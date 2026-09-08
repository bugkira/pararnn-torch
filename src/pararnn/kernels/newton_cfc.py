"""Fused ParaCfC Newton: cell + diagonal J + scan (Alg. 1).

``project_wx(x)`` is ``(B, T, 3 d_h)`` = ``(f_pre, c_x, Δt)``. Recurrent mix
is the diagonal vector ``u``. CUDA fp16/fp32 (bf16 on CC ≥ 8.0); cell+scan
algebra in fp32.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import (
    _bt_row,
    _tanh,
    alloc_fp32_update,
    fp32_omega_add,
    fused_early_exit_hit,
    gather_h0_heads,
    is_seg_head,
    log_fused_done,
    log_fused_iter,
    mark_fused_iters_done,
    prepare_h0_block_table,
    time_tiles,
)
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.kernels.scan_diag import scan_chunk_aggregates

log = logging.getLogger(__name__)

# Same tiles as scan_diag: 128 × 32 × 2 × 4 B = 32 KiB for (J, r).
_BLOCK_T = 128
_BLOCK_D = 32
# Leaf pad matches scan_diag._CHUNK_PAD (third-level superchunks past 8192).
_CHUNK_PAD = 64


@triton.jit
def _compose_diag(j_left, r_left, j_right, r_right):
    return j_right * j_left, j_right * r_left + r_right


@triton.jit
def _cfc_pred_j(h_prev, f_pre, cx, u, dt):
    """CfC: a=exp(-softplus(f)·dt); J = a + (1-a)*(1-n^2)*u."""
    soft = tl.where(f_pre > 20.0, f_pre, tl.log(1.0 + tl.exp(f_pre)))
    a = tl.exp(-soft * dt)
    n = _tanh(cx + u * h_prev)
    h_new = a * h_prev + (1.0 - a) * n
    n_p = 1.0 - n * n
    j = a + (1.0 - a) * n_p * u
    return h_new, j


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    f_pre = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    cx = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    dt = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 1.0)
    return f_pre, cx, dt


@triton.jit
def _cfc_init_kernel(
    wx_ptr,
    h_ptr,
    u_ptr,
    h0_ptr,
    d_h,
    time,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """App. A: ``h_t = f(h_{t-1}, x_t)`` in parallel; heads see packed ``h0``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    f_pre, cx, dt = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    if HAS_CU:
        h_prev = gather_h0_heads(
            h0_ptr,
            offs_t,
            offs_d,
            dmask,
            cs_ptr,
            n_seq,
            stride_h0b,
            stride_h0d,
            bt_ptr,
            BLOCK_T,
            BLOCK_D,
            HAS_BT,
        )
    else:
        row = _bt_row(pid_b, bt_ptr, HAS_BT)
        h0 = load_acc(h0_ptr + row * stride_h0b + offs_d * stride_h0d, dmask, 0.0)
        h_prev = tl.where((offs_t == 0)[:, None], h0[None, :], 0.0)
    h_new, _ = _cfc_pred_j(h_prev, f_pre, cx, u, dt)
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h_new,
        mask,
    )


@triton.jit
def _cfc_cell_local_scan_kernel(
    h_ptr,
    wx_ptr,
    u_ptr,
    h0_ptr,
    j_loc_ptr,
    r_loc_ptr,
    agg_j_ptr,
    agg_r_ptr,
    time,
    d_h,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_h0b,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_ojb,
    stride_ojt,
    stride_ojd,
    stride_orb,
    stride_ort,
    stride_ord,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    cs_ptr,
    n_seq,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """One Newton linearization: cell+J+residual, then local inclusive scan."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h

    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    h_prev = load_acc(
        h_ptr + pid_b * stride_hb + offs_tm1[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask_prev,
        0.0,
    )
    row = _bt_row(pid_b, bt_ptr, HAS_BT)
    h0 = load_acc(h0_ptr + row * stride_h0b + offs_d * stride_h0d, dmask, 0.0)
    if HAS_CU:
        head = is_seg_head(offs_t, cs_ptr, n_seq)
        h_prev = tl.where(
            head[:, None],
            gather_h0_heads(
                h0_ptr,
                offs_t,
                offs_d,
                dmask,
                cs_ptr,
                n_seq,
                stride_h0b,
                stride_h0d,
                bt_ptr,
                BLOCK_T,
                BLOCK_D,
                HAS_BT,
            ),
            h_prev,
        )
    else:
        h_prev = tl.where((offs_t == 0)[:, None], h0[None, :], h_prev)
    f_pre, cx, dt = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    u = load_acc(u_ptr + offs_d, dmask, 0.0)
    h_new, j = _cfc_pred_j(h_prev, f_pre, cx, u, dt)
    residual = h_new - h
    if HAS_CU:
        j = tl.where(head[:, None], 0.0, j)
    j = tl.where(mask, j, 1.0)
    residual = tl.where(mask, residual, 0.0)
    j_s, r_s = tl.associative_scan((j, residual), 0, _compose_diag)
    store_acc(
        j_loc_ptr
        + pid_b * stride_ojb
        + offs_t[:, None] * stride_ojt
        + offs_d[None, :] * stride_ojd,
        j_s,
        mask,
    )
    store_acc(
        r_loc_ptr
        + pid_b * stride_orb
        + offs_t[:, None] * stride_ort
        + offs_d[None, :] * stride_ord,
        r_s,
        mask,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    agg_j = tl.sum(tl.where(last, j_s, 0.0), axis=0)
    agg_r = tl.sum(tl.where(last, r_s, 0.0), axis=0)
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        agg_j,
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + offs_d * stride_ard,
        agg_r,
        dmask,
    )


@triton.jit
def _cfc_apply_update_kernel(
    h_ptr,
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_jb,
    stride_jt,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rd,
    stride_ib,
    stride_ic,
    stride_id,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """h += omega * (J_loc @ carry + r_loc)."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    j_loc = load_acc(
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    r_loc = load_acc(
        r_loc_ptr + pid_b * stride_rb + offs_t[:, None] * stride_rt + offs_d[None, :] * stride_rd,
        mask,
        0.0,
    )
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    carry = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
        offs_d < d_h,
        0.0,
    )
    carry = tl.where(pid_c > 0, carry, 0.0)
    delta = j_loc * carry[None, :] + r_loc
    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h + omega * delta,
        mask,
    )


def _newton_cfc_fused_impl(
    wx: Tensor,
    u: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_fn=None,
    iters_done_out: list[int] | None = None,
) -> Tensor:
    """Alg. 1 for diagonal ParaCfC. Public entry: ``pararnn::newton_cfc_fused``."""
    wx = wx.contiguous()
    u = u.contiguous()
    batch, time, three_d = wx.shape
    d_h = u.numel()
    if three_d != 3 * d_h:
        raise ValueError(f"wx last dim {three_d} != 3 * d_h={3 * d_h}")
    has_cu = cu_seqlens is not None
    if has_cu:
        if batch != 1:
            raise ValueError(f"cu_seqlens packs wx with batch=1, got batch={batch}")
        cs = cu_seqlens.to(device=wx.device, dtype=torch.int32).contiguous()
        if cs.dim() != 1 or int(cs[0]) != 0 or int(cs[-1]) != time:
            raise ValueError(
                f"cu_seqlens must be (S+1,) with [0]=0 and [-1]=time={time}, got {tuple(cs.shape)}"
            )
        n_seq = int(cs.numel()) - 1
        h0, bt, has_bt = prepare_h0_block_table(wx, h0, n_seq, (d_h,), block_table)
    else:
        cs = u
        n_seq = 0
        h0, bt, has_bt = prepare_h0_block_table(wx, h0, batch, (d_h,), block_table)
    validate_cuda_tensors(wx, u, h0, name="newton_cfc_fused")
    n_chunks, n_dtiles = time_tiles(time, d_h, _BLOCK_T, _BLOCK_D, _CHUNK_PAD, cap=False)
    h = wx.new_empty(batch, time, d_h)
    grid_td = (batch, n_chunks, n_dtiles)
    _cfc_init_kernel[grid_td](
        wx,
        h,
        u,
        h0,
        d_h,
        time,
        *wx.stride(),
        *h.stride(),
        h0.stride(0),
        h0.stride(1),
        cs,
        n_seq,
        bt,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        HAS_CU=has_cu,
        HAS_BT=has_bt,
    )
    if max_iters <= 0:
        return h

    j_loc = h.new_empty(batch, time, d_h)
    r_loc = h.new_empty(batch, time, d_h)
    agg_j = h.new_empty(batch, n_chunks, d_h)
    agg_r = h.new_empty(batch, n_chunks, d_h)
    incl_r = h.new_empty(batch, n_chunks, d_h) if n_chunks > 1 else None
    omega_f = float(omega)
    h32, r32 = alloc_fp32_update(h, r_loc, n_chunks)

    for it in range(max_iters):
        _cfc_cell_local_scan_kernel[grid_td](
            h,
            wx,
            u,
            h0,
            j_loc,
            r_loc,
            agg_j,
            agg_r,
            time,
            d_h,
            *h.stride(),
            h0.stride(0),
            h0.stride(1),
            *wx.stride(),
            *j_loc.stride(),
            *r_loc.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            cs,
            n_seq,
            bt,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_CU=has_cu,
            HAS_BT=has_bt,
        )
        if h32 is not None and r32 is not None:
            fp32_omega_add(h, r_loc, omega_f, h32, r32)
        else:
            if incl_r is None:
                raise RuntimeError("fused CfC scan buffer missing for n_chunks>1")
            incl_r.copy_(scan_chunk_aggregates(agg_j, agg_r))
            _cfc_apply_update_kernel[grid_td](
                h,
                j_loc,
                r_loc,
                incl_r,
                time,
                d_h,
                float(omega),
                *h.stride(),
                *j_loc.stride(),
                *r_loc.stride(),
                *incl_r.stride(),
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
            )
        log_fused_iter(
            log,
            "newton_cfc_fused_iter",
            it=it,
            time=time,
            batch=batch,
            d_h=d_h,
            n_chunks=n_chunks,
        )
        if fused_early_exit_hit(
            residual_fn,
            early_exit_atol,
            h,
            iters_done_out=iters_done_out,
            it=it,
        ):
            log_fused_done(
                log,
                "newton_cfc_fused",
                time=time,
                batch=batch,
                d_h=d_h,
                max_iters=it + 1,
                n_chunks=n_chunks,
                early_exit=True,
            )
            return h
    mark_fused_iters_done(iters_done_out, max_iters)
    log_fused_done(
        log,
        "newton_cfc_fused",
        time=time,
        batch=batch,
        d_h=d_h,
        max_iters=max_iters,
        n_chunks=n_chunks,
    )
    return h
