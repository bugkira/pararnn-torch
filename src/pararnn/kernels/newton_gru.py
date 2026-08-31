"""Fused ParaGRU Newton: cell + diagonal J + scan (paper Alg. 1 / eq. 3.1a, 3.2a).

``W_x(x)`` stays a cuBLAS GEMM. This kernel is the rest of one Newton step.
Not Apple's fused CUDA. CUDA float16/float32; cell+scan algebra in fp32.
Not bf16 (Turing has no bf16 tensor cores).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.prec import check_cuda_real, load_acc, store_acc

log = logging.getLogger(__name__)

# Same tiles as scan_diag: 128 × 32 × 2 × 4 B = 32 KiB for (J, r).
_BLOCK_T = 128
_BLOCK_D = 32
_CHUNK_PAD = 64  # 64 * 128 = 8192.


@triton.jit
def _tanh(x):
    """CUDA libdevice tanh — same family as torch.tanh, not the sigmoid identity."""
    return _nv_tanh(x)


@triton.jit
def _compose_diag(j_left, r_left, j_right, r_right):
    return j_right * j_left, j_right * r_left + r_right


@triton.jit
def _gru_pred_j(h_prev, zx, rx, nx, az, ar, an):
    """Eq. 3.1a and diagonal J (eq. 3.2a), elementwise A_*."""
    z = tl.sigmoid(az * h_prev + zx)
    r = tl.sigmoid(ar * h_prev + rx)
    n = _tanh(an * (h_prev * r) + nx)
    h_new = (1.0 - z) * h_prev + z * n
    z_p = z * (1.0 - z)
    r_p = r * (1.0 - r)
    n_p = 1.0 - n * n
    j = (
        (1.0 - z)
        + (n - h_prev) * z_p * az
        + z * n_p * an * (r + h_prev * r_p * ar)
    )
    return h_new, j


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd, FP16: tl.constexpr):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    zx = load_acc(base + offs_d[None, :] * sd, mask, 0.0, FP16)
    rx = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0, FP16)
    nx = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0, FP16)
    return zx, rx, nx


@triton.jit
def _gru_init_kernel(
    wx_ptr,
    h_ptr,
    d_h,
    time,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_hb,
    stride_ht,
    stride_hd,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    """App. A: h_t = f(0, x_t). Fully parallel, no scan."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    zx, rx, nx = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd, FP16
    )
    h_new, _ = _gru_pred_j(0.0, zx, rx, nx, 0.0, 0.0, 0.0)
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h_new,
        mask,
        FP16,
    )


@triton.jit
def _gru_cell_local_scan_kernel(
    h_ptr,
    wx_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    j_loc_ptr,
    r_loc_ptr,
    agg_j_ptr,
    agg_r_ptr,
    time,
    d_h,
    stride_hb,
    stride_ht,
    stride_hd,
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
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
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
        FP16,
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    h_prev = load_acc(
        h_ptr
        + pid_b * stride_hb
        + offs_tm1[:, None] * stride_ht
        + offs_d[None, :] * stride_hd,
        mask_prev,
        0.0,
        FP16,
    )
    zx, rx, nx = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd, FP16
    )
    az = load_acc(az_ptr + offs_d, dmask, 0.0, FP16)
    ar = load_acc(ar_ptr + offs_d, dmask, 0.0, FP16)
    an = load_acc(an_ptr + offs_d, dmask, 0.0, FP16)
    h_new, j = _gru_pred_j(h_prev, zx, rx, nx, az, ar, an)
    residual = h_new - h
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
        FP16,
    )
    store_acc(
        r_loc_ptr
        + pid_b * stride_orb
        + offs_t[:, None] * stride_ort
        + offs_d[None, :] * stride_ord,
        r_s,
        mask,
        FP16,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    agg_j = tl.sum(tl.where(last, j_s, 0.0), axis=0)
    agg_r = tl.sum(tl.where(last, r_s, 0.0), axis=0)
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        agg_j,
        dmask,
        FP16,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + offs_d * stride_ard,
        agg_r,
        dmask,
        FP16,
    )


@triton.jit
def _chunk_incl_kernel(
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j = load_acc(
        agg_j_ptr
        + pid_b * stride_ajb
        + offs_c[:, None] * stride_ajc
        + offs_d[None, :] * stride_ajd,
        mask,
        1.0,
        FP16,
    )
    r = load_acc(
        agg_r_ptr
        + pid_b * stride_arb
        + offs_c[:, None] * stride_arc
        + offs_d[None, :] * stride_ard,
        mask,
        0.0,
        FP16,
    )
    _, r_s = tl.associative_scan((j, r), 0, _compose_diag)
    store_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + offs_c[:, None] * stride_ic
        + offs_d[None, :] * stride_id,
        r_s,
        mask,
        FP16,
    )


@triton.jit
def _gru_apply_update_kernel(
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
    FP16: tl.constexpr,
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
        j_loc_ptr
        + pid_b * stride_jb
        + offs_t[:, None] * stride_jt
        + offs_d[None, :] * stride_jd,
        mask,
        1.0,
        FP16,
    )
    r_loc = load_acc(
        r_loc_ptr
        + pid_b * stride_rb
        + offs_t[:, None] * stride_rt
        + offs_d[None, :] * stride_rd,
        mask,
        0.0,
        FP16,
    )
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    carry = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
        offs_d < d_h,
        0.0,
        FP16,
    )
    carry = tl.where(pid_c > 0, carry, 0.0)
    delta = j_loc * carry[None, :] + r_loc
    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
        FP16,
    )
    store_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        h + omega * delta,
        mask,
        FP16,
    )


def newton_gru_fused(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Alg. 1 for diagonal ParaGRU. ``wx`` is ``W_x(x)`` with shape ``(B, T, 3 d_h)``."""
    fp16 = check_cuda_real(wx, a_z, a_r, a_n, name="newton_gru_fused")
    wx = wx.contiguous()
    a_z = a_z.contiguous()
    a_r = a_r.contiguous()
    a_n = a_n.contiguous()
    batch, time, three_d = wx.shape
    d_h = a_z.numel()
    if three_d != 3 * d_h:
        raise ValueError(f"wx last dim {three_d} != 3 * d_h={3 * d_h}")
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD}. "
            "Increase BLOCK_T rather than copying a longer Apple kernel."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    h = wx.new_empty(batch, time, d_h)
    grid_td = (batch, n_chunks, n_dtiles)
    _gru_init_kernel[grid_td](
        wx,
        h,
        d_h,
        time,
        *wx.stride(),
        *h.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        FP16=fp16,
    )
    if max_iters <= 0:
        return h

    j_loc = h.new_empty(batch, time, d_h)
    r_loc = h.new_empty(batch, time, d_h)
    agg_j = h.new_empty(batch, n_chunks, d_h)
    agg_r = h.new_empty(batch, n_chunks, d_h)
    incl_r = h.new_empty(batch, n_chunks, d_h) if n_chunks > 1 else None

    for it in range(max_iters):
        _gru_cell_local_scan_kernel[grid_td](
            h,
            wx,
            a_z,
            a_r,
            a_n,
            j_loc,
            r_loc,
            agg_j,
            agg_r,
            time,
            d_h,
            *h.stride(),
            *wx.stride(),
            *j_loc.stride(),
            *r_loc.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
        FP16=fp16,
        )
        if n_chunks == 1:
            h.copy_((h.float() + float(omega) * r_loc.float()).to(h.dtype))
        else:
            _chunk_incl_kernel[(batch, n_dtiles)](
                agg_j,
                agg_r,
                incl_r,
                n_chunks,
                d_h,
                *agg_j.stride(),
                *agg_r.stride(),
                *incl_r.stride(),
                CHUNK_PAD=_CHUNK_PAD,
                BLOCK_D=_BLOCK_D,
        FP16=fp16,
            )
            _gru_apply_update_kernel[grid_td](
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
        FP16=fp16,
            )
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "newton_gru_fused_iter",
                extra={
                    "iter": it,
                    "seq_len": time,
                    "batch": batch,
                    "d_h": d_h,
                    "n_chunks": n_chunks,
                },
            )
    log.debug(
        "newton_gru_fused",
        extra={
            "seq_len": time,
            "batch": batch,
            "d_h": d_h,
            "max_iters": max_iters,
            "n_chunks": n_chunks,
        },
    )
    return h

