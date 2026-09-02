"""Fused ParaLSTM Newton: CIFG cell + 2×2 J + scan (paper Alg. 1 / eq. 3.1b, 3.2b).

``W_x(x)`` stays a cuBLAS GEMM. CUDA float16/float32, and bf16 on compute
capability ≥ 8.0; cell+scan algebra in fp32. DRAM is the tensor dtype.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import LSTM_CELL, LSTM_HIDDEN

log = logging.getLogger(__name__)

# Same tiles as scan_block2 (6 scan lanes).
_BLOCK_T = 64
_BLOCK_D = 16
_CHUNK_PAD = 64  # 64 * 64 = 4096.


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _compose_block2(
    a00,
    a01,
    a10,
    a11,
    u0,
    u1,
    b00,
    b01,
    b10,
    b11,
    v0,
    v1,
):
    o00 = b00 * a00 + b01 * a10
    o01 = b00 * a01 + b01 * a11
    o10 = b10 * a00 + b11 * a10
    o11 = b10 * a01 + b11 * a11
    w0 = b00 * u0 + b01 * u1 + v0
    w1 = b10 * u0 + b11 * u1 + v1
    return o00, o01, o10, o11, w0, w1


@triton.jit
def _lstm_pred_j(
    c_prev,
    h_prev,
    fx,
    zx,
    ox,
    a_f,
    a_z,
    a_o,
    peephole_f,
    peephole_o,
):
    """Eq. 3.1b and 2×2 J (eq. 3.2b). Returns c, h, Jcc, Jch, Jhc, Jhh."""
    f = tl.sigmoid(a_f * h_prev + peephole_f * c_prev + fx)
    z = _tanh(a_z * h_prev + zx)
    c = f * c_prev + (1.0 - f) * z
    o = tl.sigmoid(a_o * h_prev + peephole_o * c + ox)
    h_act = _tanh(c)
    h = o * h_act
    f_p = f * (1.0 - f)
    z_p = 1.0 - z * z
    o_p = o * (1.0 - o)
    h_act_p = 1.0 - h_act * h_act
    j_cc = f + (c_prev - z) * f_p * peephole_f
    j_ch = (c_prev - z) * f_p * a_f + (1.0 - f) * z_p * a_z
    j_hc = (h_act * o_p * peephole_o + o * h_act_p) * j_cc
    j_hh = h_act * o_p * (a_o + peephole_o * j_ch) + o * h_act_p * j_ch
    return c, h, j_cc, j_ch, j_hc, j_hh


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    fx = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    zx = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    ox = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    return fx, zx, ox


@triton.jit
def _load_state(s_ptr, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    return load_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _store_state(s_ptr, val, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    store_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        val,
        mask,
    )


@triton.jit
def _lstm_init_kernel(
    wx_ptr,
    s_ptr,
    af_ptr,
    az_ptr,
    ao_ptr,
    cf_ptr,
    co_ptr,
    h0_ptr,
    d_h,
    time,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    SLOT_C: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """App. A: only t=0 sees ``h0``; later t still ``f(0, x_t)``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    fx, zx, ox = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    a_f = load_acc(af_ptr + offs_d, dmask, 0.0)
    a_z = load_acc(az_ptr + offs_d, dmask, 0.0)
    a_o = load_acc(ao_ptr + offs_d, dmask, 0.0)
    peephole_f = load_acc(cf_ptr + offs_d, dmask, 0.0)
    peephole_o = load_acc(co_ptr + offs_d, dmask, 0.0)
    c0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    h0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_H * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], 0.0)
    h_prev = tl.where(is_t0, h0[None, :], 0.0)
    c, h, _, _, _, _ = _lstm_pred_j(
        c_prev, h_prev, fx, zx, ox, a_f, a_z, a_o, peephole_f, peephole_o
    )
    _store_state(s_ptr, c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd)
    _store_state(s_ptr, h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd)


@triton.jit
def _lstm_cell_local_scan_kernel(
    s_ptr,
    wx_ptr,
    af_ptr,
    az_ptr,
    ao_ptr,
    cf_ptr,
    co_ptr,
    h0_ptr,
    j_loc_ptr,
    r_loc_ptr,
    agg_j_ptr,
    agg_r_ptr,
    time,
    d_h,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_jb,
    stride_jt,
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    SLOT_C: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h

    c = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd)
    h = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd)
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    c_prev = _load_state(
        s_ptr, pid_b, offs_tm1, offs_d, SLOT_C, mask_prev, stride_sb, stride_st, stride_ss, stride_sd
    )
    h_prev = _load_state(
        s_ptr, pid_b, offs_tm1, offs_d, SLOT_H, mask_prev, stride_sb, stride_st, stride_ss, stride_sd
    )
    c0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    h0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_H * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], c_prev)
    h_prev = tl.where(is_t0, h0[None, :], h_prev)
    fx, zx, ox = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    a_f = load_acc(af_ptr + offs_d, dmask, 0.0)
    a_z = load_acc(az_ptr + offs_d, dmask, 0.0)
    a_o = load_acc(ao_ptr + offs_d, dmask, 0.0)
    peephole_f = load_acc(cf_ptr + offs_d, dmask, 0.0)
    peephole_o = load_acc(co_ptr + offs_d, dmask, 0.0)
    c_new, h_new, j00, j01, j10, j11 = _lstm_pred_j(
        c_prev, h_prev, fx, zx, ox, a_f, a_z, a_o, peephole_f, peephole_o
    )
    r0 = c_new - c
    r1 = h_new - h
    ident00 = tl.where(mask, j00, 1.0)
    ident01 = tl.where(mask, j01, 0.0)
    ident10 = tl.where(mask, j10, 0.0)
    ident11 = tl.where(mask, j11, 1.0)
    r0 = tl.where(mask, r0, 0.0)
    r1 = tl.where(mask, r1, 0.0)
    j00s, j01s, j10s, j11s, r0s, r1s = tl.associative_scan(
        (ident00, ident01, ident10, ident11, r0, r1), 0, _compose_block2
    )
    store_acc(
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 0 * stride_jk + offs_d[None, :] * stride_jd,
        j00s,
        mask,
    )
    store_acc(
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 1 * stride_jk + offs_d[None, :] * stride_jd,
        j01s,
        mask,
    )
    store_acc(
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 2 * stride_jk + offs_d[None, :] * stride_jd,
        j10s,
        mask,
    )
    store_acc(
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 3 * stride_jk + offs_d[None, :] * stride_jd,
        j11s,
        mask,
    )
    _store_state(r_loc_ptr, r0s, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    _store_state(r_loc_ptr, r1s, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    agg00 = tl.sum(tl.where(last, j00s, 0.0), axis=0)
    agg01 = tl.sum(tl.where(last, j01s, 0.0), axis=0)
    agg10 = tl.sum(tl.where(last, j10s, 0.0), axis=0)
    agg11 = tl.sum(tl.where(last, j11s, 0.0), axis=0)
    aggr0 = tl.sum(tl.where(last, r0s, 0.0), axis=0)
    aggr1 = tl.sum(tl.where(last, r1s, 0.0), axis=0)
    store_acc(agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 0 * stride_ajk + offs_d * stride_ajd, agg00, dmask)
    store_acc(agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 1 * stride_ajk + offs_d * stride_ajd, agg01, dmask)
    store_acc(agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 2 * stride_ajk + offs_d * stride_ajd, agg10, dmask)
    store_acc(agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 3 * stride_ajk + offs_d * stride_ajd, agg11, dmask)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_C * stride_ars + offs_d * stride_ard, aggr0, dmask)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_H * stride_ars + offs_d * stride_ard, aggr1, dmask)


@triton.jit
def _chunk_incl_kernel(
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j00 = load_acc(agg_j_ptr + pid_b * stride_ajb + offs_c[:, None] * stride_ajc + 0 * stride_ajk + offs_d[None, :] * stride_ajd, mask, 1.0)
    j01 = load_acc(agg_j_ptr + pid_b * stride_ajb + offs_c[:, None] * stride_ajc + 1 * stride_ajk + offs_d[None, :] * stride_ajd, mask, 0.0)
    j10 = load_acc(agg_j_ptr + pid_b * stride_ajb + offs_c[:, None] * stride_ajc + 2 * stride_ajk + offs_d[None, :] * stride_ajd, mask, 0.0)
    j11 = load_acc(agg_j_ptr + pid_b * stride_ajb + offs_c[:, None] * stride_ajc + 3 * stride_ajk + offs_d[None, :] * stride_ajd, mask, 1.0)
    r0 = load_acc(agg_r_ptr + pid_b * stride_arb + offs_c[:, None] * stride_arc + 0 * stride_ars + offs_d[None, :] * stride_ard, mask, 0.0)
    r1 = load_acc(agg_r_ptr + pid_b * stride_arb + offs_c[:, None] * stride_arc + 1 * stride_ars + offs_d[None, :] * stride_ard, mask, 0.0)
    _, _, _, _, u0, u1 = tl.associative_scan((j00, j01, j10, j11, r0, r1), 0, _compose_block2)
    store_acc(incl_r_ptr + pid_b * stride_ib + offs_c[:, None] * stride_ic + 0 * stride_is + offs_d[None, :] * stride_id, u0, mask)
    store_acc(incl_r_ptr + pid_b * stride_ib + offs_c[:, None] * stride_ic + 1 * stride_is + offs_d[None, :] * stride_id, u1, mask)


@triton.jit
def _lstm_apply_update_kernel(
    s_ptr,
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_jb,
    stride_jt,
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    SLOT_C: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    j00 = load_acc(j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 0 * stride_jk + offs_d[None, :] * stride_jd, mask, 1.0)
    j01 = load_acc(j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 1 * stride_jk + offs_d[None, :] * stride_jd, mask, 0.0)
    j10 = load_acc(j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 2 * stride_jk + offs_d[None, :] * stride_jd, mask, 0.0)
    j11 = load_acc(j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + 3 * stride_jk + offs_d[None, :] * stride_jd, mask, 1.0)
    r0 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    r1 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    c0 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_C * stride_is + offs_d * stride_id, dmask, 0.0)
    c1 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_H * stride_is + offs_d * stride_id, dmask, 0.0)
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    delta_c = j00 * c0[None, :] + j01 * c1[None, :] + r0
    delta_h = j10 * c0[None, :] + j11 * c1[None, :] + r1
    c = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd)
    h = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd)
    _store_state(
        s_ptr, c + omega * delta_c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, h + omega * delta_h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )


def newton_lstm_fused(
    wx: Tensor,
    a_f: Tensor,
    a_z: Tensor,
    a_o: Tensor,
    c_f: Tensor,
    c_o: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
) -> Tensor:
    """Alg. 1 for CIFG ParaLSTM. ``wx`` is ``W_x(x)`` with shape ``(B, T, 3 d_h)``.

    ``h0`` is paper ``h_0`` (default zeros), shape ``(B, 2, d_h)``.
    """
    wx = wx.contiguous()
    a_f = a_f.contiguous()
    a_z = a_z.contiguous()
    a_o = a_o.contiguous()
    c_f = c_f.contiguous()
    c_o = c_o.contiguous()
    batch, time, three_d = wx.shape
    d_h = a_f.numel()
    if three_d != 3 * d_h:
        raise ValueError(f"wx last dim {three_d} != 3 * d_h={3 * d_h}")
    if h0 is None:
        h0 = wx.new_zeros(batch, 2, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, 2, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, 2, d_h)}")
        if h0.dtype != wx.dtype:
            h0 = h0.to(dtype=wx.dtype)
    validate_cuda_tensors(wx, a_f, a_z, a_o, c_f, c_o, h0, name="newton_lstm_fused")
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD}. "
            "Increase BLOCK_T or CHUNK_PAD."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    states = wx.new_empty(batch, time, 2, d_h)
    grid_td = (batch, n_chunks, n_dtiles)
    _lstm_init_kernel[grid_td](
        wx,
        states,
        a_f,
        a_z,
        a_o,
        c_f,
        c_o,
        h0,
        d_h,
        time,
        *wx.stride(),
        *states.stride(),
        *h0.stride(),
        SLOT_C=LSTM_CELL,
        SLOT_H=LSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    if max_iters <= 0:
        return states

    j_loc = states.new_empty(batch, time, 4, d_h)
    r_loc = states.new_empty(batch, time, 2, d_h)
    agg_j = j_loc.new_empty(batch, n_chunks, 4, d_h)
    agg_r = r_loc.new_empty(batch, n_chunks, 2, d_h)
    incl_r = r_loc.new_empty(batch, n_chunks, 2, d_h) if n_chunks > 1 else None
    omega_f = float(omega)
    states32 = r32 = None
    if n_chunks == 1:
        states32 = states.new_empty(states.shape, dtype=torch.float32)
        r32 = r_loc.new_empty(r_loc.shape, dtype=torch.float32)

    for it in range(max_iters):
        _lstm_cell_local_scan_kernel[grid_td](
            states,
            wx,
            a_f,
            a_z,
            a_o,
            c_f,
            c_o,
            h0,
            j_loc,
            r_loc,
            agg_j,
            agg_r,
            time,
            d_h,
            *states.stride(),
            *h0.stride(),
            *wx.stride(),
            *j_loc.stride(),
            *r_loc.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            SLOT_C=LSTM_CELL,
            SLOT_H=LSTM_HIDDEN,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
        )
        if n_chunks == 1:
            states32.copy_(states)
            r32.copy_(r_loc)
            states32.add_(r32, alpha=omega_f)
            states.copy_(states32)
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
            )
            _lstm_apply_update_kernel[grid_td](
                states,
                j_loc,
                r_loc,
                incl_r,
                time,
                d_h,
                float(omega),
                *states.stride(),
                *j_loc.stride(),
                *r_loc.stride(),
                *incl_r.stride(),
                SLOT_C=LSTM_CELL,
                SLOT_H=LSTM_HIDDEN,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
            )
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "newton_lstm_fused_iter",
                extra={
                    "iter": it,
                    "seq_len": time,
                    "batch": batch,
                    "d_h": d_h,
                    "n_chunks": n_chunks,
                },
            )
    log.debug(
        "newton_lstm_fused",
        extra={
            "seq_len": time,
            "batch": batch,
            "d_h": d_h,
            "max_iters": max_iters,
            "n_chunks": n_chunks,
        },
    )
    return states
