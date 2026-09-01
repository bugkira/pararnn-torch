"""Fused frozen-gate sLSTM scan (Picard). Not Newton, not 4x4.

Gates are frozen in ``pre``: ``m`` is max-plus, ``n``/``c`` are two 1D
``a x + b`` scans, ``h`` is the readout. Same contract as
``slstm_frozen_gate_scan`` (eager Blelloch). Span is two-level prefix
scans (tile + chunk), not a serial ``chunk_len`` loop.

Newton still needs the 4x4 ``J`` of ``(c, n, m, h)`` because ``R h``
couples the next gates. This kernel is the Picard *guess*.

After the max-plus apply, phase-m temps ``a_loc``/``b_loc``/``agg_*``/
``incl_m`` are dead and alias the ``n``/``c`` scan lanes. ``m`` stays live.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)

log = logging.getLogger(__name__)

# 3 scan lanes (j, n, c) x 128 x 16 x 4 B = 24 KiB plus max-plus temps.
_BLOCK_T = 128
_BLOCK_D = 16
_CHUNK_PAD = 128  # 128 * 128 = 16384. 3 lanes × 128 × 16 × 4 B = 24 KiB.
_NEG_INF = -1.0e30


@triton.jit
def _compose_maxplus(a_l, b_l, a_r, b_r):
    """``m |-> max(a + m, b)``. left = earlier, right = later."""
    return a_r + a_l, tl.maximum(a_r + b_l, b_r)


@triton.jit
def _compose_diag2(j_l, n_l, c_l, j_r, n_r, c_r):
    """Two ``δ = j δ_prev + r`` scans sharing ``j`` (forget gate)."""
    return j_r * j_l, j_r * n_l + n_r, j_r * c_l + c_r


@triton.jit
def _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, gate, mask, sb, st, sd):
    idx = offs_d + gate * d_h
    return load_acc(
        pre_ptr + pid_b * sb + offs_t[:, None] * st + idx[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _m_local_kernel(
    pre_ptr,
    a_ptr,
    b_ptr,
    agg_a_ptr,
    agg_b_ptr,
    time,
    d_h,
    stride_pb, stride_pt, stride_pd,
    stride_ab, stride_at, stride_ad,
    stride_bb, stride_bt, stride_bd,
    stride_aab, stride_aac, stride_aad,
    stride_bab, stride_bac, stride_bad,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    z_i = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 0, mask, stride_pb, stride_pt, stride_pd)
    z_f = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 1, mask, stride_pb, stride_pt, stride_pd)
    a = tl.where(mask, z_f, 0.0)
    b = tl.where(mask, z_i, NEG_INF)
    a_s, b_s = tl.associative_scan((a, b), 0, _compose_maxplus)
    store_acc(a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad, a_s, mask)
    store_acc(b_ptr + pid_b * stride_bb + offs_t[:, None] * stride_bt + offs_d[None, :] * stride_bd, b_s, mask)
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    dmask = offs_d < d_h
    store_acc(
        agg_a_ptr + pid_b * stride_aab + pid_c * stride_aac + offs_d * stride_aad,
        tl.sum(tl.where(last, a_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_b_ptr + pid_b * stride_bab + pid_c * stride_bac + offs_d * stride_bad,
        tl.sum(tl.where(last, b_s, 0.0), axis=0),
        dmask,
    )


@triton.jit
def _m_chunk_kernel(
    agg_a_ptr,
    agg_b_ptr,
    m0_ptr,
    incl_m_ptr,
    n_chunks,
    d_h,
    stride_aab, stride_aac, stride_aad,
    stride_bab, stride_bac, stride_bad,
    stride_m0b, stride_m0d,
    stride_ib, stride_ic, stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    a = load_acc(
        agg_a_ptr + pid_b * stride_aab + offs_c[:, None] * stride_aac + offs_d[None, :] * stride_aad,
        mask,
        0.0,
    )
    b = load_acc(
        agg_b_ptr + pid_b * stride_bab + offs_c[:, None] * stride_bac + offs_d[None, :] * stride_bad,
        mask,
        NEG_INF,
    )
    a_s, b_s = tl.associative_scan((a, b), 0, _compose_maxplus)
    m0 = load_acc(m0_ptr + pid_b * stride_m0b + offs_d * stride_m0d, dmask, 0.0)
    m_end = tl.maximum(a_s + m0[None, :], b_s)
    store_acc(
        incl_m_ptr + pid_b * stride_ib + offs_c[:, None] * stride_ic + offs_d[None, :] * stride_id,
        m_end,
        mask,
    )


@triton.jit
def _m_apply_kernel(
    a_ptr,
    b_ptr,
    m0_ptr,
    incl_m_ptr,
    m_ptr,
    time,
    d_h,
    stride_ab, stride_at, stride_ad,
    stride_bb, stride_bt, stride_bd,
    stride_m0b, stride_m0d,
    stride_ib, stride_ic, stride_id,
    stride_mb, stride_mt, stride_md,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NEG_INF: tl.constexpr,
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
    a = load_acc(
        a_ptr + pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad,
        mask,
        0.0,
    )
    b = load_acc(
        b_ptr + pid_b * stride_bb + offs_t[:, None] * stride_bt + offs_d[None, :] * stride_bd,
        mask,
        NEG_INF,
    )
    m0 = load_acc(m0_ptr + pid_b * stride_m0b + offs_d * stride_m0d, dmask, 0.0)
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    prev = load_acc(
        incl_m_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
        dmask,
        0.0,
    )
    carry = tl.where(pid_c > 0, prev, m0)
    m = tl.maximum(a + carry[None, :], b)
    store_acc(
        m_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        m,
        mask,
    )


@triton.jit
def _nc_local_kernel(
    pre_ptr,
    m_ptr,
    h0_ptr,
    j_ptr,
    n_ptr,
    c_ptr,
    agg_j_ptr,
    agg_n_ptr,
    agg_c_ptr,
    time,
    d_h,
    stride_pb, stride_pt, stride_pd,
    stride_mb, stride_mt, stride_md,
    stride_h0b, stride_h0s, stride_h0d,
    stride_jb, stride_jt, stride_jd,
    stride_nb, stride_nt, stride_nd,
    stride_cb, stride_ct, stride_cd,
    stride_ajb, stride_ajc, stride_ajd,
    stride_anb, stride_anc, stride_and,
    stride_acb, stride_acc, stride_acd,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
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
    z_i = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 0, mask, stride_pb, stride_pt, stride_pd)
    z_f = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 1, mask, stride_pb, stride_pt, stride_pd)
    z_z = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 2, mask, stride_pb, stride_pt, stride_pd)
    m = load_acc(
        m_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask,
        0.0,
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    m_prev = load_acc(
        m_ptr + pid_b * stride_mb + offs_tm1[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask_prev,
        0.0,
    )
    m0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_M * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    n0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_N * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    c0 = load_acc(
        h0_ptr + pid_b * stride_h0b + SLOT_C * stride_h0s + offs_d * stride_h0d,
        dmask,
        0.0,
    )
    is_t0 = (offs_t == 0)[:, None]
    m_prev = tl.where(is_t0, m0[None, :], m_prev)
    i_t = tl.exp(z_i - m)
    f_t = tl.exp(z_f + m_prev - m)
    z = _nv_tanh(z_z)
    res_n = i_t
    res_c = i_t * z
    res_n = tl.where(is_t0, f_t * n0[None, :] + i_t, res_n)
    res_c = tl.where(is_t0, f_t * c0[None, :] + i_t * z, res_c)
    j = tl.where(mask, f_t, 1.0)
    res_n = tl.where(mask, res_n, 0.0)
    res_c = tl.where(mask, res_c, 0.0)
    j_s, n_s, c_s = tl.associative_scan((j, res_n, res_c), 0, _compose_diag2)
    store_acc(j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd, j_s, mask)
    store_acc(n_ptr + pid_b * stride_nb + offs_t[:, None] * stride_nt + offs_d[None, :] * stride_nd, n_s, mask)
    store_acc(c_ptr + pid_b * stride_cb + offs_t[:, None] * stride_ct + offs_d[None, :] * stride_cd, c_s, mask)
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        tl.sum(tl.where(last, j_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_n_ptr + pid_b * stride_anb + pid_c * stride_anc + offs_d * stride_and,
        tl.sum(tl.where(last, n_s, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_c_ptr + pid_b * stride_acb + pid_c * stride_acc + offs_d * stride_acd,
        tl.sum(tl.where(last, c_s, 0.0), axis=0),
        dmask,
    )


@triton.jit
def _nc_chunk_kernel(
    agg_j_ptr,
    agg_n_ptr,
    agg_c_ptr,
    incl_n_ptr,
    incl_c_ptr,
    n_chunks,
    d_h,
    stride_ajb, stride_ajc, stride_ajd,
    stride_anb, stride_anc, stride_and,
    stride_acb, stride_acc, stride_acd,
    stride_inb, stride_inc, stride_ind,
    stride_icb, stride_icc, stride_icd,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j = load_acc(
        agg_j_ptr + pid_b * stride_ajb + offs_c[:, None] * stride_ajc + offs_d[None, :] * stride_ajd,
        mask,
        1.0,
    )
    n = load_acc(
        agg_n_ptr + pid_b * stride_anb + offs_c[:, None] * stride_anc + offs_d[None, :] * stride_and,
        mask,
        0.0,
    )
    c = load_acc(
        agg_c_ptr + pid_b * stride_acb + offs_c[:, None] * stride_acc + offs_d[None, :] * stride_acd,
        mask,
        0.0,
    )
    _, n_s, c_s = tl.associative_scan((j, n, c), 0, _compose_diag2)
    store_acc(
        incl_n_ptr + pid_b * stride_inb + offs_c[:, None] * stride_inc + offs_d[None, :] * stride_ind,
        n_s,
        mask,
    )
    store_acc(
        incl_c_ptr + pid_b * stride_icb + offs_c[:, None] * stride_icc + offs_d[None, :] * stride_icd,
        c_s,
        mask,
    )


@triton.jit
def _nc_apply_h_kernel(
    pre_ptr,
    j_ptr,
    n_loc_ptr,
    c_loc_ptr,
    incl_n_ptr,
    incl_c_ptr,
    out_ptr,
    time,
    d_h,
    eps,
    stride_pb, stride_pt, stride_pd,
    stride_jb, stride_jt, stride_jd,
    stride_nb, stride_nt, stride_nd,
    stride_cb, stride_ct, stride_cd,
    stride_inb, stride_inc, stride_ind,
    stride_icb, stride_icc, stride_icd,
    stride_ob, stride_ot, stride_os, stride_od,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
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
    j = load_acc(
        j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    n_loc = load_acc(
        n_loc_ptr + pid_b * stride_nb + offs_t[:, None] * stride_nt + offs_d[None, :] * stride_nd,
        mask,
        0.0,
    )
    c_loc = load_acc(
        c_loc_ptr + pid_b * stride_cb + offs_t[:, None] * stride_ct + offs_d[None, :] * stride_cd,
        mask,
        0.0,
    )
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    cn = load_acc(
        incl_n_ptr + pid_b * stride_inb + idx_c * stride_inc + offs_d * stride_ind,
        dmask,
        0.0,
    )
    cc = load_acc(
        incl_c_ptr + pid_b * stride_icb + idx_c * stride_icc + offs_d * stride_icd,
        dmask,
        0.0,
    )
    cn = tl.where(pid_c > 0, cn, 0.0)
    cc = tl.where(pid_c > 0, cc, 0.0)
    n = j * cn[None, :] + n_loc
    c = j * cc[None, :] + c_loc
    z_o = _load_pre(pre_ptr, pid_b, offs_t, offs_d, d_h, 3, mask, stride_pb, stride_pt, stride_pd)
    h = tl.sigmoid(z_o) * (c / (n + eps))
    store_acc(
        out_ptr + pid_b * stride_ob + offs_t[:, None] * stride_ot + SLOT_C * stride_os + offs_d[None, :] * stride_od,
        c,
        mask,
    )
    store_acc(
        out_ptr + pid_b * stride_ob + offs_t[:, None] * stride_ot + SLOT_N * stride_os + offs_d[None, :] * stride_od,
        n,
        mask,
    )
    store_acc(
        out_ptr + pid_b * stride_ob + offs_t[:, None] * stride_ot + SLOT_H * stride_os + offs_d[None, :] * stride_od,
        h,
        mask,
    )


def frozen_gate_scan_triton(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """CUDA twin of ``slstm_frozen_gate_scan``. ``pre`` is ``(B, T, 4 d_h)``."""
    pre = pre.contiguous()
    batch, time, four_d = pre.shape
    if four_d % 4 != 0:
        raise ValueError(f"pre last dim {four_d} is not 4 * d_h")
    d_h = four_d // 4
    if h0 is None:
        h0 = pre.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0.dtype != pre.dtype:
            h0 = h0.to(dtype=pre.dtype)
    validate_cuda_tensors(pre, h0, name="frozen_gate_scan_triton")
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD} "
            f"(T≤{_BLOCK_T * _CHUNK_PAD}). Not serial chunk_len."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    grid = (batch, n_chunks, n_dtiles)
    grid_d = (batch, n_dtiles)
    m0 = h0[:, SLSTM_STABILIZER, :].contiguous()

    a_loc = pre.new_empty(batch, time, d_h)
    b_loc = pre.new_empty(batch, time, d_h)
    agg_a = pre.new_empty(batch, n_chunks, d_h)
    agg_b = pre.new_empty(batch, n_chunks, d_h)
    m = pre.new_empty(batch, time, d_h)
    incl_m = m.new_empty(batch, n_chunks, d_h)

    _m_local_kernel[grid](
        pre,
        a_loc,
        b_loc,
        agg_a,
        agg_b,
        time,
        d_h,
        *pre.stride(),
        *a_loc.stride(),
        *b_loc.stride(),
        *agg_a.stride(),
        *agg_b.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NEG_INF=_NEG_INF,
    )
    if n_chunks == 1:
        incl_m.zero_()
    else:
        _m_chunk_kernel[grid_d](
            agg_a,
            agg_b,
            m0,
            incl_m,
            n_chunks,
            d_h,
            *agg_a.stride(),
            *agg_b.stride(),
            m0.stride(0),
            m0.stride(1),
            *incl_m.stride(),
            CHUNK_PAD=_CHUNK_PAD,
            BLOCK_D=_BLOCK_D,
            NEG_INF=_NEG_INF,
        )
    _m_apply_kernel[grid](
        a_loc,
        b_loc,
        m0,
        incl_m,
        m,
        time,
        d_h,
        *a_loc.stride(),
        *b_loc.stride(),
        m0.stride(0),
        m0.stride(1),
        *incl_m.stride(),
        *m.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        NEG_INF=_NEG_INF,
    )

    # Phase-m temps are dead after apply. Reuse as nc scan lanes (not ``m``:
    # that is the stabilizer and is copied into ``out``).
    j_loc = a_loc
    n_loc = b_loc
    c_loc = pre.new_empty(batch, time, d_h)
    agg_j = agg_a
    agg_n = agg_b
    agg_c = pre.new_empty(batch, n_chunks, d_h)
    _nc_local_kernel[grid](
        pre,
        m,
        h0,
        j_loc,
        n_loc,
        c_loc,
        agg_j,
        agg_n,
        agg_c,
        time,
        d_h,
        *pre.stride(),
        *m.stride(),
        *h0.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *agg_j.stride(),
        *agg_n.stride(),
        *agg_c.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    out = pre.new_empty(batch, time, SLSTM_SLOTS, d_h)
    out[:, :, SLSTM_STABILIZER, :] = m
    incl_n = incl_m
    incl_c = c_loc.new_empty(batch, n_chunks, d_h)
    if n_chunks > 1:
        _nc_chunk_kernel[grid_d](
            agg_j,
            agg_n,
            agg_c,
            incl_n,
            incl_c,
            n_chunks,
            d_h,
            *agg_j.stride(),
            *agg_n.stride(),
            *agg_c.stride(),
            *incl_n.stride(),
            *incl_c.stride(),
            CHUNK_PAD=_CHUNK_PAD,
            BLOCK_D=_BLOCK_D,
        )
    else:
        incl_n.zero_()
        incl_c.zero_()
    _nc_apply_h_kernel[grid](
        pre,
        j_loc,
        n_loc,
        c_loc,
        incl_n,
        incl_c,
        out,
        time,
        d_h,
        float(eps),
        *pre.stride(),
        *j_loc.stride(),
        *n_loc.stride(),
        *c_loc.stride(),
        *incl_n.stride(),
        *incl_c.stride(),
        *out.stride(),
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    log.debug(
        "frozen_gate_scan_triton",
        extra={"seq_len": time, "batch": batch, "d_h": d_h, "n_chunks": n_chunks},
    )
    return out
