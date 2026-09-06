"""Factorized / fused Newton for ``ParaSLSTM(mix='head')`` — no dense ``(4d)×(4d)``.

Alg. 1 with factorized ``J δ`` / ``J^T μ`` per head. Rectangular CUDA batches:
``d_head ≤ 32`` fused (all four ``R_g`` in SRAM); ``32 < d_head ≤ 128``
streamed-``R`` SRAM; larger / CPU: PyTorch factor scan. Picard / zero-hidden
init come from ``fused_newton``. Dense-J oracle stays on ``scan_backend='eager'``.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels.precision import load_acc, store_acc
from pararnn.kernels.slstm_head_factor import (
    slstm_head_gates,
    slstm_head_jt_mvp,
    slstm_head_jvp,
)
from pararnn.layout import SLSTM_SLOTS, prepend_state, slstm_pack_heads, slstm_unpack_heads

log = logging.getLogger(__name__)

# All four ``R_g`` in SRAM (4 × d² floats).
_BLOCK_D = 32
# Stream one ``R`` at a time up to this head width.
_REVERSE_SRAM_D = 128
_NEWTON_SRAM_D = 128

SLOT_C = tl.constexpr(0)
SLOT_N = tl.constexpr(1)
SLOT_M = tl.constexpr(2)
SLOT_H = tl.constexpr(3)


def newton_slstm_head_factorized(
    cell: ParaSLSTM,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    states: Tensor | None = None,
) -> Tensor:
    """Alg. 1 for head sLSTM without materializing ``J``.

    Prefer ``pararnn::newton_slstm_head_fused`` via ``fused_newton`` on CUDA.
    """
    if cell.mix != "head" or cell.n_heads is None or cell.d_head is None:
        raise TypeError("newton_slstm_head_factorized needs ParaSLSTM(mix='head')")
    r = cell.clipped_r_head()
    return _newton_slstm_head_fused_impl(
        wx,
        r,
        h0=h0,
        states=states,
        max_iters=max_iters,
        omega=omega,
        eps=cell.eps,
    )


def _newton_slstm_head_fused_impl(
    wx: Tensor,
    r_head: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Tensor Alg. 1 for head sLSTM. Public entry: ``pararnn::newton_slstm_head_fused``."""
    if cu_seqlens is not None:
        raise TypeError(
            "ParaSLSTM(mix='head') fused Newton does not support cu_seqlens yet; "
            "use scan_backend='eager' for ragged packs, or pad to a rectangular batch"
        )
    if r_head.dim() != 4 or r_head.shape[-1] != r_head.shape[-2]:
        raise ValueError(f"r_head needs (4,H,d,d), got {tuple(r_head.shape)}")
    n_gates, n_heads, d_head, _ = r_head.shape
    if n_gates != SLSTM_SLOTS:
        raise ValueError(f"r_head first dim must be 4, got {n_gates}")
    d_h = n_heads * d_head
    if wx.shape[-1] != 4 * d_h:
        raise ValueError(f"wx last dim {wx.shape[-1]} != 4 * d_h={4 * d_h}")
    batch, time, _ = wx.shape

    fused = wx.is_cuda and d_head <= _BLOCK_D
    stream = wx.is_cuda and _BLOCK_D < d_head <= _NEWTON_SRAM_D

    if fused:
        states_out = _newton_fused_triton(
            wx,
            r_head,
            h0,
            states=states,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            n_heads=n_heads,
            d_head=d_head,
        )
    elif stream:
        states_out = _newton_stream_r_triton(
            wx,
            r_head,
            h0,
            states=states,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            n_heads=n_heads,
            d_head=d_head,
        )
    else:
        states_out = _newton_factor_eager(
            wx,
            r_head,
            h0=h0,
            states=states,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            n_heads=n_heads,
            d_head=d_head,
        )

    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        path = "fused" if fused else ("stream" if stream else "eager")
        log.debug(
            "newton_slstm_head_fused",
            extra={
                "batch": batch,
                "seq_len": time,
                "n_heads": n_heads,
                "d_head": d_head,
                "max_iters": max_iters,
                "path": path,
            },
        )
    return states_out


def reverse_factor_scan_slstm_head(
    cell: ParaSLSTM,
    h_prev: Tensor,
    partial: Tensor,
    *,
    wx: Tensor,
) -> Tensor:
    """Eq. 2.6 reverse with factorized ``J^T`` (no dense ``(4d)×(4d)``)."""
    r = cell.clipped_r_head()
    assert cell.n_heads is not None and cell.d_head is not None
    if partial.is_cuda:
        from pararnn.kernels.custom_ops import reverse_slstm_head_factor

        return reverse_slstm_head_factor(h_prev, wx, partial, r, eps=cell.eps)
    return _reverse_slstm_head_factor_impl(
        h_prev, wx, partial, r, n_heads=cell.n_heads, d_head=cell.d_head, eps=cell.eps
    )


def _reverse_slstm_head_factor_impl(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> Tensor:
    """Tensor eq. 2.6 reverse. Public entry: ``pararnn::reverse_slstm_head_factor``."""
    if partial.is_cuda and d_head <= _BLOCK_D:
        return _reverse_fused_triton(
            h_prev, wx, partial, r_head, n_heads=n_heads, d_head=d_head, eps=eps
        )
    if partial.is_cuda and d_head <= _REVERSE_SRAM_D:
        return _reverse_stream_r_triton(
            h_prev, wx, partial, r_head, n_heads=n_heads, d_head=d_head, eps=eps
        )
    _, acts = slstm_head_gates(
        h_prev, wx, r_head, n_heads=n_heads, d_head=d_head, eps=eps
    )
    return _factor_reverse(acts, partial, r_head, n_heads=n_heads, d_head=d_head)


def slstm_head_t0_vjp(
    cell: ParaSLSTM,
    h_prev: Tensor,
    mu: Tensor,
    *,
    wx: Tensor,
) -> Tensor:
    """``J_0^T μ_0`` for the paper ``h_0`` adjoint (factorized)."""
    r = cell.clipped_r_head()
    assert cell.n_heads is not None and cell.d_head is not None
    n_heads, d_head = cell.n_heads, cell.d_head
    _, acts = slstm_head_gates(
        h_prev, wx, r, n_heads=n_heads, d_head=d_head, eps=cell.eps
    )
    acts0 = {k: v[:, 0] for k, v in acts.items()}
    mu0 = slstm_pack_heads(mu[:, 0], n_heads, d_head)
    g = slstm_head_jt_mvp(acts0, r, mu0)
    return slstm_unpack_heads(g, n_heads, d_head)


def _newton_factor_eager(
    wx: Tensor,
    r_head: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    n_heads: int,
    d_head: int,
    h0: Tensor | None = None,
    states: Tensor | None = None,
) -> Tensor:
    d_h = n_heads * d_head
    batch, time, _ = wx.shape
    if states is None:
        zeros = wx.new_zeros(batch, time, SLSTM_SLOTS, d_h)
        h_prev0 = prepend_state(zeros, h0)
        states, _ = slstm_head_gates(
            h_prev0, wx, r_head, n_heads=n_heads, d_head=d_head, eps=eps
        )
    for _ in range(max_iters):
        h_prev = prepend_state(states, h0)
        pred, acts = slstm_head_gates(
            h_prev, wx, r_head, n_heads=n_heads, d_head=d_head, eps=eps
        )
        residual = pred - states
        delta = _factor_scan(acts, residual, r_head, n_heads=n_heads, d_head=d_head)
        states = states + omega * delta
    return states


def _factor_scan(
    acts: dict[str, Tensor],
    residual: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time = residual.shape[:2]
    res_h = slstm_pack_heads(residual, n_heads, d_head)
    delta = residual.new_zeros(batch, n_heads, 4 * d_head)
    out = residual.new_empty(batch, time, n_heads, 4 * d_head)
    for t in range(time):
        acts_t = {k: v[:, t] for k, v in acts.items()}
        delta = slstm_head_jvp(acts_t, r_head, delta)
        delta = delta + res_h[:, t]
        out[:, t] = delta
    return slstm_unpack_heads(out, n_heads, d_head)


def _factor_reverse(
    acts: dict[str, Tensor],
    partial: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time = partial.shape[:2]
    part_h = slstm_pack_heads(partial, n_heads, d_head)
    mu = partial.new_zeros(batch, n_heads, 4 * d_head)
    out = partial.new_empty(batch, time, n_heads, 4 * d_head)
    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            acts_tp1 = {k: v[:, t + 1] for k, v in acts.items()}
            mu = slstm_head_jt_mvp(acts_tp1, r_head, mu)
        mu = mu + part_h[:, t]
        out[:, t] = mu
    return slstm_unpack_heads(out, n_heads, d_head)


def _block_d(d_head: int, cap: int) -> int:
    block = 1 << (d_head - 1).bit_length()
    return min(max(block, 16), cap)


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _mix(h, r):
    """``h @ R`` with ``R`` ``(d_in, d_out)`` in SRAM."""
    return tl.sum(h[:, None] * r, axis=0)


@triton.jit
def _mix_t(v, r):
    """``R @ v`` (= ``v @ R^T``)."""
    return tl.sum(r * v[None, :], axis=1)


@triton.jit
def _load_slot(s_ptr, b, t, slot, head_off, offs, mask, sb, st, ss, sd):
    return load_acc(
        s_ptr + b * sb + t * st + slot * ss + (head_off + offs) * sd,
        mask,
        0.0,
    )


@triton.jit
def _store_slot(s_ptr, val, b, t, slot, head_off, offs, mask, sb, st, ss, sd):
    store_acc(
        s_ptr + b * sb + t * st + slot * ss + (head_off + offs) * sd,
        val,
        mask,
    )


@triton.jit
def _load_wx_gate(wx_ptr, b, t, gate, d_h, head_off, offs, mask, wb, wt, wd):
    return load_acc(
        wx_ptr + b * wb + t * wt + (gate * d_h + head_off + offs) * wd,
        mask,
        0.0,
    )


@triton.jit
def _load_r_gate(r_ptr, gate, head, offs, mask_ij, rg, rh, rin, rout):
    return load_acc(
        r_ptr
        + gate * rg
        + head * rh
        + offs[:, None] * rin
        + offs[None, :] * rout,
        mask_ij,
        0.0,
    )


@triton.jit
def _slstm_gates(c, n, m, h, wxi, wxf, wxz, wxo, ri, rf, rz, ro, eps):
    """Head sLSTM step; returns new state + acts for factorized J."""
    zi = _mix(h, ri) + wxi
    zf = _mix(h, rf) + wxf
    zz = _mix(h, rz) + wxz
    zo = _mix(h, ro) + wxo
    left = zf + m
    m_new = tl.maximum(left, zi)
    gt = (left > zi).to(tl.float32)
    eq = (left == zi).to(tl.float32)
    alpha = gt + 0.5 * eq
    i_t = tl.exp(zi - m_new)
    f_t = tl.exp(zf + m - m_new)
    z = _tanh(zz)
    n_new = f_t * n + i_t
    c_new = f_t * c + i_t * z
    o = tl.sigmoid(zo)
    denom = n_new + eps
    h_new = o * (c_new / denom)
    return c_new, n_new, m_new, h_new, c, n, m, h, i_t, f_t, z, o, denom, alpha, c_new


@triton.jit
def _slstm_gates_stream(
    c,
    n,
    m,
    h,
    wxi,
    wxf,
    wxz,
    wxo,
    r_ptr,
    head,
    offs,
    mask_ij,
    rg,
    rh,
    rin,
    rout,
    eps,
):
    """Same as ``_slstm_gates`` but loads one ``R`` at a time."""
    ri = _load_r_gate(r_ptr, 0, head, offs, mask_ij, rg, rh, rin, rout)
    zi = _mix(h, ri) + wxi
    rf = _load_r_gate(r_ptr, 1, head, offs, mask_ij, rg, rh, rin, rout)
    zf = _mix(h, rf) + wxf
    rz = _load_r_gate(r_ptr, 2, head, offs, mask_ij, rg, rh, rin, rout)
    zz = _mix(h, rz) + wxz
    ro = _load_r_gate(r_ptr, 3, head, offs, mask_ij, rg, rh, rin, rout)
    zo = _mix(h, ro) + wxo
    left = zf + m
    m_new = tl.maximum(left, zi)
    gt = (left > zi).to(tl.float32)
    eq = (left == zi).to(tl.float32)
    alpha = gt + 0.5 * eq
    i_t = tl.exp(zi - m_new)
    f_t = tl.exp(zf + m - m_new)
    z = _tanh(zz)
    n_new = f_t * n + i_t
    c_new = f_t * c + i_t * z
    o = tl.sigmoid(zo)
    denom = n_new + eps
    h_new = o * (c_new / denom)
    return c_new, n_new, m_new, h_new, c, n, m, h, i_t, f_t, z, o, denom, alpha, c_new


@triton.jit
def _slstm_jvp(c, n, i_t, f, z, o, denom, alpha, c_new, ri, rf, rz, ro, vc, vn, vm, vh):
    """Factorized ``J @ v`` (packed ``c,n,m,h``)."""
    beta = 1.0 - alpha
    d_zi = _mix(vh, ri)
    d_zf = _mix(vh, rf)
    d_zz = _mix(vh, rz)
    d_zo = _mix(vh, ro)
    dm_dh_v = alpha * d_zf + beta * d_zi
    di_dm = -i_t * alpha
    df_dm = f * beta
    di_dh_v = i_t * (d_zi - dm_dh_v)
    df_dh_v = f * (d_zf - dm_dh_v)
    dz_dh_v = (1.0 - z * z) * d_zz
    do_dh_v = o * (1.0 - o) * d_zo
    j_ch_v = df_dh_v * c + di_dh_v * z + i_t * dz_dh_v
    j_nh_v = df_dh_v * n + di_dh_v
    inv = o / denom
    dn = -o * c_new / (denom * denom)
    du = c_new / denom
    j_hh_v = inv * j_ch_v + dn * j_nh_v + du * do_dh_v
    j_cm = df_dm * c + di_dm * z
    j_nm = df_dm * n + di_dm
    c_out = f * vc + j_cm * vm + j_ch_v
    n_out = f * vn + j_nm * vm + j_nh_v
    m_out = alpha * vm + dm_dh_v
    h_out = inv * f * vc + dn * f * vn + (inv * j_cm + dn * j_nm) * vm + j_hh_v
    return c_out, n_out, m_out, h_out


@triton.jit
def _slstm_jvp_stream(
    c,
    n,
    i_t,
    f,
    z,
    o,
    denom,
    alpha,
    c_new,
    r_ptr,
    head,
    offs,
    mask_ij,
    rg,
    rh,
    rin,
    rout,
    vc,
    vn,
    vm,
    vh,
):
    ri = _load_r_gate(r_ptr, 0, head, offs, mask_ij, rg, rh, rin, rout)
    rf = _load_r_gate(r_ptr, 1, head, offs, mask_ij, rg, rh, rin, rout)
    rz = _load_r_gate(r_ptr, 2, head, offs, mask_ij, rg, rh, rin, rout)
    ro = _load_r_gate(r_ptr, 3, head, offs, mask_ij, rg, rh, rin, rout)
    return _slstm_jvp(c, n, i_t, f, z, o, denom, alpha, c_new, ri, rf, rz, ro, vc, vn, vm, vh)


@triton.jit
def _slstm_jt(
    c, n, i_t, f, z, o, denom, alpha, c_new, ri, rf, rz, ro, mc, mn, mm, mh
):
    """Factorized ``J^T @ μ``."""
    beta = 1.0 - alpha
    inv = o / denom
    dn = -o * c_new / (denom * denom)
    du = c_new / denom
    di_dm = -i_t * alpha
    df_dm = f * beta
    j_cm = df_dm * c + di_dm * z
    j_nm = df_dm * n + di_dm
    g_c = f * mc + inv * f * mh
    g_n = f * mn + dn * f * mh
    g_m = j_cm * mc + j_nm * mn + alpha * mm + (inv * j_cm + dn * j_nm) * mh
    w_c_like = mc + inv * mh
    w_n_like = mn + dn * mh
    w_do = du * mh
    adj_df = w_c_like * c + w_n_like * n
    adj_di = w_c_like * z + w_n_like
    adj_dz = w_c_like * i_t
    adj_dzf = f * adj_df
    adj_dzi = i_t * adj_di
    adj_dm = -(f * adj_df + i_t * adj_di) + mm
    adj_dzf = adj_dzf + alpha * adj_dm
    adj_dzi = adj_dzi + beta * adj_dm
    adj_dzz = (1.0 - z * z) * adj_dz
    adj_dzo = o * (1.0 - o) * w_do
    g_h = (
        _mix_t(adj_dzi, ri)
        + _mix_t(adj_dzf, rf)
        + _mix_t(adj_dzz, rz)
        + _mix_t(adj_dzo, ro)
    )
    return g_c, g_n, g_m, g_h


@triton.jit
def _slstm_jt_stream(
    c,
    n,
    i_t,
    f,
    z,
    o,
    denom,
    alpha,
    c_new,
    r_ptr,
    head,
    offs,
    mask_ij,
    rg,
    rh,
    rin,
    rout,
    mc,
    mn,
    mm,
    mh,
):
    ri = _load_r_gate(r_ptr, 0, head, offs, mask_ij, rg, rh, rin, rout)
    rf = _load_r_gate(r_ptr, 1, head, offs, mask_ij, rg, rh, rin, rout)
    rz = _load_r_gate(r_ptr, 2, head, offs, mask_ij, rg, rh, rin, rout)
    ro = _load_r_gate(r_ptr, 3, head, offs, mask_ij, rg, rh, rin, rout)
    return _slstm_jt(c, n, i_t, f, z, o, denom, alpha, c_new, ri, rf, rz, ro, mc, mn, mm, mh)


@triton.jit
def _newton_slstm_head_fused_kernel(
    wx_ptr,
    s_ptr,
    d_ptr,
    r_ptr,
    h0_ptr,
    has_h0,
    do_init,
    n_heads,
    time,
    d,
    d_h,
    omega,
    max_iters,
    eps,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_s_b,
    stride_s_t,
    stride_s_s,
    stride_s_d,
    stride_d_b,
    stride_d_t,
    stride_d_s,
    stride_d_d,
    stride_r_g,
    stride_r_h,
    stride_r_in,
    stride_r_out,
    stride_h0_b,
    stride_h0_s,
    stride_h0_d,
    BLOCK_D: tl.constexpr,
):
    """Warm-start or App. A init + K Newton iters; all four ``R_g`` in SRAM."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    ri = _load_r_gate(r_ptr, 0, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    rf = _load_r_gate(r_ptr, 1, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    rz = _load_r_gate(r_ptr, 2, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    ro = _load_r_gate(r_ptr, 3, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)

    c0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    n0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h0v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if has_h0 != 0:
        c0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_C * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        n0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_N * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        m0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_M * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        h0v = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_H * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )

    # App. A guess only when no Picard / zero-hidden warm start was supplied.
    if do_init != 0:
        for t in range(0, time):
            wxi = _load_wx_gate(wx_ptr, b, t, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, t, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, t, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, t, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_p = tl.where(t == 0, c0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            n_p = tl.where(t == 0, n0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            m_p = tl.where(t == 0, m0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            h_p = tl.where(t == 0, h0v, tl.zeros((BLOCK_D,), dtype=tl.float32))
            c_n, n_n, m_n, h_n, _, _, _, _, _, _, _, _, _, _, _ = _slstm_gates(
                c_p, n_p, m_p, h_p, wxi, wxf, wxz, wxo, ri, rf, rz, ro, eps
            )
            _store_slot(s_ptr, c_n, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, n_n, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, m_n, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, h_n, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)

    for _it in range(0, max_iters):
        dc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dn = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dm = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dh = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, time):
            wxi = _load_wx_gate(wx_ptr, b, t, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, t, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, t, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, t, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_C, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_N, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_M, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_H, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            c_p = tl.where(t == 0, c0, c_nm1)
            n_p = tl.where(t == 0, n0, n_nm1)
            m_p = tl.where(t == 0, m0, m_nm1)
            h_p = tl.where(t == 0, h0v, h_nm1)
            c_g = _load_slot(s_ptr, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_g = _load_slot(s_ptr, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_g = _load_slot(s_ptr, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_g = _load_slot(s_ptr, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            (
                c_pred,
                n_pred,
                m_pred,
                h_pred,
                c_act,
                n_act,
                _m_act,
                _h_act,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
            ) = _slstm_gates(c_p, n_p, m_p, h_p, wxi, wxf, wxz, wxo, ri, rf, rz, ro, eps)
            rc = c_pred - c_g
            rn = n_pred - n_g
            rm = m_pred - m_g
            rh = h_pred - h_g
            dc, dn, dm, dh = _slstm_jvp(
                c_act, n_act, i_t, f_t, z, o, denom, alpha, c_new, ri, rf, rz, ro, dc, dn, dm, dh
            )
            dc = dc + rc
            dn = dn + rn
            dm = dm + rm
            dh = dh + rh
            _store_slot(d_ptr, dc, b, t, SLOT_C, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dn, b, t, SLOT_N, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dm, b, t, SLOT_M, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dh, b, t, SLOT_H, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
        for t in range(0, time):
            c_g = _load_slot(s_ptr, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_g = _load_slot(s_ptr, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_g = _load_slot(s_ptr, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_g = _load_slot(s_ptr, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            dc = _load_slot(d_ptr, b, t, SLOT_C, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dn = _load_slot(d_ptr, b, t, SLOT_N, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dm = _load_slot(d_ptr, b, t, SLOT_M, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dh = _load_slot(d_ptr, b, t, SLOT_H, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(s_ptr, c_g + omega * dc, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, n_g + omega * dn, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, m_g + omega * dm, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, h_g + omega * dh, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)


@triton.jit
def _newton_slstm_head_stream_kernel(
    wx_ptr,
    s_ptr,
    d_ptr,
    r_ptr,
    h0_ptr,
    has_h0,
    do_init,
    n_heads,
    time,
    d,
    d_h,
    omega,
    max_iters,
    eps,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_s_b,
    stride_s_t,
    stride_s_s,
    stride_s_d,
    stride_d_b,
    stride_d_t,
    stride_d_s,
    stride_d_d,
    stride_r_g,
    stride_r_h,
    stride_r_in,
    stride_r_out,
    stride_h0_b,
    stride_h0_s,
    stride_h0_d,
    BLOCK_D: tl.constexpr,
):
    """Newton with one ``R_g`` in SRAM at a time (``32 < d ≤ 128``)."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    c0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    n0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    m0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    h0v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if has_h0 != 0:
        c0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_C * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        n0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_N * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        m0 = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_M * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )
        h0v = load_acc(
            h0_ptr + b * stride_h0_b + SLOT_H * stride_h0_s + (head_off + offs) * stride_h0_d,
            mask,
            0.0,
        )

    if do_init != 0:
        for t in range(0, time):
            wxi = _load_wx_gate(wx_ptr, b, t, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, t, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, t, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, t, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_p = tl.where(t == 0, c0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            n_p = tl.where(t == 0, n0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            m_p = tl.where(t == 0, m0, tl.zeros((BLOCK_D,), dtype=tl.float32))
            h_p = tl.where(t == 0, h0v, tl.zeros((BLOCK_D,), dtype=tl.float32))
            c_n, n_n, m_n, h_n, _, _, _, _, _, _, _, _, _, _, _ = _slstm_gates_stream(
                c_p,
                n_p,
                m_p,
                h_p,
                wxi,
                wxf,
                wxz,
                wxo,
                r_ptr,
                head,
                offs,
                mask_ij,
                stride_r_g,
                stride_r_h,
                stride_r_in,
                stride_r_out,
                eps,
            )
            _store_slot(s_ptr, c_n, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, n_n, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, m_n, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, h_n, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)

    for _it in range(0, max_iters):
        dc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dn = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dm = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dh = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, time):
            wxi = _load_wx_gate(wx_ptr, b, t, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, t, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, t, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, t, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_C, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_N, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_M, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_nm1 = _load_slot(s_ptr, b, t - 1, SLOT_H, head_off, offs, mask & (t > 0), stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            c_p = tl.where(t == 0, c0, c_nm1)
            n_p = tl.where(t == 0, n0, n_nm1)
            m_p = tl.where(t == 0, m0, m_nm1)
            h_p = tl.where(t == 0, h0v, h_nm1)
            c_g = _load_slot(s_ptr, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_g = _load_slot(s_ptr, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_g = _load_slot(s_ptr, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_g = _load_slot(s_ptr, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            (
                c_pred,
                n_pred,
                m_pred,
                h_pred,
                c_act,
                n_act,
                _m_act,
                _h_act,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
            ) = _slstm_gates_stream(
                c_p,
                n_p,
                m_p,
                h_p,
                wxi,
                wxf,
                wxz,
                wxo,
                r_ptr,
                head,
                offs,
                mask_ij,
                stride_r_g,
                stride_r_h,
                stride_r_in,
                stride_r_out,
                eps,
            )
            rc = c_pred - c_g
            rn = n_pred - n_g
            rm = m_pred - m_g
            rh = h_pred - h_g
            dc, dn, dm, dh = _slstm_jvp_stream(
                c_act,
                n_act,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
                r_ptr,
                head,
                offs,
                mask_ij,
                stride_r_g,
                stride_r_h,
                stride_r_in,
                stride_r_out,
                dc,
                dn,
                dm,
                dh,
            )
            dc = dc + rc
            dn = dn + rn
            dm = dm + rm
            dh = dh + rh
            _store_slot(d_ptr, dc, b, t, SLOT_C, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dn, b, t, SLOT_N, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dm, b, t, SLOT_M, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(d_ptr, dh, b, t, SLOT_H, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
        for t in range(0, time):
            c_g = _load_slot(s_ptr, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            n_g = _load_slot(s_ptr, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            m_g = _load_slot(s_ptr, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            h_g = _load_slot(s_ptr, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            dc = _load_slot(d_ptr, b, t, SLOT_C, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dn = _load_slot(d_ptr, b, t, SLOT_N, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dm = _load_slot(d_ptr, b, t, SLOT_M, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            dh = _load_slot(d_ptr, b, t, SLOT_H, head_off, offs, mask, stride_d_b, stride_d_t, stride_d_s, stride_d_d)
            _store_slot(s_ptr, c_g + omega * dc, b, t, SLOT_C, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, n_g + omega * dn, b, t, SLOT_N, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, m_g + omega * dm, b, t, SLOT_M, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)
            _store_slot(s_ptr, h_g + omega * dh, b, t, SLOT_H, head_off, offs, mask, stride_s_b, stride_s_t, stride_s_s, stride_s_d)


@triton.jit
def _reverse_slstm_head_fused_kernel(
    wx_ptr,
    hp_ptr,
    part_ptr,
    out_ptr,
    r_ptr,
    n_heads,
    time,
    d,
    d_h,
    eps,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_hp_b,
    stride_hp_t,
    stride_hp_s,
    stride_hp_d,
    stride_p_b,
    stride_p_t,
    stride_p_s,
    stride_p_d,
    stride_o_b,
    stride_o_t,
    stride_o_s,
    stride_o_d,
    stride_r_g,
    stride_r_h,
    stride_r_in,
    stride_r_out,
    BLOCK_D: tl.constexpr,
):
    """Eq. 2.6 reverse; all four ``R_g`` in SRAM."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    ri = _load_r_gate(r_ptr, 0, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    rf = _load_r_gate(r_ptr, 1, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    rz = _load_r_gate(r_ptr, 2, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)
    ro = _load_r_gate(r_ptr, 3, head, offs, mask_ij, stride_r_g, stride_r_h, stride_r_in, stride_r_out)

    mc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mn = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mm = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mh = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            tp1 = t + 1
            wxi = _load_wx_gate(wx_ptr, b, tp1, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, tp1, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, tp1, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, tp1, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_p = _load_slot(hp_ptr, b, tp1, SLOT_C, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            n_p = _load_slot(hp_ptr, b, tp1, SLOT_N, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            m_p = _load_slot(hp_ptr, b, tp1, SLOT_M, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            h_p = _load_slot(hp_ptr, b, tp1, SLOT_H, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            (
                _cp,
                _np,
                _mp,
                _hp,
                c_act,
                n_act,
                _ma,
                _ha,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
            ) = _slstm_gates(c_p, n_p, m_p, h_p, wxi, wxf, wxz, wxo, ri, rf, rz, ro, eps)
            mc, mn, mm, mh = _slstm_jt(
                c_act, n_act, i_t, f_t, z, o, denom, alpha, c_new, ri, rf, rz, ro, mc, mn, mm, mh
            )
        pc = _load_slot(part_ptr, b, t, SLOT_C, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        pn = _load_slot(part_ptr, b, t, SLOT_N, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        pm = _load_slot(part_ptr, b, t, SLOT_M, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        ph = _load_slot(part_ptr, b, t, SLOT_H, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        mc = mc + pc
        mn = mn + pn
        mm = mm + pm
        mh = mh + ph
        _store_slot(out_ptr, mc, b, t, SLOT_C, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mn, b, t, SLOT_N, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mm, b, t, SLOT_M, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mh, b, t, SLOT_H, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)


@triton.jit
def _reverse_slstm_head_stream_kernel(
    wx_ptr,
    hp_ptr,
    part_ptr,
    out_ptr,
    r_ptr,
    n_heads,
    time,
    d,
    d_h,
    eps,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_hp_b,
    stride_hp_t,
    stride_hp_s,
    stride_hp_d,
    stride_p_b,
    stride_p_t,
    stride_p_s,
    stride_p_d,
    stride_o_b,
    stride_o_t,
    stride_o_s,
    stride_o_d,
    stride_r_g,
    stride_r_h,
    stride_r_in,
    stride_r_out,
    BLOCK_D: tl.constexpr,
):
    """Eq. 2.6 reverse; stream one ``R`` at a time."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    mc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mn = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mm = tl.zeros((BLOCK_D,), dtype=tl.float32)
    mh = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            tp1 = t + 1
            wxi = _load_wx_gate(wx_ptr, b, tp1, 0, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxf = _load_wx_gate(wx_ptr, b, tp1, 1, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxz = _load_wx_gate(wx_ptr, b, tp1, 2, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            wxo = _load_wx_gate(wx_ptr, b, tp1, 3, d_h, head_off, offs, mask, stride_wx_b, stride_wx_t, stride_wx_d)
            c_p = _load_slot(hp_ptr, b, tp1, SLOT_C, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            n_p = _load_slot(hp_ptr, b, tp1, SLOT_N, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            m_p = _load_slot(hp_ptr, b, tp1, SLOT_M, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            h_p = _load_slot(hp_ptr, b, tp1, SLOT_H, head_off, offs, mask, stride_hp_b, stride_hp_t, stride_hp_s, stride_hp_d)
            (
                _cp,
                _np,
                _mp,
                _hp,
                c_act,
                n_act,
                _ma,
                _ha,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
            ) = _slstm_gates_stream(
                c_p,
                n_p,
                m_p,
                h_p,
                wxi,
                wxf,
                wxz,
                wxo,
                r_ptr,
                head,
                offs,
                mask_ij,
                stride_r_g,
                stride_r_h,
                stride_r_in,
                stride_r_out,
                eps,
            )
            mc, mn, mm, mh = _slstm_jt_stream(
                c_act,
                n_act,
                i_t,
                f_t,
                z,
                o,
                denom,
                alpha,
                c_new,
                r_ptr,
                head,
                offs,
                mask_ij,
                stride_r_g,
                stride_r_h,
                stride_r_in,
                stride_r_out,
                mc,
                mn,
                mm,
                mh,
            )
        pc = _load_slot(part_ptr, b, t, SLOT_C, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        pn = _load_slot(part_ptr, b, t, SLOT_N, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        pm = _load_slot(part_ptr, b, t, SLOT_M, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        ph = _load_slot(part_ptr, b, t, SLOT_H, head_off, offs, mask, stride_p_b, stride_p_t, stride_p_s, stride_p_d)
        mc = mc + pc
        mn = mn + pn
        mm = mm + pm
        mh = mh + ph
        _store_slot(out_ptr, mc, b, t, SLOT_C, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mn, b, t, SLOT_N, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mm, b, t, SLOT_M, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)
        _store_slot(out_ptr, mh, b, t, SLOT_H, head_off, offs, mask, stride_o_b, stride_o_t, stride_o_s, stride_o_d)


def _newton_fused_triton(
    wx: Tensor,
    r_head: Tensor,
    h0: Tensor | None,
    *,
    states: Tensor | None,
    max_iters: int,
    omega: float,
    eps: float,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time, _ = wx.shape
    d_h = n_heads * d_head
    do_init = 1 if states is None else 0
    if states is None:
        states_buf = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        states_buf = states.contiguous().clone()
    delta = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    wx = wx.contiguous()
    r_head = r_head.contiguous()
    has_h0 = 0 if h0 is None else 1
    if h0 is None:
        h0_t = wx.new_empty(0)
        h0_strides = (0, 0, 0)
    else:
        h0_t = h0.contiguous()
        h0_strides = h0_t.stride()
    block = _block_d(d_head, _BLOCK_D)
    _newton_slstm_head_fused_kernel[(batch * n_heads,)](
        wx,
        states_buf,
        delta,
        r_head,
        h0_t,
        has_h0,
        do_init,
        n_heads,
        time,
        d_head,
        d_h,
        float(omega),
        max_iters,
        float(eps),
        *wx.stride(),
        *states_buf.stride(),
        *delta.stride(),
        *r_head.stride(),
        *h0_strides,
        BLOCK_D=block,
    )
    return states_buf


def _newton_stream_r_triton(
    wx: Tensor,
    r_head: Tensor,
    h0: Tensor | None,
    *,
    states: Tensor | None,
    max_iters: int,
    omega: float,
    eps: float,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time, _ = wx.shape
    d_h = n_heads * d_head
    do_init = 1 if states is None else 0
    if states is None:
        states_buf = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    else:
        states_buf = states.contiguous().clone()
    delta = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
    wx = wx.contiguous()
    r_head = r_head.contiguous()
    has_h0 = 0 if h0 is None else 1
    if h0 is None:
        h0_t = wx.new_empty(0)
        h0_strides = (0, 0, 0)
    else:
        h0_t = h0.contiguous()
        h0_strides = h0_t.stride()
    block = _block_d(d_head, _NEWTON_SRAM_D)
    _newton_slstm_head_stream_kernel[(batch * n_heads,)](
        wx,
        states_buf,
        delta,
        r_head,
        h0_t,
        has_h0,
        do_init,
        n_heads,
        time,
        d_head,
        d_h,
        float(omega),
        max_iters,
        float(eps),
        *wx.stride(),
        *states_buf.stride(),
        *delta.stride(),
        *r_head.stride(),
        *h0_strides,
        BLOCK_D=block,
    )
    return states_buf


def _reverse_fused_triton(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> Tensor:
    out = torch.empty_like(partial)
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    partial = partial.contiguous()
    r_head = r_head.contiguous()
    batch = partial.shape[0]
    time = partial.shape[1]
    d_h = n_heads * d_head
    block = _block_d(d_head, _BLOCK_D)
    _reverse_slstm_head_fused_kernel[(batch * n_heads,)](
        wx,
        h_prev,
        partial,
        out,
        r_head,
        n_heads,
        time,
        d_head,
        d_h,
        float(eps),
        *wx.stride(),
        *h_prev.stride(),
        *partial.stride(),
        *out.stride(),
        *r_head.stride(),
        BLOCK_D=block,
    )
    return out


def _reverse_stream_r_triton(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> Tensor:
    out = torch.empty_like(partial)
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    partial = partial.contiguous()
    r_head = r_head.contiguous()
    batch = partial.shape[0]
    time = partial.shape[1]
    d_h = n_heads * d_head
    block = _block_d(d_head, _REVERSE_SRAM_D)
    _reverse_slstm_head_stream_kernel[(batch * n_heads,)](
        wx,
        h_prev,
        partial,
        out,
        r_head,
        n_heads,
        time,
        d_head,
        d_h,
        float(eps),
        *wx.stride(),
        *h_prev.stride(),
        *partial.stride(),
        *out.stride(),
        *r_head.stride(),
        BLOCK_D=block,
    )
    return out
