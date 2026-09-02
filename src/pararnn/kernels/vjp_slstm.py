"""Triton VJP of the channelwise ParaSLSTM recurrence (Beck et al. 2024).

``state_prev`` detached (eq. 2.6 already scanned ``J^T``). ``W_x`` GEMM stays
in PyTorch. ``mix='diag'`` only. Dtype gate is ``validate_cuda_tensors``
(bf16 if CC ≥ 8.0).

Formulas live in ``slstm_recurrence_vjp_eager``. The kernel is the fused
CUDA implementation of the same lines. ``∇R``: tile ``tl.sum`` into fp32
``(B, n_tiles, d_h)``, then ``.sum`` — deterministic, no atomics. ``∇wx``
stays ``(B, T, 4 d_h)`` for the GEMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_STABILIZER,
)

_BLOCK_T = 64
_BLOCK_D = 32
SLOT_C = tl.constexpr(0)
SLOT_N = tl.constexpr(1)
SLOT_M = tl.constexpr(2)
SLOT_H = tl.constexpr(3)


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _load_state(s_ptr, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    return load_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _slstm_vjp_kernel(
    s_ptr,
    wx_ptr,
    ri_ptr,
    rf_ptr,
    rz_ptr,
    ro_ptr,
    mu_ptr,
    gwx_ptr,
    gri_ptr,
    grf_ptr,
    grz_ptr,
    gro_ptr,
    time,
    d_h,
    eps,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_mb,
    stride_mt,
    stride_ms,
    stride_md,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_ab,
    stride_ac,
    stride_ad,
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

    c_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    mu_c = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_n = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_m = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_h = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_mb, stride_mt, stride_ms, stride_md
    )

    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    wx_i = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    wx_f = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    wx_z = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    wx_o = load_acc(base + (offs_d[None, :] + 3 * d_h) * stride_wd, mask, 0.0)
    r_i = load_acc(ri_ptr + offs_d, dmask, 0.0)
    r_f = load_acc(rf_ptr + offs_d, dmask, 0.0)
    r_z = load_acc(rz_ptr + offs_d, dmask, 0.0)
    r_o = load_acc(ro_ptr + offs_d, dmask, 0.0)

    z_i = wx_i + r_i * h_prev
    z_f = wx_f + r_f * h_prev
    z_z = wx_z + r_z * h_prev
    z_o = wx_o + r_o * h_prev
    left = z_f + m_prev
    m_new = tl.maximum(left, z_i)
    # ∂maximum(left, z_i)/∂left; ties 0.5/0.5 like torch.maximum.
    alpha = tl.where(left > z_i, 1.0, 0.0) + 0.5 * tl.where(left == z_i, 1.0, 0.0)
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = 1.0 / (1.0 + tl.exp(-z_o))
    denom = n_new + eps

    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / (denom * denom))
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z * z)
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_zi, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_zf, mask)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, d_zz, mask)
    store_acc(gout + (offs_d[None, :] + 3 * d_h) * stride_gd, d_zo, mask)

    off = pid_b * stride_ab + pid_c * stride_ac + offs_d * stride_ad
    store_acc(gri_ptr + off, tl.sum(tl.where(mask, d_zi * h_prev, 0.0), 0), dmask)
    store_acc(grf_ptr + off, tl.sum(tl.where(mask, d_zf * h_prev, 0.0), 0), dmask)
    store_acc(grz_ptr + off, tl.sum(tl.where(mask, d_zz * h_prev, 0.0), 0), dmask)
    store_acc(gro_ptr + off, tl.sum(tl.where(mask, d_zo * h_prev, 0.0), 0), dmask)


def slstm_recurrence_vjp_eager(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    mu: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Channelwise sLSTM VJP. fp32 algebra, fp32 ``(B, T)`` reduce, cast to ``wx.dtype``."""
    dt = wx.dtype
    state_prev = state_prev.float()
    wx = wx.float()
    r = r.float()
    mu = mu.float()
    c_prev = state_prev[..., SLSTM_CELL, :]
    n_prev = state_prev[..., SLSTM_NORMALIZER, :]
    m_prev = state_prev[..., SLSTM_STABILIZER, :]
    h_prev = state_prev[..., SLSTM_HIDDEN, :]
    mu_c = mu[..., SLSTM_CELL, :]
    mu_n = mu[..., SLSTM_NORMALIZER, :]
    mu_m = mu[..., SLSTM_STABILIZER, :]
    mu_h = mu[..., SLSTM_HIDDEN, :]
    wx_i, wx_f, wx_z, wx_o = wx.chunk(4, dim=-1)
    r_i, r_f, r_z, r_o = r.unbind(0)
    z_i = wx_i + r_i * h_prev
    z_f = wx_f + r_f * h_prev
    z_z = wx_z + r_z * h_prev
    z_o = wx_o + r_o * h_prev
    left = z_f + m_prev
    m_new = torch.maximum(left, z_i)
    gt = (left > z_i).to(dtype=left.dtype)
    eq = (left == z_i).to(dtype=left.dtype)
    alpha = gt + 0.5 * eq
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m_prev - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + eps
    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / denom.square())
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z.square())
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha
    g_wx = torch.cat((d_zi, d_zf, d_zz, d_zo), dim=-1)
    dims = (0, 1)
    g_r = torch.stack(
        (
            (d_zi * h_prev).sum(dim=dims),
            (d_zf * h_prev).sum(dim=dims),
            (d_zz * h_prev).sum(dim=dims),
            (d_zo * h_prev).sum(dim=dims),
        ),
        dim=0,
    )
    return g_wx.to(dt), g_r.to(dt)


def slstm_recurrence_vjp(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    mu: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    if not state_prev.is_cuda:
        return slstm_recurrence_vjp_eager(state_prev, wx, r, mu, eps)
    validate_cuda_tensors(state_prev, wx, r, mu, name="slstm_recurrence_vjp")
    state_prev = state_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    r = r.contiguous()
    r_i, r_f, r_z, r_o = r.unbind(0)
    batch, time, _, d_h = state_prev.shape
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    g_wx = wx.new_empty(wx.shape)
    acc_shape = (batch, n_chunks, d_h)
    g_ri = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_rf = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_rz = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    g_ro = torch.zeros(acc_shape, device=state_prev.device, dtype=torch.float32)
    _slstm_vjp_kernel[(batch, n_chunks, n_dtiles)](
        state_prev,
        wx,
        r_i,
        r_f,
        r_z,
        r_o,
        mu,
        g_wx,
        g_ri,
        g_rf,
        g_rz,
        g_ro,
        time,
        d_h,
        float(eps),
        *state_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_ri.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    dt = r.dtype
    dims = (0, 1)
    g_r = torch.stack(
        (
            g_ri.sum(dim=dims),
            g_rf.sum(dim=dims),
            g_rz.sum(dim=dims),
            g_ro.sum(dim=dims),
        ),
        dim=0,
    )
    return g_wx, g_r.to(dt)
