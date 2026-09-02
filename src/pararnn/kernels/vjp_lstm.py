"""Triton VJP of the CIFG peephole ParaLSTM recurrence (eq. 3.1b).

``state_prev`` detached. ``W_x`` GEMM stays in PyTorch.
Dtype gate is ``validate_cuda_tensors`` (bf16 if CC ≥ 8.0).
"""

from __future__ import annotations

import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors

_BLOCK_T = 64
_BLOCK_D = 32
SLOT_C = tl.constexpr(0)
SLOT_H = tl.constexpr(1)


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
def _lstm_vjp_kernel(
    s_ptr,
    wx_ptr,
    af_ptr,
    az_ptr,
    ao_ptr,
    cf_ptr,
    co_ptr,
    mu_ptr,
    gwx_ptr,
    gaf_ptr,
    gaz_ptr,
    gao_ptr,
    gcf_ptr,
    gco_ptr,
    time,
    d_h,
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
    stride_at,
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
    h_prev = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    mu_c = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_mb, stride_mt, stride_ms, stride_md
    )
    mu_h = _load_state(
        mu_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_mb, stride_mt, stride_ms, stride_md
    )

    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    fx = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    zx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    ox = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    a_f = load_acc(af_ptr + offs_d, dmask, 0.0)
    a_z = load_acc(az_ptr + offs_d, dmask, 0.0)
    a_o = load_acc(ao_ptr + offs_d, dmask, 0.0)
    peephole_f = load_acc(cf_ptr + offs_d, dmask, 0.0)
    peephole_o = load_acc(co_ptr + offs_d, dmask, 0.0)

    f = 1.0 / (1.0 + tl.exp(-(a_f * h_prev + peephole_f * c_prev + fx)))
    z = _tanh(a_z * h_prev + zx)
    c = f * c_prev + (1.0 - f) * z
    o = 1.0 / (1.0 + tl.exp(-(a_o * h_prev + peephole_o * c + ox)))
    h_act = _tanh(c)

    d_o = mu_h * h_act
    d_h_act = mu_h * o
    d_opre = d_o * o * (1.0 - o)
    d_c = mu_c + d_h_act * (1.0 - h_act * h_act) + d_opre * peephole_o
    d_f = d_c * (c_prev - z)
    d_z = d_c * (1.0 - f)
    d_zpre = d_z * (1.0 - z * z)
    d_fpre = d_f * f * (1.0 - f)

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_fpre, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_zpre, mask)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, d_opre, mask)

    acc = pid_b * stride_ab + offs_t[:, None] * stride_at + offs_d[None, :] * stride_ad
    store_acc(gaf_ptr + acc, d_fpre * h_prev, mask)
    store_acc(gaz_ptr + acc, d_zpre * h_prev, mask)
    store_acc(gao_ptr + acc, d_opre * h_prev, mask)
    store_acc(gcf_ptr + acc, d_fpre * c_prev, mask)
    store_acc(gco_ptr + acc, d_opre * c, mask)


def lstm_recurrence_vjp(
    state_prev: Tensor,
    wx: Tensor,
    a_f: Tensor,
    a_z: Tensor,
    a_o: Tensor,
    c_f: Tensor,
    c_o: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    validate_cuda_tensors(state_prev, wx, a_f, a_z, a_o, c_f, c_o, mu, name="lstm_recurrence_vjp")
    state_prev = state_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    a_f = a_f.contiguous()
    a_z = a_z.contiguous()
    a_o = a_o.contiguous()
    c_f = c_f.contiguous()
    c_o = c_o.contiguous()
    batch, time, _, d_h = state_prev.shape
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    g_wx = wx.new_empty(wx.shape)
    acc_shape = (batch, time, d_h)
    g_af_bt = wx.new_empty(acc_shape)
    g_az_bt = wx.new_empty(acc_shape)
    g_ao_bt = wx.new_empty(acc_shape)
    g_cf_bt = wx.new_empty(acc_shape)
    g_co_bt = wx.new_empty(acc_shape)
    _lstm_vjp_kernel[(batch, n_chunks, n_dtiles)](
        state_prev,
        wx,
        a_f,
        a_z,
        a_o,
        c_f,
        c_o,
        mu,
        g_wx,
        g_af_bt,
        g_az_bt,
        g_ao_bt,
        g_cf_bt,
        g_co_bt,
        time,
        d_h,
        *state_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_af_bt.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    dims = (0, 1)
    return (
        g_wx,
        g_af_bt.sum(dim=dims),
        g_az_bt.sum(dim=dims),
        g_ao_bt.sum(dim=dims),
        g_cf_bt.sum(dim=dims),
        g_co_bt.sum(dim=dims),
    )
