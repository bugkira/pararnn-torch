"""Triton VJP of the ParaGRU recurrence w.r.t. ``wx`` and ``a_*`` (eq. 3.1a).

``h_prev`` is detached (eq. 2.6 already scanned ``J^T``). ``W_x`` GEMM stays
in PyTorch. Not a translation of anyone else's kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import check_cuda_real, load_acc, store_acc

_BLOCK_T = 128
_BLOCK_D = 32


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _gru_vjp_kernel(
    h_ptr,
    wx_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    mu_ptr,
    gwx_ptr,
    gaz_ptr,
    gar_ptr,
    gan_ptr,
    time,
    d_h,
    stride_hb,
    stride_ht,
    stride_hd,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_mb,
    stride_mt,
    stride_md,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_azb,
    stride_azt,
    stride_azd,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
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

    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
        FP16,
    )
    mu = load_acc(
        mu_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask,
        0.0,
        FP16,
    )
    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    zx = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0, FP16)
    rx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0, FP16)
    nx = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0, FP16)
    az = load_acc(az_ptr + offs_d, dmask, 0.0, FP16)
    ar = load_acc(ar_ptr + offs_d, dmask, 0.0, FP16)
    an = load_acc(an_ptr + offs_d, dmask, 0.0, FP16)

    z = 1.0 / (1.0 + tl.exp(-(az * h + zx)))
    r = 1.0 / (1.0 + tl.exp(-(ar * h + rx)))
    n = _tanh(an * (h * r) + nx)

    d_z = mu * (n - h)
    d_npre = mu * z * (1.0 - n * n)
    d_zpre = d_z * z * (1.0 - z)
    d_rpre = (d_npre * an * h) * r * (1.0 - r)

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_zpre, mask, FP16)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_rpre, mask, FP16)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, d_npre, mask, FP16)

    store_acc(
        gaz_ptr + pid_b * stride_azb + offs_t[:, None] * stride_azt + offs_d[None, :] * stride_azd,
        d_zpre * h,
        mask,
        FP16,
    )
    store_acc(
        gar_ptr + pid_b * stride_azb + offs_t[:, None] * stride_azt + offs_d[None, :] * stride_azd,
        d_rpre * h,
        mask,
        FP16,
    )
    store_acc(
        gan_ptr + pid_b * stride_azb + offs_t[:, None] * stride_azt + offs_d[None, :] * stride_azd,
        d_npre * (h * r),
        mask,
        FP16,
    )


def gru_recurrence_vjp(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    fp16 = check_cuda_real(h_prev, wx, a_z, a_r, a_n, mu, name="gru_recurrence_vjp")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    a_z = a_z.contiguous()
    a_r = a_r.contiguous()
    a_n = a_n.contiguous()
    batch, time, d_h = h_prev.shape
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    g_wx = wx.new_empty(wx.shape)
    g_az_bt = h_prev.new_empty(h_prev.shape)
    g_ar_bt = h_prev.new_empty(h_prev.shape)
    g_an_bt = h_prev.new_empty(h_prev.shape)
    _gru_vjp_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        a_z,
        a_r,
        a_n,
        mu,
        g_wx,
        g_az_bt,
        g_ar_bt,
        g_an_bt,
        time,
        d_h,
        *h_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_az_bt.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        FP16=fp16,
    )
    dims = (0, 1)
    return g_wx, g_az_bt.sum(dim=dims), g_ar_bt.sum(dim=dims), g_an_bt.sum(dim=dims)
