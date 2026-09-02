"""Triton VJP of the ParaGRU recurrence w.r.t. ``wx`` and ``a_*`` (eq. 3.1a).

``h_prev`` detached (eq. 2.6 already scanned ``J^T``). ``W_x`` GEMM stays
in PyTorch. Dtype gate is ``validate_cuda_tensors`` (bf16 if CC ≥ 8.0).

Formulas live in ``gru_recurrence_vjp_eager`` (eq. 3.1a). The kernel is the
fused CUDA implementation of the same lines. ``∇a_*``: tile ``tl.sum`` into
fp32 ``(B, n_tiles, d_h)``, then ``.sum`` — deterministic, no atomics.
``∇wx`` stays ``(B, T, 3 d_h)`` for the GEMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors

_BLOCK_T = 128
_BLOCK_D = 32


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

    h = load_acc(
        h_ptr + pid_b * stride_hb + offs_t[:, None] * stride_ht + offs_d[None, :] * stride_hd,
        mask,
        0.0,
    )
    mu = load_acc(
        mu_ptr + pid_b * stride_mb + offs_t[:, None] * stride_mt + offs_d[None, :] * stride_md,
        mask,
        0.0,
    )
    base = wx_ptr + pid_b * stride_wb + offs_t[:, None] * stride_wt
    zx = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    rx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    nx = load_acc(base + (offs_d[None, :] + 2 * d_h) * stride_wd, mask, 0.0)
    # (1, BLOCK_D) so az * h is rank-aligned (h is (BLOCK_T, BLOCK_D)).
    az = load_acc(az_ptr + offs_d[None, :], dmask[None, :], 0.0)
    ar = load_acc(ar_ptr + offs_d[None, :], dmask[None, :], 0.0)
    an = load_acc(an_ptr + offs_d[None, :], dmask[None, :], 0.0)

    z = tl.sigmoid(az * h + zx)
    r = tl.sigmoid(ar * h + rx)
    n = _nv_tanh(an * (h * r) + nx)

    d_z = mu * (n - h)
    d_npre = mu * z * (1.0 - n * n)
    d_zpre = d_z * z * (1.0 - z)
    d_rpre = (d_npre * an * h) * r * (1.0 - r)

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_zpre, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_rpre, mask)
    store_acc(gout + (offs_d[None, :] + 2 * d_h) * stride_gd, d_npre, mask)

    off = pid_b * stride_ab + pid_c * stride_ac + offs_d * stride_ad
    store_acc(gaz_ptr + off, tl.sum(tl.where(mask, d_zpre * h, 0.0), 0), dmask)
    store_acc(gar_ptr + off, tl.sum(tl.where(mask, d_rpre * h, 0.0), 0), dmask)
    store_acc(gan_ptr + off, tl.sum(tl.where(mask, d_npre * (h * r), 0.0), 0), dmask)


def gru_recurrence_vjp_eager(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eq. 3.1a VJP. fp32 algebra, fp32 ``(B, T)`` reduce, cast to ``h_prev.dtype``."""
    dt = h_prev.dtype
    h_prev = h_prev.float()
    wx = wx.float()
    a_z = a_z.float()
    a_r = a_r.float()
    a_n = a_n.float()
    mu = mu.float()
    zx, rx, nx = wx.chunk(3, dim=-1)
    z = torch.sigmoid(a_z * h_prev + zx)
    r = torch.sigmoid(a_r * h_prev + rx)
    n = torch.tanh(a_n * (h_prev * r) + nx)
    d_z = mu * (n - h_prev)
    d_npre = mu * z * (1.0 - n.square())
    d_zpre = d_z * z * (1.0 - z)
    d_rpre = (d_npre * a_n * h_prev) * r * (1.0 - r)
    g_wx = torch.cat((d_zpre, d_rpre, d_npre), dim=-1)
    dims = (0, 1)
    return (
        g_wx.to(dt),
        (d_zpre * h_prev).sum(dim=dims).to(dt),
        (d_rpre * h_prev).sum(dim=dims).to(dt),
        (d_npre * (h_prev * r)).sum(dim=dims).to(dt),
    )


def gru_recurrence_vjp(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not h_prev.is_cuda:
        return gru_recurrence_vjp_eager(h_prev, wx, a_z, a_r, a_n, mu)
    validate_cuda_tensors(h_prev, wx, a_z, a_r, a_n, mu, name="gru_recurrence_vjp")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    a_z = a_z.contiguous()
    a_r = a_r.contiguous()
    a_n = a_n.contiguous()
    batch, time, d_h = h_prev.shape
    if wx.shape != (batch, time, 3 * d_h):
        raise ValueError(
            f"gru_recurrence_vjp: wx shape {tuple(wx.shape)} != {(batch, time, 3 * d_h)}"
        )
    if mu.shape != (batch, time, d_h):
        raise ValueError(f"gru_recurrence_vjp: mu shape {tuple(mu.shape)} != {(batch, time, d_h)}")
    if a_z.shape != (d_h,) or a_r.shape != (d_h,) or a_n.shape != (d_h,):
        raise ValueError(
            f"gru_recurrence_vjp: a_* must be ({d_h},), got "
            f"{tuple(a_z.shape)}, {tuple(a_r.shape)}, {tuple(a_n.shape)}"
        )
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    g_wx = wx.new_empty(wx.shape)
    acc_shape = (batch, n_chunks, d_h)
    g_az = torch.zeros(acc_shape, device=h_prev.device, dtype=torch.float32)
    g_ar = torch.zeros(acc_shape, device=h_prev.device, dtype=torch.float32)
    g_an = torch.zeros(acc_shape, device=h_prev.device, dtype=torch.float32)
    _gru_vjp_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        a_z,
        a_r,
        a_n,
        mu,
        g_wx,
        g_az,
        g_ar,
        g_an,
        time,
        d_h,
        *h_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *g_az.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    dt = a_z.dtype
    dims = (0, 1)
    return g_wx, g_az.sum(dim=dims).to(dt), g_ar.sum(dim=dims).to(dt), g_an.sum(dim=dims).to(dt)
