"""Triton / eager VJP of the ParaNLRU recurrence (diag ``u``).

``h_prev`` is detached (eq. 2.6 already applied ``J^T``). ``∇u`` reduces with
per-batch fp32 tiles then ``.sum`` — no atomics.
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
def _nlru_vjp_kernel(
    h_ptr,
    wx_ptr,
    u_ptr,
    mu_ptr,
    gwx_ptr,
    gu_ptr,
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
    stride_ub,
    stride_uc,
    stride_ud,
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
    ax = load_acc(base + offs_d[None, :] * stride_wd, mask, 0.0)
    cx = load_acc(base + (offs_d[None, :] + d_h) * stride_wd, mask, 0.0)
    u = load_acc(u_ptr + offs_d[None, :], dmask[None, :], 0.0)

    a = tl.sigmoid(ax)
    n = _nv_tanh(cx + u * h)
    d_h_da = mu * (h - n)
    d_apre = d_h_da * a * (1.0 - a)
    d_npre = mu * (1.0 - a) * (1.0 - n * n)

    gout = gwx_ptr + pid_b * stride_gb + offs_t[:, None] * stride_gt
    store_acc(gout + offs_d[None, :] * stride_gd, d_apre, mask)
    store_acc(gout + (offs_d[None, :] + d_h) * stride_gd, d_npre, mask)

    off = pid_b * stride_ub + pid_c * stride_uc + offs_d * stride_ud
    store_acc(gu_ptr + off, tl.sum(tl.where(mask, d_npre * h, 0.0), 0), dmask)


def nlru_recurrence_vjp_eager(
    h_prev: Tensor,
    wx: Tensor,
    u: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor]:
    """NLRU VJP. Returns ``(∇wx, ∇u)`` in ``h_prev.dtype``."""
    dt = h_prev.dtype
    h_prev = h_prev.float()
    wx = wx.float()
    u = u.float()
    mu = mu.float()
    ax, cx = wx.chunk(2, dim=-1)
    a = torch.sigmoid(ax)
    n = torch.tanh(cx + u * h_prev)
    d_h_da = mu * (h_prev - n)
    d_apre = d_h_da * a * (1.0 - a)
    d_npre = mu * (1.0 - a) * (1.0 - n.square())
    g_wx = torch.cat((d_apre, d_npre), dim=-1)
    g_u = (d_npre * h_prev).sum(dim=(0, 1))
    return g_wx.to(dt), g_u.to(dt)


def nlru_recurrence_vjp(
    h_prev: Tensor,
    wx: Tensor,
    u: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor]:
    """CUDA tile VJP when available; eager otherwise."""
    if h_prev.device.type != "cuda":
        return nlru_recurrence_vjp_eager(h_prev, wx, u, mu)
    validate_cuda_tensors(h_prev, wx, u, mu, name="nlru_recurrence_vjp")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    u = u.contiguous()
    mu = mu.contiguous()
    batch, time, d_h = h_prev.shape
    if wx.shape[-1] != 2 * d_h:
        raise ValueError(f"wx last dim {wx.shape[-1]} != 2*d_h={2 * d_h}")
    n_chunks = triton.cdiv(time, _BLOCK_T)
    n_dtiles = triton.cdiv(d_h, _BLOCK_D)
    g_wx = wx.new_empty(batch, time, 2 * d_h)
    gu_tiles = wx.new_empty(batch, n_chunks, d_h)
    _nlru_vjp_kernel[(batch, n_chunks, n_dtiles)](
        h_prev,
        wx,
        u,
        mu,
        g_wx,
        gu_tiles,
        time,
        d_h,
        *h_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *gu_tiles.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    g_u = gu_tiles.float().sum(dim=(0, 1)).to(dtype=h_prev.dtype)
    return g_wx, g_u
