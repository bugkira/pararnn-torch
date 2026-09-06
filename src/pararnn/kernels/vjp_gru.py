"""Triton VJP of the ParaGRU recurrence w.r.t. ``wx`` and ``a_*`` (eq. 3.1a).

``h_prev`` detached (eq. 2.6 already scanned ``J^T``). ``W_x`` GEMM stays
in PyTorch. Dtype gate is ``validate_cuda_tensors`` (bf16 if CC ≥ 8.0).

Diag: ``gru_recurrence_vjp`` / ``gru_recurrence_vjp_eager``. Head:
``gru_head_recurrence_vjp`` (SRAM ``d_head≤64``, tiled otherwise) /
``gru_head_recurrence_vjp_eager``. ``∇A``: per-batch fp32 tiles + ``.sum`` —
deterministic, no atomics. ``∇wx`` is ``(B, T, 3 d_h)`` for the GEMM.
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


def gru_head_recurrence_vjp_eager(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eager block-diagonal GRU VJP (``mix='head'`` recurrence).

    ``a_*`` use layout ``(H, d_in, d_out)`` with ``y = h @ A``. Algebra in
    fp32; outputs cast to ``h_prev.dtype``. Reference path for CPU and for
    checking the CUDA SRAM / tiled kernels.

    Parameters
    ----------
    h_prev : Tensor
        Previous hidden. Tensor of shape ``(B, T, d_h)``.
    wx : Tensor
        Precomputed ``W_x(x)``. Tensor of shape ``(B, T, 3 d_h)``.
    a_z, a_r, a_n : Tensor
        Per-head recurrent matrices. Each of shape
        ``(n_heads, d_head, d_head)``.
    mu : Tensor
        Upstream adjoint w.r.t. ``h_new``. Tensor of shape ``(B, T, d_h)``.
    n_heads : int
        Head count; ``d_h == n_heads * d_head``.
    d_head : int
        Width of each head block.

    Returns
    -------
    g_wx : Tensor
        Gradient w.r.t. ``wx``. Tensor of shape ``(B, T, 3 d_h)``.
    g_az, g_ar, g_an : Tensor
        Gradients w.r.t. ``A_*``. Each of shape ``(n_heads, d_head, d_head)``.
    """
    dt = h_prev.dtype
    prefix = h_prev.shape[:-1]
    h = h_prev.float().reshape(*prefix, n_heads, d_head)
    mu_h = mu.float().reshape(*prefix, n_heads, d_head)
    wx_f = wx.float()
    a_z = a_z.float()
    a_r = a_r.float()
    a_n = a_n.float()
    zx, rx, nx = wx_f.chunk(3, dim=-1)
    zx = zx.reshape(*prefix, n_heads, d_head)
    rx = rx.reshape(*prefix, n_heads, d_head)
    nx = nx.reshape(*prefix, n_heads, d_head)

    z_pre = torch.matmul(h.unsqueeze(-2), a_z).squeeze(-2) + zx
    r_pre = torch.matmul(h.unsqueeze(-2), a_r).squeeze(-2) + rx
    z = torch.sigmoid(z_pre)
    r = torch.sigmoid(r_pre)
    u = h * r
    n_pre = torch.matmul(u.unsqueeze(-2), a_n).squeeze(-2) + nx
    n = torch.tanh(n_pre)

    d_z = mu_h * (n - h)
    d_npre = mu_h * z * (1.0 - n.square())
    d_zpre = d_z * z * (1.0 - z)
    # ∂L/∂u = d_npre @ A_n^T
    d_u = torch.matmul(d_npre.unsqueeze(-2), a_n.transpose(-1, -2)).squeeze(-2)
    d_r = d_u * h
    d_rpre = d_r * r * (1.0 - r)

    g_wx = torch.cat(
        (
            d_zpre.reshape(*prefix, n_heads * d_head),
            d_rpre.reshape(*prefix, n_heads * d_head),
            d_npre.reshape(*prefix, n_heads * d_head),
        ),
        dim=-1,
    )
    g_az = torch.einsum("...hi,...ho->hio", h, d_zpre)
    g_ar = torch.einsum("...hi,...ho->hio", h, d_rpre)
    g_an = torch.einsum("...hi,...ho->hio", u, d_npre)
    return g_wx.to(dt), g_az.to(dt), g_ar.to(dt), g_an.to(dt)


_HEAD_VJP_BLOCK = 64
_HEAD_VJP_TILE = 32


@triton.jit
def _gemv_xA_tile(
    a_ptr,
    x_ptr,
    o0,
    d,
    stride_a_in,
    stride_a_out,
    stride_x,
    BLOCK: tl.constexpr,
):
    """One out-tile of ``y = x @ A`` (``A`` is ``(in, out)``)."""
    o = o0 + tl.arange(0, BLOCK)
    mask_o = o < d
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, d, BLOCK):
        k = k0 + tl.arange(0, BLOCK)
        mask_k = k < d
        a = load_acc(
            a_ptr + k[:, None] * stride_a_in + o[None, :] * stride_a_out,
            mask_k[:, None] & mask_o[None, :],
            0.0,
        )
        x = load_acc(x_ptr + k * stride_x, mask_k, 0.0)
        acc += tl.sum(a * x[:, None], axis=0)
    return acc, o, mask_o


@triton.jit
def _gemv_Av_tile(
    a_ptr,
    v_ptr,
    i0,
    d,
    stride_a_in,
    stride_a_out,
    stride_v,
    BLOCK: tl.constexpr,
):
    """One in-tile of ``y = A @ v`` (``A`` is ``(in, out)``)."""
    i = i0 + tl.arange(0, BLOCK)
    mask_i = i < d
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for o0 in range(0, d, BLOCK):
        o = o0 + tl.arange(0, BLOCK)
        mask_o = o < d
        a = load_acc(
            a_ptr + i[:, None] * stride_a_in + o[None, :] * stride_a_out,
            mask_i[:, None] & mask_o[None, :],
            0.0,
        )
        v = load_acc(v_ptr + o * stride_v, mask_o, 0.0)
        acc += tl.sum(a * v[None, :], axis=1)
    return acc, i, mask_i


@triton.jit
def _gru_head_vjp_sram_kernel(
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
    n_heads,
    time,
    d,
    d_h,
    stride_h_b,
    stride_h_t,
    stride_h_d,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_mu_b,
    stride_mu_t,
    stride_mu_d,
    stride_gwx_b,
    stride_gwx_t,
    stride_gwx_d,
    stride_ga_b,
    stride_ga_h,
    stride_ga_in,
    stride_ga_out,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    BLOCK_D: tl.constexpr,
):
    """One ``(batch, head)``: loop ``T``, accumulate ``∇A`` in SRAM (``d≤64``)."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d

    a_base = head * stride_a_h
    az = load_acc(
        az_ptr + a_base + offs[:, None] * stride_a_in + offs[None, :] * stride_a_out,
        mask_ij,
        0.0,
    )
    ar = load_acc(
        ar_ptr + a_base + offs[:, None] * stride_a_in + offs[None, :] * stride_a_out,
        mask_ij,
        0.0,
    )
    an = load_acc(
        an_ptr + a_base + offs[:, None] * stride_a_in + offs[None, :] * stride_a_out,
        mask_ij,
        0.0,
    )

    gaz = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    gar = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    gan = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    for t in range(0, time):
        h = load_acc(
            h_ptr + b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d,
            mask,
            0.0,
        )
        mu = load_acc(
            mu_ptr + b * stride_mu_b + t * stride_mu_t + (head_off + offs) * stride_mu_d,
            mask,
            0.0,
        )
        base_wx = b * stride_wx_b + t * stride_wx_t
        zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
        rx = load_acc(wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0)
        nx = load_acc(wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0)

        hz = tl.sum(az * h[:, None], axis=0)
        hr = tl.sum(ar * h[:, None], axis=0)
        z = tl.sigmoid(hz + zx)
        r = tl.sigmoid(hr + rx)
        u = h * r
        hn = tl.sum(an * u[:, None], axis=0)
        n = _nv_tanh(hn + nx)

        d_z = mu * (n - h)
        d_npre = mu * z * (1.0 - n * n)
        d_zpre = d_z * z * (1.0 - z)
        # d_u = d_npre @ A_n^T → (A_n @ d_npre) with A (in,out): sum_out A[in,out]*d_npre[out]
        d_u = tl.sum(an * d_npre[None, :], axis=1)
        d_rpre = (d_u * h) * r * (1.0 - r)

        base_g = b * stride_gwx_b + t * stride_gwx_t
        store_acc(gwx_ptr + base_g + (head_off + offs) * stride_gwx_d, d_zpre, mask)
        store_acc(gwx_ptr + base_g + (d_h + head_off + offs) * stride_gwx_d, d_rpre, mask)
        store_acc(gwx_ptr + base_g + (2 * d_h + head_off + offs) * stride_gwx_d, d_npre, mask)

        # ∇A[in,out] += h[in] * d_pre[out]
        gaz += h[:, None] * d_zpre[None, :]
        gar += h[:, None] * d_rpre[None, :]
        gan += u[:, None] * d_npre[None, :]

    ga_base = b * stride_ga_b + head * stride_ga_h
    store_acc(
        gaz_ptr + ga_base + offs[:, None] * stride_ga_in + offs[None, :] * stride_ga_out,
        gaz,
        mask_ij,
    )
    store_acc(
        gar_ptr + ga_base + offs[:, None] * stride_ga_in + offs[None, :] * stride_ga_out,
        gar,
        mask_ij,
    )
    store_acc(
        gan_ptr + ga_base + offs[:, None] * stride_ga_in + offs[None, :] * stride_ga_out,
        gan,
        mask_ij,
    )


@triton.jit
def _gru_head_vjp_tiled_pre_kernel(
    h_ptr,
    wx_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    mu_ptr,
    gwx_ptr,
    z_ptr,
    r_ptr,
    n_heads,
    time,
    d,
    d_h,
    stride_h_b,
    stride_h_t,
    stride_h_d,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_mu_b,
    stride_mu_t,
    stride_mu_d,
    stride_gwx_b,
    stride_gwx_t,
    stride_gwx_d,
    stride_z_b,
    stride_z_t,
    stride_z_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    BLOCK: tl.constexpr,
):
    """``(B*H, T)``: one timestep per program — gates + ``g_wx``; stash ``z,r`` for outer.

    One-``t``-per-program avoids a Triton loop-carried bug on ``d_rpre`` when a
    ``BLOCK_T`` inner loop wraps the ``A_n @ d_npre`` GEMV (seen for ``BLOCK_T=4``).
    """
    pid_bh = tl.program_id(0)
    t = tl.program_id(1)
    b = pid_bh // n_heads
    head = pid_bh % n_heads
    head_off = head * d
    a_base = head * stride_a_h

    h_base = b * stride_h_b + t * stride_h_t + head_off * stride_h_d
    wx_base = b * stride_wx_b + t * stride_wx_t
    mu_base = b * stride_mu_b + t * stride_mu_t + head_off * stride_mu_d
    gwx_base = b * stride_gwx_b + t * stride_gwx_t
    z_base = b * stride_z_b + t * stride_z_t + head_off * stride_z_d

    for o0 in range(0, d, BLOCK):
        hz, o, mask_o = _gemv_xA_tile(
            az_ptr + a_base,
            h_ptr + h_base,
            o0,
            d,
            stride_a_in,
            stride_a_out,
            stride_h_d,
            BLOCK,
        )
        hr, _, _ = _gemv_xA_tile(
            ar_ptr + a_base,
            h_ptr + h_base,
            o0,
            d,
            stride_a_in,
            stride_a_out,
            stride_h_d,
            BLOCK,
        )
        zx = load_acc(wx_ptr + wx_base + (head_off + o) * stride_wx_d, mask_o, 0.0)
        rx = load_acc(wx_ptr + wx_base + (d_h + head_off + o) * stride_wx_d, mask_o, 0.0)
        z = tl.sigmoid(hz + zx)
        r = tl.sigmoid(hr + rx)
        store_acc(z_ptr + z_base + o * stride_z_d, z, mask_o)
        store_acc(
            r_ptr + b * stride_z_b + t * stride_z_t + (head_off + o) * stride_z_d,
            r,
            mask_o,
        )

    for o0 in range(0, d, BLOCK):
        o = o0 + tl.arange(0, BLOCK)
        mask_o = o < d
        hn = tl.zeros((BLOCK,), dtype=tl.float32)
        for k0 in range(0, d, BLOCK):
            k = k0 + tl.arange(0, BLOCK)
            mask_k = k < d
            h_k = load_acc(h_ptr + h_base + k * stride_h_d, mask_k, 0.0)
            r_k = load_acc(
                r_ptr + b * stride_z_b + t * stride_z_t + (head_off + k) * stride_z_d,
                mask_k,
                0.0,
            )
            u_k = h_k * r_k
            a = load_acc(
                an_ptr + a_base + k[:, None] * stride_a_in + o[None, :] * stride_a_out,
                mask_k[:, None] & mask_o[None, :],
                0.0,
            )
            hn += tl.sum(a * u_k[:, None], axis=0)
        nx = load_acc(
            wx_ptr + wx_base + (2 * d_h + head_off + o) * stride_wx_d, mask_o, 0.0
        )
        n = _nv_tanh(hn + nx)
        store_acc(gwx_ptr + gwx_base + (2 * d_h + head_off + o) * stride_gwx_d, n, mask_o)

    for o0 in range(0, d, BLOCK):
        o = o0 + tl.arange(0, BLOCK)
        mask_o = o < d
        h = load_acc(h_ptr + h_base + o * stride_h_d, mask_o, 0.0)
        mu = load_acc(mu_ptr + mu_base + o * stride_mu_d, mask_o, 0.0)
        z = load_acc(z_ptr + z_base + o * stride_z_d, mask_o, 0.0)
        n = load_acc(
            gwx_ptr + gwx_base + (2 * d_h + head_off + o) * stride_gwx_d, mask_o, 0.0
        )
        d_z = mu * (n - h)
        d_npre = mu * z * (1.0 - n * n)
        d_zpre = d_z * z * (1.0 - z)
        store_acc(gwx_ptr + gwx_base + (head_off + o) * stride_gwx_d, d_zpre, mask_o)
        store_acc(
            gwx_ptr + gwx_base + (2 * d_h + head_off + o) * stride_gwx_d,
            d_npre,
            mask_o,
        )

    for i0 in range(0, d, BLOCK):
        d_u, i, mask_i = _gemv_Av_tile(
            an_ptr + a_base,
            gwx_ptr + gwx_base + (2 * d_h + head_off) * stride_gwx_d,
            i0,
            d,
            stride_a_in,
            stride_a_out,
            stride_gwx_d,
            BLOCK,
        )
        h = load_acc(h_ptr + h_base + i * stride_h_d, mask_i, 0.0)
        r = load_acc(
            r_ptr + b * stride_z_b + t * stride_z_t + (head_off + i) * stride_z_d,
            mask_i,
            0.0,
        )
        d_rpre = (d_u * h) * r * (1.0 - r)
        store_acc(
            gwx_ptr + gwx_base + (d_h + head_off + i) * stride_gwx_d,
            d_rpre,
            mask_i,
        )


@triton.jit
def _gru_head_vjp_tiled_outer_kernel(
    h_ptr,
    r_ptr,
    gwx_ptr,
    gaz_ptr,
    gar_ptr,
    gan_ptr,
    n_heads,
    time,
    d,
    d_h,
    stride_h_b,
    stride_h_t,
    stride_h_d,
    stride_r_b,
    stride_r_t,
    stride_r_d,
    stride_gwx_b,
    stride_gwx_t,
    stride_gwx_d,
    stride_ga_b,
    stride_ga_h,
    stride_ga_in,
    stride_ga_out,
    BLOCK: tl.constexpr,
):
    """``(B*H, i_tile, o_tile)``: sum_T outer products into ``∇A`` (one write, no RMW)."""
    pid_bh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_o = tl.program_id(2)
    b = pid_bh // n_heads
    head = pid_bh % n_heads
    head_off = head * d
    i0 = pid_i * BLOCK
    o0 = pid_o * BLOCK
    i = i0 + tl.arange(0, BLOCK)
    o = o0 + tl.arange(0, BLOCK)
    mask_i = i < d
    mask_o = o < d
    mask_ij = mask_i[:, None] & mask_o[None, :]

    gaz = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    gar = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    gan = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    for t in range(0, time):
        h_i = load_acc(
            h_ptr + b * stride_h_b + t * stride_h_t + (head_off + i) * stride_h_d,
            mask_i,
            0.0,
        )
        r_i = load_acc(
            r_ptr + b * stride_r_b + t * stride_r_t + (head_off + i) * stride_r_d,
            mask_i,
            0.0,
        )
        u_i = h_i * r_i
        d_z = load_acc(
            gwx_ptr + b * stride_gwx_b + t * stride_gwx_t + (head_off + o) * stride_gwx_d,
            mask_o,
            0.0,
        )
        d_r = load_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (d_h + head_off + o) * stride_gwx_d,
            mask_o,
            0.0,
        )
        d_n = load_acc(
            gwx_ptr
            + b * stride_gwx_b
            + t * stride_gwx_t
            + (2 * d_h + head_off + o) * stride_gwx_d,
            mask_o,
            0.0,
        )
        gaz += h_i[:, None] * d_z[None, :]
        gar += h_i[:, None] * d_r[None, :]
        gan += u_i[:, None] * d_n[None, :]

    base = b * stride_ga_b + head * stride_ga_h
    store_acc(
        gaz_ptr + base + i[:, None] * stride_ga_in + o[None, :] * stride_ga_out,
        gaz,
        mask_ij,
    )
    store_acc(
        gar_ptr + base + i[:, None] * stride_ga_in + o[None, :] * stride_ga_out,
        gar,
        mask_ij,
    )
    store_acc(
        gan_ptr + base + i[:, None] * stride_ga_in + o[None, :] * stride_ga_out,
        gan,
        mask_ij,
    )


def gru_head_recurrence_vjp(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Head GRU VJP with CUDA SRAM / tiled kernels, eager on CPU.

    ``d_head ≤ 64`` keeps ``∇A`` accumulators in SRAM and reduces over ``T``
    inside one program per ``(batch, head)``. Larger heads use a tiled
    pre-pass (gates + ``g_wx``) plus an outer-product write so ``∇A`` sums
    are deterministic (one write per tile, no RMW races).

    Parameters
    ----------
    h_prev : Tensor
        Previous hidden. Tensor of shape ``(B, T, d_h)``.
    wx : Tensor
        Precomputed ``W_x(x)``. Tensor of shape ``(B, T, 3 d_h)``.
    a_z, a_r, a_n : Tensor
        Per-head recurrent matrices. Each of shape
        ``(n_heads, d_head, d_head)``.
    mu : Tensor
        Upstream adjoint w.r.t. ``h_new``. Tensor of shape ``(B, T, d_h)``.
    n_heads : int
        Head count; ``d_h == n_heads * d_head``.
    d_head : int
        Width of each head block.

    Returns
    -------
    g_wx : Tensor
        Gradient w.r.t. ``wx``. Tensor of shape ``(B, T, 3 d_h)``.
    g_az, g_ar, g_an : Tensor
        Gradients w.r.t. ``A_*``. Each of shape ``(n_heads, d_head, d_head)``.

    Raises
    ------
    ValueError
        When ``d_h`` or ``a_*`` shapes disagree with ``n_heads`` / ``d_head``.
    """
    if not h_prev.is_cuda:
        return gru_head_recurrence_vjp_eager(
            h_prev, wx, a_z, a_r, a_n, mu, n_heads=n_heads, d_head=d_head
        )
    validate_cuda_tensors(h_prev, wx, a_z, a_r, a_n, mu, name="gru_head_recurrence_vjp")
    batch, time, d_h = h_prev.shape
    if d_h != n_heads * d_head:
        raise ValueError(f"d_h={d_h} != n_heads*d_head={n_heads * d_head}")
    if a_z.shape != (n_heads, d_head, d_head):
        raise ValueError(f"a_z shape {tuple(a_z.shape)} != {(n_heads, d_head, d_head)}")
    h_prev = h_prev.contiguous()
    wx = wx.contiguous()
    mu = mu.contiguous()
    a_z = a_z.contiguous()
    a_r = a_r.contiguous()
    a_n = a_n.contiguous()
    g_wx = wx.new_empty(wx.shape)
    dt = a_z.dtype
    n_rows = batch * n_heads

    if d_head <= _HEAD_VJP_BLOCK:
        g_az_b = torch.zeros(
            batch, n_heads, d_head, d_head, device=h_prev.device, dtype=torch.float32
        )
        g_ar_b = torch.zeros_like(g_az_b)
        g_an_b = torch.zeros_like(g_az_b)
        block = 1 << (d_head - 1).bit_length()
        block = min(max(block, 16), _HEAD_VJP_BLOCK)
        _gru_head_vjp_sram_kernel[(n_rows,)](
            h_prev,
            wx,
            a_z,
            a_r,
            a_n,
            mu,
            g_wx,
            g_az_b,
            g_ar_b,
            g_an_b,
            n_heads,
            time,
            d_head,
            d_h,
            *h_prev.stride(),
            *wx.stride(),
            *mu.stride(),
            *g_wx.stride(),
            *g_az_b.stride(),
            *a_z.stride(),
            BLOCK_D=block,
        )
        return (
            g_wx,
            g_az_b.sum(0).to(dt),
            g_ar_b.sum(0).to(dt),
            g_an_b.sum(0).to(dt),
        )

    g_az_b = torch.zeros(
        batch, n_heads, d_head, d_head, device=h_prev.device, dtype=torch.float32
    )
    g_ar_b = torch.zeros_like(g_az_b)
    g_an_b = torch.zeros_like(g_az_b)
    z_buf = torch.empty(batch, time, d_h, device=h_prev.device, dtype=torch.float32)
    r_buf = torch.empty_like(z_buf)
    n_d = triton.cdiv(d_head, _HEAD_VJP_TILE)
    _gru_head_vjp_tiled_pre_kernel[(n_rows, time)](
        h_prev,
        wx,
        a_z,
        a_r,
        a_n,
        mu,
        g_wx,
        z_buf,
        r_buf,
        n_heads,
        time,
        d_head,
        d_h,
        *h_prev.stride(),
        *wx.stride(),
        *mu.stride(),
        *g_wx.stride(),
        *z_buf.stride(),
        *a_z.stride(),
        BLOCK=_HEAD_VJP_TILE,
    )
    _gru_head_vjp_tiled_outer_kernel[(n_rows, n_d, n_d)](
        h_prev,
        r_buf,
        g_wx,
        g_az_b,
        g_ar_b,
        g_an_b,
        n_heads,
        time,
        d_head,
        d_h,
        *h_prev.stride(),
        *r_buf.stride(),
        *g_wx.stride(),
        *g_az_b.stride(),
        BLOCK=_HEAD_VJP_TILE,
    )
    return (
        g_wx,
        g_az_b.sum(0).to(dt),
        g_ar_b.sum(0).to(dt),
        g_an_b.sum(0).to(dt),
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
