"""Factorized / fused Newton for ``ParaGRU(mix='head')`` — no dense ``d×d``.

Forward: one Triton program per ``(batch, head)`` walks time — gates from
``wx`` + ``A_*``, residual, factorized ``J δ`` (Alg. 1). Backward: factorized
``J^T`` reverse (eq. 2.6). ``d_head ≤ 64``: full fused SRAM path;
``64 < d_head ≤ 128``: streamed-``A`` SRAM Newton + reverse; larger: PyTorch
gates + tiled Triton factor scan / reverse. CPU / ``scan_backend='eager'``
keep the dense-J oracle.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.cells.para_gru import ParaGRU
from pararnn.kernels._fused_common import _tanh
from pararnn.kernels.gru_head_factor import gru_head_jvp, gru_head_jt_mvp
from pararnn.kernels.precision import load_acc, store_acc
from pararnn.layout import prepend_state, prepend_state_ragged

log = logging.getLogger(__name__)

_BLOCK_D = 64
# Reverse / Newton can stream one ``A`` at a time up to this head width.
_REVERSE_SRAM_D = 128
_NEWTON_SRAM_D = 128
# Tile size for factorized matvecs when ``d_head > _NEWTON_SRAM_D``.
_TILE_D = 32


def newton_gru_head_factorized(
    cell: ParaGRU,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Alg. 1 for block-diagonal ParaGRU without materializing ``J``.

    ``wx = cell.W_x(x)`` is computed outside (same split as diag fused).
    Prefer ``pararnn::newton_gru_head_fused`` via ``fused_newton`` on CUDA.
    """
    if cell.mix != "head" or cell.n_heads is None or cell.d_head is None:
        raise TypeError("newton_gru_head_factorized needs ParaGRU(mix='head')")
    a_z, a_r, a_n = cell.clipped_a_head()
    return _newton_gru_head_fused_impl(
        wx,
        a_z,
        a_r,
        a_n,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
    )


def _newton_gru_head_fused_impl(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Tensor Alg. 1 for head GRU. Public entry: ``pararnn::newton_gru_head_fused``."""
    if a_z.dim() != 3 or a_z.shape[-1] != a_z.shape[-2]:
        raise ValueError(f"a_z needs (H,d,d), got {tuple(a_z.shape)}")
    n_heads, d_head, _ = a_z.shape
    if a_r.shape != a_z.shape or a_n.shape != a_z.shape:
        raise ValueError("a_z, a_r, a_n shape mismatch")
    d_h = n_heads * d_head
    if wx.shape[-1] != 3 * d_h:
        raise ValueError(f"wx last dim {wx.shape[-1]} != 3 * d_h={3 * d_h}")
    if cu_seqlens is not None and wx.shape[0] != 1:
        raise ValueError("cu_seqlens packs need batch=1")
    batch, time, _ = wx.shape
    # Rectangular CUDA paths only. Packed packs stay on factorized eager;
    # fused_newton raises before the custom op when cu_seqlens is set.
    fused = wx.is_cuda and d_head <= _BLOCK_D and cu_seqlens is None
    stream = (
        wx.is_cuda and _BLOCK_D < d_head <= _NEWTON_SRAM_D and cu_seqlens is None
    )
    hybrid = wx.is_cuda and d_head > _NEWTON_SRAM_D and cu_seqlens is None

    if fused:
        states = _newton_fused_triton(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            max_iters=max_iters,
            omega=omega,
            n_heads=n_heads,
            d_head=d_head,
        )
    elif stream:
        states = _newton_stream_a_triton(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            max_iters=max_iters,
            omega=omega,
            n_heads=n_heads,
            d_head=d_head,
        )
    elif hybrid:
        states = _newton_hybrid_tiled(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            max_iters=max_iters,
            omega=omega,
            n_heads=n_heads,
            d_head=d_head,
            d_h=d_h,
        )
    else:
        if cu_seqlens is not None and wx.is_cuda:
            log.warning(
                "newton_gru_head: cu_seqlens uses factorized eager "
                "(fused/stream/hybrid are rectangular-batch only)"
            )
        states = _newton_factor_eager(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            max_iters=max_iters,
            omega=omega,
            cu_seqlens=cu_seqlens,
            n_heads=n_heads,
            d_head=d_head,
            d_h=d_h,
        )

    if log.isEnabledFor(logging.DEBUG):
        path = (
            "fused"
            if fused
            else ("stream" if stream else ("hybrid" if hybrid else "eager"))
        )
        log.debug(
            "newton_gru_head_fused",
            extra={
                "batch": batch,
                "seq_len": time,
                "n_heads": n_heads,
                "d_head": d_head,
                "max_iters": max_iters,
                "path": path,
            },
        )
    return states


def reverse_factor_scan_gru_head(
    cell: ParaGRU,
    h_prev: Tensor,
    partial: Tensor,
    *,
    wx: Tensor,
) -> Tensor:
    """Eq. 2.6 reverse with factorized ``J^T`` (no dense ``d×d``)."""
    a_z, a_r, a_n = cell.clipped_a_head()
    if partial.is_cuda:
        from pararnn.kernels.custom_ops import reverse_gru_head_factor

        return reverse_gru_head_factor(h_prev, wx, partial, a_z, a_r, a_n)
    return _reverse_gru_head_factor_impl(h_prev, wx, partial, a_z, a_r, a_n)


def _reverse_gru_head_factor_impl(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
) -> Tensor:
    """Tensor eq. 2.6 reverse. Public entry: ``pararnn::reverse_gru_head_factor``."""
    n_heads, d_head, _ = a_z.shape
    d_h = n_heads * d_head
    if partial.is_cuda and d_head <= _BLOCK_D:
        return _reverse_fused_triton(
            h_prev, wx, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
        )
    if partial.is_cuda and d_head <= _REVERSE_SRAM_D:
        return _reverse_stream_a_triton(
            h_prev, wx, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
        )
    _, gates = _gates(h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h)
    if partial.is_cuda:
        return _factor_reverse_tiled_triton(
            gates, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
        )
    return _factor_reverse_eager(
        gates, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
    )


def gru_head_t0_vjp(
    cell: ParaGRU,
    h_prev: Tensor,
    mu: Tensor,
    *,
    wx: Tensor,
) -> Tensor:
    """``J_0^T μ_0`` for the paper ``h_0`` adjoint (factorized)."""
    a_z, a_r, a_n = cell.clipped_a_head()
    n_heads, d_head = cell.n_heads, cell.d_head
    assert n_heads is not None and d_head is not None
    d_h = n_heads * d_head
    _, gates = _gates(h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h)
    h, z, r, n = gates
    mu0 = mu[:, 0].reshape(mu.shape[0], n_heads, d_head)
    g = gru_head_jt_mvp(h[:, 0], z[:, 0], r[:, 0], n[:, 0], a_z, a_r, a_n, mu0)
    return g.reshape(mu.shape[0], n_heads * d_head)


def _newton_factor_eager(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    cu_seqlens: Tensor | None,
    n_heads: int,
    d_head: int,
    d_h: int,
) -> Tensor:
    h_prev0 = _init_prev(wx, h0, d_h=d_h, cu_seqlens=cu_seqlens)
    states, _ = _gates(
        h_prev0, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
    )
    for _ in range(max_iters):
        h_prev = (
            prepend_state_ragged(states, h0, cu_seqlens)
            if cu_seqlens is not None
            else prepend_state(states, h0)
        )
        pred, gates = _gates(
            h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
        )
        residual = pred - states
        delta = _factor_scan_eager(
            gates, residual, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
        )
        states = states + omega * delta
    return states


def _init_prev(
    wx: Tensor, h0: Tensor | None, *, d_h: int, cu_seqlens: Tensor | None
) -> Tensor:
    zeros = wx.new_zeros(wx.shape[0], wx.shape[1], d_h)
    if cu_seqlens is not None:
        return prepend_state_ragged(zeros, h0, cu_seqlens)
    return prepend_state(zeros, h0)


def _gates(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
    d_h: int,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Batched head gates; returns ``(h_new, (h,z,r,n)_heads)``."""
    prefix = h_prev.shape[:-1]
    h_v = h_prev.reshape(*prefix, n_heads, d_head)
    zx, rx, nx = wx.chunk(3, dim=-1)
    zx_v = zx.reshape(*prefix, n_heads, d_head)
    rx_v = rx.reshape(*prefix, n_heads, d_head)
    nx_v = nx.reshape(*prefix, n_heads, d_head)
    z = torch.sigmoid(torch.matmul(h_v.unsqueeze(-2), a_z).squeeze(-2) + zx_v)
    r = torch.sigmoid(torch.matmul(h_v.unsqueeze(-2), a_r).squeeze(-2) + rx_v)
    n = torch.tanh(torch.matmul((h_v * r).unsqueeze(-2), a_n).squeeze(-2) + nx_v)
    h_new_v = torch.lerp(h_v, n, z)
    h_new = h_new_v.reshape(*prefix, d_h)
    return h_new, (h_v, z, r, n)


def _factor_scan_eager(
    gates: tuple[Tensor, Tensor, Tensor, Tensor],
    residual: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    h, z, r, n = gates
    batch, time = residual.shape[:2]
    res_h = residual.reshape(batch, time, n_heads, d_head)
    delta = residual.new_zeros(batch, n_heads, d_head)
    out = residual.new_empty(batch, time, n_heads, d_head)
    for t in range(time):
        delta = gru_head_jvp(h[:, t], z[:, t], r[:, t], n[:, t], a_z, a_r, a_n, delta)
        delta = delta + res_h[:, t]
        out[:, t] = delta
    return out.reshape(batch, time, n_heads * d_head)


def _factor_reverse_eager(
    gates: tuple[Tensor, Tensor, Tensor, Tensor],
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    """``μ_t = J_{t+1}^T μ_{t+1} + g_t`` (eq. 2.6)."""
    h, z, r, n = gates
    batch, time = partial.shape[:2]
    part_h = partial.reshape(batch, time, n_heads, d_head)
    mu = partial.new_zeros(batch, n_heads, d_head)
    out = partial.new_empty(batch, time, n_heads, d_head)
    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            mu = gru_head_jt_mvp(
                h[:, t + 1], z[:, t + 1], r[:, t + 1], n[:, t + 1], a_z, a_r, a_n, mu
            )
        mu = mu + part_h[:, t]
        out[:, t] = mu
    return out.reshape(batch, time, n_heads * d_head)


def _block_d(d_head: int) -> int:
    block = 1 << (d_head - 1).bit_length()
    return min(max(block, 16), _BLOCK_D)


@triton.jit
def _head_gates_from_prev(h_prev, zx, rx, nx, az, ar, an):
    """Cho GRU head step + gate activations for factorized J."""
    hz = tl.sum(az * h_prev[:, None], axis=0)
    hr = tl.sum(ar * h_prev[:, None], axis=0)
    z = tl.sigmoid(hz + zx)
    r = tl.sigmoid(hr + rx)
    u = h_prev * r
    hn = tl.sum(an * u[:, None], axis=0)
    n = _tanh(hn + nx)
    h_new = (1.0 - z) * h_prev + z * n
    return h_new, h_prev, z, r, n


@triton.jit
def _head_jvp(h, z, r, n, az, ar, an, delta):
    """Factorized ``J @ delta`` (same algebra as ``gru_head_jvp``)."""
    z_p = z * (1.0 - z)
    r_p = r * (1.0 - r)
    n_p = 1.0 - n * n
    daz = tl.sum(az * delta[:, None], axis=0)
    dar = tl.sum(ar * delta[:, None], axis=0)
    dz_v = z_p * daz
    du = r * delta + h * (r_p * dar)
    dan = tl.sum(an * du[:, None], axis=0)
    dn_v = n_p * dan
    return (1.0 - z) * delta + (n - h) * dz_v + z * dn_v


@triton.jit
def _head_jt_mvp(h, z, r, n, az, ar, an, mu):
    """Factorized ``J^T @ mu`` (same algebra as ``gru_head_jt_mvp``)."""
    z_p = z * (1.0 - z)
    r_p = r * (1.0 - r)
    n_p = 1.0 - n * n
    t1 = (1.0 - z) * mu
    w_z = z_p * ((n - h) * mu)
    t2 = tl.sum(az * w_z[None, :], axis=1)
    w_n = n_p * (z * mu)
    g_du = tl.sum(an * w_n[None, :], axis=1)
    w_r = r_p * (h * g_du)
    t3 = r * g_du + tl.sum(ar * w_r[None, :], axis=1)
    return t1 + t2 + t3


@triton.jit
def _newton_head_all_kernel(
    wx_ptr,
    h_ptr,
    delta_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    h0_ptr,
    has_h0,
    n_heads,
    time,
    d,
    d_h,
    omega,
    max_iters,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_h_b,
    stride_h_t,
    stride_h_d,
    stride_d_b,
    stride_d_t,
    stride_d_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    stride_h0_b,
    stride_h0_d,
    BLOCK_D: tl.constexpr,
):
    """Init + K Newton iters on native ``(B,T,d_h)`` / ``(H,d,d)`` layout."""
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

    h0_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if has_h0 != 0:
        h0_v = load_acc(
            h0_ptr + b * stride_h0_b + (head_off + offs) * stride_h0_d, mask, 0.0
        )

    # App. A parallel guess: only t=0 sees h0; other steps see 0.
    for t in range(0, time):
        base_wx = b * stride_wx_b + t * stride_wx_t
        zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
        rx = load_acc(wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0)
        nx = load_acc(wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0)
        h_prev = tl.where(t == 0, h0_v, tl.zeros((BLOCK_D,), dtype=tl.float32))
        h_new, _, _, _, _ = _head_gates_from_prev(h_prev, zx, rx, nx, az, ar, an)
        store_acc(
            h_ptr + b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d,
            h_new,
            mask,
        )

    for _it in range(0, max_iters):
        delta = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, time):
            base_wx = b * stride_wx_b + t * stride_wx_t
            zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
            rx = load_acc(
                wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            nx = load_acc(
                wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            h_prev_nm1 = load_acc(
                h_ptr
                + b * stride_h_b
                + (t - 1) * stride_h_t
                + (head_off + offs) * stride_h_d,
                mask & (t > 0),
                0.0,
            )
            h_prev = tl.where(t == 0, h0_v, h_prev_nm1)
            h_guess = load_acc(
                h_ptr + b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d,
                mask,
                0.0,
            )
            pred, h_g, z, r, n = _head_gates_from_prev(h_prev, zx, rx, nx, az, ar, an)
            res = pred - h_guess
            delta = _head_jvp(h_g, z, r, n, az, ar, an, delta) + res
            store_acc(
                delta_ptr
                + b * stride_d_b
                + t * stride_d_t
                + (head_off + offs) * stride_d_d,
                delta,
                mask,
            )
        for t in range(0, time):
            base_h = b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d
            base_d = b * stride_d_b + t * stride_d_t + (head_off + offs) * stride_d_d
            h_guess = load_acc(h_ptr + base_h, mask, 0.0)
            dlt = load_acc(delta_ptr + base_d, mask, 0.0)
            store_acc(h_ptr + base_h, h_guess + omega * dlt, mask)


@triton.jit
def _reverse_head_native_kernel(
    wx_ptr,
    h_prev_ptr,
    part_ptr,
    out_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    n_heads,
    time,
    d,
    d_h,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_hp_b,
    stride_hp_t,
    stride_hp_d,
    stride_p_b,
    stride_p_t,
    stride_p_d,
    stride_o_b,
    stride_o_t,
    stride_o_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    BLOCK_D: tl.constexpr,
):
    """Eq. 2.6 reverse on native ``(B,T,d_h)`` layout."""
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

    mu = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t_rev in range(0, time):
        t = time - 1 - t_rev
        part = load_acc(
            part_ptr + b * stride_p_b + t * stride_p_t + (head_off + offs) * stride_p_d,
            mask,
            0.0,
        )
        if t_rev > 0:
            tp = t + 1
            base_wx = b * stride_wx_b + tp * stride_wx_t
            zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
            rx = load_acc(
                wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            nx = load_acc(
                wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            hp = load_acc(
                h_prev_ptr
                + b * stride_hp_b
                + tp * stride_hp_t
                + (head_off + offs) * stride_hp_d,
                mask,
                0.0,
            )
            _, h_g, z, r, n = _head_gates_from_prev(hp, zx, rx, nx, az, ar, an)
            mu = _head_jt_mvp(h_g, z, r, n, az, ar, an, mu)
        mu = mu + part
        store_acc(
            out_ptr + b * stride_o_b + t * stride_o_t + (head_off + offs) * stride_o_d,
            mu,
            mask,
        )


def _as_fp32_work(t: Tensor) -> tuple[Tensor, bool]:
    narrow = t.dtype in (torch.float16, torch.bfloat16)
    return (t.float().contiguous() if narrow else t.contiguous()), narrow


def _newton_fused_triton(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time, three_dh = wx.shape
    d_h = three_dh // 3
    dt = wx.dtype
    wx_w, narrow = _as_fp32_work(wx)
    az, _ = _as_fp32_work(a_z)
    ar, _ = _as_fp32_work(a_r)
    an, _ = _as_fp32_work(a_n)
    h = torch.empty(batch, time, d_h, device=wx.device, dtype=torch.float32)
    delta = torch.empty_like(h)
    has_h0 = 1 if h0 is not None else 0
    if h0 is not None:
        h0_w = h0.float().contiguous()
    else:
        h0_w = torch.empty(batch, d_h, device=wx.device, dtype=torch.float32)
    block = _block_d(d_head)
    _newton_head_all_kernel[(batch * n_heads,)](
        wx_w,
        h,
        delta,
        az,
        ar,
        an,
        h0_w,
        has_h0,
        n_heads,
        time,
        d_head,
        d_h,
        float(omega),
        max_iters,
        wx_w.stride(0),
        wx_w.stride(1),
        wx_w.stride(2),
        h.stride(0),
        h.stride(1),
        h.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        h0_w.stride(0),
        h0_w.stride(1) if h0_w.dim() > 1 else 1,
        BLOCK_D=block,
    )
    return h.to(dtype=dt) if narrow else h


@triton.jit
def _load_a_square(a_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out):
    return load_acc(
        a_ptr + a_base + offs[:, None] * stride_a_in + offs[None, :] * stride_a_out,
        mask_ij,
        0.0,
    )


@triton.jit
def _stream_gates(h_prev, zx, rx, nx, az_ptr, ar_ptr, an_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out):
    """Cho gates; peak one ``A`` in SRAM. Returns ``(h_new, h_prev, z, r, n, an)``."""
    az = _load_a_square(az_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out)
    hz = tl.sum(az * h_prev[:, None], axis=0)
    ar = _load_a_square(ar_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out)
    hr = tl.sum(ar * h_prev[:, None], axis=0)
    z = tl.sigmoid(hz + zx)
    r = tl.sigmoid(hr + rx)
    an = _load_a_square(an_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out)
    n = _tanh(tl.sum(an * (h_prev * r)[:, None], axis=0) + nx)
    h_new = (1.0 - z) * h_prev + z * n
    return h_new, h_prev, z, r, n, an


@triton.jit
def _stream_jvp(h, z, r, n, an, delta, az_ptr, ar_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out):
    """Factorized ``J @ delta``; ``an`` still live from gates; reload ``az``/``ar``."""
    z_p = z * (1.0 - z)
    r_p = r * (1.0 - r)
    n_p = 1.0 - n * n
    az = _load_a_square(az_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out)
    daz = tl.sum(az * delta[:, None], axis=0)
    ar = _load_a_square(ar_ptr, a_base, offs, mask_ij, stride_a_in, stride_a_out)
    dar = tl.sum(ar * delta[:, None], axis=0)
    du = r * delta + h * (r_p * dar)
    dan = tl.sum(an * du[:, None], axis=0)
    return (1.0 - z) * delta + (n - h) * (z_p * daz) + z * (n_p * dan)


@triton.jit
def _newton_head_stream_a_kernel(
    wx_ptr,
    h_ptr,
    delta_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    h0_ptr,
    has_h0,
    n_heads,
    time,
    d,
    d_h,
    omega,
    max_iters,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_h_b,
    stride_h_t,
    stride_h_d,
    stride_d_b,
    stride_d_t,
    stride_d_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    stride_h0_b,
    stride_h0_d,
    BLOCK_D: tl.constexpr,
):
    """Alg. 1 with streamed ``A_*`` (``64 < d_head ≤ 128``)."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d
    a_base = head * stride_a_h

    h0_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
    if has_h0 != 0:
        h0_v = load_acc(
            h0_ptr + b * stride_h0_b + (head_off + offs) * stride_h0_d, mask, 0.0
        )

    for t in range(0, time):
        base_wx = b * stride_wx_b + t * stride_wx_t
        zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
        rx = load_acc(wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0)
        nx = load_acc(
            wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0
        )
        h_prev = tl.where(t == 0, h0_v, tl.zeros((BLOCK_D,), dtype=tl.float32))
        h_new, _, _, _, _, _ = _stream_gates(
            h_prev,
            zx,
            rx,
            nx,
            az_ptr,
            ar_ptr,
            an_ptr,
            a_base,
            offs,
            mask_ij,
            stride_a_in,
            stride_a_out,
        )
        store_acc(
            h_ptr + b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d,
            h_new,
            mask,
        )

    for _it in range(0, max_iters):
        delta = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t in range(0, time):
            base_wx = b * stride_wx_b + t * stride_wx_t
            zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
            rx = load_acc(
                wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            nx = load_acc(
                wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            h_prev_nm1 = load_acc(
                h_ptr
                + b * stride_h_b
                + (t - 1) * stride_h_t
                + (head_off + offs) * stride_h_d,
                mask & (t > 0),
                0.0,
            )
            h_prev = tl.where(t == 0, h0_v, h_prev_nm1)
            h_guess = load_acc(
                h_ptr + b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d,
                mask,
                0.0,
            )
            pred, h_g, z, r, n, an = _stream_gates(
                h_prev,
                zx,
                rx,
                nx,
                az_ptr,
                ar_ptr,
                an_ptr,
                a_base,
                offs,
                mask_ij,
                stride_a_in,
                stride_a_out,
            )
            res = pred - h_guess
            delta = (
                _stream_jvp(
                    h_g,
                    z,
                    r,
                    n,
                    an,
                    delta,
                    az_ptr,
                    ar_ptr,
                    a_base,
                    offs,
                    mask_ij,
                    stride_a_in,
                    stride_a_out,
                )
                + res
            )
            store_acc(
                delta_ptr
                + b * stride_d_b
                + t * stride_d_t
                + (head_off + offs) * stride_d_d,
                delta,
                mask,
            )
        for t in range(0, time):
            base_h = b * stride_h_b + t * stride_h_t + (head_off + offs) * stride_h_d
            base_d = b * stride_d_b + t * stride_d_t + (head_off + offs) * stride_d_d
            h_guess = load_acc(h_ptr + base_h, mask, 0.0)
            dlt = load_acc(delta_ptr + base_d, mask, 0.0)
            store_acc(h_ptr + base_h, h_guess + omega * dlt, mask)


def _newton_stream_a_triton(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    n_heads: int,
    d_head: int,
) -> Tensor:
    """``64 < d_head ≤ 128``: fused Newton with streamed ``A_*``."""
    batch, time, three_dh = wx.shape
    d_h = three_dh // 3
    dt = wx.dtype
    wx_w, narrow = _as_fp32_work(wx)
    az, _ = _as_fp32_work(a_z)
    ar, _ = _as_fp32_work(a_r)
    an, _ = _as_fp32_work(a_n)
    h = torch.empty(batch, time, d_h, device=wx.device, dtype=torch.float32)
    delta = torch.empty_like(h)
    has_h0 = 1 if h0 is not None else 0
    if h0 is not None:
        h0_w = h0.float().contiguous()
    else:
        h0_w = torch.empty(batch, d_h, device=wx.device, dtype=torch.float32)
    block = _reverse_sram_block(d_head)
    _newton_head_stream_a_kernel[(batch * n_heads,)](
        wx_w,
        h,
        delta,
        az,
        ar,
        an,
        h0_w,
        has_h0,
        n_heads,
        time,
        d_head,
        d_h,
        float(omega),
        max_iters,
        wx_w.stride(0),
        wx_w.stride(1),
        wx_w.stride(2),
        h.stride(0),
        h.stride(1),
        h.stride(2),
        delta.stride(0),
        delta.stride(1),
        delta.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        h0_w.stride(0),
        h0_w.stride(1) if h0_w.dim() > 1 else 1,
        BLOCK_D=block,
        num_warps=8 if block >= 64 else 4,
    )
    return h.to(dtype=dt) if narrow else h


def _reverse_fused_triton(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    batch, time, d_h = partial.shape
    dt = partial.dtype
    wx_w, _ = _as_fp32_work(wx)
    hp, _ = _as_fp32_work(h_prev)
    part, narrow = _as_fp32_work(partial)
    out = torch.empty_like(part)
    az, _ = _as_fp32_work(a_z)
    ar, _ = _as_fp32_work(a_r)
    an, _ = _as_fp32_work(a_n)
    block = _block_d(d_head)
    _reverse_head_native_kernel[(batch * n_heads,)](
        wx_w,
        hp,
        part,
        out,
        az,
        ar,
        an,
        n_heads,
        time,
        d_head,
        d_h,
        wx_w.stride(0),
        wx_w.stride(1),
        wx_w.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        part.stride(0),
        part.stride(1),
        part.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        BLOCK_D=block,
    )
    return out.to(dtype=dt) if narrow else out


def _reverse_sram_block(d_head: int) -> int:
    block = 1 << (d_head - 1).bit_length()
    return min(max(block, 16), _REVERSE_SRAM_D)


@triton.jit
def _reverse_head_stream_a_kernel(
    wx_ptr,
    h_prev_ptr,
    part_ptr,
    out_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    n_heads,
    time,
    d,
    d_h,
    stride_wx_b,
    stride_wx_t,
    stride_wx_d,
    stride_hp_b,
    stride_hp_t,
    stride_hp_d,
    stride_p_b,
    stride_p_t,
    stride_p_d,
    stride_o_b,
    stride_o_t,
    stride_o_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    BLOCK_D: tl.constexpr,
):
    """Eq. 2.6 reverse; one ``A`` in SRAM at a time (``d_head ≤ 128``)."""
    pid = tl.program_id(0)
    b = pid // n_heads
    head = pid % n_heads
    offs = tl.arange(0, BLOCK_D)
    mask = offs < d
    mask_ij = mask[:, None] & mask[None, :]
    head_off = head * d
    a_base = head * stride_a_h

    mu = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t_rev in range(0, time):
        t = time - 1 - t_rev
        part = load_acc(
            part_ptr + b * stride_p_b + t * stride_p_t + (head_off + offs) * stride_p_d,
            mask,
            0.0,
        )
        if t_rev > 0:
            tp = t + 1
            base_wx = b * stride_wx_b + tp * stride_wx_t
            zx = load_acc(wx_ptr + base_wx + (head_off + offs) * stride_wx_d, mask, 0.0)
            rx = load_acc(
                wx_ptr + base_wx + (d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            nx = load_acc(
                wx_ptr + base_wx + (2 * d_h + head_off + offs) * stride_wx_d, mask, 0.0
            )
            hp = load_acc(
                h_prev_ptr
                + b * stride_hp_b
                + tp * stride_hp_t
                + (head_off + offs) * stride_hp_d,
                mask,
                0.0,
            )
            # Gates: stream A_* so peak SRAM is one (BLOCK_D, BLOCK_D).
            az = load_acc(
                az_ptr
                + a_base
                + offs[:, None] * stride_a_in
                + offs[None, :] * stride_a_out,
                mask_ij,
                0.0,
            )
            hz = tl.sum(az * hp[:, None], axis=0)
            ar = load_acc(
                ar_ptr
                + a_base
                + offs[:, None] * stride_a_in
                + offs[None, :] * stride_a_out,
                mask_ij,
                0.0,
            )
            hr = tl.sum(ar * hp[:, None], axis=0)
            z = tl.sigmoid(hz + zx)
            r = tl.sigmoid(hr + rx)
            an = load_acc(
                an_ptr
                + a_base
                + offs[:, None] * stride_a_in
                + offs[None, :] * stride_a_out,
                mask_ij,
                0.0,
            )
            n = _tanh(tl.sum(an * (hp * r)[:, None], axis=0) + nx)
            # J^T: reload Az / keep An / reload Ar (liveness ≤ 2 mats).
            z_p = z * (1.0 - z)
            r_p = r * (1.0 - r)
            n_p = 1.0 - n * n
            w_z = z_p * ((n - hp) * mu)
            az = load_acc(
                az_ptr
                + a_base
                + offs[:, None] * stride_a_in
                + offs[None, :] * stride_a_out,
                mask_ij,
                0.0,
            )
            t2 = tl.sum(az * w_z[None, :], axis=1)
            w_n = n_p * (z * mu)
            g_du = tl.sum(an * w_n[None, :], axis=1)
            w_r = r_p * (hp * g_du)
            ar = load_acc(
                ar_ptr
                + a_base
                + offs[:, None] * stride_a_in
                + offs[None, :] * stride_a_out,
                mask_ij,
                0.0,
            )
            t3 = r * g_du + tl.sum(ar * w_r[None, :], axis=1)
            mu = (1.0 - z) * mu + t2 + t3
        mu = mu + part
        store_acc(
            out_ptr + b * stride_o_b + t * stride_o_t + (head_off + offs) * stride_o_d,
            mu,
            mask,
        )


def _reverse_stream_a_triton(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    """``64 < d_head ≤ 128``: fused reverse with streamed ``A_*`` loads."""
    batch, time, d_h = partial.shape
    dt = partial.dtype
    wx_w, _ = _as_fp32_work(wx)
    hp, _ = _as_fp32_work(h_prev)
    part, narrow = _as_fp32_work(partial)
    out = torch.empty_like(part)
    az, _ = _as_fp32_work(a_z)
    ar, _ = _as_fp32_work(a_r)
    an, _ = _as_fp32_work(a_n)
    block = _reverse_sram_block(d_head)
    _reverse_head_stream_a_kernel[(batch * n_heads,)](
        wx_w,
        hp,
        part,
        out,
        az,
        ar,
        an,
        n_heads,
        time,
        d_head,
        d_h,
        wx_w.stride(0),
        wx_w.stride(1),
        wx_w.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        part.stride(0),
        part.stride(1),
        part.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        BLOCK_D=block,
        num_warps=8 if block >= 64 else 4,
    )
    return out.to(dtype=dt) if narrow else out


def _newton_hybrid_tiled(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    n_heads: int,
    d_head: int,
    d_h: int,
) -> Tensor:
    """``d_head > 128``: PyTorch gates + tiled Triton factorized scan.

    cuBLAS ``h @ A`` beats a naive in-kernel tiled GEMV on this class of GPUs;
    the scan stays in Triton.
    """
    h_prev0 = _init_prev(wx, h0, d_h=d_h, cu_seqlens=None)
    states, _ = _gates(
        h_prev0, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
    )
    for _ in range(max_iters):
        h_prev = prepend_state(states, h0)
        pred, gates = _gates(
            h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
        )
        residual = pred - states
        delta = _factor_scan_tiled_triton(
            gates, residual, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
        )
        states = states + omega * delta
    return states


@triton.jit
def _mix_col_tile(a_ptr, v_ptr, i, j_offs, d, stride_a_in, stride_a_out, stride_v, BLOCK: tl.constexpr):
    """``sum_j A[j, i] * v[j]`` over one ``j`` tile."""
    j = j_offs + tl.arange(0, BLOCK)
    mask_j = j < d
    mask_ij = mask_j[:, None] & (i < d)[None, :]
    a_tile = load_acc(
        a_ptr + j[:, None] * stride_a_in + i[None, :] * stride_a_out,
        mask_ij,
        0.0,
    )
    v_tile = load_acc(v_ptr + j * stride_v, mask_j, 0.0)
    return tl.sum(a_tile * v_tile[:, None], axis=0)


@triton.jit
def _factor_scan_tiled_kernel(
    h_ptr,
    z_ptr,
    r_ptr,
    n_ptr,
    res_ptr,
    out_ptr,
    daz_ptr,
    dar_ptr,
    du_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    time,
    d,
    stride_row,
    stride_t,
    stride_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    n_heads,
    BLOCK: tl.constexpr,
):
    """Inclusive factorized scan; ``A`` tiled so ``d`` may exceed SRAM."""
    pid = tl.program_id(0)
    head = pid % n_heads
    a_base = head * stride_a_h
    offs = tl.arange(0, BLOCK)

    for t in range(0, time):
        base = pid * stride_row + t * stride_t
        prev_base = pid * stride_row + (t - 1) * stride_t
        for i0 in range(0, d, BLOCK):
            i = i0 + offs
            mask_i = i < d
            daz = tl.zeros((BLOCK,), dtype=tl.float32)
            dar = tl.zeros((BLOCK,), dtype=tl.float32)
            if t > 0:
                for j0 in range(0, d, BLOCK):
                    daz += _mix_col_tile(
                        az_ptr + a_base,
                        out_ptr + prev_base,
                        i,
                        j0,
                        d,
                        stride_a_in,
                        stride_a_out,
                        stride_d,
                        BLOCK,
                    )
                    dar += _mix_col_tile(
                        ar_ptr + a_base,
                        out_ptr + prev_base,
                        i,
                        j0,
                        d,
                        stride_a_in,
                        stride_a_out,
                        stride_d,
                        BLOCK,
                    )
            store_acc(daz_ptr + pid * d + i, daz, mask_i)
            store_acc(dar_ptr + pid * d + i, dar, mask_i)

        for i0 in range(0, d, BLOCK):
            i = i0 + offs
            mask_i = i < d
            h = load_acc(h_ptr + base + i * stride_d, mask_i, 0.0)
            r = load_acc(r_ptr + base + i * stride_d, mask_i, 0.0)
            dar = load_acc(dar_ptr + pid * d + i, mask_i, 0.0)
            if t == 0:
                delta = tl.zeros((BLOCK,), dtype=tl.float32)
            else:
                delta = load_acc(out_ptr + prev_base + i * stride_d, mask_i, 0.0)
            r_p = r * (1.0 - r)
            du = r * delta + h * (r_p * dar)
            store_acc(du_ptr + pid * d + i, du, mask_i)

        for i0 in range(0, d, BLOCK):
            i = i0 + offs
            mask_i = i < d
            h = load_acc(h_ptr + base + i * stride_d, mask_i, 0.0)
            z = load_acc(z_ptr + base + i * stride_d, mask_i, 0.0)
            n = load_acc(n_ptr + base + i * stride_d, mask_i, 0.0)
            res = load_acc(res_ptr + base + i * stride_d, mask_i, 0.0)
            daz = load_acc(daz_ptr + pid * d + i, mask_i, 0.0)
            if t == 0:
                delta = tl.zeros((BLOCK,), dtype=tl.float32)
            else:
                delta = load_acc(out_ptr + prev_base + i * stride_d, mask_i, 0.0)
            dan = tl.zeros((BLOCK,), dtype=tl.float32)
            for j0 in range(0, d, BLOCK):
                dan += _mix_col_tile(
                    an_ptr + a_base,
                    du_ptr + pid * d,
                    i,
                    j0,
                    d,
                    stride_a_in,
                    stride_a_out,
                    1,
                    BLOCK,
                )
            z_p = z * (1.0 - z)
            n_p = 1.0 - n * n
            outv = (1.0 - z) * delta + (n - h) * (z_p * daz) + z * (n_p * dan) + res
            store_acc(out_ptr + base + i * stride_d, outv, mask_i)


def _pack_gate_bt(t: Tensor, n_heads: int, d_head: int) -> Tensor:
    batch, time = t.shape[:2]
    return t.permute(0, 2, 1, 3).reshape(batch * n_heads, time, d_head).contiguous()


def _factor_scan_tiled_triton(
    gates: tuple[Tensor, Tensor, Tensor, Tensor],
    residual: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    h, z, r, n = gates
    batch, time = residual.shape[:2]
    dt = residual.dtype
    narrow = dt in (torch.float16, torch.bfloat16)

    def prep(x: Tensor) -> Tensor:
        return _pack_gate_bt(x.float() if narrow else x, n_heads, d_head)

    hp, zp, rp, np_ = prep(h), prep(z), prep(r), prep(n)
    res_p = (
        (residual.float() if narrow else residual)
        .reshape(batch, time, n_heads, d_head)
        .permute(0, 2, 1, 3)
        .reshape(batch * n_heads, time, d_head)
        .contiguous()
    )
    out_p = torch.empty_like(res_p)
    n_rows = batch * n_heads
    daz = torch.empty(n_rows, d_head, device=residual.device, dtype=torch.float32)
    dar = torch.empty_like(daz)
    du = torch.empty_like(daz)
    az = (a_z.float() if narrow else a_z).contiguous()
    ar = (a_r.float() if narrow else a_r).contiguous()
    an = (a_n.float() if narrow else a_n).contiguous()
    _factor_scan_tiled_kernel[(n_rows,)](
        hp,
        zp,
        rp,
        np_,
        res_p,
        out_p,
        daz,
        dar,
        du,
        az,
        ar,
        an,
        time,
        d_head,
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        n_heads,
        BLOCK=_TILE_D,
    )
    out = (
        out_p.reshape(batch, n_heads, time, d_head)
        .permute(0, 2, 1, 3)
        .reshape(batch, time, n_heads * d_head)
    )
    return out.to(dtype=dt) if narrow else out


@triton.jit
def _factor_reverse_tiled_kernel(
    h_ptr,
    z_ptr,
    r_ptr,
    n_ptr,
    part_ptr,
    out_ptr,
    mu_ptr,
    wz_ptr,
    wn_ptr,
    gdu_ptr,
    wr_ptr,
    t2_ptr,
    az_ptr,
    ar_ptr,
    an_ptr,
    time,
    d,
    stride_row,
    stride_t,
    stride_d,
    stride_a_h,
    stride_a_in,
    stride_a_out,
    n_heads,
    BLOCK: tl.constexpr,
):
    """Eq. 2.6 reverse; fused ``A_z``/``A_n`` tile pass (``d_head > 128``)."""
    pid = tl.program_id(0)
    head = pid % n_heads
    a_base = head * stride_a_h
    offs = tl.arange(0, BLOCK)

    for i0 in range(0, d, BLOCK):
        i = i0 + offs
        store_acc(mu_ptr + pid * d + i, tl.zeros((BLOCK,), dtype=tl.float32), i < d)

    for t_rev in range(0, time):
        t = time - 1 - t_rev
        base = pid * stride_row + t * stride_t
        if t_rev > 0:
            tp = t + 1
            tp_base = pid * stride_row + tp * stride_t
            for i0 in range(0, d, BLOCK):
                i = i0 + offs
                mask_i = i < d
                h = load_acc(h_ptr + tp_base + i * stride_d, mask_i, 0.0)
                z = load_acc(z_ptr + tp_base + i * stride_d, mask_i, 0.0)
                n = load_acc(n_ptr + tp_base + i * stride_d, mask_i, 0.0)
                mu = load_acc(mu_ptr + pid * d + i, mask_i, 0.0)
                z_p = z * (1.0 - z)
                n_p = 1.0 - n * n
                store_acc(wz_ptr + pid * d + i, z_p * ((n - h) * mu), mask_i)
                store_acc(wn_ptr + pid * d + i, n_p * (z * mu), mask_i)

            for i0 in range(0, d, BLOCK):
                i = i0 + offs
                mask_i = i < d
                t2 = tl.zeros((BLOCK,), dtype=tl.float32)
                g_du = tl.zeros((BLOCK,), dtype=tl.float32)
                for j0 in range(0, d, BLOCK):
                    j = j0 + offs
                    mask_j = j < d
                    mask_ij = mask_i[:, None] & mask_j[None, :]
                    az = load_acc(
                        az_ptr
                        + a_base
                        + i[:, None] * stride_a_in
                        + j[None, :] * stride_a_out,
                        mask_ij,
                        0.0,
                    )
                    an = load_acc(
                        an_ptr
                        + a_base
                        + i[:, None] * stride_a_in
                        + j[None, :] * stride_a_out,
                        mask_ij,
                        0.0,
                    )
                    wz = load_acc(wz_ptr + pid * d + j, mask_j, 0.0)
                    wn = load_acc(wn_ptr + pid * d + j, mask_j, 0.0)
                    t2 += tl.sum(az * wz[None, :], axis=1)
                    g_du += tl.sum(an * wn[None, :], axis=1)
                store_acc(t2_ptr + pid * d + i, t2, mask_i)
                store_acc(gdu_ptr + pid * d + i, g_du, mask_i)

            for i0 in range(0, d, BLOCK):
                i = i0 + offs
                mask_i = i < d
                h = load_acc(h_ptr + tp_base + i * stride_d, mask_i, 0.0)
                r = load_acc(r_ptr + tp_base + i * stride_d, mask_i, 0.0)
                g_du = load_acc(gdu_ptr + pid * d + i, mask_i, 0.0)
                r_p = r * (1.0 - r)
                store_acc(wr_ptr + pid * d + i, r_p * (h * g_du), mask_i)

            for i0 in range(0, d, BLOCK):
                i = i0 + offs
                mask_i = i < d
                t3_ar = tl.zeros((BLOCK,), dtype=tl.float32)
                for j0 in range(0, d, BLOCK):
                    j = j0 + offs
                    mask_j = j < d
                    ar = load_acc(
                        ar_ptr
                        + a_base
                        + i[:, None] * stride_a_in
                        + j[None, :] * stride_a_out,
                        mask_i[:, None] & mask_j[None, :],
                        0.0,
                    )
                    wr = load_acc(wr_ptr + pid * d + j, mask_j, 0.0)
                    t3_ar += tl.sum(ar * wr[None, :], axis=1)
                z = load_acc(z_ptr + tp_base + i * stride_d, mask_i, 0.0)
                r = load_acc(r_ptr + tp_base + i * stride_d, mask_i, 0.0)
                mu = load_acc(mu_ptr + pid * d + i, mask_i, 0.0)
                t2 = load_acc(t2_ptr + pid * d + i, mask_i, 0.0)
                g_du = load_acc(gdu_ptr + pid * d + i, mask_i, 0.0)
                store_acc(
                    mu_ptr + pid * d + i,
                    (1.0 - z) * mu + t2 + r * g_du + t3_ar,
                    mask_i,
                )

        for i0 in range(0, d, BLOCK):
            i = i0 + offs
            mask_i = i < d
            part = load_acc(part_ptr + base + i * stride_d, mask_i, 0.0)
            mu = load_acc(mu_ptr + pid * d + i, mask_i, 0.0) + part
            store_acc(mu_ptr + pid * d + i, mu, mask_i)
            store_acc(out_ptr + base + i * stride_d, mu, mask_i)


def _factor_reverse_tiled_triton(
    gates: tuple[Tensor, Tensor, Tensor, Tensor],
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    *,
    n_heads: int,
    d_head: int,
) -> Tensor:
    """Large-head reverse: tiled Triton ``J^T`` (``d_head > 128``)."""
    h, z, r, n = gates
    batch, time = partial.shape[:2]
    dt = partial.dtype
    narrow = dt in (torch.float16, torch.bfloat16)

    def prep(x: Tensor) -> Tensor:
        return _pack_gate_bt(x.float() if narrow else x, n_heads, d_head)

    hp, zp, rp, np_ = prep(h), prep(z), prep(r), prep(n)
    part_p = (
        (partial.float() if narrow else partial)
        .reshape(batch, time, n_heads, d_head)
        .permute(0, 2, 1, 3)
        .reshape(batch * n_heads, time, d_head)
        .contiguous()
    )
    out_p = torch.empty_like(part_p)
    n_rows = batch * n_heads
    mu = torch.empty(n_rows, d_head, device=partial.device, dtype=torch.float32)
    wz = torch.empty_like(mu)
    wn = torch.empty_like(mu)
    gdu = torch.empty_like(mu)
    wr = torch.empty_like(mu)
    t2 = torch.empty_like(mu)
    az = (a_z.float() if narrow else a_z).contiguous()
    ar = (a_r.float() if narrow else a_r).contiguous()
    an = (a_n.float() if narrow else a_n).contiguous()
    block = 64 if d_head >= 192 else _TILE_D
    _factor_reverse_tiled_kernel[(n_rows,)](
        hp,
        zp,
        rp,
        np_,
        part_p,
        out_p,
        mu,
        wz,
        wn,
        gdu,
        wr,
        t2,
        az,
        ar,
        an,
        time,
        d_head,
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        az.stride(0),
        az.stride(1),
        az.stride(2),
        n_heads,
        BLOCK=block,
        num_warps=8,
    )
    out = (
        out_p.reshape(batch, n_heads, time, d_head)
        .permute(0, 2, 1, 3)
        .reshape(batch, time, n_heads * d_head)
    )
    return out.to(dtype=dt) if narrow else out
