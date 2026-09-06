"""Factorized Newton for ``ParaM2RNN`` (no dense ``(KV)×(KV)``).

Alg. 1 with ``m2rnn_jvp`` inclusive scan. CUDA: Triton fused time-loop
(SRAM ``K,V ≤ 64``). CPU / large ``K,V`` / ``residual_history``: eager
factor scan. Dense-J oracle stays off this path.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.cells.para_m2rnn import ParaM2RNN
from pararnn.kernels._fused_common import _tanh
from pararnn.kernels.m2rnn_factor import (
    m2rnn_frozen_w_scan,
    m2rnn_gates,
    m2rnn_jt_mvp,
    m2rnn_jvp,
)
from pararnn.kernels.precision import is_fused_dtype_supported, load_acc, store_acc
from pararnn.layout import prepend_state

log = logging.getLogger(__name__)

# Full W + state tiles in SRAM. 64²×4×2 ≈ 32 KiB for W+delta, plus z/R.
_M2RNN_SRAM_MAX = 64
# Hybrid tiled factor scan (PyTorch gates + Triton T-loop) for larger K,V.
_M2RNN_TILE = 32


def _pow2_ge(n: int, *, minimum: int = 16) -> int:
    p = minimum
    while p < n:
        p *= 2
    return p


def _sram_ok(k_dim: int, v_dim: int) -> bool:
    return k_dim <= _M2RNN_SRAM_MAX and v_dim <= _M2RNN_SRAM_MAX


def can_fuse_m2rnn(k_dim: int, v_dim: int, ref: Tensor) -> bool:
    """Whether CUDA factorized Newton (SRAM or hybrid tiled) can run."""
    del k_dim, v_dim
    if not ref.is_cuda:
        return False
    return is_fused_dtype_supported(ref.dtype, ref.device)


def newton_m2rnn_factorized(
    cell: ParaM2RNN,
    x: Tensor,
    *,
    max_iters: int,
    omega: float = 1.0,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    residual_history: list[float] | None = None,
    force_eager: bool = False,
    residual_atol: float | None = None,
    frozen_w_init: bool = False,
) -> Tensor:
    """Alg. 1 for M²RNN with factorized ``J δ``.

    Parameters
    ----------
    cell : ParaM2RNN
    x : Tensor
        ``(B, T, d_in)``.
    max_iters : int
        Newton steps cap ``K``. Under App. A init, ``K*(T)≈Θ(log T)``;
        prefer ``residual_atol`` early-stop over a fixed over-provisioned ``K``.
    omega : float
        Damping (``1.0`` = undamped).
    h0, states : optional
        Initial state / warm start ``(B, K, V)`` / ``(B, T, K, V)``.
    residual_history : list, optional
        Appended with ``max |F|`` after each Newton step (forces eager).
    force_eager : bool
        Skip Triton even when geometry fits (dense-oracle / debug).
    residual_atol : float or None
        Host early-stop when ``max|F| < atol`` (hybrid / eager loops).
    frozen_w_init : bool
        If True and ``states is None``, warm-start with ``m2rnn_frozen_w_scan``
        (``W=0`` linear scan) before Newton. Wired from
        ``NewtonConfig.picard_iters >= 1`` for ``ParaM2RNN``.
    """
    if not isinstance(cell, ParaM2RNN):
        raise TypeError(f"newton_m2rnn_factorized needs ParaM2RNN, got {type(cell).__name__}")
    batch, time, _ = x.shape
    k_dim, v_dim = cell.k_dim, cell.v_dim
    wx = cell.W_x(x)
    k = wx[..., :k_dim]
    v = wx[..., k_dim : k_dim + v_dim]
    f = torch.sigmoid(wx[..., -1])
    w = cell.W

    use_fused = (
        not force_eager
        and residual_history is None
        and states is None
        and can_fuse_m2rnn(k_dim, v_dim, x)
    )
    if use_fused:
        if _sram_ok(k_dim, v_dim) and residual_atol is None:
            from pararnn.kernels.custom_ops import newton_m2rnn_fused

            out = newton_m2rnn_fused(
                k.contiguous(),
                v.contiguous(),
                f.contiguous(),
                w.contiguous(),
                h0.contiguous() if h0 is not None else None,
                max_iters=int(max_iters),
                omega=float(omega),
                frozen_w_init=bool(frozen_w_init),
            )
        elif _sram_ok(k_dim, v_dim):
            # residual_atol: one Newton iter per launch so we can early-stop.
            out = _newton_m2rnn_sram_early(
                k,
                v,
                f,
                w,
                h0,
                max_iters=max_iters,
                omega=omega,
                residual_atol=residual_atol,
                residual_history=None,
                frozen_w_init=frozen_w_init,
            )
        else:
            out = _newton_m2rnn_hybrid_tiled(
                k,
                v,
                f,
                w,
                h0,
                max_iters=max_iters,
                omega=omega,
                residual_atol=residual_atol,
                residual_history=None,
                frozen_w_init=frozen_w_init,
            )
        if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "newton_m2rnn_fused",
                extra={
                    "batch": batch,
                    "seq_len": time,
                    "k_dim": k_dim,
                    "v_dim": v_dim,
                    "max_iters": max_iters,
                    "residual_atol": residual_atol,
                    "frozen_w_init": frozen_w_init,
                },
            )
        return out

    return _newton_m2rnn_eager(
        k,
        v,
        f,
        w,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        states=states,
        residual_history=residual_history,
        k_dim=k_dim,
        v_dim=v_dim,
        residual_atol=residual_atol,
        frozen_w_init=frozen_w_init,
    )


def _newton_m2rnn_eager(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None,
    states: Tensor | None,
    residual_history: list[float] | None,
    k_dim: int,
    v_dim: int,
    residual_atol: float | None = None,
    frozen_w_init: bool = False,
) -> Tensor:
    batch, time = k.shape[0], k.shape[1]
    if states is None:
        if frozen_w_init:
            states = m2rnn_frozen_w_scan(k, v, f, h0=h0)
        else:
            zeros = k.new_zeros(batch, time, k_dim, v_dim)
            h_prev0 = prepend_state(zeros, h0)
            states, _ = m2rnn_gates(h_prev0, k, v, f, w)

    for _ in range(max_iters):
        h_prev = prepend_state(states, h0)
        pred, acts = m2rnn_gates(h_prev, k, v, f, w)
        residual = pred - states
        res_amax = float(residual.detach().abs().amax())
        if residual_history is not None and not torch.compiler.is_compiling():
            residual_history.append(res_amax)
        if residual_atol is not None and res_amax < residual_atol:
            break
        delta = _factor_scan(acts["z"], acts["f"], acts["w"], residual)
        states = states + omega * delta
    return states


def reverse_factor_scan_m2rnn(
    cell: ParaM2RNN,
    h_prev: Tensor,
    partial: Tensor,
    *,
    x: Tensor | None = None,
    wx: Tensor | None = None,
) -> Tensor:
    """Eq. 2.6 reverse with factorized ``Jᵀ``."""
    k_dim, v_dim = cell.k_dim, cell.v_dim
    if wx is None:
        if x is None:
            raise ValueError("reverse_factor_scan_m2rnn needs x= or wx=")
        wx = cell.W_x(x)
    k = wx[..., :k_dim]
    v = wx[..., k_dim : k_dim + v_dim]
    f = torch.sigmoid(wx[..., -1])
    _, acts = m2rnn_gates(h_prev, k, v, f, cell.W)
    z = acts["z"]
    f_act = acts["f"]
    while f_act.dim() > 2:
        f_act = f_act.squeeze(-1)
    if can_fuse_m2rnn(k_dim, v_dim, partial) and _sram_ok(k_dim, v_dim):
        from pararnn.kernels.custom_ops import reverse_m2rnn_factor

        return reverse_m2rnn_factor(
            z.contiguous(),
            f_act.contiguous(),
            cell.W.contiguous(),
            partial.contiguous(),
        )
    return _factor_reverse(z, f_act, acts["w"], partial)


def m2rnn_t0_vjp(
    cell: ParaM2RNN,
    h_prev: Tensor,
    mu: Tensor,
    *,
    x: Tensor | None = None,
    wx: Tensor | None = None,
) -> Tensor:
    """``J_0^T μ_0`` for the paper ``h_0`` adjoint (factorized)."""
    k_dim, v_dim = cell.k_dim, cell.v_dim
    if wx is None:
        if x is None:
            raise ValueError("m2rnn_t0_vjp needs x= or wx=")
        wx = cell.W_x(x)
    k0 = wx[:, 0, :k_dim]
    v0 = wx[:, 0, k_dim : k_dim + v_dim]
    f0 = torch.sigmoid(wx[:, 0, -1])
    _, acts0 = m2rnn_gates(h_prev[:, 0], k0, v0, f0, cell.W)
    return m2rnn_jt_mvp(acts0, mu[:, 0])


def _factor_scan(z: Tensor, f: Tensor, w: Tensor, residual: Tensor) -> Tensor:
    """Inclusive ``δ_t = J_t δ_{t-1} + R_t`` (eager)."""
    batch, time = residual.shape[:2]
    f_b = f
    while f_b.dim() < residual.dim():
        f_b = f_b.unsqueeze(-1)
    scale = (1.0 - f_b) * (1.0 - z.square())
    delta = residual.new_zeros(batch, *residual.shape[2:])
    out = residual.new_empty(residual.shape)
    for t in range(time):
        delta = f_b[:, t] * delta + scale[:, t] * (delta @ w) + residual[:, t]
        out[:, t] = delta
    return out


def _factor_reverse(z: Tensor, f: Tensor, w: Tensor, partial: Tensor) -> Tensor:
    batch, time = partial.shape[:2]
    f_b = f
    while f_b.dim() < partial.dim():
        f_b = f_b.unsqueeze(-1)
    scale = (1.0 - f_b) * (1.0 - z.square())
    w_t = w.transpose(-1, -2)
    mu = partial.new_zeros(batch, *partial.shape[2:])
    out = partial.new_empty(partial.shape)
    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            mu = f_b[:, t + 1] * mu + (scale[:, t + 1] * mu) @ w_t
        mu = mu + partial[:, t]
        out[:, t] = mu
    return out


# ---------------------------------------------------------------------------
# Triton fused Newton / reverse (SRAM path, K,V ≤ 64)
# ---------------------------------------------------------------------------


@triton.jit
def _m2_mul_kw(a, b):
    """``a @ b`` with ``a`` ``(K,V)``, ``b`` ``(V,V)``, padded power-of-two blocks."""
    return tl.dot(a, b)


@triton.jit
def _m2_gates(h_prev, k, v, f, w):
    s = _m2_mul_kw(h_prev, w) + k[:, None] * v[None, :]
    z = _tanh(s)
    h_new = f * h_prev + (1.0 - f) * z
    return h_new, z


@triton.jit
def _m2_jvp(z, f, w, delta):
    dz = (1.0 - z * z) * _m2_mul_kw(delta, w)
    return f * delta + (1.0 - f) * dz


@triton.jit
def _m2_jt(z, f, w_t, mu):
    dz_bar = (1.0 - f) * (1.0 - z * z) * mu
    return f * mu + _m2_mul_kw(dz_bar, w_t)


@triton.jit
def _newton_m2rnn_kernel(
    k_ptr,
    v_ptr,
    f_ptr,
    w_ptr,
    h_ptr,
    delta_ptr,
    h0_ptr,
    has_h0,
    time,
    k_dim,
    v_dim,
    omega,
    max_iters,
    stride_k_b,
    stride_k_t,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_d,
    stride_f_b,
    stride_f_t,
    stride_w_r,
    stride_w_c,
    stride_h_b,
    stride_h_t,
    stride_h_k,
    stride_h_v,
    stride_d_b,
    stride_d_t,
    stride_d_k,
    stride_d_v,
    stride_h0_b,
    stride_h0_k,
    stride_h0_v,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    USE_FROZEN_W: tl.constexpr,
):
    """Guess + ``max_iters`` factorized Newton; one program per batch.

    ``USE_FROZEN_W=0``: App. A parallel guess (zero prev, full ``W``).
    ``USE_FROZEN_W=1``: ``W=0`` carry scan ``H_t=f H_{t-1}+(1-f)tanh(kvᵀ)``.
    Each Newton step freezes ``H``, scans ``δ`` into ``delta_ptr``, then
    ``H ← H + ω δ`` (Alg. 1).
    """
    b = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < k_dim
    mask_v = offs_v < v_dim
    mask_kv = mask_k[:, None] & mask_v[None, :]
    mask_vv = mask_v[:, None] & mask_v[None, :]

    w = load_acc(
        w_ptr + offs_v[:, None] * stride_w_r + offs_v[None, :] * stride_w_c,
        mask_vv,
        0.0,
    )

    h0_m = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    if has_h0 != 0:
        h0_m = load_acc(
            h0_ptr
            + b * stride_h0_b
            + offs_k[:, None] * stride_h0_k
            + offs_v[None, :] * stride_h0_v,
            mask_kv,
            0.0,
        )

    h_carry = h0_m
    for t in range(0, time):
        kk = load_acc(k_ptr + b * stride_k_b + t * stride_k_t + offs_k * stride_k_d, mask_k, 0.0)
        vv = load_acc(v_ptr + b * stride_v_b + t * stride_v_t + offs_v * stride_v_d, mask_v, 0.0)
        ff = load_acc(f_ptr + b * stride_f_b + t * stride_f_t, True, 0.0)
        if USE_FROZEN_W:
            z = _tanh(kk[:, None] * vv[None, :])
            h_new = ff * h_carry + (1.0 - ff) * z
            h_carry = h_new
        else:
            # App. A: parallel guess from zero prev (only t=0 sees h0).
            h_prev = tl.where(t == 0, h0_m, tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32))
            h_new, _ = _m2_gates(h_prev, kk, vv, ff, w)
        store_acc(
            h_ptr
            + b * stride_h_b
            + t * stride_h_t
            + offs_k[:, None] * stride_h_k
            + offs_v[None, :] * stride_h_v,
            h_new,
            mask_kv,
        )

    for _it in range(0, max_iters):
        delta = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
        for t in range(0, time):
            kk = load_acc(
                k_ptr + b * stride_k_b + t * stride_k_t + offs_k * stride_k_d, mask_k, 0.0
            )
            vv = load_acc(
                v_ptr + b * stride_v_b + t * stride_v_t + offs_v * stride_v_d, mask_v, 0.0
            )
            ff = load_acc(f_ptr + b * stride_f_b + t * stride_f_t, True, 0.0)
            h_prev_nm1 = load_acc(
                h_ptr
                + b * stride_h_b
                + (t - 1) * stride_h_t
                + offs_k[:, None] * stride_h_k
                + offs_v[None, :] * stride_h_v,
                mask_kv & (t > 0),
                0.0,
            )
            h_prev = tl.where(t == 0, h0_m, h_prev_nm1)
            h_guess = load_acc(
                h_ptr
                + b * stride_h_b
                + t * stride_h_t
                + offs_k[:, None] * stride_h_k
                + offs_v[None, :] * stride_h_v,
                mask_kv,
                0.0,
            )
            pred, z = _m2_gates(h_prev, kk, vv, ff, w)
            residual = pred - h_guess
            delta = _m2_jvp(z, ff, w, delta) + residual
            store_acc(
                delta_ptr
                + b * stride_d_b
                + t * stride_d_t
                + offs_k[:, None] * stride_d_k
                + offs_v[None, :] * stride_d_v,
                delta,
                mask_kv,
            )
        for t in range(0, time):
            h_guess = load_acc(
                h_ptr
                + b * stride_h_b
                + t * stride_h_t
                + offs_k[:, None] * stride_h_k
                + offs_v[None, :] * stride_h_v,
                mask_kv,
                0.0,
            )
            dlt = load_acc(
                delta_ptr
                + b * stride_d_b
                + t * stride_d_t
                + offs_k[:, None] * stride_d_k
                + offs_v[None, :] * stride_d_v,
                mask_kv,
                0.0,
            )
            store_acc(
                h_ptr
                + b * stride_h_b
                + t * stride_h_t
                + offs_k[:, None] * stride_h_k
                + offs_v[None, :] * stride_h_v,
                h_guess + omega * dlt,
                mask_kv,
            )


@triton.jit
def _reverse_m2rnn_kernel(
    z_ptr,
    f_ptr,
    w_ptr,
    partial_ptr,
    out_ptr,
    time,
    k_dim,
    v_dim,
    stride_z_b,
    stride_z_t,
    stride_z_k,
    stride_z_v,
    stride_f_b,
    stride_f_t,
    stride_w_r,
    stride_w_c,
    stride_p_b,
    stride_p_t,
    stride_p_k,
    stride_p_v,
    stride_o_b,
    stride_o_t,
    stride_o_k,
    stride_o_v,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    b = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < k_dim
    mask_v = offs_v < v_dim
    mask_kv = mask_k[:, None] & mask_v[None, :]
    mask_vv = mask_v[:, None] & mask_v[None, :]

    w = load_acc(
        w_ptr + offs_v[:, None] * stride_w_r + offs_v[None, :] * stride_w_c,
        mask_vv,
        0.0,
    )
    w_t = tl.trans(w)

    mu = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    for t in range(time - 1, -1, -1):
        if t + 1 < time:
            z = load_acc(
                z_ptr
                + b * stride_z_b
                + (t + 1) * stride_z_t
                + offs_k[:, None] * stride_z_k
                + offs_v[None, :] * stride_z_v,
                mask_kv,
                0.0,
            )
            ff = load_acc(f_ptr + b * stride_f_b + (t + 1) * stride_f_t, True, 0.0)
            mu = _m2_jt(z, ff, w_t, mu)
        part = load_acc(
            partial_ptr
            + b * stride_p_b
            + t * stride_p_t
            + offs_k[:, None] * stride_p_k
            + offs_v[None, :] * stride_p_v,
            mask_kv,
            0.0,
        )
        mu = mu + part
        store_acc(
            out_ptr
            + b * stride_o_b
            + t * stride_o_t
            + offs_k[:, None] * stride_o_k
            + offs_v[None, :] * stride_o_v,
            mu,
            mask_kv,
        )


def _newton_m2rnn_fused_impl(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    frozen_w_init: bool = False,
) -> Tensor:
    """Tensor Alg. 1 for M²RNN. Public entry: ``pararnn::newton_m2rnn_fused``."""
    from pararnn.kernels.precision import validate_cuda_tensors

    validate_cuda_tensors(k, v, f, w, name="newton_m2rnn_fused")
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    if v.shape != (batch, time, v_dim):
        raise ValueError(f"v shape {tuple(v.shape)} != {(batch, time, v_dim)}")
    if f.shape != (batch, time):
        raise ValueError(f"f shape {tuple(f.shape)} != {(batch, time)}")
    if w.shape != (v_dim, v_dim):
        raise ValueError(f"w shape {tuple(w.shape)} != {(v_dim, v_dim)}")
    if h0 is not None:
        validate_cuda_tensors(h0, name="newton_m2rnn_fused.h0")
        if h0.shape != (batch, k_dim, v_dim):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, k_dim, v_dim)}")

    if _sram_ok(k_dim, v_dim):
        return _newton_m2rnn_sram(
            k, v, f, w, h0, max_iters=max_iters, omega=omega, frozen_w_init=frozen_w_init
        )
    return _newton_m2rnn_hybrid_tiled(
        k, v, f, w, h0, max_iters=max_iters, omega=omega, frozen_w_init=frozen_w_init
    )


def _newton_m2rnn_sram(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    frozen_w_init: bool = False,
) -> Tensor:
    """SRAM fused Alg. 1 (``K,V ≤ 64``)."""
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    states = k.new_empty(batch, time, k_dim, v_dim)
    deltas = k.new_empty(batch, time, k_dim, v_dim)
    block_k = _pow2_ge(k_dim)
    block_v = _pow2_ge(v_dim)
    h0_buf = h0 if h0 is not None else k.new_zeros(batch, k_dim, v_dim)
    _newton_m2rnn_kernel[(batch,)](
        k,
        v,
        f,
        w,
        states,
        deltas,
        h0_buf,
        0 if h0 is None else 1,
        time,
        k_dim,
        v_dim,
        float(omega),
        int(max_iters),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        f.stride(0),
        f.stride(1),
        w.stride(0),
        w.stride(1),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        deltas.stride(0),
        deltas.stride(1),
        deltas.stride(2),
        deltas.stride(3),
        h0_buf.stride(0),
        h0_buf.stride(1),
        h0_buf.stride(2),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        USE_FROZEN_W=bool(frozen_w_init),
    )
    return states


def _newton_m2rnn_hybrid_tiled(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    residual_atol: float | None = None,
    residual_history: list[float] | None = None,
    frozen_w_init: bool = False,
) -> Tensor:
    """``K,V > 64``: cuBLAS gates + Triton tiled factor scan (Alg. 1).

    Early-stop when ``max|F| < residual_atol`` — large-state wall-clock is
    ``≈ K_used × (gates + scan)``; over-running past ``K*`` loses to sequential.
    """
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    if frozen_w_init:
        states = m2rnn_frozen_w_scan(k, v, f, h0=h0)
    else:
        zeros = k.new_zeros(batch, time, k_dim, v_dim)
        h_prev0 = prepend_state(zeros, h0)
        states, _ = m2rnn_gates(h_prev0, k, v, f, w)
    for _ in range(max_iters):
        h_prev = prepend_state(states, h0)
        pred, acts = m2rnn_gates(h_prev, k, v, f, w)
        residual = pred - states
        res_amax = float(residual.detach().abs().amax())
        if residual_history is not None and not torch.compiler.is_compiling():
            residual_history.append(res_amax)
        if residual_atol is not None and res_amax < residual_atol:
            break
        delta = _factor_scan_tiled_triton(acts["z"], acts["f"], w, residual)
        states = states + omega * delta
    return states


def _newton_m2rnn_sram_early(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None,
    *,
    max_iters: int,
    omega: float,
    residual_atol: float,
    residual_history: list[float] | None = None,
    frozen_w_init: bool = False,
) -> Tensor:
    """SRAM path with host residual early-stop (one Newton iter per launch)."""
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    if frozen_w_init:
        states = m2rnn_frozen_w_scan(k, v, f, h0=h0)
    else:
        zeros = k.new_zeros(batch, time, k_dim, v_dim)
        h_prev0 = prepend_state(zeros, h0)
        states, _ = m2rnn_gates(h_prev0, k, v, f, w)
    for _ in range(max_iters):
        h_prev = prepend_state(states, h0)
        pred, acts = m2rnn_gates(h_prev, k, v, f, w)
        residual = pred - states
        res_amax = float(residual.detach().abs().amax())
        if residual_history is not None and not torch.compiler.is_compiling():
            residual_history.append(res_amax)
        if res_amax < residual_atol:
            break
        # One Newton correction via factor scan (SRAM geometry: tiled or eager scan).
        if _sram_ok(k_dim, v_dim):
            # Reuse tiled scan even for small KV when in early-stop host loop —
            # avoids a separate 1-iter SRAM kernel.
            delta = _factor_scan_tiled_triton(acts["z"], acts["f"], w, residual)
        else:
            delta = _factor_scan(acts["z"], acts["f"], w, residual)
        states = states + omega * delta
    return states


def _reverse_m2rnn_factor_impl(
    z: Tensor,
    f: Tensor,
    w: Tensor,
    partial: Tensor,
) -> Tensor:
    """Factorized reverse scan. Public entry: ``pararnn::reverse_m2rnn_factor``."""
    from pararnn.kernels.precision import validate_cuda_tensors

    validate_cuda_tensors(z, f, w, partial, name="reverse_m2rnn_factor")
    batch, time, k_dim, v_dim = z.shape
    if f.dim() > 2:
        f = f.reshape(batch, time)
    if f.shape != (batch, time):
        raise ValueError(f"f shape {tuple(f.shape)} != {(batch, time)}")
    if w.shape != (v_dim, v_dim):
        raise ValueError(f"w shape {tuple(w.shape)} != {(v_dim, v_dim)}")
    if partial.shape != z.shape:
        raise ValueError(f"partial {tuple(partial.shape)} != z {tuple(z.shape)}")
    if not _sram_ok(k_dim, v_dim):
        return _factor_reverse(z, f, w, partial)

    out = torch.empty_like(partial)
    block_k = _pow2_ge(k_dim)
    block_v = _pow2_ge(v_dim)
    _reverse_m2rnn_kernel[(batch,)](
        z,
        f,
        w,
        partial,
        out,
        time,
        k_dim,
        v_dim,
        z.stride(0),
        z.stride(1),
        z.stride(2),
        z.stride(3),
        f.stride(0),
        f.stride(1),
        w.stride(0),
        w.stride(1),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )
    return out


# ---------------------------------------------------------------------------
# Hybrid tiled factor scan / reverse (K or V > 64)
# ---------------------------------------------------------------------------


@triton.jit
def _factor_scan_m2_tiled_kernel(
    z_ptr,
    f_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    time,
    k_dim,
    v_dim,
    stride_z_b,
    stride_z_t,
    stride_z_k,
    stride_z_v,
    stride_f_b,
    stride_f_t,
    stride_r_b,
    stride_r_t,
    stride_r_k,
    stride_r_v,
    stride_w_r,
    stride_w_c,
    stride_o_b,
    stride_o_t,
    stride_o_k,
    stride_o_v,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Inclusive ``δ_t = J_t δ_{t-1} + R_t`` with tiled ``(K×V)(V×V)``."""
    b = tl.program_id(0)
    for t in range(0, time):
        for i0 in range(0, k_dim, BLOCK_K):
            for j0 in range(0, v_dim, BLOCK_V):
                offs_i = i0 + tl.arange(0, BLOCK_K)
                offs_j = j0 + tl.arange(0, BLOCK_V)
                mask_i = offs_i < k_dim
                mask_j = offs_j < v_dim
                mask = mask_i[:, None] & mask_j[None, :]

                if t == 0:
                    d_prev = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
                else:
                    d_prev = load_acc(
                        out_ptr
                        + b * stride_o_b
                        + (t - 1) * stride_o_t
                        + offs_i[:, None] * stride_o_k
                        + offs_j[None, :] * stride_o_v,
                        mask,
                        0.0,
                    )

                acc = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
                if t > 0:
                    for p0 in range(0, v_dim, BLOCK_V):
                        offs_p = p0 + tl.arange(0, BLOCK_V)
                        mask_p = offs_p < v_dim
                        mask_dp = mask_i[:, None] & mask_p[None, :]
                        mask_wp = mask_p[:, None] & mask_j[None, :]
                        d_tile = load_acc(
                            out_ptr
                            + b * stride_o_b
                            + (t - 1) * stride_o_t
                            + offs_i[:, None] * stride_o_k
                            + offs_p[None, :] * stride_o_v,
                            mask_dp,
                            0.0,
                        )
                        w_tile = load_acc(
                            w_ptr
                            + offs_p[:, None] * stride_w_r
                            + offs_j[None, :] * stride_w_c,
                            mask_wp,
                            0.0,
                        )
                        acc += tl.dot(d_tile, w_tile)

                ff = load_acc(f_ptr + b * stride_f_b + t * stride_f_t, True, 0.0)
                zz = load_acc(
                    z_ptr
                    + b * stride_z_b
                    + t * stride_z_t
                    + offs_i[:, None] * stride_z_k
                    + offs_j[None, :] * stride_z_v,
                    mask,
                    0.0,
                )
                rr = load_acc(
                    res_ptr
                    + b * stride_r_b
                    + t * stride_r_t
                    + offs_i[:, None] * stride_r_k
                    + offs_j[None, :] * stride_r_v,
                    mask,
                    0.0,
                )
                scale = (1.0 - ff) * (1.0 - zz * zz)
                outv = ff * d_prev + scale * acc + rr
                store_acc(
                    out_ptr
                    + b * stride_o_b
                    + t * stride_o_t
                    + offs_i[:, None] * stride_o_k
                    + offs_j[None, :] * stride_o_v,
                    outv,
                    mask,
                )


def _factor_scan_tiled_triton(
    z: Tensor,
    f: Tensor,
    w: Tensor,
    residual: Tensor,
) -> Tensor:
    """Tiled Triton inclusive factor scan (``K`` or ``V`` may exceed SRAM)."""
    from pararnn.kernels.precision import validate_cuda_tensors

    validate_cuda_tensors(z, f, w, residual, name="m2rnn_factor_scan_tiled")
    batch, time, k_dim, v_dim = z.shape
    f_b = f
    while f_b.dim() > 2:
        f_b = f_b.squeeze(-1)
    if f_b.shape != (batch, time):
        f_b = f_b.reshape(batch, time)
    out = torch.empty_like(residual)
    block = _M2RNN_TILE
    _factor_scan_m2_tiled_kernel[(batch,)](
        z.contiguous(),
        f_b.contiguous(),
        residual.contiguous(),
        w.contiguous(),
        out,
        time,
        k_dim,
        v_dim,
        z.stride(0),
        z.stride(1),
        z.stride(2),
        z.stride(3),
        f_b.stride(0),
        f_b.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        residual.stride(3),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_K=block,
        BLOCK_V=block,
    )
    return out

