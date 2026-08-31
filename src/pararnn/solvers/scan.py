"""Parallel prefix scan for the Newton linear system (eq. 2.4).

Work-efficient Blelloch scan on the monoid
``(J_r, r_r) ⊕ (J_l, r_l) = (J_r J_l, J_r r_l + r_r)``.
Pad time to the next power of two with the identity ``(I, 0)``.
Do not use the HTML ``l - 2^i + 1`` index; see ``pararnn.layout``.

2×2 blocks are four elementwise muls (not ``einsum`` → tiny ``bmm``).
Reverse scan is paper eq. 2.6 (Jacobian transpose, unroll backwards).

``torch.associative_scan`` is a CUDA/compile prototype without autograd and
without CPU — we do not use it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

_Compose = Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
_FillIdent = Callable[[Tensor], None]


def scan_diag(jac: Tensor, residual: Tensor, *, backend: str = "eager") -> Tensor:
    """Solve ``δ_t = jac_t * δ_{t-1} + residual_t`` with ``δ_{<0} = 0``.

    ``jac`` and ``residual``: (batch, time, d). ``jac`` is the diagonal of J.
    ``backend``: ``eager`` (default) or ``triton`` (CUDA float16/float32).
    """
    if backend == "triton":
        from pararnn.kernels import scan_diag_triton

        return scan_diag_triton(jac, residual)
    if backend != "eager":
        raise ValueError(f"unknown scan backend {backend!r}")
    return _scan_acc(jac, residual, _compose_diag, _fill_ident_diag)


def scan_block2(jac: Tensor, residual: Tensor, *, backend: str = "eager") -> Tensor:
    """Same recurrence with 2×2 blocks per feature.

    ``jac``: (batch, time, 2, 2, d) with ``[..., out, in, d]``.
    ``residual`` / result: (batch, time, 2, d).
    ``backend``: ``eager`` (default) or ``triton`` (CUDA float16/float32).
    """
    if backend == "triton":
        from pararnn.kernels.scan_block2 import scan_block2_triton

        return scan_block2_triton(jac, residual)
    if backend != "eager":
        raise ValueError(f"unknown scan backend {backend!r}")
    return _scan_acc(jac, residual, _compose_block2, _fill_ident_block2)


def scan_dense(jac: Tensor, residual: Tensor, *, backend: str = "eager") -> Tensor:
    """Exact Newton scan for a full ``d_h × d_h`` Jacobian (DEER).

    ``jac``: (batch, time, d, d) with ``[..., out, in]``. ``residual``: (batch, time, d).
    Compose is ``bmm`` — ``O(T d^3)`` after the log-depth scan. Triton is not
    this path; ``backend='triton'`` still runs the eager Blelloch.
    """
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    return _scan_acc(jac, residual, _compose_dense, _fill_ident_dense)


def reverse_scan_diag(
    jac: Tensor, partial: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Total adjoint ``∇_{h_t} L`` from direct ``∂_{h_t} L`` (eq. 2.6, diagonal).

    ``∇_{h_{t-1}} L = J_t ∇_{h_t} L + ∂_{h_{t-1}} L``, ``∇_{h_{T-1}} L = ∂_{h_{T-1}} L``.
    Diagonal ``J`` is symmetric. ``jac[:, 0]`` is unused (no ``h_{-1}`` in the trajectory).
    """
    j_rev = jac.new_zeros(jac.shape)
    j_rev[:, 1:] = jac.flip(1)[:, :-1]
    return scan_diag(j_rev, partial.flip(1), backend=backend).flip(1)


def reverse_scan_block2(
    jac: Tensor, partial: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Eq. 2.6 with 2×2 blocks: uses ``J^T`` (swap ``out``/``in``)."""
    j_t = jac.transpose(-3, -2)
    j_rev = j_t.new_zeros(j_t.shape)
    j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    return scan_block2(j_rev, partial.flip(1), backend=backend).flip(1)


def reverse_scan_dense(
    jac: Tensor, partial: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Eq. 2.6 with a full matrix: uses ``J^T``."""
    j_t = jac.transpose(-1, -2)
    j_rev = j_t.new_zeros(j_t.shape)
    j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    return scan_dense(j_rev, partial.flip(1), backend=backend).flip(1)


def _scan_acc(
    jac: Tensor,
    residual: Tensor,
    compose: _Compose,
    fill_ident: _FillIdent,
) -> Tensor:
    """Blelloch in fp32 when DRAM is fp16 (Newton accumulators)."""
    if jac.dtype == torch.float16:
        out = _blelloch_inclusive(jac.float(), residual.float(), compose, fill_ident)
        return out.to(dtype=torch.float16)
    return _blelloch_inclusive(jac, residual, compose, fill_ident)


def _blelloch_inclusive(
    jac: Tensor,
    residual: Tensor,
    compose: _Compose,
    fill_ident: _FillIdent,
) -> Tensor:
    time = residual.shape[1]
    if time <= 1:
        return residual.clone()
    n = 1 << (time - 1).bit_length()
    j = jac.new_empty(jac.shape[0], n, *jac.shape[2:])
    r = residual.new_empty(residual.shape[0], n, *residual.shape[2:])
    j[:, :time] = jac
    r[:, :time] = residual
    if n > time:
        fill_ident(j[:, time:])
        r[:, time:] = 0
    _blelloch_exclusive_(j, r, compose, fill_ident)
    # exclusive r-component at t is δ_{t-1}; inclusive δ_t = J_t δ_{t-1} + r_t
    prefix = r[:, :time]
    if jac.dim() == residual.dim():
        return jac * prefix + residual
    if _is_dense(jac, residual):
        return (jac @ prefix.unsqueeze(-1)).squeeze(-1) + residual
    return _mv2(jac, prefix) + residual


def _blelloch_exclusive_(
    j: Tensor,
    r: Tensor,
    compose: _Compose,
    fill_ident: _FillIdent,
) -> None:
    """In-place exclusive scan along time. ``j``/``r`` length is a power of two."""
    n = j.shape[1]
    step = 2
    while step <= n:
        right = torch.arange(step - 1, n, step, device=j.device)
        left = right - (step // 2)
        j[:, right], r[:, right] = compose(j[:, right], r[:, right], j[:, left], r[:, left])
        step *= 2
    fill_ident(j[:, n - 1])
    r[:, n - 1] = 0
    step = n
    while step >= 2:
        right = torch.arange(step - 1, n, step, device=j.device)
        left = right - (step // 2)
        j_left = j[:, left].clone()
        r_left = r[:, left].clone()
        j[:, left] = j[:, right]
        r[:, left] = r[:, right]
        # parent prefix is already in ``right``; it must run *before* the left half.
        j[:, right], r[:, right] = compose(j_left, r_left, j[:, right], r[:, right])
        step //= 2


def _compose_diag(
    j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor
) -> tuple[Tensor, Tensor]:
    return j_r * j_l, j_r * r_l + r_r


def _compose_block2(
    j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor
) -> tuple[Tensor, Tensor]:
    return _mm2(j_r, j_l), _mv2(j_r, r_l) + r_r


def _compose_dense(
    j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor
) -> tuple[Tensor, Tensor]:
    return j_r @ j_l, (j_r @ r_l.unsqueeze(-1)).squeeze(-1) + r_r


def _is_dense(jac: Tensor, residual: Tensor) -> bool:
    return (
        jac.dim() == residual.dim() + 1
        and jac.shape[-1] == residual.shape[-1]
        and jac.shape[-2] == residual.shape[-1]
    )


def _mv2(jac: Tensor, vec: Tensor) -> Tensor:
    """``J @ v`` per feature: four muls, layout ``[..., out, in, d]``."""
    j00, j01 = jac[..., 0, 0, :], jac[..., 0, 1, :]
    j10, j11 = jac[..., 1, 0, :], jac[..., 1, 1, :]
    v0, v1 = vec[..., 0, :], vec[..., 1, :]
    return torch.stack((j00 * v0 + j01 * v1, j10 * v0 + j11 * v1), dim=-2)


def _mm2(j_right: Tensor, j_left: Tensor) -> Tensor:
    """``J_right @ J_left`` per feature."""
    a00, a01 = j_right[..., 0, 0, :], j_right[..., 0, 1, :]
    a10, a11 = j_right[..., 1, 0, :], j_right[..., 1, 1, :]
    b00, b01 = j_left[..., 0, 0, :], j_left[..., 0, 1, :]
    b10, b11 = j_left[..., 1, 0, :], j_left[..., 1, 1, :]
    out00 = a00 * b00 + a01 * b10
    out01 = a00 * b01 + a01 * b11
    out10 = a10 * b00 + a11 * b10
    out11 = a10 * b01 + a11 * b11
    return torch.stack(
        (
            torch.stack((out00, out01), dim=-2),
            torch.stack((out10, out11), dim=-2),
        ),
        dim=-3,
    )


def _fill_ident_diag(slot: Tensor) -> None:
    slot.fill_(1.0)


def _fill_ident_block2(slot: Tensor) -> None:
    slot.zero_()
    slot[..., 0, 0, :] = 1
    slot[..., 1, 1, :] = 1


def _fill_ident_dense(slot: Tensor) -> None:
    slot.zero_()
    slot.diagonal(dim1=-2, dim2=-1).fill_(1)
