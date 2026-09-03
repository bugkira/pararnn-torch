"""Sequence-parallel Newton scan: tile locally, then apply a carry.

Same monoid as ``scan_diag``. Two streams on one GPU, not NCCL.
Two CUDA streams on one device are virtual ranks.

Rank 1's local scan runs concurrently with rank 0. The carry is an axpy
with the inclusive Jacobian prefix of the right tile.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pararnn.solvers.scan import scan_diag


def scan_diag_two_ranks(
    jac: Tensor,
    residual: Tensor,
    *,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None,
) -> Tensor:
    """Inclusive diag scan by splitting time in half, then a carry.

    Rank 1's local scan runs concurrently with rank 0. The carry apply waits.
    ``streams``: virtual ranks on one GPU. ``None`` runs both tiles in one
    stream (correctness path / CPU).
    """
    _check_diag(jac, residual)
    time = residual.shape[1]
    if time < 2:
        return scan_diag(jac, residual)
    mid = time // 2
    j0, r0 = jac[:, :mid], residual[:, :mid]
    j1, r1 = jac[:, mid:], residual[:, mid:]
    if streams is None or residual.device.type != "cuda":
        left = scan_diag(j0, r0)
        right_local = scan_diag(j1, r1)
        prefix = _diag_prefix_products(j1)
        right = _compose_diag_carry(right_local, prefix, left[:, -1])
        return torch.cat((left, right), dim=1)

    s0, s1 = streams
    left = residual.new_empty(r0.shape)
    right_local = residual.new_empty(r1.shape)
    prefix = residual.new_empty(r1.shape)
    with torch.cuda.stream(s0):
        left.copy_(scan_diag(j0, r0))
    with torch.cuda.stream(s1):
        right_local.copy_(scan_diag(j1, r1))
        prefix.copy_(_diag_prefix_products(j1))
    torch.cuda.current_stream().wait_stream(s0)
    torch.cuda.current_stream().wait_stream(s1)
    right = _compose_diag_carry(right_local, prefix, left[:, -1])
    return torch.cat((left, right), dim=1)


def sequential_prefix_two_ranks(jac: Tensor, residual: Tensor) -> Tensor:
    """Naive prefix split: rank 1 starts after rank 0's last state exists.

    Same numeric result as ``scan_diag``. The right tile is scanned from the
    carry (dummy identity step). That data dependence serializes the two
    CUDA streams.
    """
    _check_diag(jac, residual)
    time = residual.shape[1]
    if time < 2:
        return scan_diag(jac, residual)
    mid = time // 2
    left = scan_diag(jac[:, :mid], residual[:, :mid])
    right = _scan_diag_from_carry(jac[:, mid:], residual[:, mid:], left[:, -1])
    return torch.cat((left, right), dim=1)


def _scan_diag_from_carry(jac: Tensor, residual: Tensor, carry: Tensor) -> Tensor:
    """Scan a tile whose ``δ`` before the first step is ``carry``.

    Dummy identity step: ``δ = 1·0 + carry``, then the real tile.
    """
    ones = jac.new_ones(jac.shape[0], 1, jac.shape[-1])
    jac_pad = torch.cat((ones, jac), dim=1)
    res_pad = torch.cat((carry.unsqueeze(1), residual), dim=1)
    return scan_diag(jac_pad, res_pad)[:, 1:]


def _diag_prefix_products(jac: Tensor) -> Tensor:
    """Inclusive products ``P_t = J_t ⋯ J_0`` (elementwise) of a tile.

    Scan of residual ``(J_0, 0, …)``: ``δ_t = J_t δ_{t-1}`` with ``δ_0 = J_0``.
    """
    r_prod = torch.zeros_like(jac)
    r_prod[:, 0] = jac[:, 0]
    return scan_diag(jac, r_prod)


def _compose_diag_carry(local: Tensor, prefix: Tensor, carry: Tensor) -> Tensor:
    """``δ_t = P_t carry + δ^{local}_t`` after a zero-init scan of the tile."""
    return local + prefix * carry.unsqueeze(1)


def _check_diag(jac: Tensor, residual: Tensor) -> None:
    if jac.shape != residual.shape:
        raise ValueError(f"jac {tuple(jac.shape)} != residual {tuple(residual.shape)}")
    if jac.dim() != 3:
        raise ValueError("scan_diag_two_ranks is diagonal (B, T, D) only")
