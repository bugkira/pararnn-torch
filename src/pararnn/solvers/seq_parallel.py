"""Sequence-parallel Newton scan: tile locally, then apply a carry.

Same monoid as ``scan_diag``. Two CUDA streams on one device are virtual
ranks. ``scan_diag_context_parallel`` splits time across a process group
(NCCL / gloo): each rank scans its tile, AllGathers the tile monoid
``(P_end, δ_end)``, and applies the exclusive prefix carry. Communication
is ``O(B d)`` per rank, independent of local ``T``.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch import Tensor

from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.solvers.scan import scan_diag

log = logging.getLogger(__name__)


def scan_diag_two_ranks(
    jac: Tensor,
    residual: Tensor,
    *,
    streams: tuple[torch.cuda.Stream, torch.cuda.Stream] | None = None,
    backend: str = "eager",
) -> Tensor:
    """Inclusive diag scan by splitting time in half, then a carry.

    Rank 1's local scan runs concurrently with rank 0. The carry apply waits.
    ``streams``: virtual ranks on one GPU. ``None`` runs both tiles in one
    stream (correctness path / CPU).
    """
    _check_diag(jac, residual)
    time = residual.shape[1]
    if time < 2:
        return scan_diag(jac, residual, backend=backend)
    mid = time // 2
    j0, r0 = jac[:, :mid], residual[:, :mid]
    j1, r1 = jac[:, mid:], residual[:, mid:]
    if streams is None or residual.device.type != "cuda":
        left = scan_diag(j0, r0, backend=backend)
        right_local = scan_diag(j1, r1, backend=backend)
        prefix = _diag_prefix_products(j1, backend=backend)
        right = _compose_diag_carry(right_local, prefix, left[:, -1])
        return torch.cat((left, right), dim=1)

    s0, s1 = streams
    left = residual.new_empty(r0.shape)
    right_local = residual.new_empty(r1.shape)
    prefix = residual.new_empty(r1.shape)
    with torch.cuda.stream(s0):
        left.copy_(scan_diag(j0, r0, backend=backend))
    with torch.cuda.stream(s1):
        right_local.copy_(scan_diag(j1, r1, backend=backend))
        prefix.copy_(_diag_prefix_products(j1, backend=backend))
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


def _scan_diag_from_carry(
    jac: Tensor, residual: Tensor, carry: Tensor, *, backend: str = "eager"
) -> Tensor:
    """Scan a tile whose ``δ`` before the first step is ``carry``.

    Dummy identity step: ``δ = 1·0 + carry``, then the real tile.
    """
    ones = jac.new_ones(jac.shape[0], 1, jac.shape[-1])
    jac_pad = torch.cat((ones, jac), dim=1)
    res_pad = torch.cat((carry.unsqueeze(1), residual), dim=1)
    return scan_diag(jac_pad, res_pad, backend=backend)[:, 1:]


def _diag_prefix_products(jac: Tensor, *, backend: str = "eager") -> Tensor:
    """Inclusive products ``P_t = J_t ⋯ J_0`` (elementwise) of a tile.

    Scan of residual ``(J_0, 0, …)``: ``δ_t = J_t δ_{t-1}`` with ``δ_0 = J_0``.
    """
    r_prod = torch.zeros_like(jac)
    r_prod[:, 0] = jac[:, 0]
    return scan_diag(jac, r_prod, backend=backend)


def _compose_diag_carry(local: Tensor, prefix: Tensor, carry: Tensor) -> Tensor:
    """``δ_t = P_t carry + δ^{local}_t`` after a zero-init scan of the tile."""
    return local + prefix * carry.unsqueeze(1)


def _check_diag(jac: Tensor, residual: Tensor) -> None:
    if jac.shape != residual.shape:
        raise ValueError(f"jac {tuple(jac.shape)} != residual {tuple(residual.shape)}")
    if jac.dim() != 3:
        raise ValueError("scan_diag_two_ranks is diagonal (B, T, D) only")


def _resolve_scan_backend(x: Tensor, backend: str) -> str:
    if backend == "auto":
        return "triton" if is_fused_dtype_supported(x.dtype, x.device) else "eager"
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    return backend


def time_shard_bounds(time: int, rank: int, world_size: int) -> tuple[int, int]:
    """Contiguous ``[start, end)`` along time. The last rank takes the remainder."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} outside [0, {world_size})")
    if time < 0:
        raise ValueError(f"time must be >= 0, got {time}")
    base = time // world_size
    if rank < world_size - 1:
        start = rank * base
        return start, start + base
    return (world_size - 1) * base, time


def scan_diag_context_parallel(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Inclusive diag scan on a time tile owned by this rank.

    Rank 0 holds the left of the sequence. After a local scan, ranks
    AllGather ``(P_end, δ_end)`` of shape ``(B, d)`` and apply the exclusive
    prefix carry of that monoid. ``world_size==1`` (or no process group) is
    ``scan_diag``.

    When ``jac``/``residual`` require grad, the local scan is eager and the
    AllGather is ``torch.distributed.nn.functional.all_gather`` so the VJP
    of the carry reaches earlier ranks (Rank 1 → Rank 0). Triton has no
    scan autograd; it stays on the ``no_grad`` path.
    """
    _check_diag(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or residual.requires_grad)
    backend = "eager" if needs_grad else _resolve_scan_backend(jac, backend)
    if world == 1:
        return scan_diag(jac, residual, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, dim = residual.shape
    if local_t == 0:
        local = residual
        prefix = residual
        p_end = residual.new_ones(batch, dim)
        d_end = residual.new_zeros(batch, dim)
    else:
        local = scan_diag(jac, residual, backend=backend)
        prefix = _diag_prefix_products(jac, backend=backend)
        p_end = prefix[:, -1].contiguous()
        d_end = local[:, -1].contiguous()
    p_list, d_list = _all_gather_monoid(p_end, d_end, world, group, needs_grad)
    # Exclusive prefix of the gathered monoid. Rank 0's carry is 0; later
    # ranks apply (P_k, δ_k) from the left. `acc` after the loop uses every
    # gathered tensor so AllGather stays in the autograd graph on Rank 0
    # (otherwise that rank skips the backward collective and the carry VJP
    # never reaches jac/residual on the left tile).
    acc = p_end.new_zeros(batch, dim)
    carry = acc
    for k in range(world):
        if k == rank:
            carry = acc
        acc = p_list[k] * acc + d_list[k]
    out = _compose_diag_carry(local, prefix, carry)
    if needs_grad:
        out = out + acc.unsqueeze(1) * 0
    if not torch.compiler.is_compiling():
        log.debug(
            "scan_diag_context_parallel",
            extra={
                "rank": rank,
                "world": world,
                "local_t": int(jac.shape[1]),
                "batch": batch,
                "d": dim,
                "backend": backend,
                "needs_grad": needs_grad,
            },
        )
    return out


def reverse_scan_diag_context_parallel(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Eq. 2.6 reverse diag scan on a time tile. Carry travels Rank N-1 → 0.

    Last rank reverse-scans with zero right state, then sends ``μ`` at the
    tile head to the left neighbor. Rank 0 is the one that applies a
    nonzero carry.
    """
    from pararnn.solvers.scan import reverse_scan_diag

    _check_diag(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    needs_grad = torch.is_grad_enabled() and (jac.requires_grad or partial.requires_grad)
    backend = "eager" if needs_grad else _resolve_scan_backend(jac, backend)
    if world == 1:
        return reverse_scan_diag(jac, partial, backend=backend)

    rank = dist.get_rank(group=group)
    batch, local_t, dim = partial.shape
    payload_shape = (batch, 2, dim)
    if rank == world - 1:
        if local_t == 0:
            mu = partial
            payload = partial.new_zeros(payload_shape)
        else:
            mu = reverse_scan_diag(jac, partial, backend=backend)
            payload = torch.stack((mu[:, 0].contiguous(), jac[:, 0].contiguous()), dim=1)
        if rank > 0:
            dist.send(payload, dst=rank - 1, group=group)
        return mu

    incoming = torch.empty(payload_shape, device=partial.device, dtype=partial.dtype)
    dist.recv(incoming, src=rank + 1, group=group)
    carry_mu, j_next = incoming[:, 0], incoming[:, 1]
    if local_t == 0:
        mu = partial
        payload = incoming
    else:
        mu = _reverse_scan_diag_from_carry(jac, partial, carry_mu, j_next, backend=backend)
        payload = torch.stack((mu[:, 0].contiguous(), jac[:, 0].contiguous()), dim=1)
    if rank > 0:
        dist.send(payload, dst=rank - 1, group=group)
    return mu


def scan_diag_context_parallel_full(
    jac: Tensor,
    residual: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    """Full ``(B, T, d)`` on every rank: shard time, CP-scan, AllGather concat.

    This is the Newton hook: ``H`` stays replicated; the scan work is ``T/N``.
    """
    _check_diag(jac, residual)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return scan_diag(jac, residual, backend=tile_backend)
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    local = scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        residual[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


def reverse_scan_diag_context_parallel_full(
    jac: Tensor,
    partial: Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    backend: str = "auto",
) -> Tensor:
    from pararnn.solvers.scan import reverse_scan_diag

    _check_diag(jac, partial)
    world = dist.get_world_size(group=group) if dist.is_initialized() else 1
    if world == 1:
        tile_backend = _resolve_scan_backend(jac, backend)
        return reverse_scan_diag(jac, partial, backend=tile_backend)
    rank = dist.get_rank(group=group)
    time = int(jac.shape[1])
    start, end = time_shard_bounds(time, rank, world)
    local = reverse_scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        partial[:, start:end].contiguous(),
        group=group,
        backend=backend,
    )
    return all_gather_time_tiles(local, time, rank, world, group)


def all_gather_time_tiles(
    tile: Tensor,
    time: int,
    rank: int,
    world: int,
    group: dist.ProcessGroup | None,
) -> Tensor:
    """Rebuild ``(B, T, …)`` from this rank's ``(B, T_local, …)`` tile."""
    bounds = [time_shard_bounds(time, r, world) for r in range(world)]
    max_len = max(end - start for start, end in bounds)
    loc = int(tile.shape[1])
    if loc == max_len:
        pad = tile
    elif loc == 0:
        pad = tile.new_zeros(tile.shape[0], max_len, *tile.shape[2:])
    else:
        pad = torch.cat(
            (tile, tile.new_zeros(tile.shape[0], max_len - loc, *tile.shape[2:])),
            dim=1,
        )
    if torch.is_grad_enabled() and tile.requires_grad:
        from torch.distributed.nn.functional import all_gather

        gathered = all_gather(pad.contiguous(), group=group)
    else:
        gathered = [torch.empty_like(pad) for _ in range(world)]
        dist.all_gather(gathered, pad.contiguous(), group=group)
    parts = [gathered[r][:, : bounds[r][1] - bounds[r][0]] for r in range(world)]
    return torch.cat(parts, dim=1)


def _all_gather_monoid(
    p_end: Tensor,
    d_end: Tensor,
    world: int,
    group: dist.ProcessGroup | None,
    needs_grad: bool,
) -> tuple[tuple[Tensor, ...] | list[Tensor], tuple[Tensor, ...] | list[Tensor]]:
    if needs_grad:
        from torch.distributed.nn.functional import all_gather

        return all_gather(p_end, group=group), all_gather(d_end, group=group)
    p_list = [torch.empty_like(p_end) for _ in range(world)]
    d_list = [torch.empty_like(d_end) for _ in range(world)]
    dist.all_gather(p_list, p_end.contiguous(), group=group)
    dist.all_gather(d_list, d_end.contiguous(), group=group)
    return p_list, d_list


def _reverse_scan_diag_from_carry(
    jac: Tensor,
    partial: Tensor,
    carry_mu: Tensor,
    j_next: Tensor,
    *,
    backend: str = "eager",
) -> Tensor:
    """Reverse diag scan; ``carry_mu`` is ``μ`` at the first time of the next tile.

    ``j_next`` is ``J`` at that same time (needed for ``μ_{e-1} = J_e μ_e + ∂_{e-1}``).
    """
    j_rev = jac.new_zeros(jac.shape)
    if jac.shape[1] > 1:
        j_rev[:, 1:] = jac.flip(1)[:, :-1]
    j_rev[:, 0] = j_next
    scanned = _scan_diag_from_carry(j_rev, partial.flip(1), carry_mu, backend=backend)
    return scanned.flip(1)
