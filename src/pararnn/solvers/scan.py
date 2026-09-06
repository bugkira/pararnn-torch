"""Parallel prefix scan for the Newton linear system (eq. 2.4).

Work-efficient Blelloch scan on the monoid
``(J_r, r_r) ⊕ (J_l, r_l) = (J_r J_l, J_r r_l + r_r)``.
Pad time to the next power of two with the identity ``(I, 0)``.
Indices: ``pararnn.layout`` (0-based ``t``).

Ragged: ``cu_seqlens`` packs sequences into ``(1, N, …)``. Combine is
segmented — a head flag drops the left carry (CUB-style). Same monoid.
Eager: log-depth Hillis–Steele over packed ``N``. Triton: ``cu_seqlens``
starts compared to ``offs_t`` in the tile (``J=0`` at heads), then PCR.
Fused ParaGRU does the same heads in-kernel.

2×2 / 4×4: elementwise mul. Reverse scan is paper eq. 2.6
(Jacobian transpose, unroll backwards).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from pararnn.layout import segment_start_flags, validate_cu_seqlens

_Compose = Callable[[Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]
_FillIdent = Callable[[Tensor], None]


def scan_diag(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Solve ``δ_t = jac_t * δ_{t-1} + residual_t`` with ``δ_{<0} = 0``.

    ``jac`` and ``residual``: (batch, time, d). ``jac`` is the diagonal of J.
    ``backend``: ``eager`` (default), ``triton`` (CUDA float16/float32), or
    ``context_parallel`` (shard ``T`` across an initialized process group,
    AllGather the full ``δ``). ``cu_seqlens`` packs ragged time; carry resets
    at heads. ``context_parallel`` does not take ``cu_seqlens``.
    """
    if backend not in ("eager", "triton", "context_parallel"):
        raise ValueError(f"unknown scan backend {backend!r}")
    if backend == "context_parallel":
        if cu_seqlens is not None:
            raise ValueError("scan_backend='context_parallel' cannot combine with cu_seqlens")
        from pararnn.solvers.seq_parallel import scan_diag_context_parallel_full

        return scan_diag_context_parallel_full(jac, residual, backend="auto")
    if backend == "triton":
        from pararnn.kernels import scan_diag_triton

        return scan_diag_triton(jac, residual, cu_seqlens=cu_seqlens)
    return _scan_acc(jac, residual, _compose_diag, _fill_ident_diag, cu_seqlens=cu_seqlens)


def scan_block2(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Same recurrence with 2×2 blocks per feature.

    ``jac``: (batch, time, 2, 2, d) with ``[..., out, in, d]``.
    ``residual`` / result: (batch, time, 2, d).
    """
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    if backend == "triton":
        from pararnn.kernels import scan_block2_triton

        return scan_block2_triton(_jac_drop_left(jac, cu_seqlens), residual)
    return _scan_acc(jac, residual, _compose_block2, _fill_ident_block2, cu_seqlens=cu_seqlens)


def scan_block4(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Same recurrence with 4x4 blocks per feature (sLSTM channelwise).

    ``jac``: (batch, time, 4, 4, d) with ``[..., out, in, d]``.
    ``residual`` / result: (batch, time, 4, d).
    """
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    if backend == "triton":
        from pararnn.kernels import scan_block4_triton

        return scan_block4_triton(_jac_drop_left(jac, cu_seqlens), residual)
    return _scan_acc(jac, residual, _compose_block4, _fill_ident_block4, cu_seqlens=cu_seqlens)


def scan_dense(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Exact Newton scan for a full ``d_h × d_h`` Jacobian (DEER).

    ``jac``: (batch, time, d, d) with ``[..., out, in]``. ``residual``: (batch, time, d).
    Compose is ``bmm`` — ``O(T d^3)`` after the log-depth scan.
    ``backend='triton'``: CUDA tiled row-wise Triton inclusive scan (any
    ``d``; fp32 algebra); ragged ``cu_seqlens`` stays on the eager segmented path.
    """
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    if backend == "triton":
        from pararnn.kernels import scan_dense_triton

        return scan_dense_triton(jac, residual, cu_seqlens=cu_seqlens)
    return _scan_acc(jac, residual, _compose_dense, _fill_ident_dense, cu_seqlens=cu_seqlens)


def reverse_scan_diag(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Total adjoint ``∇_{h_t} L`` from direct ``∂_{h_t} L`` (eq. 2.6, diagonal).

    ``∇_{h_{t-1}} L = J_t ∇_{h_t} L + ∂_{h_{t-1}} L``, ``∇_{h_{T-1}} L = ∂_{h_{T-1}} L``.
    Diagonal ``J`` is symmetric. Reverse scan starts at t=0 (eq. 2.6).
    ``context_parallel`` shards ``T`` and sends the adjoint carry Rank N-1 → 0.
    """
    if backend == "context_parallel":
        if cu_seqlens is not None:
            raise ValueError("scan_backend='context_parallel' cannot combine with cu_seqlens")
        from pararnn.solvers.seq_parallel import reverse_scan_diag_context_parallel_full

        return reverse_scan_diag_context_parallel_full(jac, partial, backend="auto")
    j_rev = jac.new_zeros(jac.shape)
    j_rev[:, 1:] = jac.flip(1)[:, :-1]
    cs_rev = _reverse_cu_seqlens(cu_seqlens, jac.shape[1]) if cu_seqlens is not None else None
    return scan_diag(j_rev, partial.flip(1), backend=backend, cu_seqlens=cs_rev).flip(1)


def reverse_scan_block2(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Eq. 2.6 with 2×2 blocks: uses ``J^T`` (swap ``out``/``in``)."""
    j_t = jac.transpose(-3, -2)
    j_rev = j_t.new_zeros(j_t.shape)
    j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    cs_rev = _reverse_cu_seqlens(cu_seqlens, jac.shape[1]) if cu_seqlens is not None else None
    return scan_block2(j_rev, partial.flip(1), backend=backend, cu_seqlens=cs_rev).flip(1)


def reverse_scan_block4(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Eq. 2.6 with 4×4 blocks: uses ``J^T`` (swap ``out``/``in``)."""
    j_t = jac.transpose(-3, -2)
    j_rev = j_t.new_zeros(j_t.shape)
    j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    cs_rev = _reverse_cu_seqlens(cu_seqlens, jac.shape[1]) if cu_seqlens is not None else None
    return scan_block4(j_rev, partial.flip(1), backend=backend, cu_seqlens=cs_rev).flip(1)


def reverse_scan_dense(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Eq. 2.6 with a full matrix: uses ``J^T``.

    ``backend='triton'``: tiled reverse walk on CUDA (no flip + forward scan).
    Ragged ``cu_seqlens`` stays on the eager flip path.
    """
    if backend not in ("eager", "triton"):
        raise ValueError(f"unknown scan backend {backend!r}")
    if backend == "triton" and cu_seqlens is None:
        from pararnn.kernels import reverse_scan_dense_triton

        return reverse_scan_dense_triton(jac, partial, cu_seqlens=None)
    j_t = jac.transpose(-1, -2)
    j_rev = j_t.new_zeros(j_t.shape)
    j_rev[:, 1:] = j_t.flip(1)[:, :-1]
    cs_rev = _reverse_cu_seqlens(cu_seqlens, jac.shape[1]) if cu_seqlens is not None else None
    return scan_dense(j_rev, partial.flip(1), backend=backend, cu_seqlens=cs_rev).flip(1)


def _jac_drop_left(jac: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    """Head flags: ``δ_t = r_t`` by zeroing ``J_t`` (left annihilator).

    The Triton PCR scan is rectangular. A segment start with ``J=0`` drops
    the incoming carry; later tiles still compose through ``r``.
    """
    if cu_seqlens is None:
        return jac
    flags = segment_start_flags(cu_seqlens, jac.shape[1], batch=jac.shape[0]).to(device=jac.device)
    extra = (1,) * (jac.dim() - 2)
    return jac.masked_fill(flags.view(*flags.shape, *extra), 0)


def _reverse_cu_seqlens(cu_seqlens: Tensor, time: int) -> Tensor:
    """Head flags of the time-reversed packed stream (eq. 2.6)."""
    cs = validate_cu_seqlens(cu_seqlens, time)
    lengths = cs[1:] - cs[:-1]
    return torch.cat(
        (cs.new_zeros(1), torch.cumsum(lengths.flip(0), dim=0)),
        dim=0,
    )


def _scan_acc(
    jac: Tensor,
    residual: Tensor,
    compose: _Compose,
    fill_ident: _FillIdent,
    *,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Blelloch in fp32 when DRAM is fp16/bf16 (Newton accumulators)."""
    flags = None
    if cu_seqlens is not None:
        flags = segment_start_flags(cu_seqlens, jac.shape[1], batch=jac.shape[0]).to(
            device=jac.device
        )
    if jac.dtype in (torch.float16, torch.bfloat16):
        out = _inclusive_scan(jac.float(), residual.float(), compose, fill_ident, flags=flags)
        return out.to(dtype=jac.dtype)
    return _inclusive_scan(jac, residual, compose, fill_ident, flags=flags)


def _inclusive_scan(
    jac: Tensor,
    residual: Tensor,
    compose: _Compose,
    fill_ident: _FillIdent,
    *,
    flags: Tensor | None = None,
) -> Tensor:
    if flags is not None:
        return _hillis_steele_seg_inclusive(jac, residual, compose, flags)
    return _blelloch_inclusive(jac, residual, compose, fill_ident)


def _hillis_steele_seg_inclusive(
    jac: Tensor,
    residual: Tensor,
    compose: _Compose,
    flags: Tensor,
) -> Tensor:
    """Log-depth inclusive scan; a head flag drops the left operand.

    The r-component of the inclusive ``(J, r)`` prefix is ``δ_t``.

    Work is ``O(T log T)``: each doubling composes ~``T`` sites and clones
    ``(J, r, flags)``. Rectangular Blelloch composes ``O(T)`` tree sites.
    ``where`` on the head flag is length-``T`` every round (head count
    does not change the kernel shape).
    """
    time = residual.shape[1]
    if time <= 1:
        return residual.clone()
    j = jac.clone()
    r = residual.clone()
    f = flags.clone()
    step = 1
    while step < time:
        j_l, r_l, f_l = j[:, :-step], r[:, :-step], f[:, :-step]
        j_r, r_r, f_r = j[:, step:], r[:, step:], f[:, step:]
        j_c, r_c, f_c = _compose_seg(compose, j_r, r_r, f_r, j_l, r_l, f_l)
        j = j.clone()
        r = r.clone()
        f = f.clone()
        j[:, step:] = j_c
        r[:, step:] = r_c
        f[:, step:] = f_c
        step *= 2
    return r


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
    prefix = r[:, :time]
    if jac.dim() == residual.dim():
        return jac * prefix + residual
    if _is_dense(jac, residual):
        return (jac @ prefix.unsqueeze(-1)).squeeze(-1) + residual
    if jac.shape[-3] == 4:
        return _mv4(jac, prefix) + residual
    return _mv2(jac, prefix) + residual


def _flag_as(flag: Tensor, ref: Tensor) -> Tensor:
    extra = (1,) * (ref.dim() - flag.dim())
    return flag.reshape(*flag.shape, *extra)


def _compose_seg(
    compose: _Compose,
    j_r: Tensor,
    r_r: Tensor,
    f_r: Tensor,
    j_l: Tensor,
    r_l: Tensor,
    f_l: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Later ⊕ earlier; a head flag on the later node drops the left carry."""
    j_c, r_c = compose(j_r, r_r, j_l, r_l)
    take_r = _flag_as(f_r, j_r)
    take_rr = _flag_as(f_r, r_r)
    return (
        torch.where(take_r, j_r, j_c),
        torch.where(take_rr, r_r, r_c),
        f_r | f_l,
    )


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


def _compose_diag(j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor) -> tuple[Tensor, Tensor]:
    return j_r * j_l, j_r * r_l + r_r


def _compose_block2(j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor) -> tuple[Tensor, Tensor]:
    return _mm2(j_r, j_l), _mv2(j_r, r_l) + r_r


def _compose_block4(j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor) -> tuple[Tensor, Tensor]:
    return _mm4(j_r, j_l), _mv4(j_r, r_l) + r_r


def _compose_dense(j_r: Tensor, r_r: Tensor, j_l: Tensor, r_l: Tensor) -> tuple[Tensor, Tensor]:
    return j_r @ j_l, (j_r @ r_l.unsqueeze(-1)).squeeze(-1) + r_r


def _is_dense(jac: Tensor, residual: Tensor) -> bool:
    """Full-matrix J is ``(B, T, d, d)``. ``(B, T, S, S, d)`` is a block Jacobian
    (sLSTM tests use ``d_h=4`` with 4×4 blocks).
    """
    return (
        jac.dim() == 4
        and residual.dim() == 3
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


def _mv4(jac: Tensor, vec: Tensor) -> Tensor:
    """``J @ v`` per feature: 4×4 elementwise, layout ``[..., out, in, d]``."""
    return (jac * vec.unsqueeze(-3)).sum(dim=-2)


def _mm4(j_right: Tensor, j_left: Tensor) -> Tensor:
    """``J_right @ J_left`` per feature. Reduce over the inner 4."""
    return (j_right.unsqueeze(-2) * j_left.unsqueeze(-4)).sum(dim=-3)


def _fill_ident_block4(slot: Tensor) -> None:
    slot.zero_()
    slot[..., 0, 0, :] = 1
    slot[..., 1, 1, :] = 1
    slot[..., 2, 2, :] = 1
    slot[..., 3, 3, :] = 1


def _fill_ident_dense(slot: Tensor) -> None:
    slot.zero_()
    slot.diagonal(dim1=-2, dim2=-1).fill_(1)
