"""Paged O(1) state pool for continuous batching.

sLSTM / LSTM / GRU state does not grow with T. Prefill and decode still
share one GPU pool: a request owns a **slot** (one page). ``block_table``
is that slot id. Gather ``h0`` by slot, run Newton or ``step``, scatter
the last state back.

Triton kernels still index ``batch * stride``. Indirect loads through
``block_table[req_id]`` and CPU↔GPU page swap are a later pass. This
module is the host allocator plus ``index_select`` / ``index_copy_``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import Tensor

from pararnn.layers.para_rnn import (
    ParaRNN,
    _hidden_slot,
    _next_layer_input,
    _state_shape,
    _validate_input,
)
from pararnn.layout import validate_cu_seqlens
from pararnn.solvers.newton import newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)


class SlotAllocator:
    """CPU free-list of integer slots. Same id is used on every layer buffer."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._free: list[int] = list(range(capacity - 1, -1, -1))
        self._used: set[int] = set()

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_used(self) -> int:
        return len(self._used)

    def allocate(self, n: int) -> list[int]:
        if n < 1:
            raise ValueError(f"allocate n must be >= 1, got {n}")
        if n > len(self._free):
            raise RuntimeError(
                f"paged cache OOM: need {n} slots, free {len(self._free)}, "
                f"capacity {self.capacity}"
            )
        ids = [self._free.pop() for _ in range(n)]
        self._used.update(ids)
        log.debug(
            "paged_allocate",
            extra={"n": n, "n_free": self.n_free, "n_used": self.n_used},
        )
        return ids

    def free(self, ids: Sequence[int]) -> None:
        for i in ids:
            ii = int(i)
            if ii not in self._used:
                raise KeyError(f"slot {ii} is not allocated")
            self._used.remove(ii)
            self._free.append(ii)
        log.debug(
            "paged_free",
            extra={"n": len(ids), "n_free": self.n_free, "n_used": self.n_used},
        )


class PagedStatePool:
    """One physical buffer per ``ParaRNN`` layer, shared slot ids.

    Buffer layout: ``(capacity, *state_tail)`` — GRU ``(C, d_h)``, LSTM
    ``(C, 2, d_h)``, sLSTM ``(C, 4, d_h)`` = (c, n, m, h).
    """

    def __init__(self, model: ParaRNN, capacity: int) -> None:
        if not model.layers:
            raise ValueError("model has no layers")
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        self.allocator = SlotAllocator(capacity)
        self.buffers: list[Tensor] = []
        for cell in model.layers:
            tail = _state_shape(cell, 1)[1:]
            self.buffers.append(
                torch.zeros((capacity, *tail), device=self.device, dtype=self.dtype)
            )
        self.model = model

    @property
    def capacity(self) -> int:
        return self.allocator.capacity

    def allocate(self, n: int) -> Tensor:
        """New requests. Slots are zeroed (fresh ``h0``)."""
        ids = self.allocator.allocate(n)
        slot_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._zero(slot_ids)
        return slot_ids

    def free(self, slot_ids: Tensor | Sequence[int]) -> None:
        ids = _as_id_list(slot_ids)
        self.allocator.free(ids)
        self._zero(torch.tensor(ids, dtype=torch.long, device=self.device))

    def gather(self, slot_ids: Tensor) -> Tensor | tuple[Tensor, ...]:
        """``h0`` for ``paged_apply``, one tensor per layer."""
        ids = slot_ids.to(device=self.device, dtype=torch.long)
        got = [buf.index_select(0, ids) for buf in self.buffers]
        return got[0] if len(got) == 1 else tuple(got)

    def scatter(self, slot_ids: Tensor, states: Tensor | Sequence[Tensor]) -> None:
        """Write last states back. ``states`` matches ``gather``."""
        ids = slot_ids.to(device=self.device, dtype=torch.long)
        packed = (states,) if isinstance(states, Tensor) else tuple(states)
        if len(packed) != len(self.buffers):
            raise ValueError(
                f"scatter expected {len(self.buffers)} layer states, got {len(packed)}"
            )
        for buf, st in zip(self.buffers, packed, strict=True):
            buf.index_copy_(0, ids, st)

    def _zero(self, slot_ids: Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        for buf in self.buffers:
            z_l = torch.zeros(
                slot_ids.shape[0], *buf.shape[1:], device=self.device, dtype=self.dtype
            )
            buf.index_copy_(0, slot_ids, z_l)


def paged_apply(
    pool: PagedStatePool,
    slot_ids: Tensor,
    x: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
    solver: str = "newton",
) -> Tensor:
    """Prefill or decode through the pool.

    Dense: ``x`` is ``(B, T, d_in)`` with ``B == slot_ids.numel()``.
    Packed mix: ``x`` is ``(1, N, d_in)`` and ``cu_seqlens`` has ``S+1``
    entries, ``S == slot_ids.numel()``. Decode is ``T=1`` (or a length-1
    packed span). Last state of each request is scattered back.
    """
    if solver not in ("newton", "sequential"):
        raise ValueError(f"solver must be 'newton' or 'sequential', got {solver!r}")
    model = pool.model
    if not model.batch_first:
        raise ValueError("paged_apply needs batch_first=True")
    _validate_input(x, model.layers[0], batch_first=True)
    ids = slot_ids.to(device=x.device, dtype=torch.long)
    n_req = int(ids.numel())
    if cu_seqlens is not None:
        cs = validate_cu_seqlens(cu_seqlens.to(device=x.device), x.shape[1])
        n_seq = int(cs.numel()) - 1
        if n_seq != n_req:
            raise ValueError(f"cu_seqlens S={n_seq} != slot_ids {n_req}")
        if x.shape[0] != 1:
            raise ValueError(f"packed x needs batch=1, got {x.shape[0]}")
    elif x.shape[0] != n_req:
        raise ValueError(f"x batch {x.shape[0]} != slot_ids {n_req}")

    h0s = pool.gather(ids)
    if isinstance(h0s, Tensor):
        h0_list: list[Tensor] = [h0s]
    else:
        h0_list = list(h0s)

    h = x
    lasts: list[Tensor] = []
    for i, cell in enumerate(model.layers):
        if solver == "newton":
            traj = newton_apply(cell, h, model.config, h0=h0_list[i], cu_seqlens=cu_seqlens)
        else:
            traj = sequential_apply(cell, h, h0_list[i], cu_seqlens=cu_seqlens)
        lasts.append(_last_states(traj, cu_seqlens))
        h = _next_layer_input(traj, cell) if i + 1 < len(model.layers) else traj

    pool.scatter(ids, lasts[0] if len(lasts) == 1 else tuple(lasts))
    slot = _hidden_slot(model.layers[-1])
    y = h if slot is None else h[:, :, slot, :]
    if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
        log.debug(
            "paged_apply",
            extra={
                "solver": solver,
                "n_req": n_req,
                "N": int(x.shape[1]),
                "packed": cu_seqlens is not None,
                "n_used": pool.allocator.n_used,
            },
        )
    return y


def _last_states(traj: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    if cu_seqlens is None:
        return traj[:, -1]
    cs = cu_seqlens.to(device=traj.device, dtype=torch.long)
    ends = cs[1:] - 1
    return traj[0].index_select(0, ends)


def _as_id_list(slot_ids: Tensor | Sequence[int]) -> list[int]:
    if isinstance(slot_ids, Tensor):
        return [int(i) for i in slot_ids.detach().cpu().tolist()]
    return [int(i) for i in slot_ids]
