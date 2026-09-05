"""Paged O(1) state pool for continuous batching.

sLSTM / LSTM / GRU state does not grow with T. Prefill and decode still
share one GPU pool: a request owns a **slot** (one page). ``block_table``
is that slot id. Gather ``h0`` by slot, run Newton or ``step``, scatter
the last state back.

Triton ``decode_step`` indexes the pool through ``block_table``. Fused
Newton loads ``h0`` the same way (pool-shaped buffer + slot ids). A paused
request can leave the GPU: ``offload`` copies the slot to pinned host RAM
and frees it; ``reload`` allocates a (possibly new) GPU slot and copies
back. Sequential ``T>=1`` on CUDA writes resident slots in-kernel.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.kernels.decode import can_decode_step, decode_step
from pararnn.layers.para_rnn import (
    ParaRNN,
    _hidden_slot,
    _next_layer_input,
    _state_shape,
    _validate_input,
)
from pararnn.layout import validate_cu_seqlens
from pararnn.solvers.newton import newton_apply
from pararnn.solvers.newton.dispatch import _can_fuse
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)


class SlotAllocator:
    """CPU free-list of integer slots. Same id is used on every layer buffer."""

    def __init__(
        self,
        capacity: int,
        *,
        oom: str = "paged cache OOM",
        unit: str = "slots",
        item: str = "slot",
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._oom = oom
        self._unit = unit
        self._item = item
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
                f"{self._oom}: need {n} {self._unit}, free {len(self._free)}, "
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
                raise KeyError(f"{self._item} {ii} is not allocated")
            self._used.remove(ii)
            self._free.append(ii)
        log.debug(
            "paged_free",
            extra={"n": len(ids), "n_free": self.n_free, "n_used": self.n_used},
        )

    def require_used(self, ids: Sequence[int]) -> None:
        seen: set[int] = set()
        for i in ids:
            ii = int(i)
            if ii in seen:
                raise ValueError(f"duplicate {self._item} id {ii}")
            seen.add(ii)
            if ii not in self._used:
                raise KeyError(f"{self._item} {ii} is not allocated")

    def adopt(self, ids: Sequence[int]) -> None:
        """Mark existing ids as used (external cache bind). Ids must be free."""
        for i in ids:
            ii = int(i)
            if ii < 0 or ii >= self.capacity:
                raise ValueError(f"{self._item} {ii} outside capacity {self.capacity}")
            if ii in self._used:
                continue
            if ii not in self._free:
                raise KeyError(f"{self._item} {ii} is not free")
            self._free.remove(ii)
            self._used.add(ii)

class PagedStatePool:
    """One physical buffer per ``ParaRNN`` layer, shared slot ids.

    Buffer layout: ``(capacity, *state_tail)`` — GRU ``(C, d_h)``, LSTM
    ``(C, 2, d_h)``, sLSTM ``(C, 4, d_h)`` = (c, n, m, h).

    ``capacity`` is max GPU-resident requests (vLLM ``max_num_seqs`` analogue;
    the example uses 8 as a toy bound). ``host_capacity`` defaults to the same
    value: one full GPU batch can sit in pinned RAM while another occupies
    the device (vLLM ``swap_space`` oversubscribe; RNN state is O(d_h),
    so GiB sizing is unnecessary). Raise ``host_capacity`` if the pause queue
    is longer; ``offload`` raises ``paged host OOM`` when it is full.
    """

    def __init__(
        self,
        model: ParaRNN,
        capacity: int,
        *,
        host_capacity: int | None = None,
        allocator: SlotAllocator | None = None,
        host_allocator: SlotAllocator | None = None,
    ) -> None:
        if not model.layers:
            raise ValueError("model has no layers")
        param = next(model.parameters())
        self.device = param.device
        self.dtype = param.dtype
        # Shared allocators: one slot id across a stack of ``ParaRNN`` pools
        # (``BlockStackPool`` / continuous-batch CausalLM).
        if allocator is not None and allocator.capacity != capacity:
            raise ValueError(
                f"allocator.capacity={allocator.capacity} != capacity={capacity}"
            )
        self.allocator = allocator if allocator is not None else SlotAllocator(capacity)
        n_host = capacity if host_capacity is None else host_capacity
        if host_allocator is not None and host_allocator.capacity != n_host:
            raise ValueError(
                f"host_allocator.capacity={host_allocator.capacity} != host_capacity={n_host}"
            )
        self.host_allocator = (
            host_allocator
            if host_allocator is not None
            else SlotAllocator(
                n_host,
                oom="paged host OOM",
                unit="pages",
                item="host page",
            )
        )
        pin = self.device.type == "cuda"
        self.buffers: list[Tensor] = []
        self.host_buffers: list[Tensor] = []
        for cell in model.layers:
            tail = _state_shape(cell, 1)[1:]
            self.buffers.append(
                torch.zeros((capacity, *tail), device=self.device, dtype=self.dtype)
            )
            self.host_buffers.append(
                torch.zeros(
                    (n_host, *tail),
                    device="cpu",
                    dtype=self.dtype,
                    pin_memory=pin,
                )
            )
        self.model = model

    @property
    def capacity(self) -> int:
        return self.allocator.capacity

    @property
    def host_capacity(self) -> int:
        return self.host_allocator.capacity

    def allocate(self, n: int) -> Tensor:
        """New requests. Slots are zeroed (fresh ``h0``)."""
        ids = self.allocator.allocate(n)
        slot_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._zero(self.buffers, slot_ids)
        return slot_ids

    def free(self, slot_ids: Tensor | Sequence[int]) -> None:
        ids = _as_id_list(slot_ids)
        self.allocator.free(ids)
        self._zero(self.buffers, torch.tensor(ids, dtype=torch.long, device=self.device))

    def offload(self, slot_ids: Tensor | Sequence[int]) -> Tensor:
        """Copy GPU slots to pinned host pages and free the GPU slots.

        Returns host page ids. The GPU slot ids may be reused. Copy finishes
        before the GPU rows are zeroed.
        """
        gpu_list = _as_id_list(slot_ids)
        n = len(gpu_list)
        self.allocator.require_used(gpu_list)
        host_list = self.host_allocator.allocate(n)
        gpu_ids = torch.tensor(gpu_list, dtype=torch.long, device=self.device)
        host_ids_cpu = torch.tensor(host_list, dtype=torch.long)
        _copy_rows(
            self.buffers,
            self.host_buffers,
            gpu_ids,
            host_ids_cpu,
            non_blocking=self.device.type == "cuda",
        )
        self._sync()
        self.free(gpu_list)
        host_ids = torch.tensor(host_list, dtype=torch.long, device=self.device)
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "paged_offload",
                extra={
                    "n": n,
                    "n_gpu_used": self.allocator.n_used,
                    "n_host_used": self.host_allocator.n_used,
                    "device": str(self.device),
                },
            )
        return host_ids

    def reload(self, host_ids: Tensor | Sequence[int]) -> Tensor:
        """Allocate GPU slots, copy host pages back, free the host pages.

        Returns new GPU slot ids (they may differ from the ids at ``offload``).
        Host pages stay allocated if GPU allocate raises.
        """
        host_list = _as_id_list(host_ids)
        n = len(host_list)
        self.host_allocator.require_used(host_list)
        gpu_ids = self.allocate(n)
        host_ids_cpu = torch.tensor(host_list, dtype=torch.long)
        _copy_rows(
            self.host_buffers,
            self.buffers,
            host_ids_cpu,
            gpu_ids,
            non_blocking=self.device.type == "cuda",
        )
        self._sync()
        self.free_host(host_list)
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "paged_reload",
                extra={
                    "n": n,
                    "n_gpu_used": self.allocator.n_used,
                    "n_host_used": self.host_allocator.n_used,
                    "device": str(self.device),
                },
            )
        return gpu_ids

    def free_host(self, host_ids: Tensor | Sequence[int]) -> None:
        """Discard paused state (request cancelled while off GPU)."""
        ids = _as_id_list(host_ids)
        self.host_allocator.free(ids)
        self._zero(self.host_buffers, torch.tensor(ids, dtype=torch.long))

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

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _zero(self, buffers: Sequence[Tensor], slot_ids: Tensor) -> None:
        if slot_ids.numel() == 0:
            return
        dev = buffers[0].device
        ids = slot_ids.to(device=dev, dtype=torch.long)
        for buf in buffers:
            z_l = torch.zeros(ids.shape[0], *buf.shape[1:], device=dev, dtype=self.dtype)
            buf.index_copy_(0, ids, z_l)


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
    cs: Tensor | None = None
    if cu_seqlens is not None:
        cs = validate_cu_seqlens(cu_seqlens.to(device=x.device), x.shape[1])
        n_seq = int(cs.numel()) - 1
        if n_seq != n_req:
            raise ValueError(f"cu_seqlens S={n_seq} != slot_ids {n_req}")
        if x.shape[0] != 1:
            raise ValueError(f"packed x needs batch=1, got {x.shape[0]}")
    elif x.shape[0] != n_req:
        raise ValueError(f"x batch {x.shape[0]} != slot_ids {n_req}")

    if (
        solver == "sequential"
        and cu_seqlens is None
        and not torch.is_grad_enabled()
        and all(can_decode_step(c, x) for c in model.layers)
    ):
        return _paged_decode_seq(pool, ids, x)

    packed_cs = cs
    if (
        solver == "newton"
        and not torch.is_grad_enabled()
        and model.config.scan_backend in ("auto", "fused")
        and all(_can_fuse(c, x) for c in model.layers)
        and (packed_cs is None or all(isinstance(c, ParaGRU) for c in model.layers))
    ):
        return _paged_newton_block_table(pool, ids, x, packed_cs)

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


def _paged_decode_seq(pool: PagedStatePool, ids: Tensor, x: Tensor) -> Tensor:
    """T>=1 sequential: Triton decode writes pool slots via ``block_table``."""
    h_in = x
    y = x
    for i, cell in enumerate(pool.model.layers):
        lin = getattr(cell, "W_x", None)
        wx_all = lin(h_in) if lin is not None else None
        outs: list[Tensor] = []
        for t in range(int(h_in.shape[1])):
            wx_t = None if wx_all is None else wx_all[:, t]
            decode_step(cell, pool.buffers[i], x=h_in[:, t], wx=wx_t, block_table=ids)
            outs.append(_gather_hidden(pool.buffers[i], ids, cell))
        y = torch.stack(outs, dim=1)
        h_in = y
    if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
        log.debug(
            "paged_decode_seq",
            extra={
                "n_req": int(ids.numel()),
                "T": int(x.shape[1]),
                "n_used": pool.allocator.n_used,
            },
        )
    return y


def _paged_newton_block_table(
    pool: PagedStatePool,
    ids: Tensor,
    x: Tensor,
    cu_seqlens: Tensor | None,
) -> Tensor:
    """Fused Newton: ``h0`` is the pool, indexed by ``block_table``."""
    model = pool.model
    h = x
    lasts: list[Tensor] = []
    for i, cell in enumerate(model.layers):
        traj = newton_apply(
            cell,
            h,
            model.config,
            h0=pool.buffers[i],
            cu_seqlens=cu_seqlens,
            block_table=ids,
        )
        lasts.append(_last_states(traj, cu_seqlens))
        h = _next_layer_input(traj, cell) if i + 1 < len(model.layers) else traj
    pool.scatter(ids, lasts[0] if len(lasts) == 1 else tuple(lasts))
    slot = _hidden_slot(model.layers[-1])
    y = h if slot is None else h[:, :, slot, :]
    if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
        log.debug(
            "paged_newton_block_table",
            extra={
                "n_req": int(ids.numel()),
                "N": int(x.shape[1]),
                "packed": cu_seqlens is not None,
                "n_used": pool.allocator.n_used,
            },
        )
    return y


def _gather_hidden(buf: Tensor, ids: Tensor, cell: nn.Module) -> Tensor:
    st = buf.index_select(0, ids)
    slot = _hidden_slot(cell)
    if slot is None:
        return st
    return st[:, slot, :]


def _last_states(traj: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    if cu_seqlens is None:
        return traj[:, -1]
    cs = cu_seqlens.to(device=traj.device, dtype=torch.long)
    ends = cs[1:] - 1
    return traj[0].index_select(0, ends)


def _copy_rows(
    srcs: Sequence[Tensor],
    dsts: Sequence[Tensor],
    src_ids: Tensor,
    dst_ids: Tensor,
    *,
    non_blocking: bool,
) -> None:
    src_list = [int(i) for i in src_ids.detach().cpu().tolist()]
    dst_list = [int(i) for i in dst_ids.detach().cpu().tolist()]
    for src, dst in zip(srcs, dsts, strict=True):
        for s, d in zip(src_list, dst_list, strict=True):
            dst[d].copy_(src[s], non_blocking=non_blocking)


def _as_id_list(slot_ids: Tensor | Sequence[int]) -> list[int]:
    if isinstance(slot_ids, Tensor):
        return [int(i) for i in slot_ids.detach().cpu().tolist()]
    return [int(i) for i in slot_ids]
