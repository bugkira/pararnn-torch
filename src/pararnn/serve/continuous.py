"""Continuous-batch engine: one slot id → all ``ParaSLSTMBlock`` recurrent pools."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import Tensor

from pararnn.layers.para_slstm_block import ParaSLSTMBlock
from pararnn.layout import SLSTM_SLOTS
from pararnn.paged import PagedStatePool, SlotAllocator, _as_id_list, _copy_rows, paged_apply

log = logging.getLogger(__name__)


class BlockStackPool:
    """Shared-slot pool across a stack of ``ParaSLSTMBlock`` modules.

    Each block owns a one-layer ``ParaRNN``; every request keeps the **same**
    slot id in every layer (vLLM ``state_indices`` / ``block_table`` analogue).
    Prefill uses Newton or sequential ``paged_apply``; decode is ``T=1`` with
    Triton ``decode_step(..., block_table=)`` when CUDA allows.
    """

    def __init__(
        self,
        blocks: Sequence[ParaSLSTMBlock],
        capacity: int,
        *,
        host_capacity: int | None = None,
    ) -> None:
        if not blocks:
            raise ValueError("blocks must be non-empty")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.blocks = list(blocks)
        self.allocator = SlotAllocator(capacity)
        n_host = capacity if host_capacity is None else host_capacity
        self.host_allocator = SlotAllocator(
            n_host,
            oom="paged host OOM",
            unit="pages",
            item="host page",
        )
        self.pools: list[PagedStatePool] = [
            PagedStatePool(
                b.rnn,
                capacity,
                host_capacity=n_host,
                allocator=self.allocator,
                host_allocator=self.host_allocator,
            )
            for b in self.blocks
        ]
        self.device = self.pools[0].device
        self.dtype = self.pools[0].dtype

    @property
    def capacity(self) -> int:
        return self.allocator.capacity

    @property
    def host_capacity(self) -> int:
        return self.host_allocator.capacity

    @property
    def n_layers(self) -> int:
        return len(self.pools)

    def allocate(self, n: int) -> Tensor:
        """Allocate ``n`` fresh slots (zeroed on every layer)."""
        # Shared allocator: only ``pools[0]`` marks ids used; zero the rest.
        ids = self.pools[0].allocate(n)
        for pool in self.pools[1:]:
            pool._zero(pool.buffers, ids)
        return ids

    def free(self, slot_ids: Tensor | Sequence[int]) -> None:
        """Free slots and zero every layer buffer row."""
        id_list = _as_id_list(slot_ids)
        t = torch.tensor(id_list, dtype=torch.long, device=self.device)
        for pool in self.pools:
            pool._zero(pool.buffers, t)
        self.allocator.free(id_list)

    def offload(self, slot_ids: Tensor | Sequence[int]) -> Tensor:
        """Copy all layer slots to host; free GPU rows. Returns host page ids."""
        gpu_list = _as_id_list(slot_ids)
        n = len(gpu_list)
        self.allocator.require_used(gpu_list)
        host_list = self.host_allocator.allocate(n)
        gpu_ids = torch.tensor(gpu_list, dtype=torch.long, device=self.device)
        host_ids_cpu = torch.tensor(host_list, dtype=torch.long)
        non_blocking = self.device.type == "cuda"
        for pool in self.pools:
            _copy_rows(
                pool.buffers,
                pool.host_buffers,
                gpu_ids,
                host_ids_cpu,
                non_blocking=non_blocking,
            )
        if non_blocking:
            torch.cuda.synchronize(self.device)
        self.free(gpu_list)
        return torch.tensor(host_list, dtype=torch.long, device=self.device)

    def reload(self, host_ids: Tensor | Sequence[int]) -> Tensor:
        """Reload host pages onto (possibly new) GPU slots for every layer."""
        host_list = _as_id_list(host_ids)
        n = len(host_list)
        self.host_allocator.require_used(host_list)
        gpu_ids = self.allocate(n)
        host_ids_cpu = torch.tensor(host_list, dtype=torch.long)
        non_blocking = self.device.type == "cuda"
        for pool in self.pools:
            _copy_rows(
                pool.host_buffers,
                pool.buffers,
                host_ids_cpu,
                gpu_ids,
                non_blocking=non_blocking,
            )
        if non_blocking:
            torch.cuda.synchronize(self.device)
        self.host_allocator.free(host_list)
        ht = torch.tensor(host_list, dtype=torch.long)
        for pool in self.pools:
            pool._zero(pool.host_buffers, ht)
        return gpu_ids

    def bind_external(
        self,
        layer_states: Sequence[Tensor],
        *,
        mark_used: Sequence[int] | None = None,
    ) -> None:
        """Point pool buffers at external tensors (vLLM-bound ``kv_cache``).

        Each ``layer_states[i]`` must be ``(C, SLSTM_SLOTS, d_h)`` matching
        capacity. Ownership stays with the caller; we do not free them.
        """
        if len(layer_states) != self.n_layers:
            raise ValueError(
                f"expected {self.n_layers} layer state tensors, got {len(layer_states)}"
            )
        for i, (pool, buf) in enumerate(zip(self.pools, layer_states, strict=True)):
            if buf.dim() != 3 or buf.shape[0] != self.capacity:
                raise ValueError(
                    f"layer {i}: expected (C={self.capacity}, {SLSTM_SLOTS}, d), "
                    f"got {tuple(buf.shape)}"
                )
            if buf.shape[1] != SLSTM_SLOTS:
                raise ValueError(
                    f"layer {i}: expected SLSTM_SLOTS={SLSTM_SLOTS} on dim 1, "
                    f"got {buf.shape[1]}"
                )
            pool.buffers[0] = buf
        if mark_used is not None:
            self.allocator.adopt(mark_used)
        log.info(
            "block_stack_bind_external",
            extra={"n_layers": self.n_layers, "capacity": self.capacity},
        )

    @torch.no_grad()
    def forward_hidden(
        self,
        h: Tensor,
        slot_ids: Tensor,
        *,
        cu_seqlens: Tensor | None = None,
        solver: str = "sequential",
    ) -> Tensor:
        """Run the block stack; update pool slots in place. Returns hidden ``h``."""
        if h.dim() != 3:
            raise ValueError(f"h must be (B, T, d) or packed (1, N, d), got {tuple(h.shape)}")
        ids = slot_ids.to(device=h.device, dtype=torch.long)
        for block, pool in zip(self.blocks, self.pools, strict=True):
            x_n = block.norm_rnn(h)
            y = paged_apply(pool, ids, x_n, cu_seqlens=cu_seqlens, solver=solver)
            h = h + y
            h = h + block.mlp(block.norm_mlp(h))
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "block_stack_forward",
                extra={
                    "n_req": int(ids.numel()),
                    "N": int(h.shape[1]),
                    "packed": cu_seqlens is not None,
                    "solver": solver,
                    "n_used": self.allocator.n_used,
                },
            )
        return h


def state_shape(hidden_size: int) -> tuple[int, int]:
    """Per-layer recurrent page shape ``(SLSTM_SLOTS, d_h)``."""
    return (SLSTM_SLOTS, int(hidden_size))


def state_shapes_for_vllm(hidden_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mamba-shaped ``(conv, temporal)`` pair for ``get_mamba_state_shape_from_config``.

    Conv is a length-1 dummy (unused). Temporal is the sLSTM carry
    ``(SLSTM_SLOTS, d_h)`` that ``bind_external`` / ``decode_step`` consume.
    """
    return ((1,), state_shape(hidden_size))
