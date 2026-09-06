"""``MambaBase`` recurrent layer: vLLM cache pages → ``decode_step`` / scan."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels.decode import can_decode_step, decode_step, decode_wx
from pararnn.layout import SLSTM_HIDDEN, SLSTM_SLOTS
from pararnn.serve.continuous import state_shape, state_shapes_for_vllm
from pararnn.solvers.newton import NewtonConfig, newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)

try:
    from vllm.config import get_current_vllm_config
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateCopyFuncCalculator,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    _HAS_VLLM = True
except ImportError:  # pragma: no cover - optional
    _HAS_VLLM = False
    MambaBase = nn.Module  # type: ignore[misc, assignment]
    MambaAttentionBackendEnum = None  # type: ignore[assignment]
    MambaStateCopyFuncCalculator = None  # type: ignore[assignment]

    def get_current_vllm_config():  # type: ignore[misc]
        raise RuntimeError("vllm not installed")

    def get_forward_context():  # type: ignore[misc]
        raise RuntimeError("vllm not installed")


def _indices_1d(indices: Tensor | None, n: int) -> Tensor:
    if indices is None:
        raise ValueError("missing state_indices for continuous batch")
    if indices.dim() == 1:
        out = indices
    elif indices.dim() == 2:
        # ``mamba_cache_mode != all``: (B, 1+spec); take the live slot column.
        out = indices[:, 0]
    else:
        raise ValueError(f"state_indices rank {indices.dim()} unsupported")
    if int(out.numel()) != n:
        raise ValueError(f"state_indices len {int(out.numel())} != batch {n}")
    return out.to(dtype=torch.long)


class ParaSLSTMRecurrentLayer(MambaBase):
    """Attention-free layer: sLSTM carry in ``kv_cache[1]`` as ``(C, 4, d)``.

    Declares ``mamba_type=MAMBA1`` so the engine builds
    ``Mamba1AttentionMetadata`` (``state_indices_*``, ``query_start_loc_*``).
    Conv page ``kv_cache[0]`` is unused (shape ``(1,)``). Temporal page holds
    the four sLSTM slots; updates go through ``decode_step`` / sequential
    scan with ``block_table``.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        newton: NewtonConfig | None = None,
        model_config: Any = None,
        cache_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cell_kw: dict = {"mix": mix, "max_recurrent_norm": max_recurrent_norm}
        if mix == "head":
            if n_heads is None:
                raise ValueError("mix='head' requires n_heads")
            cell_kw["n_heads"] = n_heads
        self.hidden_size = int(hidden_size)
        self.cell = ParaSLSTM(hidden_size, hidden_size, **cell_kw)
        self.newton = newton or NewtonConfig(max_iters=3)
        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix
        # Worker replaces these via ``bind_kv_cache``.
        self.kv_cache: tuple[Tensor, ...] = (torch.tensor([]), torch.tensor([]))

        if _HAS_VLLM:
            try:
                cfg = get_current_vllm_config()
                if prefix in cfg.compilation_config.static_forward_context:
                    raise ValueError(f"Duplicate layer name: {prefix}")
                cfg.compilation_config.static_forward_context[prefix] = self
            except Exception as exc:  # pragma: no cover - outside engine
                log.debug("skip static_forward_context register: %s", exc)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        dtype = torch.float32
        if self.model_config is not None:
            dtype = getattr(self.model_config, "dtype", dtype)
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype, torch.float32)
        return (dtype, dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return state_shapes_for_vllm(self.hidden_size)

    @property
    def mamba_type(self):
        if not _HAS_VLLM:
            return "MAMBA1"
        return MambaAttentionBackendEnum.MAMBA1

    def forward(self, hidden_states: Tensor) -> Tensor:
        """``hidden_states`` is ``(num_tokens, d)`` from the v1 worker."""
        if hidden_states.dim() != 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"expected (N, {self.hidden_size}), got {tuple(hidden_states.shape)}")
        meta = self._attn_metadata()
        if meta is None or not self._cache_ready():
            return self._profile_forward(hidden_states)
        return self._engine_forward(hidden_states, meta)

    def _attn_metadata(self) -> Any | None:
        if not _HAS_VLLM:
            return None
        try:
            raw = get_forward_context().attn_metadata
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw.get(self.prefix)
        return raw

    def _cache_ready(self) -> bool:
        if not self.kv_cache or len(self.kv_cache) < 2:
            return False
        return int(self.kv_cache[1].numel()) > 0

    def _temporal(self) -> Tensor:
        """``(C, 4, d)`` sLSTM pool."""
        st = self.kv_cache[1]
        if st.dim() == 2 and st.shape[-1] == SLSTM_SLOTS * self.hidden_size:
            return st.view(st.shape[0], SLSTM_SLOTS, self.hidden_size)
        if st.dim() == 3 and st.shape[1] == SLSTM_SLOTS:
            return st
        raise ValueError(
            f"temporal cache expected (C, {SLSTM_SLOTS}, d) or flat, got {tuple(st.shape)}"
        )

    def _profile_forward(self, hidden_states: Tensor) -> Tensor:
        """No bound cache (memory profile): zero-state step, discard carry."""
        n, d = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        state = torch.zeros(n, SLSTM_SLOTS, d, device=device, dtype=dtype)
        if can_decode_step(self.cell, state):
            wx = decode_wx(self.cell, hidden_states)
            state = decode_step(self.cell, state, wx=wx, out=state)
        else:
            for i in range(n):
                state[i : i + 1] = self.cell.step(state[i : i + 1], hidden_states[i : i + 1])
        return state[:, SLSTM_HIDDEN, :]

    def _engine_forward(self, hidden_states: Tensor, meta: Any) -> Tensor:
        num_decode = int(getattr(meta, "num_decode_tokens", 0) or 0)
        num_prefill = int(getattr(meta, "num_prefill_tokens", 0) or 0)
        num_actual = num_decode + num_prefill
        h = hidden_states[:num_actual] if num_actual > 0 else hidden_states
        pool = self._temporal()
        pieces: list[Tensor] = []

        if num_decode > 0:
            h_d = h[:num_decode]
            idx_d = _indices_1d(getattr(meta, "state_indices_tensor_d", None), num_decode)
            pieces.append(self._decode_tokens(h_d, pool, idx_d))

        if num_prefill > 0:
            h_p = h[num_decode : num_decode + num_prefill]
            idx_p = getattr(meta, "state_indices_tensor_p", None)
            qsl = getattr(meta, "query_start_loc_p", None)
            if qsl is None:
                # Single contiguous prefill request spanning all prefill tokens.
                if idx_p is None:
                    idx = torch.zeros(1, dtype=torch.long, device=h.device)
                else:
                    idx = _indices_1d(idx_p, 1)
                qsl = torch.tensor([0, num_prefill], device=h.device, dtype=torch.int32)
            else:
                n_req = int(qsl.numel()) - 1
                idx = _indices_1d(idx_p, n_req)
            pieces.append(self._prefill_tokens(h_p, pool, idx, qsl))

        if not pieces:
            return self._profile_forward(hidden_states)
        out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        if num_actual > 0 and hidden_states.shape[0] > num_actual:
            pad = torch.zeros(
                hidden_states.shape[0] - num_actual,
                self.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            out = torch.cat([out, pad], dim=0)
        return out

    def _decode_tokens(self, h: Tensor, pool: Tensor, slot_ids: Tensor) -> Tensor:
        """One token per request; write pool rows via ``block_table``."""
        # (B, d) → treat as T=1 batch for decode_step.
        if can_decode_step(self.cell, pool):
            wx = decode_wx(self.cell, h)
            decode_step(self.cell, pool, wx=wx, block_table=slot_ids)
        else:
            for b in range(int(slot_ids.numel())):
                sid = int(slot_ids[b].item())
                pool[sid] = self.cell.step(pool[sid : sid + 1], h[b : b + 1])[0]
        return pool.index_select(0, slot_ids.to(dtype=torch.long))[:, SLSTM_HIDDEN, :]

    def _prefill_tokens(
        self, h: Tensor, pool: Tensor, slot_ids: Tensor, cu_seqlens: Tensor
    ) -> Tensor:
        """Packed ``(N, d)`` → ``(1, N, d)`` scan; scatter last state to pool."""
        x = h.unsqueeze(0)
        cs = cu_seqlens.to(device=h.device)
        ids = slot_ids.to(device=h.device, dtype=torch.long)
        # Gather h0, run sequential (CPU/portable) or newton, scatter.
        h0 = pool.index_select(0, ids)
        if (
            not torch.is_grad_enabled()
            and self.newton.scan_backend in ("auto", "fused", "eager")
            and h.is_cuda
            and self.newton.scan_backend != "eager"
        ):
            try:
                traj = newton_apply(
                    self.cell, x, self.newton, h0=pool, cu_seqlens=cs, block_table=ids
                )
            except Exception:
                traj = sequential_apply(self.cell, x, h0, cu_seqlens=cs)
                self._scatter_last(pool, ids, traj, cs)
                return traj[0, :, SLSTM_HIDDEN, :]
            self._scatter_last(pool, ids, traj, cs)
            return traj[0, :, SLSTM_HIDDEN, :]

        traj = sequential_apply(self.cell, x, h0, cu_seqlens=cs)
        self._scatter_last(pool, ids, traj, cs)
        return traj[0, :, SLSTM_HIDDEN, :]

    @staticmethod
    def _scatter_last(pool: Tensor, ids: Tensor, traj: Tensor, cs: Tensor) -> None:
        ends = cs[1:].to(dtype=torch.long) - 1
        lasts = traj[0].index_select(0, ends)
        pool.index_copy_(0, ids, lasts)


class ParaSLSTMDecoderLayer(nn.Module):
    """Pre-norm residual: RMSNorm → recurrent mixer → RMSNorm → SwiGLU."""

    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float = 4.0,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        newton: NewtonConfig | None = None,
        eps: float = 1e-6,
        model_config: Any = None,
        cache_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from pararnn.layers.para_slstm_block import SwiGLU

        self.norm_rnn = nn.RMSNorm(hidden_size, eps=eps)
        self.mixer = ParaSLSTMRecurrentLayer(
            hidden_size,
            mix=mix,
            n_heads=n_heads,
            max_recurrent_norm=max_recurrent_norm,
            newton=newton,
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.mixer",
        )
        self.norm_mlp = nn.RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, hidden_states: Tensor) -> Tensor:
        h = hidden_states + self.mixer(self.norm_rnn(hidden_states))
        return h + self.mlp(self.norm_mlp(h))


def mamba1_copy_funcs():
    if MambaStateCopyFuncCalculator is None:
        return ()
    return MambaStateCopyFuncCalculator.mamba1_state_copy_func()


__all__ = [
    "ParaSLSTMDecoderLayer",
    "ParaSLSTMRecurrentLayer",
    "mamba1_copy_funcs",
    "state_shape",
]
