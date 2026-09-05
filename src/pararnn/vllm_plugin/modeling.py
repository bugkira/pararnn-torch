"""vLLM architecture: ``ParaSLSTMForCausalLM`` with ``MambaBase`` layers.

Each decoder layer registers a ``ParaSLSTMRecurrentLayer`` (``MambaBase``,
``mamba_type=MAMBA1``). The worker binds cache pages; the mixer reads
``Mamba1AttentionMetadata`` and updates sLSTM carries via ``decode_step`` /
packed scan. Offline ``slot_ids`` kwargs still route through the library
``ParaSLSTMForCausalLM.forward_continuous`` helper attached as ``.library``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
from torch import Tensor, nn

from pararnn.models.causal_lm import ParaSLSTMForCausalLM as LibraryParaSLSTMForCausalLM
from pararnn.models.config import ParaSLSTMConfig
from pararnn.serve.continuous import state_shapes_for_vllm
from pararnn.solvers.newton import LIBRARY_NEWTON_ITERS, NewtonConfig
from pararnn.vllm_plugin.layers import ParaSLSTMDecoderLayer, mamba1_copy_funcs

log = logging.getLogger(__name__)

_AF_BASES: tuple[type, ...] = ()
try:
    from vllm.model_executor.models.interfaces import (
        HasInnerState,
        IsAttentionFree,
        SupportsMambaPrefixCaching,
    )

    _AF_BASES = (HasInnerState, IsAttentionFree, SupportsMambaPrefixCaching)
except ImportError:  # pragma: no cover
    try:
        from vllm.model_executor.models.interfaces import HasInnerState, IsAttentionFree

        _AF_BASES = (HasInnerState, IsAttentionFree)
    except ImportError:
        pass


def _config_from_hf(hf_config: Any) -> ParaSLSTMConfig:
    if isinstance(hf_config, ParaSLSTMConfig):
        return hf_config
    data = dict(getattr(hf_config, "__dict__", {}))
    if hasattr(hf_config, "to_dict"):
        with contextlib.suppress(Exception):
            data = hf_config.to_dict()
    if "text_config" in data and isinstance(data["text_config"], dict):
        data = {**data, **data["text_config"]}
    return ParaSLSTMConfig.from_dict(data)


def _newton(cfg: ParaSLSTMConfig) -> NewtonConfig:
    return NewtonConfig(
        max_iters=int(cfg.newton_iters or LIBRARY_NEWTON_ITERS),
        scan_backend=str(cfg.scan_backend),
        picard_iters=cfg.picard_iters,
    )


def _max_num_seqs(vllm_config: Any) -> int:
    sched = getattr(vllm_config, "scheduler_config", None)
    n = getattr(sched, "max_num_seqs", None) if sched is not None else None
    return max(1, int(n if n is not None else 8))


class VLLMParaSLSTMForCausalLM(nn.Module, *_AF_BASES):
    """Registered as ``ParaSLSTMForCausalLM`` for ``ModelRegistry``."""

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        hf = getattr(vllm_config, "model_config", None)
        hf_config = getattr(hf, "hf_config", None) if hf is not None else None
        if hf_config is None:
            raise TypeError(
                "VLLMParaSLSTMForCausalLM requires vllm_config.model_config.hf_config"
            )
        self.vllm_config = vllm_config
        self.config = _config_from_hf(hf_config)
        model_config = getattr(vllm_config, "model_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        d = int(self.config.hidden_size)
        ncfg = _newton(self.config)
        root = f"{prefix}.model" if prefix else "model"

        self.embed = nn.Embedding(int(self.config.vocab_size), d)
        self.layers = nn.ModuleList(
            [
                ParaSLSTMDecoderLayer(
                    d,
                    mlp_ratio=float(self.config.mlp_ratio),
                    mix=str(self.config.mix),
                    n_heads=self.config.n_heads,
                    max_recurrent_norm=self.config.max_recurrent_norm,
                    newton=ncfg,
                    eps=float(self.config.rms_norm_eps),
                    model_config=model_config,
                    cache_config=cache_config,
                    prefix=f"{root}.layers.{i}",
                )
                for i in range(int(self.config.num_hidden_layers))
            ]
        )
        self.norm = nn.RMSNorm(d, eps=float(self.config.rms_norm_eps))
        self.lm_head = nn.Linear(d, int(self.config.vocab_size), bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

        # Offline continuous-batch helper (same weights via state_dict copy on demand).
        self.library = LibraryParaSLSTMForCausalLM(self.config)
        self.library.attach_pool(_max_num_seqs(vllm_config))
        self._sync_library_weights()

        log.info(
            "built VLLMParaSLSTMForCausalLM layers=%s d=%s vocab=%s",
            self.config.num_hidden_layers,
            d,
            self.config.vocab_size,
        )

    def _sync_library_weights(self) -> None:
        """Best-effort mirror of engine weights into the library CausalLM."""
        try:
            src = self.state_dict()
            dst = self.library.state_dict()
            mapped: dict[str, Tensor] = {}
            for k, v in src.items():
                if k.startswith("library."):
                    continue
                # embed / norm / lm_head
                if k in dst:
                    mapped[k] = v
                    continue
                # layers.i.mixer.cell.* → blocks.i.rnn.layers.0.*
                if k.startswith("layers.") and ".mixer.cell." in k:
                    # layers.0.mixer.cell.W_x.weight → blocks.0.rnn.layers.0.W_x.weight
                    rest = k.replace(".mixer.cell.", ".rnn.layers.0.")
                    alt = "blocks." + rest[len("layers.") :]
                    if alt in dst:
                        mapped[alt] = v
                if k.startswith("layers.") and ".norm_rnn." in k:
                    alt = "blocks." + k[len("layers.") :]
                    if alt in dst:
                        mapped[alt] = v
                if k.startswith("layers.") and (".norm_mlp." in k or ".mlp." in k):
                    alt = "blocks." + k[len("layers.") :]
                    if alt in dst:
                        mapped[alt] = v
            if mapped:
                self.library.load_state_dict(mapped, strict=False)
        except Exception as exc:
            log.debug("library weight sync skipped: %s", exc)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: Any
    ) -> tuple[torch.dtype, torch.dtype]:
        model_cfg = getattr(vllm_config, "model_config", None)
        dtype = getattr(model_cfg, "dtype", torch.float32) if model_cfg else torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)
        return (dtype, dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: Any
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        if hf is None:
            raise TypeError("get_mamba_state_shape_from_config needs hf_config")
        hidden = int(getattr(hf, "hidden_size", 0) or 0)
        if hidden < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden}")
        return state_shapes_for_vllm(hidden)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return mamba1_copy_funcs()

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embed(input_ids)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del intermediate_tensors, positions
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif input_ids is not None:
            # Offline continuous path (tests / custom runners).
            slot_ids = kwargs.get("slot_ids", kwargs.get("block_table"))
            if slot_ids is not None:
                self._sync_library_weights()
                cu = kwargs.get("cu_seqlens", kwargs.get("query_start_loc"))
                solver = str(kwargs.get("solver", "sequential"))
                ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
                if (
                    ids.dim() == 2
                    and ids.shape[0] == 1
                    and cu is None
                    and int(slot_ids.numel()) == int(ids.shape[1])
                ):
                    cu = torch.arange(
                        int(slot_ids.numel()) + 1,
                        device=ids.device,
                        dtype=torch.int32,
                    )
                logits = self.library.forward_continuous(
                    ids, slot_ids, cu_seqlens=cu, solver=solver
                )
                return logits
            hidden = self.embed(input_ids)
            if hidden.dim() == 3:
                # Dense (B, T, d) → flatten tokens for mixer profile path.
                b, t, d = hidden.shape
                hidden = hidden.reshape(b * t, d)
                for layer in self.layers:
                    hidden = layer(hidden)
                hidden = self.norm(hidden).view(b, t, d)
                return hidden
        else:
            raise ValueError("input_ids or inputs_embeds required")

        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(-1)
        # v1 worker: (num_tokens, d)
        if hidden.dim() != 2:
            raise ValueError(f"expected (N, d) hidden, got {tuple(hidden.shape)}")
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)

    def compute_logits(self, hidden_states: Tensor, sampling_metadata: Any = None) -> Tensor:
        del sampling_metadata
        if hidden_states.shape[-1] == int(self.config.vocab_size):
            return hidden_states
        return self.lm_head(hidden_states)

    def load_weights(self, weights) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            key = name
            if key not in params and key.startswith("model."):
                key = key[len("model.") :]
            if key not in params and key.startswith("backbone."):
                key = key[len("backbone.") :]
            if key not in params:
                continue
            params[key].data.copy_(tensor.to(dtype=params[key].dtype))
            loaded.add(name)
        self._sync_library_weights()
        return loaded

    def bind_kv_caches(
        self, layer_states: list[Tensor], *, mark_used: list[int] | None = None
    ) -> None:
        """Test helper: set temporal ``kv_cache`` on each mixer."""
        if len(layer_states) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} states, got {len(layer_states)}"
            )
        for layer, st in zip(self.layers, layer_states, strict=True):
            conv = torch.zeros(st.shape[0], 1, device=st.device, dtype=st.dtype)
            layer.mixer.kv_cache = (conv, st)
        if self.library.pool is not None:
            self.library.pool.bind_external(layer_states, mark_used=mark_used)
