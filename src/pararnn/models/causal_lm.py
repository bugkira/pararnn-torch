"""Standalone CausalLM: embed + ``ParaSLSTMBlock`` stack + tied LM head."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from pararnn.kernels.decode import can_decode_step, decode_step, decode_wx
from pararnn.layers.para_slstm_block import ParaSLSTMBlock
from pararnn.layout import SLSTM_HIDDEN, SLSTM_SLOTS
from pararnn.models.config import ParaSLSTMConfig
from pararnn.serve.continuous import BlockStackPool
from pararnn.solvers.newton import LIBRARY_NEWTON_ITERS, NewtonConfig

log = logging.getLogger(__name__)


def _newton_config(cfg: ParaSLSTMConfig) -> NewtonConfig:
    return NewtonConfig(
        max_iters=int(cfg.newton_iters or LIBRARY_NEWTON_ITERS),
        scan_backend=str(cfg.scan_backend),
        picard_iters=cfg.picard_iters,
    )


class ParaSLSTMForCausalLM(nn.Module):
    """Library CausalLM for export and as the vLLM architecture payload.

    Prefill / training: ``ParaSLSTMBlock`` (Newton while ``.train()``).
    Autoregressive decode: per-layer ``decode_step`` on full sLSTM carries.
    Continuous batch: ``attach_pool`` + ``forward_continuous`` (packed
    ``cu_seqlens`` + shared ``slot_ids`` / ``block_table``).
    """

    config_class = ParaSLSTMConfig

    def __init__(self, config: ParaSLSTMConfig) -> None:
        super().__init__()
        self.config = config
        d = int(config.hidden_size)
        self.embed = nn.Embedding(int(config.vocab_size), d)
        ncfg = _newton_config(config)
        self.blocks = nn.ModuleList(
            ParaSLSTMBlock(
                d,
                mlp_ratio=float(config.mlp_ratio),
                mix=str(config.mix),
                n_heads=config.n_heads,
                config=ncfg,
                solver="auto",
                max_recurrent_norm=config.max_recurrent_norm,
                eps=float(config.rms_norm_eps),
            )
            for _ in range(int(config.num_hidden_layers))
        )
        self.norm = nn.RMSNorm(d, eps=float(config.rms_norm_eps))
        self.lm_head = nn.Linear(d, int(config.vocab_size), bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self._pool: BlockStackPool | None = None

    def attach_pool(
        self, capacity: int, *, host_capacity: int | None = None
    ) -> BlockStackPool:
        """Create / replace the continuous-batch ``BlockStackPool``."""
        self._pool = BlockStackPool(
            list(self.blocks), capacity, host_capacity=host_capacity
        )
        log.info(
            "attached BlockStackPool capacity=%s layers=%s device=%s",
            capacity,
            len(self.blocks),
            self._pool.device,
        )
        return self._pool

    @property
    def pool(self) -> BlockStackPool | None:
        return self._pool

    def forward(self, input_ids: Tensor, *, positions: Tensor | None = None) -> Tensor:
        del positions
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, T), got {tuple(input_ids.shape)}")
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm(h))

    @torch.no_grad()
    def forward_continuous(
        self,
        input_ids: Tensor,
        slot_ids: Tensor,
        *,
        cu_seqlens: Tensor | None = None,
        solver: str = "sequential",
        pool: BlockStackPool | None = None,
    ) -> Tensor:
        """Prefill / decode through the paged block stack.

        Dense: ``input_ids`` is ``(B, T)`` with ``B == slot_ids.numel()``.
        Packed mix: ``input_ids`` is ``(1, N)`` and ``cu_seqlens`` has ``S+1``
        entries (``S == slot_ids.numel()``). Decode is ``T=1`` (or a length-1
        packed span). Pool slots are updated in place via ``block_table``.
        """
        eng = pool if pool is not None else self._pool
        if eng is None:
            raise RuntimeError("call attach_pool(capacity) before forward_continuous")
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be (B, T) or packed (1, N), got {tuple(input_ids.shape)}"
            )
        h = self.embed(input_ids)
        h = eng.forward_hidden(h, slot_ids, cu_seqlens=cu_seqlens, solver=solver)
        return self.lm_head(self.norm(h))

    def _cell_at(self, block: ParaSLSTMBlock) -> nn.Module:
        cells = block.rnn.layers if hasattr(block.rnn, "layers") else block.rnn.cells
        return cells[0]

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> Tensor:
        """Greedy (default) or sampled continuation with O(1) recurrent steps."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        self.eval()
        if self._pool is not None and int(input_ids.shape[0]) <= self._pool.capacity:
            return self._generate_paged(input_ids, max_new_tokens, temperature)
        return self._generate_carries(input_ids, max_new_tokens, temperature)

    def _generate_paged(
        self, input_ids: Tensor, max_new_tokens: int, temperature: float
    ) -> Tensor:
        assert self._pool is not None
        pool = self._pool
        batch = int(input_ids.shape[0])
        ids = pool.allocate(batch)
        try:
            out = input_ids.clone()
            logits = self.forward_continuous(out, ids, solver="sequential")
            logits_t = logits[:, -1]
            for _ in range(max_new_tokens):
                if temperature and temperature > 0:
                    probs = torch.softmax(logits_t / float(temperature), dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = logits_t.argmax(dim=-1, keepdim=True)
                out = torch.cat([out, next_id], dim=-1)
                logits_t = self.forward_continuous(next_id, ids, solver="sequential")[:, 0]
            return out
        finally:
            pool.free(ids)

    def _generate_carries(
        self, input_ids: Tensor, max_new_tokens: int, temperature: float
    ) -> Tensor:
        device = input_ids.device
        dtype = self.embed.weight.dtype
        out = input_ids.clone()
        batch = int(out.shape[0])
        carries = [
            torch.zeros(
                batch, SLSTM_SLOTS, int(self.config.hidden_size), device=device, dtype=dtype
            )
            for _ in self.blocks
        ]

        def _step_token(token_emb: Tensor) -> Tensor:
            h_t = token_emb
            for i, block in enumerate(self.blocks):
                cell = self._cell_at(block)
                x_n = block.norm_rnn(h_t)
                if can_decode_step(cell, carries[i]):
                    wx = decode_wx(cell, x_n)
                    carries[i] = decode_step(cell, carries[i], wx=wx, out=carries[i])
                else:
                    carries[i] = cell.step(carries[i], x_n)
                h_t = h_t + carries[i][..., SLSTM_HIDDEN, :]
                h_t = h_t + block.mlp(block.norm_mlp(h_t))
            return self.lm_head(self.norm(h_t))

        logits_t = None
        for t in range(int(out.shape[1])):
            logits_t = _step_token(self.embed(out[:, t]))
        assert logits_t is not None

        for _ in range(max_new_tokens):
            if temperature and temperature > 0:
                probs = torch.softmax(logits_t / float(temperature), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits_t.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=-1)
            logits_t = _step_token(self.embed(next_id[:, 0]))
        return out

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        (path / "pararnn_model_type.json").write_text(
            json.dumps({"model": "ParaSLSTMForCausalLM"}) + "\n"
        )
        log.info("saved ParaSLSTMForCausalLM to %s", path)

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location=None
    ) -> ParaSLSTMForCausalLM:
        path = Path(directory)
        cfg = ParaSLSTMConfig.from_pretrained(path)
        model = cls(cfg)
        state = torch.load(path / "pytorch_model.bin", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
