"""Standalone CausalLM: embed + ``ParaSLSTMBlock`` stack + tied LM head."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.kernels.decode import can_decode_step, decode_step, decode_wx
from pararnn.layers.para_slstm_block import ParaSLSTMBlock
from pararnn.layout import SLSTM_HIDDEN, SLSTM_SLOTS
from pararnn.models.config import ParaSLSTMConfig
from pararnn.serve.continuous import BlockStackPool
from pararnn.solvers.newton import LIBRARY_NEWTON_ITERS, NewtonConfig

log = logging.getLogger(__name__)

_SAFETENSORS_NAME = "model.safetensors"
_BIN_NAME = "pytorch_model.bin"


def _newton_config(cfg: ParaSLSTMConfig) -> NewtonConfig:
    return NewtonConfig(
        max_iters=int(cfg.newton_iters or LIBRARY_NEWTON_ITERS),
        scan_backend=str(cfg.scan_backend),
        picard_iters=cfg.picard_iters,
    )


class ParaSLSTMForCausalLM(nn.Module):
    """Embed + ``ParaSLSTMBlock`` stack + LM head.

    Prefill / train: Newton via each block's ``ParaRNN``. Autoregressive
    decode: ``decode_step`` on full sLSTM carries. Continuous batch:
    ``attach_pool`` + ``forward_continuous``.

    Parameters
    ----------
    config : ParaSLSTMConfig

    Attributes
    ----------
    config : ParaSLSTMConfig
    embed : nn.Embedding
    blocks : nn.ModuleList of ParaSLSTMBlock
    norm : nn.RMSNorm
    lm_head : nn.Linear
        Tied to ``embed`` when ``config.tie_word_embeddings``.
    pool : BlockStackPool or None
        Set by ``attach_pool``.
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

    def attach_pool(self, capacity: int, *, host_capacity: int | None = None) -> BlockStackPool:
        """Create or replace the continuous-batch ``BlockStackPool``.

        Parameters
        ----------
        capacity : int
            Max GPU-resident request slots (shared across all layers).
        host_capacity : int, optional
            Pinned host pages for ``offload`` / ``reload``. Defaults to
            ``capacity``.

        Returns
        -------
        BlockStackPool
            The attached pool (also stored on ``self.pool``).
        """
        self._pool = BlockStackPool(list(self.blocks), capacity, host_capacity=host_capacity)
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

    def forward(
        self,
        input_ids: Tensor,
        *,
        positions: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Dense prefill / training forward through the block stack.

        Parameters
        ----------
        input_ids : Tensor of shape (batch, time)
            Token ids.
        positions : Tensor, optional
            Ignored; accepted for HF / vLLM call-site compatibility.
        labels : Tensor of shape (batch, time), optional
            Token targets. When set, returns ``(logits, loss)`` with shifted
            cross-entropy (``logits[:, :-1]`` vs ``labels[:, 1:]``); ``-100``
            is ignored. Without ``labels``, returns logits only.

        Returns
        -------
        Tensor of shape (batch, time, vocab_size)
            LM-head logits when ``labels`` is omitted.
        (logits, loss) : tuple[Tensor, Tensor]
            When ``labels`` is provided.

        Raises
        ------
        ValueError
            If ``input_ids`` is not rank-2, or ``labels`` shape mismatches.
        """
        del positions
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, T), got {tuple(input_ids.shape)}")
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h)
        logits = self.lm_head(self.norm(h))
        if labels is None:
            return logits
        if labels.shape != input_ids.shape:
            raise ValueError(
                f"labels must match input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(labels.shape)}"
            )
        # Next-token CE: predict t+1 from positions 0..t.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss

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
        """Prefill or decode through the paged block stack.

        Dense layout: ``input_ids`` is ``(batch, time)`` with
        ``batch == slot_ids.numel()``. Packed mix: ``input_ids`` is
        ``(1, N)`` and ``cu_seqlens`` has ``S+1`` entries with
        ``S == slot_ids.numel()``. Decode uses ``time=1`` (or a length-1
        packed span). Pool slots update in place via ``block_table``.

        Parameters
        ----------
        input_ids : Tensor of shape (batch, time) or (1, N)
            Token ids (dense or packed).
        slot_ids : Tensor of shape (S,)
            GPU pool slot ids for each request (``block_table``).
        cu_seqlens : Tensor of shape (S + 1,), optional
            Cumulative sequence lengths for packed ``(1, N)`` input.
        solver : {"newton", "sequential"}, default "sequential"
            Prefill / decode solver passed to ``paged_apply``.
        pool : BlockStackPool, optional
            Override for ``self.pool``.

        Returns
        -------
        Tensor of shape (batch, time, vocab_size) or (1, N, vocab_size)
            LM-head logits matching the input layout.

        Raises
        ------
        RuntimeError
            If no pool is attached and ``pool`` is omitted.
        ValueError
            If ``input_ids`` rank is invalid.
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
        """Autoregressive continuation with O(1) recurrent steps per token.

        Uses the attached ``BlockStackPool`` when batch fits ``capacity``;
        otherwise keeps per-layer carries on the device. ``temperature=0``
        is greedy argmax; positive temperature samples from softmax.

        Parameters
        ----------
        input_ids : Tensor of shape (batch, time)
            Prompt token ids.
        max_new_tokens : int, default 32
            Number of new tokens to append (must be ``>= 1``).
        temperature : float, default 0.0
            Sampling temperature; ``0`` selects greedy decoding.

        Returns
        -------
        Tensor of shape (batch, time + max_new_tokens)
            Prompt concatenated with generated ids.

        Raises
        ------
        ValueError
            If ``max_new_tokens < 1``.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        self.eval()
        if self._pool is not None and int(input_ids.shape[0]) <= self._pool.capacity:
            return self._generate_paged(input_ids, max_new_tokens, temperature)
        return self._generate_carries(input_ids, max_new_tokens, temperature)

    def _generate_paged(self, input_ids: Tensor, max_new_tokens: int, temperature: float) -> Tensor:
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
        """Write ``config.json``, ``model.safetensors``, and model-type marker.

        Also writes ``pytorch_model.bin`` so older loaders keep working.
        ``safetensors.torch.save_model`` keeps tied ``embed`` / ``lm_head`` weights.
        """
        from safetensors.torch import save_model

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        save_model(self, str(path / _SAFETENSORS_NAME))
        torch.save(self.state_dict(), path / _BIN_NAME)
        (path / "pararnn_model_type.json").write_text(
            json.dumps({"model": "ParaSLSTMForCausalLM"}) + "\n"
        )
        log.info("saved ParaSLSTMForCausalLM to %s", path)

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, map_location=None) -> ParaSLSTMForCausalLM:
        """Load from a directory written by ``save_pretrained``.

        Prefers ``model.safetensors``; falls back to ``pytorch_model.bin``.
        Weights load on CPU, then move when ``map_location`` is set.
        """
        path = Path(directory)
        cfg = ParaSLSTMConfig.from_pretrained(path)
        model = cls(cfg)
        st_path = path / _SAFETENSORS_NAME
        bin_path = path / _BIN_NAME
        if st_path.is_file():
            from safetensors.torch import load_model

            load_model(model, str(st_path), device="cpu")
        elif bin_path.is_file():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"no {_SAFETENSORS_NAME} or {_BIN_NAME} under {path}")
        if map_location is not None:
            model.to(map_location)
        return model
