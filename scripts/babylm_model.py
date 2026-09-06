"""Pre-norm causal LM: ParaSLSTM cell + SwiGLU, mixing set by ``cell_type``."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pararnn import NewtonConfig, NewtonStats, ParaRNN, ParaSLSTM

log = logging.getLogger("babylm_model")

CELL_TYPES = ("dense", "diag_seq", "diag_fused")


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mult: int) -> None:
        super().__init__()
        hidden = int(mult) * d_model
        self.up = nn.Linear(d_model, 2 * hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(u) * v)


class GELUMLP(nn.Module):
    def __init__(self, d_model: int, mult: int) -> None:
        super().__init__()
        hidden = int(mult) * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class BabyLMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        mix: str,
        n_heads: int | None,
        solver: str,
        newton_cfg: NewtonConfig,
        max_recurrent_norm: float | None,
        mlp_mult: int,
        mlp_act: str,
    ) -> None:
        super().__init__()
        kw: dict = {
            "mix": mix,
            "max_recurrent_norm": max_recurrent_norm,
        }
        if mix == "head":
            kw["n_heads"] = n_heads
        self.norm_rnn = nn.LayerNorm(d_model)
        self.rnn = ParaRNN(
            ParaSLSTM(d_model, d_model, **kw),
            config=newton_cfg,
            output_hidden=True,
            solver=solver,
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        if mlp_act == "gelu":
            self.mlp = GELUMLP(d_model, mlp_mult)
        elif mlp_act == "swiglu":
            self.mlp = SwiGLUMLP(d_model, mlp_mult)
        else:
            raise ValueError(f"mlp_act must be gelu or swiglu, got {mlp_act!r}")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.rnn(self.norm_rnn(x))
        return x + self.mlp(self.norm_mlp(x))


class BabyLMModel(nn.Module):
    """Tied-embedding causal LM. Residual stack lives here, not in ``ParaRNN``."""

    def __init__(self, spec: dict, *, cell_type: str) -> None:
        super().__init__()
        if cell_type not in CELL_TYPES:
            raise ValueError(f"cell_type must be one of {CELL_TYPES}, got {cell_type!r}")
        self.cell_type = cell_type
        d_model = int(spec["d_model"])
        n_heads = int(spec["n_heads"])
        vocab = int(spec["vocab_size"])
        seq_len = int(spec["seq_len"])
        mix, solver, n_heads_cell = _mix_solver(cell_type, n_heads)
        self.newton_cfg = _newton_config(spec, cell_type)
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            BabyLMBlock(
                d_model,
                mix=mix,
                n_heads=n_heads_cell,
                solver=solver,
                newton_cfg=self.newton_cfg,
                max_recurrent_norm=spec.get("max_recurrent_norm", 0.5),
                mlp_mult=int(spec["mlp_mult"]),
                mlp_act=str(spec["mlp_act"]),
            )
            for _ in range(int(spec["num_layers"]))
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight
        self._init_embeddings()
        log.info(
            "babylm_model cell_type=%s mix=%s solver=%s d_model=%d layers=%d params=%d",
            cell_type,
            mix,
            solver,
            d_model,
            len(self.blocks),
            count_params(self),
        )

    def _init_embeddings(self) -> None:
        # GPT-2: token/pos std 0.02 (Radford et al. 2019). Default N(0,1)
        # embeddings send sLSTM n outside the Newton basin at d_model=384.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        _b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        h = self.embed(tokens) + self.pos(pos)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm_f(h))

    def newton_stats(self) -> list[NewtonStats]:
        out: list[NewtonStats] = []
        for block in self.blocks:
            out.extend(block.rnn.last_stats)
        return out

    def rnn_io_at_layer(self, tokens: Tensor, layer: int) -> tuple[Tensor, Tensor]:
        """Post-LN input and RNN hidden at ``layer``. Stem is blocks ``[:layer]``."""
        n = len(self.blocks)
        if not 0 <= layer < n:
            raise IndexError(f"layer={layer} outside [0, {n})")
        _b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        h = self.embed(tokens) + self.pos(pos)
        for i, block in enumerate(self.blocks):
            if i == layer:
                x_in = block.norm_rnn(h)
                return x_in, block.rnn(x_in)
            h = block(h)
        raise RuntimeError("rnn_io_at_layer fell through the stack")


def _mix_solver(cell_type: str, n_heads: int) -> tuple[str, str, int | None]:
    if cell_type == "dense":
        return "head", "sequential", n_heads
    if cell_type == "diag_seq":
        return "diag", "sequential", None
    return "diag", "auto", None


def _newton_config(spec: dict, cell_type: str) -> NewtonConfig:
    verify_first = bool(spec.get("verify_first_step", False))
    if cell_type != "diag_fused":
        return NewtonConfig(
            max_iters=int(spec["newton_iters"]),
            scan_backend="eager",
            verify_first_step=verify_first,
        )
    kw: dict = {
        "max_iters": int(spec["newton_iters"]),
        "scan_backend": "fused",
        "picard_iters": int(spec["picard_iters"]),
        "picard_adapt": bool(spec["picard_adapt"]),
        "verify_first_step": verify_first,
    }
    # Optional windowed fused path (NewtonConfig.fused_time_loop).
    if bool(spec.get("fused_time_loop", False)):
        kw["fused_time_loop"] = True
        if spec.get("fused_window_len") is not None:
            kw["fused_window_len"] = int(spec["fused_window_len"])
    if spec.get("chunk_len") is not None:
        kw["chunk_len"] = int(spec["chunk_len"])
    return NewtonConfig(**kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def mixing_label(cell_type: str) -> str:
    if cell_type == "dense":
        return r"Dense $d_{\mathrm{head}}\times d_{\mathrm{head}}$"
    return r"Vector $d_h$"


def solver_label(cell_type: str) -> str:
    if cell_type == "diag_fused":
        return "Fused Newton-Scan"
    return "Sequential"


def arm_label(cell_type: str) -> str:
    return {
        "dense": "`sLSTM-Dense`",
        "diag_seq": "`Diag-sLSTM-Seq`",
        "diag_fused": "`ParaSLSTM` (Ours)",
    }[cell_type]
