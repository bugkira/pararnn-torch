"""Pre-norm ParaSLSTM + SwiGLU residual block for stacking LMs / SSM hybrids."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton import NewtonConfig, NewtonStats


class SwiGLU(nn.Module):
    """Gated MLP used in LLaMA-style blocks (SiLU(u) * v).

    ``mlp_ratio=4`` → hidden width ``4 * d_model`` (BabyLM / common LLM default).
    Intermediate is ``2 * hidden`` before the gate split.
    """

    def __init__(self, d_model: int, *, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio!r}")
        hidden = int(mlp_ratio * d_model)
        if hidden < 1:
            raise ValueError(f"mlp_ratio={mlp_ratio} yields hidden={hidden} for d_model={d_model}")
        self.up = nn.Linear(d_model, 2 * hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(u) * v)


class ParaSLSTMBlock(nn.Module):
    """One pre-norm residual layer around fused-ready ``ParaSLSTM``.

    ::

        x → RMSNorm → ParaRNN(ParaSLSTM) → + → RMSNorm → SwiGLU → +

    For people stacking an LM / torchtitan-style trunk without the NX-AI
    ``xlstm`` package. The recurrent core stays ``ParaRNN`` + ``ParaSLSTM``;
    LayerNorm/FFN are not part of the Newton cell.

    ``mix='diag'`` (default) is the fused training cell. ``mix='head'`` /
    ``'dense'`` stay ablations (unfused scan). No FlashAttention hybrid here —
    that is a separate stack composition.
    """

    def __init__(
        self,
        d_model: int,
        *,
        mlp_ratio: float = 4.0,
        mix: str = "diag",
        n_heads: int | None = None,
        config: NewtonConfig | None = None,
        solver: str = "auto",
        max_recurrent_norm: float | None = 0.5,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model!r}")
        # App. C.1 default clip 0.5 on R (Danieli / this repo BabyLM).
        cell_kw: dict = {"mix": mix, "max_recurrent_norm": max_recurrent_norm}
        if mix == "head":
            if n_heads is None:
                raise ValueError("mix='head' requires n_heads")
            cell_kw["n_heads"] = n_heads
        elif n_heads is not None:
            raise ValueError("n_heads is only used with mix='head'")
        self.d_model = int(d_model)
        self.norm_rnn = nn.RMSNorm(d_model, eps=eps)
        self.rnn = ParaRNN(
            ParaSLSTM(d_model, d_model, **cell_kw),
            config=config or NewtonConfig(max_iters=3),
            output_hidden=True,
            solver=solver,
        )
        self.norm_mlp = nn.RMSNorm(d_model, eps=eps)
        self.mlp = SwiGLU(d_model, mlp_ratio=mlp_ratio)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @property
    def last_stats(self) -> list[NewtonStats]:
        return list(self.rnn.last_stats)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected x (B, T, {self.d_model}), got {tuple(x.shape)}"
            )
        x = x + self.drop(self.rnn(self.norm_rnn(x)))
        return x + self.drop(self.mlp(self.norm_mlp(x)))
