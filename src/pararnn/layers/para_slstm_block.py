"""Pre-norm ParaSLSTM + SwiGLU residual block for stacking LMs / SSM hybrids."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton import NewtonConfig, NewtonStats


class SwiGLU(nn.Module):
    """Gated MLP used in LLaMA-style blocks (``SiLU(u) * v``).

    ``mlp_ratio=4`` yields hidden width ``4 * d_model`` (BabyLM / common
    LLM default). The up-projection is ``2 * hidden`` before the gate split.

    Attributes
    ----------
    up : nn.Linear
        Projection ``d_model → 2 * hidden``.
    down : nn.Linear
        Projection ``hidden → d_model``.
    """

    def __init__(self, d_model: int, *, mlp_ratio: float = 4.0) -> None:
        """
        Parameters
        ----------
        d_model : int
            Model / residual stream width.
        mlp_ratio : float, default=4.0
            Expansion factor: ``hidden = int(mlp_ratio * d_model)``.

        Raises
        ------
        ValueError
            When ``mlp_ratio <= 0`` or the resulting ``hidden`` is ``< 1``.
        """
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio!r}")
        hidden = int(mlp_ratio * d_model)
        if hidden < 1:
            raise ValueError(f"mlp_ratio={mlp_ratio} yields hidden={hidden} for d_model={d_model}")
        self.up = nn.Linear(d_model, 2 * hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input. Tensor of shape ``(..., d_model)``.

        Returns
        -------
        y : Tensor
            Output. Tensor of shape ``(..., d_model)``.
        """
        u, v = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(u) * v)


class ParaSLSTMBlock(nn.Module):
    """Pre-norm residual: RMSNorm → ParaRNN(ParaSLSTM) → + → RMSNorm → SwiGLU → +.

    Attributes
    ----------
    d_model : int
        Residual stream width (also ``ParaSLSTM`` ``d_in`` / ``d_h``).
    norm_rnn : nn.RMSNorm
        Pre-norm before the recurrent branch.
    rnn : ParaRNN
        Single-layer ``ParaSLSTM`` wrapper with ``output_hidden=True``.
    norm_mlp : nn.RMSNorm
        Pre-norm before the SwiGLU branch.
    mlp : SwiGLU
        Feed-forward network.
    drop : nn.Dropout or nn.Identity
        Residual dropout.

    See Also
    --------
    ParaSLSTM, ParaRNN, SwiGLU
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
        """
        Parameters
        ----------
        d_model : int
            Model width (``>= 1``).
        mlp_ratio : float, default=4.0
            SwiGLU expansion factor.
        mix : {'diag', 'head', 'dense'}, default='diag'
            ``ParaSLSTM`` recurrent mixing mode.
        n_heads : int or None, default=None
            Required when ``mix='head'``.
        config : NewtonConfig or None, default=None
            Newton settings. Defaults to ``NewtonConfig(max_iters=3)``
            (paper uses 3 Newton iterations for ParaGRU/ParaLSTM; sLSTM
            training in this repo follows the same ``K=3`` default).
        solver : {'auto', 'newton', 'sequential'}, default='auto'
            Forwarded to ``ParaRNN``.
        max_recurrent_norm : float or None, default=0.5
            App. C.1 clip on ``R`` (Danieli / this repo BabyLM default).
        dropout : float, default=0.0
            Residual dropout probability.
        eps : float, default=1e-6
            Epsilon for both ``RMSNorm`` layers.

        Raises
        ------
        ValueError
            When ``d_model < 1``, ``mix='head'`` lacks ``n_heads``, or
            ``n_heads`` is set with a non-head mix.
        """
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
        """Per-layer Newton stats from the most recent recurrent forward."""
        return list(self.rnn.last_stats)

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-norm RNN and SwiGLU residual branches.

        Parameters
        ----------
        x : Tensor
            Input. Tensor of shape ``(batch, time, d_model)``.

        Returns
        -------
        y : Tensor
            Output. Tensor of shape ``(batch, time, d_model)``.

        Raises
        ------
        ValueError
            When ``x`` is not rank-3 or the last dim differs from
            ``d_model``.
        """
        if x.dim() != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected x (B, T, {self.d_model}), got {tuple(x.shape)}"
            )
        x = x + self.drop(self.rnn(self.norm_rnn(x)))
        return x + self.drop(self.mlp(self.norm_mlp(x)))
