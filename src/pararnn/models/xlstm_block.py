"""Pre-norm residual sLSTM block (Beck et al. 2024).

LayerNorm + residual around sLSTM. This is the sLSTM half of an xLSTM
stack. ``solver`` matches ``ParaRNN``: Newton while
``self.training`` when ``solver='auto'``.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton import NewtonConfig

_SOLVERS = ("auto", "newton", "sequential")


class xLSTMBlock(nn.Module):
    """``(B, T, d_model) → (B, T, d_model)``: LN → sLSTM → residual add.

    ``mix='diag'`` is the fused Newton cell. Head mix is the xLSTM-style
    block-diagonal ``R`` and stays on ``step``. ``d_model`` is both input
    and hidden size (no projection).
    """

    def __init__(
        self,
        d_model: int,
        *,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        config: NewtonConfig | None = None,
        solver: Literal["auto", "newton", "sequential"] = "auto",
        batch_first: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        cell = ParaSLSTM(
            input_size=d_model,
            hidden_size=d_model,
            mix=mix,
            n_heads=n_heads,
            max_recurrent_norm=max_recurrent_norm,
            **factory_kwargs,
        )
        self.rnn = ParaRNN(
            cell,
            config=config or NewtonConfig(),
            output_hidden=True,
            solver=solver,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    @property
    def cell(self) -> ParaSLSTM:
        cell = self.rnn.layers[0]
        if not isinstance(cell, ParaSLSTM):
            raise TypeError(f"expected ParaSLSTM, got {type(cell).__name__}")
        return cell

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.rnn.reset_parameters()

    def extra_repr(self) -> str:
        return f"{self.d_model}, solver={self.rnn.solver!r}"

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        z = self.norm(x)
        h = self.rnn(z, h0)
        return x + h
