"""Pre-norm residual sLSTM block. Ours — not NX-AI's ``xlstm`` package.

Beck et al. 2024 put LayerNorm + residual around sLSTM in the backbone.
``ParaRNN`` does not: stacking there is naive. This module is that missing
block so an xLSTM-style stack can swap the recurrence backend.

``backend="newton"``: ``ParaRNN`` ``solver="auto"`` (Newton while
``self.training``, sequential ``step`` in ``eval()``).
``backend="eager"``: ``solver="sequential"`` always. Pass ``solver=`` to
override. FlashRNN is a bench in ``scripts/``, not a switch here —
``cuda_fused`` needs compute capability ≥ 8.0, and we do not stub it.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton import NewtonConfig

_BACKENDS = ("newton", "eager")


class xLSTMBlock(nn.Module):
    """``(B, T, d_model) → (B, T, d_model)``: LN → sLSTM → residual add.

    ``mix='diag'`` is the fused Newton cell. Head mix is the xLSTM compromise
    and stays on ``step``. ``max_recurrent_norm`` is App. C.1 (0.5 LM; ``None``
    on parity). This block is the sLSTM half an xLSTM stack could swap in
    for parallel *training*; mLSTM stays theirs. Not NX-AI. Not xLSTM-7B.
    """

    def __init__(
        self,
        d_model: int,
        *,
        backend: str = "newton",
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        config: NewtonConfig | None = None,
        solver: Literal["auto", "newton", "sequential"] | None = None,
        batch_first: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {_BACKENDS}, got {backend!r}. "
                "FlashRNN is scripts/ only."
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.backend = backend
        self.norm = nn.LayerNorm(d_model, **factory_kwargs)
        cell = ParaSLSTM(
            d_in=d_model,
            d_h=d_model,
            mix=mix,
            n_heads=n_heads,
            max_recurrent_norm=max_recurrent_norm,
            **factory_kwargs,
        )
        if solver is None:
            solver = "sequential" if backend == "eager" else "auto"
        self.rnn = ParaRNN(
            cell,
            config=config or NewtonConfig(),
            output_hidden=True,
            solver=solver,
            batch_first=batch_first,
        )

    @property
    def cell(self) -> ParaSLSTM:
        cell = self.rnn.layers[0]
        if not isinstance(cell, ParaSLSTM):
            raise TypeError(f"expected ParaSLSTM, got {type(cell).__name__}")
        return cell

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        z = self.norm(x)
        h = self.rnn(z, h0)
        return x + h
