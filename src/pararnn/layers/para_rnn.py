"""Sequence wrapper: Newton in ``train()``, sequential unroll in ``eval()``."""

from __future__ import annotations

import inspect
import logging

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import check_cell
from pararnn.layout import LSTM_HIDDEN
from pararnn.solvers.newton import NewtonConfig, newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)


class ParaRNN(nn.Module):
    """Stack of RNN cells as a batch-first sequence module.

    ``self.training`` selects the solver: Newton+scan (Alg. 1) while training,
    sequential ``step`` unroll in ``eval()``. No residual or LayerNorm here —
    stacking is naive; the caller owns the backbone.

    ``x`` is ``(batch, time, d_in)``. Output is the last cell's full state
    (GRU: ``(B, T, d_h)``; LSTM: ``(B, T, 2, d_h)``). Intermediate LSTM
    layers feed only the hidden slot into the next layer.
    """

    def __init__(
        self,
        cell: nn.Module,
        *,
        num_layers: int = 1,
        config: NewtonConfig | None = None,
    ) -> None:
        super().__init__()
        check_cell(cell)
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.config = config or NewtonConfig()
        layers: list[nn.Module] = [cell]
        if num_layers > 1:
            layers.extend(_extra_layers(cell, num_layers - 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, h0: Tensor | list | tuple | None = None) -> Tensor:
        h0s = _split_h0(h0, len(self.layers))
        h = x
        n = len(self.layers)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "para_rnn_forward",
                extra={
                    "training": self.training,
                    "solver": "newton" if self.training else "sequential",
                    "num_layers": n,
                    "batch": x.shape[0],
                    "seq_len": x.shape[1],
                },
            )
        for i, cell in enumerate(self.layers):
            if self.training:
                h = newton_apply(cell, h, self.config, h0=h0s[i])
            else:
                h = sequential_apply(cell, h, h0s[i])
            if i + 1 < n:
                h = _next_layer_input(h, cell)
        return h


def _extra_layers(cell: nn.Module, n_extra: int) -> list[nn.Module]:
    """Further layers: ``type(cell)(d_in=cell.d_h, d_h=cell.d_h, ...)``."""
    cls = type(cell)
    kw = _stack_kwargs(cell)
    extras: list[nn.Module] = []
    device, dtype = _param_device_dtype(cell)
    for _ in range(n_extra):
        try:
            extra = cls(d_in=cell.d_h, d_h=cell.d_h, **kw)
        except TypeError as exc:
            raise TypeError(
                f"{cls.__name__} cannot be stacked (num_layers>1): need "
                "type(cell)(d_in=cell.d_h, d_h=cell.d_h, ...). "
                "Use num_layers=1 for custom cells without that constructor."
            ) from exc
        extras.append(extra.to(device=device, dtype=dtype))
    return extras


def _stack_kwargs(cell: nn.Module) -> dict:
    sig = inspect.signature(type(cell).__init__)
    kw: dict = {}
    if "max_recurrent_norm" in sig.parameters and hasattr(cell, "max_recurrent_norm"):
        kw["max_recurrent_norm"] = cell.max_recurrent_norm
    return kw


def _param_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is not None:
        return param.device, param.dtype
    buf = next(module.buffers(), None)
    if buf is not None:
        return buf.device, buf.dtype
    return torch.device("cpu"), torch.float32


def _next_layer_input(states: Tensor, cell: nn.Module) -> Tensor:
    slots = getattr(cell, "state_slots", 1)
    if slots == 2:
        return states[:, :, LSTM_HIDDEN, :]
    return states


def _split_h0(h0: Tensor | list | tuple | None, n_layers: int) -> list:
    if h0 is None:
        return [None] * n_layers
    if n_layers == 1:
        if isinstance(h0, (list, tuple)):
            if len(h0) != 1:
                raise TypeError(
                    f"h0 for 1 layer must be a state tensor or a length-1 sequence, got {len(h0)}"
                )
            return [h0[0]]
        return [h0]
    if not isinstance(h0, (list, tuple)) or len(h0) != n_layers:
        raise TypeError(
            f"h0 for {n_layers} layers must be a tuple/list of length {n_layers}"
        )
    return list(h0)
