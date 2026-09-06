"""Sequence wrapper: Newton in ``train()``, sequential unroll in ``eval()``."""

from __future__ import annotations

import inspect
import logging
import warnings
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.protocol import check_cell
from pararnn.layout import LSTM_CELL, LSTM_HIDDEN, swap_lstm_ch, validate_cu_seqlens
from pararnn.solvers.newton import NewtonConfig, NewtonStats, newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)

_SOLVERS = ("auto", "newton", "sequential")
_HIDDEN_LAYOUTS = ("paper", "pytorch")


class ParaRNN(nn.Module):
    """Stacked RNN cells: Newton+scan in ``train()``, sequential ``step`` in ``eval()``.

    ``solver='auto'`` follows ``self.training``; ``'newton'`` / ``'sequential'``
    force a path. Default layout is batch-first ``(B, T, d_in)``. Multi-slot
    cells (LSTM / sLSTM) emit the hidden slot unless ``output_hidden=False``.
    See ``__init__`` / ``forward`` for packed ``cu_seqlens`` and
    ``hidden_layout='pytorch'``.

    Attributes
    ----------
    layers : nn.ModuleList
    config : NewtonConfig
    return_hidden, output_hidden : bool
    solver : {'auto', 'newton', 'sequential'}
    batch_first : bool
    hidden_layout : {'paper', 'pytorch'}
    dropout : float
    last_stats : list of NewtonStats
        Per-layer diagnostics from the last Newton forward.

    See Also
    --------
    newton_apply, sequential_apply, ParaSLSTMBlock, RNNCell
    """

    def __init__(
        self,
        cell: nn.Module | Sequence[nn.Module],
        *,
        num_layers: int = 1,
        config: NewtonConfig | None = None,
        return_hidden: bool = False,
        output_hidden: bool | None = None,
        solver: Literal["auto", "newton", "sequential"] = "auto",
        batch_first: bool = True,
        hidden_layout: Literal["paper", "pytorch"] = "paper",
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cell : nn.Module or sequence of nn.Module
            Template cell, or one cell per layer. With a single cell and
            ``num_layers > 1``, deeper layers use ``input_size = hidden_size``.
        num_layers : int, default=1
            Stack depth (must match ``len(cell)`` when ``cell`` is a sequence).
        config : NewtonConfig or None, default=None
            Defaults to ``NewtonConfig()``.
        return_hidden : bool, default=False
            Also return last-step state of the final layer (paper layout).
        output_hidden : bool or None, default=None
            Emit only the hidden slot; default True when last cell is multi-slot.
        solver : {'auto', 'newton', 'sequential'}, default='auto'
            ``'auto'``: Newton while ``training``, sequential in ``eval()``.
        batch_first : bool, default=True
            ``(B, T, …)`` vs ``(T, B, …)``.
        hidden_layout : {'paper', 'pytorch'}, default='paper'
            LSTM packing; ``'pytorch'`` needs ``ParaLSTM`` layers.
        dropout : float, default=0.0
            Between-layer dropout in ``[0, 1]``.
        device, dtype : optional
            Applied to constructed / provided cells.

        Raises
        ------
        ValueError
            Invalid ``solver``, ``hidden_layout``, ``dropout``, or layer counts.
        TypeError
            ``hidden_layout='pytorch'`` on non-LSTM cells, or ``check_cell`` fails.
        """
        super().__init__()
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {solver!r}")
        if hidden_layout not in _HIDDEN_LAYOUTS:
            raise ValueError(
                f"hidden_layout must be one of {_HIDDEN_LAYOUTS}, got {hidden_layout!r}"
            )
        if not 0 <= dropout <= 1:
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                f"inclusive, but got a ratio of {dropout}"
            )
        self.config = config or NewtonConfig()
        self.return_hidden = return_hidden
        self.solver = solver
        self.batch_first = batch_first
        self.hidden_layout = hidden_layout
        self.dropout = dropout
        self.last_stats: list[NewtonStats] = []
        self.layers = nn.ModuleList(_build_layers(cell, num_layers, device=device, dtype=dtype))
        if dropout > 0 and len(self.layers) == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                "num_layers greater than 1, but got dropout="
                f"{dropout} and num_layers={len(self.layers)}",
                stacklevel=2,
            )
        if hidden_layout == "pytorch":
            _require_lstm_layout(self.layers)
        if output_hidden is None:
            self.output_hidden = _hidden_slot(self.layers[-1]) is not None
        else:
            self.output_hidden = output_hidden

    def _use_newton(self) -> bool:
        if self.solver == "newton":
            return True
        if self.solver == "sequential":
            return False
        return self.training

    def reset_parameters(self) -> None:
        for cell in self.layers:
            reset = getattr(cell, "reset_parameters", None)
            if callable(reset):
                reset()

    def extra_repr(self) -> str:
        return (
            f"solver={self.solver!r}, effective={self._effective_solver()}, "
            f"batch_first={self.batch_first}, dropout={self.dropout}, "
            f"output_hidden={self.output_hidden}, "
            f"hidden_layout={self.hidden_layout!r}"
        )

    def _effective_solver(self) -> str:
        return "newton" if self._use_newton() else "sequential"

    def forward(
        self,
        x: Tensor,
        h0: Tensor | Sequence[Tensor] | None = None,
        cu_seqlens: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        x : Tensor
            Input sequence. Tensor of shape ``(batch, time, d_in)`` when
            ``batch_first=True``, else ``(time, batch, d_in)``. With
            ``cu_seqlens``, use packed batch-first layout ``(1, N, d_in)``.
        h0 : Tensor or sequence of Tensor or None, default=None
            Initial state(s). Batch-leading. For one layer, a single state
            tensor; for ``num_layers > 1``, a sequence of length
            ``num_layers``. Shape matches each cell's
            ``(batch, …, d_h)`` / ``(batch, state_slots, d_h)``. With
            packing, ``batch`` is the number of sequences ``S``.
        cu_seqlens : Tensor or None, default=None
            Cumulative sequence lengths for packed input
            (``int32`` / ``int64``, length ``S + 1``).

        Returns
        -------
        output : Tensor
            Sequence output. Tensor of shape ``(batch, time, d_h)`` (or
            ``(time, batch, d_h)``) when ``output_hidden`` extracts the
            hidden slot; full multi-slot trajectories keep the slot axis.
        h_n : Tensor, optional
            With ``return_hidden`` and ``hidden_layout='paper'``: last
            time-step state of the final layer.
        (h_n, c_n) : tuple of Tensor, optional
            With ``hidden_layout='pytorch'``: ``nn.LSTM``-style finals,
            each Tensor of shape ``(num_layers, batch, hidden_size)``.

        Raises
        ------
        ValueError
            When input rank/features, ``h0`` shapes, or packed batch size
            are invalid.
        """
        _validate_input(x, self.layers[0], batch_first=self.batch_first)
        if not self.batch_first:
            x = x.transpose(0, 1)
        h0s = _split_h0(h0, len(self.layers))
        cs = None
        h0_batch = x.shape[0]
        if cu_seqlens is not None:
            if x.shape[0] != 1:
                raise ValueError(f"cu_seqlens packs x with batch=1, got batch={x.shape[0]}")
            cs = validate_cu_seqlens(cu_seqlens, x.shape[1])
            h0_batch = int(cs.numel()) - 1
        _validate_h0s(h0s, self.layers, batch=h0_batch)
        if self.hidden_layout == "pytorch":
            h0s = [None if h is None else swap_lstm_ch(h) for h in h0s]
        h = x
        n = len(self.layers)
        self.last_stats = []
        use_newton = self._use_newton()
        collect_lasts = self.hidden_layout == "pytorch"
        lasts: list[Tensor] = []
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "para_rnn_forward",
                extra={
                    "training": self.training,
                    "solver": self._effective_solver(),
                    "solver_flag": self.solver,
                    "num_layers": n,
                    "batch": x.shape[0],
                    "seq_len": x.shape[1],
                    "n_seq": None if cs is None else int(cs.numel()) - 1,
                },
            )
        for i, cell in enumerate(self.layers):
            if use_newton:
                st = NewtonStats()
                h = newton_apply(cell, h, self.config, h0=h0s[i], stats=st, cu_seqlens=cs)
                self.last_stats.append(st)
            else:
                h = sequential_apply(cell, h, h0s[i], cu_seqlens=cs)
            if collect_lasts:
                lasts.append(h[:, -1])
            if i + 1 < n:
                h = _next_layer_input(h, cell)
                if self.dropout > 0.0 and self.training:
                    h = F.dropout(h, p=self.dropout, training=True)
        y = h
        last = h[:, -1]
        slot = _hidden_slot(self.layers[-1])
        if self.output_hidden and slot is not None:
            y = h[:, :, slot, :]
        if not self.batch_first:
            y = y.transpose(0, 1)
        if self.hidden_layout == "pytorch":
            return y, _lstm_hn_cn(lasts)
        if self.return_hidden:
            return y, last
        return y


def _build_layers(
    cell: nn.Module | Sequence[nn.Module],
    num_layers: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> list[nn.Module]:
    if isinstance(cell, (list, tuple)):
        if not cell:
            raise ValueError("cell list must be non-empty")
        if num_layers not in (1, len(cell)):
            raise ValueError(f"num_layers={num_layers} does not match len(cells)={len(cell)}")
        layers = list(cell)
        for c in layers:
            check_cell(c)
        return [_maybe_to(c, device, dtype) for c in layers]
    check_cell(cell)
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    layers = [_maybe_to(cell, device, dtype)]
    if num_layers > 1:
        layers.extend(_extra_layers(layers[0], num_layers - 1, device=device, dtype=dtype))
    return layers


def _extra_layers(
    cell: nn.Module,
    n_extra: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> list[nn.Module]:
    """Further layers: same class, ``hidden_size`` → ``input_size``."""
    cls = type(cell)
    kw = _stack_kwargs(cell)
    extras: list[nn.Module] = []
    if device is None or dtype is None:
        inferred_device, inferred_dtype = _param_device_dtype(cell)
        if device is None:
            device = inferred_device
        if dtype is None:
            dtype = inferred_dtype
    hid = getattr(cell, "hidden_size", cell.d_h)
    sig = inspect.signature(cls.__init__)
    if "device" in sig.parameters:
        kw["device"] = device
    if "dtype" in sig.parameters:
        kw["dtype"] = dtype
    for _ in range(n_extra):
        try:
            extra = cls(input_size=hid, hidden_size=hid, **kw)
        except TypeError:
            try:
                extra = cls(d_in=cell.d_h, d_h=cell.d_h, **kw)
            except TypeError as exc:
                raise TypeError(
                    f"{cls.__name__} cannot be stacked (num_layers>1): need "
                    "type(cell)(input_size=hidden_size, hidden_size=hidden_size) "
                    "or pass a list of cells. Use num_layers=1 for custom cells "
                    "without that constructor."
                ) from exc
        extras.append(_maybe_to(extra, device, dtype))
    return extras


def _stack_kwargs(cell: nn.Module) -> dict:
    sig = inspect.signature(type(cell).__init__)
    kw: dict = {}
    for name in ("max_recurrent_norm", "mix", "eps", "n_heads"):
        if name in sig.parameters and hasattr(cell, name):
            kw[name] = getattr(cell, name)
    return kw


def _param_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters(), None)
    if param is not None:
        return param.device, param.dtype
    buf = next(module.buffers(), None)
    if buf is not None:
        return buf.device, buf.dtype
    return torch.device("cpu"), torch.float32


def _maybe_to(
    module: nn.Module,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.Module:
    if device is None and dtype is None:
        return module
    kw: dict = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    return module.to(**kw)


def _next_layer_input(states: Tensor, cell: nn.Module) -> Tensor:
    slot = _hidden_slot(cell)
    if slot is None:
        return states
    return states[:, :, slot, :]


def _hidden_slot(cell: nn.Module) -> int | None:
    slots = getattr(cell, "state_slots", 1)
    if slots == 1:
        return None
    return getattr(cell, "hidden_slot", 1)


def _require_lstm_layout(layers: nn.ModuleList) -> None:
    for cell in layers:
        if not isinstance(cell, ParaLSTM):
            raise TypeError(
                "hidden_layout='pytorch' is ParaLSTM only "
                "(nn.LSTM (h, c) vs paper (c, h)); "
                f"got {type(cell).__name__}"
            )


def _lstm_hn_cn(lasts: list[Tensor]) -> tuple[Tensor, Tensor]:
    stacked = torch.stack(lasts, dim=0)
    return stacked[:, :, LSTM_HIDDEN, :], stacked[:, :, LSTM_CELL, :]


def _input_size(cell: nn.Module) -> int | None:
    size = getattr(cell, "input_size", None)
    if size is None:
        size = getattr(cell, "d_in", None)
    return size if isinstance(size, int) else None


def _state_shape(cell: nn.Module, batch: int) -> tuple[int, ...]:
    tail = getattr(cell, "state_shape", None)
    if tail is not None:
        return (batch, *tuple(tail))
    hid = getattr(cell, "hidden_size", None)
    if hid is None:
        hid = cell.d_h
    slots = getattr(cell, "state_slots", 1)
    if slots == 1:
        return (batch, hid)
    return (batch, slots, hid)


def _validate_input(x: Tensor, cell: nn.Module, *, batch_first: bool) -> None:
    layout = "(batch, time, features)" if batch_first else "(time, batch, features)"
    if x.ndim != 3:
        raise ValueError(f"expected 3D input {layout}, got {x.ndim}D of shape {tuple(x.shape)}")
    d_in = _input_size(cell)
    feat = x.shape[-1]
    if d_in is not None and feat != d_in:
        raise ValueError(f"expected input features {d_in}, got {feat} (shape {tuple(x.shape)})")


def _validate_h0s(h0s: list[Tensor | None], layers: nn.ModuleList, *, batch: int) -> None:
    n_layers = len(layers)
    for i, (h0, cell) in enumerate(zip(h0s, layers, strict=True)):
        if h0 is None:
            continue
        expected = _state_shape(cell, batch)
        if tuple(h0.shape) != expected:
            where = f"h0 for layer {i}" if n_layers > 1 else "h0"
            raise ValueError(f"{where} expected shape {expected}, got {tuple(h0.shape)}")


def _split_h0(h0: Tensor | Sequence[Tensor] | None, n_layers: int) -> list[Tensor | None]:
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
        raise TypeError(f"h0 for {n_layers} layers must be a tuple/list of length {n_layers}")
    return list(h0)
