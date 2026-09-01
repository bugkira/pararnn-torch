"""Cell contract for Newton / sequential / ``ParaRNN``.

A cell is any ``nn.Module`` with ``d_h`` and ``step(h_prev, x)``. No base class.
Ones-JVP (default Autograd Jacobian) is exact Newton iff ``f`` is channelwise
in ``h``. Four-slot channelwise cells (sLSTM diag mix) set
``cell.jac_structure='block4'``. Head-block mixing uses ``'head'``; mixing
all channels needs ``'dense'``.
Fused Newton is a handwritten Triton kernel for ParaGRU / ParaLSTM / ParaSLSTM
``mix='diag'``, not a generic ``f``. Head/dense sLSTM stay on ``cell.step``.
``scan_backend='auto'`` picks fused on CUDA for those cells (fp16/fp32), else
Triton scan + ``step``, else eager.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class RNNCell(Protocol):
    """Minimal surface the solvers call.

    ``state_slots``: ``1`` → ``(..., d_h)``; ``2`` → ``(..., 2, d_h)`` (c, h);
    ``4`` → ``(..., 4, d_h)`` (sLSTM: c, n, m, h). Default 1 if omitted.
    ``step_with_jacobian`` and ``wx=`` on ``step`` are optional fast paths.
    """

    d_h: int

    def step(self, h_prev: Tensor, x: Tensor) -> Tensor: ...


def check_cell(cell: nn.Module) -> None:
    """Raise ``TypeError`` if ``cell`` cannot be unrolled or Newton-solved."""
    if not hasattr(cell, "d_h"):
        raise TypeError(
            f"{type(cell).__name__} needs integer attribute d_h (hidden size)"
        )
    d_h = cell.d_h
    if not isinstance(d_h, int) or d_h < 1:
        raise TypeError(f"{type(cell).__name__}.d_h must be a positive int, got {d_h!r}")
    if not callable(getattr(cell, "step", None)):
        raise TypeError(f"{type(cell).__name__} needs a step(h_prev, x) method")
    slots = getattr(cell, "state_slots", 1)
    if slots not in (1, 2, 4):
        raise TypeError(
            f"{type(cell).__name__}.state_slots must be 1, 2, or 4, got {slots!r}"
        )
