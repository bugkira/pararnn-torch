"""Cell contract for Newton / sequential / ``ParaRNN``.

A cell is any ``nn.Module`` with ``hidden_size`` (alias ``d_h``) and
``step(h_prev, x)``. Optional fast paths: ``step_with_jacobian``,
``jac_structure``, and ``wx=`` on ``step``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


def resolve_layer_sizes(
    input_size: int | None = None,
    hidden_size: int | None = None,
    *,
    d_in: int | None = None,
    d_h: int | None = None,
) -> tuple[int, int]:
    """``input_size`` / ``hidden_size`` with aliases ``d_in`` / ``d_h``."""
    in_sz = _pick_size("input_size", input_size, "d_in", d_in)
    hid = _pick_size("hidden_size", hidden_size, "d_h", d_h)
    if in_sz < 1 or hid < 1:
        raise ValueError(f"input_size and hidden_size must be positive, got {in_sz}, {hid}")
    return in_sz, hid


def _pick_size(name: str, value: int | None, alias: str, alias_value: int | None) -> int:
    if value is not None and alias_value is not None and value != alias_value:
        raise ValueError(f"{name}={value!r} conflicts with {alias}={alias_value!r}")
    picked = value if value is not None else alias_value
    if picked is None:
        raise TypeError(f"missing {name} (or alias {alias})")
    if not isinstance(picked, int):
        raise TypeError(f"{name} must be int, got {type(picked).__name__}")
    return picked


@runtime_checkable
class RNNCell(Protocol):
    """Minimal surface the solvers call.

    ``hidden_size`` / ``d_h``: channel width. ``state_slots``: ``1`` →
    ``(..., d_h)``; ``2`` → ``(..., 2, d_h)`` (c, h); ``4`` → ``(..., 4, d_h)``
    (sLSTM: c, n, m, h). Default 1 if omitted.
    ``step_with_jacobian`` and ``wx=`` on ``step`` are optional fast paths.
    ``jac_structure``: ``diag`` | ``block2`` | ``block4`` | ``head`` | ``dense``.
    """

    d_h: int
    hidden_size: int

    def step(self, h_prev: Tensor, x: Tensor) -> Tensor: ...


def check_cell(cell: nn.Module) -> None:
    """Require ``hidden_size`` / ``d_h`` and ``step(h_prev, x)``."""
    d_h = getattr(cell, "d_h", None)
    if d_h is None:
        d_h = getattr(cell, "hidden_size", None)
    if not isinstance(d_h, int) or d_h < 1:
        raise TypeError(
            f"{type(cell).__name__} needs integer d_h / hidden_size (hidden size), got {d_h!r}"
        )
    if not callable(getattr(cell, "step", None)):
        raise TypeError(f"{type(cell).__name__} needs a step(h_prev, x) method")
    slots = getattr(cell, "state_slots", 1)
    if slots not in (1, 2, 4):
        raise TypeError(f"{type(cell).__name__}.state_slots must be 1, 2, or 4, got {slots!r}")
