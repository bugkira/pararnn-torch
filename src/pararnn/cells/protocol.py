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
    """Resolve ``input_size`` / ``hidden_size`` with aliases ``d_in`` / ``d_h``.

    Parameters
    ----------
    input_size : int or None, default=None
        Input feature width. Alias of ``d_in``.
    hidden_size : int or None, default=None
        Hidden / channel width. Alias of ``d_h``.
    d_in : int or None, default=None
        Alias for ``input_size``.
    d_h : int or None, default=None
        Alias for ``hidden_size``.

    Returns
    -------
    input_size : int
        Resolved positive input width.
    hidden_size : int
        Resolved positive hidden width.

    Raises
    ------
    ValueError
        When a name and its alias disagree, or a resolved size is ``< 1``.
    TypeError
        When both forms of a size are missing, or a value is not ``int``.
    """
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
    """Minimal surface Newton, sequential unroll, and ``ParaRNN`` call.

    Implementations are ``nn.Module`` subclasses. Solvers read ``hidden_size``
    / ``d_h`` and call ``step``. Optional attributes and methods unlock fused
    or structured Jacobian paths.

    Attributes
    ----------
    d_h : int
        Channel / hidden width (alias of ``hidden_size``).
    hidden_size : int
        Same width as ``d_h``.
    state_slots : int, optional
        Number of state tensors stacked on a slot axis before the channel
        dim. ``1`` → state ``(..., d_h)`` (GRU); ``2`` → ``(..., 2, d_h)``
        with slots ``(c, h)`` (LSTM); ``4`` → ``(..., 4, d_h)`` with slots
        ``(c, n, m, h)`` (sLSTM). Defaults to ``1`` when omitted.
    jac_structure : str, optional
        Jacobian layout hint: ``'diag'``, ``'block2'``, ``'block4'``,
        ``'head'``, or ``'dense'``.

    Notes
    -----
    Optional fast paths:

    - ``step(..., wx=...)`` accepts a precomputed ``W_x(x)`` so Newton can
      reuse one GEMM across initialization and ``K`` iterations.
    - ``step_with_jacobian(h_prev, x)`` returns ``(state_new, J)`` with
      structure matching ``jac_structure``.

    See Also
    --------
    check_cell : Runtime validation of the required surface.
    pararnn.layers.para_rnn.ParaRNN : Sequence wrapper over cells.
    """

    d_h: int
    hidden_size: int

    def step(self, h_prev: Tensor, x: Tensor) -> Tensor:
        """Advance one time step.

        Parameters
        ----------
        h_prev : Tensor
            Previous state. Shape ``(..., d_h)`` for one slot, or
            ``(..., state_slots, d_h)`` for multi-slot cells.
        x : Tensor
            Input at this step. Shape ``(..., d_in)``.

        Returns
        -------
        state_new : Tensor
            Next state, same layout as ``h_prev``.
        """
        ...


def check_cell(cell: nn.Module) -> None:
    """Validate that ``cell`` exposes the ``RNNCell`` surface.

    Parameters
    ----------
    cell : nn.Module
        Module expected to implement ``hidden_size`` / ``d_h`` and
        ``step(h_prev, x)``.

    Raises
    ------
    TypeError
        When ``d_h`` / ``hidden_size`` is missing or invalid, ``step`` is
        missing, or ``state_slots`` is outside ``{1, 2, 4}``.

    See Also
    --------
    RNNCell : Protocol describing the expected attributes and methods.
    """
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
