"""Sequential unroll — the numerical oracle for Newton+scan."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from torch import Tensor, nn

from pararnn.cells.para_lstm import ParaLSTM

_compiled_steps: dict[tuple[int, str], Callable[[Tensor, Tensor], Tensor]] = {}


def sequential_apply(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None = None,
    *,
    step: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Unroll ``cell.step`` along time. ``x`` is (batch, time, d_in).

    ``h0`` is the paper's ``h_0`` (default 0). Output ``[:, t]`` is ``h_{t+1}``.
    Eager Python loop: correctness oracle. For a timing baseline see
    ``sequential_apply_compiled``. When ``cell`` has ``W_x``, compute it once
    over ``(B, T)`` (eq. 3.1) instead of ``T`` GEMVs. Custom ``step`` is
    unchanged (no ``wx=``).
    """
    batch, time, _ = x.shape
    if time < 1:
        raise ValueError(f"sequential_apply needs time >= 1, got {time}")
    if h0 is None:
        h = _zero_state(cell, batch, x)
    else:
        h = h0
    step_fn = step or cell.step
    wx_all = None
    if step is None and _accepts_wx(cell.step):
        lin = getattr(cell, "W_x", None)
        if lin is not None:
            wx_all = lin(x)
    outs = []
    for t in range(time):
        if wx_all is None:
            h = step_fn(h, x[:, t])
        else:
            h = step_fn(h, x[:, t], wx=wx_all[:, t])
        outs.append(h)
    return torch.stack(outs, dim=1)


def sequential_apply_compiled(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None = None,
    *,
    mode: str = "reduce-overhead",
) -> Tensor:
    """Same unroll with ``torch.compile`` on ``cell.step``.

    Honest sequential baseline (bottlenecks.md #12). ``cell.step`` shapes are
    static across time, so ``reduce-overhead`` can CUDA-graph the cell. Does
    not compile Newton. Mode: PyTorch 2 compile tutorial (graphs for repeated
    small ops), not a paper hyperparameter.
    """
    batch, time, _ = x.shape
    if time < 1:
        raise ValueError(f"sequential_apply_compiled needs time >= 1, got {time}")
    key = (id(cell), mode)
    compiled = _compiled_steps.get(key)
    if compiled is None:

        def _step(h: Tensor, xt: Tensor, c: nn.Module = cell) -> Tensor:
            return c.step(h, xt)

        compiled = torch.compile(_step, mode=mode)
        _compiled_steps[key] = compiled
    h = h0 if h0 is not None else _zero_state(cell, batch, x)
    outs = []
    for t in range(time):
        # reduce-overhead CUDA-graphs reuse output storage; the next step
        # would otherwise read a tensor that the subsequent capture overwrote.
        torch.compiler.cudagraph_mark_step_begin()
        h = compiled(h, x[:, t]).clone()
        outs.append(h)
    return torch.stack(outs, dim=1)


def _zero_state(cell: nn.Module, batch: int, ref: Tensor) -> Tensor:
    d_h = cell.d_h
    slots = getattr(cell, "state_slots", None)
    if slots is None:
        slots = 2 if isinstance(cell, ParaLSTM) else 1
    if slots == 1:
        return ref.new_zeros(batch, d_h)
    return ref.new_zeros(batch, slots, d_h)


def _accepts_wx(fn: Callable) -> bool:
    try:
        return "wx" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
