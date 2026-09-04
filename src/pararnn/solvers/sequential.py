"""Sequential unroll — the numerical oracle for Newton+scan.

``T=1`` on CUDA (no grad) uses the decode Triton kernel: one SRAM trip for
the recurrent step. ``W_x`` is still a GEMM. Longer eval unrolls stay the
eager ``cell.step`` loop (or ``sequential_apply_compiled``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from torch import Tensor, nn

from pararnn.cells.para_lstm import ParaLSTM
from pararnn.layout import validate_cu_seqlens

_compiled_steps: dict[tuple[int, str], Callable[[Tensor, Tensor], Tensor]] = {}


def sequential_apply(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None = None,
    *,
    step: Callable[[Tensor, Tensor], Tensor] | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Unroll ``cell.step`` along time. ``x`` is (batch, time, d_in).

    ``h0`` is the paper's ``h_0`` (default 0). Output ``[:, t]`` is ``h_{t+1}``.
    Eager Python loop: correctness oracle. For a timing baseline see
    ``sequential_apply_compiled``. When ``cell`` has ``W_x``, compute it once
    over ``(B, T)`` (eq. 3.1). Custom ``step`` is unchanged.

    ``cu_seqlens`` packs sequences into ``x`` of shape ``(1, N, …)``; ``h0``
    is ``(S, …)``. Each packed span is an independent unroll.

    On CUDA, ``T=1`` with gradients disabled uses ``decode_step`` (one Triton
    launch for the recurrent step). ``W_x`` is still a GEMM. Custom ``step``
    and ``T>1`` stay on this Python loop.
    """
    batch, time, _ = x.shape
    if time < 1:
        raise ValueError(f"sequential_apply needs time >= 1, got {time}")
    if cu_seqlens is not None:
        return _sequential_ragged(cell, x, h0, step=step, cu_seqlens=cu_seqlens)
    h = h0 if h0 is not None else _zero_state(cell, batch, x)
    step_fn = step or cell.step
    wx_all = None
    if step is None and _accepts_wx(cell.step):
        lin = getattr(cell, "W_x", None)
        if lin is not None:
            wx_all = lin(x)
    if step is None and time == 1 and not torch.is_grad_enabled() and _can_decode_triton(cell, x):
        from pararnn.kernels.decode import decode_step

        wx_t = None if wx_all is None else wx_all[:, 0]
        return decode_step(cell, h, x[:, 0], wx=wx_t).unsqueeze(1)
    outs = []
    for t in range(time):
        h = step_fn(h, x[:, t]) if wx_all is None else step_fn(h, x[:, t], wx=wx_all[:, t])
        outs.append(h)
    return torch.stack(outs, dim=1)


def _sequential_ragged(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None,
    *,
    step: Callable[[Tensor, Tensor], Tensor] | None,
    cu_seqlens: Tensor,
) -> Tensor:
    if x.shape[0] != 1:
        raise ValueError(f"cu_seqlens packs x with batch=1, got batch={x.shape[0]}")
    cs = validate_cu_seqlens(cu_seqlens, x.shape[1])
    n_seq = int(cs.numel()) - 1
    if h0 is not None and h0.shape[0] != n_seq:
        raise ValueError(f"h0 batch {h0.shape[0]} != n_seq {n_seq}")
    parts = []
    for s in range(n_seq):
        t0, t1 = int(cs[s]), int(cs[s + 1])
        h0s = None if h0 is None else h0[s : s + 1]
        parts.append(sequential_apply(cell, x[:, t0:t1], h0s, step=step))
    return torch.cat(parts, dim=1)


def sequential_apply_compiled(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None = None,
    *,
    mode: str = "reduce-overhead",
) -> Tensor:
    """Same unroll with ``torch.compile`` on ``cell.step``.

    Honest sequential baseline (bottlenecks.md #12). ``cell.step`` shapes are
    static across time, so ``reduce-overhead`` can CUDA-graph the cell.
    Mode: PyTorch 2 compile tutorial (graphs for repeated small ops).
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


def _can_decode_triton(cell: nn.Module, x: Tensor) -> bool:
    from pararnn.kernels.decode import can_decode_step

    return can_decode_step(cell, x)


def _accepts_wx(fn: Callable) -> bool:
    try:
        return "wx" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
