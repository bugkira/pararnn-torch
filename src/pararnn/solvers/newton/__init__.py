"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3 (App. A) for ParaGRU/ParaLSTM. Init is eq. A.1 except ParaSLSTM, which
starts from the zero-hidden unroll. Backward is eq. 2.6 (one reverse scan).

``scan_backend='fused'`` is a handwritten Triton kernel for ParaGRU, ParaLSTM,
and ParaSLSTM ``mix='diag'``. ``'auto'`` picks fused on CUDA for those cells.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layout import prepend_state
from pararnn.solvers.jacobian import step_and_jacobian
from pararnn.solvers.newton import config, dispatch, picard
from pararnn.solvers.newton.config import (
    _RESIDUAL_WARN,
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    _validate_config,
)
from pararnn.solvers.newton.dispatch import _resolve_backend, _reverse_scan, _scan, _t0_state_vjp
from pararnn.solvers.newton.picard import _newton_forward_picard_adapt, _slstm_newton_guess
from pararnn.solvers.slstm_log import (
    SLSTMLogCoords,
    slstm_clamp_log_coords,
    slstm_decode_log,
    slstm_encode_log,
)
from pararnn.solvers.vjp import cell_vjp, uses_packed_vjp

LIBRARY_NEWTON_ITERS = config.LIBRARY_NEWTON_ITERS
_DEFAULT_RESIDUAL_ATOL = config._DEFAULT_RESIDUAL_ATOL
_DEFAULT_RESIDUAL_FAIL = config._DEFAULT_RESIDUAL_FAIL
_PICARD_RUNGS = picard._PICARD_RUNGS
slstm_auto_picard = picard.slstm_auto_picard
slstm_picard_next = picard.slstm_picard_next
_resolve_picard = picard._resolve_picard
_picard_retry_needed = picard._picard_retry_needed
_can_triton_scan = dispatch._can_triton_scan
_can_fuse = dispatch._can_fuse
_pick_auto = dispatch._pick_auto
_fused_error = dispatch._fused_error
_scan_named = dispatch._scan_named
_reverse_scan_named = dispatch._reverse_scan_named
_scan_infer = dispatch._scan_infer
_reverse_scan_infer = dispatch._reverse_scan_infer
_dense_slot_pack = dispatch._dense_slot_pack
_head_slot_pack = dispatch._head_slot_pack
_head_slot_unpack = dispatch._head_slot_unpack

log = logging.getLogger(__name__)


def newton_apply(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig | None = None,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
) -> Tensor:
    """Parallel forward: Newton on F(H)=0, inner solve via associative scan.

    ``h0`` is the paper's ``h_0`` (default 0). Fused kernels prepend ``h0``.

    If gradients are enabled, the backward is eq. 2.6 (one reverse scan).
    """
    config = config or NewtonConfig()
    _validate_config(config)
    config = _resolve_backend(cell, x, config)
    params = tuple(cell.parameters())
    has_h0 = h0 is not None
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or (has_h0 and h0.requires_grad) or any(p.requires_grad for p in params)
    )
    if not needs_grad:
        return _newton_forward(cell, x, config, h0=h0, stats=stats)

    # Nested Autograd.Function.apply takes tensors. cell/config/stats close
    # over this frame; a module-level class storing them is racy under
    # concurrent newton_apply. h0 is an apply() input so ∇_{h0} L = J_0^T μ_0
    # (eq. 2.6).
    h0_leaf = h0 if has_h0 else x.new_zeros(())

    class _NewtonFixedPoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_in: Tensor, h0_in: Tensor, *param_tensors: Tensor) -> Tensor:
            del param_tensors
            h0_fwd = h0_in if has_h0 else None
            with torch.no_grad():
                states = _newton_forward(cell, x_in, config, h0=h0_fwd, stats=stats)
            ctx.has_h0 = has_h0
            ctx.scan_backend = "triton" if config.scan_backend == "fused" else config.scan_backend
            ctx.jacobian = config.jacobian
            ctx.jac_structure = config.jac_structure
            ctx.save_for_backward(states, x_in, h0_in)
            return states

        @staticmethod
        def backward(ctx, grad_states: Tensor):
            states, x_in, h0_in = ctx.saved_tensors
            h0_fwd = h0_in if ctx.has_h0 else None
            grad_x, param_grads, grad_h0 = _eq26_vjp(
                cell,
                states,
                x_in,
                grad_states,
                backend=ctx.scan_backend,
                jacobian=ctx.jacobian,
                jac_structure=ctx.jac_structure,
                h0=h0_fwd,
            )
            if not x_in.requires_grad:
                grad_x = None
            if not ctx.has_h0 or not h0_in.requires_grad:
                grad_h0 = None
            return (grad_x, grad_h0, *param_grads)

    return _NewtonFixedPoint.apply(x, h0_leaf, *params)


def _newton_forward(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
) -> Tensor:
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, stats=stats)
    if config.picard_adapt and isinstance(cell, ParaSLSTM) and not torch.compiler.is_compiling():
        return _newton_forward_picard_adapt(cell, x, config, h0=h0, stats=stats)
    return _newton_solve(cell, x, config, h0=h0, stats=stats)


def _newton_solve(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
) -> Tensor:
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, stats=stats)
    if config.scan_backend == "fused":
        states = _newton_fused(cell, x, config, h0=h0)
        # Fused kernels run exactly max_iters (no residual early-stop).
        _fill_stats(cell, x, states, h0, config, iters=config.max_iters, stats=stats)
        return states
    wx = _wx_if_analytic(cell, x, config.jacobian)
    if isinstance(cell, ParaSLSTM):
        states = _slstm_newton_guess(cell, x, config, h0=h0, wx=wx)
    else:
        h_prev0 = _init_h_prev(cell, x, h0)
        states, _ = step_and_jacobian(
            cell,
            h_prev0,
            x,
            wx=wx,
            jacobian=config.jacobian,
            jac_structure=config.jac_structure,
        )

    native = cell
    h0_loop = h0
    if config.coords == "log":
        eps = native.eps
        states = slstm_encode_log(states, eps=eps)
        if h0 is None:
            h0_loop = slstm_encode_log(
                x.new_zeros(x.shape[0], native.state_slots, native.d_h),
                eps=eps,
            )
        else:
            h0_loop = slstm_encode_log(h0, eps=eps)
        cell = SLSTMLogCoords(native)
        if not torch.compiler.is_compiling():
            log.debug(
                "newton_slstm_log_coords",
                extra={"seq_len": x.shape[1], "batch": x.shape[0], "d_h": native.d_h},
            )

    structure = config.jac_structure or getattr(cell, "jac_structure", None)
    iters_done = 0
    last_res = float("nan")
    history: list[float] = []
    residual_is_current = False
    atol = config.residual_atol
    for it in range(config.max_iters):
        h_prev = prepend_state(states, h0_loop)
        pred, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=config.jacobian,
            jac_structure=config.jac_structure,
        )
        residual = pred - states
        iters_done = it + 1
        if atol is not None:
            last_res = float(residual.detach().abs().amax())
            history.append(last_res)
            residual_is_current = True
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "newton_iter",
                extra={
                    "iter": it,
                    "max_residual": last_res,
                    "seq_len": x.shape[1],
                    "batch": x.shape[0],
                    "jacobian": config.jacobian,
                    "scan_backend": config.scan_backend,
                    "coords": config.coords,
                },
            )
        if atol is not None and last_res < atol:
            if log.isEnabledFor(logging.INFO) and not torch.compiler.is_compiling():
                log.info(
                    "newton_early_stop",
                    extra={
                        "iters": iters_done,
                        "max_residual": last_res,
                        "atol": atol,
                        "seq_len": x.shape[1],
                        "coords": config.coords,
                    },
                )
            break
        delta = _scan(jac, residual, backend=config.scan_backend, structure=structure)
        if states.dtype == torch.float16:
            states = (states.float() + config.omega * delta.float()).to(states.dtype)
        else:
            states = states + config.omega * delta
        if config.coords == "log":
            states = slstm_clamp_log_coords(states)
        residual_is_current = False
    if config.coords == "log":
        states = slstm_decode_log(states, eps=native.eps)
        cell = native
        # last_res was in LSE coords; fail-loud / stats need native F.
        residual_is_current = False
    _fill_stats(
        cell,
        x,
        states,
        h0,
        config,
        iters=iters_done,
        stats=stats,
        residual_history=history,
        known_residual=last_res if residual_is_current else None,
    )
    return states


def _newton_chunked(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    stats: NewtonStats | None,
) -> Tensor:
    """Newton on windows of ``chunk_len``; last state of a chunk is the next ``h0``.

    Span is linear in the number of chunks. Each window is the usual Alg. 1
    (eager / triton / fused). ``chunk_len=64`` from T=64 K=3 at ``d_h=256``.
    """
    length = int(config.chunk_len)
    inner = replace(config, chunk_len=None)
    parts: list[Tensor] = []
    carry = h0
    iters_done = 0
    for t0 in range(0, x.shape[1], length):
        chunk_stats = NewtonStats() if stats is not None else None
        piece = _newton_forward(cell, x[:, t0 : t0 + length], inner, h0=carry, stats=chunk_stats)
        parts.append(piece)
        carry = piece[:, -1]
        if chunk_stats is not None:
            iters_done += chunk_stats.iters
    states = torch.cat(parts, dim=1)
    _fill_stats(cell, x, states, h0, config, iters=iters_done, stats=stats)
    return states


def _fill_stats(
    cell: nn.Module,
    x: Tensor,
    states: Tensor,
    h0: Tensor | None,
    config: NewtonConfig,
    *,
    iters: int,
    stats: NewtonStats | None,
    residual_history: list[float] | tuple[float, ...] = (),
    known_residual: float | None = None,
) -> None:
    if stats is None and config.residual_fail is None:
        return
    if known_residual is None:
        pred = cell.step(prepend_state(states, h0), x)
        res = float((pred - states).detach().abs().amax())
    else:
        res = known_residual
    hist = tuple(residual_history)
    if stats is not None:
        stats.max_residual = res
        stats.iters = iters
        stats.scan_backend = config.scan_backend
        stats.picard_iters = int(config.picard_iters or 0)
        stats.residual_history = hist
    cap = config.residual_fail
    extra = {
        "max_residual": res,
        "residual_fail": cap,
        "iters": iters,
        "seq_len": int(x.shape[1]),
        "batch": int(x.shape[0]),
        "cell": type(cell).__name__,
        "picard_iters": int(config.picard_iters or 0),
        "scan_backend": config.scan_backend,
        "residual_history": hist[-8:],
    }
    diverged = not math.isfinite(res) or (cap is not None and res > cap)
    if diverged and cap is not None:
        if not torch.compiler.is_compiling():
            log.error("newton_diverged", extra=extra)
        raise NewtonDivergenceError(
            f"Newton residual {res:.3e} after {iters} iters exceeds residual_fail="
            f"{cap:g} (seq_len={x.shape[1]}, picard={int(config.picard_iters or 0)}, "
            f"history={hist[-8:]!r}). For ParaSLSTM raise picard_iters."
        )
    if (
        res > _RESIDUAL_WARN
        and log.isEnabledFor(logging.WARNING)
        and not torch.compiler.is_compiling()
    ):
        log.warning("newton_residual_high", extra=extra)


def _newton_fused(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
) -> Tensor:
    """Alg. 1 with cell+J+scan in Triton. ``W_x(x)`` is still one PyTorch GEMM."""
    wx = _input_affine(cell, x)
    if wx is None:
        raise TypeError(f"fused Newton needs cell.W_x; got {type(cell).__name__}")
    log.debug(
        "newton_fused",
        extra={
            "cell": type(cell).__name__,
            "seq_len": x.shape[1],
            "batch": x.shape[0],
            "d_h": cell.d_h,
            "max_iters": config.max_iters,
            "device": str(x.device),
            "h0": h0 is not None,
            "picard_iters": int(config.picard_iters or 0),
            "scan_tile": config.scan_tile,
        },
    )
    from pararnn.kernels.fused_newton import fused_newton

    return fused_newton(
        cell,
        wx,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        log_coords=config.coords == "log",
        picard_iters=int(config.picard_iters or 0),
        scan_tile=config.scan_tile,
    )


def _eq26_vjp(
    cell: nn.Module,
    states: Tensor,
    x: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    jacobian: str = "auto",
    jac_structure: str | None = None,
    h0: Tensor | None = None,
) -> tuple[Tensor | None, tuple[Tensor | None, ...], Tensor | None]:
    """``∇_x L``, per-parameter grads, and ``∇_{h0} L`` (eq. 2.6 + cell VJP).

    ``∇_{h0} L = J_0^T μ_0``. Reverse scan over ``H`` starts at t=0; ``J_0``
    is the h0 adjoint.
    """
    h_prev = prepend_state(states, h0)
    wx = _wx_if_analytic(cell, x, jacobian)
    structure = jac_structure or getattr(cell, "jac_structure", None)
    with torch.no_grad():
        _, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=jacobian,
            jac_structure=jac_structure,
        )
        mu = _reverse_scan(jac, partial, backend=backend, structure=structure)
    packed = uses_packed_vjp(cell)
    grad_x, param_grads = cell_vjp(cell, h_prev, x, mu, packed=packed)
    grad_h0 = None if h0 is None else _t0_state_vjp(jac, mu)
    return grad_x, param_grads, grad_h0


def _init_h_prev(cell: nn.Module, x: Tensor, h0: Tensor | None) -> Tensor:
    """App. A: ``H^0_t = f(h_{t-1}, x_t)`` in parallel; only t=0 sees ``h0``."""
    h_prev0 = _zero_state_like_input(cell, x)
    if h0 is not None:
        h_prev0[:, 0] = h0
    return h_prev0


def _wx_if_analytic(cell: nn.Module, x: Tensor, jacobian: str) -> Tensor | None:
    """Reuse ``W_x(x)`` only on the analytic-J path (eq. 3.1)."""
    mode = jacobian
    if mode == "auto":
        mode = "analytic" if hasattr(cell, "step_with_jacobian") else "autograd"
    if mode != "analytic":
        return None
    return _input_affine(cell, x)


def _input_affine(cell: nn.Module, x: Tensor) -> Tensor | None:
    """``W_x(x)`` when the cell has a dense input map; else ``None`` (compute inside step)."""
    lin = getattr(cell, "W_x", None)
    if lin is None:
        return None
    return lin(x)


def _zero_state_like_input(cell: nn.Module, x: Tensor) -> Tensor:
    """Zeros with a time axis, matching ``step``'s previous-state layout."""
    batch, time, _ = x.shape
    d_h = cell.d_h
    slots = getattr(cell, "state_slots", None)
    if slots is None:
        slots = 2 if isinstance(cell, ParaLSTM) else 1
    if slots == 1:
        return x.new_zeros(batch, time, d_h)
    return x.new_zeros(batch, time, slots, d_h)
