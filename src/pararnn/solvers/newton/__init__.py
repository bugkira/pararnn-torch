"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3 (App. A) for ParaGRU/ParaLSTM. Init is eq. A.1 except ParaSLSTM, which
starts from the zero-hidden unroll. Backward is eq. 2.6 (one reverse scan).

``scan_backend='fused'`` is a handwritten Triton kernel for ParaGRU,
ParaLSTM, and ParaSLSTM ``mix='diag'``, and factorized Newton for
``ParaGRU(mix='head')`` / ``ParaSLSTM(mix='head')``. ``'auto'`` picks fused
on CUDA for those cells. ``ParaM2RNN`` uses a factorized Kronecker scan
(``newton_m2rnn_factorized``) on every backend alias; no dense ``(KV)²``.
``ParaRWKV7`` is linear in ``S``: ``newton_apply`` redirects to the
associative ``(G,U)`` monoid scan (or sequential when ``scan_backend='eager'``).
Packed ``cu_seqlens``: fused diag ParaGRU stays in-kernel; head ParaGRU
raises on explicit ``fused`` and remaps ``auto`` → ``eager``; LSTM/sLSTM
fused falls back to Triton scan with ``J=0`` at segment heads.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layout import prepend_state, prepend_state_ragged, validate_cu_seqlens
from pararnn.solvers.jacobian import step_and_jacobian
from pararnn.solvers.newton import config, dispatch, picard
from pararnn.solvers.newton.config import (
    _RESIDUAL_WARN,
    FUSED_WINDOW_DEFAULT,
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    _validate_config,
)
from pararnn.solvers.newton.dispatch import (
    _resolve_backend,
    _reverse_scan,
    _scan,
    _state_vjp_at_times,
    _t0_state_vjp,
)
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

# Pack for the duration of ``_NewtonFixedPoint.apply`` only (forward thread).
# Backward reads ``ctx`` alone — never this pack (autograd may run on another
# worker thread). Overlapping concurrent ``newton_apply`` calls are unsupported;
# no ``threading.Lock`` here (Dynamo cannot trace that context manager).
_fwd_pack: dict[str, object] = {}


class _NewtonFixedPoint(torch.autograd.Function):
    """Eq. 2.6 backward. Module-level so Dynamo does not hit ``__build_class__``.

    Non-tensor state for backward lives on ``ctx`` (legal and thread-safe).
    ``_fwd_pack`` is only read inside ``forward``, then discarded.
    """

    @staticmethod
    def forward(ctx, x_in: Tensor, h0_in: Tensor, *param_tensors: Tensor) -> Tensor:
        del param_tensors
        cell = _fwd_pack["cell"]
        config = _fwd_pack["config"]
        cu_seqlens = _fwd_pack["cu_seqlens"]
        stats = _fwd_pack["stats"]
        has_h0 = bool(_fwd_pack["has_h0"])
        block_table = _fwd_pack["block_table"]
        assert isinstance(cell, nn.Module)
        assert isinstance(config, NewtonConfig)
        h0_fwd = h0_in if has_h0 else None
        with torch.no_grad():
            states = _newton_forward(
                cell,
                x_in,
                config,
                h0=h0_fwd,
                stats=stats,  # type: ignore[arg-type]
                cu_seqlens=cu_seqlens,  # type: ignore[arg-type]
                block_table=block_table,  # type: ignore[arg-type]
            )
        # Everything backward needs — on ctx (autograd worker threads safe).
        ctx.cell = cell
        ctx.has_h0 = has_h0
        ctx.recompute = bool(config.recompute)
        # Rematerialize with the same resolved config (incl. fused). Eq. 2.6
        # reverse scan maps fused → triton (no fused reverse kernel).
        ctx.fwd_config = config
        ctx.scan_backend = "triton" if config.scan_backend == "fused" else config.scan_backend
        ctx.jacobian = config.jacobian
        ctx.jac_structure = config.jac_structure
        ctx.chunk_len = config.chunk_len
        ctx.has_cu_seqlens = cu_seqlens is not None
        saved_cs = (
            cu_seqlens if isinstance(cu_seqlens, Tensor) else x_in.new_zeros(0, dtype=torch.long)
        )
        if ctx.recompute:
            # Level 2: do not retain H* on the Function; rematerialize in backward.
            ctx.save_for_backward(x_in, h0_in, saved_cs)
        else:
            ctx.save_for_backward(states, x_in, h0_in, saved_cs)
        return states

    @staticmethod
    def backward(ctx, grad_states: Tensor):
        if ctx.recompute:
            x_in, h0_in, saved_cs = ctx.saved_tensors
            h0_fwd = h0_in if ctx.has_h0 else None
            cs_bwd = saved_cs if ctx.has_cu_seqlens else None
            with torch.no_grad():
                states = _newton_forward(
                    ctx.cell,
                    x_in,
                    ctx.fwd_config,
                    h0=h0_fwd,
                    stats=None,
                    cu_seqlens=cs_bwd,
                )
        else:
            states, x_in, h0_in, saved_cs = ctx.saved_tensors
            h0_fwd = h0_in if ctx.has_h0 else None
            cs_bwd = saved_cs if ctx.has_cu_seqlens else None
        with _newton_precision_region(x_in.device):
            if ctx.chunk_len is not None:
                grad_x, param_grads, grad_h0 = _eq26_vjp_chunked(
                    ctx.cell,
                    states,
                    x_in,
                    grad_states,
                    chunk_len=int(ctx.chunk_len),
                    backend=ctx.scan_backend,
                    jacobian=ctx.jacobian,
                    jac_structure=ctx.jac_structure,
                    h0=h0_fwd,
                )
            else:
                grad_x, param_grads, grad_h0 = _eq26_vjp(
                    ctx.cell,
                    states,
                    x_in,
                    grad_states,
                    backend=ctx.scan_backend,
                    jacobian=ctx.jacobian,
                    jac_structure=ctx.jac_structure,
                    h0=h0_fwd,
                    cu_seqlens=cs_bwd,
                )
        if not x_in.requires_grad:
            grad_x = None
        if not ctx.has_h0 or not h0_in.requires_grad:
            grad_h0 = None
        return (grad_x, grad_h0, *param_grads)


@contextlib.contextmanager
def _newton_precision_region(device: torch.device) -> Iterator[None]:
    """Keep Newton / eq. 2.6 in the tensor dtype under outer ``autocast``.

    AMP would half ``W_x(x)`` while states stay in the module dtype and break
    the VJP dtype check. Disable autocast for the solve; callers that want
    fp16/bf16 Newton put the module and ``x`` in that dtype explicitly.
    """
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        with torch.autocast(device_type="cuda", enabled=False):
            yield
        return
    if device.type == "cpu" and torch.is_autocast_enabled("cpu"):
        with torch.autocast(device_type="cpu", enabled=False):
            yield
        return
    yield


def newton_apply(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig | None = None,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Solve ``F(H)=0`` with Newton + associative scan (Danieli et al. Alg. 1).

    Inner solve is a scan on the local Jacobian monoid. With gradients,
    backward is one reverse scan (eq. 2.6). For train/eval switching use
    :class:`~pararnn.layers.ParaRNN`; call this for custom cells or benches.

    Parameters
    ----------
    cell : nn.Module
        Cell with ``step`` (prefer ``step_with_jacobian``). Fused: ParaGRU,
        ParaLSTM, ParaSLSTM ``mix='diag'|'head'``. Factorized Kronecker:
        ``ParaM2RNN`` (recipe ``max_iters=4``).
    x : Tensor of shape (batch, time, d_in)
        With ``cu_seqlens``, shape ``(1, N, d_in)``.
    config : NewtonConfig, optional
        Defaults to ``NewtonConfig()`` (``K=3``; use ``K=4`` for ``ParaM2RNN``).
    h0 : Tensor, optional
        Paper ``h_0`` (default zeros). Packed: ``(S, ...)``. With
        ``block_table``: pool ``(C, ...)``.
    stats : NewtonStats, optional
        Filled in-place with residual / backend info.
    cu_seqlens : Tensor of shape (S + 1,), optional
        Packed-batch offsets; segment heads use ``J=0``.
    block_table : Tensor of shape (batch,) or (S,), optional
        Slot ids into a paged ``h0`` pool (fused inference; no ``chunk_len``).

    Returns
    -------
    H : Tensor of shape (batch, time, *state)
        Solved trajectory (GRU ``d_h``; LSTM ``(2, d_h)``; sLSTM ``(4, d_h)``;
        M²RNN ``(K, V)``).

    Raises
    ------
    NewtonDivergenceError
        ``max |F|`` above ``config.residual_fail``.
    ValueError
        Bad ``block_table`` / backend / chunk combo.

    See Also
    --------
    sequential_apply, ParaRNN, NewtonConfig

    Notes
    -----
    Packed fused diag ParaGRU stays in-kernel; head + ``cu_seqlens`` needs
    ``eager`` (``fused`` raises; ``auto`` remaps). LSTM/sLSTM fused +
    ``cu_seqlens`` falls back to Triton scan. ``chunk_len`` windows forward
    and reverse. ``recompute=True`` rematerializes ``H*`` in backward (Level 2).
    """
    config = config or NewtonConfig()
    _validate_config(config)
    # RWKV-7 Goose is linear in S: no nonlinear fixed point. Redirect before
    # the IFT Autograd.Function so grads are ordinary BPTT through the scan.
    if getattr(cell, "jac_structure", None) == "rwkv7":
        return _newton_rwkv7(
            cell, x, config, h0=h0, stats=stats, cu_seqlens=cu_seqlens, block_table=block_table
        )
    # max_iters=None → measured K*(T) envelope (or newton_iters_by_t pin table).
    from pararnn.solvers.newton.k_star import resolve_max_iters

    config = resolve_max_iters(cell, x, config)
    assert config.max_iters is not None  # resolved: int pin or auto K*(T)
    config = _resolve_backend(cell, x, config, cu_seqlens=cu_seqlens)
    if block_table is not None:
        if config.chunk_len is not None:
            raise ValueError("block_table cannot be combined with chunk_len")
        if config.scan_backend != "fused":
            raise ValueError("block_table newton needs scan_backend='fused' (or auto on CUDA)")
    cs = None
    if cu_seqlens is not None:
        if config.scan_backend == "fused" and not isinstance(cell, ParaGRU):
            if block_table is not None:
                raise ValueError("block_table packed fused is ParaGRU only")
            config = replace(config, scan_backend="triton")
        cs = _prepare_ragged(x, h0, config, cu_seqlens, block_table=block_table)
    params = tuple(cell.parameters())
    has_h0 = h0 is not None
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or (has_h0 and h0.requires_grad) or any(p.requires_grad for p in params)
    )
    if block_table is not None and needs_grad:
        raise RuntimeError("block_table newton is inference-only")
    with _newton_precision_region(x.device):
        if not needs_grad:
            return _newton_forward(
                cell, x, config, h0=h0, stats=stats, cu_seqlens=cs, block_table=block_table
            )

        h0_leaf = h0 if has_h0 else x.new_zeros(())
        _fwd_pack.clear()
        _fwd_pack.update(
            cell=cell,
            config=config,
            cu_seqlens=cs,
            stats=stats,
            has_h0=has_h0,
            block_table=block_table,
        )
        try:
            return _NewtonFixedPoint.apply(x, h0_leaf, *params)
        finally:
            _fwd_pack.clear()


def _prepare_ragged(
    x: Tensor,
    h0: Tensor | None,
    config: NewtonConfig,
    cu_seqlens: Tensor,
    *,
    block_table: Tensor | None = None,
) -> Tensor:
    if x.shape[0] != 1:
        raise ValueError(f"cu_seqlens packs x with batch=1, got batch={x.shape[0]}")
    if config.chunk_len is not None:
        raise ValueError("chunk_len cannot be combined with cu_seqlens")
    if config.fused_time_loop:
        raise ValueError("fused_time_loop cannot be combined with cu_seqlens")
    cs = validate_cu_seqlens(cu_seqlens, x.shape[1])
    n_seq = int(cs.numel()) - 1
    if h0 is not None and block_table is None and h0.shape[0] != n_seq:
        raise ValueError(f"h0 batch {h0.shape[0]} != n_seq {n_seq}")
    if not torch.compiler.is_compiling() and log.isEnabledFor(logging.DEBUG):
        log.debug(
            "newton_ragged",
            extra={
                "n_seq": n_seq,
                "packed_len": int(x.shape[1]),
                "scan_backend": config.scan_backend,
                "requested_backend": config.scan_backend,
            },
        )
    return cs


def _newton_forward(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, stats=stats)
    if config.picard_adapt and isinstance(cell, ParaSLSTM) and not torch.compiler.is_compiling():
        return _newton_forward_picard_adapt(
            cell,
            x,
            config,
            h0=h0,
            stats=stats,
            cu_seqlens=cu_seqlens,
            block_table=block_table,
        )
    return _newton_solve(
        cell, x, config, h0=h0, stats=stats, cu_seqlens=cu_seqlens, block_table=block_table
    )


def _newton_solve(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    if getattr(cell, "jac_structure", None) == "m2rnn":
        return _newton_m2rnn(
            cell, x, config, h0=h0, stats=stats, cu_seqlens=cu_seqlens, block_table=block_table
        )
    if config.chunk_len is not None:
        return _newton_chunked(cell, x, config, h0=h0, stats=stats)
    if config.fused_time_loop and config.scan_backend != "fused":
        raise ValueError(
            "fused_time_loop requires scan_backend='fused' (or auto on CUDA ParaSLSTM)"
        )
    compiling = torch.compiler.is_compiling()
    if config.scan_backend == "fused":
        use_early = bool(config.fused_early_exit) and not compiling
        if use_early:
            states, iters_done, history, last_res = _newton_fused_early_exit(
                cell, x, config, h0=h0, cu_seqlens=cu_seqlens, block_table=block_table
            )
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=iters_done,
                stats=stats,
                residual_history=history,
                known_residual=last_res,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
            )
            return states
        states = _newton_fused(
            cell, x, config, h0=h0, cu_seqlens=cu_seqlens, block_table=block_table
        )
        if not compiling:
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=config.max_iters,
                stats=stats,
                cu_seqlens=cu_seqlens,
                block_table=block_table,
            )
        return states

    states, iters_done, history, last_res, residual_is_current = _newton_forward_pure(
        cell,
        x,
        config,
        h0=h0,
        cu_seqlens=cu_seqlens,
        early_stop=bool(config.residual_atol is not None and not compiling),
    )
    if not compiling:
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
            cu_seqlens=cu_seqlens,
        )
    return states


def _newton_forward_pure(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    early_stop: bool = False,
) -> tuple[Tensor, int, list[float], float, bool]:
    """Tensor Newton loop. No logging, no ``NewtonStats``, no fused path.

    ``early_stop=False`` (compile / ``residual_atol=None``): fixed ``max_iters``.
    ``early_stop=True``: host ``float(amax)`` + break — eager debug only.
    """
    wx = _wx_if_analytic(cell, x, config.jacobian)
    if isinstance(cell, ParaSLSTM):
        states = _slstm_newton_guess(cell, x, config, h0=h0, wx=wx, cu_seqlens=cu_seqlens)
    else:
        h_prev0 = _init_h_prev(cell, x, h0, cu_seqlens=cu_seqlens)
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
            n_h0 = int(cu_seqlens.numel()) - 1 if cu_seqlens is not None else x.shape[0]
            h0_loop = slstm_encode_log(
                x.new_zeros(n_h0, native.state_slots, native.d_h),
                eps=eps,
            )
        else:
            h0_loop = slstm_encode_log(h0, eps=eps)
        cell = SLSTMLogCoords(native)

    structure = config.jac_structure or getattr(cell, "jac_structure", None)
    iters_done = 0
    last_res = float("nan")
    history: list[float] = []
    residual_is_current = False
    atol = config.residual_atol if early_stop else None
    for it in range(config.max_iters):
        h_prev = _prepend(states, h0_loop, cu_seqlens)
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
            if last_res < atol:
                break
        delta = _scan(
            jac, residual, backend=config.scan_backend, structure=structure, cu_seqlens=cu_seqlens
        )
        if states.dtype in (torch.float16, torch.bfloat16):
            states = (states.float() + config.omega * delta.float()).to(states.dtype)
        else:
            states = states + config.omega * delta
        if config.coords == "log":
            states = slstm_clamp_log_coords(states)
        residual_is_current = False
    if config.coords == "log":
        states = slstm_decode_log(states, eps=native.eps)
        residual_is_current = False
    if not early_stop:
        iters_done = config.max_iters
    return states, iters_done, history, last_res, residual_is_current


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
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> None:
    if torch.compiler.is_compiling():
        return
    if stats is None and config.residual_fail is None:
        return
    h0_use = h0
    if block_table is not None and h0 is not None:
        h0_use = h0.index_select(0, block_table.to(device=h0.device, dtype=torch.long))
    if known_residual is None:
        pred = cell.step(_prepend(states, h0_use, cu_seqlens), x)
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
        log.error("newton_diverged", extra=extra)
        raise NewtonDivergenceError(
            f"Newton residual {res:.3e} after {iters} iters exceeds residual_fail="
            f"{cap:g} (seq_len={x.shape[1]}, picard={int(config.picard_iters or 0)}, "
            f"history={hist[-8:]!r}). For ParaSLSTM raise picard_iters."
        )
    if res > _RESIDUAL_WARN and log.isEnabledFor(logging.WARNING):
        log.warning("newton_residual_high", extra=extra)


def _newton_fused(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Alg. 1 with cell+J+scan in Triton. ``W_x(x)`` is still one PyTorch GEMM.

    Fused kernels are ``pararnn::newton_*_fused`` custom ops (``register_fake``)
    so Dynamo does not trace into Triton. Eq. 2.6 stays on ``_NewtonFixedPoint``.
    """
    wx = _input_affine(cell, x)
    if wx is None:
        raise TypeError(f"fused Newton needs cell.W_x; got {type(cell).__name__}")
    window = config.fused_window_len
    if config.fused_time_loop and window is None:
        window = FUSED_WINDOW_DEFAULT
    if not torch.compiler.is_compiling():
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
                "fused_time_loop": config.fused_time_loop,
                "fused_window_len": window,
                "block_table": block_table is not None,
                "fused_early_exit": config.fused_early_exit,
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
        fused_time_loop=config.fused_time_loop,
        fused_window_len=window,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


def _newton_fused_early_exit(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> tuple[Tensor, int, list[float], float]:
    """Experimental fused path: host ``max|F|`` after each Newton step.

    Bypasses the fixed-K custom op. Variable ``iters`` — train/DDP/compile
    should keep ``fused_early_exit=False``.
    """
    wx = _input_affine(cell, x)
    if wx is None:
        raise TypeError(f"fused Newton needs cell.W_x; got {type(cell).__name__}")
    atol = config.residual_atol
    if atol is None:
        raise ValueError("fused_early_exit requires residual_atol")
    history: list[float] = []
    h0_use = h0
    if block_table is not None and h0 is not None:
        h0_use = h0.index_select(0, block_table.to(device=h0.device, dtype=torch.long))

    def residual_fn(states: Tensor) -> float:
        pred = cell.step(_prepend(states, h0_use, cu_seqlens), x, wx=wx)
        res = float((pred - states).detach().abs().amax())
        history.append(res)
        return res

    iters_done_out: list[int] = []
    from pararnn.kernels.fused_newton import fused_newton

    window = config.fused_window_len
    if config.fused_time_loop and window is None:
        window = FUSED_WINDOW_DEFAULT
    log.info(
        "newton_fused_early_exit",
        extra={
            "cell": type(cell).__name__,
            "seq_len": int(x.shape[1]),
            "batch": int(x.shape[0]),
            "max_iters": config.max_iters,
            "residual_atol": float(atol),
        },
    )
    states = fused_newton(
        cell,
        wx,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        log_coords=config.coords == "log",
        picard_iters=int(config.picard_iters or 0),
        scan_tile=config.scan_tile,
        fused_time_loop=config.fused_time_loop,
        fused_window_len=window,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
        early_exit_atol=float(atol),
        residual_fn=residual_fn,
        iters_done_out=iters_done_out,
    )
    iters_done = int(iters_done_out[0]) if iters_done_out else int(config.max_iters)
    last_res = history[-1] if history else float("nan")
    return states, iters_done, history, last_res


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
    cu_seqlens: Tensor | None = None,
) -> tuple[Tensor | None, tuple[Tensor | None, ...], Tensor | None]:
    """``∇_x L``, per-parameter grads, and ``∇_{h0} L`` (eq. 2.6 + cell VJP).

    ``∇_{h0} L = J_0^T μ_0``. Reverse scan over ``H`` starts at t=0; ``J_0``
    is the h0 adjoint. Ragged: one adjoint per sequence start.
    """
    h_prev = _prepend(states, h0, cu_seqlens)
    if getattr(cell, "jac_structure", None) == "m2rnn" and cu_seqlens is None:
        from pararnn.kernels.newton_m2rnn import m2rnn_t0_vjp, reverse_factor_scan_m2rnn

        wx = _input_affine(cell, x)
        if wx is None:
            raise TypeError("ParaM2RNN factorized VJP needs cell.W_x")
        with torch.no_grad():
            mu = reverse_factor_scan_m2rnn(cell, h_prev, partial, wx=wx)
            h0_vjp = m2rnn_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads = cell_vjp(cell, h_prev, x, mu, packed=packed)
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0

    # Factorized reverse for ParaGRU / ParaSLSTM head on CUDA (matches fused forward).
    if (
        isinstance(cell, ParaGRU)
        and cell.mix == "head"
        and backend in ("triton", "fused")
        and cu_seqlens is None
        and jacobian in ("auto", "analytic")
    ):
        wx = _input_affine(cell, x)
        if wx is None:
            raise TypeError("ParaGRU head factorized VJP needs cell.W_x")
        from pararnn.kernels.newton_gru_head import (
            gru_head_t0_vjp,
            reverse_factor_scan_gru_head,
        )

        with torch.no_grad():
            mu = reverse_factor_scan_gru_head(cell, h_prev, partial, wx=wx)
            h0_vjp = gru_head_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads = cell_vjp(cell, h_prev, x, mu, packed=packed)
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0

    if (
        isinstance(cell, ParaSLSTM)
        and cell.mix == "head"
        and backend in ("triton", "fused")
        and cu_seqlens is None
        and jacobian in ("auto", "analytic")
    ):
        wx = _input_affine(cell, x)
        if wx is None:
            raise TypeError("ParaSLSTM head factorized VJP needs cell.W_x")
        from pararnn.kernels.newton_slstm_head import (
            reverse_factor_scan_slstm_head,
            slstm_head_t0_vjp,
        )

        with torch.no_grad():
            mu = reverse_factor_scan_slstm_head(cell, h_prev, partial, wx=wx)
            h0_vjp = slstm_head_t0_vjp(cell, h_prev, mu, wx=wx)
        packed = uses_packed_vjp(cell)
        grad_x, param_grads = cell_vjp(cell, h_prev, x, mu, packed=packed)
        grad_h0 = None if h0 is None else h0_vjp
        return grad_x, param_grads, grad_h0

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
        mu = _reverse_scan(
            jac, partial, backend=backend, structure=structure, cu_seqlens=cu_seqlens
        )
    packed = uses_packed_vjp(cell)
    grad_x, param_grads = cell_vjp(cell, h_prev, x, mu, packed=packed)
    if cu_seqlens is None:
        h0_vjp = _t0_state_vjp(jac, mu)
    else:
        starts = validate_cu_seqlens(cu_seqlens, states.shape[1])[:-1]
        h0_vjp = _state_vjp_at_times(jac, mu, starts.to(device=jac.device))[0]
    grad_h0 = None if h0 is None else h0_vjp
    return grad_x, param_grads, grad_h0


def _eq26_vjp_chunked(
    cell: nn.Module,
    states: Tensor,
    x: Tensor,
    partial: Tensor,
    *,
    chunk_len: int,
    backend: str = "eager",
    jacobian: str = "auto",
    jac_structure: str | None = None,
    h0: Tensor | None = None,
) -> tuple[Tensor | None, tuple[Tensor | None, ...], Tensor | None]:
    """Eq. 2.6 on the same windows as ``_newton_chunked``.

    Reverse order: ``∇_{h0}`` of window ``i+1`` adds to ``∂L/∂S`` at the last
    step of window ``i`` (that step is the next window's ``h0``).
    """
    length = int(chunk_len)
    time = int(x.shape[1])
    n_params = len(tuple(cell.parameters()))
    acc_x: Tensor | None = None
    acc_params: list[Tensor | None] = [None] * n_params
    carry_h0 = None
    starts = list(range(0, time, length))
    for t0 in reversed(starts):
        t1 = min(t0 + length, time)
        piece = states[:, t0:t1]
        xw = x[:, t0:t1]
        partial_w = partial[:, t0:t1]
        if carry_h0 is not None:
            partial_w = partial_w.clone()
            partial_w[:, -1] = partial_w[:, -1] + carry_h0
        h0_w = h0 if t0 == 0 else states[:, t0 - 1]
        gx, pgrads, gh0 = _eq26_vjp(
            cell,
            piece,
            xw,
            partial_w,
            backend=backend,
            jacobian=jacobian,
            jac_structure=jac_structure,
            h0=h0_w,
        )
        carry_h0 = gh0
        if gx is not None:
            if acc_x is None:
                acc_x = x.new_zeros(x.shape)
            acc_x[:, t0:t1] = gx
        for i, g in enumerate(pgrads):
            if g is None:
                continue
            acc_params[i] = g if acc_params[i] is None else acc_params[i] + g
    grad_h0 = None if h0 is None else carry_h0
    return acc_x, tuple(acc_params), grad_h0


def _init_h_prev(
    cell: nn.Module,
    x: Tensor,
    h0: Tensor | None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """App. A: ``H^0_t = f(h_{t-1}, x_t)`` in parallel; only starts see ``h0``."""
    h_prev0 = _zero_state_like_input(cell, x)
    if cu_seqlens is None:
        if h0 is not None:
            h_prev0[:, 0] = h0
        return h_prev0
    starts = validate_cu_seqlens(cu_seqlens, x.shape[1])[:-1]
    if h0 is not None:
        h_prev0[0, starts] = h0
    return h_prev0


def _prepend(states: Tensor, h0: Tensor | None, cu_seqlens: Tensor | None) -> Tensor:
    if cu_seqlens is None:
        return prepend_state(states, h0)
    return prepend_state_ragged(states, h0, cu_seqlens)


def _wx_if_analytic(cell: nn.Module, x: Tensor, jacobian: str) -> Tensor | None:
    """Reuse ``W_x(x)`` only on the analytic-J path (eq. 3.1)."""
    mode = jacobian
    if mode == "auto":
        mode = "analytic" if hasattr(cell, "step_with_jacobian") else "autograd"
    if mode != "analytic":
        return None
    return _input_affine(cell, x)


def _input_affine(cell: nn.Module, x: Tensor) -> Tensor | None:
    """Input affine for Newton; ``project_wx`` when present (CfC packs Δt)."""
    project = getattr(cell, "project_wx", None)
    if callable(project):
        return project(x)
    lin = getattr(cell, "W_x", None)
    if lin is None:
        return None
    return lin(x)


def _zero_state_like_input(cell: nn.Module, x: Tensor) -> Tensor:
    """Zeros with a time axis, matching ``step``'s previous-state layout."""
    batch, time, _ = x.shape
    tail = getattr(cell, "state_shape", None)
    if tail is not None:
        return x.new_zeros(batch, time, *tuple(tail))
    d_h = cell.d_h
    slots = getattr(cell, "state_slots", None)
    if slots is None:
        slots = 2 if isinstance(cell, ParaLSTM) else 1
    if slots == 1:
        return x.new_zeros(batch, time, d_h)
    return x.new_zeros(batch, time, slots, d_h)


def _newton_m2rnn(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Factorized Kronecker Newton for ``ParaM2RNN`` (no dense ``(KV)²``)."""
    from pararnn.cells.para_m2rnn import ParaM2RNN
    from pararnn.kernels.newton_m2rnn import newton_m2rnn_factorized

    if not isinstance(cell, ParaM2RNN):
        raise TypeError(f"_newton_m2rnn needs ParaM2RNN, got {type(cell).__name__}")
    if cu_seqlens is not None or block_table is not None:
        raise ValueError(
            "ParaM2RNN factorized Newton is rectangular-batch only "
            "(no cu_seqlens / block_table yet)"
        )
    if config.chunk_len is not None:
        raise ValueError("ParaM2RNN does not support chunk_len yet")
    if config.coords == "log":
        raise TypeError("NewtonConfig(coords='log') is ParaSLSTM only")
    # picard_iters>=1 → frozen-W (W=0) warm-start; see m2rnn_frozen_w_scan.
    frozen_w = int(config.picard_iters or 0) > 0

    compiling = torch.compiler.is_compiling()
    history: list[float] = []
    force_eager = config.scan_backend == "eager"
    backend_tag = "fused" if config.scan_backend == "fused" else "factor_m2rnn"
    # Default residual_atol early-stops past K*; fixed over-provisioned K loses
    # to sequential on large KV (wall ≈ K_used × scan).
    states = newton_m2rnn_factorized(
        cell,
        x,
        max_iters=config.max_iters,
        omega=config.omega,
        h0=h0,
        residual_history=None if (compiling or not force_eager) else history,
        force_eager=force_eager,
        residual_atol=None if compiling else config.residual_atol,
        frozen_w_init=frozen_w,
    )
    if not compiling:
        cfg = replace(config, scan_backend=backend_tag)
        last = history[-1] if history else None
        _fill_stats(
            cell,
            x,
            states,
            h0,
            cfg,
            iters=config.max_iters,
            stats=stats,
            residual_history=history,
            known_residual=last,
        )
    return states


def _newton_rwkv7(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
    stats: NewtonStats | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
) -> Tensor:
    """Linear RWKV-7 monoid: associative scan (default) or sequential oracle.

    The map is affine in ``S``. ``scan_backend='eager'`` uses
    ``sequential_apply``; other aliases use ``rwkv7_associative_scan``.
    """
    from pararnn.cells.para_rwkv7 import ParaRWKV7
    from pararnn.kernels.rwkv7_scan import (
        rwkv7_associative_scan,
        rwkv7_build_g,
        rwkv7_outer_vk,
    )
    from pararnn.solvers.sequential import sequential_apply

    if not isinstance(cell, ParaRWKV7):
        raise TypeError(f"_newton_rwkv7 needs ParaRWKV7, got {type(cell).__name__}")
    if cu_seqlens is not None or block_table is not None:
        raise ValueError(
            "ParaRWKV7 scan is rectangular-batch only (no cu_seqlens / block_table yet)"
        )
    if config.chunk_len is not None:
        raise ValueError("ParaRWKV7 does not support chunk_len yet")
    if config.coords == "log":
        raise TypeError("NewtonConfig(coords='log') is ParaSLSTM only")

    use_seq = config.scan_backend == "eager"
    if use_seq:
        states = sequential_apply(cell, x, h0)
        backend_tag = "rwkv7_sequential"
    else:
        w, a, kappa, v, k, _r = cell.project(x)
        g = rwkv7_build_g(w, a, kappa)
        u = rwkv7_outer_vk(v, k)
        states = rwkv7_associative_scan(g, u, s0=h0)
        backend_tag = "rwkv7_scan"

    if not torch.compiler.is_compiling():
        if log.isEnabledFor(logging.INFO):
            log.info(
                "rwkv7_linear_redirect",
                extra={
                    "backend": backend_tag,
                    "seq_len": int(x.shape[1]),
                    "batch": int(x.shape[0]),
                    "n_heads": cell.n_heads,
                    "d_head": cell.d_head,
                },
            )
        cfg = replace(config, scan_backend=backend_tag)
        _fill_stats(
            cell,
            x,
            states,
            h0,
            cfg,
            iters=0,
            stats=stats,
            residual_history=[],
            known_residual=0.0,
        )
    return states
