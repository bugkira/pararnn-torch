"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3 (App. A) for ParaGRU/ParaLSTM. Init is eq. A.1 except ParaSLSTM, which
starts from the zero-hidden unroll. Backward is eq. 2.6 (one reverse scan).

``scan_backend='fused'`` is a handwritten Triton kernel for ParaGRU, ParaLSTM,
and ParaSLSTM ``mix='diag'``. ``'auto'`` picks fused on CUDA for those cells.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.layout import prepend_state, slstm_pack_heads, slstm_unpack_heads
from pararnn.solvers.jacobian import step_and_jacobian
from pararnn.solvers.scan import (
    reverse_scan_block2,
    reverse_scan_block4,
    reverse_scan_dense,
    reverse_scan_diag,
    scan_block2,
    scan_block4,
    scan_dense,
    scan_diag,
)
from pararnn.solvers.slstm_log import (
    SLSTMLogCoords,
    slstm_clamp_log_coords,
    slstm_decode_log,
    slstm_encode_log,
)
from pararnn.solvers.slstm_picard import slstm_picard_init, slstm_zero_hidden_init
from pararnn.solvers.vjp import cell_vjp, uses_packed_vjp

log = logging.getLogger(__name__)

# App. A: K=3 reaches machine precision on these cells. Sequential agreement
# tests use 1e-4. Stop a wasted extra iter below that and above fp32 noise.
_DEFAULT_RESIDUAL_ATOL = 1e-5
# Library contract: K=3 (App. A). sLSTM basin: raise picard_iters (para-slstm.md).
LIBRARY_NEWTON_ITERS = 3
# After K steps, max|F| above this is divergence.
# Sequential agreement is 1e-4…2e-3; diverged sLSTM is 1e2…1e14 (para-slstm.md).
# 1.0 sits between. None disables (K-curves, P=0 timing benches).
_DEFAULT_RESIDUAL_FAIL = 1.0
# Warn when max|F| is past sequential-agreement but under residual_fail.
# Train can miss the P=1 basin on one batch after Adam (para-slstm.md).
_RESIDUAL_WARN = 1e-3
# Auto Picard rungs (slstm_auto_picard). Train adapt climbs picard_iters.
_PICARD_RUNGS = (1, 3, 5)


class NewtonDivergenceError(RuntimeError):
    """Newton residual exceeded residual_fail. For ParaSLSTM raise picard_iters."""


@dataclass
class NewtonStats:
    """Filled by ``newton_apply(..., stats=)`` after the forward."""

    max_residual: float = float("nan")
    # Residual evaluations in the Newton loop (≤ max_iters), including the
    # eval that triggered early-stop. 0 if max_iters=0. Fused: this is max_iters.
    iters: int = 0
    scan_backend: str = ""
    picard_iters: int = 0
    residual_history: tuple[float, ...] = ()


@dataclass
class NewtonConfig:
    # App. A: K=3. ParaSLSTM at long T uses Picard (picard_iters).
    max_iters: int = LIBRARY_NEWTON_ITERS
    omega: float = 1.0  # 1 = vanilla Newton; <1 damps (Gonzalez et al. ELK)
    # auto: fused CUDA GRU/LSTM/sLSTM-diag; else Triton scan + step; else eager Blelloch.
    scan_backend: str = "auto"
    # auto: analytic J if step_with_jacobian else Autograd. analytic: require it.
    jacobian: str = "auto"
    # None infers from state / cell.jac_structure. diag | block2 | block4 | head | dense.
    jac_structure: str | None = None
    # None: run all K. Default 1e-5 skips leftover K when max|F| is already small (App. A).
    residual_atol: float | None = _DEFAULT_RESIDUAL_ATOL
    # After last K, raise if max|F| exceeds this. Default 1.0. None: K-curves / benches.
    residual_fail: float | None = _DEFAULT_RESIDUAL_FAIL
    # native | log. log: ParaSLSTM LSE cell in (u, log n, m, h).
    coords: str = "native"
    # None = one Newton over T. int: sequential chunks; 64 from T=64 K=3 at d_h=256.
    chunk_len: int | None = None
    # None = auto P ∈ {1, 3, 5} from T for ParaSLSTM. Explicit 0 is zero-hidden.
    picard_iters: int | None = None
    # None: retry P when picard_iters was auto. True/False force. Better initial guess.
    picard_adapt: bool | None = None
    # Retry next P if max|F| exceeds this (1e-3 = sequential-agreement band). None: only residual_fail.
    picard_retry_atol: float | None = _RESIDUAL_WARN
    # assoc: tl.associative_scan. seq: serial prefix in the tile (ablation).
    scan_tile: str = "assoc"


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
    if config.scan_backend not in ("auto", "eager", "triton", "fused"):
        raise ValueError(f"unknown scan backend {config.scan_backend!r}")
    if config.jacobian not in ("auto", "analytic", "autograd"):
        raise ValueError(f"unknown jacobian {config.jacobian!r}")
    if config.coords not in ("native", "log"):
        raise ValueError(f"unknown newton coords {config.coords!r}")
    if config.chunk_len is not None and int(config.chunk_len) < 1:
        raise ValueError(f"chunk_len must be >= 1, got {config.chunk_len!r}")
    if config.max_iters < 0:
        raise ValueError(f"max_iters must be >= 0, got {config.max_iters!r}")
    if config.picard_iters is not None and int(config.picard_iters) < 0:
        raise ValueError(f"picard_iters must be >= 0, got {config.picard_iters!r}")
    if config.picard_retry_atol is not None and float(config.picard_retry_atol) < 0:
        raise ValueError(
            f"picard_retry_atol must be >= 0 or None, got {config.picard_retry_atol!r}"
        )
    if config.residual_fail is not None and float(config.residual_fail) < 0:
        raise ValueError(
            f"residual_fail must be >= 0 or None, got {config.residual_fail!r}"
        )
    if config.scan_tile not in ("assoc", "seq"):
        raise ValueError(f"unknown scan_tile {config.scan_tile!r}")
    config = _resolve_backend(cell, x, config)
    params = tuple(cell.parameters())
    has_h0 = h0 is not None
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad
        or (has_h0 and h0.requires_grad)
        or any(p.requires_grad for p in params)
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
            ctx.scan_backend = (
                "triton" if config.scan_backend == "fused" else config.scan_backend
            )
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


def _can_triton_scan(x: Tensor) -> bool:
    return is_fused_dtype_supported(x.dtype, x.device)


def _can_fuse(cell: nn.Module, x: Tensor) -> bool:
    if not _can_triton_scan(x):
        return False
    if isinstance(cell, ParaSLSTM):
        return cell.mix == "diag" and getattr(cell, "W_x", None) is not None
    if not isinstance(cell, (ParaGRU, ParaLSTM)):
        return False
    return getattr(cell, "W_x", None) is not None


def _pick_auto(cell: nn.Module, x: Tensor) -> str:
    if _can_fuse(cell, x):
        return "fused"
    if _can_triton_scan(x):
        return "triton"
    return "eager"


def slstm_auto_picard(seq_len: int) -> int:
    """Library Picard P for ParaSLSTM: 1 if T≤64, 3 if T≤2048, else 5.

    Measured at ``d_h=256``, ``x_scale=1``, K=3 (para-slstm.md). Explicit 0 is
    zero-hidden only. Fallback if ``residual_fail`` fires: raise ``picard_iters``.
    """
    t = int(seq_len)
    if t <= 64:
        return 1
    if t <= 2048:
        return 3
    return 5


def slstm_picard_next(picard_iters: int) -> int | None:
    """Next auto Picard rung after ``picard_iters``, or None at the cap (5).

    Ladder is ``{1, 3, 5}`` (slstm_auto_picard / para-slstm.md).
    """
    p = int(picard_iters)
    for rung in _PICARD_RUNGS:
        if p < rung:
            return rung
    return None


def _resolve_picard(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    if config.picard_iters is None:
        if isinstance(cell, ParaSLSTM):
            chosen = slstm_auto_picard(x.shape[1])
            if not torch.compiler.is_compiling():
                log.debug(
                    "slstm_picard_auto",
                    extra={"chosen": chosen, "seq_len": x.shape[1], "d_h": cell.d_h},
                )
            return replace(config, picard_iters=chosen)
        return replace(config, picard_iters=0)
    if config.picard_iters and not isinstance(cell, ParaSLSTM):
        raise TypeError(
            f"NewtonConfig(picard_iters=) is ParaSLSTM only (got {type(cell).__name__})"
        )
    return config


def _resolve_backend(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    auto_p = config.picard_iters is None
    config = _resolve_picard(cell, x, config)
    if config.picard_adapt is None:
        config = replace(
            config,
            picard_adapt=auto_p and isinstance(cell, ParaSLSTM),
        )
    requested = config.scan_backend
    if config.coords == "log":
        if not isinstance(cell, ParaSLSTM):
            raise TypeError(
                "NewtonConfig(coords='log') is ParaSLSTM only "
                f"(got {type(cell).__name__})"
            )
        if requested == "fused" and not _can_fuse(cell, x):
            raise TypeError(_fused_error(cell, x))
        if requested == "auto":
            chosen = _pick_auto(cell, x)
            return replace(config, scan_backend=chosen)
        return config
    if requested == "auto":
        chosen = _pick_auto(cell, x)
        if not torch.compiler.is_compiling():
            log.debug(
                "scan_backend_auto",
                extra={
                    "chosen": chosen,
                    "cell": type(cell).__name__,
                    "device": str(x.device),
                    "dtype": str(x.dtype),
                },
            )
        return replace(config, scan_backend=chosen)
    if requested == "fused" and not _can_fuse(cell, x):
        raise TypeError(_fused_error(cell, x))
    return config


def _fused_error(cell: nn.Module, x: Tensor) -> str:
    if x.dtype == torch.bfloat16 and not is_fused_dtype_supported(x.dtype, x.device):
        major, minor = torch.cuda.get_device_capability(x.device)
        return (
            "fused Newton: bfloat16 needs CUDA compute capability >= 8.0 "
            f"(Ampere+ tensor cores); got sm_{major}{minor} on {x.device}. "
            "Use float16; cell+scan algebra stays fp32."
        )
    if isinstance(cell, ParaSLSTM) and cell.mix != "diag":
        return (
            f"scan_backend='fused' is mix='diag' only (4x4 SRAM); got mix={cell.mix!r}"
        )
    return (
        "scan_backend='fused' needs CUDA ParaGRU/ParaLSTM/ParaSLSTM(mix='diag') "
        f"in float16/float32/bfloat16 (got {type(cell).__name__} {x.dtype} {x.device})"
    )


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
    if (
        config.picard_adapt
        and isinstance(cell, ParaSLSTM)
        and not torch.compiler.is_compiling()
    ):
        return _newton_forward_picard_adapt(cell, x, config, h0=h0, stats=stats)
    return _newton_solve(cell, x, config, h0=h0, stats=stats)


def _picard_retry_needed(res: float, config: NewtonConfig) -> bool:
    if not math.isfinite(res):
        return True
    cap = config.picard_retry_atol
    if cap is not None and res > cap:
        return True
    fail = config.residual_fail
    return fail is not None and res > fail


def _newton_forward_picard_adapt(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    stats: NewtonStats | None,
) -> Tensor:
    """Re-run Alg. 1 at the next P rung when the guess was outside the basin."""
    cfg = config
    while True:
        st = stats if stats is not None else NewtonStats()
        quiet = replace(cfg, residual_fail=None)
        states = _newton_solve(cell, x, quiet, h0=h0, stats=st)
        res = st.max_residual
        nxt = slstm_picard_next(int(cfg.picard_iters or 0))
        if _picard_retry_needed(res, config) and nxt is not None:
            if not torch.compiler.is_compiling():
                log.info(
                    "picard_adapt",
                    extra={
                        "from_p": int(cfg.picard_iters or 0),
                        "to_p": nxt,
                        "max_residual": res,
                        "seq_len": int(x.shape[1]),
                        "batch": int(x.shape[0]),
                        "d_h": cell.d_h,
                        "scan_backend": cfg.scan_backend,
                    },
                )
            cfg = replace(cfg, picard_iters=nxt)
            continue
        cap = config.residual_fail
        if cap is not None and (not math.isfinite(res) or res > cap):
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=st.iters,
                stats=st,
                residual_history=st.residual_history,
                known_residual=res,
            )
        return states


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
        piece = _newton_forward(
            cell, x[:, t0 : t0 + length], inner, h0=carry, stats=chunk_stats
        )
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


def _slstm_newton_guess(
    cell: ParaSLSTM,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    wx: Tensor | None,
) -> Tensor:
    """Zero-hidden, or extra frozen-gate Picard scans. Still O(log T)."""
    pre = wx if wx is not None else cell.W_x(x)
    if config.picard_iters:
        if not torch.compiler.is_compiling():
            log.debug(
                "slstm_picard_init",
                extra={
                    "n_picard": config.picard_iters,
                    "seq_len": x.shape[1],
                    "batch": x.shape[0],
                    "d_h": cell.d_h,
                },
            )
        return slstm_picard_init(cell, pre, h0=h0, n_picard=config.picard_iters)
    return slstm_zero_hidden_init(pre, eps=cell.eps, h0=h0)


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


def _scan(
    jac: Tensor,
    residual: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
) -> Tensor:
    if structure is not None:
        return _scan_named(jac, residual, backend=backend, structure=structure)
    return _scan_infer(jac, residual, backend=backend)


def _reverse_scan(
    jac: Tensor,
    partial: Tensor,
    *,
    backend: str = "eager",
    structure: str | None = None,
) -> Tensor:
    if structure is not None:
        return _reverse_scan_named(jac, partial, backend=backend, structure=structure)
    return _reverse_scan_infer(jac, partial, backend=backend)


def _scan_named(
    jac: Tensor, residual: Tensor, *, backend: str, structure: str
) -> Tensor:
    if structure == "diag":
        return scan_diag(jac, residual, backend=backend)
    if structure == "block2":
        return scan_block2(jac, residual, backend=backend)
    if structure == "block4":
        return scan_block4(jac, residual, backend=backend)
    if structure == "head":
        packed_h = _head_slot_pack(jac, residual)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} "
                f"residual {tuple(residual.shape)}"
            )
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, residual)
        if packed is not None:
            jac_f, res_f, shape = packed
            return scan_dense(jac_f, res_f, backend=backend).reshape(shape)
        return scan_dense(jac, residual, backend=backend)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _reverse_scan_named(
    jac: Tensor, partial: Tensor, *, backend: str, structure: str
) -> Tensor:
    if structure == "diag":
        return reverse_scan_diag(jac, partial, backend=backend)
    if structure == "block2":
        return reverse_scan_block2(jac, partial, backend=backend)
    if structure == "block4":
        return reverse_scan_block4(jac, partial, backend=backend)
    if structure == "head":
        packed_h = _head_slot_pack(jac, partial)
        if packed_h is None:
            raise ValueError(
                f"jac_structure='head' but jac {tuple(jac.shape)} "
                f"partial {tuple(partial.shape)}"
            )
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if structure == "dense":
        packed = _dense_slot_pack(jac, partial)
        if packed is not None:
            jac_f, part_f, shape = packed
            return reverse_scan_dense(jac_f, part_f, backend=backend).reshape(shape)
        return reverse_scan_dense(jac, partial, backend=backend)
    raise ValueError(f"unknown jac_structure {structure!r}")


def _scan_infer(jac: Tensor, residual: Tensor, *, backend: str) -> Tensor:
    packed = _dense_slot_pack(jac, residual)
    if packed is not None:
        jac_f, res_f, shape = packed
        delta = scan_dense(jac_f, res_f, backend=backend)
        return delta.reshape(shape)
    packed_h = _head_slot_pack(jac, residual)
    if packed_h is not None:
        jac_f, res_f, shape, n_heads, d_head = packed_h
        delta = scan_dense(jac_f, res_f, backend=backend)
        return _head_slot_unpack(delta, shape, n_heads, d_head)
    if jac.dim() == residual.dim():
        return scan_diag(jac, residual, backend=backend)
    if jac.dim() == 4:
        return scan_dense(jac, residual, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return scan_block4(jac, residual, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return scan_block2(jac, residual, backend=backend)
    raise ValueError(
        f"cannot dispatch scan for jac {tuple(jac.shape)} residual "
        f"{tuple(residual.shape)}; set NewtonConfig.jac_structure"
    )


def _reverse_scan_infer(jac: Tensor, partial: Tensor, *, backend: str) -> Tensor:
    packed = _dense_slot_pack(jac, partial)
    if packed is not None:
        jac_f, part_f, shape = packed
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return mu.reshape(shape)
    packed_h = _head_slot_pack(jac, partial)
    if packed_h is not None:
        jac_f, part_f, shape, n_heads, d_head = packed_h
        mu = reverse_scan_dense(jac_f, part_f, backend=backend)
        return _head_slot_unpack(mu, shape, n_heads, d_head)
    if jac.dim() == partial.dim():
        return reverse_scan_diag(jac, partial, backend=backend)
    if jac.dim() == 4:
        return reverse_scan_dense(jac, partial, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 4:
        return reverse_scan_block4(jac, partial, backend=backend)
    if jac.dim() == 5 and jac.shape[-3] == 2:
        return reverse_scan_block2(jac, partial, backend=backend)
    raise ValueError(
        f"cannot dispatch reverse scan for jac {tuple(jac.shape)} partial "
        f"{tuple(partial.shape)}; set NewtonConfig.jac_structure"
    )


def _t0_state_vjp(jac: Tensor, mu: Tensor) -> Tensor:
    """``J_0^T μ_0`` — adjoint of paper ``h_0``. Layout matches ``_scan``."""
    packed = _dense_slot_pack(jac, mu)
    if packed is not None:
        jac_f, mu_f, shape = packed
        g = torch.matmul(
            jac_f[:, 0].transpose(-1, -2), mu_f[:, 0].unsqueeze(-1)
        ).squeeze(-1)
        return g.reshape(shape[0], *shape[2:])
    packed_h = _head_slot_pack(jac, mu)
    if packed_h is not None:
        jac_f, mu_f, shape, n_heads, d_head = packed_h
        g = torch.matmul(
            jac_f[:, 0].transpose(-1, -2), mu_f[:, 0].unsqueeze(-1)
        ).squeeze(-1)
        packed_h0 = g.reshape(shape[0], n_heads, 4 * d_head)
        return slstm_unpack_heads(packed_h0, n_heads, d_head)
    if jac.dim() == mu.dim():
        return jac[:, 0] * mu[:, 0]
    if jac.dim() == 4:
        return torch.matmul(
            jac[:, 0].transpose(-1, -2), mu[:, 0].unsqueeze(-1)
        ).squeeze(-1)
    return torch.einsum("boid,bod->bid", jac[:, 0], mu[:, 0])


def _dense_slot_pack(
    jac: Tensor, vec: Tensor
) -> tuple[Tensor, Tensor, tuple[int, ...]] | None:
    """4-slot state with a flattened dense J: ``jac`` is (B, T, S d, S d)."""
    if jac.dim() != 4 or vec.dim() != 4:
        return None
    slots, d_h = vec.shape[-2], vec.shape[-1]
    sd = slots * d_h
    if jac.shape[-1] != sd or jac.shape[-2] != sd:
        return None
    return jac, vec.reshape(*vec.shape[:2], sd), vec.shape


def _head_slot_pack(
    jac: Tensor, vec: Tensor
) -> tuple[Tensor, Tensor, tuple[int, ...], int, int] | None:
    """Per-head dense J: ``jac`` is (B, T, H, 4 d_head, 4 d_head), ``vec`` is (B, T, 4, d_h)."""
    if jac.dim() != 5 or vec.dim() != 4:
        return None
    if jac.shape[-1] != jac.shape[-2]:
        return None
    slots, d_h = vec.shape[-2], vec.shape[-1]
    n_heads = jac.shape[2]
    sd = jac.shape[-1]
    if slots != 4 or n_heads < 1 or d_h % n_heads != 0:
        return None
    d_head = d_h // n_heads
    if sd != 4 * d_head:
        return None
    # block4 is (B, T, 4, 4, d_h); last dim is the channel.
    if jac.shape[2] == 4 and jac.shape[3] == 4 and jac.shape[-1] == d_h:
        return None
    packed = slstm_pack_heads(vec, n_heads, d_head)
    b, t = vec.shape[:2]
    jac_f = jac.permute(0, 2, 1, 3, 4).reshape(b * n_heads, t, sd, sd).contiguous()
    vec_f = packed.permute(0, 2, 1, 3).reshape(b * n_heads, t, sd).contiguous()
    return jac_f, vec_f, vec.shape, n_heads, d_head


def _head_slot_unpack(
    folded: Tensor, shape: tuple[int, ...], n_heads: int, d_head: int
) -> Tensor:
    b, t = shape[:2]
    sd = 4 * d_head
    packed = folded.reshape(b, n_heads, t, sd).permute(0, 2, 1, 3)
    return slstm_unpack_heads(packed, n_heads, d_head)


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
