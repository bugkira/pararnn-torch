"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3: App. A — residual to machine precision in 3–4 steps for ParaGRU/ParaLSTM.
Init: eq. A.1, only t=0 sees ``h0``; later t still ``f(0, x_t)``. ParaSLSTM
instead starts from the zero-hidden unroll (running ``m``/``n``, no ``R h``).
``picard_iters`` None is auto P from T for ParaSLSTM (still prefix scans).

Any cell with ``step(h, x)`` parallelizes: Autograd supplies ``J = ∂f/∂h``
(DEER / Lim et al.). ParaGRU/ParaLSTM keep analytic J (paper §3) as the default.

``scan_backend='fused'`` is a handwritten Triton kernel for ParaGRU, ParaLSTM,
and ParaSLSTM ``mix='diag'``, not a generic ``f``. ``'auto'`` picks fused
(CUDA, those cells, fp16/fp32), else Triton scan + ``cell.step``, else eager.
ParaSLSTM ``coords='log'`` uses the LSE fused kernel when ``scan_backend`` is
``fused`` / ``auto`` on CUDA diag mix.

Backward is **not** autograd through the K iterates. Paper eq. 2.6: one reverse
scan of J^T, then a VJP of the batched cell. ParaGRU/ParaLSTM pack that VJP in
Triton on CUDA (``W_x`` GEMM still PyTorch). Custom cells use Autograd on
``step``. IFT is not this.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import (
    ParaSLSTM,
    SLSTMLogCoords,
    slstm_clamp_log_coords,
    slstm_decode_log,
    slstm_encode_log,
    slstm_picard_init,
    slstm_zero_hidden_init,
)
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
from pararnn.solvers.vjp import cell_vjp

log = logging.getLogger(__name__)

# App. A: K=3 reaches machine precision on these cells. Sequential agreement
# tests use 1e-4. Stop a wasted extra iter below that and above fp32 noise.
_DEFAULT_RESIDUAL_ATOL = 1e-5
# Library contract (not a search): K=3 (App. A). Do not raise K when sLSTM
# misses the basin — raise Picard P instead (para-slstm.md).
LIBRARY_NEWTON_ITERS = 3
# After K steps, max|F| above this is divergence, not "needs one more Newton".
# Sequential agreement is 1e-4…2e-3; diverged sLSTM is 1e2…1e14 (para-slstm.md).
# 1.0 sits between. None disables (K-curves, P=0 timing benches).
_DEFAULT_RESIDUAL_FAIL = 1.0


class NewtonDivergenceError(RuntimeError):
    """Newton did not land in the sequential basin. Raise P for sLSTM, not K."""


@dataclass
class NewtonStats:
    """Filled by ``newton_apply(..., stats=)`` after the forward."""

    max_residual: float = float("nan")
    iters: int = 0
    scan_backend: str = ""
    picard_iters: int = 0
    residual_history: tuple[float, ...] = ()


@dataclass
class NewtonConfig:
    # App. A / library contract. ParaSLSTM at long T needs Picard, not more K.
    max_iters: int = LIBRARY_NEWTON_ITERS
    omega: float = 1.0  # 1 = vanilla Newton; <1 damps (cf. Gonzalez et al. ELK)
    # auto: fused on CUDA GRU/LSTM/sLSTM-diag fp16/fp32, else Triton scan + step, else eager.
    # eager: vectorized Blelloch (CPU+CUDA). Any f.
    # triton: CUDA scan only (fp16 DRAM / fp32 algebra, or fp32). Cell stays PyTorch.
    # fused: handwritten CUDA cell+J+scan for GRU/LSTM/sLSTM-diag. Not any f.
    scan_backend: str = "auto"
    # auto: analytic J if the cell has step_with_jacobian, else Autograd.
    # analytic: require step_with_jacobian (paper §3 cells).
    # autograd: torch.func JVP/jacrev — any step(h, x).
    jacobian: str = "auto"
    # None infers from state, or cell.jac_structure (sLSTM: block4 / head / dense).
    # diag: (B,T,D); block2: (B,T,2,D); block4: (B,T,4,D) channelwise 4×4.
    # head: per-head dense (B,T,H,4 d_head, 4 d_head). dense: full d_h×d_h.
    jac_structure: str | None = None
    # None disables early-stop. Default: skip remaining Newton steps when
    # max|F| is already below sequential-agreement scale (see App. A / 1e-4 tests).
    residual_atol: float | None = _DEFAULT_RESIDUAL_ATOL
    # After the last Newton step, raise NewtonDivergenceError if max|F| exceeds
    # this. Default 1.0 (see module comment). None: K-curves / diverged benches.
    residual_fail: float | None = _DEFAULT_RESIDUAL_FAIL
    # log: ParaSLSTM only — LSE cell in (u, log n, m, h). Not a paper default.
    # Not the snap path once picard_iters is in the basin (native is as good
    # or better; para-slstm.md). Fused diag has an LSE kernel. Fallback: native.
    coords: str = "native"
    # None = one Newton over the full T. int: sequential chunks of this length,
    # each with its own K Newton steps; carry the last state as h0. 64 because
    # T=64 K=3 snaps at d_h=256 seed 0 in this repo (para-slstm.md). Gemini /
    # Mamba-2 SRAM tile. Fallback: 32 if a seed fails at 64. Sequential span.
    chunk_len: int | None = None
    # None = auto for ParaSLSTM: library contract P ∈ {1, 3, 5} from T
    # (slstm_auto_picard). Other cells: 0. Explicit 0 is zero-hidden.
    # Fallback if residual_fail fires: raise P, not K.
    picard_iters: int | None = None
    # assoc: tl.associative_scan.
    # seq: serial tl.range prefix in the tile — ablation only.
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

    ``h0`` is the paper's ``h_0`` (default 0). Fused kernels prepend it (not zeros).

    If gradients are enabled, the backward uses eq. 2.6 (reverse scan) instead
    of differentiating the Newton loop.
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
    if config.residual_fail is not None and float(config.residual_fail) < 0:
        raise ValueError(
            f"residual_fail must be >= 0 or None, got {config.residual_fail!r}"
        )
    if config.scan_tile not in ("assoc", "seq"):
        raise ValueError(f"unknown scan_tile {config.scan_tile!r}")
    config = _resolve_backend(cell, x, config)
    params = tuple(cell.parameters())
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or any(p.requires_grad for p in params)
    )
    if not needs_grad:
        return _newton_forward(cell, x, config, h0=h0, stats=stats)

    class _NewtonFixedPoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_in: Tensor, *param_tensors: Tensor) -> Tensor:
            del param_tensors
            with torch.no_grad():
                states = _newton_forward(cell, x_in, config, h0=h0, stats=stats)
            ctx.scan_backend = (
                "triton" if config.scan_backend == "fused" else config.scan_backend
            )
            ctx.jacobian = config.jacobian
            ctx.jac_structure = config.jac_structure
            ctx.h0 = h0.detach() if h0 is not None else None
            ctx.save_for_backward(states, x_in)
            return states

        @staticmethod
        def backward(ctx, grad_states: Tensor):
            states, x_in = ctx.saved_tensors
            grad_x, param_grads = _eq26_vjp(
                cell,
                states,
                x_in,
                grad_states,
                backend=ctx.scan_backend,
                jacobian=ctx.jacobian,
                jac_structure=ctx.jac_structure,
                h0=ctx.h0,
            )
            if not x_in.requires_grad:
                grad_x = None
            return (grad_x, *param_grads)

    return _NewtonFixedPoint.apply(x, *params)


def _can_triton_scan(x: Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.float16, torch.float32)


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

    Contract is this triple, not a search over K. Measured at ``d_h=256``,
    ``x_scale=1``, K=3 (para-slstm.md). Finer seed-0 cutovers (P=2 at T=1024,
    P=4 at T=4096) failed unseeded B=8. Explicit 0 is zero-hidden only.
    Fallback if ``residual_fail`` fires: raise P, not K.
    """
    t = int(seq_len)
    if t <= 64:
        return 1
    if t <= 2048:
        return 3
    return 5


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
            "NewtonConfig(picard_iters=) is ParaSLSTM only "
            f"(got {type(cell).__name__})"
        )
    return config


def _resolve_backend(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    config = _resolve_picard(cell, x, config)
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
    if x.dtype is torch.bfloat16:
        return (
            "fused Newton: bfloat16 is not used on Turing (no bf16 tensor cores). "
            "Use float16; cell+scan algebra stays fp32."
        )
    if isinstance(cell, ParaSLSTM) and cell.mix != "diag":
        return (
            "scan_backend='fused' is mix='diag' only (4x4 SRAM); "
            f"got mix={cell.mix!r}"
        )
    return (
        "scan_backend='fused' needs CUDA ParaGRU/ParaLSTM/ParaSLSTM(mix='diag') "
        f"in float16/float32 (got {type(cell).__name__} {x.dtype} {x.device})"
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
    if config.scan_backend == "fused":
        states = _newton_fused(cell, x, config, h0=h0)
        _fill_stats(cell, x, states, h0, config, iters=config.max_iters, stats=stats)
        return states
    wx = _wx_if_analytic(cell, x, config)
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
        zeros = x.new_zeros(x.shape[0], native.state_slots, native.d_h)
        h0_loop = slstm_encode_log(
            h0 if h0 is not None else zeros, eps=eps
        )
        cell = SLSTMLogCoords(native)
        if not torch.compiler.is_compiling():
            log.debug(
                "newton_slstm_log_coords",
                extra={"seq_len": x.shape[1], "batch": x.shape[0], "d_h": native.d_h},
            )

    iters_done = 0
    last_res = float("nan")
    history: list[float] = []
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
        last_res = float(residual.detach().abs().amax())
        history.append(last_res)
        delta = _scan(jac, residual, backend=config.scan_backend)
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
        atol = config.residual_atol
        if atol is not None and last_res < atol:
            log.info(
                "newton_early_stop",
                extra={
                    "iters": it,
                    "max_residual": last_res,
                    "atol": atol,
                    "seq_len": x.shape[1],
                    "coords": config.coords,
                },
            )
            break
        if states.dtype == torch.float16:
            states = (states.float() + config.omega * delta.float()).to(states.dtype)
        else:
            states = states + config.omega * delta
        if config.coords == "log":
            states = slstm_clamp_log_coords(states)
        iters_done = it + 1
    if config.coords == "log":
        states = slstm_decode_log(states, eps=native.eps)
        cell = native
    _fill_stats(
        cell,
        x,
        states,
        h0,
        config,
        iters=iters_done,
        stats=stats,
        residual_history=history,
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
    (eager / triton / fused). T=64 K=3 snaps at bench width in this repo.
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
) -> None:
    if stats is None and config.residual_fail is None:
        return
    pred = cell.step(prepend_state(states, h0), x)
    res = float((pred - states).detach().abs().amax())
    hist = tuple(residual_history)
    if stats is not None:
        stats.max_residual = res
        stats.iters = iters
        stats.scan_backend = config.scan_backend
        stats.picard_iters = int(config.picard_iters or 0)
        stats.residual_history = hist
    cap = config.residual_fail
    if cap is None:
        return
    diverged = not math.isfinite(res) or res > cap
    if not diverged:
        return
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
    log.error("newton_diverged", extra=extra)
    raise NewtonDivergenceError(
        f"Newton residual {res:.3e} after {iters} iters exceeds residual_fail="
        f"{cap:g} (seq_len={x.shape[1]}, picard={int(config.picard_iters or 0)}, "
        f"history={hist[-8:]!r}). For ParaSLSTM raise P, not K."
    )


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
    from pararnn.kernels.fused import fused_newton

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
) -> tuple[Tensor | None, tuple[Tensor | None, ...]]:
    """``∇_x L`` and per-parameter grads from direct ``∂_H L`` (eq. 2.6 + cell VJP)."""
    h_prev = prepend_state(states, h0)
    wx = _wx_if_analytic(cell, x, NewtonConfig(jacobian=jacobian, jac_structure=jac_structure))
    with torch.no_grad():
        _, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=jacobian,
            jac_structure=jac_structure,
        )
        mu = _reverse_scan(jac, partial, backend=backend)
    packed = isinstance(cell, (ParaGRU, ParaLSTM))
    return cell_vjp(cell, h_prev, x, mu, packed=packed)


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
        return slstm_picard_init(
            cell, pre, h0=h0, n_picard=config.picard_iters
        )
    return slstm_zero_hidden_init(pre, eps=cell.eps, h0=h0)


def _init_h_prev(cell: nn.Module, x: Tensor, h0: Tensor | None) -> Tensor:
    """App. A: ``H^0_t = f(h_{t-1}, x_t)`` in parallel; only t=0 sees ``h0``."""
    h_prev0 = _zero_state_like_input(cell, x)
    if h0 is not None:
        h_prev0[:, 0] = h0
    return h_prev0


def _wx_if_analytic(cell: nn.Module, x: Tensor, config: NewtonConfig) -> Tensor | None:
    """Reuse ``W_x(x)`` only on the analytic-J path (eq. 3.1)."""
    mode = config.jacobian
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


def _scan(jac: Tensor, residual: Tensor, *, backend: str = "eager") -> Tensor:
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
    return scan_block2(jac, residual, backend=backend)


def _reverse_scan(jac: Tensor, partial: Tensor, *, backend: str = "eager") -> Tensor:
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
    return reverse_scan_block2(jac, partial, backend=backend)


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
    # block4 is (B, T, 4, 4, d_h); last dim is the feature, not 4 d_head.
    if jac.shape[2] == 4 and jac.shape[3] == 4 and jac.shape[-1] == d_h:
        return None
    packed = slstm_pack_heads(vec, n_heads, d_head)
    b, t = vec.shape[:2]
    jac_f = jac.permute(0, 2, 1, 3, 4).reshape(b * n_heads, t, sd, sd)
    vec_f = packed.permute(0, 2, 1, 3).reshape(b * n_heads, t, sd)
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
