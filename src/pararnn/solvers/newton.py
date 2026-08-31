"""Newton iterations wrapping a parallel scan (Danieli et al. 2025 Alg. 1).

K=3: App. A — residual to machine precision in 3–4 steps for ParaGRU/ParaLSTM.
Init: eq. A.1, h_l^0 = f(0, x_l), not the zero trajectory (README draft was wrong).

Any cell with ``step(h, x)`` parallelizes: Autograd supplies ``J = ∂f/∂h``
(DEER / Lim et al.). ParaGRU/ParaLSTM keep analytic J (paper §3) as the default.

Backward is **not** autograd through the K iterates. Paper eq. 2.6: one reverse
scan of J^T, then a VJP of the batched cell. ParaGRU/ParaLSTM pack that VJP in
Triton on CUDA (``W_x`` GEMM still PyTorch). Custom cells use Autograd on
``step``. IFT is not this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.layout import prepend_state
from pararnn.solvers.jacobian import step_and_jacobian
from pararnn.solvers.scan import (
    reverse_scan_block2,
    reverse_scan_dense,
    reverse_scan_diag,
    scan_block2,
    scan_dense,
    scan_diag,
)
from pararnn.solvers.vjp import cell_vjp

log = logging.getLogger(__name__)


@dataclass
class NewtonConfig:
    max_iters: int = 3
    omega: float = 1.0  # 1 = vanilla Newton; <1 damps (cf. Gonzalez et al. ELK)
    # eager: vectorized Blelloch (default, CPU+CUDA).
    # triton: CUDA scan (fp16 DRAM / fp32 algebra, or fp32).
    # fused: CUDA cell + J + scan. GEMM W_x stays in PyTorch. ParaGRU/LSTM only.
    # fp16 is Turing TC for W_x; not bf16. Not a paper hyperparameter.
    scan_backend: str = "eager"
    # auto: analytic J if the cell has step_with_jacobian, else Autograd.
    # analytic: require step_with_jacobian (paper §3 cells).
    # autograd: torch.func JVP/jacrev — any step(h, x).
    jacobian: str = "auto"
    # None infers: (B,T,D) → diag JVP; (B,T,2,D) → block2.
    # dense: full d_h×d_h (exact mixing cells; O(d^3) scan).
    jac_structure: str | None = None


def newton_apply(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig | None = None,
    *,
    h0: Tensor | None = None,
) -> Tensor:
    """Parallel forward: Newton on F(H)=0, inner solve via associative scan.

    ``h0`` is the paper's ``h_0`` (default 0). If it is nonzero and
    ``scan_backend='fused'``, fall back to eager (fused kernels prepend zeros)
    and log it.

    If gradients are enabled, the backward uses eq. 2.6 (reverse scan) instead
    of differentiating the Newton loop.
    """
    config = config or NewtonConfig()
    if config.scan_backend not in ("eager", "triton", "fused"):
        raise ValueError(f"unknown scan backend {config.scan_backend!r}")
    if config.jacobian not in ("auto", "analytic", "autograd"):
        raise ValueError(f"unknown jacobian {config.jacobian!r}")
    config = _maybe_unfuse_h0(config, h0)
    params = tuple(cell.parameters())
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or any(p.requires_grad for p in params)
    )
    if not needs_grad:
        return _newton_forward(cell, x, config, h0=h0)

    class _NewtonFixedPoint(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_in: Tensor, *param_tensors: Tensor) -> Tensor:
            del param_tensors
            with torch.no_grad():
                states = _newton_forward(cell, x_in, config, h0=h0)
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


def _h0_nonzero(h0: Tensor | None) -> bool:
    if h0 is None:
        return False
    return bool((h0.detach().abs().amax() > 0).item())


def _maybe_unfuse_h0(config: NewtonConfig, h0: Tensor | None) -> NewtonConfig:
    if config.scan_backend == "fused" and _h0_nonzero(h0):
        log.warning(
            "fused_h0_fallback_eager",
            extra={"reason": "fused kernels prepend zeros; h0 is nonzero"},
        )
        return replace(config, scan_backend="eager")
    return config


def _newton_forward(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None = None,
) -> Tensor:
    if config.scan_backend == "fused":
        return _newton_fused(cell, x, config)
    h_prev0 = _init_h_prev(cell, x, h0)
    wx = _wx_if_analytic(cell, x, config)
    states, _ = step_and_jacobian(
        cell,
        h_prev0,
        x,
        wx=wx,
        jacobian=config.jacobian,
        jac_structure=config.jac_structure,
    )

    for it in range(config.max_iters):
        h_prev = prepend_state(states, h0)
        pred, jac = step_and_jacobian(
            cell,
            h_prev,
            x,
            wx=wx,
            jacobian=config.jacobian,
            jac_structure=config.jac_structure,
        )
        residual = pred - states
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "newton_iter",
                extra={
                    "iter": it,
                    "max_residual": float(residual.detach().abs().amax()),
                    "seq_len": x.shape[1],
                    "batch": x.shape[0],
                    "jacobian": config.jacobian,
                },
            )
        delta = _scan(jac, residual, backend=config.scan_backend)
        if states.dtype == torch.float16:
            states = (states.float() + config.omega * delta.float()).to(states.dtype)
        else:
            states = states + config.omega * delta
    return states


def _newton_fused(cell: nn.Module, x: Tensor, config: NewtonConfig) -> Tensor:
    """Alg. 1 with cell+J+scan in Triton. ``W_x(x)`` is still one PyTorch GEMM."""
    if x.dtype is torch.bfloat16:
        raise TypeError(
            "fused Newton: bfloat16 is not used on Turing (no bf16 tensor cores). "
            "Use float16; cell+scan algebra stays fp32."
        )
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"fused Newton supports float16/float32, got {x.dtype}")
    if not x.is_cuda:
        raise RuntimeError("fused Newton requires CUDA")
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
        },
    )
    if isinstance(cell, ParaGRU):
        from pararnn.kernels.newton_gru import newton_gru_fused

        a_z, a_r, a_n = cell._clipped_a()
        return newton_gru_fused(
            wx, a_z, a_r, a_n, max_iters=config.max_iters, omega=config.omega
        )
    if isinstance(cell, ParaLSTM):
        from pararnn.kernels.newton_lstm import newton_lstm_fused

        return newton_lstm_fused(
            wx,
            cell._clip(cell.a_f),
            cell._clip(cell.a_z),
            cell._clip(cell.a_o),
            cell._clip(cell.c_f),
            cell._clip(cell.c_o),
            max_iters=config.max_iters,
            omega=config.omega,
        )
    raise TypeError(f"fused Newton does not support {type(cell).__name__}")


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
    if jac.dim() == residual.dim():
        return scan_diag(jac, residual, backend=backend)
    if jac.dim() == 4:
        return scan_dense(jac, residual, backend=backend)
    return scan_block2(jac, residual, backend=backend)


def _reverse_scan(jac: Tensor, partial: Tensor, *, backend: str = "eager") -> Tensor:
    if jac.dim() == partial.dim():
        return reverse_scan_diag(jac, partial, backend=backend)
    if jac.dim() == 4:
        return reverse_scan_dense(jac, partial, backend=backend)
    return reverse_scan_block2(jac, partial, backend=backend)


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
