"""Packed cell VJP for eq. 2.6 (``h_prev`` detached).

The reverse scan already applied ``J^T``. This is ``∂L/∂(x, θ)`` from
``μ = ∇_H L`` through one batched ``f`` (eq. 2.6).

ParaGRU / ParaLSTM / ParaSLSTM ``mix='diag'``: closed-form elementwise VJP +
one ``W_x`` GEMM (same split as the fused forward). Formulas live in
``pararnn.kernels.vjp_*`` (``*_recurrence_vjp_eager``); CUDA uses the
matching Triton kernel. Reduction is tile-sum then PyTorch ``.sum``
(deterministic; no atomics). Head/dense sLSTM and custom cells use
Autograd on ``step`` — that is the generic path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM


def cell_vjp(
    cell: nn.Module,
    h_prev: Tensor,
    x: Tensor,
    mu: Tensor,
    *,
    packed: bool,
) -> tuple[Tensor | None, tuple[Tensor | None, ...]]:
    """``(∇_x L, per-parameter grads)`` aligned with ``cell.parameters()``."""
    if packed and isinstance(cell, ParaGRU):
        return _gru_vjp(cell, h_prev, x, mu)
    if packed and isinstance(cell, ParaLSTM):
        return _lstm_vjp(cell, h_prev, x, mu)
    if packed and isinstance(cell, ParaSLSTM) and cell.mix == "diag":
        return _slstm_vjp(cell, h_prev, x, mu)
    return _autograd_vjp(cell, h_prev, x, mu)


def uses_packed_vjp(cell: nn.Module) -> bool:
    """True when eq. 2.6 can skip Autograd on ``step``."""
    if isinstance(cell, (ParaGRU, ParaLSTM)):
        return True
    return isinstance(cell, ParaSLSTM) and cell.mix == "diag"


def _autograd_vjp(
    cell: nn.Module,
    h_prev: Tensor,
    x: Tensor,
    mu: Tensor,
) -> tuple[Tensor | None, tuple[Tensor | None, ...]]:
    params = tuple(cell.parameters())
    x_in = x.detach().requires_grad_(True)
    param_leaves = [p for p in params if p.requires_grad]
    with torch.enable_grad():
        pred = cell.step(h_prev.detach(), x_in)
        grads = torch.autograd.grad(
            pred,
            (x_in, *param_leaves),
            grad_outputs=mu,
            retain_graph=False,
            allow_unused=True,
        )
    grad_x = grads[0]
    leaf_grads = {id(p): g for p, g in zip(param_leaves, grads[1:], strict=True)}
    return grad_x, tuple(leaf_grads.get(id(p)) for p in params)


def _clip_mask(raw: Tensor, cap: float | None) -> Tensor:
    if cap is None:
        return raw.new_ones(raw.shape)
    return (raw.abs() <= cap).to(dtype=raw.dtype)


def _linear_vjp(lin: nn.Linear, x: Tensor, grad_y: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
    """VJP of ``y = x @ Wᵀ + b``. ``x`` may be a strided ``(B, T, d_in)`` view.

    ``reshape`` copies when the batch/time axes are not a dense ``(N, d_in)``
    layout (time skip, feature skip, permute round-trip through ``(B, D, T)``).
    """
    gy = grad_y.reshape(-1, grad_y.shape[-1])
    xx = x.reshape(-1, x.shape[-1])
    grad_x = (gy @ lin.weight).reshape(x.shape)
    grad_w = gy.t() @ xx
    grad_b = gy.sum(0) if lin.bias is not None else None
    return grad_x, grad_w, grad_b


def _align_grads(
    cell: nn.Module, grad_x: Tensor, by_name: dict[str, Tensor | None]
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    packed = []
    for n, p in cell.named_parameters():
        g = by_name.get(n)
        packed.append(None if g is None or not p.requires_grad else g)
    return grad_x, tuple(packed)


def _gru_vjp(
    cell: ParaGRU, h_prev: Tensor, x: Tensor, mu: Tensor
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    from pararnn.kernels.vjp_gru import gru_recurrence_vjp

    a_z, a_r, a_n = cell.clipped_a()
    wx = cell.W_x(x)
    g_wx, g_az, g_ar, g_an = gru_recurrence_vjp(h_prev, wx, a_z, a_r, a_n, mu)
    cap = cell.max_recurrent_norm
    g_az = g_az * _clip_mask(cell.a_z, cap)
    g_ar = g_ar * _clip_mask(cell.a_r, cap)
    g_an = g_an * _clip_mask(cell.a_n, cap)
    grad_x, grad_w, grad_b = _linear_vjp(cell.W_x, x, g_wx)
    return _align_grads(
        cell,
        grad_x,
        {
            "a_z": g_az,
            "a_r": g_ar,
            "a_n": g_an,
            "W_x.weight": grad_w,
            "W_x.bias": grad_b,
        },
    )


def _lstm_vjp(
    cell: ParaLSTM, state_prev: Tensor, x: Tensor, mu: Tensor
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    from pararnn.kernels.vjp_lstm import lstm_recurrence_vjp

    a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
    wx = cell.W_x(x)
    g_wx, g_af, g_az, g_ao, g_cf, g_co = lstm_recurrence_vjp(
        state_prev, wx, a_f, a_z, a_o, c_f, c_o, mu
    )
    cap = cell.max_recurrent_norm
    g_af = g_af * _clip_mask(cell.a_f, cap)
    g_az = g_az * _clip_mask(cell.a_z, cap)
    g_ao = g_ao * _clip_mask(cell.a_o, cap)
    g_cf = g_cf * _clip_mask(cell.c_f, cap)
    g_co = g_co * _clip_mask(cell.c_o, cap)
    grad_x, grad_w, grad_b = _linear_vjp(cell.W_x, x, g_wx)
    return _align_grads(
        cell,
        grad_x,
        {
            "a_f": g_af,
            "a_z": g_az,
            "a_o": g_ao,
            "c_f": g_cf,
            "c_o": g_co,
            "W_x.weight": grad_w,
            "W_x.bias": grad_b,
        },
    )


def _slstm_vjp(
    cell: ParaSLSTM, state_prev: Tensor, x: Tensor, mu: Tensor
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    from pararnn.kernels.vjp_slstm import slstm_recurrence_vjp

    r = cell.clipped_r()
    wx = cell.W_x(x)
    g_wx, g_r = slstm_recurrence_vjp(state_prev, wx, r, mu, cell.eps)
    g_r = g_r * _clip_mask(cell.R, cell.max_recurrent_norm)
    grad_x, grad_w, grad_b = _linear_vjp(cell.W_x, x, g_wx)
    return _align_grads(
        cell,
        grad_x,
        {
            "R": g_r,
            "W_x.weight": grad_w,
            "W_x.bias": grad_b,
        },
    )
