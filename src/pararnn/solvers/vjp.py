"""Packed cell VJP for eq. 2.6 (``h_prev`` detached).

The reverse scan already applied ``J^T``. This is ``∂L/∂(x, θ)`` from
``μ = ∇_H L`` through one batched ``f`` (eq. 2.6).

ParaGRU / ParaLSTM / ParaSLSTM ``mix='diag'``: closed-form elementwise VJP +
one ``W_x`` GEMM (same split as the fused forward). On CUDA the elementwise
part is a Triton kernel. Head/dense sLSTM and custom cells use Autograd on
``step`` — that is the generic path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.layout import (
    LSTM_CELL,
    LSTM_HIDDEN,
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_STABILIZER,
)


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
    gy = grad_y.reshape(-1, grad_y.shape[-1])
    xx = x.reshape(-1, x.shape[-1])
    grad_x = (gy @ lin.weight).view_as(x)
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
    a_z, a_r, a_n = cell.clipped_a()
    wx = cell.W_x(x)
    if mu.is_cuda:
        from pararnn.kernels.vjp_gru import gru_recurrence_vjp

        g_wx, g_az, g_ar, g_an = gru_recurrence_vjp(h_prev, wx, a_z, a_r, a_n, mu)
    else:
        g_wx, g_az, g_ar, g_an = _gru_elementwise(h_prev, wx, a_z, a_r, a_n, mu)
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


def _gru_elementwise(
    h_prev: Tensor,
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    zx, rx, nx = wx.chunk(3, dim=-1)
    z = torch.sigmoid(a_z * h_prev + zx)
    r = torch.sigmoid(a_r * h_prev + rx)
    n = torch.tanh(a_n * (h_prev * r) + nx)
    d_z = mu * (n - h_prev)
    d_npre = mu * z * (1.0 - n.square())
    d_zpre = d_z * z * (1.0 - z)
    d_rpre = (d_npre * a_n * h_prev) * r * (1.0 - r)
    g_wx = torch.cat((d_zpre, d_rpre, d_npre), dim=-1)
    g_az = (d_zpre * h_prev).sum(dim=(0, 1))
    g_ar = (d_rpre * h_prev).sum(dim=(0, 1))
    g_an = (d_npre * (h_prev * r)).sum(dim=(0, 1))
    return g_wx, g_az, g_ar, g_an


def _lstm_vjp(
    cell: ParaLSTM, state_prev: Tensor, x: Tensor, mu: Tensor
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
    wx = cell.W_x(x)
    if mu.is_cuda:
        from pararnn.kernels.vjp_lstm import lstm_recurrence_vjp

        g_wx, g_af, g_az, g_ao, g_cf, g_co = lstm_recurrence_vjp(
            state_prev, wx, a_f, a_z, a_o, c_f, c_o, mu
        )
    else:
        g_wx, g_af, g_az, g_ao, g_cf, g_co = _lstm_elementwise(
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


def _lstm_elementwise(
    state_prev: Tensor,
    wx: Tensor,
    a_f: Tensor,
    a_z: Tensor,
    a_o: Tensor,
    peephole_f: Tensor,
    peephole_o: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    c_prev = state_prev[..., LSTM_CELL, :]
    h_prev = state_prev[..., LSTM_HIDDEN, :]
    mu_c = mu[..., LSTM_CELL, :]
    mu_h = mu[..., LSTM_HIDDEN, :]
    fx, zx, ox = wx.chunk(3, dim=-1)
    f = torch.sigmoid(a_f * h_prev + peephole_f * c_prev + fx)
    z = torch.tanh(a_z * h_prev + zx)
    c = f * c_prev + (1.0 - f) * z
    o = torch.sigmoid(a_o * h_prev + peephole_o * c + ox)
    h_act = torch.tanh(c)
    d_o = mu_h * h_act
    d_h_act = mu_h * o
    d_opre = d_o * o * (1.0 - o)
    d_c = mu_c + d_h_act * (1.0 - h_act.square()) + d_opre * peephole_o
    d_f = d_c * (c_prev - z)
    d_z = d_c * (1.0 - f)
    d_zpre = d_z * (1.0 - z.square())
    d_fpre = d_f * f * (1.0 - f)
    g_wx = torch.cat((d_fpre, d_zpre, d_opre), dim=-1)
    dims = (0, 1)
    return (
        g_wx,
        (d_fpre * h_prev).sum(dim=dims),
        (d_zpre * h_prev).sum(dim=dims),
        (d_opre * h_prev).sum(dim=dims),
        (d_fpre * c_prev).sum(dim=dims),
        (d_opre * c).sum(dim=dims),
    )


def _slstm_vjp(
    cell: ParaSLSTM, state_prev: Tensor, x: Tensor, mu: Tensor
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    r = cell.clipped_r()
    wx = cell.W_x(x)
    if mu.is_cuda:
        from pararnn.kernels.vjp_slstm import slstm_recurrence_vjp

        g_wx, g_r = slstm_recurrence_vjp(state_prev, wx, r, mu, cell.eps)
    else:
        g_wx, g_r = _slstm_elementwise(state_prev, wx, r, mu, cell.eps)
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


def _slstm_elementwise(
    state_prev: Tensor,
    wx: Tensor,
    r: Tensor,
    mu: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    c_prev = state_prev[..., SLSTM_CELL, :]
    n_prev = state_prev[..., SLSTM_NORMALIZER, :]
    m_prev = state_prev[..., SLSTM_STABILIZER, :]
    h_prev = state_prev[..., SLSTM_HIDDEN, :]
    mu_c = mu[..., SLSTM_CELL, :]
    mu_n = mu[..., SLSTM_NORMALIZER, :]
    mu_m = mu[..., SLSTM_STABILIZER, :]
    mu_h = mu[..., SLSTM_HIDDEN, :]
    wx_i, wx_f, wx_z, wx_o = wx.chunk(4, dim=-1)
    r_i, r_f, r_z, r_o = r.unbind(0)
    z_i = wx_i + r_i * h_prev
    z_f = wx_f + r_f * h_prev
    z_z = wx_z + r_z * h_prev
    z_o = wx_o + r_o * h_prev
    left = z_f + m_prev
    m_new = torch.maximum(left, z_i)
    gt = (left > z_i).to(dtype=left.dtype)
    eq = (left == z_i).to(dtype=left.dtype)
    alpha = gt + 0.5 * eq
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m_prev - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + eps
    d_o = mu_h * (c_new / denom)
    d_cnew = mu_c + mu_h * (o / denom)
    d_nnew = mu_n + mu_h * (-o * c_new / denom.square())
    d_f = d_cnew * c_prev + d_nnew * n_prev
    d_i = d_cnew * z + d_nnew
    d_z = d_cnew * i_t
    d_zo = d_o * o * (1.0 - o)
    d_zz = d_z * (1.0 - z.square())
    d_mnew = mu_m - d_i * i_t - d_f * f_t
    d_zi = d_i * i_t + d_mnew * (1.0 - alpha)
    d_zf = d_f * f_t + d_mnew * alpha
    g_wx = torch.cat((d_zi, d_zf, d_zz, d_zo), dim=-1)
    dims = (0, 1)
    g_r = torch.stack(
        (
            (d_zi * h_prev).sum(dim=dims),
            (d_zf * h_prev).sum(dim=dims),
            (d_zz * h_prev).sum(dim=dims),
            (d_zo * h_prev).sum(dim=dims),
        ),
        dim=0,
    )
    return g_wx, g_r
