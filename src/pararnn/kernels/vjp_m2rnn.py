"""Packed VJP for ``ParaM2RNN`` recurrence (eq. 2.6 cell term).

Closed-form through ``H ← f H + (1-f) tanh(H W + k vᵀ)``. No Autograd on
``step``. Gradients land on ``(k, v, f_logit)`` then ``W_x``, plus ``W``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def m2rnn_recurrence_vjp(
    h_prev: Tensor,
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """VJP of one M²RNN step at frozen ``(k,v,f,W)``.

    Parameters
    ----------
    h_prev : Tensor
        ``(..., K, V)``.
    k, v, f : Tensor
        ``(..., K)``, ``(..., V)``, ``(...)`` (broadcast to state).
    w : Tensor
        ``(V, V)``.
    mu : Tensor
        Upstream ``∇_{H^+} L``, same shape as ``h_prev``.

    Returns
    -------
    g_h, g_k, g_v, g_f, g_w
        ``g_f`` is w.r.t. the post-sigmoid forget (same shape as ``f``).
        ``g_w`` is ``(V, V)`` summed over batch/time prefixes.
    """
    f_b = f
    while f_b.dim() < h_prev.dim():
        f_b = f_b.unsqueeze(-1)
    s = h_prev @ w + k.unsqueeze(-1) * v.unsqueeze(-2)
    z = torch.tanh(s)
    # H+ = f H + (1-f) Z
    g_f = (mu * (h_prev - z)).sum(dim=(-1, -2))
    g_z = (1.0 - f_b) * mu
    g_s = (1.0 - z.square()) * g_z
    g_h = f_b * mu + g_s @ w.transpose(-1, -2)
    # ∂s/∂W: s = H W → g_W = Σ Hᵀ g_s over leading dims
    lead = g_s.shape[:-2]
    if lead:
        hp = h_prev.reshape(-1, h_prev.shape[-2], h_prev.shape[-1])
        gs = g_s.reshape(-1, g_s.shape[-2], g_s.shape[-1])
        g_w = torch.einsum("nku,nkv->uv", hp, gs)
    else:
        g_w = torch.einsum("ku,kv->uv", h_prev, g_s)
    g_k = (g_s * v.unsqueeze(-2)).sum(dim=-1)
    g_v = (g_s * k.unsqueeze(-1)).sum(dim=-2)
    return g_h, g_k, g_v, g_f, g_w
