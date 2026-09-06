"""Factorized M²RNN Jacobian matvecs (no dense ``(KV)×(KV)``).

See ``docs/internal/m2rnn-jacobian.md``. Row-major flatten matches PyTorch.
``J[Δ] = f Δ + (1-f) (1-Z²) ⊙ (Δ W)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def m2rnn_gates(
    h_prev: Tensor,
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One step. ``h_prev`` ``(..., K, V)``; ``k`` ``(..., K)``; ``v`` ``(..., V)``;
    ``f`` scalar-per-prefix (broadcast to ``(..., 1, 1)``); ``w`` ``(V, V)``.
    """
    while f.dim() < h_prev.dim():
        f = f.unsqueeze(-1)
    # (..., K, V)
    s = h_prev @ w + k.unsqueeze(-1) * v.unsqueeze(-2)
    z = torch.tanh(s)
    h_new = f * h_prev + (1.0 - f) * z
    acts = {"z": z, "f": f, "w": w}
    return h_new, acts


def m2rnn_frozen_w_scan(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    h0: Tensor | None = None,
) -> Tensor:
    """Warm-start with ``W=0``: linear associative recurrence in ``H``.

    Gates ``k,v,f`` depend on ``x`` only. Dropping ``H W`` yields

    .. math::

        H_t = f_t H_{t-1} + (1-f_t)\\,\\tanh(k_t v_t^\\top)

    which is an elementwise ``ax+b`` scan (Mamba / mLSTM style). Cost is one
    ``O(T K V)`` pass with no Jacobian — Newton then only corrects the ``W``
    coupling. Used when ``NewtonConfig.picard_iters >= 1`` on ``ParaM2RNN``.
    """
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    z = torch.tanh(k.unsqueeze(-1) * v.unsqueeze(-2))  # (B, T, K, V)
    f_b = f.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    h = k.new_zeros(batch, k_dim, v_dim) if h0 is None else h0
    out = k.new_empty(batch, time, k_dim, v_dim)
    for t in range(time):
        h = f_b[:, t] * h + (1.0 - f_b[:, t]) * z[:, t]
        out[:, t] = h
    return out


def m2rnn_jvp(acts: dict[str, Tensor], delta: Tensor) -> Tensor:
    """``J @ Δ`` at frozen gates. ``delta`` ``(..., K, V)``."""
    z = acts["z"]
    f = acts["f"]
    w = acts["w"]
    dz = (1.0 - z.square()) * (delta @ w)
    return f * delta + (1.0 - f) * dz


def m2rnn_jt_mvp(acts: dict[str, Tensor], mu: Tensor) -> Tensor:
    """``Jᵀ @ M``. ``mu`` ``(..., K, V)``."""
    z = acts["z"]
    f = acts["f"]
    w = acts["w"]
    dz_bar = (1.0 - f) * (1.0 - z.square()) * mu
    return f * mu + dz_bar @ w.transpose(-1, -2)
