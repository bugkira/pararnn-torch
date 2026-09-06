"""RWKV-7 Goose linear matrix-state scan (Peng et al., arXiv:2503.14456).

Transition is affine in ``S``:

.. math::

    S_t = S_{t-1} G_t + v_t^\\top k_t,
    \\quad
    G_t = \\mathrm{diag}(w_t) - \\hat\\kappa_t^\\top (a_t \\odot \\hat\\kappa_t).

``G`` is diagonal-plus-rank-1; ``step`` / ``apply_factorized`` never materialize
``G`` as a dense ``d×d``. The associative scan path composes dense ``(G, U)``
pairs for the monoid ``(G₁,U₁)∘(G₂,U₂)=(G₁G₂, U₁G₂+U₂)`` (parallel train).
Newton is unnecessary for this linear core.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rwkv7_apply_factorized(
    s_prev: Tensor,
    w: Tensor,
    a: Tensor,
    kappa: Tensor,
    v: Tensor,
    k: Tensor,
) -> Tensor:
    """One factorized step: ``S @ diag(w) - outer(S κ̂, a⊙κ̂) + outer(v, k)``.

    Parameters
    ----------
    s_prev : Tensor
        ``(..., d_head, d_head)``.
    w, a, kappa, v, k : Tensor
        ``(..., d_head)``; ``kappa`` is already L2-normalized ``κ̂``.
    """
    # S @ diag(w): scale columns.
    sw = s_prev * w.unsqueeze(-2)
    # S @ outer(κ, a⊙κ) = (S @ κ) outer (a⊙κ).
    sk = torch.matmul(s_prev, kappa.unsqueeze(-1)).squeeze(-1)
    a_kappa = a * kappa
    rank1 = sk.unsqueeze(-1) * a_kappa.unsqueeze(-2)
    write = v.unsqueeze(-1) * k.unsqueeze(-2)
    return sw - rank1 + write


def rwkv7_build_g(w: Tensor, a: Tensor, kappa: Tensor) -> Tensor:
    """Dense ``G = diag(w) - outer(κ̂, a⊙κ̂)`` for monoid composition."""
    g = torch.diag_embed(w)
    a_kappa = a * kappa
    return g - kappa.unsqueeze(-1) * a_kappa.unsqueeze(-2)


def rwkv7_outer_vk(v: Tensor, k: Tensor) -> Tensor:
    """Write term ``U = vᵀ k`` as ``(..., d, d)``."""
    return v.unsqueeze(-1) * k.unsqueeze(-2)


def rwkv7_compose_monoid(
    g_left: Tensor,
    u_left: Tensor,
    g_right: Tensor,
    u_right: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compose affine maps: apply left first, then right.

    ``(G_l, U_l) ∘ (G_r, U_r) = (G_l G_r, U_l G_r + U_r)``.
    """
    g = g_left @ g_right
    u = u_left @ g_right + u_right
    return g, u


def rwkv7_associative_scan(
    g: Tensor,
    u: Tensor,
    s0: Tensor | None = None,
) -> Tensor:
    """Inclusive Hillis–Steele scan of the ``(G, U)`` monoid.

    Parameters
    ----------
    g, u : Tensor
        ``(batch, time, n_heads, d_head, d_head)``.
    s0 : Tensor, optional
        ``(batch, n_heads, d_head, d_head)``; default zeros.

    Returns
    -------
    S : Tensor
        ``(batch, time, n_heads, d_head, d_head)``.
    """
    batch, time, n_heads, d_head, _ = g.shape
    if time < 1:
        raise ValueError(f"rwkv7_associative_scan needs time >= 1, got {time}")
    g_pref = g.clone()
    u_pref = u.clone()
    offset = 1
    while offset < time:
        g_prev = g_pref.clone()
        u_prev = u_pref.clone()
        g_l = g_prev[:, : time - offset]
        u_l = u_prev[:, : time - offset]
        g_r = g_prev[:, offset:]
        u_r = u_prev[:, offset:]
        g_pref[:, offset:], u_pref[:, offset:] = rwkv7_compose_monoid(g_l, u_l, g_r, u_r)
        offset *= 2
    if s0 is None:
        s0 = g.new_zeros(batch, n_heads, d_head, d_head)
    # S_t = S_0 @ G_pref_t + U_pref_t
    return torch.matmul(s0.unsqueeze(1), g_pref) + u_pref


def rwkv7_readout(s: Tensor, r: Tensor) -> Tensor:
    """Contract state with receptance: ``y = S @ r`` then flatten heads.

    Parameters
    ----------
    s : Tensor
        ``(..., n_heads, d_head, d_head)``.
    r : Tensor
        ``(..., n_heads, d_head)``.

    Returns
    -------
    y : Tensor
        ``(..., n_heads * d_head)``.
    """
    y_heads = torch.matmul(s, r.unsqueeze(-1)).squeeze(-1)
    return y_heads.reshape(*y_heads.shape[:-2], -1)
