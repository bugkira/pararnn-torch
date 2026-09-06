"""Factorized ParaGRU ``mix='head'`` Jacobian matvecs (no dense ``d×d``).

``A_*`` layout matches the cell: ``(n_heads, d_in, d_out)`` with ``y = h @ A``.
Gates ``h,z,r,n`` are head-local ``(..., n_heads, d_head)``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _sigmoid_prime(gate: Tensor) -> Tensor:
    return gate * (1.0 - gate)


def _tanh_prime(act: Tensor) -> Tensor:
    return 1.0 - act.square()


def _mix(h: Tensor, a: Tensor) -> Tensor:
    """``h @ A`` per head."""
    return torch.matmul(h.unsqueeze(-2), a).squeeze(-2)


def _mix_t(v: Tensor, a: Tensor) -> Tensor:
    """``A @ v`` per head (= ``v @ A^T``) for ``A`` ``(H, in, out)``."""
    return torch.matmul(a, v.unsqueeze(-1)).squeeze(-1)


def gru_head_jvp(
    h: Tensor,
    z: Tensor,
    r: Tensor,
    n: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    v: Tensor,
) -> Tensor:
    """``J @ v`` with ``J = ∂h'/∂h`` at the given gates (eq. 3.2a, block form)."""
    z_p = _sigmoid_prime(z)
    r_p = _sigmoid_prime(r)
    n_p = _tanh_prime(n)
    # A^T @ v ≡ v @ A for square head maps.
    dz_v = z_p * _mix(v, a_z)
    du_v = r * v + h * (r_p * _mix(v, a_r))
    dn_v = n_p * _mix(du_v, a_n)
    return (1.0 - z) * v + (n - h) * dz_v + z * dn_v


def gru_head_jt_mvp(
    h: Tensor,
    z: Tensor,
    r: Tensor,
    n: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    mu: Tensor,
) -> Tensor:
    """``J^T @ μ`` at the given gates (for eq. 2.6 reverse scan)."""
    z_p = _sigmoid_prime(z)
    r_p = _sigmoid_prime(r)
    n_p = _tanh_prime(n)
    # J = D1 + D_nh @ (D_zp @ A_z^T) + D_z @ (D_np @ A_n^T @ (D_r + D_h @ D_rp @ A_r^T))
    # J^T μ = D1 μ + A_z @ (z' * ((n-h)*μ)) + ...
    t1 = (1.0 - z) * mu
    w_z = z_p * ((n - h) * mu)
    t2 = _mix_t(w_z, a_z)
    # From diag(z) @ diag(n') @ A_n^T @ du_dh: backprop μ through z*n'*(A_n^T @ ·)
    w_n = n_p * (z * mu)
    # g_du = A_n @ w_n
    g_du = _mix_t(w_n, a_n)
    # du = diag(r) + diag(h) @ (diag(r') @ A_r^T)
    # g_du through diag(r): r * g_du
    # g_du through A_r path: A_r @ (r' * (h * g_du))
    t3 = r * g_du + _mix_t(r_p * (h * g_du), a_r)
    return t1 + t2 + t3
