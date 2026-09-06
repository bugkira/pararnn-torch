"""Factorized ParaSLSTM ``mix='head'`` Jacobian matvecs (no dense ``(4d)×(4d)``).

``R_head`` layout matches the cell: ``(4, n_heads, d_in, d_out)`` with
``pre_g = h @ R_g``. Packed tangent/cotangent is ``(..., n_heads, 4 * d_head)``
in slot order ``(c, n, m, h)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pararnn.layout import SLSTM_SLOTS


def _maximum_subgrad_left(left: Tensor, right: Tensor) -> Tensor:
    """∂maximum(left, right)/∂left. Ties split 0.5."""
    gt = (left > right).to(dtype=left.dtype)
    eq = (left == right).to(dtype=left.dtype)
    return gt + 0.5 * eq


def _mix(h: Tensor, r: Tensor) -> Tensor:
    """``h @ R`` per head. ``r`` is ``(H, d_in, d_out)``."""
    return torch.matmul(h.unsqueeze(-2), r).squeeze(-2)


def _mix_t(v: Tensor, r: Tensor) -> Tensor:
    """``R @ v`` per head (= ``v @ R^T``). ``r`` is ``(H, d_in, d_out)``."""
    return torch.matmul(r, v.unsqueeze(-1)).squeeze(-1)


def slstm_head_gates(
    state_prev: Tensor,
    wx: Tensor,
    r_head: Tensor,
    *,
    n_heads: int,
    d_head: int,
    eps: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One parallel step; returns ``state_new`` and head-local acts for JVP.

    Parameters
    ----------
    state_prev : Tensor
        ``(..., 4, d_h)``.
    wx : Tensor
        ``(..., 4 * d_h)`` precomputed ``W_x(x)``.
    r_head : Tensor
        ``(4, n_heads, d_head, d_head)`` clipped recurrent mix.
    """
    prefix = state_prev.shape[:-2]
    d_h = n_heads * d_head
    c = state_prev[..., 0, :].reshape(*prefix, n_heads, d_head)
    n = state_prev[..., 1, :].reshape(*prefix, n_heads, d_head)
    m = state_prev[..., 2, :].reshape(*prefix, n_heads, d_head)
    h = state_prev[..., 3, :].reshape(*prefix, n_heads, d_head)
    wx_h = wx.reshape(*prefix, 4, n_heads, d_head)
    # pre_g = h @ R_g + wx_g
    pre = torch.stack(
        [_mix(h, r_head[g]) + wx_h[..., g, :, :] for g in range(4)],
        dim=-3,
    )
    z_i, z_f, z_z, z_o = pre.unbind(dim=-3)
    left = z_f + m
    m_new = torch.maximum(left, z_i)
    alpha = _maximum_subgrad_left(left, z_i)
    i_t = torch.exp(z_i - m_new)
    f_t = torch.exp(z_f + m - m_new)
    z = torch.tanh(z_z)
    n_new = f_t * n + i_t
    c_new = f_t * c + i_t * z
    o = torch.sigmoid(z_o)
    denom = n_new + eps
    h_new = o * (c_new / denom)
    state_new = torch.stack(
        (
            c_new.reshape(*prefix, d_h),
            n_new.reshape(*prefix, d_h),
            m_new.reshape(*prefix, d_h),
            h_new.reshape(*prefix, d_h),
        ),
        dim=-2,
    )
    acts = {
        "c": c,
        "n": n,
        "m": m,
        "h": h,
        "c_new": c_new,
        "n_new": n_new,
        "i": i_t,
        "f": f_t,
        "z": z,
        "o": o,
        "denom": denom,
        "alpha": alpha,
    }
    return state_new, acts


def slstm_head_jvp(
    acts: dict[str, Tensor],
    r_head: Tensor,
    v: Tensor,
) -> Tensor:
    """``J @ v`` at the given head acts (packed ``(c,n,m,h)``).

    Parameters
    ----------
    acts : dict
        Head-local tensors from ``slstm_head_gates`` (each ``(..., H, d)``).
    r_head : Tensor
        ``(4, H, d_in, d_out)``.
    v : Tensor
        Tangent. ``(..., H, 4 * d_head)``.

    Returns
    -------
    Jv : Tensor
        ``(..., H, 4 * d_head)``.
    """
    d = acts["f"].shape[-1]
    vc, vn, vm, vh = v.reshape(*v.shape[:-1], SLSTM_SLOTS, d).unbind(-2)
    alpha = acts["alpha"]
    beta = 1.0 - alpha
    f = acts["f"]
    i_t = acts["i"]
    z = acts["z"]
    o = acts["o"]
    c = acts["c"]
    n = acts["n"]
    c_new = acts["c_new"]
    denom = acts["denom"]
    r_i, r_f, r_z, r_o = r_head[0], r_head[1], r_head[2], r_head[3]

    d_zi = _mix(vh, r_i)
    d_zf = _mix(vh, r_f)
    d_zz = _mix(vh, r_z)
    d_zo = _mix(vh, r_o)
    dm_dh_v = alpha * d_zf + beta * d_zi
    di_dm = -i_t * alpha
    df_dm = f * beta
    di_dh_v = i_t * (d_zi - dm_dh_v)
    df_dh_v = f * (d_zf - dm_dh_v)
    dz_dh_v = (1.0 - z.square()) * d_zz
    do_dh_v = o * (1.0 - o) * d_zo
    j_ch_v = df_dh_v * c + di_dh_v * z + i_t * dz_dh_v
    j_nh_v = df_dh_v * n + di_dh_v
    inv = o / denom
    dn = -o * c_new / denom.square()
    du = c_new / denom
    j_hh_v = inv * j_ch_v + dn * j_nh_v + du * do_dh_v
    j_cm = df_dm * c + di_dm * z
    j_nm = df_dm * n + di_dm

    c_out = f * vc + j_cm * vm + j_ch_v
    n_out = f * vn + j_nm * vm + j_nh_v
    m_out = alpha * vm + dm_dh_v
    h_out = inv * f * vc + dn * f * vn + (inv * j_cm + dn * j_nm) * vm + j_hh_v
    return torch.stack((c_out, n_out, m_out, h_out), dim=-2).reshape(*v.shape[:-1], 4 * d)


def slstm_head_jt_mvp(
    acts: dict[str, Tensor],
    r_head: Tensor,
    mu: Tensor,
) -> Tensor:
    """``J^T @ μ`` at the given head acts (for eq. 2.6).

    Parameters
    ----------
    acts : dict
        Head-local tensors from ``slstm_head_gates``.
    r_head : Tensor
        ``(4, H, d_in, d_out)``.
    mu : Tensor
        Cotangent. ``(..., H, 4 * d_head)``.

    Returns
    -------
    Jt_mu : Tensor
        ``(..., H, 4 * d_head)``.
    """
    d = acts["f"].shape[-1]
    mc, mn, mm, mh = mu.reshape(*mu.shape[:-1], SLSTM_SLOTS, d).unbind(-2)
    alpha = acts["alpha"]
    beta = 1.0 - alpha
    f = acts["f"]
    i_t = acts["i"]
    z = acts["z"]
    o = acts["o"]
    c = acts["c"]
    n = acts["n"]
    c_new = acts["c_new"]
    denom = acts["denom"]
    r_i, r_f, r_z, r_o = r_head[0], r_head[1], r_head[2], r_head[3]

    inv = o / denom
    dn = -o * c_new / denom.square()
    du = c_new / denom
    di_dm = -i_t * alpha
    df_dm = f * beta
    j_cm = df_dm * c + di_dm * z
    j_nm = df_dm * n + di_dm

    g_c = f * mc + inv * f * mh
    g_n = f * mn + dn * f * mh
    g_m = j_cm * mc + j_nm * mn + alpha * mm + (inv * j_cm + dn * j_nm) * mh

    # Dense h-column adjoint into pre-gate directions.
    w_c_like = mc + inv * mh
    w_n_like = mn + dn * mh
    w_do = du * mh
    adj_df = w_c_like * c + w_n_like * n
    adj_di = w_c_like * z + w_n_like
    adj_dz = w_c_like * i_t
    adj_dzf = f * adj_df
    adj_dzi = i_t * adj_di
    adj_dm = -(f * adj_df + i_t * adj_di) + mm
    adj_dzf = adj_dzf + alpha * adj_dm
    adj_dzi = adj_dzi + beta * adj_dm
    adj_dzz = (1.0 - z.square()) * adj_dz
    adj_dzo = o * (1.0 - o) * w_do

    g_h = (
        _mix_t(adj_dzi, r_i)
        + _mix_t(adj_dzf, r_f)
        + _mix_t(adj_dzz, r_z)
        + _mix_t(adj_dzo, r_o)
    )
    return torch.stack((g_c, g_n, g_m, g_h), dim=-2).reshape(*mu.shape[:-1], 4 * d)
