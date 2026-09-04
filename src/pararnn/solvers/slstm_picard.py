"""Picard / frozen-gate init for ParaSLSTM Newton."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_STABILIZER,
    prepend_state,
    prepend_state_ragged,
    segment_start_flags,
    validate_cu_seqlens,
)


def slstm_frozen_gate_scan(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """sLSTM ``(c, n, m, h)`` with gates frozen in ``pre`` ``(B, T, 4 d_h)``.

    ``m`` is the max-plus prefix ``m_t = max(z_f + m_{t-1}, z_i)``, then
    ``n`` and ``c`` are 1D scans. ``h`` is readout. Algebra in fp32; DRAM
    dtype preserved. CUDA fp16/fp32 uses the Triton tiled scan.
    """
    if (
        cu_seqlens is None
        and pre.is_cuda
        and pre.dtype in (torch.float16, torch.float32)
        and (h0 is None or h0.is_cuda)
    ):
        from pararnn.kernels.picard_slstm import (
            _BLOCK_T,
            _CHUNK_PAD,
            frozen_gate_scan_triton,
        )

        n_chunks = (pre.shape[1] + _BLOCK_T - 1) // _BLOCK_T
        if n_chunks <= _CHUNK_PAD:
            return frozen_gate_scan_triton(pre, eps=eps, h0=h0)
    return slstm_frozen_gate_scan_eager(pre, eps=eps, h0=h0, cu_seqlens=cu_seqlens)


def slstm_frozen_gate_scan_eager(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """CPU / fallback twin of ``slstm_frozen_gate_scan`` (eager Blelloch)."""
    from pararnn.solvers.scan import scan_diag

    batch, time, four_d = pre.shape
    d_h = four_d // 4
    orig = pre.dtype
    pre32 = pre.float()
    z_i, z_f, z_z, z_o = pre32.chunk(4, dim=-1)
    if h0 is None:
        c0 = n0 = m0 = pre32.new_zeros(batch, d_h)
        if cu_seqlens is not None:
            n_seq = int(validate_cu_seqlens(cu_seqlens, time).numel()) - 1
            c0 = n0 = m0 = pre32.new_zeros(n_seq, d_h)
    else:
        h0f = h0.float()
        c0 = h0f[:, SLSTM_CELL]
        n0 = h0f[:, SLSTM_NORMALIZER]
        m0 = h0f[:, SLSTM_STABILIZER]
    if cu_seqlens is None:
        p = z_f.cumsum(dim=1)
        m = p + torch.maximum((z_i - p).cummax(dim=1).values, m0.unsqueeze(1))
        m_prev = torch.cat((m0.unsqueeze(1), m[:, :-1]), dim=1)
        i_t = torch.exp(z_i - m)
        f_t = torch.exp(z_f + m_prev - m)
        z = torch.tanh(z_z)
        res_n = i_t.clone()
        res_n[:, 0] = f_t[:, 0] * n0 + i_t[:, 0]
        res_c = i_t * z
        res_c[:, 0] = f_t[:, 0] * c0 + i_t[:, 0] * z[:, 0]
    else:
        cs = validate_cu_seqlens(cu_seqlens, time)
        starts = cs[:-1]
        p = _seg_cumsum(z_f, cu_seqlens)
        zi_p = z_i - p
        zi_p_adj = zi_p.clone()
        zi_p_adj[0, starts] = torch.maximum(zi_p[0, starts], m0)
        m = p + _seg_cummax(zi_p_adj, cu_seqlens)
        i_t = torch.exp(z_i - m)
        m_prev = _prepend_feat(m, m0, cu_seqlens)
        f_t = torch.exp(z_f + m_prev - m)
        z = torch.tanh(z_z)
        res_n = i_t.clone()
        res_n[0, starts] = f_t[0, starts] * n0 + i_t[0, starts]
        res_c = i_t * z
        res_c[0, starts] = f_t[0, starts] * c0 + i_t[0, starts] * z[0, starts]
    n = scan_diag(f_t, res_n, cu_seqlens=cu_seqlens)
    c = scan_diag(f_t, res_c, cu_seqlens=cu_seqlens)
    o = torch.sigmoid(z_o)
    h = o * (c / (n + eps))
    return torch.stack((c, n, m, h), dim=-2).to(dtype=orig)


def _seg_cumsum(x: Tensor, cu_seqlens: Tensor | None) -> Tensor:
    from pararnn.solvers.scan import scan_diag

    ones = torch.ones_like(x)
    return scan_diag(ones, x, cu_seqlens=cu_seqlens)


def _seg_cummax(x: Tensor, cu_seqlens: Tensor) -> Tensor:
    flags = segment_start_flags(cu_seqlens, x.shape[1], batch=x.shape[0]).to(device=x.device)
    out = x.clone()
    f = flags.clone()
    step = 1
    time = x.shape[1]
    while step < time:
        left, f_l = out[:, :-step], f[:, :-step]
        right, f_r = out[:, step:], f[:, step:]
        take = f_r.reshape(*f_r.shape, *([1] * (right.dim() - f_r.dim())))
        combined = torch.maximum(left, right)
        out = out.clone()
        f = f.clone()
        out[:, step:] = torch.where(take, right, combined)
        f[:, step:] = f_r | f_l
        step *= 2
    return out


def _prepend_feat(feat: Tensor, h0_feat: Tensor, cu_seqlens: Tensor) -> Tensor:
    """Shift a ``(1, N, d)`` feature right; ``h0_feat`` is ``(S, d)`` at starts."""
    dummy = feat.unsqueeze(-2)
    h0 = h0_feat.unsqueeze(-2)
    return prepend_state_ragged(dummy, h0, cu_seqlens).squeeze(-2)


def slstm_zero_hidden_init(
    wx: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Newton guess: sLSTM with ``R h = 0`` (keep running ``m`` / ``n``).

    App. A ``f(0, x_t)`` zeros ``n`` independently at each t. Mixing ``R h``
    is left to Newton (or to ``slstm_picard_init``).
    """
    return slstm_frozen_gate_scan(wx, eps=eps, h0=h0, cu_seqlens=cu_seqlens)


def slstm_picard_init(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 1,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Zero-hidden scan, then ``n_picard`` frozen-gate scans using ``R h``.

    Each pass freezes mixing from the previous trajectory and rescans
    ``(c, n, m)`` over all T. Cap ``n_picard``: P ~ T is sequential.
    """
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    states = slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens)
    for _ in range(n_picard):
        if cu_seqlens is None:
            h_prev = prepend_state(states, h0)[..., SLSTM_HIDDEN, :]
        else:
            h_prev = prepend_state_ragged(states, h0, cu_seqlens)[..., SLSTM_HIDDEN, :]
        pre = wx + cell._recurrent(h_prev)
        states = slstm_frozen_gate_scan(pre, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens)
    return states
