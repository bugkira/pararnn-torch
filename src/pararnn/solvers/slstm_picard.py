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
)


def slstm_frozen_gate_scan(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """sLSTM ``(c, n, m, h)`` with gates frozen in ``pre`` ``(B, T, 4 d_h)``.

    ``m`` is the max-plus prefix ``m_t = max(z_f + m_{t-1}, z_i)``, then
    ``n`` and ``c`` are 1D scans. ``h`` is readout. Algebra in fp32; DRAM
    dtype preserved. CUDA fp16/fp32 uses the Triton tiled scan.
    """
    if pre.is_cuda and pre.dtype in (torch.float16, torch.float32) and (h0 is None or h0.is_cuda):
        from pararnn.kernels.picard_slstm import (
            _BLOCK_T,
            _CHUNK_PAD,
            frozen_gate_scan_triton,
        )

        n_chunks = (pre.shape[1] + _BLOCK_T - 1) // _BLOCK_T
        if n_chunks <= _CHUNK_PAD:
            return frozen_gate_scan_triton(pre, eps=eps, h0=h0)
    return slstm_frozen_gate_scan_eager(pre, eps=eps, h0=h0)


def slstm_frozen_gate_scan_eager(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """CPU / fallback twin of ``slstm_frozen_gate_scan`` (eager Blelloch)."""
    from pararnn.solvers.scan import scan_diag

    batch, _, four_d = pre.shape
    d_h = four_d // 4
    orig = pre.dtype
    pre32 = pre.float()
    z_i, z_f, z_z, z_o = pre32.chunk(4, dim=-1)
    if h0 is None:
        c0 = n0 = m0 = pre32.new_zeros(batch, d_h)
    else:
        h0f = h0.float()
        c0 = h0f[:, SLSTM_CELL]
        n0 = h0f[:, SLSTM_NORMALIZER]
        m0 = h0f[:, SLSTM_STABILIZER]
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
    n = scan_diag(f_t, res_n)
    c = scan_diag(f_t, res_c)
    o = torch.sigmoid(z_o)
    h = o * (c / (n + eps))
    return torch.stack((c, n, m, h), dim=-2).to(dtype=orig)


def slstm_zero_hidden_init(
    wx: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """Newton guess: sLSTM with ``R h = 0`` (keep running ``m`` / ``n``).

    App. A ``f(0, x_t)`` zeros ``n`` independently at each t. Mixing ``R h``
    is left to Newton (or to ``slstm_picard_init``).
    """
    return slstm_frozen_gate_scan(wx, eps=eps, h0=h0)


def slstm_picard_init(
    cell: nn.Module,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 1,
) -> Tensor:
    """Zero-hidden scan, then ``n_picard`` frozen-gate scans using ``R h``.

    Each pass freezes mixing from the previous trajectory and rescans
    ``(c, n, m)`` over all T. Cap ``n_picard``: P ~ T is sequential.
    """
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    states = slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0)
    for _ in range(n_picard):
        h_prev = prepend_state(states, h0)[..., SLSTM_HIDDEN, :]
        pre = wx + cell._recurrent(h_prev)
        states = slstm_frozen_gate_scan(pre, eps=cell.eps, h0=h0)
    return states
