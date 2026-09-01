"""Index and layout conventions.

The paper is 1-indexed. HTML/PDF conversions of Alg. 2 disagree on the scan
offset (``l - 2^i`` vs ``l - 2^i + 1``). Eq. 2.3 and eq. 2.4 also disagree on
whether ``J_f`` at step ``l`` is evaluated at ``h_l`` or ``h_{l-1}``.

This module is the code's source of truth. Derived from eq. 2.1–2.3, not from
the OCR'd Algorithm 2 and not from Apple's source.

Paper (1-based)
    l = 1..L, h_0 = 0, x_l, h_l = f(h_{l-1}, x_l)
    F_l = h_l - f(h_{l-1}, x_l)
    J_l := ∂f/∂h_prev at (h_{l-1}, x_l)     # footnote: J_SSM |_{h_{l-1}} ≡ A_l
    δh_l = J_l δh_{l-1} + (f(h_{l-1}, x_l) - h_l),  δh_0 = 0

Code (0-based, batch-first)
    t = 0..T-1  corresponds to paper l = t+1
    x: (batch, time, d_in)
    GRU state h: (batch, time, d_h)
    LSTM state: (batch, time, 2, d_h) with index 0 = cell c, 1 = hidden h
        (paper concatenates [c, h]; PyTorch nn.LSTM's hidden tuple is (h, c) — inverted)
    sLSTM state: (batch, time, 4, d_h) = (c, n, m, h) — Beck et al. 2024, not ParaRNN.

Scan
    Work-efficient Blelloch on 0-based time (pad to ``2^k`` with identity).
    Never ``t - 2^i + 1``. The ``+1`` in some renderings of Alg. 2 is a 1-based artefact.
"""

from __future__ import annotations

import torch

LSTM_CELL = 0
LSTM_HIDDEN = 1

# sLSTM (Beck et al. 2024): four-slot state, not Apple's CIFG pair.
SLSTM_CELL = 0
SLSTM_NORMALIZER = 1
SLSTM_STABILIZER = 2
SLSTM_HIDDEN = 3
SLSTM_SLOTS = 4


def slstm_pack_heads(state: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    """``(..., 4, n_heads * d_head)`` → ``(..., n_heads, 4 * d_head)``.

    Packed last dim is ``(c, n, m, h)`` for one head. Used by the per-head dense
    Newton scan (xLSTM memory mixing is block-diagonal across heads).
    """
    heads = state.reshape(*state.shape[:-1], n_heads, d_head)
    packed = heads.movedim(-3, -2)
    return packed.reshape(*state.shape[:-2], n_heads, 4 * d_head)


def slstm_unpack_heads(packed: torch.Tensor, n_heads: int, d_head: int) -> torch.Tensor:
    """Inverse of ``slstm_pack_heads``."""
    slots = packed.reshape(*packed.shape[:-1], SLSTM_SLOTS, d_head)
    slots = slots.movedim(-2, -3)
    return slots.reshape(*packed.shape[:-2], SLSTM_SLOTS, n_heads * d_head)


def prepend_zero_state(states: torch.Tensor) -> torch.Tensor:
    """Shift the trajectory right by one step and put zeros at t=0."""
    return prepend_state(states, None)


def prepend_state(states: torch.Tensor, h0: torch.Tensor | None) -> torch.Tensor:
    """Shift right; ``h0`` (paper ``h_0``) at t=0, else zeros.

    ``states[:, t]`` is the paper's ``h_{t+1}``. The previous state at code
    index t is paper's ``h_t``.
    """
    out = states.new_empty(states.shape)
    if h0 is None:
        out[:, 0] = 0
    else:
        out[:, 0] = h0
    out[:, 1:] = states[:, :-1]
    return out
