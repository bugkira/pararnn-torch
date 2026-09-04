"""Index and layout conventions.

Paper is 1-based: ``l = 1..L``, ``h_0 = 0``, ``h_l = f(h_{l-1}, x_l)``.
Code is 0-based: ``t = 0..T-1`` is paper ``l = t+1``. Batch-first inside
cells, solvers, and kernels.

    x: (B, T, input_size)
    GRU h: (B, T, hidden_size)
    LSTM: (B, T, 2, hidden_size) slots [c, h]
        # nn.LSTM hidden tuple is (h, c)
    sLSTM: (B, T, 4, hidden_size) slots [c, n, m, h]
        # Beck et al. 2024

    F_l = h_l - f(h_{l-1}, x_l)
    J_t = ∂f/∂h_prev at (h_{t-1}, x_t)   # eq. 2.1–2.3
    δh_t = J_t δh_{t-1} + (f(h_{t-1}, x_t) - h_t),  δh_{<0} = 0

Work-efficient Blelloch exclusive scan on 0-based ``t``, pad to ``2^k``
with identity ``(I, 0)``. Ragged batches pack time to ``N = cu_seqlens[-1]``
with ``x`` shaped ``(1, N, …)``; the scan is segmented (head flags at
``cu_seqlens[:-1]``).

``ParaRNN`` only: ``batch_first=False`` permutes ``x`` and the output like
``nn.LSTM``; ``h0`` stays batch-leading. ``hidden_layout='pytorch'`` swaps
LSTM slots 0/1 on ``h0`` and on ``return_hidden``'s last state via
``swap_lstm_ch``. Kernels stay paper ``[c, h]``.
"""

from __future__ import annotations

import torch

LSTM_CELL = 0
LSTM_HIDDEN = 1


def swap_lstm_ch(state: torch.Tensor) -> torch.Tensor:
    """Swap LSTM slots 0/1. Paper ``(c, h)`` ↔ ``nn.LSTM`` ``(h, c)``.

    ``state`` is ``(..., 2, hidden_size)`` — ``h0`` or last step.
    """
    if state.ndim < 2 or state.shape[-2] != 2:
        raise ValueError(f"swap_lstm_ch expects (..., 2, d_h), got {tuple(state.shape)}")
    return state.flip(-2)


# sLSTM (Beck et al. 2024): four-slot state (c, n, m, h).
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


def validate_cu_seqlens(cu_seqlens: torch.Tensor, time: int) -> torch.Tensor:
    """FlashAttention-style exclusive prefix: ``(S+1,)``, ``[0] = 0``, last = ``time``."""
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be (S+1,), got {tuple(cu_seqlens.shape)}")
    cs = cu_seqlens.to(dtype=torch.long)
    if int(cs[0]) != 0:
        raise ValueError(f"cu_seqlens[0] must be 0, got {int(cs[0])}")
    if int(cs[-1]) != int(time):
        raise ValueError(f"cu_seqlens[-1] must equal time={time}, got {int(cs[-1])}")
    if torch.any(cs[1:] <= cs[:-1]):
        raise ValueError("cu_seqlens must be strictly increasing")
    return cs


def segment_start_flags(cu_seqlens: torch.Tensor, time: int, *, batch: int = 1) -> torch.Tensor:
    """Bool ``(batch, time)``: True at the first token of each packed sequence."""
    cs = validate_cu_seqlens(cu_seqlens, time)
    flags = torch.zeros(batch, time, dtype=torch.bool, device=cs.device)
    flags[:, cs[:-1]] = True
    return flags


def prepend_state_ragged(
    states: torch.Tensor,
    h0: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Like ``prepend_state``, but ``h0[s]`` is the previous state at ``cu_seqlens[s]``.

    Packed ``states`` is ``(1, N, …)``. ``h0`` is ``(S, …)`` or ``None`` (zeros).
    """
    if states.shape[0] != 1:
        raise ValueError(f"ragged prepend needs packed batch 1, got batch={states.shape[0]}")
    time = states.shape[1]
    cs = validate_cu_seqlens(cu_seqlens, time)
    out = states.new_empty(states.shape)
    out[:, 1:] = states[:, :-1]
    starts = cs[:-1]
    if h0 is None:
        out[:, starts] = 0
    else:
        n_seq = int(cs.numel()) - 1
        if h0.shape[0] != n_seq:
            raise ValueError(f"h0 batch {h0.shape[0]} != n_seq {n_seq}")
        out[0, starts] = h0
    return out
