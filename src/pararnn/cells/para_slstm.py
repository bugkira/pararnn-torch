"""sLSTM (Beck et al. 2024) as a Newton cell — not Apple's ParaLSTM.

Equations: ADD_TASK / xLSTM §2. Stabilizer ``max``, exp input/forget, normalizer
``n``, memory mixing ``R h``. ``mix='diag'`` is channelwise (Newton prototype).
``mix='head'`` is the xLSTM compromise: dense ``R`` inside a head, block-
diagonal across heads. ``mix='dense'`` mixes the full ``d_h``; scan is
``O(T (4d)^3)`` — tests stay tiny.

Not fused. Not FlashRNN. ``K=3`` is a ParaGRU fact, not a promise here.
Measured prototype (diag, seed 101): K=4, omega=1, clip=0.5. ``omega=0.5``
kills the K=4 snap — do not copy ELK as the sLSTM default.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn.init import kaiming_uniform_linear_, xavier_gaussian_vec_
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)

_MIX = ("diag", "dense", "head")
_JAC = {"diag": "block4", "dense": "dense", "head": "head"}


class ParaSLSTM(nn.Module):
    """Four-slot sLSTM: state ``(..., 4, d_h)`` = (c, n, m, h).

    ``max_recurrent_norm=0.5``: same elementwise clip as ParaGRU/LSTM App. C.1.
    For ``mix='head'`` it clamps each block entry (not a spectral bound).
    ``eps=1e-6``: xLSTM-style floor on ``n`` in ``h = o * c / n``.
    ``mix='head'`` requires ``n_heads`` that divides ``d_h``.
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        *,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mix not in _MIX:
            raise ValueError(f"mix must be one of {_MIX}, got {mix!r}")
        self.d_in = d_in
        self.d_h = d_h
        self.state_slots = SLSTM_SLOTS
        self.hidden_slot = SLSTM_HIDDEN
        self.mix = mix
        self.jac_structure = _JAC[mix]
        self.max_recurrent_norm = max_recurrent_norm
        self.eps = eps
        self.n_heads = n_heads
        self.d_head = None
        # Pre-activations: input, forget, candidate, output (ADD_TASK §2.1).
        self.W_x = nn.Linear(d_in, 4 * d_h, bias=True)
        self.R = None
        self.R_dense = None
        self.R_head = None
        if mix == "diag":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R = nn.Parameter(torch.empty(4, d_h))
        elif mix == "dense":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R_dense = nn.Linear(d_h, 4 * d_h, bias=False)
        else:
            if n_heads is None or n_heads < 1 or d_h % n_heads != 0:
                raise ValueError(
                    f"mix='head' needs n_heads that divides d_h={d_h}, got {n_heads!r}"
                )
            self.d_head = d_h // n_heads
            self.R_head = nn.Parameter(torch.empty(4, n_heads, self.d_head, self.d_head))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.R is not None:
            xavier_gaussian_vec_(self.R)
        elif self.R_dense is not None:
            nn.init.orthogonal_(self.R_dense.weight, gain=0.25)
        else:
            for g in range(4):
                for hd in range(self.n_heads):
                    nn.init.orthogonal_(self.R_head[g, hd], gain=0.25)

    def clipped_r(self) -> Tensor:
        if self.R is None:
            raise TypeError("clipped_r is for mix='diag'")
        if self.max_recurrent_norm is None:
            return self.R
        return self.R.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def clipped_r_head(self) -> Tensor:
        if self.R_head is None:
            raise TypeError("clipped_r_head is for mix='head'")
        if self.max_recurrent_norm is None:
            return self.R_head
        return self.R_head.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def step(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> Tensor:
        """One sLSTM step. ``state_prev`` is ``(..., 4, d_h)``."""
        if wx is None:
            wx = self.W_x(x)
        h = state_prev[..., SLSTM_HIDDEN, :]
        return self._step_from_pre(state_prev, wx + self._recurrent(h))

    def step_head(self, state: Tensor, wx_head: Tensor, r: Tensor) -> Tensor:
        """One head: ``state`` / ``wx_head`` are ``(4, d_head)``; ``r`` is ``(4, d_head, d_head)``."""
        h = state[SLSTM_HIDDEN]
        pre = wx_head + torch.einsum("d,gde->ge", h, r)
        return self._step_from_pre(state, pre.reshape(4 * state.shape[-1]))

    def _recurrent(self, h: Tensor) -> Tensor:
        if self.R is not None:
            return (self.clipped_r() * h.unsqueeze(-2)).reshape(*h.shape[:-1], 4 * self.d_h)
        if self.R_head is not None:
            h_v = h.reshape(*h.shape[:-1], self.n_heads, self.d_head)
            rec = torch.einsum("...nd,gnde->...gne", h_v, self.clipped_r_head())
            return rec.reshape(*h.shape[:-1], 4 * self.d_h)
        return self.R_dense(h)

    def _step_from_pre(self, state_prev: Tensor, pre: Tensor) -> Tensor:
        c = state_prev[..., SLSTM_CELL, :]
        n = state_prev[..., SLSTM_NORMALIZER, :]
        m = state_prev[..., SLSTM_STABILIZER, :]
        z_i, z_f, z_z, z_o = pre.chunk(4, dim=-1)
        m_new = torch.maximum(z_f + m, z_i)
        i_t = torch.exp(z_i - m_new)
        f_t = torch.exp(z_f + m - m_new)
        z = torch.tanh(z_z)
        n_new = f_t * n + i_t
        c_new = f_t * c + i_t * z
        o = torch.sigmoid(z_o)
        h_new = o * (c_new / (n_new + self.eps))
        return torch.stack((c_new, n_new, m_new, h_new), dim=-2)
