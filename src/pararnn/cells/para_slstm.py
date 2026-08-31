"""sLSTM (Beck et al. 2024) as a Newton cell — not Apple's ParaLSTM.

Equations: ADD_TASK / xLSTM §2. Stabilizer ``max``, exp input/forget, normalizer
``n``, memory mixing ``R h``. Default ``mix='diag'`` is channelwise R (cheap
enough to prototype Newton). ``mix='dense'`` is the real mixing cell; Jacobian
is full ``4 d_h`` and the scan is ``O(T (4d)^3)`` — tests stay tiny.
``jac_structure`` is ``block4`` (diag mix) or ``dense`` (full mix).

Not fused. Not FlashRNN. ``K=3`` is a ParaGRU fact, not a promise here.
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

_MIX = ("diag", "dense")


class ParaSLSTM(nn.Module):
    """Four-slot sLSTM: state ``(..., 4, d_h)`` = (c, n, m, h).

    ``max_recurrent_norm=0.5``: same clip as ParaGRU/LSTM App. C.1, for Newton
    stability on the diagonal mix. If residual stays large, try 0.25 or
    ``NewtonConfig(omega=0.5)`` (Gonzalez et al. ELK-style damping) before
    raising K. ``eps=1e-6``: xLSTM-style floor on ``n`` in ``h = o * c / n``.
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        *,
        mix: str = "diag",
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
        self.jac_structure = "block4" if mix == "diag" else "dense"
        self.max_recurrent_norm = max_recurrent_norm
        self.eps = eps
        # Pre-activations: input, forget, candidate, output (ADD_TASK §2.1).
        self.W_x = nn.Linear(d_in, 4 * d_h, bias=True)
        if mix == "diag":
            self.R = nn.Parameter(torch.empty(4, d_h))
            self.R_dense = None
        else:
            self.R = None
            self.R_dense = nn.Linear(d_h, 4 * d_h, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.R is not None:
            xavier_gaussian_vec_(self.R)
        else:
            nn.init.orthogonal_(self.R_dense.weight, gain=0.25)

    def clipped_r(self) -> Tensor:
        if self.R is None:
            raise TypeError("clipped_r is for mix='diag'")
        if self.max_recurrent_norm is None:
            return self.R
        return self.R.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def step(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> Tensor:
        """One sLSTM step. ``state_prev`` is ``(..., 4, d_h)``."""
        c = state_prev[..., SLSTM_CELL, :]
        n = state_prev[..., SLSTM_NORMALIZER, :]
        m = state_prev[..., SLSTM_STABILIZER, :]
        h = state_prev[..., SLSTM_HIDDEN, :]
        if wx is None:
            wx = self.W_x(x)
        pre = wx + self._recurrent(h)
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

    def _recurrent(self, h: Tensor) -> Tensor:
        if self.R is not None:
            return (self.clipped_r() * h.unsqueeze(-2)).reshape(*h.shape[:-1], 4 * self.d_h)
        return self.R_dense(h)
