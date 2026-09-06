"""ParaM2RNN — matrix-state nonlinear RNN (Mishra et al., arXiv:2603.14360).

Eager research cell: state ``H ∈ ℝ^{K×V}``, transition
``H ← f H + (1-f) tanh(H W + k vᵀ)`` with input-only ``k,v,f``. Parallel
train uses factorized Newton (``newton_m2rnn_factorized``); dense ``(KV)²``
is the oracle only.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.kernels.m2rnn_factor import m2rnn_gates

log = logging.getLogger(__name__)


class ParaM2RNN(nn.Module):
    """Matrix-to-matrix RNN cell (M²RNN recurrence core).

    MVP projections: one linear ``d_in → K+V+1`` for ``(k, v, f_logit)``.
    Paper adds short causal conv + SiLU on ``q,k,v`` and a richer forget map;
    those sit outside the Newton kernel once ``k,v,f`` are frozen per ``t``.

    Attributes
    ----------
    k_dim, v_dim : int
        State shape ``(K, V)``.
    state_shape : tuple[int, int]
        ``(k_dim, v_dim)`` for solvers / ``sequential_apply``.
    d_h : int
        ``K * V`` (flat width; protocol alias).
    W : Parameter
        Right-multiply transition ``(V, V)``. Default init: identity.
    W_x : nn.Linear
        Input affine to ``K + V + 1``.
    """

    def __init__(
        self,
        input_size: int | None = None,
        *,
        k_dim: int,
        v_dim: int,
        d_in: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        d_in_r, _ = resolve_layer_sizes(input_size, k_dim * v_dim, d_in=d_in, d_h=k_dim * v_dim)
        if k_dim < 1 or v_dim < 1:
            raise ValueError(f"k_dim and v_dim must be positive, got {k_dim}, {v_dim}")
        factory = {"device": device, "dtype": dtype}
        self.input_size = self.d_in = d_in_r
        self.k_dim = int(k_dim)
        self.v_dim = int(v_dim)
        self.hidden_size = self.d_h = self.k_dim * self.v_dim
        self.state_slots = 1  # unused; state_shape wins
        self.state_shape = (self.k_dim, self.v_dim)
        self.jac_structure = "m2rnn"
        self.W = nn.Parameter(torch.empty(self.v_dim, self.v_dim, **factory))
        self.W_x = nn.Linear(self.d_in, self.k_dim + self.v_dim + 1, **factory)
        self.reset_parameters()
        log.info(
            "para_m2rnn k_dim=%d v_dim=%d d_in=%d",
            self.k_dim,
            self.v_dim,
            self.d_in,
        )

    def reset_parameters(self) -> None:
        # Paper runs often init W as identity (arXiv:2603.14360).
        with torch.no_grad():
            self.W.copy_(torch.eye(self.v_dim, device=self.W.device, dtype=self.W.dtype))
        nn.init.xavier_uniform_(self.W_x.weight)
        if self.W_x.bias is not None:
            nn.init.zeros_(self.W_x.bias)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``k, v, f`` from ``x``. ``f`` in ``(0,1)`` via sigmoid."""
        raw = self.W_x(x)
        k = raw[..., : self.k_dim]
        v = raw[..., self.k_dim : self.k_dim + self.v_dim]
        f = torch.sigmoid(raw[..., -1])
        return k, v, f

    def step(self, h_prev: Tensor, x: Tensor, *, wx: Tensor | None = None) -> Tensor:
        """Advance one step. ``h_prev`` ``(..., K, V)``."""
        if wx is None:
            k, v, f = self.project(x)
        else:
            k = wx[..., : self.k_dim]
            v = wx[..., self.k_dim : self.k_dim + self.v_dim]
            f = torch.sigmoid(wx[..., -1])
        h_new, _ = m2rnn_gates(h_prev, k, v, f, self.W)
        return h_new

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, k_dim={self.k_dim}, v_dim={self.v_dim}"
