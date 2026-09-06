"""ParaRWKV7 — RWKV-7 Goose matrix-state delta monoid (Peng et al. 2025).

Linear recurrence in matrix state ``S`` (per head):

.. math::

    S_t = S_{t-1} G_t + v_t^{\\top} k_t,
    \\quad
    G_t = \\mathrm{diag}(w_t) - \\hat\\kappa_t^{\\top}(a_t \\odot \\hat\\kappa_t),

with ``w,a,κ,v,k,r`` from ``x`` only. Library pitch: factorized matrix-state
delta monoid (same class as ``ParaM2RNN``'s linear warm-start), sequential
oracle + optional associative ``(G,U)`` scan. Newton is redirected to
sequential / scan — there is no nonlinear fixed point in ``S``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.kernels.rwkv7_scan import (
    rwkv7_apply_factorized,
    rwkv7_associative_scan,
    rwkv7_build_g,
    rwkv7_outer_vk,
    rwkv7_readout,
)
from pararnn.weight_init import kaiming_uniform_linear_

log = logging.getLogger(__name__)

# Six vectors per head: w, a, κ, v, k, r (Peng et al. arXiv:2503.14456 §3).
_N_GATE = 6


class ParaRWKV7(nn.Module):
    """RWKV-7 Goose transition cell (linear in ``S``).

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    n_heads, d_head : int
        Per-head matrix state ``(d_head, d_head)``.
    hidden_size, d_h : int
        Readout width ``n_heads * d_head``.
    state_shape : tuple[int, int, int]
        ``(n_heads, d_head, d_head)`` for ``sequential_apply``.
    jac_structure : str
        Always ``'rwkv7'`` (linear monoid; Newton redirects).
    W_x : nn.Linear
        ``d_in → 6 * n_heads * d_head`` packing ``(w, a, κ, v, k, r)``.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        *,
        d_in: int | None = None,
        d_h: int | None = None,
        n_heads: int = 1,
        d_head: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Parameters
        ----------
        input_size, d_in : int
            Input width.
        hidden_size, d_h : int, optional
            Readout width. When set with ``n_heads``, requires
            ``d_h % n_heads == 0`` and sets ``d_head = d_h // n_heads``.
        n_heads : int, default=1
            Number of independent matrix states.
        d_head : int, optional
            Head width. Required when ``d_h`` / ``hidden_size`` omitted.
        device, dtype
            Parameter placement.
        """
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        d_in_r, d_head_r, d_h_r = _resolve_rwkv7_sizes(
            input_size, hidden_size, d_in=d_in, d_h=d_h, n_heads=n_heads, d_head=d_head
        )
        factory = {"device": device, "dtype": dtype}
        self.input_size = self.d_in = d_in_r
        self.n_heads = int(n_heads)
        self.d_head = int(d_head_r)
        self.hidden_size = self.d_h = int(d_h_r)
        self.state_slots = 1  # unused; state_shape wins
        self.state_shape = (self.n_heads, self.d_head, self.d_head)
        self.jac_structure = "rwkv7"
        self.mix = "rwkv7"
        out = _N_GATE * self.n_heads * self.d_head
        self.W_x = nn.Linear(self.d_in, out, bias=True, **factory)
        self.reset_parameters()
        log.info(
            "pararwkv7_init d_in=%s n_heads=%s d_head=%s d_h=%s",
            self.d_in,
            self.n_heads,
            self.d_head,
            self.d_h,
        )

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, n_heads={self.n_heads}, d_head={self.d_head}, d_h={self.d_h}"

    def reset_parameters(self) -> None:
        """Kaiming ``W_x`` (paper C.1 style on input affines); zero bias.

        Gate logits start near mid-sigmoid (``w≈0.5``, ``a≈0.5``) so the
        DPLR ``G`` stays contractive at init. Fallback: measure
        ``max|S|`` vs sequential on a fixed batch; if it grows with ``T``,
        shift ``w`` bias toward ``+2`` (stronger decay) before changing
        ``d_head``.
        """
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def project_wx(self, x: Tensor) -> Tensor:
        """``W_x(x)`` with shape ``(..., 6 n_heads d_head)``."""
        return self.W_x(x)

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``w, a, κ̂, v, k, r`` from ``x``. Shapes ``(..., n_heads, d_head)``."""
        return self._gates_from_wx(self.W_x(x))

    def step(self, s_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        """Advance one step. ``s_prev`` ``(..., n_heads, d_head, d_head)``."""
        w, a, kappa, v, k, _r = self._resolve_gates(x, wx)
        return rwkv7_apply_factorized(s_prev, w, a, kappa, v, k)

    def readout(self, s: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        """``y = flatten_heads(S @ r)`` with ``r`` from ``x`` / ``wx``."""
        _w, _a, _kappa, _v, _k, r = self._resolve_gates(x, wx)
        return rwkv7_readout(s, r)

    def scan_apply(
        self,
        x: Tensor,
        s0: Tensor | None = None,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Parallel ``(G,U)`` monoid scan; optional receptance readout.

        Parameters
        ----------
        x : Tensor
            ``(batch, time, d_in)``.
        s0 : Tensor, optional
            Initial state ``(batch, n_heads, d_head, d_head)``.
        return_state : bool, default=False
            If True, return ``(y, S)``; else ``y`` only
            (``y`` shape ``(batch, time, d_h)``).
        """
        w, a, kappa, v, k, r = self.project(x)
        g = rwkv7_build_g(w, a, kappa)
        u = rwkv7_outer_vk(v, k)
        s = rwkv7_associative_scan(g, u, s0=s0)
        y = rwkv7_readout(s, r)
        if return_state:
            return y, s
        return y

    def _resolve_gates(
        self, x: Tensor | None, wx: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if x is None and wx is None:
            raise ValueError("ParaRWKV7.step needs x or wx")
        if wx is None:
            assert x is not None
            wx = self.W_x(x)
        return self._gates_from_wx(wx)

    def _gates_from_wx(self, wx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        expected = _N_GATE * self.n_heads * self.d_head
        if wx.shape[-1] != expected:
            raise ValueError(
                f"ParaRWKV7 wx last dim must be 6*n_heads*d_head={expected}, got {wx.shape[-1]}"
            )
        raw = wx.reshape(*wx.shape[:-1], _N_GATE, self.n_heads, self.d_head)
        w = torch.sigmoid(raw[..., 0, :, :])
        a = torch.sigmoid(raw[..., 1, :, :])
        kappa = F.normalize(raw[..., 2, :, :], dim=-1, eps=1e-6)
        v = raw[..., 3, :, :]
        k = raw[..., 4, :, :]
        r = raw[..., 5, :, :]
        return w, a, kappa, v, k, r


def _resolve_rwkv7_sizes(
    input_size: int | None,
    hidden_size: int | None,
    *,
    d_in: int | None,
    d_h: int | None,
    n_heads: int,
    d_head: int | None,
) -> tuple[int, int, int]:
    if d_head is not None and d_head < 1:
        raise ValueError(f"d_head must be positive, got {d_head}")
    if hidden_size is None and d_h is None:
        if d_head is None:
            raise TypeError("ParaRWKV7 needs d_head or hidden_size / d_h")
        d_in_r, _ = resolve_layer_sizes(
            input_size, d_head * n_heads, d_in=d_in, d_h=d_head * n_heads
        )
        return d_in_r, d_head, d_head * n_heads
    d_in_r, d_h_r = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
    if d_h_r % n_heads != 0:
        raise ValueError(f"d_h={d_h_r} must be divisible by n_heads={n_heads}")
    inferred = d_h_r // n_heads
    if d_head is not None and d_head != inferred:
        raise ValueError(f"d_head={d_head} conflicts with d_h/n_heads={inferred}")
    return d_in_r, inferred, d_h_r
