"""ParaHopfield — recurrent Modern-Hopfield slot with dense Newton Jacobian.

One-step continuous Hopfield update with input-conditioned pattern matrices
(Ramsauer et al. 2021 style; patterns from ``x_t``, query from ``h_{t-1}``):

.. math::

    h_t = V_t\\,\\mathrm{softmax}(\\beta K_t h_{t-1}),

where ``K_t, V_t ∈ R^{d_h × d_h}`` are reshaped halves of ``W_x(x_t)``.
Softmax couples channels, so ``jac_structure='dense'`` and Newton uses
``scan_dense`` (eager or Triton). Cap ``d_h ≤ 32`` for the dense path;
larger widths need a head-blocked recipe (parked).
"""

from __future__ import annotations

import logging
import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_

log = logging.getLogger(__name__)

# Dense ``d×d`` scan / Autograd jacrev cost grows as ``O(d_h^3)`` per step.
# Lab recipe: keep ``d_h ≤ 32``; above that emit a one-shot warning and expect
# a head-blocked follow-up (parked for a later minor).
_DH_DENSE_CAP = 32


class ParaHopfield(nn.Module):
    """Recurrent Hopfield / soft-attention slot with dense Jacobian.

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    hidden_size, d_h : int
        Hidden / pattern width (recommend ``≤ 32``).
    state_slots : int
        Always ``1``.
    jac_structure : str
        Always ``'dense'``.
    beta : float
        Inverse temperature on ``K h``. Default ``1/sqrt(d_h)`` (attention
        scale; Ramsauer et al. Modern Hopfield). Newton depth is **not** a
        fixed App. A constant: use ``NewtonConfig(max_iters=None)`` for the
        measured ``K*(T)`` envelope (`hopfield_auto_newton_iters`), or pin
        ``max_iters=int`` / ``newton_iters_by_t={…}``. Lab campaign
        (τ=1e-4, T≤4096): H1-ish with short-T K*∈{1,2}. If residual stays
        high after the schedule, try ``β ∈ {0.5, 1, 2}/sqrt(d_h)`` before
        raising the pin.
    W_x : nn.Linear
        ``d_in → 2 d_h²`` packing flattened ``K`` and ``V``.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        *,
        d_in: int | None = None,
        d_h: int | None = None,
        beta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Parameters
        ----------
        input_size, hidden_size : int or None
            Widths; aliases ``d_in`` / ``d_h``.
        beta : float or None, optional
            Softmax inverse temperature. Default ``1 / sqrt(d_h)``.
        device, dtype
            Parameter placement.
        """
        super().__init__()
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        factory_kwargs = {"device": device, "dtype": dtype}
        if beta is None:
            # Attention-scale default (Vaswani et al.); Modern Hopfield β is the
            # same inverse-temperature knob (Ramsauer et al. 2021). Fallback
            # search: measure Newton residual vs β on a fixed batch
            # (log-spaced around 1/sqrt(d_h)); raise K only after β is settled.
            beta = 1.0 / math.sqrt(float(hidden_size))
        if beta <= 0.0:
            raise ValueError(f"ParaHopfield beta must be positive, got {beta}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.state_slots = 1
        self.mix = "dense"
        self.jac_structure = "dense"
        self.beta = float(beta)
        self.W_x = nn.Linear(input_size, 2 * hidden_size * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()
        if hidden_size > _DH_DENSE_CAP:
            warnings.warn(
                f"ParaHopfield d_h={hidden_size} > {_DH_DENSE_CAP}: dense "
                "Jacobian + scan_dense is O(d_h³); prefer d_h≤32 or a "
                "head-blocked recipe (parked).",
                UserWarning,
                stacklevel=2,
            )
        log.info(
            "parahopfield_init d_in=%s d_h=%s beta=%s",
            input_size,
            hidden_size,
            self.beta,
        )

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}, beta={self.beta}"

    def reset_parameters(self) -> None:
        """Kaiming ``W_x`` (paper C.1 style on input affines); zero bias."""
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def project_wx(self, x: Tensor) -> Tensor:
        """``W_x(x)`` with shape ``(..., 2 d_h²)`` for Newton reuse."""
        return self.W_x(x)

    def step(self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        return self._recurrence(h_prev, x, wx=wx)[0]

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """``(h_new, J)`` with dense ``J = V (diag(a)-aaᵀ) (β K)``."""
        h_new, K, V, a = self._recurrence(h_prev, x, wx=wx)
        # Softmax Jacobian over pattern dim: diag(a) - a aᵀ.
        outer = a.unsqueeze(-1) * a.unsqueeze(-2)
        j_s = torch.diag_embed(a) - outer
        jac = V @ j_s @ (self.beta * K)
        return h_new, jac

    def _kv_from_wx(self, wx: Tensor) -> tuple[Tensor, Tensor]:
        d = self.d_h
        expected = 2 * d * d
        if wx.shape[-1] != expected:
            raise ValueError(
                f"ParaHopfield wx last dim must be 2*d_h²={expected}, got {wx.shape[-1]}"
            )
        kv = wx.reshape(*wx.shape[:-1], 2, d, d)
        return kv[..., 0, :, :], kv[..., 1, :, :]

    def _recurrence(
        self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x is None and wx is None:
            raise ValueError("ParaHopfield.step needs x or wx")
        if wx is None:
            assert x is not None
            wx = self.W_x(x)
        K, V = self._kv_from_wx(wx)
        # logits = β K h ; a = softmax(logits); h_new = V a
        logits = self.beta * torch.matmul(K, h_prev.unsqueeze(-1)).squeeze(-1)
        a = F.softmax(logits, dim=-1)
        h_new = torch.matmul(V, a.unsqueeze(-1)).squeeze(-1)
        return h_new, K, V, a
