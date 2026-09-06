"""ParaNLRU — nonlinear RG-LRU-style cell with diagonal mix.

Inspired by Griffin / RecurrentGemma RG-LRU (De et al. 2024), where the
recurrent core is linear in ``h``. This cell puts ``tanh`` and a diagonal
``u`` *inside* the step while keeping the gate ``a_t = a_t(x_t)`` only, so
the Newton Jacobian stays channelwise diagonal.

Reference math: ``docs/internal/paranlru-jacobian.md`` (gitignored lab note).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_

log = logging.getLogger(__name__)

_DEFAULT_CAP = object()


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    """tanh'(pre) when ``act = tanh(pre)``: ``1 - act^2``."""
    return 1.0 - act.square()


class _NLRUActs(NamedTuple):
    h_prev: Tensor
    a: Tensor
    n: Tensor
    u: Tensor
    h_new: Tensor


class ParaNLRU(nn.Module):
    """Nonlinear LRU-style recurrence with input-only gate and diagonal ``u``.

    .. math::

        a_t = \\sigma(W_a x_t + b_a),\\quad
        h_t = a_t \\odot h_{t-1}
        + (1-a_t)\\odot\\tanh(W_c x_t + b_c + u \\odot h_{t-1}).

    ``jac_structure='diag'``. Fused Newton reuses the diag scan class.

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    hidden_size, d_h : int
        Hidden width.
    state_slots : int
        Always ``1``.
    jac_structure : str
        Always ``'diag'``.
    max_recurrent_norm : float or None
        App. C.1 clamp on ``u`` to ``[-cap, cap]``. Default ``0.5``.
    u : Parameter
        Diagonal recurrent mix, shape ``(d_h,)``.
    W_x : nn.Linear
        ``d_in → 2 * d_h`` packing gate and candidate affines.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        *,
        d_in: int | None = None,
        d_h: int | None = None,
        max_recurrent_norm: float | object | None = _DEFAULT_CAP,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Parameters
        ----------
        input_size, hidden_size : int or None
            Widths; aliases ``d_in`` / ``d_h``.
        max_recurrent_norm : float or None, optional
            Elementwise clamp of ``u`` (App. C.1). Default ``0.5``.
        device, dtype
            Parameter placement.
        """
        super().__init__()
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        factory_kwargs = {"device": device, "dtype": dtype}
        if max_recurrent_norm is _DEFAULT_CAP:
            max_recurrent_norm = 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.state_slots = 1
        self.mix = "diag"
        self.jac_structure = "diag"
        self.max_recurrent_norm = max_recurrent_norm  # type: ignore[assignment]
        self.u = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.W_x = nn.Linear(input_size, 2 * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()
        log.info(
            "paranlru_init d_in=%s d_h=%s max_recurrent_norm=%s",
            input_size,
            hidden_size,
            self.max_recurrent_norm,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, max_recurrent_norm={self.max_recurrent_norm}"
        )

    def reset_parameters(self) -> None:
        """Kaiming ``W_x``; Xavier-Gaussian ``u`` (App. C.1 style)."""
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        xavier_gaussian_vec_(self.u)

    def clipped_u(self) -> Tensor:
        """App. C.1 clamp of the diagonal recurrent vector ``u``."""
        if self.max_recurrent_norm is None:
            return self.u
        return self.u.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def step(self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        """One NLRU step.

        Parameters
        ----------
        h_prev : Tensor
            Previous hidden. Shape ``(..., d_h)``.
        x : Tensor, optional
            Input. Shape ``(..., d_in)``. Ignored when ``wx`` is set.
        wx : Tensor, optional
            Precomputed ``W_x(x)``. Shape ``(..., 2 * d_h)``.

        Returns
        -------
        h_new : Tensor
            Shape ``(..., d_h)``.
        """
        return self._recurrence(h_prev, x, wx=wx).h_new

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Step plus channelwise-diagonal Jacobian ``(..., d_h)``."""
        acts = self._recurrence(h_prev, x, wx=wx)
        n_p = _tanh_prime_from_act(acts.n)
        jac = acts.a + (1.0 - acts.a) * n_p * acts.u
        return acts.h_new, jac

    def _recurrence(self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None) -> _NLRUActs:
        if wx is None:
            if x is None:
                raise ValueError("ParaNLRU.step needs x or wx")
            wx = self.W_x(x)
        ax, cx = wx.chunk(2, dim=-1)
        a = torch.sigmoid(ax)
        u = self.clipped_u()
        n = torch.tanh(cx + u * h_prev)
        h_new = a * h_prev + (1.0 - a) * n
        return _NLRUActs(h_prev=h_prev, a=a, n=n, u=u, h_new=h_new)
