"""ParaCfC — Liquid-style CfC brick with irregular Δt and diagonal nonlinear mix.

Research-variant vs Hasani et al. 2022 eq. (10) / ``ncps.torch.CfCCell``:
see ``docs/architecture/para_cfc.md``.

``x[..., :-1]`` are features; ``x[..., -1:]`` is ``Δt`` (``d_in >= 2``).

.. math::

    a_t = \\sigma\\bigl(-\\mathrm{softplus}(f(x_t))\\,\\Delta t_t\\bigr),\\quad
    h_t = a_t \\odot h_{t-1}
    + (1-a_t)\\odot\\tanh(W_c x_t + u \\odot h_{t-1}).

For fused Newton, ``project_wx`` returns ``(B, T, 3 d_h) = (f, c, Δt)``.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_

log = logging.getLogger(__name__)

_DEFAULT_CAP = object()
_DT_EPS = 1e-4


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    return 1.0 - act.square()


class _CfCActs(NamedTuple):
    h_prev: Tensor
    a: Tensor
    n: Tensor
    u: Tensor
    h_new: Tensor


def split_cfc_input(x: Tensor) -> tuple[Tensor, Tensor]:
    if x.shape[-1] < 2:
        raise ValueError(f"ParaCfC needs d_in >= 2 (features + Δt), got {x.shape[-1]}")
    return x[..., :-1], x[..., -1:].clamp_min(_DT_EPS)


class ParaCfC(nn.Module):
    """Nonlinear CfC; Δt is the last channel of ``x``."""

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
        super().__init__()
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        if input_size < 2:
            raise ValueError(f"ParaCfC d_in must be >= 2, got {input_size}")
        factory_kwargs = {"device": device, "dtype": dtype}
        if max_recurrent_norm is _DEFAULT_CAP:
            max_recurrent_norm = 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.feature_size = input_size - 1
        self.state_slots = 1
        self.mix = "diag"
        self.jac_structure = "diag"
        self.max_recurrent_norm = max_recurrent_norm  # type: ignore[assignment]
        self.u = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.W_x = nn.Linear(self.feature_size, 2 * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()
        log.info(
            "paracfc_init d_in=%s feature_size=%s d_h=%s",
            input_size,
            self.feature_size,
            hidden_size,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, feature_size={self.feature_size}, "
            f"max_recurrent_norm={self.max_recurrent_norm}"
        )

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        xavier_gaussian_vec_(self.u)

    def clipped_u(self) -> Tensor:
        if self.max_recurrent_norm is None:
            return self.u
        return self.u.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def project_wx(self, x: Tensor) -> Tensor:
        """Fused pack ``(f_pre, c_x, Δt)`` with shape ``(..., 3 d_h)``."""
        feat, dt = split_cfc_input(x)
        fc = self.W_x(feat)
        dt_b = dt.expand(*feat.shape[:-1], self.d_h)
        return torch.cat((fc, dt_b), dim=-1)

    def step(self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        return self._recurrence(h_prev, x, wx=wx).h_new

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        acts = self._recurrence(h_prev, x, wx=wx)
        jac = acts.a + (1.0 - acts.a) * _tanh_prime_from_act(acts.n) * acts.u
        return acts.h_new, jac

    def _recurrence(self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None) -> _CfCActs:
        if x is None and wx is None:
            raise ValueError("ParaCfC.step needs x or wx")
        if wx is not None and wx.shape[-1] == 3 * self.d_h:
            f_pre, cx, dt = wx.chunk(3, dim=-1)
        else:
            if x is None:
                raise ValueError("ParaCfC needs x to read Δt when wx is 2-wide")
            feat, dt = split_cfc_input(x)
            if wx is None:
                wx = self.W_x(feat)
            f_pre, cx = wx.chunk(2, dim=-1)
            dt = dt.expand_as(f_pre)
        a = torch.sigmoid(-F.softplus(f_pre) * dt)
        u = self.clipped_u()
        n = torch.tanh(cx + u * h_prev)
        h_new = a * h_prev + (1.0 - a) * n
        return _CfCActs(h_prev=h_prev, a=a, n=n, u=u, h_new=h_new)
