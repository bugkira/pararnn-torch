"""ParaCfC — Liquid-style CfC brick with irregular Δt and diagonal nonlinear mix.

Research-variant vs Hasani et al. 2022 eq. (10) / ``ncps.torch.CfCCell``:
see ``docs/architecture/para_cfc.md``.

``x[..., :-1]`` are features; ``x[..., -1:]`` is ``Δt`` (``d_in >= 2``).

Default (``gate_mix='input'``)::

    a_t = exp(-softplus(f(x_t)) Δt_t)          # → 1 as Δt → 0 (ODE continuity)
    h_t = a_t ⊙ h_{t-1} + (1-a_t) ⊙ tanh(W_c x_t + u ⊙ h_{t-1})

``gate_mix='diag_h'`` (quasi-linear liquid rate; Jacobian stays channelwise diag)::

    a_t = exp(-softplus(f(x_t) + v ⊙ h_{t-1}) Δt_t)

Earlier drafts used ``a = σ(-softplus·Δt)``, which forces ``a ≤ 0.5`` and
erases memory as Δt → 0 (``σ(0)=0.5``). Fused Newton / packed VJP match the
exponential gate.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_

log = logging.getLogger(__name__)

_DEFAULT_CAP = object()
_DT_EPS = 1e-4
GateMix = Literal["input", "diag_h"]


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    return 1.0 - act.square()


class _CfCActs(NamedTuple):
    h_prev: Tensor
    a: Tensor
    n: Tensor
    u: Tensor
    v: Tensor | None
    soft: Tensor
    soft_prime: Tensor
    dt: Tensor
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
        gate_mix: GateMix = "input",
        max_recurrent_norm: float | object | None = _DEFAULT_CAP,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        if input_size < 2:
            raise ValueError(f"ParaCfC d_in must be >= 2, got {input_size}")
        if gate_mix not in ("input", "diag_h"):
            raise ValueError(f"gate_mix must be 'input' or 'diag_h', got {gate_mix!r}")
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
        self.gate_mix: GateMix = gate_mix
        self.jac_structure = "diag"
        self.max_recurrent_norm = max_recurrent_norm  # type: ignore[assignment]
        self.u = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        if gate_mix == "diag_h":
            self.v = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("v", None)
        self.W_x = nn.Linear(self.feature_size, 2 * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()
        log.info(
            "paracfc_init d_in=%s feature_size=%s d_h=%s gate_mix=%s",
            input_size,
            self.feature_size,
            hidden_size,
            gate_mix,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, feature_size={self.feature_size}, "
            f"gate_mix={self.gate_mix}, max_recurrent_norm={self.max_recurrent_norm}"
        )

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        xavier_gaussian_vec_(self.u)
        if self.v is not None:
            xavier_gaussian_vec_(self.v)

    def clipped_u(self) -> Tensor:
        if self.max_recurrent_norm is None:
            return self.u
        return self.u.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def clipped_v(self) -> Tensor | None:
        if self.v is None:
            return None
        if self.max_recurrent_norm is None:
            return self.v
        return self.v.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

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
        # Candidate branch: (1-a) · n' · u.
        jac = acts.a + (1.0 - acts.a) * _tanh_prime_from_act(acts.n) * acts.u
        if acts.v is not None:
            # a = exp(-soft·dt), soft = softplus(f+v⊙h)
            # ∂a/∂h = a · (-dt) · softplus'(z) · v
            da_dh = acts.a * (-acts.dt) * acts.soft_prime * acts.v
            jac = jac + da_dh * (acts.h_prev - acts.n)
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
        u = self.clipped_u()
        v = self.clipped_v()
        z = f_pre + v * h_prev if v is not None else f_pre
        soft = F.softplus(z)
        soft_prime = torch.sigmoid(z)
        # ODE continuity: lim_{Δt→0} a = 1 (sigmoid(-soft·Δt) forced a≤0.5).
        a = torch.exp(-soft * dt)
        n = torch.tanh(cx + u * h_prev)
        h_new = a * h_prev + (1.0 - a) * n
        return _CfCActs(
            h_prev=h_prev,
            a=a,
            n=n,
            u=u,
            v=v,
            soft=soft,
            soft_prime=soft_prime,
            dt=dt,
            h_new=h_new,
        )
