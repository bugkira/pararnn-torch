"""CIFG peephole ParaLSTM — Danieli et al. 2025 eq. 3.1b, A_*, C_* diagonal (eq. 3.3)."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.layout import LSTM_CELL, LSTM_HIDDEN
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_


def _sigmoid_prime_from_act(gate: Tensor) -> Tensor:
    return gate * (1.0 - gate)


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    return 1.0 - act.square()


class ParaLSTM(nn.Module):
    """Coupled input-forget LSTM with peepholes (Greff et al. 2017), diagonal A/C.

    State layout ``(..., 2, d_h)``: index 0 = cell ``c``, 1 = hidden ``h``.
    Candidate ``z`` uses tanh (σ_z in the paper). Forget/output: sigmoid.
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        *,
        max_recurrent_norm: float | None = 0.5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_in = d_in
        self.d_h = d_h
        self.state_slots = 2
        self.hidden_slot = LSTM_HIDDEN
        self.max_recurrent_norm = max_recurrent_norm

        # a_f, a_z, a_o and peepholes c_f, c_o (paper eq. 3.3).
        self.a_f = nn.Parameter(torch.empty(d_h, **factory_kwargs))
        self.a_z = nn.Parameter(torch.empty(d_h, **factory_kwargs))
        self.a_o = nn.Parameter(torch.empty(d_h, **factory_kwargs))
        self.c_f = nn.Parameter(torch.empty(d_h, **factory_kwargs))
        self.c_o = nn.Parameter(torch.empty(d_h, **factory_kwargs))
        self.W_x = nn.Linear(d_in, 3 * d_h, bias=True, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.a_f, self.a_z, self.a_o, self.c_f, self.c_o):
            xavier_gaussian_vec_(p)
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def _clip(self, t: Tensor) -> Tensor:
        if self.max_recurrent_norm is None:
            return t
        return t.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def clipped_recurrent(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """App. C.1 clip of ``a_f, a_z, a_o, c_f, c_o``."""
        return (
            self._clip(self.a_f),
            self._clip(self.a_z),
            self._clip(self.a_o),
            self._clip(self.c_f),
            self._clip(self.c_o),
        )

    def step(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> Tensor:
        """One step. Does not build the 2×2 Jacobian (sequential unroll / decode).

        ``wx`` is optional ``W_x(x)`` (eq. 3.1, independent of state).
        """
        return self._recurrence(state_prev, x, wx=wx).state_new

    def step_with_jacobian(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """``jac`` shape ``(..., 2, 2, d_h)`` with ``jac[..., out, in, :]``.

        ``out/in`` in {0: c, 1: h} matching eq. 3.2b:
        [[Jcc, Jch], [Jhc, Jhh]].
        """
        acts = self._recurrence(state_prev, x, wx=wx)
        f_p = _sigmoid_prime_from_act(acts.f)
        z_p = _tanh_prime_from_act(acts.z)
        o_p = _sigmoid_prime_from_act(acts.o)
        h_act_p = _tanh_prime_from_act(acts.h_act)

        # eq. 3.2b, all products elementwise because A, C are diagonal.
        j_cc = acts.f + (acts.c_prev - acts.z) * f_p * acts.peephole_f
        j_ch = (acts.c_prev - acts.z) * f_p * acts.a_f + (1.0 - acts.f) * z_p * acts.a_z
        j_hc = (acts.h_act * o_p * acts.peephole_o + acts.o * h_act_p) * j_cc
        j_hh = (
            acts.h_act * o_p * (acts.a_o + acts.peephole_o * j_ch)
            + acts.o * h_act_p * j_ch
        )
        jac = torch.stack(
            (
                torch.stack((j_cc, j_ch), dim=-2),
                torch.stack((j_hc, j_hh), dim=-2),
            ),
            dim=-3,
        )
        return acts.state_new, jac

    def _recurrence(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> _LSTMActs:
        c_prev = state_prev[..., LSTM_CELL, :]
        h_prev = state_prev[..., LSTM_HIDDEN, :]
        a_f, a_z, a_o, peephole_f, peephole_o = self.clipped_recurrent()
        if wx is None:
            wx = self.W_x(x)
        fx, zx, ox = wx.chunk(3, dim=-1)

        f = torch.sigmoid(a_f * h_prev + peephole_f * c_prev + fx)
        z = torch.tanh(a_z * h_prev + zx)
        c = f * c_prev + (1.0 - f) * z
        o = torch.sigmoid(a_o * h_prev + peephole_o * c + ox)
        h_act = torch.tanh(c)
        h = o * h_act
        return _LSTMActs(
            state_new=torch.stack((c, h), dim=-2),
            c_prev=c_prev,
            h_prev=h_prev,
            f=f,
            z=z,
            o=o,
            h_act=h_act,
            a_f=a_f,
            a_z=a_z,
            a_o=a_o,
            peephole_f=peephole_f,
            peephole_o=peephole_o,
        )


class _LSTMActs(NamedTuple):
    state_new: Tensor
    c_prev: Tensor
    h_prev: Tensor
    f: Tensor
    z: Tensor
    o: Tensor
    h_act: Tensor
    a_f: Tensor
    a_z: Tensor
    a_o: Tensor
    peephole_f: Tensor
    peephole_o: Tensor
