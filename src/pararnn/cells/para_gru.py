"""Diagonal ParaGRU — Danieli et al. 2025 eq. 3.1a, A_* = diag(a_*) (eq. 3.3)."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_


def _sigmoid_prime_from_act(gate: Tensor) -> Tensor:
    """σ'(pre) when ``gate = σ(pre)``: gate * (1 - gate)."""
    return gate * (1.0 - gate)


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    """tanh'(pre) when ``act = tanh(pre)``: 1 - act^2."""
    return 1.0 - act.square()


class ParaGRU(nn.Module):
    """Fully gated GRU with diagonal recurrent weights.

    Gates: update z, reset r, candidate n (paper's c). Activations: sigmoid /
    sigmoid / tanh (Cho et al. 2014, as used in §3).

    ``max_recurrent_norm`` is an App. C.1 elementwise clamp of ``a_*`` to
    ``[-cap, cap]``.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        *,
        d_in: int | None = None,
        d_h: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.state_slots = 1
        self.max_recurrent_norm = max_recurrent_norm

        self.a_z = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.a_r = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.a_n = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.W_x = nn.Linear(input_size, 3 * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xavier_gaussian_vec_(self.a_z)
        xavier_gaussian_vec_(self.a_r)
        xavier_gaussian_vec_(self.a_n)
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def clipped_a(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.max_recurrent_norm is None:
            return self.a_z, self.a_r, self.a_n
        cap = self.max_recurrent_norm
        return (
            self.a_z.clamp(-cap, cap),
            self.a_r.clamp(-cap, cap),
            self.a_n.clamp(-cap, cap),
        )

    def step(self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        """One step. ``h_prev`` last dim ``d_h``; ``x`` last dim ``d_in``.

        Sequential unroll / decode. ``wx`` is optional ``W_x(x)`` (eq. 3.1,
        independent of ``h``) so Newton can reuse one GEMM across init + ``K``
        iterations. Pass ``x`` or ``wx``; ``wx`` wins if both are set.
        """
        return self._recurrence(h_prev, x, wx=wx).h_new

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(h_new, j_diag)`` with ``j_diag = ∂h_new/∂h_prev`` (diagonal).

        Jacobian: eq. 3.2a with A_* diagonal, so products are elementwise.
        """
        acts = self._recurrence(h_prev, x, wx=wx)
        z_p = _sigmoid_prime_from_act(acts.z)
        r_p = _sigmoid_prime_from_act(acts.r)
        n_p = _tanh_prime_from_act(acts.n)
        j = (
            (1.0 - acts.z)
            + (acts.n - acts.h_prev) * z_p * acts.a_z
            + acts.z * n_p * acts.a_n * (acts.r + acts.h_prev * r_p * acts.a_r)
        )
        return acts.h_new, j

    def _recurrence(
        self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None = None
    ) -> _GRUActs:
        a_z, a_r, a_n = self.clipped_a()
        if wx is None:
            if x is None:
                raise ValueError("ParaGRU.step needs x or wx (precomputed W_x(x))")
            wx = self.W_x(x)
        zx, rx, nx = wx.chunk(3, dim=-1)
        z = torch.sigmoid(a_z * h_prev + zx)
        r = torch.sigmoid(a_r * h_prev + rx)
        n = torch.tanh(a_n * (h_prev * r) + nx)
        h_new = torch.lerp(h_prev, n, z)
        return _GRUActs(h_new=h_new, h_prev=h_prev, z=z, r=r, n=n, a_z=a_z, a_r=a_r, a_n=a_n)


class _GRUActs(NamedTuple):
    h_new: Tensor
    h_prev: Tensor
    z: Tensor
    r: Tensor
    n: Tensor
    a_z: Tensor
    a_r: Tensor
    a_n: Tensor
