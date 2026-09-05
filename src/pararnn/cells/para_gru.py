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

    Gates: update ``z``, reset ``r``, candidate ``n`` (paper's ``c``).
    Activations: sigmoid / sigmoid / tanh (Cho et al. 2014, as used in §3).
    Recurrent matrices are diagonal (Danieli et al. 2025 eq. 3.1a, 3.3).

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    hidden_size, d_h : int
        Hidden width.
    state_slots : int
        Always ``1``; state layout is ``(..., d_h)``.
    max_recurrent_norm : float or None
        App. C.1 elementwise clamp of ``a_*`` to ``[-cap, cap]``.
    a_z, a_r, a_n : Parameter
        Diagonal recurrent vectors, each of shape ``(d_h,)``.
    W_x : nn.Linear
        Input projection to three gates, ``d_in → 3 * d_h``.

    See Also
    --------
    ParaLSTM, ParaSLSTM, ParaRNN
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
        """
        Parameters
        ----------
        input_size : int or None, default=None
            Input feature width. Alias of ``d_in``.
        hidden_size : int or None, default=None
            Hidden width. Alias of ``d_h``.
        d_in : int or None, default=None
            Alias for ``input_size``.
        d_h : int or None, default=None
            Alias for ``hidden_size``.
        max_recurrent_norm : float or None, default=0.5
            Elementwise clamp of ``a_z``, ``a_r``, ``a_n`` to
            ``[-max_recurrent_norm, max_recurrent_norm]`` (App. C.1).
            ``None`` disables clipping.
        device : torch.device or str or None, default=None
            Parameter device.
        dtype : torch.dtype or None, default=None
            Parameter dtype.
        """
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

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, max_recurrent_norm={self.max_recurrent_norm}"
        )

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
        """Advance one GRU step (sequential unroll / decode).

        Parameters
        ----------
        h_prev : Tensor
            Previous hidden state. Tensor of shape ``(..., d_h)``.
        x : Tensor or None, default=None
            Input at this step. Tensor of shape ``(..., d_in)``. Required
            when ``wx`` is omitted.
        wx : Tensor or None, default=None
            Optional precomputed ``W_x(x)`` (eq. 3.1, independent of ``h``)
            so Newton can reuse one GEMM across init and ``K`` iterations.
            When both ``x`` and ``wx`` are set, ``wx`` is used.

        Returns
        -------
        h_new : Tensor
            Next hidden state. Tensor of shape ``(..., d_h)``.

        Raises
        ------
        ValueError
            When both ``x`` and ``wx`` are ``None``.
        """
        return self._recurrence(h_prev, x, wx=wx).h_new

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Advance one step and return the diagonal Jacobian.

        Jacobian follows eq. 3.2a with diagonal ``A_*``, so all products are
        elementwise.

        Parameters
        ----------
        h_prev : Tensor
            Previous hidden state. Tensor of shape ``(..., d_h)``.
        x : Tensor or None, default=None
            Input at this step. Tensor of shape ``(..., d_in)``.
        wx : Tensor or None, default=None
            Optional precomputed ``W_x(x)``. When both ``x`` and ``wx`` are
            set, ``wx`` is used.

        Returns
        -------
        h_new : Tensor
            Next hidden state. Tensor of shape ``(..., d_h)``.
        j_diag : Tensor
            ``∂h_new/∂h_prev`` as a diagonal. Tensor of shape ``(..., d_h)``.

        Raises
        ------
        ValueError
            When both ``x`` and ``wx`` are ``None``.
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
