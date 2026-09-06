"""ParaTitans — shallow L=1 Titans-inspired neural memory (vector state).

Honest scope: memory ``h`` is a channel vector (or, equivalently, diagonal
weights of an elementwise associative map). One surprise-GD step plus a
small diagonal nonlinear correction. Deep multi-layer MLP memory ``M`` is
parked.

Reference: Behrouz et al., *Titans: Learning to Memorize at Test Time*
(arXiv:2501.00663) — surprise gradient on associative loss; this cell is a
diag-Jacobian library slot, not the full Titans attention stack.
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

# softplus(θ_pre) starts small so the surprise step does not dominate the
# (1-α) forget gate at init (Behrouz et al. use a learned LR; App. C.1-style
# small recurrent scale). Fallback: raise bias toward 0 if |J| under-contracts.
_THETA_BIAS_INIT = -1.0


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    """tanh'(pre) when ``act = tanh(pre)``: ``1 - act^2``."""
    return 1.0 - act.square()


class _TitansActs(NamedTuple):
    h_prev: Tensor
    alpha: Tensor
    theta: Tensor
    k: Tensor
    v: Tensor
    n: Tensor
    u: Tensor
    r: Tensor
    h_new: Tensor


class ParaTitans(nn.Module):
    """Shallow Titans-inspired memory slot with channelwise-diagonal Jacobian.

    .. math::

        \\alpha_t = \\sigma(\\alpha^{\\mathrm{pre}}_t),\\quad
        \\theta_t = \\mathrm{softplus}(\\theta^{\\mathrm{pre}}_t),\\\\
        g_t = (h_{t-1}\\odot k_t - v_t)\\odot k_t,\\\\
        h^{\\mathrm{lin}}_t = (1-\\alpha_t)\\odot h_{t-1} - \\theta_t\\odot g_t,\\\\
        h_t = h^{\\mathrm{lin}}_t
        + u\\odot\\tanh(W_n x_t + r\\odot h_{t-1}).

    ``(α_pre, θ_pre, k, v, n_pre) = W_x(x)`` chunked into five ``d_h`` blocks.
    ``jac_structure='diag'``. Fused Newton reuses the diag scan class.

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    hidden_size, d_h : int
        Memory / hidden width.
    state_slots : int
        Always ``1``.
    jac_structure : str
        Always ``'diag'``.
    max_recurrent_norm : float or None
        App. C.1 clamp on diagonal ``u`` and ``r``. Default ``0.5``.
    u, r : Parameter
        Diagonal nonlinear correction gains, shape ``(d_h,)``.
    W_x : nn.Linear
        ``d_in → 5 * d_h`` packing ``(α_pre, θ_pre, k, v, n_pre)``.
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
            Elementwise clamp of ``u`` and ``r`` (App. C.1). Default ``0.5``.
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
        self.r = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.W_x = nn.Linear(input_size, 5 * hidden_size, bias=True, **factory_kwargs)
        self.reset_parameters()
        log.info(
            "paratitans_init d_in=%s d_h=%s max_recurrent_norm=%s",
            input_size,
            hidden_size,
            self.max_recurrent_norm,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, max_recurrent_norm={self.max_recurrent_norm}"
        )

    def reset_parameters(self) -> None:
        """Kaiming ``W_x``; Xavier-Gaussian ``u``/``r``; small softplus θ bias."""
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        # θ_pre chunk is [d_h : 2 d_h]; softplus(-1) ≈ 0.31 keeps |θ k²| modest.
        with torch.no_grad():
            self.W_x.bias[self.d_h : 2 * self.d_h].fill_(_THETA_BIAS_INIT)
        xavier_gaussian_vec_(self.u)
        xavier_gaussian_vec_(self.r)

    def clipped_u(self) -> Tensor:
        """App. C.1 clamp of the nonlinear residual gain ``u``."""
        if self.max_recurrent_norm is None:
            return self.u
        return self.u.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def clipped_r(self) -> Tensor:
        """App. C.1 clamp of the recurrent mix ``r`` into the tanh polish."""
        if self.max_recurrent_norm is None:
            return self.r
        return self.r.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def step(self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None) -> Tensor:
        """One Titans memory step.

        Parameters
        ----------
        h_prev : Tensor
            Previous memory. Shape ``(..., d_h)``.
        x : Tensor, optional
            Input. Shape ``(..., d_in)``. Ignored when ``wx`` is set.
        wx : Tensor, optional
            Precomputed ``W_x(x)``. Shape ``(..., 5 * d_h)``.

        Returns
        -------
        h_new : Tensor
            Shape ``(..., d_h)``.
        """
        return self._recurrence(h_prev, x, wx=wx).h_new

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Step plus channelwise-diagonal Jacobian ``(..., d_h)``.

        .. math::

            J = (1-\\alpha) - \\theta\\odot k^{2}
            + u\\odot(1-n^{2})\\odot r.
        """
        acts = self._recurrence(h_prev, x, wx=wx)
        n_p = _tanh_prime_from_act(acts.n)
        jac = (1.0 - acts.alpha) - acts.theta * acts.k.square() + acts.u * n_p * acts.r
        return acts.h_new, jac

    def _recurrence(self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None) -> _TitansActs:
        if wx is None:
            if x is None:
                raise ValueError("ParaTitans.step needs x or wx")
            wx = self.W_x(x)
        alpha_pre, theta_pre, k, v, n_pre = wx.chunk(5, dim=-1)
        alpha = torch.sigmoid(alpha_pre)
        theta = F.softplus(theta_pre)
        # ℓ = 0.5 ||h⊙k - v||² → ∂ℓ/∂h = (h⊙k - v)⊙k (elementwise assoc.).
        grad = (h_prev * k - v) * k
        h_lin = (1.0 - alpha) * h_prev - theta * grad
        u = self.clipped_u()
        r = self.clipped_r()
        n = torch.tanh(n_pre + r * h_prev)
        h_new = h_lin + u * n
        return _TitansActs(
            h_prev=h_prev,
            alpha=alpha,
            theta=theta,
            k=k,
            v=v,
            n=n,
            u=u,
            r=r,
            h_new=h_new,
        )
