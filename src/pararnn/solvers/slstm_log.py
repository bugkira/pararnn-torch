"""Log-space Newton coordinates for ParaSLSTM: ``(u, log n, m, h)``."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM, _maximum_subgrad_left
from pararnn.layout import SLSTM_CELL, SLSTM_HIDDEN, SLSTM_NORMALIZER, SLSTM_STABILIZER

# u = c/(n+eps) lives in ~(-1, 1). ln clamp is fp32 exp (overflow ~88).
_LOG_RATIO_ABSMAX = 4.0
_LOG_N_MIN = -40.0
_LOG_N_MAX = 80.0


def slstm_encode_log(state: Tensor, *, eps: float) -> Tensor:
    """Newton coordinates: ``(u, log n, m, h)`` with ``u = c/n``.

    ``c`` is signed; coordinates use ``u = c/n``. ``n`` is clamped at
    ``eps`` so ``h0=0`` is ``log n = log(eps)``. Algebra in fp32.
    """
    c = state[..., SLSTM_CELL, :].float()
    n = state[..., SLSTM_NORMALIZER, :].float().clamp_min(eps)
    m = state[..., SLSTM_STABILIZER, :].float()
    h = state[..., SLSTM_HIDDEN, :].float()
    return torch.stack((c / n, torch.log(n), m, h), dim=-2).to(dtype=state.dtype)


def slstm_decode_log(coords: Tensor, *, eps: float) -> Tensor:
    """Inverse of ``slstm_encode_log``. ``n = exp(log n)``; ``c = u n``."""
    del eps
    u = coords[..., SLSTM_CELL, :].float().clamp(-_LOG_RATIO_ABSMAX, _LOG_RATIO_ABSMAX)
    ln = coords[..., SLSTM_NORMALIZER, :].float().clamp(_LOG_N_MIN, _LOG_N_MAX)
    m = coords[..., SLSTM_STABILIZER, :].float()
    h = coords[..., SLSTM_HIDDEN, :].float()
    n = torch.exp(ln)
    return torch.stack((u * n, n, m, h), dim=-2).to(dtype=coords.dtype)


def slstm_clamp_log_coords(coords: Tensor) -> Tensor:
    """Keep ``(c/n, log n)`` inside the decode clamps after a Newton step."""
    out = coords.clone()
    out[..., SLSTM_CELL, :] = out[..., SLSTM_CELL, :].clamp(
        -_LOG_RATIO_ABSMAX, _LOG_RATIO_ABSMAX
    )
    out[..., SLSTM_NORMALIZER, :] = out[..., SLSTM_NORMALIZER, :].clamp(
        _LOG_N_MIN, _LOG_N_MAX
    )
    return out


class _LogActs(NamedTuple):
    state_new: Tensor
    u: Tensor
    u_new: Tensor
    ln_new: Tensor
    gamma: Tensor
    z: Tensor
    o: Tensor
    alpha: Tensor
    da_dm: Tensor
    da_dh: Tensor
    db_dm: Tensor
    db_dh: Tensor
    dm_dh: Tensor


class SLSTMLogCoords:
    """Newton cell in ``(u, log n, m, h)``. Convex combination + LSE.

    ``u_t = (1-γ) u_{t-1} + γ tanh(z_z)``, ``log n`` via ``logaddexp``,
    ``h_t = σ(z_o) ⊙ u_t``. Mixing still reads stored ``h``. Sequential
    stays on the native ``(c, n, m, h)`` cell.
    """

    def __init__(self, cell: ParaSLSTM) -> None:
        self.cell = cell
        self.input_size = cell.input_size
        self.hidden_size = cell.hidden_size
        self.d_in = cell.d_in
        self.d_h = cell.d_h
        self.state_slots = cell.state_slots
        self.hidden_slot = cell.hidden_slot
        self.mix = cell.mix
        self.jac_structure = cell.jac_structure
        self.eps = cell.eps
        self.n_heads = cell.n_heads
        self.d_head = cell.d_head

    def parameters(self, recurse: bool = True):
        return self.cell.parameters(recurse=recurse)

    @property
    def W_x(self) -> nn.Linear:
        return self.cell.W_x

    def step(
        self, coords_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> Tensor:
        return self._acts_from_coords(coords_prev, x, wx=wx).state_new

    def step_with_jacobian(
        self, coords_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        acts = self._acts_from_coords(coords_prev, x, wx=wx)
        if self.mix != "diag":
            from pararnn.solvers.jacobian import jacobian_autograd

            return jacobian_autograd(
                self, coords_prev, x, structure=self.jac_structure
            )
        return acts.state_new, self._jac_log_diag(acts)

    def _acts_from_coords(
        self, coords_prev: Tensor, x: Tensor, *, wx: Tensor | None
    ) -> _LogActs:
        if wx is None:
            wx = self.cell.W_x(x)
        u = coords_prev[..., SLSTM_CELL, :].float()
        ln = coords_prev[..., SLSTM_NORMALIZER, :].float()
        m = coords_prev[..., SLSTM_STABILIZER, :].float()
        h = coords_prev[..., SLSTM_HIDDEN, :]
        pre = wx + self.cell._recurrent(h)
        z_i, z_f, z_z, z_o = pre.chunk(4, dim=-1)
        z_i, z_f, z_z, z_o = z_i.float(), z_f.float(), z_z.float(), z_o.float()
        left = z_f + m
        m_new = torch.maximum(left, z_i)
        alpha = _maximum_subgrad_left(left, z_i)
        a = z_f + m - m_new + ln
        b = z_i - m_new
        ln_new = torch.logaddexp(a, b)
        gamma = torch.exp(b - ln_new)
        z = torch.tanh(z_z)
        u_new = (1.0 - gamma) * u + gamma * z
        o = torch.sigmoid(z_o)
        h_new = o * u_new
        state_new = torch.stack((u_new, ln_new, m_new, h_new), dim=-2).to(
            dtype=coords_prev.dtype
        )
        beta = 1.0 - alpha
        if self.mix == "diag":
            r = self.cell.clipped_r()
            dm_dh = alpha * r[1] + beta * r[0]
            da_dh = r[1] - dm_dh
            db_dh = r[0] - dm_dh
        else:
            dm_dh = da_dh = db_dh = torch.zeros_like(u)
        return _LogActs(
            state_new=state_new,
            u=u,
            u_new=u_new,
            ln_new=ln_new,
            gamma=gamma,
            z=z,
            o=o,
            alpha=alpha,
            da_dm=1.0 - alpha,
            da_dh=da_dh,
            db_dm=-alpha,
            db_dh=db_dh,
            dm_dh=dm_dh,
        )

    def _jac_log_diag(self, acts: _LogActs) -> Tensor:
        """4×4 of the convex-combination / LSE map in ``(u, log n, m, h)``."""
        r = self.cell.clipped_r()
        gamma = acts.gamma
        omg = 1.0 - gamma
        dln_dln = omg
        dln_dm = omg * acts.da_dm + gamma * acts.db_dm
        dln_dh = omg * acts.da_dh + gamma * acts.db_dh
        zeros = torch.zeros_like(gamma)
        dgamma_dln = gamma * (0.0 - dln_dln)
        dgamma_dm = gamma * (acts.db_dm - dln_dm)
        dgamma_dh = gamma * (acts.db_dh - dln_dh)
        dz_dh = (1.0 - acts.z.square()) * r[2]
        uz = acts.z - acts.u
        du_du = omg
        du_dln = uz * dgamma_dln
        du_dm = uz * dgamma_dm
        du_dh = uz * dgamma_dh + gamma * dz_dh
        do_dh = acts.o * (1.0 - acts.o) * r[3]
        o = acts.o
        rows = (
            du_du,
            du_dln,
            du_dm,
            du_dh,
            zeros,
            dln_dln,
            dln_dm,
            dln_dh,
            zeros,
            zeros,
            acts.alpha,
            acts.dm_dh,
            o * du_du,
            o * du_dln,
            o * du_dm,
            o * du_dh + acts.u_new * do_dh,
        )
        return torch.stack(
            (
                torch.stack(rows[0:4], dim=-2),
                torch.stack(rows[4:8], dim=-2),
                torch.stack(rows[8:12], dim=-2),
                torch.stack(rows[12:16], dim=-2),
            ),
            dim=-3,
        ).to(dtype=acts.state_new.dtype)
