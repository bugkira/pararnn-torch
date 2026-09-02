"""sLSTM (Beck et al. 2024) as a Newton cell.

State ``(..., 4, hidden_size)`` = (c, n, m, h). Stabilizer ``max``, exp
input/forget, normalizer ``n``, memory mixing ``R h``. ``mix='diag'`` is
channelwise (fused Newton). ``mix='head'`` is dense ``R`` inside a head,
block-diagonal across heads. ``mix='dense'`` mixes the full width.

Newton init is the ``R h = 0`` unroll (running ``m``/``n``). Recurrent mix
is exactly one of ``R`` / ``R_dense`` / ``R_head``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_

_MIX = ("diag", "dense", "head")
_JAC = {"diag": "block4", "dense": "dense", "head": "head"}


class ParaSLSTM(nn.Module):
    """Four-slot sLSTM: state ``(..., 4, hidden_size)`` = (c, n, m, h).

    ``max_recurrent_norm`` is an App. C.1 elementwise clamp of recurrent
    mix entries. ``eps`` floors ``n`` in
    ``h = o * c / n``. ``mix='head'`` needs ``n_heads`` dividing ``hidden_size``.
    Recurrent mix: ``R`` (diag), ``R_head`` (head), or ``R_dense`` (dense).
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        *,
        d_in: int | None = None,
        d_h: int | None = None,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if mix not in _MIX:
            raise ValueError(f"mix must be one of {_MIX}, got {mix!r}")
        input_size, hidden_size = resolve_layer_sizes(
            input_size, hidden_size, d_in=d_in, d_h=d_h
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.state_slots = SLSTM_SLOTS
        self.hidden_slot = SLSTM_HIDDEN
        self.mix = mix
        self.jac_structure = _JAC[mix]
        self.max_recurrent_norm = max_recurrent_norm
        self.eps = eps
        self.n_heads = n_heads
        self.d_head = None
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True, **factory_kwargs)
        self.R = None
        self.R_dense = None
        self.R_head = None
        if mix == "diag":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R = nn.Parameter(torch.empty(4, hidden_size, **factory_kwargs))
        elif mix == "dense":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R_dense = nn.Linear(
                hidden_size, 4 * hidden_size, bias=False, **factory_kwargs
            )
        else:
            if n_heads is None or n_heads < 1 or hidden_size % n_heads != 0:
                raise ValueError(
                    f"mix='head' needs n_heads that divides hidden_size={hidden_size}, "
                    f"got {n_heads!r}"
                )
            self.d_head = hidden_size // n_heads
            self.R_head = nn.Parameter(
                torch.empty(4, n_heads, self.d_head, self.d_head, **factory_kwargs)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.R is not None:
            xavier_gaussian_vec_(self.R)
        elif self.R_dense is not None:
            nn.init.orthogonal_(self.R_dense.weight, gain=0.25)
        else:
            for g in range(4):
                for hd in range(self.n_heads):
                    nn.init.orthogonal_(self.R_head[g, hd], gain=0.25)

    def clipped_r(self) -> Tensor:
        if self.R is None:
            raise TypeError("clipped_r is for mix='diag'")
        if self.max_recurrent_norm is None:
            return self.R
        return self.R.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def clipped_r_head(self) -> Tensor:
        if self.R_head is None:
            raise TypeError("clipped_r_head is for mix='head'")
        if self.max_recurrent_norm is None:
            return self.R_head
        return self.R_head.clamp(-self.max_recurrent_norm, self.max_recurrent_norm)

    def step(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> Tensor:
        """One sLSTM step. ``state_prev`` is ``(..., 4, d_h)``."""
        if wx is None:
            wx = self.W_x(x)
        h = state_prev[..., SLSTM_HIDDEN, :]
        return self._acts_from_pre(state_prev, wx + self._recurrent(h)).state_new

    def step_with_jacobian(
        self, state_prev: Tensor, x: Tensor, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """``(state_new, J)``. Layout follows ``jac_structure``.

        The channelwise skeleton is ``mix='diag'``. We also chain ``tanh`` of
        the candidate, ``σ`` of the output gate, and ``n+ε`` (the forward
        uses ``eps``). ``torch.maximum`` at ties splits 0.5/0.5.
        """
        if wx is None:
            wx = self.W_x(x)
        h = state_prev[..., SLSTM_HIDDEN, :]
        acts = self._acts_from_pre(state_prev, wx + self._recurrent(h))
        if self.mix == "diag":
            jac = self._jac_diag(acts)
        elif self.mix == "head":
            jac = self._jac_head(acts)
        else:
            jac = self._jac_dense(acts)
        return acts.state_new, jac

    def step_head(self, state: Tensor, wx_head: Tensor, r: Tensor) -> Tensor:
        """One head: ``state`` / ``wx_head`` are ``(4, d_head)``; ``r`` is ``(4, d_head, d_head)``."""
        h = state[SLSTM_HIDDEN]
        pre = wx_head + torch.einsum("d,gde->ge", h, r)
        return self._step_from_pre(state, pre.reshape(4 * state.shape[-1]))

    def _recurrent(self, h: Tensor) -> Tensor:
        if self.R is not None:
            return (self.clipped_r() * h.unsqueeze(-2)).reshape(*h.shape[:-1], 4 * self.d_h)
        if self.R_head is not None:
            h_v = h.reshape(*h.shape[:-1], self.n_heads, self.d_head)
            rec = torch.einsum("...nd,gnde->...gne", h_v, self.clipped_r_head())
            return rec.reshape(*h.shape[:-1], 4 * self.d_h)
        return self.R_dense(h)

    def zero_hidden_init(
        self, x: Tensor, *, wx: Tensor | None = None, h0: Tensor | None = None
    ) -> Tensor:
        """Newton guess: running ``(c, n, m)`` with ``R h = 0``.

        See ``pararnn.solvers.slstm_picard.slstm_zero_hidden_init``.
        """
        from pararnn.solvers.slstm_picard import slstm_zero_hidden_init

        if wx is None:
            wx = self.W_x(x)
        return slstm_zero_hidden_init(wx, eps=self.eps, h0=h0)

    def picard_init(
        self,
        x: Tensor,
        *,
        wx: Tensor | None = None,
        h0: Tensor | None = None,
        n_picard: int = 1,
    ) -> Tensor:
        """Zero-hidden, then ``n_picard`` frozen-gate scans.

        See ``pararnn.solvers.slstm_picard.slstm_picard_init``.
        """
        from pararnn.solvers.slstm_picard import slstm_picard_init

        if wx is None:
            wx = self.W_x(x)
        return slstm_picard_init(self, wx, h0=h0, n_picard=n_picard)

    def _step_from_pre(self, state_prev: Tensor, pre: Tensor) -> Tensor:
        return self._acts_from_pre(state_prev, pre).state_new

    def _acts_from_pre(self, state_prev: Tensor, pre: Tensor) -> _SLSTMActs:
        c = state_prev[..., SLSTM_CELL, :]
        n = state_prev[..., SLSTM_NORMALIZER, :]
        m = state_prev[..., SLSTM_STABILIZER, :]
        z_i, z_f, z_z, z_o = pre.chunk(4, dim=-1)
        left = z_f + m
        m_new = torch.maximum(left, z_i)
        alpha = _maximum_subgrad_left(left, z_i)
        i_t = torch.exp(z_i - m_new)
        f_t = torch.exp(z_f + m - m_new)
        z = torch.tanh(z_z)
        n_new = f_t * n + i_t
        c_new = f_t * c + i_t * z
        o = torch.sigmoid(z_o)
        denom = n_new + self.eps
        h_new = o * (c_new / denom)
        return _SLSTMActs(
            state_new=torch.stack((c_new, n_new, m_new, h_new), dim=-2),
            c=c,
            n=n,
            m=m,
            c_new=c_new,
            n_new=n_new,
            i=i_t,
            f=f_t,
            z=z,
            o=o,
            denom=denom,
            alpha=alpha,
        )

    def _jac_diag(self, acts: _SLSTMActs) -> Tensor:
        r = self.clipped_r()
        rows = self._jac_channelwise(acts, r[0], r[1], r[2], r[3])
        return torch.stack(
            (
                torch.stack(rows[0:4], dim=-2),
                torch.stack(rows[4:8], dim=-2),
                torch.stack(rows[8:12], dim=-2),
                torch.stack(rows[12:16], dim=-2),
            ),
            dim=-3,
        )

    def _jac_channelwise(
        self,
        acts: _SLSTMActs,
        r_i: Tensor,
        r_f: Tensor,
        r_z: Tensor,
        r_o: Tensor,
    ) -> tuple[Tensor, ...]:
        """4×4 per channel. ``r_*`` are ∂pre_g/∂h (diag mix: the vector ``R_g``)."""
        alpha = acts.alpha
        beta = 1.0 - alpha
        dm_dh = alpha * r_f + beta * r_i
        di_dm = -acts.i * alpha
        df_dm = acts.f * beta
        di_dh = acts.i * (r_i - dm_dh)
        df_dh = acts.f * (r_f - dm_dh)
        dz_dh = (1.0 - acts.z.square()) * r_z
        do_dh = acts.o * (1.0 - acts.o) * r_o
        zeros = torch.zeros_like(acts.f)
        j_cc = acts.f
        j_cn = zeros
        j_cm = df_dm * acts.c + di_dm * acts.z
        j_ch = df_dh * acts.c + di_dh * acts.z + acts.i * dz_dh
        j_nc = zeros
        j_nn = acts.f
        j_nm = df_dm * acts.n + di_dm
        j_nh = df_dh * acts.n + di_dh
        j_mc = zeros
        j_mn = zeros
        j_mm = alpha
        j_mh = dm_dh
        inv = acts.o / acts.denom
        dn = -acts.o * acts.c_new / acts.denom.square()
        du = acts.c_new / acts.denom
        j_hc = inv * j_cc
        j_hn = dn * j_nn
        j_hm = inv * j_cm + dn * j_nm
        j_hh = inv * j_ch + dn * j_nh + du * do_dh
        return (
            j_cc,
            j_cn,
            j_cm,
            j_ch,
            j_nc,
            j_nn,
            j_nm,
            j_nh,
            j_mc,
            j_mn,
            j_mm,
            j_mh,
            j_hc,
            j_hn,
            j_hm,
            j_hh,
        )

    def _jac_head(self, acts: _SLSTMActs) -> Tensor:
        r = self.clipped_r_head()
        # R[g, hd, in, out]; J_pre_g is (out, in).
        j_g = r.permute(0, 1, 3, 2)
        assert self.n_heads is not None and self.d_head is not None
        return self._jac_packed(acts, j_g, n_heads=self.n_heads, d_head=self.d_head)

    def _jac_dense(self, acts: _SLSTMActs) -> Tensor:
        w = self.R_dense.weight.view(4, self.d_h, self.d_h)
        j_g = w.unsqueeze(1)
        packed = self._jac_packed(acts, j_g, n_heads=1, d_head=self.d_h)
        return packed.squeeze(-3)

    def _jac_packed(
        self, acts: _SLSTMActs, j_g: Tensor, *, n_heads: int, d_head: int
    ) -> Tensor:
        """``j_g`` is ``(4, H, d_out, d_in)``. Result ``(..., H, 4 d, 4 d)``."""
        prefix = acts.f.shape[:-1]
        d = d_head
        sd = 4 * d

        def split(t: Tensor) -> Tensor:
            return t.reshape(*prefix, n_heads, d)

        alpha = split(acts.alpha)
        f = split(acts.f)
        i_t = split(acts.i)
        z = split(acts.z)
        o = split(acts.o)
        c = split(acts.c)
        n = split(acts.n)
        c_new = split(acts.c_new)
        denom = split(acts.denom)
        beta = 1.0 - alpha
        j_zi, j_zf, j_zz, j_zo = j_g[0], j_g[1], j_g[2], j_g[3]
        dm_dh = alpha.unsqueeze(-1) * j_zf + beta.unsqueeze(-1) * j_zi
        di_dm = -i_t * alpha
        df_dm = f * beta
        di_dh = i_t.unsqueeze(-1) * (j_zi - dm_dh)
        df_dh = f.unsqueeze(-1) * (j_zf - dm_dh)
        dz_dh = (1.0 - z.square()).unsqueeze(-1) * j_zz
        do_dh = (o * (1.0 - o)).unsqueeze(-1) * j_zo
        j_cm = df_dm * c + di_dm * z
        j_nm = df_dm * n + di_dm
        j_ch = df_dh * c.unsqueeze(-1) + di_dh * z.unsqueeze(-1) + i_t.unsqueeze(-1) * dz_dh
        j_nh = df_dh * n.unsqueeze(-1) + di_dh
        inv = o / denom
        dn = -o * c_new / denom.square()
        du = c_new / denom
        j_hh = inv.unsqueeze(-1) * j_ch + dn.unsqueeze(-1) * j_nh + du.unsqueeze(-1) * do_dh
        j_hc = inv * f
        j_hn = dn * f
        j_hm = inv * j_cm + dn * j_nm
        jac = acts.f.new_zeros(*prefix, n_heads, sd, sd)

        def put_diag(out_s: int, in_s: int, diag: Tensor) -> None:
            jac[
                ...,
                out_s * d : (out_s + 1) * d,
                in_s * d : (in_s + 1) * d,
            ] = torch.diag_embed(diag)

        def put_h(out_s: int, mat: Tensor) -> None:
            jac[..., out_s * d : (out_s + 1) * d, 3 * d : 4 * d] = mat

        put_diag(0, 0, f)
        put_diag(0, 2, j_cm)
        put_h(0, j_ch)
        put_diag(1, 1, f)
        put_diag(1, 2, j_nm)
        put_h(1, j_nh)
        put_diag(2, 2, alpha)
        put_h(2, dm_dh)
        put_diag(3, 0, j_hc)
        put_diag(3, 1, j_hn)
        put_diag(3, 2, j_hm)
        put_h(3, j_hh)
        return jac


def _maximum_subgrad_left(left: Tensor, right: Tensor) -> Tensor:
    """∂maximum(left, right)/∂left. Ties split 0.5 (PyTorch ``maximum``)."""
    gt = (left > right).to(dtype=left.dtype)
    eq = (left == right).to(dtype=left.dtype)
    return gt + 0.5 * eq


class _SLSTMActs(NamedTuple):
    state_new: Tensor
    c: Tensor
    n: Tensor
    m: Tensor
    c_new: Tensor
    n_new: Tensor
    i: Tensor
    f: Tensor
    z: Tensor
    o: Tensor
    denom: Tensor
    alpha: Tensor
