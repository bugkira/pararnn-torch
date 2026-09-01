"""sLSTM (Beck et al. 2024) as a Newton cell — not Apple's ParaLSTM.

Equations: ADD_TASK / xLSTM §2. Stabilizer ``max``, exp input/forget, normalizer
``n``, memory mixing ``R h``. ``mix='diag'`` is channelwise (Newton prototype).
``mix='head'`` is the xLSTM compromise: dense ``R`` inside a head, block-
diagonal across heads. ``mix='dense'`` mixes the full ``d_h``; scan is
``O(T (4d)^3)`` — tests stay tiny.

``mix='diag'`` has a fused Triton Newton (cell + 4x4 J + scan). Head/dense
do not. Not FlashRNN. Newton init is not App. A ``f(0, x_t)``: ``n`` is a
running normalizer, so we start from the ``R h = 0`` unroll (max-plus ``m``,
linear ``n``/``c``). ``K=3`` then matches GRU on the T=48 prototype.
``omega=0.5`` still damps the snap — do not copy ELK as the sLSTM default.
``NewtonConfig(coords='log')`` iterates the convex-combination / LSE map
``(u, log n, m, h)`` with ``u=c/n`` (opt-in; fused diag has a matching
Triton cell). Not the long-T snap: that is ``picard_iters`` (frozen-gate
1D scans, still O(log T); CUDA uses the Triton twin) plus native K=3.
Newton itself keeps the 4x4 ``J`` of ``R h`` feedback.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.init import kaiming_uniform_linear_, xavier_gaussian_vec_
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
    prepend_state,
)

_MIX = ("diag", "dense", "head")
_JAC = {"diag": "block4", "dense": "dense", "head": "head"}
# ã = c/(n+eps) lives in ~(-1, 1). ln clamp is fp32 exp (overflow ~88).
_LOG_RATIO_ABSMAX = 4.0
_LOG_N_MIN = -40.0
_LOG_N_MAX = 80.0


class ParaSLSTM(nn.Module):
    """Four-slot sLSTM: state ``(..., 4, d_h)`` = (c, n, m, h).

    ``max_recurrent_norm=0.5``: same elementwise clip as ParaGRU/LSTM App. C.1.
    For ``mix='head'`` it clamps each block entry (not a spectral bound).
    ``eps=1e-6``: xLSTM-style floor on ``n`` in ``h = o * c / n``.
    ``mix='head'`` requires ``n_heads`` that divides ``d_h``.
    """

    def __init__(
        self,
        d_in: int,
        d_h: int,
        *,
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if mix not in _MIX:
            raise ValueError(f"mix must be one of {_MIX}, got {mix!r}")
        self.d_in = d_in
        self.d_h = d_h
        self.state_slots = SLSTM_SLOTS
        self.hidden_slot = SLSTM_HIDDEN
        self.mix = mix
        self.jac_structure = _JAC[mix]
        self.max_recurrent_norm = max_recurrent_norm
        self.eps = eps
        self.n_heads = n_heads
        self.d_head = None
        # Pre-activations: input, forget, candidate, output (ADD_TASK §2.1).
        self.W_x = nn.Linear(d_in, 4 * d_h, bias=True)
        self.R = None
        self.R_dense = None
        self.R_head = None
        if mix == "diag":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R = nn.Parameter(torch.empty(4, d_h))
        elif mix == "dense":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.R_dense = nn.Linear(d_h, 4 * d_h, bias=False)
        else:
            if n_heads is None or n_heads < 1 or d_h % n_heads != 0:
                raise ValueError(
                    f"mix='head' needs n_heads that divides d_h={d_h}, got {n_heads!r}"
                )
            self.d_head = d_h // n_heads
            self.R_head = nn.Parameter(torch.empty(4, n_heads, self.d_head, self.d_head))
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

        ADD_TASK §2.4 is the channelwise skeleton. We also chain ``tanh`` of
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
        """Newton guess: running ``(c, n, m)`` with ``R h = 0``. See module fn."""
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
        """Zero-hidden, then ``n_picard`` frozen-gate scans. See module fn."""
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


def slstm_frozen_gate_scan(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """sLSTM ``(c, n, m, h)`` with gates frozen in ``pre`` (full preactivation).

    ``m`` is the max-plus prefix ``m_t = max(z_f + m_{t-1}, z_i)``, then
    ``n`` and ``c`` are 1D scans ``q_t = f_t q_{t-1} + ...``. ``h`` is
    readout. Algebra in fp32; DRAM dtype preserved. Span is a prefix
    scan, not a time loop. CUDA fp16/fp32 uses the Triton tiled scan
    (not eager Blelloch; not serial ``chunk_len``).
    """
    if (
        pre.is_cuda
        and pre.dtype in (torch.float16, torch.float32)
        and (h0 is None or h0.is_cuda)
    ):
        from pararnn.kernels.picard_slstm import (
            _BLOCK_T,
            _CHUNK_PAD,
            frozen_gate_scan_triton,
        )

        n_chunks = (pre.shape[1] + _BLOCK_T - 1) // _BLOCK_T
        if n_chunks <= _CHUNK_PAD:
            return frozen_gate_scan_triton(pre, eps=eps, h0=h0)
    return slstm_frozen_gate_scan_eager(pre, eps=eps, h0=h0)


def slstm_frozen_gate_scan_eager(
    pre: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """CPU / fallback twin of ``slstm_frozen_gate_scan`` (eager Blelloch)."""
    from pararnn.solvers.scan import scan_diag

    batch, _, four_d = pre.shape
    d_h = four_d // 4
    orig = pre.dtype
    pre32 = pre.float()
    z_i, z_f, z_z, z_o = pre32.chunk(4, dim=-1)
    if h0 is None:
        c0 = n0 = m0 = pre32.new_zeros(batch, d_h)
    else:
        h0f = h0.float()
        c0 = h0f[:, SLSTM_CELL]
        n0 = h0f[:, SLSTM_NORMALIZER]
        m0 = h0f[:, SLSTM_STABILIZER]
    p = z_f.cumsum(dim=1)
    m = p + torch.maximum((z_i - p).cummax(dim=1).values, m0.unsqueeze(1))
    m_prev = torch.cat((m0.unsqueeze(1), m[:, :-1]), dim=1)
    i_t = torch.exp(z_i - m)
    f_t = torch.exp(z_f + m_prev - m)
    z = torch.tanh(z_z)
    res_n = i_t.clone()
    res_n[:, 0] = f_t[:, 0] * n0 + i_t[:, 0]
    res_c = i_t * z
    res_c[:, 0] = f_t[:, 0] * c0 + i_t[:, 0] * z[:, 0]
    n = scan_diag(f_t, res_n)
    c = scan_diag(f_t, res_c)
    o = torch.sigmoid(z_o)
    h = o * (c / (n + eps))
    return torch.stack((c, n, m, h), dim=-2).to(dtype=orig)


def slstm_zero_hidden_init(
    wx: Tensor,
    *,
    eps: float,
    h0: Tensor | None = None,
) -> Tensor:
    """Newton guess: sLSTM with ``R h = 0`` (keep running ``m`` / ``n``).

    App. A ``f(0, x_t)`` zeros ``n`` independently at each t. Mixing ``R h``
    is left to Newton (or to ``slstm_picard_init``).
    """
    return slstm_frozen_gate_scan(wx, eps=eps, h0=h0)


def slstm_picard_init(
    cell: ParaSLSTM,
    wx: Tensor,
    *,
    h0: Tensor | None = None,
    n_picard: int = 1,
) -> Tensor:
    """Zero-hidden scan, then ``n_picard`` frozen-gate scans using ``R h``.

    Each pass freezes mixing from the previous trajectory and rescans
    ``(c, n, m)`` over all T (cumsum / cummax / ``scan_diag``). Not Jacobi
    ``H := f(H_prev, x)`` (that only moves one token per iter). Not Mamba-2
    (no extra parameters). Cap ``n_picard`` at a handful: P ~ T is sequential.
    """
    if n_picard < 0:
        raise ValueError(f"n_picard must be >= 0, got {n_picard!r}")
    states = slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0)
    for _ in range(n_picard):
        h_prev = prepend_state(states, h0)[..., SLSTM_HIDDEN, :]
        pre = wx + cell._recurrent(h_prev)
        states = slstm_frozen_gate_scan(pre, eps=cell.eps, h0=h0)
    return states


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


def slstm_encode_log(state: Tensor, *, eps: float) -> Tensor:
    """Newton coordinates: ``(u, log n, m, h)`` with ``u = c/n``.

    ``c`` is signed, so ``log c`` is not a coordinate. ``n`` is clamped at
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


class SLSTMLogCoords(nn.Module):
    """Newton cell in ``(u, log n, m, h)``. Convex combination + LSE.

    ``u_t = (1-γ) u_{t-1} + γ tanh(z_z)``, ``log n`` via ``logaddexp``,
    ``h_t = σ(z_o) ⊙ u_t``. Mixing still reads stored ``h``. Sequential
    stays on the native ``(c, n, m, h)`` cell. The fused LSE kernel is a
    Triton twin of this map, not a call into this module.
    """

    def __init__(self, cell: ParaSLSTM) -> None:
        super().__init__()
        object.__setattr__(self, "cell", cell)
        self.d_in = cell.d_in
        self.d_h = cell.d_h
        self.state_slots = cell.state_slots
        self.hidden_slot = cell.hidden_slot
        self.mix = cell.mix
        self.jac_structure = cell.jac_structure
        self.eps = cell.eps
        self.n_heads = cell.n_heads
        self.d_head = cell.d_head

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
        """4×4 of the convex-combination / LSE map. No ``c``, no ``c/n²``."""
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
