"""ParaGRU — Danieli et al. 2025 eq. 3.1a.

``mix='diag'``: ``A_* = diag(a_*)`` (eq. 3.3), fused Newton path.
``mix='head'``: block-diagonal recurrent weights (Dreamer-style block
recurrence) with a full ``W_x`` input mix; CUDA factorized Newton (no
dense ``d×d``); ``scan_backend='eager'`` keeps the dense-J oracle.
Cho gates only — DreamerV3 ports often wrap LayerNorm around the GRU;
that LN is outside this cell.
"""

from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import torch
from torch import Tensor, nn

from pararnn.cells.protocol import resolve_layer_sizes
from pararnn.weight_init import kaiming_uniform_linear_, xavier_gaussian_vec_

log = logging.getLogger(__name__)

_MIX = ("diag", "head")
_JAC = {"diag": "diag", "head": "head"}
_DEFAULT_CAP = object()
_HEAD_WARN = (
    "mix='head' is block-diagonal ParaGRU (Dreamer-style). On CUDA, "
    "Newton uses a factorized-J fused path; scan_backend='eager' keeps the "
    "dense-J oracle. Cho gates only — Dreamer LN-GRU wraps LayerNorm outside "
    "this cell."
)


def _sigmoid_prime_from_act(gate: Tensor) -> Tensor:
    """σ'(pre) when ``gate = σ(pre)``: gate * (1 - gate)."""
    return gate * (1.0 - gate)


def _tanh_prime_from_act(act: Tensor) -> Tensor:
    """tanh'(pre) when ``act = tanh(pre)``: 1 - act^2."""
    return 1.0 - act.square()


class ParaGRU(nn.Module):
    """Fully gated GRU with diagonal or block-diagonal recurrent weights.

    Gates: update ``z``, reset ``r``, candidate ``n`` (paper's ``c``).
    Activations: sigmoid / sigmoid / tanh (Cho et al. 2014, as used in §3).
    Default ``mix='diag'`` is the fused path (Danieli et al. 2025 eq. 3.1a,
    3.3). ``mix='head'`` is block-diagonal recurrent ``A_*`` per head with
    a full input projection (Dreamer-style); CUDA uses factorized Newton,
    ``scan_backend='eager'`` keeps the dense-J oracle.

    Attributes
    ----------
    input_size, d_in : int
        Input feature width.
    hidden_size, d_h : int
        Hidden width.
    state_slots : int
        Always ``1``; state layout is ``(..., d_h)``.
    mix : {'diag', 'head'}
        Recurrent mixing mode.
    jac_structure : str
        ``'diag'`` or ``'head'``, matching ``mix``.
    n_heads, d_head : int or None
        Head geometry for ``mix='head'``.
    max_recurrent_norm : float or None
        App. C.1 elementwise clamp of recurrent entries to ``[-cap, cap]``.
        Default ``0.5`` for ``mix='diag'``; ``None`` for ``mix='head'``.
    a_z, a_r, a_n : Parameter or None
        Diagonal recurrent vectors for ``mix='diag'``, each ``(d_h,)``.
    A_z, A_r, A_n : Parameter or None
        Per-head recurrent matrices for ``mix='head'``, each
        ``(n_heads, d_head, d_head)`` with last dims ``(d_in, d_out)``.
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
        mix: str = "diag",
        n_heads: int | None = None,
        max_recurrent_norm: float | None | object = _DEFAULT_CAP,
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
        mix : {'diag', 'head'}, default='diag'
            ``'diag'`` uses channelwise ``a_*`` (fused). ``'head'`` uses
            dense ``A_*`` inside each of ``n_heads`` blocks (Dreamer-style;
            CUDA factorized Newton; ``scan_backend='eager'`` dense-J oracle).
            Cho gates only — LayerNorm in Dreamer LN-GRU sits outside this cell.
        n_heads : int or None, default=None
            Head count for ``mix='head'`` only; must divide ``hidden_size``.
            Factorized Newton is linear in ``d_head`` matvecs. CUDA path
            tiers: ``d_head≤64`` full fused SRAM; ``64 < d_head ≤128``
            streamed-``A``; larger heads use hybrid tiled Triton (PyTorch
            gates + tiled factor scan / reverse). DreamerV3 often uses
            8 blocks (``d_head=64`` at width 512).
        max_recurrent_norm : float or None, optional
            Elementwise clamp of recurrent entries to
            ``[-max_recurrent_norm, max_recurrent_norm]`` (App. C.1).
            Default ``0.5`` when ``mix='diag'``; ``None`` when ``mix='head'``
            so dense ``A_*`` are not silently clipped on weight load.
        device : torch.device or str or None, default=None
            Parameter device.
        dtype : torch.dtype or None, default=None
            Parameter dtype.

        Raises
        ------
        ValueError
            When ``mix`` is unknown, ``n_heads`` is misused, or
            ``mix='head'`` lacks a dividing ``n_heads``.
        """
        super().__init__()
        if mix not in _MIX:
            raise ValueError(f"mix must be one of {_MIX}, got {mix!r}")
        input_size, hidden_size = resolve_layer_sizes(input_size, hidden_size, d_in=d_in, d_h=d_h)
        factory_kwargs = {"device": device, "dtype": dtype}
        if max_recurrent_norm is _DEFAULT_CAP:
            max_recurrent_norm = None if mix == "head" else 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_in = input_size
        self.d_h = hidden_size
        self.state_slots = 1
        self.mix = mix
        self.jac_structure = _JAC[mix]
        self.max_recurrent_norm = max_recurrent_norm  # type: ignore[assignment]
        self.n_heads = n_heads
        self.d_head = None
        self.a_z = None
        self.a_r = None
        self.a_n = None
        self.A_z = None
        self.A_r = None
        self.A_n = None
        self.W_x = nn.Linear(input_size, 3 * hidden_size, bias=True, **factory_kwargs)
        if mix == "diag":
            if n_heads is not None:
                raise ValueError("n_heads is only for mix='head'")
            self.a_z = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.a_r = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.a_n = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            if n_heads is None or n_heads < 1 or hidden_size % n_heads != 0:
                raise ValueError(
                    f"mix='head' needs n_heads that divides hidden_size={hidden_size}, "
                    f"got {n_heads!r}"
                )
            self.d_head = hidden_size // n_heads
            self.A_z = nn.Parameter(
                torch.empty(n_heads, self.d_head, self.d_head, **factory_kwargs)
            )
            self.A_r = nn.Parameter(
                torch.empty(n_heads, self.d_head, self.d_head, **factory_kwargs)
            )
            self.A_n = nn.Parameter(
                torch.empty(n_heads, self.d_head, self.d_head, **factory_kwargs)
            )
            self.register_buffer(
                "_eye_head",
                torch.eye(self.d_head, **factory_kwargs),
                persistent=False,
            )
            warnings.warn(_HEAD_WARN, UserWarning, stacklevel=2)
            log.warning(
                "paragru_mix_head factorized_cuda_path hidden_size=%d n_heads=%d d_head=%d",
                hidden_size,
                n_heads,
                self.d_head,
            )
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = f"{self.input_size}, {self.hidden_size}, mix={self.mix!r}"
        if self.n_heads is not None:
            s += f", n_heads={self.n_heads}"
        s += f", max_recurrent_norm={self.max_recurrent_norm}"
        return s

    def reset_parameters(self) -> None:
        """Initialize ``W_x`` (Kaiming) and recurrent ``a_*`` / ``A_*``.

        Diagonal ``a_*`` use Xavier-Gaussian vectors. Head ``A_*`` use
        orthogonal init per head at gain ``0.25`` (float32 workspace on
        CUDA when the parameter dtype lacks ``geqrf``).
        """
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        if self.a_z is not None:
            xavier_gaussian_vec_(self.a_z)
            xavier_gaussian_vec_(self.a_r)
            xavier_gaussian_vec_(self.a_n)
            return
        for A in (self.A_z, self.A_r, self.A_n):
            for hd in range(self.n_heads):
                # orthogonal_ needs float32/64 on CUDA (no bf16 geqrf).
                w = torch.empty_like(A[hd], dtype=torch.float32)
                nn.init.orthogonal_(w, gain=0.25)
                A[hd].data.copy_(w.to(dtype=A.dtype))

    def clipped_a(self) -> tuple[Tensor, Tensor, Tensor]:
        """App. C.1 clamp of diagonal recurrent vectors (``mix='diag'``).

        Returns
        -------
        a_z, a_r, a_n : Tensor
            Each of shape ``(d_h,)``. When ``max_recurrent_norm`` is set,
            entries are clamped to ``[-cap, cap]``; otherwise the raw
            parameters are returned.

        Raises
        ------
        TypeError
            When ``mix != 'diag'`` (use :meth:`clipped_a_head`).
        """
        if self.a_z is None:
            raise TypeError("clipped_a is for mix='diag'")
        if self.max_recurrent_norm is None:
            return self.a_z, self.a_r, self.a_n
        cap = self.max_recurrent_norm
        return (
            self.a_z.clamp(-cap, cap),
            self.a_r.clamp(-cap, cap),
            self.a_n.clamp(-cap, cap),
        )

    def clipped_a_head(self) -> tuple[Tensor, Tensor, Tensor]:
        """App. C.1 clamp of per-head recurrent matrices (``mix='head'``).

        Returns
        -------
        A_z, A_r, A_n : Tensor
            Each of shape ``(n_heads, d_head, d_head)`` with last dims
            ``(d_in, d_out)`` so ``y = h @ A``. When ``max_recurrent_norm``
            is set (unusual for head; default is ``None``), entries are
            clamped elementwise to ``[-cap, cap]``.

        Raises
        ------
        TypeError
            When ``mix != 'head'`` (use :meth:`clipped_a`).

        See Also
        --------
        clipped_a : Diagonal counterpart for ``mix='diag'``.
        """
        if self.A_z is None:
            raise TypeError("clipped_a_head is for mix='head'")
        if self.max_recurrent_norm is None:
            return self.A_z, self.A_r, self.A_n
        cap = self.max_recurrent_norm
        return (
            self.A_z.clamp(-cap, cap),
            self.A_r.clamp(-cap, cap),
            self.A_n.clamp(-cap, cap),
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

    def step_head(self, h: Tensor, wx_head: Tensor, a_z: Tensor, a_r: Tensor, a_n: Tensor) -> Tensor:
        """Advance one head with packed head-local tensors.

        Parameters
        ----------
        h : Tensor
            Head hidden. Tensor of shape ``(d_head,)``.
        wx_head : Tensor
            Head input pre-activations. Tensor of shape ``(3, d_head)``.
        a_z, a_r, a_n : Tensor
            Head recurrent matrices. Each of shape ``(d_head, d_head)``
            with dims ``(d_in, d_out)``.

        Returns
        -------
        h_new : Tensor
            Next head hidden. Tensor of shape ``(d_head,)``.
        """
        zx, rx, nx = wx_head.unbind(0)
        z = torch.sigmoid(h @ a_z + zx)
        r = torch.sigmoid(h @ a_r + rx)
        n = torch.tanh((h * r) @ a_n + nx)
        return torch.lerp(h, n, z)

    def step_with_jacobian(
        self, h_prev: Tensor, x: Tensor | None = None, *, wx: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Advance one step and return the structured Jacobian.

        ``mix='diag'``: diagonal ``∂h_new/∂h_prev`` (eq. 3.2a).
        ``mix='head'``: per-head dense ``(..., n_heads, d_head, d_head)``.

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
        jac : Tensor
            Jacobian whose layout matches ``jac_structure``:

            - ``'diag'``: ``(..., d_h)``
            - ``'head'``: ``(..., n_heads, d_head, d_head)`` with
              ``[..., out, in]``

        Raises
        ------
        ValueError
            When both ``x`` and ``wx`` are ``None``.
        """
        acts = self._recurrence(h_prev, x, wx=wx)
        if self.mix == "diag":
            return acts.h_new, self._jac_diag(acts)
        return acts.h_new, self._jac_head(acts)

    def _jac_diag(self, acts: _GRUActs) -> Tensor:
        z_p = _sigmoid_prime_from_act(acts.z)
        r_p = _sigmoid_prime_from_act(acts.r)
        n_p = _tanh_prime_from_act(acts.n)
        return (
            (1.0 - acts.z)
            + (acts.n - acts.h_prev) * z_p * acts.a_z
            + acts.z * n_p * acts.a_n * (acts.r + acts.h_prev * r_p * acts.a_r)
        )

    def _jac_head(self, acts: _GRUActs) -> Tensor:
        """Per-head ``∂h'/∂h`` with layout ``(..., H, d_out, d_in)``."""
        if self.n_heads is None or self.d_head is None:
            raise TypeError(
                f"mix='head' needs n_heads and d_head, got n_heads={self.n_heads!r}, "
                f"d_head={self.d_head!r}"
            )
        h = acts.h_heads
        z = acts.z_heads
        r = acts.r_heads
        n = acts.n_heads
        assert h is not None and z is not None and r is not None and n is not None
        a_z, a_r, a_n = acts.a_z, acts.a_r, acts.a_n
        z_p = _sigmoid_prime_from_act(z)
        r_p = _sigmoid_prime_from_act(r)
        n_p = _tanh_prime_from_act(n)
        # A is (H, in, out); column form A^T is (H, out, in).
        a_z_t = a_z.transpose(-1, -2)
        a_r_t = a_r.transpose(-1, -2)
        a_n_t = a_n.transpose(-1, -2)
        dz_dh = z_p.unsqueeze(-1) * a_z_t
        dr_dh = r_p.unsqueeze(-1) * a_r_t
        eye = self._eye_head.to(device=h.device, dtype=h.dtype)
        du_dh = torch.add(h.unsqueeze(-1) * dr_dh, r.unsqueeze(-1) * eye)
        dn_dh = torch.matmul(n_p.unsqueeze(-1) * a_n_t, du_dh)
        return (
            (1.0 - z).unsqueeze(-1) * eye
            + (n - h).unsqueeze(-1) * dz_dh
            + z.unsqueeze(-1) * dn_dh
        )

    def _head_mix(self, h_v: Tensor, a: Tensor) -> Tensor:
        """``h @ A`` per head. ``h_v`` is ``(..., H, d)``, ``a`` is ``(H, d_in, d_out)``."""
        return torch.matmul(h_v.unsqueeze(-2), a).squeeze(-2)

    def _recurrence(
        self, h_prev: Tensor, x: Tensor | None, *, wx: Tensor | None = None
    ) -> _GRUActs:
        if wx is None:
            if x is None:
                raise ValueError("ParaGRU.step needs x or wx (precomputed W_x(x))")
            wx = self.W_x(x)
        zx, rx, nx = wx.chunk(3, dim=-1)
        if self.mix == "diag":
            a_z, a_r, a_n = self.clipped_a()
            z = torch.sigmoid(a_z * h_prev + zx)
            r = torch.sigmoid(a_r * h_prev + rx)
            n = torch.tanh(a_n * (h_prev * r) + nx)
            h_new = torch.lerp(h_prev, n, z)
            return _GRUActs(
                h_new=h_new,
                h_prev=h_prev,
                z=z,
                r=r,
                n=n,
                a_z=a_z,
                a_r=a_r,
                a_n=a_n,
            )

        a_z, a_r, a_n = self.clipped_a_head()
        prefix = h_prev.shape[:-1]
        h_v = h_prev.reshape(*prefix, self.n_heads, self.d_head)
        zx_v = zx.reshape(*prefix, self.n_heads, self.d_head)
        rx_v = rx.reshape(*prefix, self.n_heads, self.d_head)
        nx_v = nx.reshape(*prefix, self.n_heads, self.d_head)
        z = torch.sigmoid(self._head_mix(h_v, a_z) + zx_v)
        r = torch.sigmoid(self._head_mix(h_v, a_r) + rx_v)
        n = torch.tanh(self._head_mix(h_v * r, a_n) + nx_v)
        h_new_v = torch.lerp(h_v, n, z)
        h_new = h_new_v.reshape(*prefix, self.d_h)
        return _GRUActs(
            h_new=h_new,
            h_prev=h_prev,
            z=z.reshape(*prefix, self.d_h),
            r=r.reshape(*prefix, self.d_h),
            n=n.reshape(*prefix, self.d_h),
            a_z=a_z,
            a_r=a_r,
            a_n=a_n,
            h_heads=h_v,
            z_heads=z,
            r_heads=r,
            n_heads=n,
        )


class _GRUActs(NamedTuple):
    h_new: Tensor
    h_prev: Tensor
    z: Tensor
    r: Tensor
    n: Tensor
    a_z: Tensor
    a_r: Tensor
    a_n: Tensor
    h_heads: Tensor | None = None
    z_heads: Tensor | None = None
    r_heads: Tensor | None = None
    n_heads: Tensor | None = None
