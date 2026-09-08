"""``torch.library.custom_op`` wrappers for Triton scans and fused Newton.

Dynamo treats these as opaque ops: ``register_fake`` supplies shape/dtype for
the meta graph so the compiler does not walk into the Triton launch path.

Training grads for fused Newton go through ``newton_apply`` / eq. 2.6
(``Autograd.Function``), not through these ops: the fused forward only sees
``W_x(x)`` and recurrent weights, not ``W_x`` itself. Direct
``requires_grad`` through a fused op is unsupported.
"""

from __future__ import annotations

import torch
from torch import Tensor

from pararnn.kernels.newton_cfc import _newton_cfc_fused_impl
from pararnn.kernels.newton_gru import _newton_gru_fused_impl
from pararnn.kernels.newton_gru_head import (
    _newton_gru_head_fused_impl,
    _reverse_gru_head_factor_impl,
)
from pararnn.kernels.newton_lstm import _newton_lstm_fused_impl
from pararnn.kernels.newton_nlru import _newton_nlru_fused_impl
from pararnn.kernels.newton_slstm import _newton_slstm_fused_impl
from pararnn.kernels.newton_slstm_head import (
    _newton_slstm_head_fused_impl,
    _reverse_slstm_head_factor_impl,
)
from pararnn.kernels.newton_titans import _newton_titans_fused_impl
from pararnn.kernels.scan_dense import _reverse_dense_triton_impl, _scan_dense_triton_impl
from pararnn.kernels.scan_diag import _scan_diag_triton_impl
from pararnn.kernels.scan_lstm_block import _scan_block2_triton_impl
from pararnn.kernels.scan_slstm_block import _scan_block4_triton_impl
from pararnn.layout import SLSTM_SLOTS


@torch.library.custom_op("pararnn::scan_diag", mutates_args=())
def scan_diag_triton(
    jac: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """CUDA diagonal Newton scan. Optional ``cu_seqlens`` for packed time."""
    return _scan_diag_triton_impl(jac, residual, cu_seqlens=cu_seqlens)


@scan_diag_triton.register_fake
def _(jac: Tensor, residual: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
    del jac, cu_seqlens
    return torch.empty_like(residual)


@torch.library.custom_op("pararnn::scan_dense", mutates_args=())
def scan_dense_triton(
    jac: Tensor,
    residual: Tensor,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """CUDA dense ``d×d`` inclusive Newton scan (head GRU / head sLSTM / DEER).

    Parameters
    ----------
    jac : Tensor
        Per-step Jacobians. Tensor of shape ``(B, T, d, d)`` with layout
        ``[..., out, in]``.
    residual : Tensor
        Newton residual. Tensor of shape ``(B, T, d)``.
    cu_seqlens : Tensor or None, default=None
        Optional packed-time offsets; ragged packs use the eager segmented
        path inside the impl.

    Returns
    -------
    delta : Tensor
        Inclusive scan ``δ_t = J_t δ_{t-1} + r_t``. Tensor of shape
        ``(B, T, d)``.
    """
    return _scan_dense_triton_impl(jac, residual, cu_seqlens=cu_seqlens)


@scan_dense_triton.register_fake
def _(jac: Tensor, residual: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
    del jac, cu_seqlens
    return torch.empty_like(residual)


@torch.library.custom_op("pararnn::reverse_scan_dense", mutates_args=())
def reverse_scan_dense_triton(
    jac: Tensor,
    partial: Tensor,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """CUDA dense ``d×d`` reverse Newton scan (eq. 2.6).

    Parameters
    ----------
    jac : Tensor
        Per-step Jacobians. Tensor of shape ``(B, T, d, d)`` with layout
        ``[..., out, in]``.
    partial : Tensor
        Incoming reverse partial. Tensor of shape ``(B, T, d)``.
    cu_seqlens : Tensor or None, default=None
        Optional packed-time offsets; ragged packs use the eager segmented
        path inside the impl.

    Returns
    -------
    mu : Tensor
        Reverse adjoint ``μ_t = J_{t+1}^T μ_{t+1} + g_t``. Tensor of shape
        ``(B, T, d)``.
    """
    return _reverse_dense_triton_impl(jac, partial, cu_seqlens=cu_seqlens)


@reverse_scan_dense_triton.register_fake
def _(jac: Tensor, partial: Tensor, cu_seqlens: Tensor | None = None) -> Tensor:
    del jac, cu_seqlens
    return torch.empty_like(partial)


@torch.library.custom_op("pararnn::scan_block2", mutates_args=())
def scan_block2_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """CUDA 2×2 block Newton scan (LSTM channelwise)."""
    return _scan_block2_triton_impl(jac, residual)


@scan_block2_triton.register_fake
def _(jac: Tensor, residual: Tensor) -> Tensor:
    del jac
    return torch.empty_like(residual)


@torch.library.custom_op("pararnn::scan_block4", mutates_args=())
def scan_block4_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """CUDA 4×4 block Newton scan (sLSTM channelwise-diagonal)."""
    return _scan_block4_triton_impl(jac, residual)


@scan_block4_triton.register_fake
def _(jac: Tensor, residual: Tensor) -> Tensor:
    del jac
    return torch.empty_like(residual)


@torch.library.custom_op("pararnn::newton_gru_fused", mutates_args=())
def newton_gru_fused(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused Alg. 1 for diagonal ParaGRU. ``wx`` is ``W_x(x)`` ``(B, T, 3 d_h)``."""
    return _newton_gru_fused_impl(
        wx,
        a_z,
        a_r,
        a_n,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


@newton_gru_fused.register_fake
def _(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del a_r, a_n, h0, cu_seqlens, block_table, max_iters, omega
    batch, time, _ = wx.shape
    return wx.new_empty(batch, time, int(a_z.numel()))


@torch.library.custom_op("pararnn::newton_nlru_fused", mutates_args=())
def newton_nlru_fused(
    wx: Tensor,
    u: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused Alg. 1 for ParaNLRU. ``wx`` is ``W_x(x)`` ``(B, T, 2 d_h)``."""
    return _newton_nlru_fused_impl(
        wx,
        u,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


@newton_nlru_fused.register_fake
def _(
    wx: Tensor,
    u: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del h0, cu_seqlens, block_table, max_iters, omega
    batch, time, _ = wx.shape
    return wx.new_empty(batch, time, int(u.numel()))


@torch.library.custom_op("pararnn::newton_cfc_fused", mutates_args=())
def newton_cfc_fused(
    wx: Tensor,
    u: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused Alg. 1 for ParaCfC. ``wx`` is ``project_wx(x)`` ``(B, T, 4 d_h)``."""
    return _newton_cfc_fused_impl(
        wx,
        u,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


@newton_cfc_fused.register_fake
def _(
    wx: Tensor,
    u: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del h0, cu_seqlens, block_table, max_iters, omega
    batch, time, _ = wx.shape
    return wx.new_empty(batch, time, int(u.numel()))


@torch.library.custom_op("pararnn::newton_titans_fused", mutates_args=())
def newton_titans_fused(
    wx: Tensor,
    u: Tensor,
    r: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused Alg. 1 for ParaTitans. ``wx`` is ``W_x(x)`` ``(B, T, 5 d_h)``."""
    return _newton_titans_fused_impl(
        wx,
        u,
        r,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
        block_table=block_table,
    )


@newton_titans_fused.register_fake
def _(
    wx: Tensor,
    u: Tensor,
    r: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del r, h0, cu_seqlens, block_table, max_iters, omega
    batch, time, _ = wx.shape
    return wx.new_empty(batch, time, int(u.numel()))


@torch.library.custom_op("pararnn::newton_gru_head_fused", mutates_args=())
def newton_gru_head_fused(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused / factorized Alg. 1 for ``ParaGRU(mix='head')``.

    Rectangular CUDA batches use path tiers by ``d_head``: ``≤64`` full
    fused SRAM; ``64 < d_head ≤128`` streamed-``A``; larger hybrid tiled
    Triton (PyTorch gates + tiled factor scan). No dense ``d×d`` Jacobian
    buffer. ``cu_seqlens`` is rejected at ``fused_newton`` / dispatch with
    :exc:`TypeError` before this op; pad to a rectangular batch or use
    ``scan_backend='eager'`` for ragged packs.

    Parameters
    ----------
    wx : Tensor
        Precomputed ``W_x(x)``. Tensor of shape ``(B, T, 3 d_h)``.
    a_z, a_r, a_n : Tensor
        Per-head recurrent matrices. Each of shape
        ``(n_heads, d_head, d_head)`` with last dims ``(d_in, d_out)``.
    h0 : Tensor or None, default=None
        Optional initial hidden. Tensor of shape ``(B, d_h)``.
    cu_seqlens : Tensor or None, default=None
        Packed-time offsets. Callers via ``fused_newton`` pass ``None``;
        that dispatcher raises :exc:`TypeError` when packs are requested.
    max_iters : int
        Newton steps ``K``.
    omega : float
        Step damping (``1.0`` = undamped).

    Returns
    -------
    H : Tensor
        Parallel Newton states. Tensor of shape ``(B, T, d_h)``.

    Notes
    -----
    Training grads go through ``newton_apply`` / eq. 2.6
    (``Autograd.Function``).
    """
    return _newton_gru_head_fused_impl(
        wx,
        a_z,
        a_r,
        a_n,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        cu_seqlens=cu_seqlens,
    )


@newton_gru_head_fused.register_fake
def _(
    wx: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
    h0: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del a_r, a_n, h0, cu_seqlens, max_iters, omega
    batch, time, _ = wx.shape
    d_h = int(a_z.shape[0] * a_z.shape[-1])
    return wx.new_empty(batch, time, d_h)


@torch.library.custom_op("pararnn::reverse_gru_head_factor", mutates_args=())
def reverse_gru_head_factor(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
) -> Tensor:
    """Eq. 2.6 factorized reverse for head ParaGRU (opaque to Dynamo).

    Parameters
    ----------
    h_prev : Tensor
        Prepended previous hidden (layout from the Newton reverse). Tensor
        of shape ``(B, T, d_h)``.
    wx : Tensor
        Precomputed ``W_x(x)`` aligned with ``h_prev`` time. Tensor of shape
        ``(B, T, 3 d_h)``.
    partial : Tensor
        Incoming reverse partial ``g`` / ``μ`` seed. Tensor of shape
        ``(B, T, d_h)``.
    a_z, a_r, a_n : Tensor
        Per-head recurrent matrices. Each of shape
        ``(n_heads, d_head, d_head)`` with last dims ``(d_in, d_out)``.

    Returns
    -------
    mu : Tensor
        Reverse adjoint ``μ_t = J_{t+1}^T μ_{t+1} + g_t``. Tensor of shape
        ``(B, T, d_h)``.
    """
    return _reverse_gru_head_factor_impl(h_prev, wx, partial, a_z, a_r, a_n)


@reverse_gru_head_factor.register_fake
def _(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    a_z: Tensor,
    a_r: Tensor,
    a_n: Tensor,
) -> Tensor:
    del h_prev, wx, a_z, a_r, a_n
    return torch.empty_like(partial)


@torch.library.custom_op("pararnn::newton_lstm_fused", mutates_args=())
def newton_lstm_fused(
    wx: Tensor,
    a_f: Tensor,
    a_z: Tensor,
    a_o: Tensor,
    c_f: Tensor,
    c_o: Tensor,
    h0: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    """Fused Alg. 1 for CIFG ParaLSTM. ``wx`` is ``W_x(x)`` ``(B, T, 3 d_h)``."""
    return _newton_lstm_fused_impl(
        wx,
        a_f,
        a_z,
        a_o,
        c_f,
        c_o,
        max_iters=max_iters,
        omega=omega,
        h0=h0,
        block_table=block_table,
    )


@newton_lstm_fused.register_fake
def _(
    wx: Tensor,
    a_f: Tensor,
    a_z: Tensor,
    a_o: Tensor,
    c_f: Tensor,
    c_o: Tensor,
    h0: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
) -> Tensor:
    del a_z, a_o, c_f, c_o, h0, block_table, max_iters, omega
    batch, time, _ = wx.shape
    d_h = int(a_f.numel())
    return wx.new_empty(batch, time, 2, d_h)


@torch.library.custom_op("pararnn::newton_slstm_fused", mutates_args=())
def newton_slstm_fused(
    wx: Tensor,
    r: Tensor,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    log_coords: bool = False,
    scan_tile: str = "assoc",
    time_loop: bool = False,
    window_len: int = 0,
) -> Tensor:
    """Fused Alg. 1 for diag ParaSLSTM. ``window_len=0`` means library default."""
    return _newton_slstm_fused_impl(
        wx,
        r,
        max_iters=max_iters,
        omega=omega,
        eps=eps,
        h0=h0,
        states=states,
        log_coords=log_coords,
        scan_tile=scan_tile,
        time_loop=time_loop,
        window_len=None if window_len == 0 else window_len,
        block_table=block_table,
    )


@newton_slstm_fused.register_fake
def _(
    wx: Tensor,
    r: Tensor,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    block_table: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    log_coords: bool = False,
    scan_tile: str = "assoc",
    time_loop: bool = False,
    window_len: int = 0,
) -> Tensor:
    del h0, block_table, max_iters, omega, eps, log_coords, scan_tile, time_loop, window_len
    if states is not None:
        return torch.empty_like(states)
    batch, time, _ = wx.shape
    d_h = int(r.shape[-1])
    return wx.new_empty(batch, time, SLSTM_SLOTS, d_h)


@torch.library.custom_op("pararnn::newton_slstm_head_fused", mutates_args=())
def newton_slstm_head_fused(
    wx: Tensor,
    r_head: Tensor,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    eps: float,
) -> Tensor:
    """Factorized Alg. 1 for ``ParaSLSTM(mix='head')``.

    Parameters
    ----------
    wx : Tensor
        ``W_x(x)`` of shape ``(B, T, 4 d_h)``.
    r_head : Tensor
        Clipped recurrent mix ``(4, n_heads, d_head, d_head)``.
    h0 : Tensor or None
        Optional initial state ``(B, 4, d_h)``.
    states : Tensor or None
        Optional Picard / zero-hidden guess ``(B, T, 4, d_h)``.
    max_iters : int
        Newton iterations (head recipe often ``K=4``).
    omega : float
        Damping (``1.0`` = vanilla Newton).
    eps : float
        Floor on the normalizer ``n`` in the readout.

    Returns
    -------
    states : Tensor
        Converged trajectory ``(B, T, 4, d_h)``.

    Notes
    -----
    Rectangular batches only. ``cu_seqlens`` raises in the impl / dispatch.
    """
    return _newton_slstm_head_fused_impl(
        wx,
        r_head,
        max_iters=max_iters,
        omega=omega,
        eps=eps,
        h0=h0,
        states=states,
    )


@newton_slstm_head_fused.register_fake
def _(
    wx: Tensor,
    r_head: Tensor,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    eps: float,
) -> Tensor:
    del h0, max_iters, omega, eps
    if states is not None:
        return torch.empty_like(states)
    batch, time, _ = wx.shape
    d_h = int(r_head.shape[1] * r_head.shape[-1])
    return wx.new_empty(batch, time, SLSTM_SLOTS, d_h)


@torch.library.custom_op("pararnn::reverse_slstm_head_factor", mutates_args=())
def reverse_slstm_head_factor(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    r_head: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Eq. 2.6 factorized reverse for head ParaSLSTM (opaque to Dynamo)."""
    n_heads = int(r_head.shape[1])
    d_head = int(r_head.shape[-1])
    return _reverse_slstm_head_factor_impl(
        h_prev, wx, partial, r_head, n_heads=n_heads, d_head=d_head, eps=eps
    )


@reverse_slstm_head_factor.register_fake
def _(
    h_prev: Tensor,
    wx: Tensor,
    partial: Tensor,
    r_head: Tensor,
    *,
    eps: float,
) -> Tensor:
    del h_prev, wx, r_head, eps
    return torch.empty_like(partial)


@torch.library.custom_op("pararnn::newton_m2rnn_fused", mutates_args=())
def newton_m2rnn_fused(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    frozen_w_init: bool = False,
) -> Tensor:
    """Fused factorized Alg. 1 for ``ParaM2RNN`` (SRAM ``K,V ≤ 64``).

    Training grads go through ``newton_apply`` / eq. 2.6.
    """
    from pararnn.kernels.newton_m2rnn import _newton_m2rnn_fused_impl

    return _newton_m2rnn_fused_impl(
        k,
        v,
        f,
        w,
        h0,
        max_iters=max_iters,
        omega=omega,
        frozen_w_init=frozen_w_init,
    )


@newton_m2rnn_fused.register_fake
def _(
    k: Tensor,
    v: Tensor,
    f: Tensor,
    w: Tensor,
    h0: Tensor | None = None,
    *,
    max_iters: int,
    omega: float,
    frozen_w_init: bool = False,
) -> Tensor:
    del f, w, h0, max_iters, omega, frozen_w_init
    batch, time, k_dim = k.shape
    v_dim = int(v.shape[-1])
    return k.new_empty(batch, time, k_dim, v_dim)


@torch.library.custom_op("pararnn::reverse_m2rnn_factor", mutates_args=())
def reverse_m2rnn_factor(
    z: Tensor,
    f: Tensor,
    w: Tensor,
    partial: Tensor,
) -> Tensor:
    """Eq. 2.6 factorized reverse for ``ParaM2RNN`` (opaque to Dynamo)."""
    from pararnn.kernels.newton_m2rnn import _reverse_m2rnn_factor_impl

    return _reverse_m2rnn_factor_impl(z, f, w, partial)


@reverse_m2rnn_factor.register_fake
def _(z: Tensor, f: Tensor, w: Tensor, partial: Tensor) -> Tensor:
    del z, f, w
    return torch.empty_like(partial)
