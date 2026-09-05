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

from pararnn.kernels.newton_gru import _newton_gru_fused_impl
from pararnn.kernels.newton_lstm import _newton_lstm_fused_impl
from pararnn.kernels.newton_slstm import _newton_slstm_fused_impl
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
