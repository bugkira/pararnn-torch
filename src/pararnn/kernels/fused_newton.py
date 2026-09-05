"""Dispatch fused Newton for ParaGRU, ParaLSTM, and ParaSLSTM ``mix='diag'``.

Head/dense sLSTM stay on the PyTorch cell + scan path. ``log_coords`` uses
the LSE cell in the same kernel; ``picard_iters`` is the frozen-gate scan
before 4×4 Newton. ``chunk_len`` stays a Python loop.

Default path is ``pararnn::newton_*_fused`` custom ops (fixed ``max_iters``).
``residual_fn`` + ``early_exit_atol`` bypass the custom op and call the impl
directly (experimental ``NewtonConfig.fused_early_exit``).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM
from pararnn.cells.para_slstm import ParaSLSTM


def fused_newton(
    cell: nn.Module,
    wx: Tensor,
    *,
    max_iters: int,
    omega: float,
    h0: Tensor | None,
    log_coords: bool = False,
    picard_iters: int = 0,
    scan_tile: str = "assoc",
    fused_time_loop: bool = False,
    fused_window_len: int | None = None,
    cu_seqlens: Tensor | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_fn: Callable[[Tensor], float] | None = None,
    iters_done_out: list[int] | None = None,
) -> Tensor:
    early = residual_fn is not None and early_exit_atol is not None
    if isinstance(cell, ParaGRU):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        if scan_tile != "assoc":
            raise TypeError("fused scan_tile is ParaSLSTM only")
        if fused_time_loop:
            raise TypeError("fused_time_loop is ParaSLSTM only")
        a_z, a_r, a_n = cell.clipped_a()
        if early:
            from pararnn.kernels.newton_gru import _newton_gru_fused_impl

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
                early_exit_atol=early_exit_atol,
                residual_fn=residual_fn,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_gru_fused

        return newton_gru_fused(
            wx,
            a_z,
            a_r,
            a_n,
            h0,
            cu_seqlens,
            block_table,
            max_iters=max_iters,
            omega=omega,
        )
    if isinstance(cell, ParaLSTM):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        if scan_tile != "assoc":
            raise TypeError("fused scan_tile is ParaSLSTM only")
        if fused_time_loop:
            raise TypeError("fused_time_loop is ParaSLSTM only")
        a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
        if early:
            from pararnn.kernels.newton_lstm import _newton_lstm_fused_impl

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
                early_exit_atol=early_exit_atol,
                residual_fn=residual_fn,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_lstm_fused

        return newton_lstm_fused(
            wx,
            a_f,
            a_z,
            a_o,
            c_f,
            c_o,
            h0,
            block_table,
            max_iters=max_iters,
            omega=omega,
        )
    if isinstance(cell, ParaSLSTM):
        if cell.mix != "diag":
            raise TypeError(f"fused Newton is mix='diag' only (4x4 SRAM); got mix={cell.mix!r}")
        from pararnn.solvers.slstm_picard import (
            slstm_picard_init,
            slstm_zero_hidden_init,
        )

        h0_init = h0
        if block_table is not None and h0 is not None:
            h0_init = h0.index_select(0, block_table.long())
        if picard_iters:
            states = slstm_picard_init(cell, wx, h0=h0_init, n_picard=picard_iters)
        else:
            states = slstm_zero_hidden_init(wx, eps=cell.eps, h0=h0_init)
        if early:
            from pararnn.kernels.newton_slstm import _newton_slstm_fused_impl

            return _newton_slstm_fused_impl(
                wx,
                cell.clipped_r(),
                h0=h0,
                states=states,
                block_table=block_table,
                max_iters=max_iters,
                omega=omega,
                eps=cell.eps,
                log_coords=log_coords,
                scan_tile=scan_tile,
                time_loop=fused_time_loop,
                window_len=0 if fused_window_len is None else int(fused_window_len),
                early_exit_atol=early_exit_atol,
                residual_fn=residual_fn,
                iters_done_out=iters_done_out,
            )
        from pararnn.kernels.custom_ops import newton_slstm_fused

        return newton_slstm_fused(
            wx,
            cell.clipped_r(),
            h0,
            states,
            block_table,
            max_iters=max_iters,
            omega=omega,
            eps=cell.eps,
            log_coords=log_coords,
            scan_tile=scan_tile,
            time_loop=fused_time_loop,
            window_len=0 if fused_window_len is None else int(fused_window_len),
        )
    raise TypeError(
        f"fused Newton is ParaGRU/ParaLSTM/ParaSLSTM(diag) only; got {type(cell).__name__}"
    )
