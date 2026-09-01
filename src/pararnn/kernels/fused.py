"""Dispatch handwritten fused Newton. Not any ``f``.

ParaGRU / ParaLSTM, and ParaSLSTM with ``mix='diag'`` (4x4 SRAM). Head/dense
sLSTM stay on the PyTorch cell + scan_dense path. ParaSLSTM ``log_coords``
uses the LSE cell inside the same kernel; ``picard_iters`` is a PyTorch
frozen-gate scan before the fused Newton (still O(log T)). Chunking stays
a Python loop.
"""

from __future__ import annotations

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
) -> Tensor:
    if isinstance(cell, ParaGRU):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        from pararnn.kernels.newton_gru import newton_gru_fused

        a_z, a_r, a_n = cell.clipped_a()
        return newton_gru_fused(
            wx, a_z, a_r, a_n, max_iters=max_iters, omega=omega, h0=h0
        )
    if isinstance(cell, ParaLSTM):
        if log_coords:
            raise TypeError("fused log coords is ParaSLSTM only")
        if picard_iters:
            raise TypeError("fused Picard is ParaSLSTM only")
        from pararnn.kernels.newton_lstm import newton_lstm_fused

        a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
        return newton_lstm_fused(
            wx, a_f, a_z, a_o, c_f, c_o, max_iters=max_iters, omega=omega, h0=h0
        )
    if isinstance(cell, ParaSLSTM):
        if cell.mix != "diag":
            raise TypeError(
                "fused Newton is mix='diag' only (4x4 SRAM); "
                f"got mix={cell.mix!r}"
            )
        from pararnn.cells.para_slstm import slstm_picard_init, slstm_zero_hidden_init
        from pararnn.kernels.newton_slstm import newton_slstm_fused

        if picard_iters:
            states = slstm_picard_init(
                cell, wx, h0=h0, n_picard=picard_iters
            )
        else:
            states = slstm_zero_hidden_init(wx, eps=cell.eps, h0=h0)
        return newton_slstm_fused(
            wx,
            cell.clipped_r(),
            max_iters=max_iters,
            omega=omega,
            eps=cell.eps,
            h0=h0,
            states=states,
            log_coords=log_coords,
        )
    raise TypeError(
        f"fused Newton is ParaGRU/ParaLSTM/ParaSLSTM(diag) only, not any f; "
        f"got {type(cell).__name__}"
    )
