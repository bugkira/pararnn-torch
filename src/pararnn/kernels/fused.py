"""Dispatch handwritten fused Newton. Not any ``f``.

ParaGRU / ParaLSTM, and ParaSLSTM with ``mix='diag'`` (4x4 SRAM). Head/dense
sLSTM stay on the PyTorch cell + scan_dense path.
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
) -> Tensor:
    if isinstance(cell, ParaGRU):
        from pararnn.kernels.newton_gru import newton_gru_fused

        a_z, a_r, a_n = cell.clipped_a()
        return newton_gru_fused(
            wx, a_z, a_r, a_n, max_iters=max_iters, omega=omega, h0=h0
        )
    if isinstance(cell, ParaLSTM):
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
        from pararnn.cells.para_slstm import slstm_zero_hidden_init
        from pararnn.kernels.newton_slstm import newton_slstm_fused

        return newton_slstm_fused(
            wx,
            cell.clipped_r(),
            max_iters=max_iters,
            omega=omega,
            eps=cell.eps,
            h0=h0,
            states=slstm_zero_hidden_init(wx, eps=cell.eps, h0=h0),
        )
    raise TypeError(
        f"fused Newton is ParaGRU/ParaLSTM/ParaSLSTM(diag) only, not any f; "
        f"got {type(cell).__name__}"
    )
