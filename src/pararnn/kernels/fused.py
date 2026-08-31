"""Dispatch handwritten fused Newton. ParaGRU / ParaLSTM only, not any ``f``."""

from __future__ import annotations

from torch import Tensor, nn

from pararnn.cells.para_gru import ParaGRU
from pararnn.cells.para_lstm import ParaLSTM


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
    raise TypeError(
        f"fused Newton is ParaGRU/ParaLSTM only, not any f; got {type(cell).__name__}"
    )
