"""Overflow / NaN stress: residual_fail must abort; log decode stays finite."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaGRU, ParaSLSTM
from pararnn.layout import SLSTM_CELL, SLSTM_NORMALIZER, SLSTM_SLOTS
from pararnn.solvers import NewtonConfig, NewtonDivergenceError, newton_apply
from pararnn.solvers.slstm_log import slstm_decode_log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_nan_input_raises_residual_fail() -> None:
    torch.manual_seed(1)
    cell = ParaGRU(d_in=4, d_h=4).to(device)
    x = torch.randn(2, 12, 4, device=device)
    x[0, 0, 0] = float("nan")
    with pytest.raises(NewtonDivergenceError, match="residual_fail"):
        newton_apply(
            cell,
            x,
            NewtonConfig(max_iters=3, residual_fail=1e-2, residual_atol=None),
        )


@torch.no_grad()
def test_exploded_slstm_weights_raise_residual_fail() -> None:
    """Huge logits / scale: Newton does not converge; residual_fail aborts."""
    torch.manual_seed(2)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    with torch.no_grad():
        for p in cell.parameters():
            p.mul_(1e3)
    x = 1e3 * torch.randn(2, 32, 4, device=device)
    with pytest.raises(NewtonDivergenceError, match="residual_fail"):
        newton_apply(
            cell,
            x,
            NewtonConfig(
                max_iters=1,
                picard_iters=0,
                residual_fail=1e-3,
                residual_atol=None,
                scan_backend="eager",
            ),
        )


@torch.no_grad()
def test_slstm_log_decode_exploded_n_stays_finite() -> None:
    """Running-max style log decode: huge cell / n stay finite (no Inf)."""
    u = torch.zeros(2, SLSTM_SLOTS, 8, device=device)
    u[:, SLSTM_CELL] = 500.0
    u[:, SLSTM_NORMALIZER] = 500.0
    out = slstm_decode_log(u, eps=1e-6)
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_huge_input_gru_stays_finite() -> None:
    """Saturated GRU gates: large |x| still yields a finite fixed point."""
    torch.manual_seed(3)
    cell = ParaGRU(d_in=4, d_h=4).to(device)
    x = 1e6 * torch.randn(2, 16, 4, device=device)
    y = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, residual_fail=1e-2, residual_atol=None),
    )
    assert torch.isfinite(y).all()
