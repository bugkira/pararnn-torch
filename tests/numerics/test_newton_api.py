"""scan_backend=auto, NewtonStats, early-stop."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn import (
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    device,
    newton_apply,
    sequential_apply,
)
from pararnn.cells import ParaGRU, ParaSLSTM


class _DiagTanh(nn.Module):
    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.a = nn.Parameter(0.4 * torch.ones(d_h))
        self.W_x = nn.Linear(d_in, d_h)

    def step(self, h: Tensor, x: Tensor) -> Tensor:
        return torch.tanh(self.a * h + self.W_x(x))


@torch.no_grad()
def test_auto_picks_fused_or_eager():
    torch.manual_seed(90)
    cell = ParaGRU(d_in=4, d_h=6).to(device)
    x = torch.randn(2, 16, 4, device=device)
    st = NewtonStats()
    par = newton_apply(cell, x, NewtonConfig(max_iters=3), stats=st)
    seq = sequential_apply(cell, x)
    assert (par - seq).abs().amax() < 1e-4
    if device.type == "cuda":
        assert st.scan_backend == "fused"
    else:
        assert st.scan_backend == "eager"
    assert st.max_residual < 1e-4
    assert st.iters >= 0


@torch.no_grad()
def test_auto_custom_cell_is_not_fused():
    torch.manual_seed(91)
    cell = _DiagTanh(d_in=4, d_h=5).to(device)
    x = torch.randn(2, 12, 4, device=device)
    st = NewtonStats()
    newton_apply(cell, x, NewtonConfig(max_iters=3, jacobian="autograd"), stats=st)
    if device.type == "cuda":
        assert st.scan_backend == "triton"
    else:
        assert st.scan_backend == "eager"


@torch.no_grad()
def test_early_stop_fewer_than_max_iters():
    torch.manual_seed(92)
    cell = ParaGRU(d_in=4, d_h=6).to(device)
    x = torch.randn(2, 16, 4, device=device)
    st = NewtonStats()
    newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=8, scan_backend="eager", residual_atol=1e-4),
        stats=st,
    )
    assert st.iters < 8
    assert st.max_residual < 1e-4
    assert st.scan_backend == "eager"


def test_fused_rejects_cpu_and_custom_cell():
    torch.manual_seed(93)
    cell = ParaGRU(d_in=4, d_h=6)
    x = torch.randn(2, 8, 4)
    with pytest.raises(TypeError, match="fused"):
        newton_apply(cell, x, NewtonConfig(max_iters=1, scan_backend="fused"))
    custom = _DiagTanh(d_in=4, d_h=5).to(device)
    xc = torch.randn(2, 8, 4, device=device)
    with pytest.raises(TypeError, match="fused"):
        newton_apply(
            custom,
            xc,
            NewtonConfig(max_iters=1, scan_backend="fused", jacobian="autograd"),
        )


@torch.no_grad()
def test_newton_fail_loud_on_zero_iters_slstm():
    torch.manual_seed(94)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 12, 4, device=device)
    with pytest.raises(NewtonDivergenceError, match="residual_fail"):
        newton_apply(
            cell,
            x,
            NewtonConfig(
                max_iters=0,
                scan_backend="eager",
                picard_iters=0,
                residual_atol=None,
                residual_fail=1e-3,
            ),
        )


@torch.no_grad()
def test_newton_fail_loud_can_be_disabled():
    torch.manual_seed(95)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 12, 4, device=device)
    st = NewtonStats()
    newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=0,
            scan_backend="eager",
            picard_iters=0,
            residual_atol=None,
            residual_fail=None,
        ),
        stats=st,
    )
    assert st.iters == 0
    assert st.max_residual > 1e-3
