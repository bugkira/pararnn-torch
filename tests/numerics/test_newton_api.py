"""scan_backend=auto, NewtonStats, early-stop."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn import NewtonConfig
from pararnn.cells import ParaGRU, ParaSLSTM
from pararnn.solvers import (
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    sequential_apply,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _DiagTanh(nn.Module):
    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.a = nn.Parameter(0.4 * torch.ones(d_h))
        self.W_x = nn.Linear(d_in, d_h)

    def step(self, h: Tensor, x: Tensor) -> Tensor:
        return torch.tanh(self.a * h + self.W_x(x))


@torch.no_grad()
def test_auto_picks_eager_on_cpu():
    torch.manual_seed(90)
    cpu = torch.device("cpu")
    cell = ParaGRU(d_in=4, d_h=6).to(cpu)
    x = torch.randn(2, 16, 4, device=cpu)
    st = NewtonStats()
    par = newton_apply(cell, x, NewtonConfig(max_iters=3), stats=st)
    seq = sequential_apply(cell, x)
    assert (par - seq).abs().amax() < 1e-4
    assert st.scan_backend == "eager"
    assert st.max_residual < 1e-4
    assert st.iters >= 0


@pytest.mark.cuda
@torch.no_grad()
def test_auto_picks_fused(cuda_device: torch.device) -> None:
    torch.manual_seed(90)
    cell = ParaGRU(d_in=4, d_h=6).to(cuda_device)
    x = torch.randn(2, 16, 4, device=cuda_device)
    st = NewtonStats()
    par = newton_apply(cell, x, NewtonConfig(max_iters=3), stats=st)
    seq = sequential_apply(cell, x)
    assert (par - seq).abs().amax() < 1e-4
    assert st.scan_backend == "fused"
    assert st.max_residual < 1e-4
    assert st.iters >= 0


@torch.no_grad()
def test_auto_custom_cell_picks_eager_on_cpu():
    torch.manual_seed(91)
    cpu = torch.device("cpu")
    cell = _DiagTanh(d_in=4, d_h=5).to(cpu)
    x = torch.randn(2, 12, 4, device=cpu)
    st = NewtonStats()
    newton_apply(cell, x, NewtonConfig(max_iters=3, jacobian="autograd"), stats=st)
    assert st.scan_backend == "eager"


@pytest.mark.cuda
@torch.no_grad()
def test_auto_custom_cell_picks_triton(cuda_device: torch.device) -> None:
    torch.manual_seed(91)
    cell = _DiagTanh(d_in=4, d_h=5).to(cuda_device)
    x = torch.randn(2, 12, 4, device=cuda_device)
    st = NewtonStats()
    newton_apply(cell, x, NewtonConfig(max_iters=3, jacobian="autograd"), stats=st)
    assert st.scan_backend == "triton"


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
    assert st.iters >= 1
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


def test_fused_window_len_requires_time_loop():
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="fused_window_len requires fused_time_loop"):
        newton_apply(cell, x, NewtonConfig(fused_window_len=64, residual_fail=None))


def test_fused_early_exit_requires_residual_atol():
    from pararnn.solvers.newton.config import _validate_config

    with pytest.raises(ValueError, match="fused_early_exit requires residual_atol"):
        _validate_config(NewtonConfig(fused_early_exit=True, residual_atol=None))


def test_fused_early_exit_rejects_time_loop():
    from pararnn.solvers.newton.config import _validate_config

    with pytest.raises(ValueError, match="fused_early_exit cannot combine"):
        _validate_config(
            NewtonConfig(fused_early_exit=True, fused_time_loop=True, fused_window_len=64)
        )


@pytest.mark.cuda
@torch.no_grad()
def test_fused_early_exit_fewer_than_max_iters(cuda_device: torch.device) -> None:
    torch.manual_seed(96)
    cell = ParaGRU(d_in=4, d_h=8).to(cuda_device)
    x = torch.randn(2, 32, 4, device=cuda_device)
    st = NewtonStats()
    y = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=8,
            scan_backend="fused",
            residual_atol=1e-4,
            fused_early_exit=True,
            residual_fail=None,
        ),
        stats=st,
    )
    assert st.iters < 8
    assert st.iters >= 1
    assert st.max_residual < 1e-4
    assert st.scan_backend == "fused"
    y_fixed = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, scan_backend="fused", residual_atol=None, residual_fail=None),
    )
    torch.testing.assert_close(y, y_fixed, atol=2e-4, rtol=2e-4)


def test_fused_time_loop_rejects_chunk_len():
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="fused_time_loop and chunk_len"):
        newton_apply(
            cell,
            x,
            NewtonConfig(
                fused_time_loop=True,
                chunk_len=64,
                residual_fail=None,
            ),
        )


def test_unknown_scan_tile_rejected():
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="unknown scan_tile"):
        newton_apply(cell, x, NewtonConfig(scan_tile="pcr", residual_fail=None))


def test_fused_time_loop_rejects_eager_backend():
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="fused_time_loop requires"):
        newton_apply(
            cell,
            x,
            NewtonConfig(
                fused_time_loop=True,
                scan_backend="eager",
                residual_fail=None,
                picard_iters=0,
            ),
        )


@pytest.mark.cuda
@torch.no_grad()
def test_fused_time_loop_rejects_gru(cuda_device: torch.device) -> None:
    gru = ParaGRU(4, 4).to(cuda_device)
    x = torch.randn(2, 8, 4, device=cuda_device)
    with pytest.raises(TypeError, match="fused_time_loop is ParaSLSTM"):
        newton_apply(
            gru,
            x,
            NewtonConfig(
                scan_backend="fused",
                fused_time_loop=True,
                residual_fail=None,
            ),
        )
