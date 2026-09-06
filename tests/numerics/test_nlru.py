"""ParaNLRU: sequential↔Newton numerics and fused CUDA agreement."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaNLRU
from pararnn.solvers import NewtonConfig, NewtonStats, newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_paranlru_newton_matches_sequential() -> None:
    torch.manual_seed(11)
    cell = ParaNLRU(d_in=8, d_h=16).to(device)
    x = torch.randn(3, 32, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_paranlru_newton_odd_t() -> None:
    torch.manual_seed(12)
    cell = ParaNLRU(d_in=4, d_h=8).to(device)
    x = torch.randn(2, 17, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_paranlru_newton_b1_t1() -> None:
    torch.manual_seed(13)
    cell = ParaNLRU(d_in=8, d_h=16).to(device)
    x = torch.randn(1, 1, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4


def test_paranlru_bwd_matches_sequential_bptt() -> None:
    torch.manual_seed(14)
    d_in, d_h = 6, 10
    cell_s = ParaNLRU(d_in, d_h).to(device)
    cell_n = ParaNLRU(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x = torch.randn(2, 16, d_in, device=device, requires_grad=True)
    x_n = x.detach().clone().requires_grad_(True)
    w = torch.randn(2, 16, d_h, device=device)
    loss_s = (sequential_apply(cell_s, x) * w).sum()
    loss_n = (newton_apply(cell_n, x_n, NewtonConfig(max_iters=3, scan_backend="eager")) * w).sum()
    loss_s.backward()
    loss_n.backward()
    assert x.grad is not None and x_n.grad is not None
    torch.testing.assert_close(x_n.grad, x.grad, atol=2e-5, rtol=2e-5)
    for (n_s, p_s), (n_n, p_n) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert n_s == n_n
        assert p_s.grad is not None and p_n.grad is not None
        torch.testing.assert_close(p_n.grad, p_s.grad, atol=2e-5, rtol=2e-5)


@pytest.mark.cuda
@torch.no_grad()
def test_paranlru_triton_scan_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(15)
    cell = ParaNLRU(d_in=8, d_h=16).to(cuda_device)
    x = torch.randn(2, 64, 8, device=cuda_device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="triton"))
    assert (par - seq).abs().amax() < 2e-4


@pytest.mark.cuda
@torch.no_grad()
def test_fused_paranlru_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(16)
    cell = ParaNLRU(d_in=4, d_h=8).to(cuda_device)
    x = torch.randn(2, 32, 4, device=cuda_device)
    fused = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="fused"))
    eager = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (fused - eager).abs().amax() < 2e-4


@pytest.mark.cuda
@torch.no_grad()
def test_paranlru_auto_picks_fused(cuda_device: torch.device) -> None:
    torch.manual_seed(17)
    cell = ParaNLRU(4, 8).to(cuda_device)
    x = torch.randn(2, 16, 4, device=cuda_device)
    stats = NewtonStats()
    y = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="auto"), stats=stats)
    assert y.shape == (2, 16, 8)
    assert stats.scan_backend == "fused"
