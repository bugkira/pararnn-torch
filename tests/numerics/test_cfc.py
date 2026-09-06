"""ParaCfC: sequential↔Newton numerics and fused CUDA agreement."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaCfC
from pararnn.solvers import NewtonConfig, NewtonStats, newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfc_x(batch: int, time: int, d_in: int, *, seed: int | None = None) -> torch.Tensor:
    """Features + positive Δt in the last channel (``d_in = features + 1``)."""
    if seed is not None:
        torch.manual_seed(seed)
    feat = torch.randn(batch, time, d_in - 1, device=device)
    dt = 0.05 + torch.rand(batch, time, 1, device=device)
    return torch.cat((feat, dt), dim=-1)


@torch.no_grad()
def test_paracfc_newton_matches_sequential() -> None:
    torch.manual_seed(11)
    # d_in=9 → 8 features + Δt (user layout for d_h=16).
    cell = ParaCfC(d_in=9, d_h=16).to(device)
    x = _cfc_x(3, 32, 9, seed=11)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_paracfc_newton_odd_t() -> None:
    torch.manual_seed(12)
    cell = ParaCfC(d_in=5, d_h=8).to(device)
    x = _cfc_x(2, 17, 5, seed=12)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_paracfc_newton_b1_t1() -> None:
    torch.manual_seed(13)
    cell = ParaCfC(d_in=9, d_h=16).to(device)
    x = _cfc_x(1, 1, 9, seed=13)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4


def test_paracfc_bwd_matches_sequential_bptt() -> None:
    torch.manual_seed(14)
    d_in, d_h = 7, 10
    cell_s = ParaCfC(d_in, d_h).to(device)
    cell_n = ParaCfC(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x = _cfc_x(2, 16, d_in, seed=14).requires_grad_(True)
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
def test_paracfc_triton_scan_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(15)
    cell = ParaCfC(d_in=9, d_h=16).to(cuda_device)
    feat = torch.randn(2, 64, 8, device=cuda_device)
    dt = 0.05 + torch.rand(2, 64, 1, device=cuda_device)
    x = torch.cat((feat, dt), dim=-1)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="triton"))
    assert (par - seq).abs().amax() < 2e-4


@pytest.mark.cuda
@torch.no_grad()
def test_fused_paracfc_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(16)
    cell = ParaCfC(d_in=5, d_h=8).to(cuda_device)
    feat = torch.randn(2, 32, 4, device=cuda_device)
    dt = 0.05 + torch.rand(2, 32, 1, device=cuda_device)
    x = torch.cat((feat, dt), dim=-1)
    fused = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="fused"))
    eager = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    assert (fused - eager).abs().amax() < 2e-4


@pytest.mark.cuda
@torch.no_grad()
def test_paracfc_auto_picks_fused(cuda_device: torch.device) -> None:
    torch.manual_seed(17)
    cell = ParaCfC(5, 8).to(cuda_device)
    feat = torch.randn(2, 16, 4, device=cuda_device)
    dt = 0.05 + torch.rand(2, 16, 1, device=cuda_device)
    x = torch.cat((feat, dt), dim=-1)
    stats = NewtonStats()
    y = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="auto"), stats=stats)
    assert y.shape == (2, 16, 8)
    assert stats.scan_backend == "fused"
