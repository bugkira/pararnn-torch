"""ParaRWKV7: linear matrix-state delta monoid (sequential + scan)."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaRWKV7
from pararnn.cells.protocol import check_cell
from pararnn.kernels.rwkv7_scan import (
    rwkv7_apply_factorized,
    rwkv7_associative_scan,
    rwkv7_build_g,
    rwkv7_outer_vk,
    rwkv7_readout,
)
from pararnn.solvers import NewtonConfig, NewtonStats, newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_pararwkv7_shapes_and_protocol() -> None:
    cell = ParaRWKV7(d_in=8, n_heads=2, d_head=4)
    check_cell(cell)
    assert cell.d_h == 8
    assert cell.state_shape == (2, 4, 4)
    assert cell.jac_structure == "rwkv7"
    s = torch.randn(3, 2, 4, 4)
    x = torch.randn(3, 8)
    s2 = cell.step(s, x)
    assert s2.shape == s.shape
    y = cell.readout(s2, x)
    assert y.shape == (3, 8)


@torch.no_grad()
def test_pararwkv7_dh_alias() -> None:
    cell = ParaRWKV7(d_in=6, d_h=12, n_heads=3)
    assert cell.d_head == 4
    assert cell.n_heads == 3


@torch.no_grad()
def test_pararwkv7_factorized_matches_dense_g() -> None:
    torch.manual_seed(0)
    b, h, d = 2, 1, 5
    s = torch.randn(b, h, d, d)
    w = torch.sigmoid(torch.randn(b, h, d))
    a = torch.sigmoid(torch.randn(b, h, d))
    kappa = torch.nn.functional.normalize(torch.randn(b, h, d), dim=-1)
    v = torch.randn(b, h, d)
    k = torch.randn(b, h, d)
    fac = rwkv7_apply_factorized(s, w, a, kappa, v, k)
    g = rwkv7_build_g(w, a, kappa)
    dense = s @ g + rwkv7_outer_vk(v, k)
    torch.testing.assert_close(fac, dense, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_pararwkv7_scan_matches_sequential() -> None:
    torch.manual_seed(1)
    cell = ParaRWKV7(d_in=7, n_heads=2, d_head=4).to(device).eval()
    x = 0.2 * torch.randn(3, 17, 7, device=device)
    seq = sequential_apply(cell, x)
    w, a, kappa, v, k, r = cell.project(x)
    g = rwkv7_build_g(w, a, kappa)
    u = rwkv7_outer_vk(v, k)
    scanned = rwkv7_associative_scan(g, u)
    torch.testing.assert_close(scanned, seq, atol=1e-5, rtol=1e-5)
    y_scan, s_scan = cell.scan_apply(x, return_state=True)
    assert isinstance(s_scan, torch.Tensor)
    torch.testing.assert_close(s_scan, seq, atol=1e-5, rtol=1e-5)
    y_seq = rwkv7_readout(seq, r)
    torch.testing.assert_close(y_scan, y_seq, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_pararwkv7_newton_redirect_matches_sequential() -> None:
    torch.manual_seed(2)
    cell = ParaRWKV7(d_in=5, n_heads=1, d_head=4).to(device).eval()
    x = 0.2 * torch.randn(2, 24, 5, device=device)
    seq = sequential_apply(cell, x)
    stats = NewtonStats()
    # Default backend → associative scan redirect (iters=0).
    par = newton_apply(cell, x, NewtonConfig(), stats=stats)
    torch.testing.assert_close(par, seq, atol=1e-5, rtol=1e-5)
    assert stats.iters == 0
    assert stats.scan_backend == "rwkv7_scan"
    # Explicit eager → sequential redirect.
    stats2 = NewtonStats()
    par2 = newton_apply(cell, x, NewtonConfig(scan_backend="eager"), stats=stats2)
    torch.testing.assert_close(par2, seq, atol=1e-5, rtol=1e-5)
    assert stats2.scan_backend == "rwkv7_sequential"


@torch.no_grad()
def test_pararwkv7_scan_with_h0() -> None:
    torch.manual_seed(3)
    cell = ParaRWKV7(d_in=4, n_heads=1, d_head=3).eval()
    x = 0.15 * torch.randn(2, 9, 4)
    h0 = torch.randn(2, 1, 3, 3)
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(cell, x, NewtonConfig(), h0=h0)
    torch.testing.assert_close(par, seq, atol=1e-5, rtol=1e-5)


def test_pararwkv7_step_gradcheck() -> None:
    torch.manual_seed(4)
    cell = ParaRWKV7(d_in=3, n_heads=1, d_head=2, dtype=torch.float64)
    s = torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True)
    x = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)

    def f(ss: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        return cell.step(ss, xx)

    assert torch.autograd.gradcheck(f, (s, x), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_pararwkv7_sequential_gradcheck() -> None:
    torch.manual_seed(5)
    cell = ParaRWKV7(d_in=2, n_heads=1, d_head=2, dtype=torch.float64)
    x = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)

    def f(xx: torch.Tensor) -> torch.Tensor:
        return sequential_apply(cell, xx)

    assert torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_pararwkv7_newton_grad_matches_sequential() -> None:
    torch.manual_seed(6)
    cell_s = ParaRWKV7(d_in=4, n_heads=1, d_head=3).to(device)
    cell_n = ParaRWKV7(d_in=4, n_heads=1, d_head=3).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x = torch.randn(2, 8, 4, device=device, requires_grad=True)
    x_n = x.detach().clone().requires_grad_(True)
    w = torch.randn(2, 8, 1, 3, 3, device=device)
    loss_s = (sequential_apply(cell_s, x) * w).sum()
    loss_n = (newton_apply(cell_n, x_n, NewtonConfig()) * w).sum()
    loss_s.backward()
    loss_n.backward()
    assert x.grad is not None and x_n.grad is not None
    torch.testing.assert_close(x_n.grad, x.grad, atol=1e-5, rtol=1e-5)
    for p_s, p_n in zip(cell_s.parameters(), cell_n.parameters(), strict=True):
        assert p_s.grad is not None and p_n.grad is not None
        torch.testing.assert_close(p_n.grad, p_s.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("t", [1, 7, 16])
@torch.no_grad()
def test_pararwkv7_odd_and_power_t(t: int) -> None:
    torch.manual_seed(7 + t)
    cell = ParaRWKV7(d_in=5, n_heads=1, d_head=4).eval()
    x = 0.2 * torch.randn(2, t, 5)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig())
    torch.testing.assert_close(par, seq, atol=1e-5, rtol=1e-5)
