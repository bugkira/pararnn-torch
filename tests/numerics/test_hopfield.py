"""ParaHopfield: sequential↔Newton dense-scan numerics."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaHopfield
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply
from pararnn.solvers.jacobian import jacobian_autograd
from pararnn.solvers.vjp import uses_packed_vjp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# d_h=8: dense recipe for v0.15 (cap 32). max_iters=6: MixTanh-class dense
# Newton in test_generic; Hopfield softmax is the same dense-J band. Fallback:
# measure residual vs K on a fixed batch; raise K before raising d_h.
_NEWTON = NewtonConfig(max_iters=6, scan_backend="eager", jac_structure="dense")


@torch.no_grad()
def test_parahopfield_newton_matches_sequential() -> None:
    torch.manual_seed(21)
    cell = ParaHopfield(d_in=6, d_h=8).to(device)
    x = torch.randn(3, 24, 6, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, _NEWTON)
    err = (par - seq).abs().amax()
    assert err < 1e-4, float(err)


@torch.no_grad()
def test_parahopfield_newton_odd_t() -> None:
    torch.manual_seed(22)
    cell = ParaHopfield(d_in=4, d_h=8).to(device)
    x = torch.randn(2, 17, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, _NEWTON)
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_parahopfield_newton_b1_t1() -> None:
    torch.manual_seed(23)
    cell = ParaHopfield(d_in=5, d_h=8).to(device)
    x = torch.randn(1, 1, 5, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, _NEWTON)
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_parahopfield_analytic_jac_matches_autograd() -> None:
    torch.manual_seed(24)
    cell = ParaHopfield(d_in=5, d_h=8).to(device)
    h = torch.randn(2, 11, 8, device=device)
    x = torch.randn(2, 11, 5, device=device)
    pred_a, jac_a = cell.step_with_jacobian(h, x)
    pred_g, jac_g = jacobian_autograd(cell, h, x, structure="dense")
    torch.testing.assert_close(pred_g, pred_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jac_g, jac_a, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_parahopfield_autograd_newton_matches_sequential() -> None:
    torch.manual_seed(25)
    cell = ParaHopfield(d_in=4, d_h=8).to(device)
    x = torch.randn(2, 16, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=6,
            scan_backend="eager",
            jacobian="autograd",
            jac_structure="dense",
        ),
    )
    assert (par - seq).abs().amax() < 1e-4


def test_parahopfield_bwd_matches_sequential_bptt() -> None:
    torch.manual_seed(26)
    d_in, d_h, t = 4, 8, 12
    cell_s = ParaHopfield(d_in, d_h).to(device)
    cell_n = ParaHopfield(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x = torch.randn(2, t, d_in, device=device, requires_grad=True)
    x_n = x.detach().clone().requires_grad_(True)
    w = torch.randn(2, t, d_h, device=device)
    loss_s = (sequential_apply(cell_s, x) * w).sum()
    loss_n = (newton_apply(cell_n, x_n, _NEWTON) * w).sum()
    loss_s.backward()
    loss_n.backward()
    assert x.grad is not None and x_n.grad is not None
    torch.testing.assert_close(x_n.grad, x.grad, atol=2e-4, rtol=1e-4)
    for (n_s, p_s), (n_n, p_n) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert n_s == n_n
        assert p_s.grad is not None and p_n.grad is not None
        torch.testing.assert_close(p_n.grad, p_s.grad, atol=2e-4, rtol=1e-4)


def test_parahopfield_uses_autograd_vjp() -> None:
    cell = ParaHopfield(d_in=4, d_h=8)
    assert uses_packed_vjp(cell) is False


def test_parahopfield_warns_large_dh() -> None:
    with pytest.warns(UserWarning, match="d_h=48"):
        ParaHopfield(d_in=4, d_h=48)


@pytest.mark.cuda
@torch.no_grad()
def test_parahopfield_triton_scan_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(27)
    cell = ParaHopfield(d_in=5, d_h=8).to(cuda_device)
    x = torch.randn(2, 32, 5, device=cuda_device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=6, scan_backend="triton", jac_structure="dense"),
    )
    assert (par - seq).abs().amax() < 2e-4
