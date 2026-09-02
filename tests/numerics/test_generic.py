"""Generic cell via Autograd Jacobian; packed eq. 2.6 VJP; dense scan."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pararnn import NewtonConfig
from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.solvers import newton_apply, sequential_apply
from pararnn.solvers.jacobian import jacobian_autograd
from pararnn.solvers.scan import scan_dense
from pararnn.solvers.vjp import cell_vjp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiagTanh(nn.Module):
    """Channelwise tanh RNN — exact Newton with a ones-JVP."""

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.state_slots = 1
        self.a = nn.Parameter(0.4 * torch.ones(d_h))
        self.W_x = nn.Linear(d_in, d_h)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def step(self, h: Tensor, x: Tensor) -> Tensor:
        return torch.tanh(self.a * h + self.W_x(x))


class MixTanh(nn.Module):
    """Full recurrent mix — needs ``jac_structure='dense'``."""

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.state_slots = 1
        self.W_h = nn.Linear(d_h, d_h, bias=False)
        self.W_x = nn.Linear(d_in, d_h)
        nn.init.orthogonal_(self.W_h.weight, gain=0.25)
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)

    def step(self, h: Tensor, x: Tensor) -> Tensor:
        return torch.tanh(self.W_h(h) + self.W_x(x))


def test_scan_dense_matches_forward_substitution():
    torch.manual_seed(50)
    b, t, d = 2, 7, 3
    jac = torch.randn(b, t, d, d, device=device) * 0.15
    residual = torch.randn(b, t, d, device=device)
    got = scan_dense(jac, residual)
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        ref[:, s] = (jac[:, s] @ ref[:, s - 1].unsqueeze(-1)).squeeze(-1) + residual[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paragru_autograd_jac_matches_analytic():
    torch.manual_seed(51)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    h = torch.randn(2, 11, 8, device=device)
    x = torch.randn(2, 11, 6, device=device)
    pred_a, jac_a = cell.step_with_jacobian(h, x)
    pred_g, jac_g = jacobian_autograd(cell, h, x, structure="diag")
    torch.testing.assert_close(pred_g, pred_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jac_g, jac_a, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paralstm_autograd_jac_matches_analytic():
    torch.manual_seed(52)
    cell = ParaLSTM(d_in=5, d_h=6).to(device)
    state = torch.randn(2, 9, 2, 6, device=device)
    x = torch.randn(2, 9, 5, device=device)
    pred_a, jac_a = cell.step_with_jacobian(state, x)
    pred_g, jac_g = jacobian_autograd(cell, state, x, structure="block2")
    torch.testing.assert_close(pred_g, pred_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jac_g, jac_a, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_custom_diag_newton_matches_sequential():
    torch.manual_seed(53)
    cell = DiagTanh(d_in=4, d_h=7).to(device)
    x = torch.randn(3, 20, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, jacobian="autograd"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, float(err)


@torch.no_grad()
def test_custom_dense_newton_matches_sequential():
    torch.manual_seed(54)
    cell = MixTanh(d_in=3, d_h=4).to(device)
    x = torch.randn(2, 12, 3, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=6, jacobian="autograd", jac_structure="dense"),
    )
    err = (par - seq).abs().amax()
    assert err < 1e-4, float(err)


@torch.no_grad()
def test_paragru_newton_autograd_matches_sequential():
    torch.manual_seed(55)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    x = torch.randn(2, 16, 6, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, jacobian="autograd"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, float(err)


def test_custom_diag_bwd_matches_sequential_bptt():
    torch.manual_seed(56)
    d_in, d_h, t = 4, 6, 10
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = DiagTanh(d_in, d_h).to(device)
    cell_n = DiagTanh(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    (sequential_apply(cell_s, x_s) * w).sum().backward()
    (newton_apply(cell_n, x_n, NewtonConfig(max_iters=3, jacobian="autograd")) * w).sum().backward()
    for (n, p_a), (_, p_b) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert p_a.grad is not None, n
        assert p_b.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=2e-4, rtol=1e-4)


def test_paragru_packed_vjp_matches_autograd_vjp():
    torch.manual_seed(57)
    cell = ParaGRU(d_in=5, d_h=7).to(device)
    h = torch.randn(2, 9, 7, device=device)
    x = torch.randn(2, 9, 5, device=device)
    mu = torch.randn(2, 9, 7, device=device)
    gx_p, gp_p = cell_vjp(cell, h, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h, x, mu, packed=False)
    torch.testing.assert_close(gx_p, gx_a, atol=2e-5, rtol=2e-5)
    for a, b in zip(gp_p, gp_a, strict=True):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_paralstm_packed_vjp_matches_autograd_vjp():
    torch.manual_seed(58)
    cell = ParaLSTM(d_in=5, d_h=6).to(device)
    h = torch.randn(2, 8, 2, 6, device=device)
    x = torch.randn(2, 8, 5, device=device)
    mu = torch.randn(2, 8, 2, 6, device=device)
    gx_p, gp_p = cell_vjp(cell, h, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h, x, mu, packed=False)
    torch.testing.assert_close(gx_p, gx_a, atol=5e-5, rtol=5e-5)
    for a, b in zip(gp_p, gp_a, strict=True):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)


def test_paraslstm_diag_packed_vjp_matches_autograd_vjp():
    torch.manual_seed(59)
    cell = ParaSLSTM(d_in=5, d_h=7, mix="diag").to(device)
    with torch.no_grad():
        cell.R[0, 0] = 0.8
        cell.R[1, 3] = -0.9
    h = torch.randn(2, 9, 4, 7, device=device)
    x = torch.randn(2, 9, 5, device=device)
    mu = torch.randn(2, 9, 4, 7, device=device)
    gx_p, gp_p = cell_vjp(cell, h, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h, x, mu, packed=False)
    torch.testing.assert_close(gx_p, gx_a, atol=5e-5, rtol=5e-5)
    for a, b in zip(gp_p, gp_a, strict=True):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)


def test_paraslstm_head_packed_vjp_falls_back_to_autograd():
    torch.manual_seed(60)
    cell = ParaSLSTM(d_in=5, d_h=6, mix="head", n_heads=2).to(device)
    h = torch.randn(2, 5, 4, 6, device=device)
    x = torch.randn(2, 5, 5, device=device)
    mu = torch.randn(2, 5, 4, 6, device=device)
    gx_p, gp_p = cell_vjp(cell, h, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h, x, mu, packed=False)
    torch.testing.assert_close(gx_p, gx_a, atol=5e-5, rtol=5e-5)
    for a, b in zip(gp_p, gp_a, strict=True):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)
