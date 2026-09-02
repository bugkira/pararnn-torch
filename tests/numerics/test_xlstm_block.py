"""Pre-norm residual sLSTM block: Newton vs sequential, residual add."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, xLSTMBlock
from pararnn.layout import SLSTM_HIDDEN
from pararnn.solvers import newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_xlstm_block_shape_and_residual():
    torch.manual_seed(81)
    d, t = 8, 12
    block = xLSTMBlock(d, solver="sequential", device=device)
    x = torch.randn(2, t, d, device=device)
    y = block(x)
    assert y.shape == x.shape
    z = block.norm(x)
    h = sequential_apply(block.cell, z)[:, :, SLSTM_HIDDEN, :]
    torch.testing.assert_close(y, x + h, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_xlstm_block_newton_matches_eager_eval():
    torch.manual_seed(82)
    d, t = 8, 12
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    newton = xLSTMBlock(d, solver="auto", config=cfg, device=device)
    eager = xLSTMBlock(d, solver="sequential", device=device)
    eager.load_state_dict(newton.state_dict())
    x = 0.3 * torch.randn(2, t, d, device=device)
    newton.eval()
    torch.testing.assert_close(newton(x), eager(x), atol=2e-4, rtol=1e-4)


def test_xlstm_block_rejects_unknown_solver():
    with pytest.raises(ValueError, match="solver"):
        xLSTMBlock(8, solver="flashrnn")


@torch.no_grad()
def test_xlstm_block_head_mix_shape():
    torch.manual_seed(83)
    d, t = 8, 10
    cfg = NewtonConfig(max_iters=4, scan_backend="eager")
    block = xLSTMBlock(d, solver="auto", mix="head", n_heads=2, config=cfg, device=device)
    x = 0.3 * torch.randn(2, t, d, device=device)
    y = block(x)
    assert y.shape == x.shape
    assert block.cell.mix == "head"
    assert block.cell.n_heads == 2


@torch.no_grad()
def test_xlstm_block_head_newton_matches_eval():
    torch.manual_seed(84)
    d, t = 8, 8
    cfg = NewtonConfig(max_iters=4, scan_backend="eager")
    block = xLSTMBlock(d, solver="auto", mix="head", n_heads=2, config=cfg, device=device)
    x = 0.3 * torch.randn(2, t, d, device=device)
    block.train()
    y_train = block(x)
    block.eval()
    y_eval = block(x)
    torch.testing.assert_close(y_train, y_eval, atol=2e-4, rtol=1e-4)


@torch.no_grad()
def test_xlstm_block_solver_newton_in_eval():
    torch.manual_seed(85)
    d, t = 8, 10
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    block = xLSTMBlock(d, solver="newton", config=cfg, device=device)
    block.eval()
    x = 0.3 * torch.randn(2, t, d, device=device)
    z = block.norm(x)
    h = newton_apply(block.cell, z, cfg)[:, :, SLSTM_HIDDEN, :]
    torch.testing.assert_close(block(x), x + h, atol=2e-4, rtol=1e-4)
    assert block.rnn.solver == "newton"


def test_xlstm_block_reset_parameters():
    torch.manual_seed(86)
    block = xLSTMBlock(8, solver="sequential", device=device)
    assert block.cell.R is not None
    w0 = block.cell.W_x.weight.detach().clone()
    r0 = block.cell.R.detach().clone()
    block.reset_parameters()
    assert not torch.equal(w0, block.cell.W_x.weight)
    assert not torch.equal(r0, block.cell.R)
