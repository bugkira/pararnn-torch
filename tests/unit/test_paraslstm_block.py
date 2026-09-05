"""``ParaSLSTMBlock``: pre-norm + SwiGLU residual around fused-ready sLSTM."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, ParaSLSTMBlock


def test_paraslstm_block_shape_and_grad() -> None:
    torch.manual_seed(0)
    block = ParaSLSTMBlock(
        32,
        mlp_ratio=2.0,
        config=NewtonConfig(max_iters=3, scan_backend="eager", residual_fail=None),
        solver="newton",
    )
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in block.parameters())


def test_paraslstm_block_stack() -> None:
    torch.manual_seed(1)
    cfg = NewtonConfig(max_iters=2, scan_backend="eager", residual_fail=None)
    stack = torch.nn.Sequential(
        ParaSLSTMBlock(24, mlp_ratio=2.0, config=cfg, solver="newton"),
        ParaSLSTMBlock(24, mlp_ratio=2.0, config=cfg, solver="newton"),
    )
    x = torch.randn(2, 8, 24)
    y = stack(x)
    assert y.shape == (2, 8, 24)


def test_paraslstm_block_rejects_bad_shape() -> None:
    block = ParaSLSTMBlock(
        16,
        config=NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None),
        solver="sequential",
    )
    with pytest.raises(ValueError, match="expected x"):
        block(torch.randn(2, 8, 15))
