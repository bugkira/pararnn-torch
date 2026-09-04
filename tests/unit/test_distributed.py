"""CPU helpers for data-parallel wrapping (no process group)."""

from __future__ import annotations

import math

import torch

from pararnn import NewtonConfig, ParaGRU, ParaRNN
from pararnn.distributed import last_newton_residuals, unwrap_distributed, warmup_scan_kernels


def _cfg() -> NewtonConfig:
    return NewtonConfig(
        max_iters=3,
        scan_backend="eager",
        residual_atol=None,
        residual_fail=None,
    )


def test_unwrap_identity() -> None:
    model = ParaRNN(ParaGRU(4, 4), config=_cfg())
    assert unwrap_distributed(model) is model


def test_warmup_and_residuals_cpu() -> None:
    torch.manual_seed(0)
    model = ParaRNN(ParaGRU(8, 8), config=_cfg())
    x = torch.randn(2, 12, 8)
    y = warmup_scan_kernels(model, x)
    assert y.shape == (2, 12, 8)
    model.train()
    model(x).sum().backward()
    res = last_newton_residuals(model)
    assert len(res) == 1
    assert math.isfinite(res[0])
