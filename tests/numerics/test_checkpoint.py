"""Activation checkpoint and ``state_dict`` round-trip for ``ParaRNN``."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from pararnn import NewtonConfig, ParaGRU, ParaRNN

# Eager scan: fused Newton mutates the work buffer in-place; non-reentrant
# checkpoint is validated on the pure eager path first.
_ATOL = 1e-5
_RTOL = 1e-5


def _eager_cfg() -> NewtonConfig:
    return NewtonConfig(max_iters=3, scan_backend="eager")


def test_checkpoint_non_reentrant_matches_eager_grads() -> None:
    torch.manual_seed(0)
    model = ParaRNN(ParaGRU(8, 16), config=_eager_cfg())
    model.train()
    x = torch.randn(2, 16, 8, requires_grad=True)

    y_ref = model(x)
    w = torch.randn_like(y_ref)
    (y_ref * w).sum().backward()
    grads_ref = [p.grad.detach().clone() for p in model.parameters()]
    x_grad_ref = x.grad.detach().clone()
    assert x.grad is not None
    model.zero_grad(set_to_none=True)
    x.grad = None

    y_ck = checkpoint(model, x, use_reentrant=False)
    (y_ck * w).sum().backward()
    torch.testing.assert_close(y_ck, y_ref, atol=_ATOL, rtol=_RTOL)
    assert x.grad is not None
    torch.testing.assert_close(x.grad, x_grad_ref, atol=_ATOL, rtol=_RTOL)
    for p, g_ref in zip(model.parameters(), grads_ref, strict=True):
        assert p.grad is not None
        torch.testing.assert_close(p.grad, g_ref, atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_state_dict_round_trip_forward() -> None:
    torch.manual_seed(1)
    src = ParaRNN(ParaGRU(8, 16), config=_eager_cfg())
    dst = ParaRNN(ParaGRU(8, 16), config=_eager_cfg())
    dst.load_state_dict(src.state_dict())
    x = torch.randn(2, 16, 8)
    src.train()
    dst.train()
    torch.testing.assert_close(dst(x), src(x), atol=_ATOL, rtol=_RTOL)
