"""Chunked eq. 2.6 adjoint matches one-window Newton and runs past the scan cap."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pararnn.cells import ParaGRU
from pararnn.solvers import NewtonConfig, newton_apply

# Two-level tile scan: BLOCK_T=128 × CHUNK_PAD=64 = 8192.
# Third-level superchunks continue past that; this file tests windowed VJP.
_SCAN_CAP = 8192


def _assert_param_grads_close(a: nn.Module, b: nn.Module, *, atol: float) -> None:
    for (n, p_a), (_, p_b) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert p_a.grad is not None, n
        assert p_b.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=atol, rtol=1e-4)


def test_chunk_len_equals_t_matches_unchunked_grads() -> None:
    """One window is the global IFT adjoint."""
    torch.manual_seed(21)
    d_in, d_h, t = 4, 6, 12
    x = torch.randn(2, t, d_in)
    w = torch.randn(2, t, d_h)
    cell_a = ParaGRU(d_in, d_h)
    cell_b = ParaGRU(d_in, d_h)
    cell_b.load_state_dict(cell_a.state_dict())
    x_a = x.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    base = NewtonConfig(max_iters=3, scan_backend="eager", residual_atol=None, residual_fail=None)
    (newton_apply(cell_a, x_a, base) * w).sum().backward()
    chunked = NewtonConfig(
        max_iters=3,
        scan_backend="eager",
        residual_atol=None,
        residual_fail=None,
        chunk_len=t,
    )
    (newton_apply(cell_b, x_b, chunked) * w).sum().backward()
    _assert_param_grads_close(cell_a, cell_b, atol=1e-6)
    torch.testing.assert_close(x_a.grad, x_b.grad, atol=1e-6, rtol=1e-5)


def test_chunked_adjoint_gradcheck_input() -> None:
    torch.manual_seed(22)
    cell = ParaGRU(input_size=2, hidden_size=3, dtype=torch.float64, max_recurrent_norm=None)
    cfg = NewtonConfig(
        max_iters=3,
        scan_backend="eager",
        residual_atol=None,
        residual_fail=None,
        chunk_len=2,
    )
    x = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)

    def fn(xx: torch.Tensor) -> torch.Tensor:
        return newton_apply(cell, xx, cfg)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_chunked_adjoint_gradcheck_h0() -> None:
    torch.manual_seed(23)
    cell = ParaGRU(input_size=2, hidden_size=3, dtype=torch.float64, max_recurrent_norm=None)
    cfg = NewtonConfig(
        max_iters=3,
        scan_backend="eager",
        residual_atol=None,
        residual_fail=None,
        chunk_len=2,
    )
    x = torch.randn(2, 4, 2, dtype=torch.float64)
    h0 = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    def fn(hh: torch.Tensor) -> torch.Tensor:
        return newton_apply(cell, x, cfg, h0=hh)

    assert torch.autograd.gradcheck(fn, (h0,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.cuda
def test_chunked_backward_past_scan_cap(cuda_device: torch.device) -> None:
    """T=2× two-level tile pad; windowed eq. 2.6 adjoint still runs."""
    torch.manual_seed(24)
    t = 2 * _SCAN_CAP
    cell = ParaGRU(6, 32).to(cuda_device)
    x = torch.randn(1, t, 6, device=cuda_device, requires_grad=True)
    cfg = NewtonConfig(max_iters=3, scan_backend="auto", chunk_len=_SCAN_CAP)
    y = newton_apply(cell, x, cfg)
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in cell.parameters())
