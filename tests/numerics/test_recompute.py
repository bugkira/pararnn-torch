"""Level-2 Newton ``recompute=True``: rematerialize H* in backward."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.solvers import NewtonConfig, newton_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ATOL = 2e-4
_RTOL = 2e-4


def _clone_cell(src: torch.nn.Module) -> torch.nn.Module:
    if isinstance(src, ParaSLSTM):
        dst = ParaSLSTM(d_in=src.d_in, d_h=src.d_h, mix=src.mix).to(device)
    else:
        dst = type(src)(d_in=src.d_in, d_h=src.d_h).to(device)
    dst.load_state_dict(src.state_dict())
    return dst


def _grad_bundle(cell: torch.nn.Module, x: torch.Tensor, cfg: NewtonConfig, w: torch.Tensor):
    cell.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(True)
    y = newton_apply(cell, x, cfg)
    (y * w).sum().backward()
    assert x.grad is not None
    return y.detach(), x.grad.detach(), [p.grad.detach().clone() for p in cell.parameters()]


@pytest.mark.parametrize(
    ("cell_ctor", "kwargs", "cfg_extra"),
    [
        (ParaGRU, {"d_in": 5, "d_h": 7}, {}),
        (ParaLSTM, {"d_in": 5, "d_h": 6}, {}),
        (ParaSLSTM, {"d_in": 4, "d_h": 4, "mix": "diag"}, {"picard_iters": 1}),
    ],
    ids=["gru", "lstm", "slstm"],
)
def test_recompute_grads_match_level1(cell_ctor, kwargs, cfg_extra) -> None:
    torch.manual_seed(11)
    cell_a = cell_ctor(**kwargs).to(device)
    cell_b = _clone_cell(cell_a)
    t = 24 if not isinstance(cell_a, ParaSLSTM) else 20
    x = torch.randn(2, t, kwargs["d_in"], device=device)
    y0 = newton_apply(
        cell_a,
        x,
        NewtonConfig(max_iters=3, scan_backend="eager", residual_fail=None, **cfg_extra),
    )
    w = torch.randn_like(y0)
    base = {"max_iters": 3, "scan_backend": "eager", "residual_fail": None, **cfg_extra}
    y_l1, gx_l1, gp_l1 = _grad_bundle(cell_a, x, NewtonConfig(**base, recompute=False), w)
    y_l2, gx_l2, gp_l2 = _grad_bundle(cell_b, x, NewtonConfig(**base, recompute=True), w)
    torch.testing.assert_close(y_l2, y_l1, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(gx_l2, gx_l1, atol=_ATOL, rtol=_RTOL)
    for a, b in zip(gp_l1, gp_l2, strict=True):
        torch.testing.assert_close(b, a, atol=_ATOL, rtol=_RTOL)


@pytest.mark.cuda
def test_recompute_fused_gru_grads_match_level1(cuda_device: torch.device) -> None:
    torch.manual_seed(12)
    cell_a = ParaGRU(d_in=8, d_h=16).to(cuda_device)
    cell_b = _clone_cell(cell_a).to(cuda_device)
    x = torch.randn(2, 48, 8, device=cuda_device)
    cfg0 = NewtonConfig(max_iters=3, scan_backend="fused", residual_fail=None)
    y0 = newton_apply(cell_a, x, cfg0)
    w = torch.randn_like(y0)
    base = {"max_iters": 3, "scan_backend": "fused", "residual_fail": None}
    y_l1, gx_l1, gp_l1 = _grad_bundle(cell_a, x, NewtonConfig(**base, recompute=False), w)
    y_l2, gx_l2, gp_l2 = _grad_bundle(cell_b, x, NewtonConfig(**base, recompute=True), w)
    torch.testing.assert_close(y_l2, y_l1, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(gx_l2, gx_l1, atol=_ATOL, rtol=_RTOL)
    for a, b in zip(gp_l1, gp_l2, strict=True):
        torch.testing.assert_close(b, a, atol=_ATOL, rtol=_RTOL)


def test_recompute_with_h0_grads_match() -> None:
    torch.manual_seed(13)
    cell_a = ParaGRU(d_in=4, d_h=5).to(device)
    cell_b = _clone_cell(cell_a)
    x = torch.randn(2, 16, 4, device=device)
    h0 = torch.randn(2, 5, device=device, requires_grad=True)
    cfg_l1 = NewtonConfig(max_iters=3, scan_backend="eager", residual_fail=None, recompute=False)
    cfg_l2 = NewtonConfig(max_iters=3, scan_backend="eager", residual_fail=None, recompute=True)
    w = torch.randn(2, 16, 5, device=device)

    def run(cell, cfg, h0_in):
        cell.zero_grad(set_to_none=True)
        xx = x.detach().requires_grad_(True)
        hh = h0_in.detach().requires_grad_(True)
        y = newton_apply(cell, xx, cfg, h0=hh)
        (y * w).sum().backward()
        assert xx.grad is not None
        assert hh.grad is not None
        return (
            y.detach(),
            xx.grad.detach(),
            hh.grad.detach(),
            [p.grad.detach().clone() for p in cell.parameters()],
        )

    y1, gx1, gh1, gp1 = run(cell_a, cfg_l1, h0)
    y2, gx2, gh2, gp2 = run(cell_b, cfg_l2, h0)
    torch.testing.assert_close(y2, y1, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(gx2, gx1, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(gh2, gh1, atol=_ATOL, rtol=_RTOL)
    for a, b in zip(gp1, gp2, strict=True):
        torch.testing.assert_close(b, a, atol=_ATOL, rtol=_RTOL)
