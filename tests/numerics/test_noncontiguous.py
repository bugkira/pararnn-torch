"""Strided ``(B, T, d_in)`` views: fused/eager Newton vs sequential.

Wrappers copy before Triton; ``W_x`` and packed VJP still see the caller's
strides. Layouts: time skip ``[:, ::2]``, feature skip ``[:, :, ::2]``, and a
contiguous ``(B, D, T)`` permuted back to ``(B, T, D)`` (strided last dim).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.layout import SLSTM_SLOTS
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply

_LAYOUTS = ("time_skip", "feat_skip", "permute_roundtrip")
_KINDS = ("gru", "lstm", "slstm")


def _cell(kind: str, d_in: int, d_h: int, device: torch.device) -> nn.Module:
    kw = {"d_in": d_in, "d_h": d_h, "device": device}
    if kind == "gru":
        return ParaGRU(**kw)
    if kind == "lstm":
        return ParaLSTM(**kw)
    return ParaSLSTM(mix="diag", **kw)


def _newton_cfg(kind: str, *, fused: bool) -> NewtonConfig:
    backend = "fused" if fused else "eager"
    if kind == "slstm":
        # K=5, residual_atol off: same band as test_slstm fused vs sequential on d_h=4.
        return NewtonConfig(max_iters=5, scan_backend=backend, residual_atol=None)
    # K=3: Danieli et al. 2025 §2.1 / App. A (ParaGRU / ParaLSTM).
    return NewtonConfig(max_iters=3, scan_backend=backend)


def _strided_pair(
    layout: str,
    batch: int,
    time: int,
    dim: int,
    device: torch.device,
    *,
    scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    if layout == "time_skip":
        base_s = scale * torch.randn(batch, 2 * time, dim, device=device)
        base_n = base_s.clone()
        xs = base_s[:, ::2, :].detach().requires_grad_(True)
        xn = base_n[:, ::2, :].detach().requires_grad_(True)
    elif layout == "feat_skip":
        base_s = scale * torch.randn(batch, time, 2 * dim, device=device)
        base_n = base_s.clone()
        xs = base_s[:, :, ::2].detach().requires_grad_(True)
        xn = base_n[:, :, ::2].detach().requires_grad_(True)
    elif layout == "permute_roundtrip":
        # Contiguous (B, D, T) permuted back to (B, T, D): last-dim stride is T.
        full_s = scale * torch.randn(batch, time, dim, device=device)
        full_n = full_s.clone()
        xs = full_s.permute(0, 2, 1).contiguous().permute(0, 2, 1).detach().requires_grad_(True)
        xn = full_n.permute(0, 2, 1).contiguous().permute(0, 2, 1).detach().requires_grad_(True)
    else:
        raise ValueError(layout)
    assert xs.shape == (batch, time, dim)
    assert not xs.is_contiguous()
    assert not xn.is_contiguous()
    return xs, xn


def _loss_weight(kind: str, batch: int, time: int, d_h: int, device: torch.device) -> Tensor:
    if kind == "gru":
        return torch.randn(batch, time, d_h, device=device)
    slots = 2 if kind == "lstm" else SLSTM_SLOTS
    return torch.randn(batch, time, slots, d_h, device=device)


def _fwd_atol(kind: str) -> float:
    if kind == "slstm":
        return 2e-3
    return 1e-4


def _bwd_atol(kind: str) -> float:
    if kind == "gru":
        return 2e-4
    return 5e-4


def _assert_param_grads_close(cell_a: nn.Module, cell_b: nn.Module, atol: float) -> None:
    for (n, p_a), (_, p_b) in zip(
        cell_a.named_parameters(), cell_b.named_parameters(), strict=True
    ):
        assert p_a.grad is not None, n
        assert p_b.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=atol, rtol=1e-4)


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_paragru_eager_noncontiguous_matches_sequential(layout: str) -> None:
    """CPU packed VJP + eager Newton on a strided ``x`` (CI, no GPU)."""
    torch.manual_seed(80)
    device = torch.device("cpu")
    d_in, d_h, batch, time = 6, 8, 2, 12
    xs, xn = _strided_pair(layout, batch, time, d_in, device)
    cell_s = _cell("gru", d_in, d_h, device)
    cell_n = _cell("gru", d_in, d_h, device)
    cell_n.load_state_dict(cell_s.state_dict())
    cfg = _newton_cfg("gru", fused=False)
    w = _loss_weight("gru", batch, time, d_h, device)
    seq = sequential_apply(cell_s, xs)
    par = newton_apply(cell_n, xn, cfg)
    torch.testing.assert_close(par, seq, atol=_fwd_atol("gru"), rtol=1e-4)
    (seq * w).sum().backward()
    (par * w).sum().backward()
    _assert_param_grads_close(cell_s, cell_n, atol=_bwd_atol("gru"))
    torch.testing.assert_close(xs.grad, xn.grad, atol=_bwd_atol("gru"), rtol=1e-4)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_fused_noncontiguous_matches_sequential(
    kind: str, layout: str, cuda_device: torch.device
) -> None:
    torch.manual_seed(81)
    d_in, d_h = (4, 4) if kind == "slstm" else (8, 8)
    batch, time = 2, 16 if kind != "slstm" else 8
    scale = 0.3 if kind == "slstm" else 1.0
    xs, xn = _strided_pair(layout, batch, time, d_in, cuda_device, scale=scale)
    cell_s = _cell(kind, d_in, d_h, cuda_device)
    cell_n = _cell(kind, d_in, d_h, cuda_device)
    cell_n.load_state_dict(cell_s.state_dict())
    cfg = _newton_cfg(kind, fused=True)
    w = _loss_weight(kind, batch, time, d_h, cuda_device)
    seq = sequential_apply(cell_s, xs)
    par = newton_apply(cell_n, xn, cfg)
    err = (par - seq).abs().amax()
    assert err < _fwd_atol(kind), float(err)
    (seq * w).sum().backward()
    (par * w).sum().backward()
    _assert_param_grads_close(cell_s, cell_n, atol=_bwd_atol(kind))
    torch.testing.assert_close(xs.grad, xn.grad, atol=_bwd_atol(kind), rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_fused_paragru_strided_h0_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(82)
    d_in, d_h, batch, time = 8, 8, 2, 16
    cell = ParaGRU(d_in=d_in, d_h=d_h, device=cuda_device)
    x = torch.randn(batch, time, d_in, device=cuda_device)
    h0_wide = torch.randn(batch, 2 * d_h, device=cuda_device)
    h0 = h0_wide[:, ::2]
    assert not h0.is_contiguous()
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(cell, x, _newton_cfg("gru", fused=True), h0=h0)
    err = (par - seq).abs().amax()
    assert err < 1e-4, float(err)
