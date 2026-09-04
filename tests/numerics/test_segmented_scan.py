"""Segmented scan on packed ``cu_seqlens`` vs concat of independent scans."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.layers import ParaRNN
from pararnn.layout import prepend_state_ragged, segment_start_flags, validate_cu_seqlens
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply
from pararnn.solvers.scan import (
    reverse_scan_block2,
    reverse_scan_block4,
    reverse_scan_diag,
    scan_block2,
    scan_block4,
    scan_dense,
    scan_diag,
)

_LENS = (50, 150)  # starts at 0 and 50; T=200 sits inside a 128-tile


def _cu(lens: tuple[int, ...]) -> torch.Tensor:
    cs = [0]
    for length in lens:
        cs.append(cs[-1] + length)
    return torch.tensor(cs, dtype=torch.long)


def _inclusive_diag(jac: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    out = residual.clone()
    for t in range(1, residual.shape[1]):
        out[:, t] = jac[:, t] * out[:, t - 1] + residual[:, t]
    return out


def test_validate_cu_seqlens_and_flags():
    cs = _cu(_LENS)
    cs = validate_cu_seqlens(cs, 200)
    flags = segment_start_flags(cs, 200, batch=1)
    assert flags.shape == (1, 200)
    assert bool(flags[0, 0])
    assert bool(flags[0, 50])
    assert int(flags.sum()) == 2


def test_prepend_state_ragged_writes_h0_at_starts():
    cs = _cu((3, 2))
    states = torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(1, 5, 2).clone()
    h0 = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    prev = prepend_state_ragged(states, h0, cs)
    torch.testing.assert_close(prev[0, 0], h0[0])
    torch.testing.assert_close(prev[0, 3], h0[1])
    torch.testing.assert_close(prev[0, 1], states[0, 0])
    torch.testing.assert_close(prev[0, 4], states[0, 3])


def test_scan_diag_single_span_matches_unsegmented():
    torch.manual_seed(0)
    t, d = 17, 4
    jac = torch.randn(1, t, d) * 0.3
    residual = torch.randn(1, t, d)
    cs = _cu((t,))
    got = scan_diag(jac, residual, cu_seqlens=cs)
    ref = scan_diag(jac, residual)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_scan_diag_packed_matches_concat_and_substitution():
    torch.manual_seed(1)
    d = 5
    cs = _cu(_LENS)
    jac = torch.randn(1, 200, d) * 0.3
    residual = torch.randn(1, 200, d)
    got = scan_diag(jac, residual, cu_seqlens=cs)
    parts = []
    for s in range(len(_LENS)):
        t0, t1 = int(cs[s]), int(cs[s + 1])
        piece = scan_diag(jac[:, t0:t1], residual[:, t0:t1])
        parts.append(piece)
        torch.testing.assert_close(
            got[:, t0:t1],
            _inclusive_diag(jac[:, t0:t1], residual[:, t0:t1]),
            atol=1e-5,
            rtol=1e-5,
        )
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_scan_block2_packed_matches_concat():
    torch.manual_seed(2)
    cs = _cu(_LENS)
    jac = torch.randn(1, 200, 2, 2, 3) * 0.2
    residual = torch.randn(1, 200, 2, 3)
    got = scan_block2(jac, residual, cu_seqlens=cs)
    parts = [
        scan_block2(jac[:, int(cs[s]) : int(cs[s + 1])], residual[:, int(cs[s]) : int(cs[s + 1])])
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_scan_block4_packed_matches_concat():
    torch.manual_seed(3)
    cs = _cu((7, 9))
    t = 16
    jac = torch.randn(1, t, 4, 4, 2) * 0.15
    residual = torch.randn(1, t, 4, 2)
    got = scan_block4(jac, residual, cu_seqlens=cs)
    parts = [
        scan_block4(jac[:, int(cs[s]) : int(cs[s + 1])], residual[:, int(cs[s]) : int(cs[s + 1])])
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_scan_dense_packed_matches_concat():
    torch.manual_seed(4)
    cs = _cu((5, 6))
    t, d = 11, 3
    jac = torch.randn(1, t, d, d) * 0.15
    residual = torch.randn(1, t, d)
    got = scan_dense(jac, residual, cu_seqlens=cs)
    parts = [
        scan_dense(jac[:, int(cs[s]) : int(cs[s + 1])], residual[:, int(cs[s]) : int(cs[s + 1])])
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_reverse_scan_diag_packed_matches_concat():
    torch.manual_seed(5)
    d = 4
    cs = _cu(_LENS)
    jac = torch.randn(1, 200, d) * 0.3
    partial = torch.randn(1, 200, d)
    got = reverse_scan_diag(jac, partial, cu_seqlens=cs)
    parts = [
        reverse_scan_diag(
            jac[:, int(cs[s]) : int(cs[s + 1])],
            partial[:, int(cs[s]) : int(cs[s + 1])],
        )
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_reverse_scan_block2_packed_matches_concat():
    torch.manual_seed(6)
    cs = _cu((8, 5))
    jac = torch.randn(1, 13, 2, 2, 3) * 0.2
    partial = torch.randn(1, 13, 2, 3)
    got = reverse_scan_block2(jac, partial, cu_seqlens=cs)
    parts = [
        reverse_scan_block2(
            jac[:, int(cs[s]) : int(cs[s + 1])], partial[:, int(cs[s]) : int(cs[s + 1])]
        )
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


def test_reverse_scan_block4_packed_matches_concat():
    torch.manual_seed(7)
    cs = _cu((4, 5))
    jac = torch.randn(1, 9, 4, 4, 2) * 0.15
    partial = torch.randn(1, 9, 4, 2)
    got = reverse_scan_block4(jac, partial, cu_seqlens=cs)
    parts = [
        reverse_scan_block4(
            jac[:, int(cs[s]) : int(cs[s + 1])], partial[:, int(cs[s]) : int(cs[s + 1])]
        )
        for s in range(2)
    ]
    torch.testing.assert_close(got, torch.cat(parts, dim=1), atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paragru_newton_packed_matches_sequential_and_concat():
    torch.manual_seed(8)
    cell = ParaGRU(d_in=6, d_h=8)
    lens = _LENS
    cs = _cu(lens)
    x1 = torch.randn(1, lens[0], 6)
    x2 = torch.randn(1, lens[1], 6)
    h01 = torch.randn(1, 8)
    h02 = torch.randn(1, 8)
    x = torch.cat((x1, x2), dim=1)
    h0 = torch.cat((h01, h02), dim=0)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    packed = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    seq = sequential_apply(cell, x, h0, cu_seqlens=cs)
    a = newton_apply(cell, x1, cfg, h0=h01)
    b = newton_apply(cell, x2, cfg, h0=h02)
    torch.testing.assert_close(packed, seq, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(packed, torch.cat((a, b), dim=1), atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_paralstm_newton_packed_matches_sequential():
    torch.manual_seed(9)
    cell = ParaLSTM(d_in=5, d_h=7)
    lens = (12, 19)
    cs = _cu(lens)
    x = torch.randn(1, sum(lens), 5)
    h0 = torch.randn(2, 2, 7)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    packed = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    seq = sequential_apply(cell, x, h0, cu_seqlens=cs)
    torch.testing.assert_close(packed, seq, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_paraslstm_newton_packed_matches_sequential():
    torch.manual_seed(10)
    cell = ParaSLSTM(d_in=4, d_h=6, mix="diag")
    lens = (11, 13)
    cs = _cu(lens)
    x = torch.randn(1, sum(lens), 4)
    h0 = torch.randn(2, 4, 6)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager", picard_iters=1)
    packed = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    seq = sequential_apply(cell, x, h0, cu_seqlens=cs)
    torch.testing.assert_close(packed, seq, atol=1e-4, rtol=1e-4)


def test_paragru_eq26_packed_matches_concat_grads():
    torch.manual_seed(11)
    cell = ParaGRU(d_in=5, d_h=6)
    lens = (9, 14)
    cs = _cu(lens)
    x1 = torch.randn(1, lens[0], 5, requires_grad=True)
    x2 = torch.randn(1, lens[1], 5, requires_grad=True)
    h01 = torch.randn(1, 6, requires_grad=True)
    h02 = torch.randn(1, 6, requires_grad=True)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    y1 = newton_apply(cell, x1, cfg, h0=h01)
    y2 = newton_apply(cell, x2, cfg, h0=h02)
    (y1.sum() + y2.sum()).backward()

    cell.zero_grad()
    x = torch.cat((x1.detach(), x2.detach()), dim=1).requires_grad_(True)
    h0 = torch.cat((h01.detach(), h02.detach()), dim=0).requires_grad_(True)
    y = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    y.sum().backward()
    torch.testing.assert_close(x.grad[:, : lens[0]], x1.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(x.grad[:, lens[0] :], x2.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(h0.grad[0:1], h01.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(h0.grad[1:2], h02.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_scan_diag_triton_packed_matches_eager(cuda_device: torch.device):
    torch.manual_seed(13)
    d = 8
    cs = _cu(_LENS)
    jac = (torch.randn(1, 200, d, device=cuda_device) * 0.3).contiguous()
    residual = torch.randn(1, 200, d, device=cuda_device).contiguous()
    eager = scan_diag(jac, residual, cu_seqlens=cs)
    tri = scan_diag(jac, residual, backend="triton", cu_seqlens=cs)
    torch.testing.assert_close(tri, eager, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_scan_diag_triton_packed_crosses_tile():
    """Flag at t=50 (tile 0) and t=178 (tile 1 of BLOCK_T=128)."""
    torch.manual_seed(14)
    device = torch.device("cuda")
    t, d = 256, 16
    cs = _cu((50, 128, 78))
    jac = (torch.randn(1, t, d, device=device) * 0.3).contiguous()
    residual = torch.randn(1, t, d, device=device).contiguous()
    eager = scan_diag(jac, residual, cu_seqlens=cs)
    tri = scan_diag(jac, residual, backend="triton", cu_seqlens=cs)
    torch.testing.assert_close(tri, eager, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_scan_block2_triton_packed_matches_eager(cuda_device: torch.device):
    torch.manual_seed(16)
    cs = _cu(_LENS).to(cuda_device)
    jac = (torch.randn(1, 200, 2, 2, 3, device=cuda_device) * 0.2).contiguous()
    residual = torch.randn(1, 200, 2, 3, device=cuda_device).contiguous()
    eager = scan_block2(jac, residual, cu_seqlens=cs)
    tri = scan_block2(jac, residual, backend="triton", cu_seqlens=cs)
    torch.testing.assert_close(tri, eager, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_newton_fused_packed_matches_sequential(cuda_device: torch.device):
    torch.manual_seed(17)
    cell = ParaGRU(d_in=6, d_h=8).to(cuda_device)
    lens = _LENS
    cs = _cu(lens).to(cuda_device)
    x1 = torch.randn(1, lens[0], 6, device=cuda_device)
    x2 = torch.randn(1, lens[1], 6, device=cuda_device)
    h01 = torch.randn(1, 8, device=cuda_device)
    h02 = torch.randn(1, 8, device=cuda_device)
    x = torch.cat((x1, x2), dim=1)
    h0 = torch.cat((h01, h02), dim=0)
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    packed = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    seq = sequential_apply(cell, x, h0, cu_seqlens=cs)
    torch.testing.assert_close(packed, seq, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_newton_triton_packed_matches_sequential(cuda_device: torch.device):
    torch.manual_seed(15)
    cell = ParaGRU(d_in=6, d_h=8).to(cuda_device)
    lens = _LENS
    cs = _cu(lens).to(cuda_device)
    x1 = torch.randn(1, lens[0], 6, device=cuda_device)
    x2 = torch.randn(1, lens[1], 6, device=cuda_device)
    h01 = torch.randn(1, 8, device=cuda_device)
    h02 = torch.randn(1, 8, device=cuda_device)
    x = torch.cat((x1, x2), dim=1)
    h0 = torch.cat((h01, h02), dim=0)
    cfg = NewtonConfig(max_iters=3, scan_backend="triton")
    packed = newton_apply(cell, x, cfg, h0=h0, cu_seqlens=cs)
    seq = sequential_apply(cell, x, h0, cu_seqlens=cs)
    torch.testing.assert_close(packed, seq, atol=1e-4, rtol=1e-4)


def test_pararnn_forward_cu_seqlens_matches_sequential():
    torch.manual_seed(12)
    cell = ParaGRU(d_in=4, d_h=5)
    cs = _cu((8, 11))
    x = torch.randn(1, 19, 4)
    h0 = torch.randn(2, 5)
    layer = ParaRNN(cell, config=NewtonConfig(max_iters=3, scan_backend="eager"), solver="newton")
    got = layer(x, h0, cu_seqlens=cs)
    ref = sequential_apply(cell, x, h0, cu_seqlens=cs)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
