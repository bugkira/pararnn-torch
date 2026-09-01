"""Scan vs forward substitution; Newton vs sequential unroll."""

from __future__ import annotations

import logging

import pytest
import torch

from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.solvers import (
    NewtonConfig,
    newton_apply,
    sequential_apply,
    sequential_apply_compiled,
)
from pararnn.solvers.scan import (
    reverse_scan_block2,
    reverse_scan_diag,
    scan_block2,
    scan_diag,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_scan_diag_matches_forward_substitution():
    torch.manual_seed(2)
    b, t, d = 4, 17, 5  # 17 is not a power of two
    jac = torch.randn(b, t, d, device=device) * 0.3
    residual = torch.randn(b, t, d, device=device)
    got = scan_diag(jac, residual)
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        ref[:, s] = jac[:, s] * ref[:, s - 1] + residual[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_scan_diag_power_of_two():
    torch.manual_seed(13)
    b, t, d = 2, 8, 3
    jac = torch.randn(b, t, d, device=device) * 0.3
    residual = torch.randn(b, t, d, device=device)
    got = scan_diag(jac, residual)
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        ref[:, s] = jac[:, s] * ref[:, s - 1] + residual[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_scan_block2_matches_forward_substitution():
    torch.manual_seed(3)
    b, t, d = 2, 9, 4
    jac = torch.randn(b, t, 2, 2, d, device=device) * 0.2
    residual = torch.randn(b, t, 2, d, device=device)
    got = scan_block2(jac, residual)
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        # J @ delta_prev + r  (einsum out,in)
        ref[:, s] = (
            torch.einsum("boid,bid->bod", jac[:, s], ref[:, s - 1]) + residual[:, s]
        )
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paragru_newton_matches_sequential():
    torch.manual_seed(4)
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(3, 32, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_paralstm_newton_matches_sequential():
    torch.manual_seed(5)
    cell = ParaLSTM(d_in=8, d_h=12).to(device)
    x = torch.randn(3, 32, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


def test_sequential_rejects_empty_time():
    cell = ParaGRU(d_in=4, d_h=8).to(device)
    x = torch.randn(2, 0, 4, device=device)
    with pytest.raises(ValueError, match="time >= 1"):
        sequential_apply(cell, x)
    with pytest.raises(ValueError, match="time >= 1"):
        sequential_apply_compiled(cell, x)


@torch.no_grad()
def test_paragru_newton_t1_matches_sequential():
    torch.manual_seed(41)
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(3, 1, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_paragru_newton_eager_no_residual_atol():
    torch.manual_seed(42)
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(2, 16, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, scan_backend="eager", residual_atol=None),
    )
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


def test_reverse_scan_diag_matches_backward_substitution():
    torch.manual_seed(8)
    b, t, d = 3, 11, 6
    jac = torch.randn(b, t, d, device=device) * 0.3
    partial = torch.randn(b, t, d, device=device)
    got = reverse_scan_diag(jac, partial)
    ref = torch.zeros_like(partial)
    ref[:, -1] = partial[:, -1]
    for s in range(t - 2, -1, -1):
        ref[:, s] = jac[:, s + 1] * ref[:, s + 1] + partial[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_reverse_scan_block2_matches_backward_substitution():
    torch.manual_seed(9)
    b, t, d = 2, 7, 4
    jac = torch.randn(b, t, 2, 2, d, device=device) * 0.2
    partial = torch.randn(b, t, 2, d, device=device)
    got = reverse_scan_block2(jac, partial)
    ref = torch.zeros_like(partial)
    ref[:, -1] = partial[:, -1]
    for s in range(t - 2, -1, -1):
        j_t = jac[:, s + 1].transpose(-3, -2)
        ref[:, s] = torch.einsum("boid,bid->bod", j_t, ref[:, s + 1]) + partial[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def _assert_param_grads_close(
    cell_a: torch.nn.Module, cell_b: torch.nn.Module, atol: float
) -> None:
    for (n, p_a), (_, p_b) in zip(cell_a.named_parameters(), cell_b.named_parameters()):
        assert p_a.grad is not None, n
        assert p_b.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=atol, rtol=1e-4)


def test_paragru_newton_bwd_matches_sequential_bptt():
    torch.manual_seed(10)
    d_in, d_h, t = 5, 7, 16
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = ParaGRU(d_in, d_h).to(device)
    cell_n = ParaGRU(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x_n, NewtonConfig(max_iters=3)) * w).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=2e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=2e-4, rtol=1e-4)


def test_paragru_newton_h0_grad_matches_sequential_bptt():
    torch.manual_seed(14)
    d_in, d_h, t = 5, 7, 16
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = ParaGRU(d_in, d_h).to(device)
    cell_n = ParaGRU(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    h0 = 0.3 * torch.randn(2, d_h, device=device)
    h0_s = h0.clone().requires_grad_(True)
    h0_n = h0.clone().requires_grad_(True)
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    loss_s = (sequential_apply(cell_s, x_s, h0_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x_n, cfg, h0=h0_n) * w).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=2e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(h0_s.grad, h0_n.grad, atol=2e-4, rtol=1e-4)


def test_paragru_h0_only_requires_grad():
    """Eq. 2.6 must see h0 even when x and θ are frozen."""
    torch.manual_seed(15)
    cell = ParaGRU(d_in=4, d_h=6).to(device)
    for p in cell.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, 12, 4, device=device)
    h0 = (0.3 * torch.randn(2, 6, device=device)).detach().requires_grad_(True)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    y = newton_apply(cell, x, cfg, h0=h0)
    y.sum().backward()
    assert h0.grad is not None
    assert h0.grad.abs().sum() > 0


def test_paralstm_newton_h0_grad_matches_sequential_bptt():
    torch.manual_seed(16)
    d_in, d_h, t = 5, 6, 12
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, 2, d_h, device=device)
    cell_s = ParaLSTM(d_in, d_h).to(device)
    cell_n = ParaLSTM(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    h0 = 0.3 * torch.randn(2, 2, d_h, device=device)
    h0_s = h0.clone().requires_grad_(True)
    h0_n = h0.clone().requires_grad_(True)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    loss_s = (sequential_apply(cell_s, x, h0_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x, cfg, h0=h0_n) * w).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=5e-4)
    torch.testing.assert_close(h0_s.grad, h0_n.grad, atol=5e-4, rtol=1e-4)


def test_paralstm_newton_bwd_matches_sequential_bptt():
    torch.manual_seed(11)
    d_in, d_h, t = 5, 6, 12
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, 2, d_h, device=device)
    cell_s = ParaLSTM(d_in, d_h).to(device)
    cell_n = ParaLSTM(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x_n, NewtonConfig(max_iters=3)) * w).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=5e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


@torch.no_grad()
def test_compiled_sequential_matches_eager():
    torch.manual_seed(12)
    if device.type != "cuda":
        return
    cell = ParaGRU(d_in=8, d_h=16).to(device).eval()
    x = torch.randn(2, 32, 8, device=device)
    eager = sequential_apply(cell, x)
    compiled = sequential_apply_compiled(cell, x)
    torch.testing.assert_close(compiled, eager, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_compiled_newton_matches_sequential():
    """Call-site torch.compile(newton_apply); not a library solver."""
    torch.manual_seed(16)
    if device.type != "cuda":
        return
    cfg = NewtonConfig(max_iters=3)
    for cell in (
        ParaGRU(d_in=8, d_h=16).to(device).eval(),
        ParaLSTM(d_in=8, d_h=12).to(device).eval(),
    ):
        x = torch.randn(2, 32, cell.d_in, device=device)
        seq = sequential_apply(cell, x)
        eager = newton_apply(cell, x, cfg)

        def _fwd(xx, c=cell, conf=cfg):
            return newton_apply(c, xx, conf)

        compiled = torch.compile(_fwd, mode="reduce-overhead")
        torch.compiler.cudagraph_mark_step_begin()
        out = compiled(x)
        torch.testing.assert_close(out, seq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(out, eager, atol=1e-5, rtol=1e-5)


def _cuda_or_skip() -> torch.device | None:
    if device.type != "cuda":
        return None
    return device


@torch.no_grad()
def test_triton_scan_diag_matches_substitution():
    torch.manual_seed(20)
    device = _cuda_or_skip()
    if device is None:
        return
    from pararnn.kernels import scan_diag_triton

    for t in (17, 128, 129, 256):
        jac = torch.randn(3, t, 7, device=device) * 0.3
        residual = torch.randn(3, t, 7, device=device)
        got = scan_diag_triton(jac, residual)
        ref = torch.zeros_like(residual)
        ref[:, 0] = residual[:, 0]
        for s in range(1, t):
            ref[:, s] = jac[:, s] * ref[:, s - 1] + residual[:, s]
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_triton_scan_diag_matches_eager():
    torch.manual_seed(21)
    device = _cuda_or_skip()
    if device is None:
        return
    jac = torch.randn(2, 300, 40, device=device) * 0.3
    residual = torch.randn(2, 300, 40, device=device)
    eager = scan_diag(jac, residual)
    tri = scan_diag(jac, residual, backend="triton")
    torch.testing.assert_close(tri, eager, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paragru_newton_triton_scan_matches_sequential():
    torch.manual_seed(22)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(3, 64, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="triton"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_triton_scan_block2_matches_substitution():
    torch.manual_seed(24)
    device = _cuda_or_skip()
    if device is None:
        return
    from pararnn.kernels import scan_block2_triton

    for t in (9, 64, 65, 128):
        jac = torch.randn(2, t, 2, 2, 5, device=device) * 0.2
        residual = torch.randn(2, t, 2, 5, device=device)
        got = scan_block2_triton(jac, residual)
        ref = torch.zeros_like(residual)
        ref[:, 0] = residual[:, 0]
        for s in range(1, t):
            ref[:, s] = (
                torch.einsum("boid,bid->bod", jac[:, s], ref[:, s - 1]) + residual[:, s]
            )
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_triton_scan_block2_matches_eager():
    torch.manual_seed(25)
    device = _cuda_or_skip()
    if device is None:
        return
    jac = torch.randn(2, 200, 2, 2, 11, device=device) * 0.2
    residual = torch.randn(2, 200, 2, 11, device=device)
    eager = scan_block2(jac, residual)
    tri = scan_block2(jac, residual, backend="triton")
    torch.testing.assert_close(tri, eager, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_paralstm_newton_triton_scan_matches_sequential():
    torch.manual_seed(26)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaLSTM(d_in=8, d_h=12).to(device)
    x = torch.randn(3, 48, 8, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="triton"))
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


def test_paralstm_newton_triton_bwd_matches_sequential_bptt():
    torch.manual_seed(27)
    device = _cuda_or_skip()
    if device is None:
        return
    d_in, d_h, t = 5, 6, 12
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, 2, d_h, device=device)
    cell_s = ParaLSTM(d_in, d_h).to(device)
    cell_n = ParaLSTM(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (
        newton_apply(cell_n, x_n, NewtonConfig(max_iters=3, scan_backend="triton")) * w
    ).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=5e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


@torch.no_grad()
def test_paragru_newton_fused_matches_sequential():
    torch.manual_seed(28)
    device = _cuda_or_skip()
    if device is None:
        return
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    for t in (32, 200):
        cell = ParaGRU(d_in=8, d_h=16).to(device)
        x = torch.randn(3, t, 8, device=device)
        seq = sequential_apply(cell, x)
        eager = newton_apply(cell, x, NewtonConfig(max_iters=3))
        par = newton_apply(cell, x, cfg)
        err_s = (par - seq).abs().amax()
        err_e = (par - eager).abs().amax()
        assert err_s < 1e-4, (t, float(err_s))
        assert err_e < 1e-4, (t, float(err_e))


@torch.no_grad()
def test_paralstm_newton_fused_matches_sequential():
    torch.manual_seed(29)
    device = _cuda_or_skip()
    if device is None:
        return
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    for t in (48, 100):
        cell = ParaLSTM(d_in=8, d_h=12).to(device)
        x = torch.randn(3, t, 8, device=device)
        seq = sequential_apply(cell, x)
        eager = newton_apply(cell, x, NewtonConfig(max_iters=3))
        par = newton_apply(cell, x, cfg)
        err_s = (par - seq).abs().amax()
        err_e = (par - eager).abs().amax()
        assert err_s < 1e-4, (t, float(err_s))
        assert err_e < 1e-4, (t, float(err_e))


def test_paragru_newton_fused_bwd_matches_sequential_bptt():
    torch.manual_seed(30)
    device = _cuda_or_skip()
    if device is None:
        return
    d_in, d_h, t = 5, 7, 16
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = ParaGRU(d_in, d_h).to(device)
    cell_n = ParaGRU(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (
        newton_apply(cell_n, x_n, NewtonConfig(max_iters=3, scan_backend="fused")) * w
    ).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=2e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=2e-4, rtol=1e-4)


def test_paralstm_newton_fused_bwd_matches_sequential_bptt():
    torch.manual_seed(31)
    device = _cuda_or_skip()
    if device is None:
        return
    d_in, d_h, t = 5, 6, 80
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, 2, d_h, device=device)
    cell_s = ParaLSTM(d_in, d_h).to(device)
    cell_n = ParaLSTM(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (
        newton_apply(cell_n, x_n, NewtonConfig(max_iters=3, scan_backend="fused")) * w
    ).sum()
    loss_n.backward()
    _assert_param_grads_close(cell_s, cell_n, atol=5e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


_FP16_ATOL = 2e-3  # bottlenecks.md: fp16 residual ~1e-3; LSTM a bit looser


@torch.no_grad()
def test_paragru_newton_fused_fp16_matches_sequential():
    torch.manual_seed(40)
    device = _cuda_or_skip()
    if device is None:
        return
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    for t in (32, 200):
        cell = ParaGRU(d_in=8, d_h=16).to(device=device, dtype=torch.float16)
        x = torch.randn(3, t, 8, device=device, dtype=torch.float16)
        seq = sequential_apply(cell, x)
        par = newton_apply(cell, x, cfg)
        err = (par.float() - seq.float()).abs().amax()
        assert err < _FP16_ATOL, (t, float(err))
        assert par.dtype is torch.float16


@torch.no_grad()
def test_paralstm_newton_fused_fp16_matches_sequential():
    torch.manual_seed(41)
    device = _cuda_or_skip()
    if device is None:
        return
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    for t in (48, 100):
        cell = ParaLSTM(d_in=8, d_h=12).to(device=device, dtype=torch.float16)
        x = torch.randn(3, t, 8, device=device, dtype=torch.float16)
        seq = sequential_apply(cell, x)
        par = newton_apply(cell, x, cfg)
        err = (par.float() - seq.float()).abs().amax()
        assert err < _FP16_ATOL, (t, float(err))
        assert par.dtype is torch.float16


@torch.no_grad()
def test_triton_scan_diag_fp16_matches_fp32():
    torch.manual_seed(42)
    device = _cuda_or_skip()
    if device is None:
        return
    jac = torch.randn(2, 200, 16, device=device) * 0.3
    residual = torch.randn(2, 200, 16, device=device)
    ref = scan_diag(jac, residual, backend="triton")
    got = scan_diag(jac.half(), residual.half(), backend="triton")
    torch.testing.assert_close(got.float(), ref, atol=2e-3, rtol=2e-3)


@torch.no_grad()
def test_paragru_newton_eager_fp16_matches_sequential():
    torch.manual_seed(43)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaGRU(d_in=8, d_h=16).to(device=device, dtype=torch.float16)
    x = torch.randn(3, 40, 8, device=device, dtype=torch.float16)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    err = (par.float() - seq.float()).abs().amax()
    assert err < _FP16_ATOL, float(err)


@torch.no_grad()
def test_paragru_newton_fused_fp16_matches_eager():
    torch.manual_seed(44)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaGRU(d_in=8, d_h=16).to(device=device, dtype=torch.float16)
    x = torch.randn(3, 200, 8, device=device, dtype=torch.float16)
    eager = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="eager"))
    fused = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="fused"))
    err = (fused.float() - eager.float()).abs().amax()
    assert err < _FP16_ATOL, float(err)


@torch.no_grad()
def test_fused_bf16_capability():
    """Fused bf16: CC ≥ 8.0 agrees with sequential; CC 7.x raises."""
    device = _cuda_or_skip()
    if device is None:
        return
    from pararnn.kernels.precision import is_fused_dtype_supported

    cell = ParaGRU(d_in=4, d_h=8).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 16, 4, device=device, dtype=torch.bfloat16)
    if not is_fused_dtype_supported(torch.bfloat16, device):
        with pytest.raises(TypeError, match="compute capability"):
            newton_apply(cell, x, NewtonConfig(max_iters=1, scan_backend="fused"))
        return
    seq = sequential_apply(cell, x)
    fused = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="fused"))
    err = (fused.float() - seq.float()).abs().amax()
    assert err < _FP16_ATOL, float(err)


@torch.no_grad()
def test_auto_bf16_matches_sequential():
    """``auto`` uses fused on CC≥8, eager on SM 7.x — neither should raise."""
    device = _cuda_or_skip()
    if device is None:
        return
    from pararnn.kernels.precision import is_fused_dtype_supported

    cell = ParaGRU(d_in=4, d_h=8).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 16, 4, device=device, dtype=torch.bfloat16)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="auto"))
    if not is_fused_dtype_supported(torch.bfloat16, device):
        return
    seq = sequential_apply(cell, x)
    err = (par.float() - seq.float()).abs().amax()
    assert err < _FP16_ATOL, float(err)


def test_is_fused_dtype_supported_cpu_false():
    from pararnn.kernels.precision import is_fused_dtype_supported

    assert is_fused_dtype_supported(torch.float32, torch.device("cpu")) is False
    assert is_fused_dtype_supported(torch.float16, torch.device("cpu")) is False
    assert is_fused_dtype_supported(torch.bfloat16, torch.device("cpu")) is False
    if device.type == "cuda":
        assert is_fused_dtype_supported(torch.float32, device) is True
        assert is_fused_dtype_supported(torch.float16, device) is True
        major, _minor = torch.cuda.get_device_capability(device)
        assert is_fused_dtype_supported(torch.bfloat16, device) is (major >= 8)


def test_validate_cuda_tensors_dtype_and_device():
    from pararnn.kernels.precision import validate_cuda_tensors

    a = torch.zeros(2, device="cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_cuda_tensors(a, name="t")
    if device.type != "cuda":
        return
    x = torch.zeros(2, device=device)
    assert validate_cuda_tensors(x, name="t") is False
    h = torch.zeros(2, device=device, dtype=torch.float16)
    assert validate_cuda_tensors(h, name="t") is True
    with pytest.raises(TypeError, match="dtype mismatch"):
        validate_cuda_tensors(x, h, name="t")
    if torch.cuda.device_count() < 2:
        return
    other = torch.device("cuda:1")
    y = torch.zeros(2, device=other)
    with pytest.raises(RuntimeError, match="different devices"):
        validate_cuda_tensors(x, y, name="t")

@torch.no_grad()
def test_paragru_newton_h0_matches_sequential():
    torch.manual_seed(80)
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(3, 24, 8, device=device)
    # App. A init is for h0=0; unit-scale randn leaves ~2e-4 after K=3.
    h0 = 0.3 * torch.randn(3, 16, device=device)
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3), h0=h0)
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_paralstm_newton_h0_matches_sequential():
    torch.manual_seed(81)
    cell = ParaLSTM(d_in=8, d_h=12).to(device)
    x = torch.randn(3, 20, 8, device=device)
    h0 = 0.3 * torch.randn(3, 2, 12, device=device)
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3), h0=h0)
    err = (par - seq).abs().amax()
    assert err < 1e-4, err


@torch.no_grad()
def test_fused_nonzero_h0_matches_sequential(caplog):
    torch.manual_seed(82)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaGRU(d_in=8, d_h=16).to(device)
    x = torch.randn(2, 32, 8, device=device)
    h0 = 0.3 * torch.randn(2, 16, device=device)
    cfg = NewtonConfig(max_iters=3, scan_backend="fused")
    with caplog.at_level(logging.WARNING, logger="pararnn.solvers.newton"):
        par = newton_apply(cell, x, cfg, h0=h0)
    assert "fused_h0_fallback_eager" not in caplog.text
    seq = sequential_apply(cell, x, h0)
    err = (par - seq).abs().amax()
    assert err < 1e-4, err
    zeros = torch.zeros_like(h0)
    par_z = newton_apply(cell, x, cfg, h0=zeros)
    seq_z = sequential_apply(cell, x, zeros)
    assert (par_z - seq_z).abs().amax() < 1e-4


@torch.no_grad()
def test_fused_lstm_nonzero_h0_matches_sequential():
    torch.manual_seed(83)
    device = _cuda_or_skip()
    if device is None:
        return
    cell = ParaLSTM(d_in=8, d_h=12).to(device)
    x = torch.randn(2, 24, 8, device=device)
    h0 = 0.3 * torch.randn(2, 2, 12, device=device)
    par = newton_apply(cell, x, NewtonConfig(max_iters=3, scan_backend="fused"), h0=h0)
    seq = sequential_apply(cell, x, h0)
    err = (par - seq).abs().amax()
    assert err < 1e-4, err
