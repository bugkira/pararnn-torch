"""M²RNN factorized J and Newton vs sequential."""

from __future__ import annotations

import pytest
import torch
from torch.func import jacrev

from pararnn.cells.para_m2rnn import ParaM2RNN
from pararnn.kernels.m2rnn_factor import m2rnn_gates, m2rnn_jt_mvp, m2rnn_jvp
from pararnn.kernels.newton_m2rnn import newton_m2rnn_factorized
from pararnn.solvers.newton import NewtonConfig, newton_apply
from pararnn.solvers.sequential import sequential_apply


def test_m2rnn_jvp_matches_jacrev():
    torch.manual_seed(0)
    k_dim, v_dim = 4, 5
    cell = ParaM2RNN(d_in=8, k_dim=k_dim, v_dim=v_dim)
    h = torch.randn(2, k_dim, v_dim)
    k = torch.randn(2, k_dim)
    v = torch.randn(2, v_dim)
    f = torch.sigmoid(torch.randn(2))
    w = cell.W.detach()
    h_new, acts = m2rnn_gates(h, k, v, f, w)
    assert h_new.shape == h.shape

    def f_one(h1: torch.Tensor, kk, vv, ff) -> torch.Tensor:
        out, _ = m2rnn_gates(h1, kk, vv, ff, w)
        return out

    # Per-batch jacrev
    jacs = []
    for b in range(2):
        j = jacrev(lambda hh: f_one(hh, k[b], v[b], f[b]))(h[b])
        # j: (K,V,K,V) → apply to delta
        jacs.append(j)
    delta = torch.randn_like(h)
    jvp = m2rnn_jvp(acts, delta)
    dense = torch.stack(
        [
            torch.einsum("abcd,cd->ab", jacs[b], delta[b])
            for b in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(jvp, dense, atol=1e-5, rtol=1e-5)


def test_m2rnn_jt_matches_jacrev():
    torch.manual_seed(1)
    k_dim, v_dim = 3, 4
    cell = ParaM2RNN(d_in=6, k_dim=k_dim, v_dim=v_dim)
    h = torch.randn(2, k_dim, v_dim)
    k = torch.randn(2, k_dim)
    v = torch.randn(2, v_dim)
    f = torch.sigmoid(torch.randn(2))
    w = cell.W.detach()
    _, acts = m2rnn_gates(h, k, v, f, w)
    mu = torch.randn_like(h)

    def f_one(hh, kk, vv, ff):
        out, _ = m2rnn_gates(hh, kk, vv, ff, w)
        return out

    jt = m2rnn_jt_mvp(acts, mu)
    dense = []
    for b in range(2):
        j = jacrev(lambda hh: f_one(hh, k[b], v[b], f[b]))(h[b])
        # J^T: (K,V,K,V) with out,in → einsum mu_ab J_ab,cd -> cd
        dense.append(torch.einsum("ab,abcd->cd", mu[b], j))
    torch.testing.assert_close(jt, torch.stack(dense), atol=1e-5, rtol=1e-5)


def test_m2rnn_sequential_runs():
    torch.manual_seed(2)
    cell = ParaM2RNN(d_in=7, k_dim=4, v_dim=4)
    x = 0.2 * torch.randn(2, 12, 7)
    h = sequential_apply(cell, x)
    assert h.shape == (2, 12, 4, 4)


def test_m2rnn_newton_matches_sequential():
    """Tiny K×V; raise K_Newton if residual stays large (see m2rnn-jacobian.md)."""
    torch.manual_seed(3)
    cell = ParaM2RNN(d_in=8, k_dim=4, v_dim=4).eval()
    x = 0.15 * torch.randn(2, 16, 8)
    seq = sequential_apply(cell, x)
    # Measure residual drop
    errs = []
    with torch.no_grad():
        for k in (1, 2, 3, 4, 6, 8):
            par = newton_m2rnn_factorized(cell, x, max_iters=k, omega=1.0)
            errs.append(float((par - seq).abs().amax()))
    assert errs[-1] < 5e-4, errs
    assert errs[0] > errs[-1], errs


def test_m2rnn_newton_bit_stable_on_cpu():
    torch.manual_seed(4)
    cell = ParaM2RNN(d_in=5, k_dim=3, v_dim=3)
    x = 0.2 * torch.randn(1, 8, 5)
    a = newton_m2rnn_factorized(cell, x, max_iters=5)
    b = newton_m2rnn_factorized(cell, x, max_iters=5)
    assert torch.equal(a, b)


def test_m2rnn_newton_apply_matches_sequential():
    torch.manual_seed(5)
    cell = ParaM2RNN(d_in=8, k_dim=4, v_dim=4).eval()
    x = 0.15 * torch.randn(2, 16, 8)
    seq = sequential_apply(cell, x)
    cfg = NewtonConfig(max_iters=4, omega=1.0, residual_atol=None)
    par = newton_apply(cell, x, cfg)
    torch.testing.assert_close(par, seq, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused M²RNN")
def test_m2rnn_fused_matches_eager_and_sequential():
    torch.manual_seed(7)
    device = torch.device("cuda")
    cell = ParaM2RNN(d_in=8, k_dim=8, v_dim=8).to(device).eval()
    x = (0.15 * torch.randn(2, 32, 8, device=device)).detach()
    seq = sequential_apply(cell, x)
    fused = newton_apply(
        cell, x, NewtonConfig(max_iters=4, scan_backend="fused", residual_atol=None)
    )
    eager = newton_apply(
        cell, x, NewtonConfig(max_iters=4, scan_backend="eager", residual_atol=None)
    )
    torch.testing.assert_close(fused, seq, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(eager, seq, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(fused, eager, atol=5e-5, rtol=5e-5)


def test_m2rnn_newton_apply_grad_finite():
    torch.manual_seed(6)
    cell = ParaM2RNN(d_in=6, k_dim=3, v_dim=3)
    x = torch.randn(2, 8, 6, requires_grad=True)
    with torch.no_grad():
        x.mul_(0.2)
    cfg = NewtonConfig(max_iters=4, omega=1.0, residual_atol=None)
    h = newton_apply(cell, x, cfg)
    h.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in cell.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hybrid tiled M²RNN")
def test_m2rnn_hybrid_tiled_matches_eager():
    torch.manual_seed(8)
    device = torch.device("cuda")
    # K,V > 64 → hybrid tiled path
    cell = ParaM2RNN(d_in=16, k_dim=96, v_dim=96).to(device).eval()
    x = (0.1 * torch.randn(1, 24, 16, device=device)).detach()
    seq = sequential_apply(cell, x)
    fused = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=6, scan_backend="fused", residual_atol=None, residual_fail=None),
    )
    eager = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=6, scan_backend="eager", residual_atol=None, residual_fail=None),
    )
    torch.testing.assert_close(fused, seq, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(eager, seq, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(fused, eager, atol=1e-4, rtol=1e-4)


def test_m2rnn_packed_vjp_matches_autograd():
    torch.manual_seed(9)
    from pararnn.solvers.vjp import cell_vjp, uses_packed_vjp

    cell = ParaM2RNN(d_in=5, k_dim=4, v_dim=4)
    assert uses_packed_vjp(cell)
    h = torch.randn(2, 7, 4, 4)
    x = torch.randn(2, 7, 5)
    mu = torch.randn(2, 7, 4, 4)
    gx_p, gp_p = cell_vjp(cell, h, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h, x, mu, packed=False)
    torch.testing.assert_close(gx_p, gx_a, atol=2e-5, rtol=2e-5)
    for a, b in zip(gp_p, gp_a, strict=True):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_m2rnn_frozen_w_scan_matches_w0_sequential():
    """Frozen-W warm-start equals sequential with ``W=0``."""
    from pararnn.kernels.m2rnn_factor import m2rnn_frozen_w_scan, m2rnn_gates
    from pararnn.layout import prepend_state

    torch.manual_seed(10)
    b, t, kd, vd = 2, 12, 4, 5
    k = torch.randn(b, t, kd)
    v = torch.randn(b, t, vd)
    f = torch.sigmoid(torch.randn(b, t))
    h0 = torch.randn(b, kd, vd)
    w0 = torch.zeros(vd, vd)
    warm = m2rnn_frozen_w_scan(k, v, f, h0=h0)
    # Sequential W=0
    h = h0
    parts = []
    for ti in range(t):
        h, _ = m2rnn_gates(h, k[:, ti], v[:, ti], f[:, ti], w0)
        parts.append(h)
    ref = torch.stack(parts, dim=1)
    torch.testing.assert_close(warm, ref, atol=1e-6, rtol=1e-6)
    # Sanity: differs from App. A (full W, zero-prev) when W is identity-ish
    zeros = k.new_zeros(b, t, kd, vd)
    app_a, _ = m2rnn_gates(prepend_state(zeros, h0), k, v, f, torch.eye(vd))
    assert (warm - app_a).abs().amax() > 1e-3


def test_m2rnn_frozen_w_newton_matches_sequential():
    torch.manual_seed(11)
    cell = ParaM2RNN(d_in=8, k_dim=4, v_dim=4).eval()
    x = 0.15 * torch.randn(2, 24, 8)
    seq = sequential_apply(cell, x)
    cfg = NewtonConfig(
        max_iters=6,
        omega=1.0,
        picard_iters=1,
        residual_atol=None,
        residual_fail=None,
    )
    par = newton_apply(cell, x, cfg)
    torch.testing.assert_close(par, seq, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused frozen-W")
def test_m2rnn_fused_frozen_w_matches_eager():
    torch.manual_seed(12)
    device = torch.device("cuda")
    cell = ParaM2RNN(d_in=8, k_dim=8, v_dim=8).to(device).eval()
    x = (0.15 * torch.randn(2, 32, 8, device=device)).detach()
    cfg_kw = dict(
        max_iters=5,
        picard_iters=1,
        residual_atol=None,
        residual_fail=None,
    )
    fused = newton_apply(cell, x, NewtonConfig(scan_backend="fused", **cfg_kw))
    eager = newton_apply(cell, x, NewtonConfig(scan_backend="eager", **cfg_kw))
    seq = sequential_apply(cell, x)
    torch.testing.assert_close(fused, seq, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(fused, eager, atol=5e-5, rtol=5e-5)
