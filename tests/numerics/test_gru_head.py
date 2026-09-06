"""ParaGRU mix='head' (Dreamer-style block-diagonal recurrent)."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, ParaGRU, newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytestmark = pytest.mark.filterwarnings(
    "ignore:mix='head' is block-diagonal ParaGRU:UserWarning"
)


def _residual_vs_k(cell: ParaGRU, x: torch.Tensor, ks: tuple[int, ...] = (1, 2, 3, 4, 5)):
    ref = sequential_apply(cell, x)
    rows = []
    for k in ks:
        cfg = NewtonConfig(max_iters=k, scan_backend="eager", residual_atol=None)
        h = newton_apply(cell, x, cfg)
        err = float((h - ref).abs().amax())
        rows.append((k, err, err))
    return rows


def test_paragru_head_needs_dividing_n_heads():
    with pytest.raises(ValueError, match="n_heads"):
        ParaGRU(d_in=4, d_h=4, mix="head")
    with pytest.raises(ValueError, match="n_heads"):
        ParaGRU(d_in=4, d_h=4, mix="head", n_heads=3)
    with pytest.raises(ValueError, match="n_heads"):
        ParaGRU(d_in=4, d_h=4, mix="diag", n_heads=2)


def test_paragru_head_default_no_clip():
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    assert cell.max_recurrent_norm is None


@torch.no_grad()
def test_paragru_head_newton_vs_sequential():
    torch.manual_seed(301)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4).to(device)
    assert cell.jac_structure == "head"
    assert cell.d_head == 2
    x = 0.3 * torch.randn(2, 16, 4, device=device)
    rows = _residual_vs_k(cell, x)
    err_k5 = rows[-1][2]
    assert err_k5 < 5e-3, rows
    assert min(err for _, _, err in rows) < 1e-4, rows


@torch.no_grad()
def test_paragru_head_matches_forced_dense():
    torch.manual_seed(301)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4).to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    cfg = {
        "max_iters": 5,
        "scan_backend": "eager",
        "residual_atol": None,
        "jacobian": "autograd",
    }
    hh = newton_apply(cell, x, NewtonConfig(**cfg))
    hd = newton_apply(cell, x, NewtonConfig(**cfg, jac_structure="dense"))
    torch.testing.assert_close(hh, hd, atol=2e-5, rtol=1e-5)


@torch.no_grad()
def test_paragru_step_head_matches_step():
    torch.manual_seed(302)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4).to(device)
    h = torch.randn(3, 8, device=device)
    x = 0.3 * torch.randn(3, 4, device=device)
    out = cell.step(h, x)
    wx = cell.W_x(x).reshape(3, 3, 4, 2)
    a_z, a_r, a_n = cell.clipped_a_head()
    parts = []
    for b in range(3):
        heads = [
            cell.step_head(h[b].reshape(4, 2)[hd], wx[b, :, hd], a_z[hd], a_r[hd], a_n[hd])
            for hd in range(4)
        ]
        parts.append(torch.cat(heads, dim=0))
    torch.testing.assert_close(torch.stack(parts), out)


@torch.no_grad()
def test_paragru_head_analytic_jac_matches_autograd():
    torch.manual_seed(303)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4).to(device)
    h = torch.randn(2, 5, 8, device=device)
    x = 0.3 * torch.randn(2, 5, 4, device=device)
    pred_a, jac_a = cell.step_with_jacobian(h, x)
    from pararnn.solvers.jacobian import jacobian_autograd

    pred_g, jac_g = jacobian_autograd(cell, h, x, structure="head")
    torch.testing.assert_close(pred_a, pred_g, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jac_a, jac_g, atol=1e-5, rtol=1e-5)


def test_paragru_head_newton_bwd_matches_sequential_bptt():
    torch.manual_seed(304)
    d_in, d_h, t = 4, 8, 12
    x = 0.3 * torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = ParaGRU(d_in, d_h, mix="head", n_heads=4).to(device)
    cell_n = ParaGRU(d_in, d_h, mix="head", n_heads=4).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    cfg = NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x_n, cfg) * w).sum()
    loss_n.backward()
    for (n, p_a), (_, p_b) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert p_a.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


@torch.no_grad()
def test_paragru_head_fused_cpu_refuses():
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    x = torch.randn(1, 8, 4)
    with pytest.raises(TypeError, match="fused"):
        newton_apply(cell, x, NewtonConfig(max_iters=2, scan_backend="fused"))


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_fused_vs_eager():
    torch.manual_seed(410)
    cell = ParaGRU(d_in=16, d_h=64, mix="head", n_heads=8, device="cuda")
    x = 0.2 * torch.randn(2, 32, 16, device="cuda")
    cfg_e = NewtonConfig(max_iters=4, scan_backend="eager", residual_atol=None)
    cfg_f = NewtonConfig(max_iters=4, scan_backend="fused", residual_atol=None)
    he = newton_apply(cell, x, cfg_e)
    hf = newton_apply(cell, x, cfg_f)
    torch.testing.assert_close(hf, he, atol=2e-4, rtol=1e-4)
    ref = sequential_apply(cell, x)
    assert float((hf - ref).abs().amax()) < 5e-3


@pytest.mark.cuda
@torch.no_grad()
@pytest.mark.parametrize("d_head,n_heads", [(64, 8), (96, 2), (128, 2)])
def test_paragru_head_long_t_fused_smoke(d_head: int, n_heads: int):
    """No T-cap on head fused path: T=4096 stays finite and self-consistent."""
    torch.manual_seed(430 + d_head)
    d_h = n_heads * d_head
    cell = ParaGRU(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads, device="cuda")
    x = 0.12 * torch.randn(1, 4096, d_h, device="cuda")
    cfg = NewtonConfig(max_iters=3, scan_backend="fused", residual_atol=None)
    h1 = newton_apply(cell, x, cfg)
    h2 = newton_apply(cell, x, cfg)
    assert torch.isfinite(h1).all()
    torch.testing.assert_close(h1, h2, atol=0.0, rtol=0.0)
    # Factor reverse at same T (stream / tiled).
    from pararnn.kernels.newton_gru_head import (
        _factor_reverse_eager,
        _gates,
        _reverse_gru_head_factor_impl,
    )

    wx = cell.W_x(x)
    h_prev = torch.cat([torch.zeros(1, 1, d_h, device="cuda"), h1[:, :-1]], dim=1)
    a_z, a_r, a_n = cell.clipped_a_head()
    part = 0.1 * torch.randn_like(h1)
    _, gates = _gates(h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h)
    eager = _factor_reverse_eager(
        gates, part, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
    )
    tri = _reverse_gru_head_factor_impl(h_prev, wx, part, a_z, a_r, a_n)
    torch.testing.assert_close(eager, tri, atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_factorized_d_head_grid():
    torch.manual_seed(411)
    for d_head in (8, 64, 96, 128, 192, 256):
        n_heads = 2
        d_h = n_heads * d_head
        cell = ParaGRU(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads, device="cuda")
        x = 0.15 * torch.randn(2, 24, d_h, device="cuda")
        h = newton_apply(
            cell, x, NewtonConfig(max_iters=4, scan_backend="fused", residual_atol=None)
        )
        ref = sequential_apply(cell, x)
        err = float((h - ref).abs().amax())
        assert err < 5e-3, (d_head, err)

def test_paragru_diag_still_default():
    cell = ParaGRU(4, 8)
    assert cell.mix == "diag"
    assert cell.jac_structure == "diag"
    assert cell.a_z is not None
    assert cell.A_z is None
    assert cell.max_recurrent_norm == 0.5
    assert "mix='diag'" in cell.extra_repr()


def test_paragru_head_uses_packed_vjp():
    from pararnn.solvers.vjp import uses_packed_vjp

    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    assert uses_packed_vjp(cell) is True


@torch.no_grad()
def test_paragru_head_scale_newton_vs_sequential():
    """Dreamer-ish width: d_h=64, n_heads=8, T=64."""
    torch.manual_seed(401)
    cell = ParaGRU(d_in=32, d_h=64, mix="head", n_heads=8).to(device)
    x = 0.2 * torch.randn(2, 64, 32, device=device)
    ref = sequential_apply(cell, x)
    h = newton_apply(
        cell, x, NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None)
    )
    err = float((h - ref).abs().amax())
    assert err < 5e-3, err


@torch.no_grad()
def test_paragru_head_ragged_matches_concat():
    torch.manual_seed(402)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    lengths = (5, 7, 4)
    xs = [0.3 * torch.randn(1, t, 4) for t in lengths]
    refs = [sequential_apply(cell, x) for x in xs]
    x = torch.cat(xs, dim=1)
    cs = torch.tensor([0, 5, 12, 16], dtype=torch.long)
    cfg = NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None)
    packed = newton_apply(cell, x, cfg, cu_seqlens=cs)
    torch.testing.assert_close(packed[:, 0:5], refs[0], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(packed[:, 5:12], refs[1], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(packed[:, 12:16], refs[2], atol=2e-4, rtol=1e-4)


@torch.no_grad()
def test_paragru_head_fused_cu_seqlens_raises():
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    x = 0.2 * torch.randn(1, 16, 4)
    cs = torch.tensor([0, 5, 12, 16], dtype=torch.long)
    cfg = NewtonConfig(max_iters=3, scan_backend="fused", residual_atol=None)
    with pytest.raises(TypeError, match="cu_seqlens"):
        newton_apply(cell, x, cfg, cu_seqlens=cs)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_auto_cu_seqlens_eager(cuda_device: torch.device):
    torch.manual_seed(403)
    cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4, device=cuda_device)
    lengths = (5, 7, 4)
    xs = [0.3 * torch.randn(1, t, 4, device=cuda_device) for t in lengths]
    refs = [sequential_apply(cell, x) for x in xs]
    x = torch.cat(xs, dim=1)
    cs = torch.tensor([0, 5, 12, 16], dtype=torch.long, device=cuda_device)
    cfg = NewtonConfig(max_iters=5, scan_backend="auto", residual_atol=None)
    with pytest.warns(UserWarning, match="cu_seqlens"):
        packed = newton_apply(cell, x, cfg, cu_seqlens=cs)
    torch.testing.assert_close(packed[:, 0:5], refs[0], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(packed[:, 5:12], refs[1], atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(packed[:, 12:16], refs[2], atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_reverse_scan_dense_triton_matches_eager():
    from pararnn.solvers.scan import reverse_scan_dense

    torch.manual_seed(412)
    b, t, d = 3, 19, 16
    jac = torch.randn(b, t, d, d, device="cuda") * 0.15
    part = torch.randn(b, t, d, device="cuda")
    eager = reverse_scan_dense(jac, part, backend="eager")
    tri = reverse_scan_dense(jac, part, backend="triton")
    torch.testing.assert_close(eager, tri, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
@pytest.mark.parametrize("d_head,n_heads", [(16, 4), (96, 2)])
@pytest.mark.parametrize("time", [32, 256, 512])
def test_paragru_head_vjp_triton_matches_eager(d_head: int, n_heads: int, time: int):
    from pararnn.kernels.vjp_gru import (
        gru_head_recurrence_vjp,
        gru_head_recurrence_vjp_eager,
    )

    torch.manual_seed(420 + d_head + time)
    b, t, d_h = 2, time, n_heads * d_head
    h_prev = torch.randn(b, t, d_h, device="cuda")
    wx = torch.randn(b, t, 3 * d_h, device="cuda")
    a_z = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.05
    a_r = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.05
    a_n = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.05
    mu = torch.randn(b, t, d_h, device="cuda")
    e = gru_head_recurrence_vjp_eager(
        h_prev, wx, a_z, a_r, a_n, mu, n_heads=n_heads, d_head=d_head
    )
    k = gru_head_recurrence_vjp(
        h_prev, wx, a_z, a_r, a_n, mu, n_heads=n_heads, d_head=d_head
    )
    for a, b_ in zip(e, k, strict=True):
        torch.testing.assert_close(a, b_, atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_reverse_tiled_matches_eager():
    from pararnn.kernels.newton_gru_head import (
        _factor_reverse_eager,
        _factor_reverse_tiled_triton,
        _gates,
    )

    torch.manual_seed(421)
    n_heads, d_head = 2, 192
    d_h = n_heads * d_head
    b, t = 2, 9
    h_prev = torch.randn(b, t, d_h, device="cuda")
    wx = torch.randn(b, t, 3 * d_h, device="cuda")
    a_z = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    a_r = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    a_n = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    partial = torch.randn(b, t, d_h, device="cuda")
    _, gates = _gates(
        h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
    )
    eager = _factor_reverse_eager(
        gates, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
    )
    tri = _factor_reverse_tiled_triton(
        gates, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
    )
    torch.testing.assert_close(eager, tri, atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
@pytest.mark.parametrize("d_head,n_heads", [(96, 2), (128, 2)])
def test_paragru_head_reverse_stream_a_matches_eager(d_head: int, n_heads: int):
    from pararnn.kernels.newton_gru_head import (
        _factor_reverse_eager,
        _gates,
        _reverse_gru_head_factor_impl,
    )

    torch.manual_seed(422 + d_head)
    d_h = n_heads * d_head
    b, t = 2, 17
    h_prev = torch.randn(b, t, d_h, device="cuda")
    wx = torch.randn(b, t, 3 * d_h, device="cuda")
    a_z = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    a_r = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    a_n = torch.randn(n_heads, d_head, d_head, device="cuda") * 0.04
    partial = torch.randn(b, t, d_h, device="cuda")
    _, gates = _gates(
        h_prev, wx, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head, d_h=d_h
    )
    eager = _factor_reverse_eager(
        gates, partial, a_z, a_r, a_n, n_heads=n_heads, d_head=d_head
    )
    tri = _reverse_gru_head_factor_impl(h_prev, wx, partial, a_z, a_r, a_n)
    torch.testing.assert_close(eager, tri, atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
def test_paragru_head_fused_bwd_matches_sequential_bptt():
    torch.manual_seed(413)
    d_in, d_h, t = 16, 32, 16
    x = 0.2 * torch.randn(2, t, d_in, device="cuda")
    w = torch.randn(2, t, d_h, device="cuda")
    cell_s = ParaGRU(d_in, d_h, mix="head", n_heads=4, device="cuda")
    cell_n = ParaGRU(d_in, d_h, mix="head", n_heads=4, device="cuda")
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    cfg = NewtonConfig(max_iters=5, scan_backend="fused", residual_atol=None)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x_n, cfg) * w).sum()
    loss_n.backward()
    for (n, p_a), (_, p_b) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert p_a.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_scan_dense_triton_matches_eager():
    from pararnn.solvers.scan import scan_dense

    torch.manual_seed(403)
    b, t, d = 3, 19, 16
    jac = torch.randn(b, t, d, d, device="cuda") * 0.15
    res = torch.randn(b, t, d, device="cuda")
    eager = scan_dense(jac, res, backend="eager")
    tri = scan_dense(jac, res, backend="triton")
    torch.testing.assert_close(eager, tri, atol=1e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_newton_triton_vs_sequential():
    torch.manual_seed(404)
    cell = ParaGRU(d_in=16, d_h=64, mix="head", n_heads=8, device="cuda")
    x = 0.2 * torch.randn(2, 32, 16, device="cuda")
    ref = sequential_apply(cell, x)
    h = newton_apply(
        cell, x, NewtonConfig(max_iters=4, scan_backend="triton", residual_atol=None)
    )
    err = float((h - ref).abs().amax())
    assert err < 5e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_large_d_head_scan_parity():
    """d_head=256 uses the same tiled row scan as d_head=8/64."""
    from pararnn.solvers.scan import scan_dense

    torch.manual_seed(406)
    b, t, d = 2, 17, 256
    jac = torch.randn(b, t, d, d, device="cuda") * 0.05
    res = torch.randn(b, t, d, device="cuda")
    eager = scan_dense(jac, res, backend="eager")
    tri = scan_dense(jac, res, backend="triton")
    torch.testing.assert_close(eager, tri, atol=2e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_paragru_head_bf16_smoke():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported")
    torch.manual_seed(405)
    cell = ParaGRU(d_in=16, d_h=32, mix="head", n_heads=4, device="cuda", dtype=torch.bfloat16)
    x = 0.2 * torch.randn(2, 24, 16, device="cuda", dtype=torch.bfloat16)
    ref = sequential_apply(cell, x)
    h = newton_apply(
        cell, x, NewtonConfig(max_iters=4, scan_backend="triton", residual_atol=None)
    )
    err = float((h.float() - ref.float()).abs().amax())
    assert err < 5e-2, err
