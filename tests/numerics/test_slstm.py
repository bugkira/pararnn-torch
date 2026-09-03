"""Para-sLSTM: sequential unroll vs Newton + 4x4 (diag) / dense (mix) scan.

K=3 / 1e-6 is a ParaGRU/LSTM measurement, not a gate. These tests record
residual vs K. Fused Newton is mix='diag' only (Triton 4x4); head/dense stay
eager/triton-scan.
"""

from __future__ import annotations

import logging

import pytest
import torch

from pararnn import NewtonConfig, ParaGRU, ParaSLSTM
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    slstm_pack_heads,
    slstm_unpack_heads,
)
from pararnn.solvers import (
    NewtonStats,
    newton_apply,
    sequential_apply,
)
from pararnn.solvers.jacobian import jacobian_autograd
from pararnn.solvers.newton import slstm_auto_picard, slstm_picard_next
from pararnn.solvers.scan import reverse_scan_block4, scan_block4, scan_dense
from pararnn.solvers.slstm_log import (
    SLSTMLogCoords,
    slstm_decode_log,
    slstm_encode_log,
)
from pararnn.solvers.slstm_picard import (
    slstm_frozen_gate_scan,
    slstm_frozen_gate_scan_eager,
    slstm_picard_init,
)

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _residual_vs_k(
    cell,
    x,
    ks: tuple[int, ...] = (1, 2, 3, 4, 5),
    *,
    omega: float = 1.0,
    jac_structure: str | None = None,
) -> list[tuple[int, float, float]]:
    seq = sequential_apply(cell, x)
    rows = []
    for k in ks:
        st = NewtonStats()
        par = newton_apply(
            cell,
            x,
            NewtonConfig(
                max_iters=k,
                omega=omega,
                scan_backend="eager",
                residual_atol=None,
                residual_fail=None,
                jacobian="autograd",
                jac_structure=jac_structure,
            ),
            stats=st,
        )
        err = float((par - seq).abs().amax())
        log.info(
            "slstm_newton_k k=%s omega=%.2f residual=%.3e seq_err=%.3e "
            "seq_len=%s d_h=%s mix=%s n_heads=%s clip=%s",
            k,
            omega,
            st.max_residual,
            err,
            x.shape[1],
            cell.d_h,
            cell.mix,
            cell.n_heads,
            cell.max_recurrent_norm,
        )
        rows.append((k, st.max_residual, err))
    return rows


@torch.no_grad()
def test_slstm_step_shape_and_slots():
    torch.manual_seed(100)
    cell = ParaSLSTM(d_in=5, d_h=6).to(device)
    assert cell.state_slots == SLSTM_SLOTS
    assert cell.jac_structure == "block4"
    h = torch.zeros(3, SLSTM_SLOTS, 6, device=device)
    x = torch.randn(3, 5, device=device)
    out = cell.step(h, x)
    assert out.shape == (3, SLSTM_SLOTS, 6)
    wx = cell.W_x(x)
    torch.testing.assert_close(cell.step(h, x), cell.step(h, x, wx=wx))


@torch.no_grad()
def test_slstm_zero_hidden_init_matches_forced_h0_loop():
    """Max-plus ``m`` + diag scans vs the O(T) ``h_prev=0`` unroll."""
    torch.manual_seed(230)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 48, 4, device=device)
    wx = cell.W_x(x)
    got = cell.zero_hidden_init(x, wx=wx)
    state = torch.zeros(2, SLSTM_SLOTS, 4, device=device)
    parts = []
    for t in range(x.shape[1]):
        prev = state.clone()
        prev[:, SLSTM_HIDDEN] = 0
        state = cell.step(prev, x[:, t], wx=wx[:, t])
        parts.append(state)
    ref = torch.stack(parts, dim=1)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    seq = sequential_apply(cell, x)
    # Better than App. A, not the sequential root (mixing still missing).
    assert float((got - seq).abs().amax()) < 10.0


@torch.no_grad()
def test_slstm_diag_newton_vs_sequential():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    rows = _residual_vs_k(cell, x)
    err_k3 = rows[2][2]
    err_k5 = rows[-1][2]
    # Zero-hidden init: K=3 is in the basin (App. A needed K=4 on this seed).
    assert err_k3 < 2e-3, rows
    assert err_k5 < 2e-3, rows
    assert min(err for _, _, err in rows) < 1e-4, rows


@torch.no_grad()
def test_slstm_dense_mix_newton_vs_sequential():
    torch.manual_seed(102)
    cell = ParaSLSTM(d_in=3, d_h=3, mix="dense").to(device)
    assert cell.jac_structure == "dense"
    x = 0.3 * torch.randn(2, 8, 3, device=device)
    rows = _residual_vs_k(cell, x)
    err_k5 = rows[-1][2]
    assert err_k5 < 5e-3, rows


@torch.no_grad()
def test_slstm_h0_and_hidden_slot():
    torch.manual_seed(103)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 7, 4, device=device)
    h0 = 0.2 * torch.randn(2, SLSTM_SLOTS, 4, device=device)
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None),
        h0=h0,
    )
    err = (par - seq).abs().amax()
    assert err < 2e-3, float(err)
    assert cell.hidden_slot == SLSTM_HIDDEN
    assert seq[:, :, SLSTM_HIDDEN, :].shape == (2, 7, 4)


def test_scan_block4_matches_forward_substitution():
    torch.manual_seed(201)
    # d == 4 is the layout that used to trip _is_dense (S=d).
    b, t, d = 2, 9, 4
    jac = torch.randn(b, t, 4, 4, d, device=device) * 0.15
    residual = torch.randn(b, t, 4, d, device=device)
    got = scan_block4(jac, residual)
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        ref[:, s] = torch.einsum("boid,bid->bod", jac[:, s], ref[:, s - 1]) + residual[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_triton_scan_block4_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(201)
    b, t, d = 2, 9, 4
    jac = torch.randn(b, t, 4, 4, d, device=cuda_device) * 0.15
    residual = torch.randn(b, t, 4, d, device=cuda_device)
    got = scan_block4(jac, residual)
    tri = scan_block4(jac, residual, backend="triton")
    torch.testing.assert_close(tri, got, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_triton_scan_block4_matches_eager_long(cuda_device: torch.device) -> None:
    torch.manual_seed(209)
    jac = torch.randn(2, 200, 4, 4, 5, device=cuda_device) * 0.12
    residual = torch.randn(2, 200, 4, 5, device=cuda_device)
    eager = scan_block4(jac, residual)
    tri = scan_block4(jac, residual, backend="triton")
    torch.testing.assert_close(tri, eager, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_triton_scan_block4_tile_boundaries(cuda_device: torch.device) -> None:
    """BLOCK_T=32: lengths that sit inside, on, and over a tile."""
    torch.manual_seed(210)
    for t in (9, 32, 33, 64, 128):
        jac = torch.randn(2, t, 4, 4, 6, device=cuda_device) * 0.12
        residual = torch.randn(2, t, 4, 6, device=cuda_device)
        eager = scan_block4(jac, residual)
        tri = scan_block4(jac, residual, backend="triton")
        torch.testing.assert_close(tri, eager, atol=1e-5, rtol=1e-5)


def test_reverse_scan_block4_matches_backward_substitution():
    torch.manual_seed(202)
    b, t, d = 2, 7, 4
    jac = torch.randn(b, t, 4, 4, d, device=device) * 0.15
    partial = torch.randn(b, t, 4, d, device=device)
    got = reverse_scan_block4(jac, partial)
    ref = torch.zeros_like(partial)
    ref[:, -1] = partial[:, -1]
    for s in range(t - 2, -1, -1):
        j_t = jac[:, s + 1].transpose(-3, -2)
        ref[:, s] = torch.einsum("boid,bid->bod", j_t, ref[:, s + 1]) + partial[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.cuda
def test_triton_reverse_scan_block4_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(202)
    b, t, d = 2, 7, 4
    jac = torch.randn(b, t, 4, 4, d, device=cuda_device) * 0.15
    partial = torch.randn(b, t, 4, d, device=cuda_device)
    got = reverse_scan_block4(jac, partial)
    tri = reverse_scan_block4(jac, partial, backend="triton")
    torch.testing.assert_close(tri, got, atol=1e-5, rtol=1e-5)


def test_scan_block4_matches_dense_blockdiag():
    torch.manual_seed(203)
    b, t, d = 2, 8, 4
    jac = torch.randn(b, t, 4, 4, d, device=device) * 0.15
    residual = torch.randn(b, t, 4, d, device=device)
    got = scan_block4(jac, residual)
    sd = 4 * d
    dense = jac.new_zeros(b, t, sd, sd)
    for out in range(4):
        for inn in range(4):
            dense[:, :, out * d : (out + 1) * d, inn * d : (inn + 1) * d] = torch.diag_embed(
                jac[:, :, out, inn, :]
            )
    packed = scan_dense(dense, residual.reshape(b, t, sd))
    torch.testing.assert_close(got.reshape(b, t, sd), packed, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_slstm_diag_block4_matches_forced_dense():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    cfg = {
        "max_iters": 5,
        "scan_backend": "eager",
        "residual_atol": None,
        "jacobian": "autograd",
    }
    h4 = newton_apply(cell, x, NewtonConfig(**cfg))
    hd = newton_apply(cell, x, NewtonConfig(**cfg, jac_structure="dense"))
    torch.testing.assert_close(h4, hd, atol=2e-5, rtol=1e-5)


@torch.no_grad()
def test_slstm_omega_and_clip():
    """Gonzalez ELK-style damping vs App. C.1 clip. Do not change global Newton defaults."""
    torch.manual_seed(101)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    cell_tight = ParaSLSTM(d_in=4, d_h=4, mix="diag", max_recurrent_norm=0.25).to(device)
    cell_tight.load_state_dict(cell.state_dict())
    rows_1 = _residual_vs_k(cell, x, omega=1.0)
    rows_h = _residual_vs_k(cell, x, omega=0.5)
    rows_c = _residual_vs_k(cell_tight, x, omega=1.0)
    err_k3_full = rows_1[2][2]
    err_k3_damp = rows_h[2][2]
    err_k4_full = rows_1[3][2]
    err_k4_clip = rows_c[3][2]
    log.info(
        "slstm_damp k3_omega1=%.3e k3_omega05=%.3e k4_omega1=%.3e k4_clip025=%.3e",
        err_k3_full,
        err_k3_damp,
        err_k4_full,
        err_k4_clip,
    )
    # omega=0.5 still damps relative to omega=1. Do not copy ELK as default.
    assert err_k4_full < 1e-4, rows_1
    assert err_k4_clip < 1e-4, rows_c
    assert err_k3_full < 2e-3, rows_1


def test_slstm_newton_bwd_matches_sequential_bptt():
    torch.manual_seed(204)
    d_in, d_h, t = 4, 4, 8
    x = 0.3 * torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, SLSTM_SLOTS, d_h, device=device)
    cell_s = ParaSLSTM(d_in, d_h, mix="diag").to(device)
    cell_n = ParaSLSTM(d_in, d_h, mix="diag").to(device)
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


def test_slstm_newton_h0_grad_matches_sequential_bptt():
    torch.manual_seed(2041)
    d_in, d_h, t = 4, 4, 8
    x = 0.3 * torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, SLSTM_SLOTS, d_h, device=device)
    cell_s = ParaSLSTM(d_in, d_h, mix="diag").to(device)
    cell_n = ParaSLSTM(d_in, d_h, mix="diag").to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    h0 = 0.2 * torch.randn(2, SLSTM_SLOTS, d_h, device=device)
    h0_s = h0.clone().requires_grad_(True)
    h0_n = h0.clone().requires_grad_(True)
    cfg = NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None, picard_iters=1)
    loss_s = (sequential_apply(cell_s, x, h0_s) * w).sum()
    loss_s.backward()
    loss_n = (newton_apply(cell_n, x, cfg, h0=h0_n) * w).sum()
    loss_n.backward()
    torch.testing.assert_close(h0_s.grad, h0_n.grad, atol=5e-4, rtol=1e-4)


def test_slstm_pack_heads_roundtrip():
    torch.manual_seed(205)
    state = torch.randn(2, 5, 4, 6, device=device)
    packed = slstm_pack_heads(state, 3, 2)
    assert packed.shape == (2, 5, 3, 8)
    torch.testing.assert_close(slstm_unpack_heads(packed, 3, 2), state)


def test_slstm_head_needs_dividing_n_heads():
    with pytest.raises(ValueError, match="n_heads"):
        ParaSLSTM(d_in=4, d_h=4, mix="head")
    with pytest.raises(ValueError, match="n_heads"):
        ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=3)


@torch.no_grad()
def test_slstm_head_newton_vs_sequential():
    torch.manual_seed(105)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=2).to(device)
    assert cell.jac_structure == "head"
    assert cell.d_head == 2
    x = 0.3 * torch.randn(2, 8, 4, device=device)
    rows = _residual_vs_k(cell, x)
    err_k5 = rows[-1][2]
    assert err_k5 < 5e-3, rows
    assert min(err for _, _, err in rows) < 1e-4, rows


@torch.no_grad()
def test_slstm_head_matches_forced_dense():
    torch.manual_seed(105)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=2).to(device)
    x = 0.3 * torch.randn(2, 8, 4, device=device)
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
def test_slstm_step_head_matches_step():
    torch.manual_seed(207)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=2).to(device)
    state = torch.randn(3, SLSTM_SLOTS, 4, device=device)
    x = 0.3 * torch.randn(3, 4, device=device)
    out = cell.step(state, x)
    wx_slots = cell.W_x(x).reshape(3, 4, 4)
    packed_s = slstm_pack_heads(state.unsqueeze(1), 2, 2)[:, 0]
    packed_w = slstm_pack_heads(wx_slots.unsqueeze(1), 2, 2)[:, 0]
    r = cell.clipped_r_head()
    parts = []
    for b in range(3):
        heads = [
            cell.step_head(
                packed_s[b, hd].reshape(4, 2),
                packed_w[b, hd].reshape(4, 2),
                r[:, hd],
            )
            for hd in range(2)
        ]
        parts.append(torch.stack(heads).reshape(2, 8))
    packed_out = torch.stack(parts, dim=0).unsqueeze(1)
    torch.testing.assert_close(slstm_unpack_heads(packed_out, 2, 2)[:, 0], out)


def test_slstm_head_newton_bwd_matches_sequential_bptt():
    torch.manual_seed(206)
    d_in, d_h, t = 4, 4, 8
    x = 0.3 * torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, SLSTM_SLOTS, d_h, device=device)
    cell_s = ParaSLSTM(d_in, d_h, mix="head", n_heads=2).to(device)
    cell_n = ParaSLSTM(d_in, d_h, mix="head", n_heads=2).to(device)
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
def test_slstm_analytic_jac_matches_autograd():
    torch.manual_seed(208)
    specs = (
        (ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device), "block4", 4),
        (ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=2).to(device), "head", 4),
        (ParaSLSTM(d_in=3, d_h=3, mix="dense").to(device), "dense", 3),
    )
    for cell, structure, d_h in specs:
        state = torch.randn(2, 7, SLSTM_SLOTS, d_h, device=device)
        x = 0.3 * torch.randn(2, 7, cell.d_in, device=device)
        pred_a, jac_a = cell.step_with_jacobian(state, x)
        pred_g, jac_g = jacobian_autograd(cell, state, x, structure=structure)
        torch.testing.assert_close(pred_g, pred_a, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(jac_g, jac_a, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_slstm_analytic_newton_matches_sequential():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=5,
            scan_backend="eager",
            residual_atol=None,
            jacobian="analytic",
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_diag_auto_picks_fused(cuda_device: torch.device) -> None:
    torch.manual_seed(220)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 12, 4, device=cuda_device)
    st = NewtonStats()
    newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=5, residual_atol=None),
        stats=st,
    )
    assert st.scan_backend == "fused"
    assert st.picard_iters == 1


@pytest.mark.cuda
def test_slstm_fused_rejects_head_and_dense(cuda_device: torch.device) -> None:
    x = 0.3 * torch.randn(2, 8, 4, device=cuda_device)
    head = ParaSLSTM(d_in=4, d_h=4, mix="head", n_heads=2).to(cuda_device)
    dense = ParaSLSTM(d_in=4, d_h=4, mix="dense").to(cuda_device)
    with pytest.raises(TypeError, match="diag"):
        newton_apply(head, x, NewtonConfig(max_iters=1, scan_backend="fused"))
    with pytest.raises(TypeError, match="diag"):
        newton_apply(dense, x, NewtonConfig(max_iters=1, scan_backend="fused"))


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_newton_fused_matches_sequential(cuda_device: torch.device) -> None:
    """Seed 101 / d_h=4 snaps at K=4. Do not copy ParaGRU's K=3 or long-T vs seq."""
    torch.manual_seed(101)
    cfg = NewtonConfig(max_iters=5, scan_backend="fused", residual_atol=None)
    eager_cfg = NewtonConfig(max_iters=5, scan_backend="eager", residual_atol=None)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 12, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    eager = newton_apply(cell, x, eager_cfg)
    par = newton_apply(cell, x, cfg)
    err_s = float((par - seq).abs().amax())
    err_e = float((par - eager).abs().amax())
    assert err_s < 2e-3, err_s
    assert err_e < 2e-4, err_e


@torch.no_grad()
def test_slstm_diag_long_t_snaps_at_k3():
    """T=48 snaps at K=3 with zero-hidden init. Library default K=3 is enough.

    App. A ``f(0, x_t)`` needed K=12 here (running ``n``). See
    ``docs/internal/para-slstm.md``. Do not raise the global Newton default for GRU.
    """
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 48, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, scan_backend="eager", residual_atol=None),
    )
    err3 = float((par - seq).abs().amax())
    assert err3 < 1e-4, err3


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_newton_fused_matches_eager_across_tiles(cuda_device: torch.device) -> None:
    """T=48 tile crossing. Kernel check is fused vs eager; K=3 vs seq."""
    torch.manual_seed(101)
    cfg = NewtonConfig(max_iters=3, scan_backend="fused", residual_atol=None)
    eager_cfg = NewtonConfig(max_iters=3, scan_backend="eager", residual_atol=None)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 48, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    eager = newton_apply(cell, x, eager_cfg)
    par = newton_apply(cell, x, cfg)
    err_e = float((par - eager).abs().amax())
    err_s = float((par - seq).abs().amax())
    assert err_e < 2e-4, err_e
    assert err_s < 1e-4, err_s


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_newton_fused_h0_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(222)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 20, 4, device=cuda_device)
    h0 = 0.2 * torch.randn(2, SLSTM_SLOTS, 4, device=cuda_device)
    seq = sequential_apply(cell, x, h0)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=5, scan_backend="fused", residual_atol=None),
        h0=h0,
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
def test_slstm_newton_fused_bwd_matches_sequential_bptt(cuda_device: torch.device) -> None:
    torch.manual_seed(223)
    d_in, d_h, t = 4, 4, 8
    x = 0.3 * torch.randn(2, t, d_in, device=cuda_device)
    w = torch.randn(2, t, SLSTM_SLOTS, d_h, device=cuda_device)
    cell_s = ParaSLSTM(d_in, d_h, mix="diag").to(cuda_device)
    cell_n = ParaSLSTM(d_in, d_h, mix="diag").to(cuda_device)
    cell_n.load_state_dict(cell_s.state_dict())
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    loss_s = (sequential_apply(cell_s, x_s) * w).sum()
    loss_s.backward()
    loss_n = (
        newton_apply(
            cell_n,
            x_n,
            NewtonConfig(max_iters=5, scan_backend="fused", residual_atol=None),
        )
        * w
    ).sum()
    loss_n.backward()
    for (n, p_a), (_, p_b) in zip(
        cell_s.named_parameters(), cell_n.named_parameters(), strict=True
    ):
        assert p_a.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_newton_fused_fp16_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(224)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device=cuda_device, dtype=torch.float16)
    x = (0.3 * torch.randn(2, 12, 4, device=cuda_device)).to(torch.float16)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=5, scan_backend="fused", residual_atol=None),
    )
    err = float((par.float() - seq.float()).abs().amax())
    assert err < 2e-2, err
    assert par.dtype is torch.float16


@torch.no_grad()
def test_slstm_log_coords_roundtrip():
    torch.manual_seed(301)
    eps = 1e-6
    state = torch.zeros(2, SLSTM_SLOTS, 5, device=device)
    state[:, SLSTM_CELL] = torch.tensor(
        [[-2.0, -0.1, 0.0, 0.5, 3.0], [1.0, -4.0, 0.2, -0.01, 8.0]], device=device
    )
    state[:, SLSTM_NORMALIZER] = torch.tensor(
        [[2.5, 1.0, 0.3, 12.0, 80.0], [1.0, 5.0, 4.0, 0.05, 2.5]], device=device
    )
    state[:, SLSTM_HIDDEN] = 0.2 * torch.randn(2, 5, device=device)
    got = slstm_decode_log(slstm_encode_log(state, eps=eps), eps=eps)
    torch.testing.assert_close(got, state, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_slstm_log_decode_huge_n_is_finite():
    u = torch.zeros(1, SLSTM_SLOTS, 2, device=device)
    u[:, SLSTM_CELL] = 200.0
    u[:, SLSTM_NORMALIZER] = 200.0
    out = slstm_decode_log(u, eps=1e-6)
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_slstm_log_jac_matches_autograd():
    torch.manual_seed(302)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    wrapped = SLSTMLogCoords(cell)
    native = torch.randn(2, 6, SLSTM_SLOTS, 4, device=device)
    native[:, :, SLSTM_NORMALIZER] = native[:, :, SLSTM_NORMALIZER].abs() + 1.0
    native[:, :, SLSTM_CELL] = torch.tanh(native[:, :, SLSTM_CELL])
    coords = slstm_encode_log(native, eps=cell.eps)
    x = 0.3 * torch.randn(2, 6, 4, device=device)
    pred_a, jac_a = wrapped.step_with_jacobian(coords, x)
    pred_g, jac_g = jacobian_autograd(wrapped, coords, x, structure="block4")
    torch.testing.assert_close(pred_g, pred_a, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(jac_g, jac_a, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_slstm_log_newton_vs_sequential():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="eager",
            residual_atol=None,
            coords="log",
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@torch.no_grad()
def test_slstm_log_coords_gru_rejected():
    gru = ParaGRU(4, 4).to(device)
    xg = torch.randn(2, 4, 4, device=device)
    with pytest.raises(TypeError, match="ParaSLSTM"):
        newton_apply(gru, xg, NewtonConfig(coords="log", scan_backend="eager"))


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_log_auto_picks_fused(cuda_device: torch.device) -> None:
    torch.manual_seed(220)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 12, 4, device=cuda_device)
    st = NewtonStats()
    newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, residual_atol=None, coords="log"),
        stats=st,
    )
    assert st.scan_backend == "fused"


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_log_fused_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 12, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    cfg = {"max_iters": 3, "residual_atol": None, "coords": "log"}
    eager = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="eager"))
    fused = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="fused"))
    torch.testing.assert_close(fused, eager, atol=2e-4, rtol=2e-4)
    err = float((fused - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_log_fused_matches_eager_across_tiles(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 48, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    cfg = {"max_iters": 3, "residual_atol": None, "coords": "log"}
    eager = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="eager"))
    fused = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="fused"))
    err_e = float((fused - eager).abs().amax())
    err_s = float((fused - seq).abs().amax())
    assert err_e < 2e-4, err_e
    assert err_s < 1e-4, err_s


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_fused_chunked_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 96, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="fused",
            residual_atol=None,
            chunk_len=64,
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_fused_log_chunked_matches_sequential(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 96, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="fused",
            residual_atol=None,
            chunk_len=64,
            coords="log",
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@torch.no_grad()
def test_slstm_log_step_matches_native_ratio():
    """LSE / convex combo ≈ encode(native(decode)) when n ≫ eps."""
    torch.manual_seed(303)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    wrapped = SLSTMLogCoords(cell)
    native = torch.randn(2, 5, SLSTM_SLOTS, 4, device=device)
    native[:, :, SLSTM_NORMALIZER] = native[:, :, SLSTM_NORMALIZER].abs() + 1.0
    native[:, :, SLSTM_CELL] = torch.tanh(native[:, :, SLSTM_CELL])
    coords = slstm_encode_log(native, eps=cell.eps)
    x = 0.3 * torch.randn(2, 5, 4, device=device)
    got = wrapped.step(coords, x)
    ref = slstm_encode_log(cell.step(slstm_decode_log(coords, eps=cell.eps), x), eps=cell.eps)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)


@torch.no_grad()
def test_slstm_chunked_newton_vs_sequential():
    """T=64 snaps at this width on seed 0; two chunks should still match seq."""
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 96, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="eager",
            residual_atol=None,
            chunk_len=64,
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_frozen_gate_triton_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(311)
    cell = ParaSLSTM(d_in=8, d_h=8, mix="diag").to(cuda_device)
    from pararnn.layout import prepend_state

    for T in (12, 48, 200):
        x = 0.3 * torch.randn(2, T, 8, device=cuda_device)
        wx = cell.W_x(x)
        h0 = 0.1 * torch.randn(2, 4, 8, device=cuda_device)
        eager = slstm_frozen_gate_scan_eager(wx, eps=cell.eps, h0=h0)
        fused = slstm_frozen_gate_scan(wx, eps=cell.eps, h0=h0)
        torch.testing.assert_close(fused, eager, atol=2e-4, rtol=2e-4)
        p3 = slstm_picard_init(cell, wx, h0=h0, n_picard=3)
        states = slstm_frozen_gate_scan_eager(wx, eps=cell.eps, h0=h0)
        for _ in range(3):
            h_prev = prepend_state(states, h0)[..., SLSTM_HIDDEN, :]
            pre = wx + cell._recurrent(h_prev)
            states = slstm_frozen_gate_scan_eager(pre, eps=cell.eps, h0=h0)
        torch.testing.assert_close(p3, states, atol=5e-4, rtol=5e-4)


@torch.no_grad()
def test_slstm_picard_p0_matches_zero_hidden():
    torch.manual_seed(310)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 16, 4, device=device)
    wx = cell.W_x(x)
    z = cell.zero_hidden_init(x, wx=wx)
    p0 = slstm_picard_init(cell, wx, n_picard=0)
    torch.testing.assert_close(p0, z)
    p1 = slstm_picard_init(cell, wx, n_picard=1)
    assert float((p1 - z).abs().amax()) > 1e-4


@torch.no_grad()
def test_slstm_picard_newton_vs_sequential():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="eager",
            residual_atol=None,
            picard_iters=1,
        ),
    )
    err = float((par - seq).abs().amax())
    assert err < 2e-3, err


@torch.no_grad()
def test_slstm_picard_gru_rejected():
    gru = ParaGRU(4, 4).to(device)
    xg = torch.randn(2, 4, 4, device=device)
    with pytest.raises(TypeError, match="ParaSLSTM"):
        newton_apply(gru, xg, NewtonConfig(picard_iters=1, scan_backend="eager"))


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_picard_fused_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 12, 4, device=cuda_device)
    seq = sequential_apply(cell, x)
    cfg = {"max_iters": 3, "residual_atol": None, "picard_iters": 1}
    eager = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="eager"))
    fused = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="fused"))
    torch.testing.assert_close(fused, eager, atol=2e-4, rtol=2e-4)
    err = float((fused - seq).abs().amax())
    assert err < 2e-3, err


@pytest.mark.cuda
@torch.no_grad()
def test_slstm_scan_seq_fused_matches_assoc(cuda_device: torch.device) -> None:
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(cuda_device)
    x = 0.3 * torch.randn(2, 48, 4, device=cuda_device)
    cfg = {
        "max_iters": 3,
        "residual_atol": None,
        "picard_iters": 1,
    }
    assoc = newton_apply(cell, x, NewtonConfig(**cfg, scan_backend="fused"))
    seqt = newton_apply(
        cell,
        x,
        NewtonConfig(**cfg, scan_backend="fused", scan_tile="seq"),
    )
    torch.testing.assert_close(seqt, assoc, atol=2e-4, rtol=2e-4)


def test_slstm_auto_picard_schedule():
    assert slstm_auto_picard(12) == 1
    assert slstm_auto_picard(64) == 1
    assert slstm_auto_picard(65) == 3
    assert slstm_auto_picard(256) == 3
    assert slstm_auto_picard(2048) == 3
    assert slstm_auto_picard(2049) == 5
    assert slstm_auto_picard(4096) == 5
    assert slstm_auto_picard(16384) == 5


@torch.no_grad()
def test_slstm_auto_picard_default_snaps():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    st = NewtonStats()
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=3, scan_backend="eager", residual_atol=None),
        stats=st,
    )
    assert st.picard_iters == 1
    err = float((par - sequential_apply(cell, x)).abs().amax())
    assert err < 2e-3, err


def test_slstm_picard_next_rungs():
    assert slstm_picard_next(0) == 1
    assert slstm_picard_next(1) == 3
    assert slstm_picard_next(2) == 3
    assert slstm_picard_next(3) == 5
    assert slstm_picard_next(5) is None
    assert slstm_picard_next(9) is None


@torch.no_grad()
def test_slstm_picard_adapt_climbs_on_far_guess():
    """P=1 at T=256 is below the auto rung; adapt should raise P, not K.

    Init-scale table in para-slstm.md: T=256 K=3 without enough Picard is
    outside the sequential basin. Explicit P=1 + picard_adapt=True is the
    train-diag path (docs/internal/next.md). Fallback if this seed snaps at P=1:
    the assert on seq err still holds.
    """
    torch.manual_seed(0)
    cell = ParaSLSTM(d_in=8, d_h=8, mix="diag").to(device)
    x = torch.randn(2, 256, 8, device=device)
    st = NewtonStats()
    par = newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="eager",
            picard_iters=1,
            picard_adapt=True,
        ),
        stats=st,
    )
    assert st.picard_iters in (1, 3, 5)
    if st.max_residual > 1e-3:
        assert st.picard_iters >= 3
    err = float((par - sequential_apply(cell, x)).abs().amax())
    assert err < 2e-3, err


@torch.no_grad()
def test_slstm_explicit_p_skips_adapt():
    torch.manual_seed(0)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    st = NewtonStats()
    newton_apply(
        cell,
        x,
        NewtonConfig(
            max_iters=3,
            scan_backend="eager",
            picard_iters=1,
            picard_adapt=False,
        ),
        stats=st,
    )
    assert st.picard_iters == 1
