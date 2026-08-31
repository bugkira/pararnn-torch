"""Para-sLSTM: sequential unroll vs Newton + 4×4 (diag) / dense (mix) scan.

K=3 / 1e-6 is a ParaGRU/LSTM measurement, not a gate. These tests record
residual vs K. Fused Triton is out of scope.
"""

from __future__ import annotations

import logging

import torch

from pararnn import (
    NewtonConfig,
    NewtonStats,
    ParaSLSTM,
    device,
    newton_apply,
    sequential_apply,
)
from pararnn.layout import SLSTM_HIDDEN, SLSTM_SLOTS
from pararnn.solvers.scan import reverse_scan_block4, scan_block4, scan_dense

log = logging.getLogger(__name__)


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
                jacobian="autograd",
                jac_structure=jac_structure,
            ),
            stats=st,
        )
        err = float((par - seq).abs().amax())
        log.info(
            "slstm_newton_k k=%s omega=%.2f residual=%.3e seq_err=%.3e seq_len=%s d_h=%s mix=%s clip=%s",
            k,
            omega,
            st.max_residual,
            err,
            x.shape[1],
            cell.d_h,
            cell.mix,
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
def test_slstm_diag_newton_vs_sequential():
    torch.manual_seed(101)
    cell = ParaSLSTM(d_in=4, d_h=4, mix="diag").to(device)
    x = 0.3 * torch.randn(2, 12, 4, device=device)
    rows = _residual_vs_k(cell, x)
    # Diag mix, seed 101: K=1..3 overshoot, K=4 snaps. Do not copy ParaGRU's K=3.
    err_k5 = rows[-1][2]
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
    tri = scan_block4(jac, residual, backend="triton")
    ref = torch.zeros_like(residual)
    ref[:, 0] = residual[:, 0]
    for s in range(1, t):
        ref[:, s] = torch.einsum("boid,bid->bod", jac[:, s], ref[:, s - 1]) + residual[:, s]
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tri, got, atol=1e-5, rtol=1e-5)


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
    # omega=0.5 cuts K=3 overshoot (~12 → ~4) but kills the K=4 snap
    # (err stays ~2). Clip 0.25 is a no-op vs 0.5 on this seed.
    # Prototype: K=4, omega=1, clip=0.5. Do not change NewtonConfig defaults.
    assert err_k4_full < 1e-4, rows_1
    assert err_k3_damp < err_k3_full, (err_k3_damp, err_k3_full)
    assert rows_h[3][2] > 0.1, rows_h
    assert err_k4_clip < 1e-4, rows_c


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
    for (n, p_a), (_, p_b) in zip(cell_s.named_parameters(), cell_n.named_parameters()):
        assert p_a.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=5e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=5e-4, rtol=1e-4)
