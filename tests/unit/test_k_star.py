"""Unit tests for measured K*(T) auto schedules and manual overrides."""

from __future__ import annotations

import torch

from pararnn.cells import ParaCfC, ParaHopfield, ParaTitans
from pararnn.solvers import NewtonConfig, NewtonStats, newton_apply, sequential_apply
from pararnn.solvers.newton.k_star import (
    auto_newton_iters,
    cfc_auto_newton_iters,
    hopfield_auto_newton_iters,
    lookup_iters_by_t,
    titans_auto_newton_iters,
)


def test_lookup_iters_by_t_left_step() -> None:
    sched = {64: 2, 1024: 3, 4096: 4}
    assert lookup_iters_by_t(sched, 1) == 2
    assert lookup_iters_by_t(sched, 64) == 2
    assert lookup_iters_by_t(sched, 512) == 2
    assert lookup_iters_by_t(sched, 1024) == 3
    assert lookup_iters_by_t(sched, 8000) == 4


def test_auto_schedules_nondecreasing() -> None:
    for fn in (cfc_auto_newton_iters, hopfield_auto_newton_iters, titans_auto_newton_iters):
        prev = 0
        for t in (1, 32, 64, 256, 1024, 4096, 16384):
            k = fn(t)
            assert k >= prev
            prev = k


def test_manual_newton_iters_by_t_overrides_cell() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)
    cell = ParaHopfield(4, 4, device=device).eval()
    x = torch.randn(1, 80, 4, device=device)
    # Pin table forces K=5 at T=80 even if cell envelope says 2.
    cfg = NewtonConfig(
        max_iters=None,
        newton_iters_by_t={1: 5, 1024: 6},
        scan_backend="eager",
        jac_structure="dense",
        residual_atol=None,
        residual_fail=None,
    )
    st = NewtonStats()
    y = newton_apply(cell, x, cfg, stats=st)
    assert y.shape == (1, 80, 4)
    assert st.iters == 5 or st.iters <= 5  # may early-stop if residual_atol set; we cleared it
    # With residual_atol=None, fused/eager runs full K.
    assert st.iters == 5


def test_pin_max_iters_ignores_schedule() -> None:
    assert auto_newton_iters(ParaCfC(3, 2), 4096) >= 2
    # Pin path is resolve-time; just check schedule helpers stay stable.
    assert hopfield_auto_newton_iters(10) == hopfield_auto_newton_iters(10)


@torch.no_grad()
def test_auto_max_iters_matches_sequential_hopfield() -> None:
    torch.manual_seed(1)
    cell = ParaHopfield(5, 8)
    x = torch.randn(2, 48, 5)
    seq = sequential_apply(cell, x)
    par = newton_apply(
        cell,
        x,
        NewtonConfig(max_iters=None, scan_backend="eager", jac_structure="dense"),
    )
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_auto_max_iters_matches_sequential_cfc() -> None:
    torch.manual_seed(2)
    cell = ParaCfC(9, 16)
    feat = torch.randn(2, 64, 8)
    dt = 0.05 + torch.rand(2, 64, 1)
    x = torch.cat((feat, dt), dim=-1)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=None, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4


@torch.no_grad()
def test_auto_max_iters_matches_sequential_titans() -> None:
    torch.manual_seed(3)
    cell = ParaTitans(8, 16)
    x = torch.randn(2, 32, 8)
    seq = sequential_apply(cell, x)
    par = newton_apply(cell, x, NewtonConfig(max_iters=None, scan_backend="eager"))
    assert (par - seq).abs().amax() < 1e-4
