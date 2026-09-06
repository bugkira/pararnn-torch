"""Factorized head-GRU Jacobian matvecs vs dense ``_jac_head``."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells.para_gru import ParaGRU
from pararnn.kernels.gru_head_factor import gru_head_jt_mvp, gru_head_jvp

pytestmark = pytest.mark.filterwarnings(
    "ignore:mix='head' is block-diagonal ParaGRU:UserWarning"
)


@pytest.mark.parametrize("d_h,n_heads", [(8, 4), (64, 8), (32, 2)])
@torch.no_grad()
def test_gru_head_jvp_matches_dense(d_h: int, n_heads: int):
    torch.manual_seed(1)
    cell = ParaGRU(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads)
    b, t = 2, 5
    h = torch.randn(b, t, d_h)
    x = 0.3 * torch.randn(b, t, d_h)
    acts = cell._recurrence(h, x)
    jac = cell._jac_head(acts)
    v = torch.randn(b, t, n_heads, cell.d_head)
    dense = torch.matmul(jac, v.unsqueeze(-1)).squeeze(-1)
    fact = gru_head_jvp(
        acts.h_heads, acts.z_heads, acts.r_heads, acts.n_heads, acts.a_z, acts.a_r, acts.a_n, v
    )
    torch.testing.assert_close(fact, dense, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d_h,n_heads", [(8, 4), (64, 8)])
@torch.no_grad()
def test_gru_head_jt_mvp_matches_dense(d_h: int, n_heads: int):
    torch.manual_seed(2)
    cell = ParaGRU(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads)
    b, t = 2, 5
    h = torch.randn(b, t, d_h)
    x = 0.3 * torch.randn(b, t, d_h)
    acts = cell._recurrence(h, x)
    jac = cell._jac_head(acts)
    mu = torch.randn(b, t, n_heads, cell.d_head)
    dense = torch.matmul(jac.transpose(-1, -2), mu.unsqueeze(-1)).squeeze(-1)
    fact = gru_head_jt_mvp(
        acts.h_heads, acts.z_heads, acts.r_heads, acts.n_heads, acts.a_z, acts.a_r, acts.a_n, mu
    )
    torch.testing.assert_close(fact, dense, atol=1e-5, rtol=1e-5)
