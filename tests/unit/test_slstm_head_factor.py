"""Unit tests: factorized sLSTM head JVP / Jᵀ vs dense ``_jac_packed``."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaSLSTM
from pararnn.kernels.slstm_head_factor import (
    slstm_head_gates,
    slstm_head_jt_mvp,
    slstm_head_jvp,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:mix='head' is Beck-style dense R:UserWarning",
)


@pytest.mark.parametrize("d_h,n_heads", [(4, 2), (8, 4)])
def test_slstm_head_jvp_matches_dense(d_h: int, n_heads: int) -> None:
    torch.manual_seed(11)
    cell = ParaSLSTM(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads)
    b, t = 2, 5
    x = 0.3 * torch.randn(b, t, d_h)
    state = 0.2 * torch.randn(b, t, 4, d_h)
    wx = cell.W_x(x)
    r = cell.clipped_r_head()
    assert cell.d_head is not None
    d_head = cell.d_head
    state_new, acts = slstm_head_gates(
        state, wx, r, n_heads=n_heads, d_head=d_head, eps=cell.eps
    )
    ref_new = cell.step(
        state.reshape(b * t, 4, d_h), None, wx=wx.reshape(b * t, 4 * d_h)
    )
    torch.testing.assert_close(
        state_new.reshape(b * t, 4, d_h), ref_new, atol=1e-5, rtol=1e-5
    )
    v = torch.randn(b, t, n_heads, 4 * d_head)
    jvp = slstm_head_jvp(acts, r, v)
    for bi in range(b):
        for ti in range(t):
            st = state[bi, ti]
            w = wx[bi, ti]
            _, jac = cell.step_with_jacobian(st.unsqueeze(0), None, wx=w.unsqueeze(0))
            vv = v[bi, ti].reshape(n_heads, 4 * d_head)
            dense = torch.matmul(jac[0], vv.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(jvp[bi, ti], dense, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("d_h,n_heads", [(4, 2), (8, 4)])
def test_slstm_head_jt_mvp_matches_dense(d_h: int, n_heads: int) -> None:
    torch.manual_seed(12)
    cell = ParaSLSTM(d_in=d_h, d_h=d_h, mix="head", n_heads=n_heads)
    b, t = 2, 4
    x = 0.3 * torch.randn(b, t, d_h)
    state = 0.2 * torch.randn(b, t, 4, d_h)
    wx = cell.W_x(x)
    r = cell.clipped_r_head()
    assert cell.d_head is not None
    d_head = cell.d_head
    _, acts = slstm_head_gates(state, wx, r, n_heads=n_heads, d_head=d_head, eps=cell.eps)
    mu = torch.randn(b, t, n_heads, 4 * d_head)
    jt = slstm_head_jt_mvp(acts, r, mu)
    for bi in range(b):
        for ti in range(t):
            st = state[bi, ti]
            w = wx[bi, ti]
            _, jac = cell.step_with_jacobian(st.unsqueeze(0), None, wx=w.unsqueeze(0))
            mm = mu[bi, ti].reshape(n_heads, 4 * d_head)
            dense = torch.matmul(jac[0].transpose(-1, -2), mm.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(jt[bi, ti], dense, atol=1e-4, rtol=1e-4)
