"""step must match step_with_jacobian's state and must not be the Jacobian path."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM


def test_gru_step_matches_jacobian_state():
    torch.manual_seed(6)
    cell = ParaGRU(d_in=5, d_h=7)
    h_prev = torch.randn(3, 7)
    x = torch.randn(3, 5)
    h_step = cell.step(h_prev, x)
    h_jac, _ = cell.step_with_jacobian(h_prev, x)
    torch.testing.assert_close(h_step, h_jac)


def test_lstm_step_matches_jacobian_state():
    torch.manual_seed(7)
    cell = ParaLSTM(d_in=5, d_h=6)
    state = torch.randn(2, 2, 6)
    x = torch.randn(2, 5)
    s_step = cell.step(state, x)
    s_jac, _ = cell.step_with_jacobian(state, x)
    torch.testing.assert_close(s_step, s_jac)


def test_gru_precomputed_wx_matches():
    torch.manual_seed(14)
    cell = ParaGRU(d_in=5, d_h=7)
    h_prev = torch.randn(3, 7)
    x = torch.randn(3, 5)
    wx = cell.W_x(x)
    torch.testing.assert_close(cell.step(h_prev, x), cell.step(h_prev, x, wx=wx))
    h_a, j_a = cell.step_with_jacobian(h_prev, x)
    h_b, j_b = cell.step_with_jacobian(h_prev, x, wx=wx)
    torch.testing.assert_close(h_a, h_b)
    torch.testing.assert_close(j_a, j_b)


def test_lstm_precomputed_wx_matches():
    torch.manual_seed(15)
    cell = ParaLSTM(d_in=5, d_h=6)
    state = torch.randn(2, 2, 6)
    x = torch.randn(2, 5)
    wx = cell.W_x(x)
    torch.testing.assert_close(cell.step(state, x), cell.step(state, x, wx=wx))
    s_a, j_a = cell.step_with_jacobian(state, x)
    s_b, j_b = cell.step_with_jacobian(state, x, wx=wx)
    torch.testing.assert_close(s_a, s_b)
    torch.testing.assert_close(j_a, j_b)


def test_slstm_step_matches_jacobian_state():
    torch.manual_seed(21)
    cell = ParaSLSTM(d_in=5, d_h=4, mix="diag")
    state = torch.randn(2, 4, 4)
    x = torch.randn(2, 5)
    s_step = cell.step(state, x)
    s_jac, _ = cell.step_with_jacobian(state, x)
    torch.testing.assert_close(s_step, s_jac)


@pytest.mark.filterwarnings("ignore:mix='head' is an unfused ablation:UserWarning")
def test_slstm_precomputed_wx_matches():
    torch.manual_seed(22)
    cell = ParaSLSTM(d_in=5, d_h=4, mix="head", n_heads=2)
    state = torch.randn(2, 4, 4)
    x = torch.randn(2, 5)
    wx = cell.W_x(x)
    torch.testing.assert_close(cell.step(state, x), cell.step(state, x, wx=wx))
    s_a, j_a = cell.step_with_jacobian(state, x)
    s_b, j_b = cell.step_with_jacobian(state, x, wx=wx)
    torch.testing.assert_close(s_a, s_b)
    torch.testing.assert_close(j_a, j_b)
