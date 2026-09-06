"""Analytic Jacobians vs autograd (eq. 3.2)."""

from __future__ import annotations

import pytest
import torch

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.layout import LSTM_CELL, LSTM_HIDDEN, SLSTM_SLOTS


def test_gru_jacobian_matches_autograd():
    torch.manual_seed(0)
    cell = ParaGRU(d_in=5, d_h=7)
    h_prev = torch.randn(3, 7, requires_grad=True)
    x = torch.randn(3, 5)
    h_new, j_diag = cell.step_with_jacobian(h_prev, x)
    for b in range(3):
        for i in range(7):
            g = torch.autograd.grad(h_new[b, i], h_prev, retain_graph=True)[0][b]
            expect = torch.zeros_like(g)
            expect[i] = j_diag[b, i]
            torch.testing.assert_close(g, expect, atol=1e-5, rtol=1e-5)


def test_gru_head_jacobian_matches_autograd():
    torch.manual_seed(0)
    with pytest.warns(UserWarning, match="block-diagonal ParaGRU"):
        cell = ParaGRU(d_in=4, d_h=8, mix="head", n_heads=4)
    h_prev = torch.randn(2, 8, requires_grad=True)
    x = torch.randn(2, 4)
    h_new, jac = cell.step_with_jacobian(h_prev, x)
    assert jac.shape == (2, 4, 2, 2)
    for b in range(2):
        for out_h in range(4):
            for out_d in range(2):
                i = out_h * 2 + out_d
                g = torch.autograd.grad(h_new[b, i], h_prev, retain_graph=True)[0][b]
                expect = torch.zeros_like(g)
                for in_h in range(4):
                    for in_d in range(2):
                        if in_h == out_h:
                            expect[in_h * 2 + in_d] = jac[b, out_h, out_d, in_d]
                torch.testing.assert_close(g, expect, atol=1e-5, rtol=1e-5)


def test_gru_step_wx_without_x():
    torch.manual_seed(2)
    cell = ParaGRU(d_in=5, d_h=7)
    h_prev = torch.randn(3, 7)
    x = torch.randn(3, 5)
    wx = cell.W_x(x)
    torch.testing.assert_close(cell.step(h_prev, x), cell.step(h_prev, wx=wx))
    h_new, j = cell.step_with_jacobian(h_prev, wx=wx)
    h_ref, j_ref = cell.step_with_jacobian(h_prev, x)
    torch.testing.assert_close(h_new, h_ref)
    torch.testing.assert_close(j, j_ref)


def test_gru_step_needs_x_or_wx():
    cell = ParaGRU(d_in=3, d_h=4)
    with pytest.raises(ValueError, match="x or wx"):
        cell.step(torch.zeros(2, 4))


def test_lstm_jacobian_matches_autograd():
    torch.manual_seed(1)
    cell = ParaLSTM(d_in=5, d_h=6)
    state = torch.randn(2, 2, 6, requires_grad=True)
    x = torch.randn(2, 5)
    new_state, jac = cell.step_with_jacobian(state, x)
    # jac[..., out, in, d]
    for b in range(2):
        for out in (LSTM_CELL, LSTM_HIDDEN):
            for d in range(6):
                g = torch.autograd.grad(new_state[b, out, d], state, retain_graph=True)[0][b]
                expect = torch.zeros_like(g)
                expect[LSTM_CELL, d] = jac[b, out, LSTM_CELL, d]
                expect[LSTM_HIDDEN, d] = jac[b, out, LSTM_HIDDEN, d]
                torch.testing.assert_close(g, expect, atol=1e-5, rtol=1e-5)


def test_slstm_diag_jacobian_matches_autograd():
    torch.manual_seed(20)
    cell = ParaSLSTM(d_in=5, d_h=4, mix="diag")
    state = torch.randn(2, SLSTM_SLOTS, 4, requires_grad=True)
    x = torch.randn(2, 5)
    new_state, jac = cell.step_with_jacobian(state, x)
    for b in range(2):
        for out in range(SLSTM_SLOTS):
            for d in range(4):
                g = torch.autograd.grad(new_state[b, out, d], state, retain_graph=True)[0][b]
                expect = torch.zeros_like(g)
                for inn in range(SLSTM_SLOTS):
                    expect[inn, d] = jac[b, out, inn, d]
                torch.testing.assert_close(g, expect, atol=1e-5, rtol=1e-5)
