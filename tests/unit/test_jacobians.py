"""Analytic Jacobians vs autograd (eq. 3.2)."""

from __future__ import annotations

import torch

from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.layout import LSTM_CELL, LSTM_HIDDEN


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
