"""CPU decode_step is eager cell.step; Triton needs CUDA."""

from __future__ import annotations

import torch

from pararnn import ParaGRU, ParaLSTM, ParaRNN, ParaSLSTM, can_decode_step, decode_step
from pararnn.solvers import sequential_apply


def test_cpu_can_decode_step_false() -> None:
    cell = ParaGRU(d_in=4, d_h=5)
    ref = torch.randn(2, 4)
    assert can_decode_step(cell, ref) is False


def test_cpu_decode_step_matches_cell_step() -> None:
    torch.manual_seed(3)
    cell = ParaGRU(d_in=5, d_h=7)
    h = torch.randn(3, 7)
    x = torch.randn(3, 5)
    torch.testing.assert_close(decode_step(cell, h, x), cell.step(h, x))
    wx = cell.W_x(x)
    torch.testing.assert_close(decode_step(cell, h, wx=wx), cell.step(h, x, wx=wx))


def test_cpu_lstm_slstm_decode_step_matches() -> None:
    torch.manual_seed(4)
    lstm = ParaLSTM(d_in=5, d_h=6)
    s = torch.randn(2, 2, 6)
    x = torch.randn(2, 5)
    torch.testing.assert_close(decode_step(lstm, s, x), lstm.step(s, x))
    slstm = ParaSLSTM(d_in=5, d_h=4, mix="diag")
    st = torch.randn(2, 4, 4)
    torch.testing.assert_close(decode_step(slstm, st, x), slstm.step(st, x))


def test_eval_t1_matches_step() -> None:
    torch.manual_seed(5)
    cell = ParaGRU(d_in=6, d_h=8)
    model = ParaRNN(cell)
    model.eval()
    x = torch.randn(2, 1, 6)
    h0 = torch.randn(2, 8)
    with torch.no_grad():
        y = model(x, h0)
    ref = cell.step(h0, x[:, 0]).unsqueeze(1)
    torch.testing.assert_close(y, ref)


def test_cpu_decode_step_out_reuses_buffer() -> None:
    torch.manual_seed(7)
    cell = ParaGRU(d_in=4, d_h=5)
    h = torch.randn(2, 5)
    x = torch.randn(2, 4)
    out = torch.empty_like(h)
    got = decode_step(cell, h, x, out=out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, cell.step(h, x))


def test_sequential_t1_cpu_matches_step() -> None:
    torch.manual_seed(6)
    cell = ParaLSTM(d_in=4, d_h=5)
    x = torch.randn(2, 1, 4)
    h0 = torch.randn(2, 2, 5)
    got = sequential_apply(cell, x, h0)
    torch.testing.assert_close(got[:, 0], cell.step(h0, x[:, 0]))
