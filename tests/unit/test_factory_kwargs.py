"""Cells follow nn.Module factory_kwargs (device=, dtype=). Not a GPU picker."""

from __future__ import annotations

import torch

from pararnn import ParaGRU, ParaLSTM, ParaRNN, ParaSLSTM, xLSTMBlock


def test_gru_factory_kwargs_cpu_float64():
    cell = ParaGRU(4, 8, device="cpu", dtype=torch.float64)
    assert cell.a_z.device.type == "cpu"
    assert cell.a_z.dtype == torch.float64
    assert cell.W_x.weight.dtype == torch.float64


def test_lstm_factory_kwargs_cpu_float64():
    cell = ParaLSTM(4, 8, device="cpu", dtype=torch.float64)
    assert cell.a_f.dtype == torch.float64
    assert cell.W_x.weight.device.type == "cpu"


def test_slstm_factory_kwargs_cpu_float64():
    cell = ParaSLSTM(4, 8, mix="diag", device="cpu", dtype=torch.float64)
    assert cell.R is not None
    assert cell.R.dtype == torch.float64
    assert cell.W_x.weight.device.type == "cpu"


def test_pararnn_factory_kwargs_stack():
    cell = ParaGRU(4, 8, device="cpu", dtype=torch.float32)
    model = ParaRNN(cell, num_layers=2, device="cpu", dtype=torch.float64)
    assert model.layers[0].a_z.dtype == torch.float64
    assert model.layers[1].a_z.dtype == torch.float64
    assert model.layers[1].W_x.weight.device.type == "cpu"


def test_pararnn_stack_infers_dtype_from_cell():
    cell = ParaGRU(4, 8, device="cpu", dtype=torch.float64)
    model = ParaRNN(cell, num_layers=2)
    assert model.layers[1].a_z.dtype == torch.float64
    assert model.layers[1].W_x.weight.device.type == "cpu"


def test_xlstm_block_factory_kwargs_cpu_float64():
    block = xLSTMBlock(8, device="cpu", dtype=torch.float64)
    assert block.norm.weight.dtype == torch.float64
    assert block.cell.W_x.weight.dtype == torch.float64
    assert block.cell.W_x.weight.device.type == "cpu"


def test_pararnn_reset_parameters():
    torch.manual_seed(0)
    model = ParaRNN(ParaGRU(4, 8, device="cpu"), num_layers=2)
    w0 = model.layers[0].a_z.detach().clone()
    w1 = model.layers[1].a_z.detach().clone()
    model.reset_parameters()
    assert not torch.equal(w0, model.layers[0].a_z)
    assert not torch.equal(w1, model.layers[1].a_z)


def test_cell_extra_repr():
    gru = ParaGRU(4, 8)
    assert gru.extra_repr() == "4, 8, max_recurrent_norm=0.5"
    lstm = ParaLSTM(4, 8)
    assert lstm.extra_repr() == "4, 8, max_recurrent_norm=0.5"
    slstm = ParaSLSTM(8, 8, mix="head", n_heads=2)
    text = slstm.extra_repr()
    assert text.startswith("8, 8, mix='head', n_heads=2")
    assert "max_recurrent_norm=0.5" in text
    assert "eps=1e-06" in text or "eps=1e-6" in text
