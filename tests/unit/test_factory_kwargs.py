"""Cells follow nn.Module factory_kwargs (device=, dtype=). Not a GPU picker."""

from __future__ import annotations

import torch

from pararnn import ParaGRU, ParaLSTM, ParaSLSTM


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
