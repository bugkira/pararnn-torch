"""RNNCell protocol: d_h + step, no required base class."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.cells.protocol import RNNCell, check_cell


def test_paragru_and_paralstm_satisfy_protocol():
    gru = ParaGRU(d_in=4, d_h=5)
    lstm = ParaLSTM(d_in=4, d_h=5)
    check_cell(gru)
    check_cell(lstm)
    assert isinstance(gru, RNNCell)
    assert isinstance(lstm, RNNCell)


def test_check_cell_rejects_missing_d_h():
    class Bare(nn.Module):
        def step(self, h: Tensor, x: Tensor) -> Tensor:
            return h

    with pytest.raises(TypeError, match="d_h"):
        check_cell(Bare())


def test_check_cell_rejects_missing_step():
    class NoStep(nn.Module):
        d_h = 3

    with pytest.raises(TypeError, match="step"):
        check_cell(NoStep())


def test_dummy_with_d_h_and_step_passes():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_h = 2
            self.a = nn.Parameter(torch.zeros(2))

        def step(self, h: Tensor, x: Tensor) -> Tensor:
            return h + x[..., :2] + self.a

    check_cell(Tiny())
