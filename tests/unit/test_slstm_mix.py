"""ParaSLSTM mix roles: fused diag, head ablation, dense oracle cap."""

from __future__ import annotations

import pytest

from pararnn.cells.para_slstm import DENSE_MAX_HIDDEN, ParaSLSTM


def test_head_mix_warns_ablation():
    with pytest.warns(UserWarning, match="Beck-style dense R"):
        cell = ParaSLSTM(4, 4, mix="head", n_heads=2)
    assert cell.mix == "head"
    assert cell.n_heads == 2
    assert "mix='head', n_heads=2" in cell.extra_repr()


def test_dense_mix_rejects_width_above_oracle_cap():
    with pytest.raises(ValueError, match="Jacobian oracle"):
        ParaSLSTM(DENSE_MAX_HIDDEN + 1, DENSE_MAX_HIDDEN + 1, mix="dense")


def test_dense_mix_allows_oracle_width():
    cell = ParaSLSTM(DENSE_MAX_HIDDEN, DENSE_MAX_HIDDEN, mix="dense")
    assert cell.mix == "dense"
    assert cell.R_dense is not None
