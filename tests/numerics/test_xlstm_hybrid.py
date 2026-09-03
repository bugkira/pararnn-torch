"""NX-AI sLSTMBlock + ParaRNN(ParaSLSTM). Skips unless extra ``xlstm`` is installed."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, ParaSLSTM

pytest.importorskip("xlstm")


def test_hybrid_slot_is_paraslstm():
    from examples.xlstm_hybrid import build_hybrid_slstm_block

    block = build_hybrid_slstm_block(32, 4, config=NewtonConfig(max_iters=3, scan_backend="eager"))
    cell = block.xlstm.rnn.layers[0]
    assert isinstance(cell, ParaSLSTM)
    assert cell.mix == "diag"
    assert cell.n_heads is None


@torch.no_grad()
def test_hybrid_train_matches_eval():
    from examples.xlstm_hybrid import build_hybrid_slstm_block

    torch.manual_seed(11)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = build_hybrid_slstm_block(32, 4, config=cfg).to(device)
    x = 0.3 * torch.randn(2, 8, 32, device=device)
    block.train()
    y_tr = block(x)
    block.eval()
    y_ev = block(x)
    torch.testing.assert_close(y_tr, y_ev, atol=2e-4, rtol=1e-4)
    assert y_tr.shape == x.shape
