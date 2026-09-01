"""Pre-norm residual sLSTM block: Newton vs sequential, residual add."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, sequential_apply, xLSTMBlock
from pararnn.layout import SLSTM_HIDDEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_xlstm_block_shape_and_residual():
    torch.manual_seed(81)
    d, t = 8, 12
    block = xLSTMBlock(d, backend="eager", device=device)
    x = torch.randn(2, t, d, device=device)
    y = block(x)
    assert y.shape == x.shape
    z = block.norm(x)
    h = sequential_apply(block.cell, z)[:, :, SLSTM_HIDDEN, :]
    torch.testing.assert_close(y, x + h, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_xlstm_block_newton_matches_eager_eval():
    torch.manual_seed(82)
    d, t = 8, 12
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    newton = xLSTMBlock(d, backend="newton", config=cfg, device=device)
    eager = xLSTMBlock(d, backend="eager", device=device)
    eager.load_state_dict(newton.state_dict())
    x = 0.3 * torch.randn(2, t, d, device=device)
    newton.eval()
    torch.testing.assert_close(newton(x), eager(x), atol=2e-4, rtol=1e-4)


def test_xlstm_block_rejects_flashrnn_stub():
    with pytest.raises(ValueError, match="FlashRNN"):
        xLSTMBlock(8, backend="flashrnn")
