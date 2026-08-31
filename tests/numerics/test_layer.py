"""ParaRNN nn.Module: train → Newton, eval → sequential."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn import NewtonConfig, ParaRNN, device, newton_apply, sequential_apply
from pararnn.cells import ParaGRU, ParaLSTM


@torch.no_grad()
def test_train_matches_newton_apply():
    torch.manual_seed(70)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, config=cfg)
    layer.train()
    x = torch.randn(2, 16, 6, device=device)
    torch.testing.assert_close(layer(x), newton_apply(cell, x, cfg), atol=0, rtol=0)


@torch.no_grad()
def test_eval_matches_sequential_apply():
    torch.manual_seed(71)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    layer = ParaRNN(cell)
    layer.eval()
    x = torch.randn(2, 16, 6, device=device)
    torch.testing.assert_close(layer(x), sequential_apply(cell, x), atol=0, rtol=0)


@torch.no_grad()
def test_train_eval_switch_solvers():
    torch.manual_seed(72)
    cell = ParaGRU(d_in=5, d_h=7).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, config=cfg)
    x = torch.randn(2, 12, 5, device=device)
    layer.train()
    y_train = layer(x)
    layer.eval()
    y_eval = layer(x)
    torch.testing.assert_close(y_train, newton_apply(cell, x, cfg), atol=0, rtol=0)
    torch.testing.assert_close(y_eval, sequential_apply(cell, x), atol=0, rtol=0)
    err = (y_train - y_eval).abs().amax()
    assert err < 1e-4, err


def test_grads_match_sequential_bptt():
    torch.manual_seed(73)
    d_in, d_h, t = 5, 7, 16
    x = torch.randn(2, t, d_in, device=device)
    w = torch.randn(2, t, d_h, device=device)
    cell_s = ParaGRU(d_in, d_h).to(device)
    cell_n = ParaGRU(d_in, d_h).to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    layer = ParaRNN(cell_n, config=NewtonConfig(max_iters=3))
    layer.train()
    x_s = x.clone().requires_grad_(True)
    x_n = x.clone().requires_grad_(True)
    (sequential_apply(cell_s, x_s) * w).sum().backward()
    (layer(x_n) * w).sum().backward()
    for (n, p_a), (_, p_b) in zip(cell_s.named_parameters(), cell_n.named_parameters()):
        assert p_a.grad is not None, n
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(x_s.grad, x_n.grad, atol=2e-4, rtol=1e-4)


@torch.no_grad()
def test_two_layer_gru_din_neq_dh():
    torch.manual_seed(74)
    cell = ParaGRU(d_in=6, d_h=10).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, num_layers=2, config=cfg)
    assert layer.layers[0].d_in == 6
    assert layer.layers[0].d_h == 10
    assert layer.layers[1].d_in == 10
    assert layer.layers[1].d_h == 10
    x = torch.randn(2, 8, 6, device=device)
    layer.train()
    y = layer(x)
    assert y.shape == (2, 8, 10)
    h = newton_apply(layer.layers[0], x, cfg)
    h = newton_apply(layer.layers[1], h, cfg)
    torch.testing.assert_close(y, h, atol=0, rtol=0)
    layer.eval()
    y_eval = layer(x)
    h_seq = sequential_apply(layer.layers[0], x)
    h_seq = sequential_apply(layer.layers[1], h_seq)
    torch.testing.assert_close(y_eval, h_seq, atol=0, rtol=0)
    assert (y - y_eval).abs().amax() < 1e-4


@torch.no_grad()
def test_two_layer_lstm_feeds_hidden_slot():
    torch.manual_seed(75)
    cell = ParaLSTM(d_in=5, d_h=8).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, num_layers=2, config=cfg)
    x = torch.randn(2, 7, 5, device=device)
    layer.train()
    y = layer(x)
    assert y.shape == (2, 7, 2, 8)
    h = newton_apply(layer.layers[0], x, cfg)
    h = newton_apply(layer.layers[1], h[:, :, 1, :], cfg)
    torch.testing.assert_close(y, h, atol=0, rtol=0)


@torch.no_grad()
def test_h0_eval_and_train():
    torch.manual_seed(76)
    cell = ParaGRU(d_in=4, d_h=6).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, config=cfg)
    x = torch.randn(3, 10, 4, device=device)
    h0 = 0.3 * torch.randn(3, 6, device=device)
    layer.eval()
    torch.testing.assert_close(layer(x, h0=h0), sequential_apply(cell, x, h0), atol=0, rtol=0)
    layer.train()
    torch.testing.assert_close(
        layer(x, h0=h0), newton_apply(cell, x, cfg, h0=h0), atol=0, rtol=0
    )


def test_custom_cell_num_layers_gt1_raises():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_h = 3
            self.a = nn.Parameter(torch.zeros(3))

        def step(self, h: Tensor, x: Tensor) -> Tensor:
            return h + x[..., :3] + self.a

    with pytest.raises(TypeError, match="num_layers"):
        ParaRNN(Tiny(), num_layers=2)


@torch.no_grad()
def test_list_of_cells_ctor():
    torch.manual_seed(77)
    a = ParaGRU(d_in=5, d_h=8).to(device)
    b = ParaGRU(d_in=8, d_h=8).to(device)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    layer = ParaRNN([a, b], config=cfg)
    layer.eval()
    x = torch.randn(2, 6, 5, device=device)
    y = layer(x)
    h = sequential_apply(a, x)
    h = sequential_apply(b, h)
    torch.testing.assert_close(y, h, atol=0, rtol=0)


@torch.no_grad()
def test_return_hidden_and_output_hidden_lstm():
    torch.manual_seed(78)
    cell = ParaLSTM(d_in=4, d_h=6).to(device)
    layer = ParaRNN(cell, return_hidden=True, output_hidden=True)
    layer.eval()
    x = torch.randn(2, 7, 4, device=device)
    y, h_last = layer(x)
    full = sequential_apply(cell, x)
    assert y.shape == (2, 7, 6)
    torch.testing.assert_close(y, full[:, :, 1, :], atol=0, rtol=0)
    torch.testing.assert_close(h_last, full[:, -1], atol=0, rtol=0)
