"""ParaRNN nn.Module: train → Newton, eval → sequential."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn import NewtonConfig, ParaRNN
from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.layout import swap_lstm_ch
from pararnn.solvers import newton_apply, sequential_apply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    assert y.shape == (2, 7, 8)
    h = newton_apply(layer.layers[0], x, cfg)
    h = newton_apply(layer.layers[1], h[:, :, 1, :], cfg)
    torch.testing.assert_close(y, h[:, :, 1, :], atol=0, rtol=0)


@torch.no_grad()
def test_h0_eval_and_train():
    torch.manual_seed(76)
    cell = ParaGRU(d_in=4, d_h=6).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, config=cfg)
    x = torch.randn(3, 10, 4, device=device)
    h0 = 0.3 * torch.randn(3, 6, device=device)
    layer.eval()
    torch.testing.assert_close(
        layer(x, h0=h0), sequential_apply(cell, x, h0), atol=0, rtol=0
    )
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


@torch.no_grad()
def test_lstm_default_output_is_hidden():
    torch.manual_seed(79)
    cell = ParaLSTM(d_in=4, d_h=6).to(device)
    layer = ParaRNN(cell)
    layer.eval()
    x = torch.randn(2, 5, 4, device=device)
    y = layer(x)
    full = sequential_apply(cell, x)
    assert y.shape == (2, 5, 6)
    torch.testing.assert_close(y, full[:, :, 1, :], atol=0, rtol=0)
    cell_full = ParaLSTM(d_in=4, d_h=6).to(device)
    cell_full.load_state_dict(cell.state_dict())
    layer_full = ParaRNN(cell_full, output_hidden=False)
    layer_full.eval()
    y_full = layer_full(x)
    assert y_full.shape == (2, 5, 2, 6)
    torch.testing.assert_close(y_full, full, atol=0, rtol=0)


@torch.no_grad()
def test_solver_newton_in_eval_matches_newton_apply():
    torch.manual_seed(80)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    cfg = NewtonConfig(max_iters=3)
    layer = ParaRNN(cell, config=cfg, solver="newton")
    layer.eval()
    x = torch.randn(2, 16, 6, device=device)
    torch.testing.assert_close(layer(x), newton_apply(cell, x, cfg), atol=0, rtol=0)


@torch.no_grad()
def test_solver_sequential_in_train_matches_sequential_apply():
    torch.manual_seed(81)
    cell = ParaGRU(d_in=6, d_h=8).to(device)
    layer = ParaRNN(cell, solver="sequential")
    layer.train()
    x = torch.randn(2, 16, 6, device=device)
    torch.testing.assert_close(layer(x), sequential_apply(cell, x), atol=0, rtol=0)


def test_extra_repr_lists_boundary_flags():
    cell = ParaGRU(d_in=4, d_h=5).to(device)
    layer = ParaRNN(cell, solver="auto", batch_first=True)
    text = layer.extra_repr()
    assert "solver='auto'" in text
    assert "effective=newton" in text
    assert "batch_first=True" in text
    assert "output_hidden=False" in text
    assert "hidden_layout='paper'" in text
    layer.eval()
    assert "effective=sequential" in layer.extra_repr()


def test_solver_rejects_unknown():
    with pytest.raises(ValueError, match="solver"):
        ParaRNN(ParaGRU(d_in=3, d_h=4), solver="picard")  # type: ignore[arg-type]


@torch.no_grad()
def test_batch_first_false_matches_batch_first():
    torch.manual_seed(82)
    cell_a = ParaGRU(d_in=5, d_h=7).to(device)
    cell_b = ParaGRU(d_in=5, d_h=7).to(device)
    cell_b.load_state_dict(cell_a.state_dict())
    bf = ParaRNN(cell_a)
    tf = ParaRNN(cell_b, batch_first=False)
    bf.eval()
    tf.eval()
    x_bf = torch.randn(3, 9, 5, device=device)
    x_tf = x_bf.transpose(0, 1)
    y_bf = bf(x_bf)
    y_tf = tf(x_tf)
    assert y_tf.shape == (9, 3, 7)
    torch.testing.assert_close(y_tf, y_bf.transpose(0, 1), atol=0, rtol=0)
    h0 = 0.2 * torch.randn(3, 7, device=device)
    torch.testing.assert_close(
        tf(x_tf, h0=h0),
        sequential_apply(cell_b, x_bf, h0).transpose(0, 1),
        atol=0,
        rtol=0,
    )


@torch.no_grad()
def test_hidden_layout_pytorch_swaps_h0_and_last():
    torch.manual_seed(83)
    cell_a = ParaLSTM(d_in=4, d_h=6).to(device)
    cell_b = ParaLSTM(d_in=4, d_h=6).to(device)
    cell_b.load_state_dict(cell_a.state_dict())
    paper = ParaRNN(cell_a, return_hidden=True, hidden_layout="paper")
    pytorch = ParaRNN(cell_b, return_hidden=True, hidden_layout="pytorch")
    paper.eval()
    pytorch.eval()
    x = torch.randn(2, 7, 4, device=device)
    h0_paper = 0.3 * torch.randn(2, 2, 6, device=device)
    h0_pytorch = swap_lstm_ch(h0_paper)
    y_p, last_p = paper(x, h0=h0_paper)
    y_t, last_t = pytorch(x, h0=h0_pytorch)
    torch.testing.assert_close(y_p, y_t, atol=0, rtol=0)
    torch.testing.assert_close(last_t, swap_lstm_ch(last_p), atol=0, rtol=0)
    torch.testing.assert_close(last_p[:, 0], last_t[:, 1], atol=0, rtol=0)


def test_hidden_layout_pytorch_rejects_non_lstm():
    with pytest.raises(TypeError, match="ParaLSTM"):
        ParaRNN(ParaGRU(d_in=3, d_h=5), hidden_layout="pytorch")
    from pararnn.cells import ParaSLSTM

    with pytest.raises(TypeError, match="ParaLSTM"):
        ParaRNN(ParaSLSTM(d_in=4, d_h=4), hidden_layout="pytorch")
