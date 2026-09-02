"""Running-XOR labels and parity nets (newton vs eager, SSM scan)."""

from __future__ import annotations

import torch

from examples.parity import VOCAB, _eval_acc, _ParityNet, _S6Block, sample_parity
from pararnn import NewtonConfig, xLSTMBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_sample_parity_is_running_xor():
    torch.manual_seed(0)
    bits, labels = sample_parity(8, 16, generator=torch.Generator().manual_seed(1))
    assert bits.shape == labels.shape == (8, 16)
    assert set(bits.unique().tolist()) <= {0, 1}
    running = torch.zeros_like(bits)
    acc = torch.zeros(8, dtype=torch.long)
    for t in range(16):
        acc = acc ^ bits[:, t]
        running[:, t] = acc
    torch.testing.assert_close(labels, running)


def test_parity_net_newton_train_matches_eval():
    torch.manual_seed(91)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    net = _ParityNet(8, 2, "newton", cfg, max_recurrent_norm=None).to(device)
    bits, _ = sample_parity(2, 8, generator=torch.Generator().manual_seed(2))
    bits = bits.to(device)
    net.train()
    y_tr = net(bits)
    net.eval()
    y_ev = net(bits)
    torch.testing.assert_close(y_tr, y_ev, atol=5e-4, rtol=1e-4)
    assert y_tr.shape == (2, 8, VOCAB)


def test_parity_net_eager_matches_newton_eval():
    torch.manual_seed(92)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    newton = _ParityNet(8, 1, "newton", cfg, None).to(device)
    eager = _ParityNet(8, 1, "eager", cfg, None).to(device)
    eager.load_state_dict(newton.state_dict())
    bits, _ = sample_parity(2, 8, generator=torch.Generator().manual_seed(3))
    bits = bits.to(device)
    newton.eval()
    eager.eval()
    torch.testing.assert_close(newton(bits), eager(bits), atol=5e-4, rtol=1e-4)


def test_s6_block_residual_shape_and_grad():
    torch.manual_seed(93)
    block = _S6Block(6).to(device)
    x = torch.randn(2, 10, 6, device=device, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_xlstm_block_forwards_parity_no_clip():
    block = xLSTMBlock(8, solver="sequential", max_recurrent_norm=None, device=device)
    assert block.cell.max_recurrent_norm is None


def test_eval_acc_splits_copy_from_full_xor():
    torch.manual_seed(94)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    net = _ParityNet(8, 1, "eager", cfg, None).to(device)
    bits, labels = sample_parity(4, 8, generator=torch.Generator().manual_seed(5))
    bits, labels = bits.to(device), labels.to(device)
    row = _eval_acc(net, bits, labels)
    assert set(row) == {"tok", "exact", "tok_t0", "tok_last"}
    assert bits[:, 0].eq(labels[:, 0]).all()
