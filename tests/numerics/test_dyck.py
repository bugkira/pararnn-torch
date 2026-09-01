"""Dyck-1 sampler + ParaSLSTM Newton grads vs sequential BPTT."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from pararnn import NewtonConfig, ParaRNN, ParaSLSTM, device, sequential_apply
from pararnn.layout import SLSTM_HIDDEN
from pararnn.train.dyck import VOCAB, sample_dyck1


def test_sample_dyck1_is_balanced():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    tokens = sample_dyck1(32, 16, generator=gen)
    assert tokens.shape == (32, 16)
    depth = (tokens == 0).int() - (tokens == 1).int()
    running = depth.cumsum(dim=1)
    assert (running >= 0).all()
    assert (running[:, -1] == 0).all()


def test_dyck_newton_grads_match_sequential_bptt():
    torch.manual_seed(80)
    d_h, t = 8, 8
    tokens = sample_dyck1(4, t, generator=torch.Generator().manual_seed(1)).to(device)
    cfg = NewtonConfig(max_iters=3, scan_backend="eager")
    cell_s = ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag").to(device)
    cell_n = ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag").to(device)
    cell_n.load_state_dict(cell_s.state_dict())
    emb_s = torch.nn.Embedding(VOCAB, d_h).to(device)
    emb_n = torch.nn.Embedding(VOCAB, d_h).to(device)
    emb_n.load_state_dict(emb_s.state_dict())
    head_s = torch.nn.Linear(d_h, VOCAB).to(device)
    head_n = torch.nn.Linear(d_h, VOCAB).to(device)
    head_n.load_state_dict(head_s.state_dict())
    x_s = emb_s(tokens[:, :-1])
    logits_s = head_s(sequential_apply(cell_s, x_s)[:, :, SLSTM_HIDDEN, :])
    layer = ParaRNN(cell_n, config=cfg, output_hidden=True)
    layer.train()
    logits_n = head_n(layer(emb_n(tokens[:, :-1])))
    target = tokens[:, 1:]
    F.cross_entropy(logits_s.reshape(-1, VOCAB), target.reshape(-1)).backward()
    F.cross_entropy(logits_n.reshape(-1, VOCAB), target.reshape(-1)).backward()
    for (n, p_a), (_, p_b) in zip(cell_s.named_parameters(), cell_n.named_parameters()):
        assert p_a.grad is not None and p_b.grad is not None, n
        assert torch.isfinite(p_b.grad).all()
        torch.testing.assert_close(p_a.grad, p_b.grad, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(emb_s.weight.grad, emb_n.weight.grad, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(head_s.weight.grad, head_n.weight.grad, atol=2e-4, rtol=1e-4)
