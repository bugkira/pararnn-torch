"""Linear-draft verify: Newton scan vs sequential unroll."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from pararnn import NewtonConfig, ParaGRU, ParaRNN, ParaSLSTM, verify_linear_draft

_ATOL = 1e-4
_RTOL = 1e-4


def _pack(
    *, d: int = 12, vocab: int = 17, seed: int = 4
) -> tuple[ParaRNN, nn.Embedding, nn.Linear]:
    torch.manual_seed(seed)
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), config=cfg)
    return model, nn.Embedding(vocab, d), nn.Linear(d, vocab)


def _greedy(
    model: ParaRNN, embed: nn.Embedding, head: nn.Linear, h0: torch.Tensor, k: int
) -> torch.Tensor:
    cell = model.layers[0]
    h = h0
    ids = []
    for _ in range(k):
        tok = head(h).argmax(-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    return torch.stack(ids, dim=1)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_newton_matches_sequential_on_greedy_draft(k: int) -> None:
    model, embed, head = _pack()
    batch = 3
    h0 = torch.randn(batch, 12)
    draft_ids = _greedy(model, embed, head, h0, k)
    x = embed(draft_ids)
    seq = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="sequential")
    par = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="newton")
    torch.testing.assert_close(par.n_accepted, seq.n_accepted)
    torch.testing.assert_close(par.bonus_ids, seq.bonus_ids)
    torch.testing.assert_close(par.hidden, seq.hidden, atol=_ATOL, rtol=_RTOL)
    assert int((par.n_accepted == k).sum()) == batch


def test_newton_matches_sequential_on_corrupted_draft() -> None:
    model, embed, head = _pack(seed=9)
    batch, k = 4, 8
    h0 = torch.randn(batch, 12)
    draft_ids = _greedy(model, embed, head, h0, k)
    draft_ids = draft_ids.clone()
    draft_ids[:, 2] = (draft_ids[:, 2] + 3) % 17
    draft_ids[1, 0] = (draft_ids[1, 0] + 1) % 17
    x = embed(draft_ids)
    seq = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="sequential")
    par = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="newton")
    torch.testing.assert_close(par.n_accepted, seq.n_accepted)
    torch.testing.assert_close(par.bonus_ids, seq.bonus_ids)
    torch.testing.assert_close(par.hidden, seq.hidden, atol=_ATOL, rtol=_RTOL)
    assert int(par.n_accepted[1].item()) == 0
    assert int(par.n_accepted[0].item()) == 2


def test_stacked_two_layer_newton_matches_sequential() -> None:
    torch.manual_seed(2)
    d, vocab, batch, k = 8, 13, 2, 5
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), num_layers=2, config=cfg)
    embed = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab)
    h0 = (torch.randn(batch, d), torch.randn(batch, d))
    seq_ids = []
    h = [h0[0].clone(), h0[1].clone()]
    for _ in range(k):
        tok = head(h[-1]).argmax(-1)
        seq_ids.append(tok)
        x_t = embed(tok)
        h[0] = model.layers[0].step(h[0], x_t)
        h[1] = model.layers[1].step(h[1], h[0])
    draft_ids = torch.stack(seq_ids, dim=1)
    x = embed(draft_ids)
    seq = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="sequential")
    par = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="newton")
    torch.testing.assert_close(par.n_accepted, seq.n_accepted)
    assert isinstance(par.hidden, tuple)
    for p, s in zip(par.hidden, seq.hidden, strict=True):
        torch.testing.assert_close(p, s, atol=_ATOL, rtol=_RTOL)
    assert int((par.n_accepted == k).sum()) == batch


def test_slstm_newton_matches_sequential() -> None:
    torch.manual_seed(6)
    d, vocab, batch, k = 8, 11, 2, 6
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaSLSTM(d, d, mix="diag"), config=cfg)
    embed = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab)
    h0 = torch.zeros(batch, 4, d)
    cell = model.layers[0]
    h = h0.clone()
    ids = []
    for _ in range(k):
        tok = head(h[:, 3]).argmax(-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    draft_ids = torch.stack(ids, dim=1)
    x = embed(draft_ids)
    seq = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="sequential")
    par = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="newton")
    torch.testing.assert_close(par.n_accepted, seq.n_accepted)
    torch.testing.assert_close(par.hidden, seq.hidden, atol=2e-4, rtol=2e-4)
    assert int((par.n_accepted == k).sum()) == batch


@pytest.mark.cuda
def test_linear_draft_cuda_newton_matches_sequential() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    device = torch.device("cuda:0")
    model, embed, head = _pack(d=32, vocab=19, seed=3)
    model = model.to(device)
    embed = embed.to(device)
    head = head.to(device)
    batch, k = 4, 16
    h0 = torch.randn(batch, 32, device=device)
    draft_ids = _greedy(model, embed, head, h0, k)
    x = embed(draft_ids)
    seq = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="sequential")
    par = verify_linear_draft(model, x, draft_ids, head, h0=h0, solver="newton")
    torch.testing.assert_close(par.n_accepted, seq.n_accepted)
    torch.testing.assert_close(par.hidden, seq.hidden, atol=2e-4, rtol=2e-4)
