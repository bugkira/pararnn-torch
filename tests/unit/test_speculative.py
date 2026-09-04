"""Greedy linear-draft verify: n_accepted, bonus, state gather."""

from __future__ import annotations

import torch
from torch import nn

from pararnn import NewtonConfig, ParaGRU, ParaLSTM, ParaRNN, verify_linear_draft


def _tiny_gru() -> tuple[ParaRNN, nn.Embedding, nn.Linear]:
    torch.manual_seed(0)
    d, vocab = 8, 11
    model = ParaRNN(
        ParaGRU(d, d),
        config=NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None),
        solver="newton",
    )
    embed = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab)
    return model, embed, head


def _greedy_chain(
    model: ParaRNN,
    embed: nn.Embedding,
    head: nn.Linear,
    h0: torch.Tensor | None,
    k: int,
    batch: int,
) -> torch.Tensor:
    cell = model.layers[0]
    h = h0 if h0 is not None else torch.zeros(batch, cell.hidden_size)
    ids = []
    for _ in range(k):
        tok = head(h).argmax(dim=-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    return torch.stack(ids, dim=1)


def test_full_accept_when_draft_is_greedy() -> None:
    model, embed, head = _tiny_gru()
    batch, k = 3, 6
    h0 = torch.randn(batch, 8)
    draft_ids = _greedy_chain(model, embed, head, h0, k, batch)
    got = verify_linear_draft(
        model, embed(draft_ids), draft_ids, head, h0=h0, solver="sequential"
    )
    assert int((got.n_accepted == k).sum()) == batch
    torch.testing.assert_close(got.hidden, _state_after(model, embed, h0, draft_ids))


def _state_after(
    model: ParaRNN, embed: nn.Embedding, h0: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    cell = model.layers[0]
    h = h0
    for t in range(ids.shape[1]):
        h = cell.step(h, embed(ids[:, t]))
    return h


def test_zero_accept_when_first_token_differs() -> None:
    model, embed, head = _tiny_gru()
    batch, k = 2, 5
    h0 = torch.randn(batch, 8)
    draft_ids = _greedy_chain(model, embed, head, h0, k, batch)
    draft_ids = draft_ids.clone()
    draft_ids[:, 0] = (draft_ids[:, 0] + 1) % 11
    got = verify_linear_draft(
        model, embed(draft_ids), draft_ids, head, h0=h0, solver="sequential"
    )
    assert torch.equal(got.n_accepted, torch.zeros(batch, dtype=torch.int64))
    torch.testing.assert_close(got.hidden, h0)
    greedy0 = head(h0).argmax(-1)
    torch.testing.assert_close(got.bonus_ids, greedy0)


def test_mid_mismatch_keeps_prefix_state() -> None:
    model, embed, head = _tiny_gru()
    batch, k = 1, 7
    h0 = torch.randn(batch, 8)
    draft_ids = _greedy_chain(model, embed, head, h0, k, batch)
    draft_ids = draft_ids.clone()
    draft_ids[:, 3] = (draft_ids[:, 3] + 1) % 11
    got = verify_linear_draft(
        model, embed(draft_ids), draft_ids, head, h0=h0, solver="sequential"
    )
    assert int(got.n_accepted.item()) == 3
    torch.testing.assert_close(got.hidden, _state_after(model, embed, h0, draft_ids[:, :3]))


def test_lstm_hidden_is_full_state() -> None:
    torch.manual_seed(1)
    d, vocab, batch, k = 6, 9, 2, 4
    model = ParaRNN(
        ParaLSTM(d, d),
        config=NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None),
    )
    embed = nn.Embedding(vocab, d)
    head = nn.Linear(d, vocab)
    h0 = torch.zeros(batch, 2, d)
    cell = model.layers[0]
    h = h0
    ids = []
    for _ in range(k):
        tok = head(h[:, 1]).argmax(-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    draft_ids = torch.stack(ids, dim=1)
    got = verify_linear_draft(
        model, embed(draft_ids), draft_ids, head, h0=h0, solver="sequential"
    )
    assert int((got.n_accepted == k).sum()) == batch
    assert got.hidden.shape == (batch, 2, d)
    torch.testing.assert_close(got.hidden, h)
