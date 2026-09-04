"""Greedy linear-draft verify: one Newton scan of K tokens.

Leviathan et al. speculative decoding, RNN form. The target consumes the
draft as a length-K sequence from ``h0``. Readout on
``cat(h0, H[:, :K])`` gives K+1 logits. The first mismatch against the
draft is k*; the kept state is h_{k*} (h0 when k* = 0). The leftover
logit is the bonus token.

A draft tree (Medusa / tree attention) is a different scan (branch mask
or batched branches). This module is the chain of K tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from pararnn.layers.para_rnn import (
    ParaRNN,
    _hidden_slot,
    _next_layer_input,
    _split_h0,
    _state_shape,
    _validate_h0s,
    _validate_input,
)
from pararnn.solvers.newton import newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)

_Readout = Callable[[Tensor], Tensor]


@dataclass
class LinearDraftResult:
    """Greedy verify of a length-K token chain.

    ``n_accepted`` is the exclusive length of the matching prefix (0..K).
    ``hidden`` is ``h0`` when that length is 0, else the last-layer (or
    per-layer) state after the last accepted draft token. ``bonus_ids`` is
    the target greedy token at the first mismatch, or the extra token after
    a full accept.
    """

    n_accepted: Tensor
    hidden: Tensor | tuple[Tensor, ...]
    bonus_ids: Tensor
    logits: Tensor


@torch.no_grad()
def verify_linear_draft(
    model: ParaRNN,
    draft_x: Tensor,
    draft_ids: Tensor,
    readout: _Readout,
    *,
    h0: Tensor | Sequence[Tensor] | None = None,
    solver: Literal["newton", "sequential"] = "newton",
) -> LinearDraftResult:
    """Verify a linear draft with one parallel (or sequential) unroll.

    ``draft_x``: ``(B, K, d_in)`` embeddings of the drafted tokens.
    ``draft_ids``: ``(B, K)`` int64 token ids. ``readout`` maps
    ``(B, T, d_h)`` hidden slots to ``(B, T, vocab)`` logits.

    ``solver='newton'`` is the serving path (Alg. 1, App. A K=3 on the
    model's ``NewtonConfig``). ``'sequential'`` is the ``step`` oracle.
    """
    if solver not in ("newton", "sequential"):
        raise ValueError(f"solver must be 'newton' or 'sequential', got {solver!r}")
    if not model.batch_first:
        raise ValueError("verify_linear_draft needs batch_first=True")
    _validate_input(draft_x, model.layers[0], batch_first=True)
    if draft_ids.shape[:2] != draft_x.shape[:2]:
        raise ValueError(
            f"draft_ids {tuple(draft_ids.shape)} vs draft_x {tuple(draft_x.shape)}"
        )
    if draft_x.shape[1] < 1:
        raise ValueError(f"draft length K must be >= 1, got {draft_x.shape[1]}")
    batch, time = int(draft_x.shape[0]), int(draft_x.shape[1])
    h0s = _split_h0(h0, len(model.layers))
    _validate_h0s(h0s, model.layers, batch=batch)

    apply = newton_apply if solver == "newton" else sequential_apply
    h = draft_x
    layer_states: list[Tensor] = []
    for i, cell in enumerate(model.layers):
        if solver == "newton":
            h = apply(cell, h, model.config, h0=h0s[i])
        else:
            h = apply(cell, h, h0s[i])
        layer_states.append(h)
        if i + 1 < len(model.layers):
            h = _next_layer_input(h, cell)

    last_cell = model.layers[-1]
    slot = _hidden_slot(last_cell)
    prefix_h = _hidden_from_h0(
        h0s[-1], slot, batch=batch, cell=last_cell, like=draft_x
    )
    scan_h = layer_states[-1] if slot is None else layer_states[-1][:, :, slot, :]
    hidden_bt = torch.cat((prefix_h.unsqueeze(1), scan_h), dim=1)
    logits = readout(hidden_bt)
    if logits.shape[:2] != (batch, time + 1):
        raise ValueError(
            f"readout returned {tuple(logits.shape)}, expected (B, K+1, vocab) "
            f"= ({batch}, {time + 1}, …)"
        )
    pred_ids = logits.argmax(dim=-1)
    match = pred_ids[:, :time] == draft_ids
    n_accepted = match.to(dtype=torch.int64).cumprod(dim=1).sum(dim=1)
    bonus_ids = pred_ids[torch.arange(batch, device=pred_ids.device), n_accepted]

    kept: list[Tensor] = []
    for states, h0_i, cell in zip(layer_states, h0s, model.layers, strict=True):
        kept.append(_gather_state(states, h0_i, n_accepted, cell, like=draft_x))
    hidden: Tensor | tuple[Tensor, ...] = kept[0] if len(kept) == 1 else tuple(kept)

    if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
        log.debug(
            "verify_linear_draft",
            extra={
                "solver": solver,
                "batch": batch,
                "K": time,
                "n_accepted_min": int(n_accepted.min()),
                "n_accepted_max": int(n_accepted.max()),
                "n_accepted_mean": float(n_accepted.float().mean()),
            },
        )
    return LinearDraftResult(
        n_accepted=n_accepted,
        hidden=hidden,
        bonus_ids=bonus_ids,
        logits=logits,
    )


def _hidden_from_h0(
    h0: Tensor | None,
    slot: int | None,
    *,
    batch: int,
    cell: nn.Module,
    like: Tensor,
) -> Tensor:
    if h0 is None:
        zeros = like.new_zeros(_state_shape(cell, batch))
        return zeros if slot is None else zeros[:, slot]
    return h0 if slot is None else h0[:, slot]


def _gather_state(
    states: Tensor,
    h0: Tensor | None,
    n_accepted: Tensor,
    cell: nn.Module,
    *,
    like: Tensor,
) -> Tensor:
    batch = states.shape[0]
    idx = (n_accepted - 1).clamp(min=0)
    gathered = states[torch.arange(batch, device=states.device), idx]
    if h0 is None:
        h0 = like.new_zeros(_state_shape(cell, batch))
    mask = n_accepted.gt(0).reshape((batch,) + (1,) * (gathered.ndim - 1))
    return torch.where(mask, gathered, h0)
