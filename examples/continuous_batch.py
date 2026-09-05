"""Continuous-batch prefill + decode on a ``ParaSLSTMForCausalLM`` pool.

Two requests share a ``BlockStackPool``. Prefill different lengths in one
packed batch; then decode one token each via ``block_table`` / slot ids.

Usage:
    uv run python examples/continuous_batch.py
"""

from __future__ import annotations

import logging

import torch

from pararnn.models import ParaSLSTMConfig, ParaSLSTMForCausalLM
from pararnn.solvers.newton import NewtonConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("continuous_batch")

VOCAB, D, LAYERS = 64, 32, 2
CAPACITY = 8


def main() -> None:
    torch.manual_seed(0)
    cfg = ParaSLSTMConfig(
        vocab_size=VOCAB,
        hidden_size=D,
        num_hidden_layers=LAYERS,
        mlp_ratio=2.0,
        newton_iters=2,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg).eval()
    for block in model.blocks:
        block.rnn.config = NewtonConfig(
            max_iters=2, scan_backend="eager", residual_fail=None
        )
    pool = model.attach_pool(CAPACITY)
    ids = pool.allocate(2)

    # Packed: request 0 length 5, request 1 length 3
    t0 = torch.randint(0, VOCAB, (1, 5))
    t1 = torch.randint(0, VOCAB, (1, 3))
    packed = torch.cat((t0, t1), dim=1)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    logits = model.forward_continuous(packed, ids, cu_seqlens=cu, solver="sequential")
    log.info(
        "prefill packed N=%s logits=%s slots_used=%s",
        packed.shape[1],
        tuple(logits.shape),
        pool.allocator.n_used,
    )

    # Decode one token per request (dense T=1)
    nxt = torch.randint(0, VOCAB, (2, 1))
    dec = model.forward_continuous(nxt, ids, solver="sequential")
    log.info("decode T=1 logits=%s", tuple(dec.shape))

    pool.free(ids)
    log.info("freed; n_used=%s", pool.allocator.n_used)


if __name__ == "__main__":
    main()
