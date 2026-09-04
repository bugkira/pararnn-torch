"""Paged (c,n,m,h) slot pool: mixed prefill + decode, then slot reuse.

    uv run python examples/paged_cache.py

State is O(1) per request. Continuous batching still needs a GPU pool:
allocate a slot, prefill (Newton or sequential), decode T=1 from that
slot, free when the request ends. Capacity 8 is a toy max-num-seqs, not
a production bound. Local smoke: no MLflow.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import NewtonConfig, PagedStatePool, ParaRNN, ParaSLSTM, paged_apply

log = logging.getLogger("paged_cache")

_DIM = 32
_CAPACITY = 8
_NEWTON_ITERS = 3


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    torch.manual_seed(0)
    cfg = NewtonConfig(
        max_iters=_NEWTON_ITERS, residual_atol=None, residual_fail=None
    )
    model = ParaRNN(ParaSLSTM(_DIM, _DIM, mix="diag"), config=cfg).to(device)
    pool = PagedStatePool(model, _CAPACITY)
    assert pool.buffers[0].shape == (_CAPACITY, 4, _DIM)

    a, b = pool.allocate(2).unbind(0)
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    x_a = torch.randn(1, 12, _DIM, device=device)
    x_b = torch.randn(1, 5, _DIM, device=device)
    paged_apply(pool, a, x_a, solver="newton")
    paged_apply(pool, b, x_b, solver="newton")
    log.info(
        "prefill gpu=%s slots_used=%s free=%s",
        gpu,
        pool.allocator.n_used,
        pool.allocator.n_free,
    )

    # Mixed step: decode A (T=1) and prefill C in one packed batch.
    c = pool.allocate(1)
    x_dec = torch.randn(1, 1, _DIM, device=device)
    x_c = torch.randn(1, 8, _DIM, device=device)
    packed = torch.cat((x_dec, x_c), dim=1)
    cs = torch.tensor([0, 1, 9], device=device)
    ids = torch.cat((a, c))
    paged_apply(pool, ids, packed, cu_seqlens=cs, solver="sequential")
    pool.free(b)
    d = pool.allocate(1)
    log.info(
        "after_mix used=%s free=%s reused_slot=%s (was B)",
        pool.allocator.n_used,
        pool.allocator.n_free,
        int(d.item()),
    )
    st = pool.gather(d)
    if not torch.equal(st, torch.zeros_like(st)):
        raise SystemExit("reused slot was not zeroed")
    log.info("ok sLSTM cache shape=%s", tuple(pool.buffers[0].shape))


if __name__ == "__main__":
    main()
