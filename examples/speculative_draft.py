"""Greedy linear-draft verify: Newton scan vs sequential step loop.

    uv run python examples/speculative_draft.py

Draft of K tokens from h0. One Newton unroll checks the chain; n_accepted
is the first mismatch (k*). Sequential ``step`` is the oracle. Typical
speculative γ is 4–8 (Leviathan et al.); K=64 is the length where a scan
starts to beat a Python step loop on this box. B=8, d=256. Local smoke:
no MLflow.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import NewtonConfig, ParaGRU, ParaRNN, verify_linear_draft

log = logging.getLogger("speculative_draft")

# App. A: K=3 Newton iters on ParaGRU (Danieli et al. 2025). Residual off:
# this is a timing / n_accepted smoke, not a train run.
_NEWTON_ITERS = 3
_BATCH = 8
_DIM = 256
_VOCAB = 1024
_DRAFT_KS = (8, 64)
_WARMUP = 3
_RUNS = 10


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _minmax_ms(fn, device: torch.device) -> tuple[float, float]:
    for _ in range(_WARMUP):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(_RUNS):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), float(sum(samples) / len(samples))


def _greedy_ids(
    cell: nn.Module, embed: nn.Embedding, head: nn.Linear, h0: torch.Tensor, k: int
) -> torch.Tensor:
    h = h0
    ids = []
    for _ in range(k):
        tok = head(h).argmax(dim=-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    return torch.stack(ids, dim=1)


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
    model = ParaRNN(ParaGRU(_DIM, _DIM), config=cfg).to(device)
    embed = nn.Embedding(_VOCAB, _DIM, device=device)
    head = nn.Linear(_DIM, _VOCAB, device=device)
    h0 = torch.randn(_BATCH, _DIM, device=device)
    cell = model.layers[0]

    draft_ids = _greedy_ids(cell, embed, head, h0, 8)
    full = verify_linear_draft(
        model, embed(draft_ids), draft_ids, head, h0=h0, solver="newton"
    )
    broken = draft_ids.clone()
    broken[:, 3] = (broken[:, 3] + 1) % _VOCAB
    mid = verify_linear_draft(
        model, embed(broken), broken, head, h0=h0, solver="newton"
    )
    seq = verify_linear_draft(
        model, embed(broken), broken, head, h0=h0, solver="sequential"
    )
    log.info(
        "agree gpu=%s greedy_n_accepted=%s broken_newton=%s broken_seq=%s",
        gpu,
        int(full.n_accepted.min()),
        int(mid.n_accepted.min()),
        int(seq.n_accepted.min()),
    )
    torch.testing.assert_close(mid.n_accepted, seq.n_accepted)

    for k in _DRAFT_KS:
        ids = _greedy_ids(cell, embed, head, h0, k)
        x = embed(ids)

        def _newton(xx: torch.Tensor = x, ii: torch.Tensor = ids) -> None:
            verify_linear_draft(model, xx, ii, head, h0=h0, solver="newton")

        def _seq(xx: torch.Tensor = x, ii: torch.Tensor = ids) -> None:
            verify_linear_draft(model, xx, ii, head, h0=h0, solver="sequential")

        n_min, n_mean = _minmax_ms(_newton, device)
        s_min, s_mean = _minmax_ms(_seq, device)
        log.info(
            "bench K=%s B=%s d=%s newton_min_ms=%.3f newton_mean_ms=%.3f "
            "seq_min_ms=%.3f seq_mean_ms=%.3f",
            k,
            _BATCH,
            _DIM,
            n_min,
            n_mean,
            s_min,
            s_mean,
        )


if __name__ == "__main__":
    main()
