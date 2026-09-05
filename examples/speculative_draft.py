"""Greedy linear-draft verify: Newton scan vs sequential step loop.

Draft of K tokens from h0. One Newton unroll checks the chain; n_accepted
is the first mismatch. Typical speculative γ is 4–8; K=64 is where a scan
starts to beat a Python step loop on a mid-range GPU.

Usage:
    python speculative_draft.py
"""

import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from torch import nn

from pararnn import NewtonConfig, ParaGRU, ParaRNN, verify_linear_draft

# App. A: K=3 Newton iters on ParaGRU (Danieli et al. 2025).
NEWTON_ITERS, BATCH, DIM, VOCAB = 3, 8, 256, 1024
DRAFT_KS = (8, 64)
WARMUP, RUNS = 3, 10

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = NewtonConfig(max_iters=NEWTON_ITERS, residual_atol=None, residual_fail=None)
model = ParaRNN(ParaGRU(DIM, DIM), config=cfg).to(device)
embed = nn.Embedding(VOCAB, DIM, device=device)
head = nn.Linear(DIM, VOCAB, device=device)
h0 = torch.randn(BATCH, DIM, device=device)
cell = model.layers[0]


def _sync() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _minmax_ms(fn) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    _sync()
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), sum(samples) / len(samples)


def _greedy_ids(h0_: torch.Tensor, k: int) -> torch.Tensor:
    h = h0_
    ids = []
    for _ in range(k):
        tok = head(h).argmax(dim=-1)
        ids.append(tok)
        h = cell.step(h, embed(tok))
    return torch.stack(ids, dim=1)


draft_ids = _greedy_ids(h0, 8)
full = verify_linear_draft(model, embed(draft_ids), draft_ids, head, h0=h0, solver="newton")
broken = draft_ids.clone()
broken[:, 3] = (broken[:, 3] + 1) % VOCAB
mid = verify_linear_draft(model, embed(broken), broken, head, h0=h0, solver="newton")
seq = verify_linear_draft(model, embed(broken), broken, head, h0=h0, solver="sequential")
torch.testing.assert_close(mid.n_accepted, seq.n_accepted)
print(
    f"agree device={device} greedy_n={int(full.n_accepted.min())} "
    f"broken_newton={int(mid.n_accepted.min())} broken_seq={int(seq.n_accepted.min())}"
)

for k in DRAFT_KS:
    ids = _greedy_ids(h0, k)
    x = embed(ids)
    n_min, n_mean = _minmax_ms(
        lambda xx=x, ii=ids: verify_linear_draft(model, xx, ii, head, h0=h0, solver="newton")
    )
    s_min, s_mean = _minmax_ms(
        lambda xx=x, ii=ids: verify_linear_draft(model, xx, ii, head, h0=h0, solver="sequential")
    )
    print(f"K={k} newton {n_min:.3f}/{n_mean:.3f} ms | seq {s_min:.3f}/{s_mean:.3f} ms")
