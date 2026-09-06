"""Thin RSSM recurrent slot: ParaGRU(mix='head') train vs eval.

Drop this cell into a Dreamer-style world model where
``h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))``. This script only times the
recurrent slot (parallel Newton in train, sequential step in eval) on a fake
imagination batch. No Dreamer / cleanrl stack.

Usage:
    uv run python examples/rssm_recurrent.py
"""

from __future__ import annotations

import logging
import time

import torch

from pararnn import NewtonConfig, ParaGRU, ParaRNN

# Dreamer-like block size: d_h=512, n_heads=8 → d_head=64 (SRAM factorized path).
# Horizon H=50 matches a common imagination depth; B=8 is a small batch.
D_H, N_HEADS, Z_DIM, A_DIM = 128, 4, 32, 16
BATCH, HORIZON, SEED = 4, 32, 0
NEWTON_K = 3

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rssm_recurrent")

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_in = Z_DIM + A_DIM

cell = ParaGRU(d_in=d_in, d_h=D_H, mix="head", n_heads=N_HEADS, device=device)
rnn = ParaRNN(
    cell,
    config=NewtonConfig(max_iters=NEWTON_K, scan_backend="auto", residual_fail=None),
).to(device)

# Fake latent + action stream over an imagination horizon.
z = torch.randn(BATCH, HORIZON, Z_DIM, device=device)
a = torch.randn(BATCH, HORIZON, A_DIM, device=device)
x = torch.cat([z, a], dim=-1)

log.info(
    "rssm_recurrent start device=%s B=%s H=%s d_h=%s n_heads=%s d_in=%s",
    device,
    BATCH,
    HORIZON,
    D_H,
    N_HEADS,
    d_in,
)

rnn.train()
t0 = time.perf_counter()
h_train = rnn(x)
if device.type == "cuda":
    torch.cuda.synchronize()
dt_train_ms = (time.perf_counter() - t0) * 1e3
loss = h_train.float().pow(2).mean()
loss.backward()
if device.type == "cuda":
    torch.cuda.synchronize()
log.info(
    "train Newton forward shape=%s loss=%.4f wall_fwd_ms=%.2f",
    tuple(h_train.shape),
    float(loss.detach()),
    dt_train_ms,
)

rnn.eval()
with torch.no_grad():
    t0 = time.perf_counter()
    h_eval = rnn(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_eval_ms = (time.perf_counter() - t0) * 1e3
log.info(
    "eval sequential shape=%s wall_fwd_ms=%.2f",
    tuple(h_eval.shape),
    dt_eval_ms,
)

assert h_train.shape == (BATCH, HORIZON, D_H)
assert h_eval.shape == h_train.shape
log.info("rssm_recurrent done (wire this ParaRNN into your RSSM GRU slot)")
