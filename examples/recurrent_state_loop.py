"""Minimal recurrent carry loop: prefill → T=1 decode_step with out=.

Shows the inference contract: fixed-size state between tokens (not a KV
cache). On CUDA, buffers are reusable for CUDA Graph capture (see also
``examples/decode_step.py``).

Usage:
    uv run python examples/recurrent_state_loop.py
"""

from __future__ import annotations

import torch

from pararnn import ParaSLSTM, decode_step, decode_wx, sequential_apply

SEED, BATCH, DIM, PREFILL, DECODE = 0, 2, 64, 12, 5

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaSLSTM(DIM, DIM, mix="diag").to(device).eval()

prompt = torch.randn(BATCH, PREFILL, DIM, device=device)
# Last time step is the recurrent carry for the next token.
carry = sequential_apply(cell, prompt)[:, -1].contiguous()

wx_buf = torch.empty(BATCH, cell.W_x.out_features, device=device, dtype=prompt.dtype)
out_buf = torch.empty_like(carry)

print(f"device={device} carry_shape={tuple(carry.shape)}  # fixed; does not grow with DECODE")

with torch.no_grad():
    for t in range(DECODE):
        x_t = torch.randn(BATCH, DIM, device=device)
        decode_wx(cell, x_t, out=wx_buf)
        decode_step(cell, carry, wx=wx_buf, out=out_buf)
        carry = out_buf
        # Same storage next iteration: out_buf holds the new carry.
        print(f"step={t + 1} max|h|={float(carry[..., -1, :].abs().amax()):.4f}")

print("ok")
