"""Tiny CausalLM smoke: train with labels, generate, save/load.

Usage:
    uv run python examples/causal_lm_smoke.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch

from pararnn import ParaSLSTMConfig, ParaSLSTMForCausalLM
from pararnn.solvers.newton import NewtonConfig

# Tiny width for a laptop CPU smoke. K=2 / eager keeps the Newton path cheap.
VOCAB, HIDDEN, LAYERS, SEQ, BATCH = 64, 32, 2, 16, 4
STEPS, LR, SEED = 3, 3e-3, 0
MAX_NEW = 4

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("causal_lm_smoke")

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = ParaSLSTMConfig(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    mlp_ratio=2.0,
    newton_iters=2,
    scan_backend="eager",
    max_recurrent_norm=0.5,
)
model = ParaSLSTMForCausalLM(cfg).to(device)
for block in model.blocks:
    block.rnn.config = NewtonConfig(max_iters=2, scan_backend="eager", residual_fail=None)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
log.info("causal_lm_smoke start device=%s", device)

model.train()
loss0 = None
for step in range(STEPS):
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    _logits, loss = model(ids, labels=ids)
    if step == 0:
        loss0 = float(loss.detach())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    log.info("step=%d loss=%.4f", step, float(loss.detach()))

assert loss0 is not None
log.info("train ok loss0=%.4f loss_last=%.4f", loss0, float(loss.detach()))

model.eval()
prompt = torch.randint(0, VOCAB, (1, 4), device=device)
out = model.generate(prompt, max_new_tokens=MAX_NEW, temperature=0.0)
assert out.shape == (1, 4 + MAX_NEW)
log.info("generate ok shape=%s", tuple(out.shape))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "ckpt"
    model.save_pretrained(path)
    restored = ParaSLSTMForCausalLM.from_pretrained(path, map_location=device)
    for block in restored.blocks:
        block.rnn.config = NewtonConfig(max_iters=2, scan_backend="eager", residual_fail=None)
    restored.eval()
    with torch.no_grad():
        a = model(prompt)
        b = restored(prompt)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    assert (path / "model.safetensors").is_file()
    log.info("save/load ok dir=%s", path)

log.info("causal_lm_smoke done")
