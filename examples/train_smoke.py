"""Training smoke: ParaRNN (ParaGRU) + linear head + CE + AdamW.

Verifies that Newton gradients and AdamW decrease loss in ~100 steps.
Usage:
    python train_smoke.py
"""

import torch
from torch import nn
from torch.nn import functional as F

from pararnn import NewtonConfig, ParaGRU, ParaRNN

# Smoke width: small alphabet, short T. K=3 from Danieli et al. 2025 App. A.
VOCAB, SEQ_LEN, BATCH, D_H = 8, 16, 32, 32
STEPS, LR, SEED = 100, 3e-3, 0
SCAN_BACKEND = "auto"

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn_layer = ParaRNN(
    ParaGRU(d_in=D_H, d_h=D_H),
    config=NewtonConfig(max_iters=3, scan_backend=SCAN_BACKEND),
)
model = nn.Sequential(
    nn.Embedding(VOCAB, D_H),
    rnn_layer,
    nn.Linear(D_H, VOCAB),
).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

print(f"Smoke start: device={device}, backend={SCAN_BACKEND}")

loss0 = None
for step in range(STEPS):
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
    logits = model(tokens)

    loss = F.cross_entropy(logits.view(-1, VOCAB), tokens.view(-1))
    if step == 0:
        loss0 = loss.item()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 20 == 0 or step == STEPS - 1:
        st = rnn_layer.last_stats[0] if rnn_layer.last_stats else None
        res = f"{st.max_residual:.2e}" if st else "nan"
        backend = (st.scan_backend or SCAN_BACKEND) if st else SCAN_BACKEND
        print(f"step={step:03d} | loss={loss.item():.4f} | res={res} | backend={backend}")

loss_final = loss.item()
assert loss_final < loss0, (
    f"Smoke failed: final loss ({loss_final:.4f}) >= initial ({loss0:.4f})"
)
print(f"Smoke OK: {loss0:.4f} -> {loss_final:.4f}")
