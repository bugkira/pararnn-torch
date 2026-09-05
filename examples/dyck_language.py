"""Dyck-1 next-token smoke: ParaSLSTM + Newton grads.

K=3, Picard P from T (P=1 at T=16). Fail-loud if CE does not drop.

Usage:
    python dyck_language.py
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pararnn import NewtonConfig, ParaRNN, ParaSLSTM

OPEN, CLOSE = 0, 1
VOCAB = 2
SEQ_LEN, BATCH, D_H = 16, 32, 32
STEPS, LR, SEED = 50, 3e-3, 0
SCAN_BACKEND = "auto"


def sample_dyck1(batch: int, length: int, *, generator: torch.Generator | None = None) -> Tensor:
    """Even-length Dyck-1 words. Token 0='(', 1=')'."""
    if length < 2 or length % 2:
        raise ValueError(f"Dyck-1 length must be even and >=2, got {length}")
    out = torch.empty(batch, length, dtype=torch.long)
    for b in range(batch):
        depth = 0
        for t in range(length):
            remain = length - t
            if depth == 0:
                tok = OPEN
            elif remain == depth:
                tok = CLOSE
            else:
                tok = int(torch.randint(0, 2, (1,), generator=generator).item())
                if tok == CLOSE and depth == 0:
                    tok = OPEN
            depth += 1 if tok == OPEN else -1
            out[b, t] = tok
        if depth != 0:
            raise RuntimeError(f"Dyck sampler ended at depth={depth}")
    return out


if __name__ == "__main__":
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rnn_layer = ParaRNN(
        ParaSLSTM(d_in=D_H, d_h=D_H, mix="diag"),
        config=NewtonConfig(max_iters=3, scan_backend=SCAN_BACKEND),
        output_hidden=True,
    )
    model = nn.Sequential(
        nn.Embedding(VOCAB, D_H),
        rnn_layer,
        nn.Linear(D_H, VOCAB),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # One backward: eq. 2.6 grads finite.
    tokens = sample_dyck1(4, min(SEQ_LEN, 8), generator=torch.Generator().manual_seed(0)).to(device)
    loss = F.cross_entropy(model(tokens[:, :-1]).reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    loss.backward()
    bad = [
        n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"non-finite or missing grads: {bad}"
    model.zero_grad(set_to_none=True)
    print(f"Dyck start: device={device}, grads ok")

    gen = torch.Generator().manual_seed(SEED)
    loss0 = None
    for step in range(STEPS):
        tokens = sample_dyck1(BATCH, SEQ_LEN, generator=gen).to(device)
        loss = F.cross_entropy(model(tokens[:, :-1]).reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
        if step == 0:
            loss0 = loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0 or step == STEPS - 1:
            st = rnn_layer.last_stats[0] if rnn_layer.last_stats else None
            res = f"{st.max_residual:.2e}" if st else "nan"
            backend = (st.scan_backend or SCAN_BACKEND) if st else SCAN_BACKEND
            print(f"step={step:03d} | loss={loss.item():.4f} | res={res} | backend={backend}")

    loss_final = loss.item()
    assert loss_final < loss0, (
        f"Dyck failed: final loss ({loss_final:.4f}) >= initial ({loss0:.4f})"
    )
    print(f"Dyck OK: {loss0:.4f} -> {loss_final:.4f}")
