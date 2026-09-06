# Long-T / OOM cookbook

A `CUDA out of memory` on the first train steps usually means the cell’s
memory geometry does not fit the card (hidden width, heads, or stored
\(H^*\)). This page is the cheat sheet before you open an issue.

Silent wrong answers (Newton vs sequential) are a different failure mode —
see [`numerics-contract.md`](numerics-contract.md).

## Cheat sheet

1. **Train GRU / sLSTM (diag) toward ~100k tokens?**  
   Use `NewtonConfig(recompute=True)` so backward rematerializes \(H^*\)
   instead of keeping the full trajectory in `save_for_backward`.

2. **Hopfield?**  
   Keep `d_h ≤ 32`. The dense Jacobian + `scan_dense` path is
   \(\mathcal{O}(d_h^3)\) per step; above 32 the cell warns at init.
   Lab: \(d_h=32\), \(T=2048\) peaks ~617 MiB (dense \(J\)).

3. **RWKV-7 on long context (consumer 12 GiB)?**  
   Use **slim heads**: `n_heads=1`, `d_head=16`. State is
   `(B, T, n_heads, d_head, d_head)` — a fat `4×16` (or larger) layout OOMs
   near \(T \gtrsim 64\mathrm{k}\) on 12 GiB. README long-T benches use slim.

```python
from pararnn import NewtonConfig, ParaGRU, ParaRNN

# Ultra-long train: Level-2 rematerialize H* in backward (eq. 2.6 unchanged).
cfg = NewtonConfig(max_iters=3, recompute=True)
model = ParaRNN(ParaGRU(64, 64), config=cfg)
```

## Three VRAM hungers (pick the right lever)

| Hunger | Scales like | Typical cell | First lever |
|---|---|---|---|
| Stored Newton trajectory \(H^*\) | \(B\cdot T\cdot S\cdot d\) | diag GRU / sLSTM / LSTM train | `recompute=True` |
| Dense Jacobian / scan workspace | \(B\cdot T\cdot d^2\) (+ \(d^3\) work) | Hopfield, dense oracle | smaller \(d_h\), or another cell |
| Matrix state over time | \(T\cdot n_{\mathrm{heads}}\cdot d_{\mathrm{head}}^2\) | RWKV-7, M²RNN | slim heads / fewer heads |

`recompute=True` addresses the stored-trajectory row. Hopfield dense \(J\) and
RWKV matrix state need a smaller geometry first.

## Decision tree

```
OOM / near-OOM?
├─ ParaHopfield ─────────────────── d_h ≤ 32 (warn if larger)
├─ ParaRWKV7, T ≳ 32k–64k on 12 GiB ─ n_heads=1, d_head=16 (slim)
├─ diag GRU / LSTM / sLSTM train, T ≳ 64k–128k
│     └─ NewtonConfig(recompute=True)
│           optional: outer torch.utils.checkpoint on blocks
│           optional: chunk_len / fused_time_loop (windowed solve;
│                     check agreement — see numerics-contract)
└─ head-mix GRU / sLSTM ──────────── factorized path is already leaner
                                      than dense J; still store H* unless
                                      recompute=True
```

Inference / `.eval()` / `decode_step` (\(T=1\)) rarely need `recompute`.
The pain is **train** with a stored full-sequence \(H^*\).

## Lab anchors (order of magnitude)

Measured on this repo’s cards; treat as envelopes, not SLAs.

| Setup | Note |
|---|---|
| Hopfield `d_h=32`, `T=2048` | ~617 MiB peak (dense \(J\)) |
| RWKV-7 slim `1×16` @ `T=131072` | fits long-T campaign on RTX 3060 (README) |
| RWKV-7 fat `4×16` @ `T≳64k` | OOM risk on 12 GiB |
| Diag-sLSTM / GRU long-T | K* often flat; VRAM of \(H^*\) still grows with \(T\) |

More scan/T limits: [`backward-scan-cap.md`](backward-scan-cap.md).
Cell snippets: [`cells.md`](cells.md).

## Peak-memory smoke (paste into a bug)

```python
import torch
from pararnn import NewtonConfig, ParaRNN, ParaGRU  # or your cell

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device)
model = ParaRNN(ParaGRU(64, 64), config=NewtonConfig(max_iters=3, recompute=True)).to(device)
x = torch.randn(1, 8192, 64, device=device)  # your B, T, d_in
model.train()
y = model(x)
y.sum().backward()
peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
print(f"peak_mib={peak_mib:.1f} T={x.shape[1]} recompute=True")
```

Log `peak_mib`, cell, `mix` / `n_heads` / `d_head`, dtype, and GPU name with
the issue.

## Related knobs

| Knob | Role |
|---|---|
| `NewtonConfig(recompute=True)` | Level-2: drop \(H^*\) from the Autograd Function; rematerialize in backward |
| `chunk_len` / `fused_time_loop` | Windowed Newton along \(T\) (different residual path — verify agreement) |
| `torch.utils.checkpoint` | Outer rematerialization across stacked blocks |
| `max_iters=None` | Measured \(K^*(T)\) — numerics budget, not a VRAM fix |

Tests: `tests/numerics/test_recompute.py`.
