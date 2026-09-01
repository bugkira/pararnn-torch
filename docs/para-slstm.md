# Para-sLSTM (research branch)

Sequential sLSTM + Newton. Fused Triton is `mix='diag'` only. Not FlashRNN.
Not TinyStories.

## Why this cell

xLSTM (Beck et al., NeurIPS 2024): **mLSTM** is the associative / scanable
memory; **sLSTM** is the nonlinear one (exp gates, stabilizer `max`,
normalizer `n`, mixing `R h`). FlashRNN keeps sLSTM sequential. ParaRNN fused
GRU/LSTM does not implement this cell.

This repo's earlier literature note had sLSTM/mLSTM swapped. The code
follows Beck / `ADD_TASK.md`.

## Cell

`ParaSLSTM`: state `(B, T, 4, d_h) = (c, n, m, h)`.

- `mix='diag'`: `R` is `(4, d_h)`, channelwise. Jacobian is 4×4 per channel
  (`cell.jac_structure='block4'`). Scan is `scan_block4` (eager or Triton).
  `scan_backend='fused'` / `'auto'` on CUDA runs cell + analytic J + 4×4
  scan in one Triton kernel. Head/dense are not fused (not 4×4 SRAM).
- `mix='head'`: `R` is `(4, n_heads, d_head, d_head)`. Dense mix inside a head,
  zeros across heads (Beck / xLSTM). Jacobian is `4 d_head × 4 d_head` per
  head; scan folds heads into the batch of `scan_dense`. `n_heads` must
  divide `d_h`.
- `mix='dense'`: `R` is `Linear(d_h, 4 d_h)`. Exact mixing; scan is
  `O(T (4d)^3)`. Tests use `d_h≤4`.

`K=3` is **not** assumed. Global `NewtonConfig` stays App. A (`K=3`,
`omega=1`) for ParaGRU/LSTM.

## Measured (2080 Ti, seed 101, diag mix, `T=12`, `d_h=4`)

These rows are **App. A** `f(0, x_t)` (old init). Current Newton uses
`zero_hidden_init` and snaps at **K=3** on this seed (and at T=48).

Sequential max-abs error. Same seed as `test_slstm_diag_newton_vs_sequential`.
block4 vs forced `jac_structure='dense'` agrees — the overshoot was the
Newton map (`max`/`exp`) plus a far guess, not 4×4 vs flattened packing.

| K | ω=1, clip=0.5 | ω=0.5, clip=0.5 | ω=1, clip=0.25 |
|---|---|---|---|
| 1 | 3.09 | 4.94 | 3.06 |
| 2 | 7.95 | 2.77 | 7.86 |
| 3 | 12.1 | 3.93 | 12.0 |
| 4 | **2.8e-6** | 1.91 | **2.5e-6** |
| 5 | 9.5e-7 | 0.94 | 1.9e-6 |

`omega=0.5` (Gonzalez et al. ELK) lowers K=3 overshoot but **kills the K=4
snap** under App. A. With zero-hidden init, K=3 already snaps at ω=1.
Do not copy ELK as the sLSTM default. Do not change the library Newton default.

Dense mix (seed 102, `d_h=3`): smoother; K=3 already ~1.7e-2, K=4 ~5e-7.

Head mix (seed 105, `d_h=4`, `n_heads=2`, `T=8`): smoother than diag; snaps
at K=4. Per-head J matches forced dense. `R h` is
`einsum('...nd,gnde->...gne')` (matmul inside the head, not a sum over the
output dim).

| K | seq err |
|---|---|
| 1 | 1.51 |
| 2 | 1.19 |
| 3 | 3.6e-2 |
| 4 | **9.5e-7** |
| 5 | 4.8e-7 |

Diag overshoot is the no-mixing cell, not a reason to drop heads. Prototype
head recipe: **K=4, omega=1, clip=0.5**. Scan cost is
`O(T n_heads (4 d_head)^3)`, not `O(T (4 d_h)^3)`.

## Analytic Jacobian

`step_with_jacobian` is the Newton `auto` path. ADD_TASK §2.4 is the
channelwise skeleton; the code also chains `tanh(z_z)`, `σ(z_o)`, and
`n+ε`. `torch.maximum` at ties splits 0.5/0.5 (PyTorch). Head/dense J is
packed `(4 d × 4 d)` per head: columns `c,n,m` stay channelwise, columns
`h` are dense through `R`.

## Triton

`scan_block4` tiles `BLOCK_T=32`, `BLOCK_D=16` (20 scan lanes; 2×2's 64×16
does not fit). Chunk scan uses `BLOCK_D=8`. Cap `T=2048`. Algebra is fp32
inside the kernel; DRAM may be fp16. Reverse scan is the same kernel on
`J^T`.

Fused Newton inlines the diag cell, `_jac_channelwise`, and the 4×4 scan
(same pattern as `newton_lstm.py`). `W_x(x)` stays a PyTorch GEMM. Fused
always runs `max_iters` (no residual early-stop). The guess is
`zero_hidden_init` (below), not App. A.

## Newton init: zero-hidden unroll, not App. A

App. A \(h_t^0 = f(0, x_t)\) zeros `(c,n,m)` independently. GRU/LSTM at
T=48, K=3: seq err ~5e-8. sLSTM's `n` accumulates, so that guess is ~30
from sequential at T=48 and needs K=12.

`slstm_zero_hidden_init`: drop `R h`, keep the running max/normalizer.
`m_t = P_t + max_k(z_{i,k} - P_k)` with `P = cumsum(z_f)`; then `n` and
`c` are `scan_diag`. Mixing is Newton's job. On the T=48 seed-101 draw
the guess is ~3.6 from sequential and **K=3 snaps** (~1e-6). Fused uses
the same guess. Global `NewtonConfig.max_iters=3` is enough for this
cell; do not raise it for GRU.

App. A residual vs K (same draw, for the record — not the current code):

| T | snap K | K=5 | K=12 |
|---|---|---|---|
| 12 | 5 | **5e-7** | 0 |
| 48 | 10 | 29 | **2e-6** |
| 64 | 10 | 194 | **4e-6** |

Quadratic basin is small (noise 0.1 around sequential diverges). `omega<1`
still kills the snap. Do not copy ELK as the default.

## Timing (2080 Ti smoke, not App. B)

`uv run python scripts/bench_time.py --config configs/bench/newton_slstm.yaml`.
MLflow `newton-slstm-bench`. CSV: `outputs/bench_newton_slstm.csv` (gitignored).
**10 warmup / 50 runs, min ms.** Same shapes as the GRU/LSTM fused table:
B=8, \(d_{\mathrm{in}}=d_h=256\), float32, \(K=3\), `x_scale=1`. Fused T cap
is 2048 (`BLOCK_D=8`, `_CHUNK_PAD=64`). Not FlashRNN. Not fig. 2/5.

**These times at \(T\ge 256\) are kernel throughput of a Newton that has not
matched sequential.** Do not quote the vs-RNN column as sequential-equivalent
work. GRU/LSTM at the same shapes stay \(\sim 10^{-7}\) at \(K=3\); sLSTM
does not. See the agreement table below.

Baselines: naive RNN = `sequential_apply`; naive ParaRNN = eager Newton +
Blelloch; fused = `scan_backend="fused"` (diag only). Compiled sequential is
`sequential_apply_compiled` (`reduce-overhead`).

| T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs RNN | vs compiled | vs eager N |
|---|---|---|---|---|---|---|---|
| 64 | 43.5 | 23.9 | 28.3 | **8.81** | 4.9× | 2.7× | 3.2× |
| 256 | 169 | 101 | 39.1 | **13.7** | 12× | 7.4× | 2.9× |
| 512 | 348 | 206 | 51.8 | **16.3** | 21× | 13× | 3.2× |
| 1024 | 727 | 404 | 77.4 | **21.7** | 34× | 19× | 3.6× |
| 2048 | 1411 | 826 | 140 | **30.3** | 47× | 27× | 4.6× |

Times in ms (min). Peak allocated MiB, same smoke:

| T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs eager N |
|---|---|---|---|---|---|
| 64 | 23 | 16 | 71 | **27** | 2.6× |
| 256 | 61 | 33 | 252 | **78** | 3.2× |
| 512 | 112 | 37 | 494 | **145** | 3.4× |
| 1024 | 214 | 73 | 978 | **280** | 3.5× |
| 2048 | 418 | 145 | 1946 | **550** | 3.5× |

Fused vs eager Newton is **~3–5×**, same class as GRU/LSTM's memory win
(4×4 \(J\) stays in SRAM). Wall time vs GRU fused at \(T=2048\) is **30 ms
vs 2.4 ms** (~13× slower): 20-lane 4×4 tiles vs diag 1-lane, and four slots.
Do not paste GRU's 11×/400× onto this cell.

## Agreement at width 256

Smoke above (B=8, `x_scale=1`, unseeded weights, \(K=3\)), max |par − seq|:

| T | eager | fused |
|---|---|---|
| 64 | 9.8e-3 | 9.8e-3 |
| 256 | 17 | 17 |
| 512 | 2.6e4 | 2.6e4 |
| 1024 | 5.3e3 | 3.7e5 |
| 2048 | 1.5e14 | 4.4e11 |

Tol in the YAML is \(10^{-4}\). Only \(T=64\) is even close, and it is **not**
a snap. At \(T=1024/2048\) fused and eager Newton diverge to **different**
garbage — both wrong. `require_agreement: false` so the timing run still
finishes.

K-curve, **seed 0**, B=2, same width, `residual_atol=None` (K means K).
`x_scale=1` is the bench input; `0.3` is the numerics-test scale.

| T | scale | K=3 | K=5 | K=8 |
|---|---|---|---|---|
| 64 | 1.0 | **7e-6** | 8e-6 | 8e-6 |
| 256 | 1.0 | 2.0 | **5e-5** | 2e-5 |
| 512 | 1.0 | 31 | 13 | **8e-5** |
| 1024 | 1.0 | 5.8e6 | 3.5e3 | 7.8e5 |
| 2048 | 1.0 | 1.8e5 | 6.8e5 | 1.3e5 |
| 64 | 0.3 | **1e-5** | 8e-6 | 1e-5 |
| 256 | 0.3 | 0.20 | **5e-5** | 4e-5 |
| 512 | 0.3 | 0.24 | **8e-5** | 9e-5 |
| 1024 | 0.3 | 29 | 0.70 | **2e-4** |
| 2048 | 0.3 | 94 | 110 | — |

Fused tracks eager while both are in the basin (T≤512). Zero-hidden init
fixes toy \(T=48\), \(d_h=4\). At GRU-table width it is **not** enough for
library \(K=3\) past a few hundred tokens. Raising K at \(T=2048\),
`x_scale=1` does not snap; it stays ~\(10^5\). Do not raise global
`NewtonConfig.max_iters` for GRU/LSTM to paper over this.

## Not yet

Packed VJP, fused head mix, FlashRNN bench, LM train.
