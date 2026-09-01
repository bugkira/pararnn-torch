# Para-sLSTM


Sequential sLSTM + Newton. Fused Triton is `mix='diag'` only. Not FlashRNN.
Not TinyStories.

## Why this cell

xLSTM (Beck et al., NeurIPS 2024): **mLSTM** is the associative / scanable
memory; **sLSTM** is the nonlinear one (exp gates, stabilizer `max`,
normalizer `n`, mixing `R h`). FlashRNN keeps sLSTM sequential. ParaRNN fused
GRU/LSTM does not implement this cell.

This repo's earlier literature note had sLSTM/mLSTM swapped. The code
follows Beck et al.

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

`step_with_jacobian` is the Newton `auto` path. The channelwise skeleton is
`mix='diag'`; the code also chains `tanh(z_z)`, `σ(z_o)`, and
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

### P=0 (diverged Newton)

`uv run python scripts/bench_time.py --config configs/bench/newton_slstm.yaml`.
MLflow `newton-slstm-bench` / `slstm-fused-vs-naive`. CSV:
`outputs/bench_newton_slstm.csv` (gitignored).
**10 warmup / 50 runs, min ms.** B=8, \(d_{\mathrm{in}}=d_h=256\), float32,
\(K=3\), `x_scale=1`, `picard_iters=0`. Fused T cap 2048. Not FlashRNN.

**These times at \(T\ge 256\) are kernel throughput of a Newton that has not
matched sequential.** Do not quote the vs-RNN column as sequential-equivalent
work. Kept so the P=3 table is not mixed with this.

| T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs RNN | vs compiled | vs eager N |
|---|---|---|---|---|---|---|---|
| 64 | 43.5 | 23.9 | 28.3 | **8.81** | 4.9× | 2.7× | 3.2× |
| 256 | 169 | 101 | 39.1 | **13.7** | 12× | 7.4× | 2.9× |
| 512 | 348 | 206 | 51.8 | **16.3** | 21× | 13× | 3.2× |
| 1024 | 727 | 404 | 77.4 | **21.7** | 34× | 19× | 3.6× |
| 2048 | 1411 | 826 | 140 | **30.3** | 47× | 27× | 4.6× |

### P=3 (sequential-matched)

`uv run python scripts/bench_time.py --config configs/bench/newton_slstm_picard.yaml`.
MLflow `newton-slstm-bench` / `slstm-picard-fused-vs-naive`. CSV:
`outputs/bench_newton_slstm_picard.csv`. Same protocol and shapes,
`picard_iters=3`, `require_agreement` atol \(10^{-3}\) (LM-length residual
at this width, not GRU \(10^{-4}\)). Max |fused − seq|: 8e-6 … **2.8e-4**.

| T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs RNN | vs compiled | vs eager N |
|---|---|---|---|---|---|---|---|
| 64 | 43.0 | 23.9 | 55.1 | 37.7† | 1.1× | 0.63× | 1.5× |
| 256 | 179 | 100 | 73.9 | **5.10** | 35× | 20× | 14× |
| 512 | 355 | 201 | 92.1 | **7.77** | 46× | 26× | 12× |
| 1024 | 726 | 395 | 121 | **12.1** | 60× | 33× | 10× |
| 2048 | 1484 | 814 | 207 | **23.6** | 63× | **34×** | 8.8× |

† T=64 fused **37.7 ms** is the old eager-Blelloch Picard (P=3). Auto-P
is P=1 at this length: Triton Picard fused **2.12 ms**. Sequential /
eager columns are the same process as the first P=3 smoke.

Peak MiB is the same class as P=0 (fused 27 → 550 vs eager Newton 71 →
1946). Triton Picard (max-plus + two 1D `ax+b` scans) dropped matched
fused \(T=2048\) from **77 → 23.6 ms**. P=0 fused 30 ms is still a
diverged Newton; matched fused is now in the same ballpark because
Picard is no longer eager Blelloch. At \(T=64\) auto P=1 fused is
**faster** than compiled sequential. Do not paste GRU's 400× onto this
table. Not FlashRNN.

Fused vs eager Newton is **~9–14×** after Triton Picard (was 1.2–2.7×
when Picard was eager Blelloch).

## Agreement at width 256

Smoke above (B=8, `x_scale=1`, unseeded weights, \(K=3\)), max |par − seq|.
**P=0** (the diverged timing run):

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
`NewtonConfig.max_iters` for GRU/LSTM to paper over this. Use
`picard_iters` (below), not `chunk_len`, if the span must stay
\(O(\log T)\).

## Picard predictor (frozen-gate scans)

`NewtonConfig(picard_iters=P)` is extra prefix scans after zero-hidden:
freeze \(R h\) from the previous trajectory, rescan `(c, n, m)` with
max-plus `m` and two 1D `ax+b` scans for `n`/`c`. CUDA fp16/fp32 uses
the Triton tiled scan (`kernels/picard_slstm.py`); CPU stays eager
Blelloch. Not Jacobi `H := f(H_prev, x)` (that only moves one token per
iter). Not Mamba-2 (no extra parameters). Each pass is still
\(O(\log T)\). **Library default is `P=None` (auto from T):**
P=1 if \(T\le 64\), P=3 if \(T\le 2048\), else P=5. Finer cutovers
(P=1 at T=256, P=2 at T=1024, P=4 at T=4096) are seed-0 B=2 only;
unseeded B=8 T=256 P=1 was \(2\times10^{-2}\), and T=4096 P=3/4 with
forced K=3 (no residual early-stop) still diverges on some draws.
Explicit `0` is zero-hidden only.

Isolated seed 0, 2080 Ti, B=2, \(d_h=256\), `x_scale=1`, K=3, eager,
fresh cell per T. Max |par − seq|:

| T | P=0 | P=1 | P=2 | P=3 |
|---|---|---|---|---|
| 256 | 3.8e-2 | **2.7e-5** | 2.9e-5 | 3.6e-5 |
| 1024 | 1.1e3 | 5.2e-3 | **1.1e-4** | 1.2e-4 |
| 2048 | 1.9e7 | 15 | 4.9e-3 | **4.8e-4** |

Fused P=3 tracks eager: T=256 ~3e-5, T=1024 ~1e-4, T=2048 **3.9e-4**
(eager 4.8e-4). Fallback: raise P, not K. Cap 3 at this width.

Once P=3 is in the basin, **native Newton is the corrector**. Same
successive-`randn` probe as the log table below, K=3, P=3:

| T | native | log |
|---|---|---|
| 256 | **3.6e-5** | 5.0e-5 |
| 1024 | **1.2e-4** | 1.8e-4 |
| 2048 | **1.6e-4** | 4.9e-4 |

Native is ~1.5–3× tighter and has no LSE in J. Log does not snap when
Picard does not; do not use `coords="log"` to paper over a far guess.

## Log-space Newton and chunked scan

`NewtonConfig(coords="log")` is the convex-combination / LSE cell
(\(u_t=(1-\gamma)u_{t-1}+\gamma\tanh z_z\), \(\log n=\mathrm{LSE}\),
\(h=\sigma(z_o)\odot u\)), not a pushforward through native \(c/n^2\).
\(c\) is signed: \(\log c\) is invalid. Mixing still reads stored \(h\).
Sequential stays the native cell. Fused diag has a matching LSE kernel.

**Not the long-T snap.** That is `picard_iters` + native K=3. Log stays
opt-in (ablation / far guess). Default `coords="native"`.

`NewtonConfig(chunk_len=64)` runs Alg. 1 on windows of 64 and passes the
last state as the next \(h_0\). Each window can be eager, Triton-scan, or
fused. Span is **linear in \(T/64\)**, not \(O(\log T)\). 64 is this
repo's measured snap length at \(d_h=256\), seed 0, K=3 — not Gemini's
\(10^{-7}\) claim as a theorem. Fallback: 32. Prefer Picard.

Same probe as the K-curve (2080 Ti, seed 0, B=2, \(d_h=256\), eager, K=3).
One cell, successive `randn` lengths (not a fresh seed per T).

| T | scale | native | log | chunk 64 | log+64 |
|---|---|---|---|---|---|
| 64 | 1.0 | **7e-6** | 2e-5 | **7e-6** | 2e-5 |
| 256 | 1.0 | 2.0 | 1.6 | 0.41 | **6e-5** |
| 512 | 1.0 | 31 | 23 | 2.5e-2 | **1e-4** |
| 1024 | 1.0 | 5.8e6 | 8e3 | **2e-4** | 3e-4 |
| 2048 | 1.0 | 1.8e5 | 984 | 0.36 | 0.35 |
| 256 | 0.3 | 0.20 | 3e-4 | 3e-3 | 4e-4 |
| 2048 | 0.3 | 94 | 73 | **2e-3** | 5e-3 |

Isolated seed-0 T=2048, `x_scale=1` (no prior `randn`): chunk 32 log
**6e-4**, chunk 64 ~1e-3, chunk 128 **fails** (~2). Gemini's
"T=64 always \(10^{-7}\) then the scan just carries" is false at this
width. Chunking makes K=3 usable past a few hundred tokens at the cost
of a **linear** span. Prefer `picard_iters` when the solve must stay
\(O(\log T)\). Default stays `coords="native"`, `chunk_len=None`. `picard_iters=None`
auto-selects P from T (library default).

## Not yet

Packed VJP, fused head mix, LM train. Mamba-2 predictor is not this Picard
(no extra SSM weights). FlashRNN on this 2080 Ti is `triton_fused` 8×32
(not diag mix): T=2048 **5.8 ms** vs fused auto-P **23.6 ms** (4× slower).
Longer T does **not** cross: both stay linear; fused/FR ≈ **0.19×** out
to T=16384. `cuda_fused` needs `nvcc` + CC 8.0. See `docs/bottlenecks.md`.
