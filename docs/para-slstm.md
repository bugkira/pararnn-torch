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

## Not yet

Packed VJP, fused head mix, FlashRNN bench, LM train.
