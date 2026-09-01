# Para-sLSTM (research branch)

Sequential sLSTM + Newton. No fused kernel. Not FlashRNN. Not TinyStories.

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
  (`cell.jac_structure='block4'`). Scan is `scan_block4` (eager; no Triton).
- `mix='head'`: `R` is `(4, n_heads, d_head, d_head)`. Dense mix inside a head,
  zeros across heads (Beck / xLSTM). Jacobian is `4 d_head × 4 d_head` per
  head; scan folds heads into the batch of `scan_dense`. `n_heads` must
  divide `d_h`.
- `mix='dense'`: `R` is `Linear(d_h, 4 d_h)`. Exact mixing; scan is
  `O(T (4d)^3)`. Tests use `d_h≤4`.

`K=3` is **not** assumed. Global `NewtonConfig` stays App. A (`K=3`,
`omega=1`) for ParaGRU/LSTM.

## Measured (2080 Ti, seed 101, diag mix, `T=12`, `d_h=4`)

Sequential max-abs error. Same seed as `test_slstm_diag_newton_vs_sequential`.
block4 vs forced `jac_structure='dense'` agrees — the overshoot is the Newton
map (`max`/`exp`), not 4×4 vs flattened packing. `_is_dense` must not treat
`(B,T,4,4,d)` as a full matrix when `d==4`.

| K | ω=1, clip=0.5 | ω=0.5, clip=0.5 | ω=1, clip=0.25 |
|---|---|---|---|
| 1 | 3.09 | 4.94 | 3.06 |
| 2 | 7.95 | 2.77 | 7.86 |
| 3 | 12.1 | 3.93 | 12.0 |
| 4 | **2.8e-6** | 1.91 | **2.5e-6** |
| 5 | 9.5e-7 | 0.94 | 1.9e-6 |

`omega=0.5` (Gonzalez et al. ELK) lowers K=3 overshoot but **kills the K=4
snap**. Clip 0.25 vs 0.5 is a no-op on this seed. Prototype sLSTM recipe:
**K=4, omega=1, clip=0.5**. Do not change the library Newton default. If a
seed does not snap at K=4, raise K (measure residual vs K) before damping.

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

## Not yet

Fused Triton, VJP packed kernel, FlashRNN bench, LM train.
