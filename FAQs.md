# FAQs

## Does parallel Newton match sequential unroll?

Yes within documented tolerances. Keep
`pytest -m "not cuda"` and GPU numerics (`pytest -m cuda`) green when changing
Jacobians, scans, or fused kernels. Float32 and bfloat16 have separate budgets.

## What if my GPU is older than Ampere (CC below 8.0)?

Fused Triton kernels target CC ≥ 8.0. Use
`NewtonConfig(scan_backend="auto")` — the solver falls back to eager / Triton
scan paths. Wall-clock will be slower than fused; numerics stay on the same
Alg. 1 surface.

## How many Newton iterations should I use?

- Pin with `NewtonConfig(max_iters=3)` for ParaGRU / ParaLSTM-style cells
  (Danieli et al. 2025 §2.1 / App. A empirical agreement).
- Use `max_iters=None` for measured \(K^*(T)\) auto schedules
  (`pararnn.solvers.newton.k_star`); campaign through \(T{=}131072\) via
  `scripts/bench_k_star.py`.
- Override with `newton_iters_by_t={64: 2, 1024: 3, …}` when you have a table.

If residual stays high, try Picard warm-start (`picard_iters`) before raising
\(K\).

## Does `torch.compile` work?

Yes with the compile-safe preset (fixed \(K\), no residual host sync) →
`fullgraph=True` on eager and fused. See
`tests/numerics/test_compile.py` and [README Compatibility](README.md#compatibility).

## train() vs eval()?

`.train()` runs parallel Newton+scan. `.eval()` runs sequential `step`. On CUDA
with `T=1`, decode uses Triton `decode_step` when available. Force either path
with `solver='newton'|'sequential'`.

## Where are benches?

Under [`scripts/`](scripts/README.md) (not shipped in the wheel). Peers often
name this `benchmarks/`; here the entrypoint is `scripts/bench_*.py` plus
[`scripts/README.md`](scripts/README.md).

## Is Apple ml-pararnn in this package?

No. `third_party/ml-pararnn` is a local read-only reference under Apple’s
license. Public MIT code is reimplemented from the paper equations.
