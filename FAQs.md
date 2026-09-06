# FAQs

## Does parallel Newton match sequential unroll?

Yes within documented tolerances — that is the **agreement** claim
(τ ≈ 1e-4 in float32). Keep `pytest -m "not cuda"` and GPU
numerics (`pytest -m cuda`) green when changing Jacobians, scans, or fused
kernels. Float32 and bfloat16 have separate budgets.

Full contract (residual gate vs agreement τ, four echelons):
[`docs/numerics-contract.md`](docs/numerics-contract.md).

Before opening a bug, run the public check:

```python
from pararnn import ParaGRU, NewtonConfig, verify_agreement

report = verify_agreement(cell, x, config=NewtonConfig(max_iters=3))
print(report.ok, report.max_abs)  # True, … if healthy
```

`report.ok` uses the same band as the numerics tests (default atol `1e-4` in
float32). Paste `report.to_dict()` into the issue if it fails.

At train start you can set `NewtonConfig(verify_first_step=True)` on a
`ParaRNN`: the first Newton forward runs that check once, then training
continues at full speed.

## Residual gate vs agreement τ?

**Different thresholds.**

- **Agreement τ** — gap between Newton trajectory and sequential unroll.
  Brand claim. Checked by `verify_agreement` / CI / `verify_first_step`.
- **Residual** `max|F(H)|` — Newton solver health.
  `residual_fail=1.0` (default) raises `NewtonDivergenceError`; a warn fires
  near `1e-3`. Logged as `newton_residual` next to loss.

A finite loss with no `DivergenceError` does **not** prove sequential match.
A residual ≥ 1 means the solver left its basin — do not step the
optimizer on that batch.

## CUDA OOM / long T / which flag?

Usually geometry (width / heads / stored H*). Cheat sheet:

1. Train diag GRU/sLSTM toward ~100k tokens → `NewtonConfig(recompute=True)`.
2. Hopfield → keep `d_h ≤ 32`.
3. RWKV-7 on long text (12 GiB) → slim `n_heads=1`, `d_head=16`.

Full decision tree + peak-mem smoke:
[`docs/oom-cookbook.md`](docs/oom-cookbook.md).
OOM is loud; silent wrong answers are the numerics contract
([`docs/numerics-contract.md`](docs/numerics-contract.md)).

## What about Turing GPUs (e.g. RTX 2080 Ti)?

They are first-class for **fp32 / fp16 fused** Newton — this lab’s benches
include an RTX 2080 Ti. Only **bf16 fused** needs Ampere+ (CC ≥ 8.0); on
Turing use fp16 or fp32 (algebra inside the kernel stays fp32 either way).

## What if I am on CPU, Mac, or Windows?

There is no fused CUDA path. Use
`NewtonConfig(scan_backend="auto")` — eager Newton+scan still matches the
sequential oracle within the numerics contract (`verify_agreement`).

## What is critical Newton depth (K*)?

Training walks a few **Newton iterations** over the whole sequence. **K\*(T)**
is the smallest iteration count where the parallel result still matches a
plain sequential for-loop within tolerance τ≈1e-4, as a function of sequence
length T.

For several cells here K\* stays **2** from short T through T=131072 (does not
grow with T). RWKV-7 is an exact parallel scan, so K\*=0. Set
`NewtonConfig(max_iters=None)` to use the measured schedules in
`pararnn.solvers.newton.k_star`.

## How many Newton iterations should I use?

- Pin with `NewtonConfig(max_iters=3)` for ParaGRU / ParaLSTM-style cells
  (Danieli et al. 2025 §2.1 / App. A empirical agreement).
- Use `max_iters=None` for measured K\*(T) auto schedules
  (`pararnn.solvers.newton.k_star`); campaign through T=131072 via
  `scripts/bench_k_star.py`.
- Override with `newton_iters_by_t={64: 2, 1024: 3, …}` when you have a table.

If residual stays high, try Picard warm-start (`picard_iters`) before raising
the iteration budget.

## Triton version / CompilationError / misaligned address?

Fused kernels pin **triton 3.6.x** (same wheel index as Torch 2.11 cu128). A
foreign Triton often fails at first JIT with `CompilationError` or a CUDA
misaligned-address traceback. Reinstall Triton from the torch index, then
re-run the bug-report env one-liner.

Internally, `require_fused_triton()` (first fused / Triton scan launch) checks
the 3.6.x pin, then runs a one-shot CUDA JIT smoke (masked load +
`associative_scan`). Call `pararnn.kernels.check_triton_environment()`
yourself before a long train if you want the same gate early. Fallback:
`NewtonConfig(scan_backend="eager")`. Loads use `precision.load_acc` /
`store_acc` (masked `tl.load`).

## Does `torch.compile` / autocast work?

Yes under the documented wrappers — see
[`docs/compile-amp.md`](docs/compile-amp.md).

- **`fullgraph=True`:** `compile_safe_config()` (fixed `K`, no residual host
  sync). Fused ops are Dynamo-opaque (`custom_op` + `register_fake`).
- **Autocast:** Newton opts out of outer autocast; half training uses
  explicit `.to(dtype)` on module and `x`.
- **Residual early-stop / Picard adapt:** eager path; under compile the
  library keeps a fixed-`K` loop.

```python
from pararnn import compile_safe_config, newton_apply
import torch

cfg = compile_safe_config(scan_backend="auto")
fn = torch.compile(lambda z: newton_apply(cell, z, cfg), fullgraph=True)
```

## Non-contiguous tensors / `cu_seqlens` packing?

Fused paths copy to contiguous before Triton. Head-mix + pack: `auto` →
eager; explicit `fused` raises `TypeError`. Diag GRU packs in fused kernels.

Matrix + smoke: [`docs/shapes-layout.md`](docs/shapes-layout.md).

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
