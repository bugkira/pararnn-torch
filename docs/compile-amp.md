# `torch.compile` and Autocast / AMP

Wrapper contract for Dynamo and AMP. CI:
`tests/numerics/test_compile.py`, `tests/numerics/test_autocast.py`.

## Cheat sheet

1. **Half / mixed precision** — put the module and `x` in the target dtype
   (`.to(torch.float16)` / `bfloat16`). Outer `torch.autocast` leaves the
   Newton solve in the module / input dtype.
2. **`torch.compile` + `fullgraph=True`** — use `compile_safe_config()`
   (fixed `K`, no residual host sync).
3. **Residual early-stop / Picard adapt / `fused_early_exit`** — eager /
   non-compiled training. Under Dynamo those host paths are skipped or unsafe
   for a single fixed graph.

```python
import torch
from pararnn import ParaGRU, compile_safe_config, newton_apply

cell = ParaGRU(32, 64).cuda().to(torch.bfloat16)
x = torch.randn(2, 128, 32, device="cuda", dtype=torch.bfloat16)
cfg = compile_safe_config(scan_backend="auto")  # or "fused" / "eager"

compiled = torch.compile(lambda z: newton_apply(cell, z, cfg), fullgraph=True)
y = compiled(x)
```

## Autocast / AMP policy

`newton_apply` wraps the solve and eq. 2.6 backward in
`_newton_precision_region`: if outer CUDA/CPU autocast is on, it is
**disabled** for that region so `W_x` and states share one dtype (the
module / input dtype).

| Pattern | Result |
|---|---|
| `model.to(bf16)` + `x` bf16 | Newton runs bf16 (fused needs CC ≥ 8.0) |
| float32 model inside `autocast(bf16)` | Newton stays float32 (test_autocast) |
| `GradScaler` around the step | Fine; scaler sees the loss you pass it |

Half-precision training recipe: explicit `.to(dtype)` on cell/`ParaRNN` and
`x`, then optional GradScaler. Autocast alone leaves Newton in fp32.

## `torch.compile` boundaries

Fused Newton/scan entrypoints are `torch.library.custom_op` with
`register_fake` — Dynamo treats them as opaque (shape/dtype from the fake).

The compile-safe preset (`pararnn.compile_safe_config`):

| Field | Value | Why |
|---|---|---|
| `max_iters` | `3` (App. A) | Fixed iteration count |
| `residual_atol` | `None` | No `float(max\|F\|)` in the K-loop |
| `residual_fail` | `None` | No host sync after K in `_fill_stats` |
| `picard_adapt` | `False` | No residual-driven Picard `while True` |

Under `torch.compiler.is_compiling()` the library also skips residual
early-stop even when `residual_atol` is set on a non-preset config (eager
M²RNN / fused paths). Dynamic “stop when residual is small” is an
**eager** feature (`residual_atol`, experimental `fused_early_exit`).

`verify_first_step` and Triton preflight smoke likewise skip while compiling.

## What to file when it breaks

Paste:

- `torch`, CUDA, Triton, GPU (bug-template one-liner)
- whether you used `compile_safe_config` / which `scan_backend`
- module dtype and autocast dtype
- `torch._dynamo.explain(...)` graph-break summary if compile fails

Related: [`numerics-contract.md`](numerics-contract.md),
[`oom-cookbook.md`](oom-cookbook.md), README Compatibility.
