# Numerics contract

Silent wrong answers at long context are a product risk: loss stays finite,
GPUs stay busy, and the hidden trajectory has already left the true
recurrence. This page separates **solver residual** from **sequential
agreement**.

## Invariant

With a measured iteration budget \(K^*(T)\), the parallel Newton+scan forward
is **mathematically equivalent** to an honest `sequential_apply` within
\(\tau \le 10^{-4}\) in float32 (low-precision dtypes use a wider band; see
`default_agreement_atol`).

If `verify_agreement` reports error \(> \tau\) at the recommended \(K^*(T)\)
on your shapes and device, that is a **bug** — file an issue with
`report.to_dict()`.

## Two thresholds

| Name | Symbol | Default | Meaning |
|---|---|---|---|
| **Agreement τ** | \(\|h_{\mathrm{newton}} - h_{\mathrm{seq}}\|\) | `1e-4` (fp32) | Match to the sequential oracle. Product claim. |
| **Residual gate** | \(\max\|F(H)\|\) | warn `1e-3`, fail `1.0` | Solver health on the Newton residual. Explosion fuse. |

Keep them separate. On long \(T\), \(\|F\|\) can stay moderate while the
trajectory drifts from the unroll — agreement still needs
`verify_agreement`. When \(\max\|F\| \ge 1\), Newton left its basin;
continuing the step poisons Adam moments.

- `NewtonConfig.residual_fail` (default `1.0`) → `NewtonDivergenceError`
- `newton_residual_high` warning when residual sits above the warn band
- `verify_agreement` → `AgreementReport` / optional `AgreementError`

## Four echelons

```
[1 Runtime]   every Newton forward — residual_fail + log max_residual
[2 Lab]       offline K*(T) tables through T=131072 (bench_k_star)
[3 Opt-in]    verify_agreement / verify_first_step (one batch, ~2× cost once)
[4 Docs]      this page + FAQs + bug template
```

### 1. Runtime (default on)

Cheap. After \(K\) steps, if \(\max\|F\| >\) `residual_fail`, raise
`NewtonDivergenceError`. Train loops must not `optimizer.step()` on that
batch. Log / log metrics for `max_residual` next to loss
(`ParaRNN.last_stats`, `pararnn.distributed.last_newton_residuals`).

This echelon stops **explosions**. Agreement still goes through echelon 3
(or CI).

### 2. Lab campaign — \(K^*(T)\)

Measured once (or on nightly lab jobs). Paper App. A is the starting guess;
tables in this repo are the authority for shipped cells.

- Script: [`scripts/bench_k_star.py`](../scripts/bench_k_star.py)
- Envelopes: `pararnn.solvers.newton.k_star`
- Train: `NewtonConfig(max_iters=None)` picks \(K\) from the table for current \(T\)
- Manual pin: `newton_iters_by_t={64: 2, 1024: 3, …}` or `max_iters=int`

For \(T > 8\text{k}\), prefer measured envelopes over a fixed folklore \(K=3\)
when the cell ships a table.

### 3. Opt-in oracle

`verify_agreement(module, x)` runs **one** batch through Newton and through
sequential (seconds of wall time). Use it:

1. **Start of training** — `NewtonConfig(verify_first_step=True)` on a
   `ParaRNN`: the first Newton forward checks agreement once, then continues
   at full speed.
2. **CI** — short \(T\) unit/numerics tests (fp32 τ kept at `1e-4`).
3. **Incident** — same weights + failing batch; if sequential matches Newton,
   look at LR / data / Adam; if they diverge, look at the kernel path.

```python
from pararnn import ParaGRU, NewtonConfig, verify_agreement

report = verify_agreement(cell, x, config=NewtonConfig(max_iters=3))
assert report.ok  # or raise_on_fail=True
```

### 4. Docs / product claim

This file, [`FAQs.md`](../FAQs.md), README glossary, and the bug-report
template. Public claim: parallel ≡ sequential at τ under \(K^*(T)\).
The residual gate is the explosion fuse; agreement is a separate check.

## Logging residuals

```python
from pararnn.distributed import last_newton_residuals

y = model(x)  # ParaRNN in train() → Newton
for r in last_newton_residuals(model):
    # mlflow.log_metric("newton_residual", r, step=…)
    ...
```

BabyLM’s train script already logs `newton_residual` each step.

## Related

- [`oom-cookbook.md`](oom-cookbook.md) — long-T VRAM: `recompute`, Hopfield
  \(d_h\), RWKV slim heads
- [`FAQs.md`](../FAQs.md) — Turing / K* / `verify_agreement` paste
- [`backward-scan-cap.md`](backward-scan-cap.md) — long-T scan limits
- Observability: residual, wall time, device, dtype on solver boundaries
