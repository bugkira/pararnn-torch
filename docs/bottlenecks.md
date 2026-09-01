# Bottlenecks and improvement list

Where the eager PyTorch Newton+scan actually spends time and memory on this box, what is worth changing, and how sure we are. Effects are relative to this 2080 Ti prototype, **not** to Danieli et al. fig. 2/5 (A100 + fused CUDA).

Do not switch to the 3060 to dodge OOM. Do not copy `third_party/ml-pararnn` kernels.

Rows are ordered by **priority**: (impact on training or honest numbers) × (confidence) × (cheapness). “Done” is the same rule, not chronological.

Verify runtime changes with numerics tests, then [`scripts/bench_time.py`](../scripts/bench_time.py) on the **2080 Ti by name**, MLflow `cell-forward-bench`.

## Next

Library completeness after v0.2 is **done** (`scan_backend=auto`, NewtonStats /
early-stop, fused `h0`, layer list / `return_hidden` / `output_hidden`).
Remaining items are **other science**, not package polish.

| Pri | Change | Why this high | Effect | Confidence |
|---|---|---|---|---|
| 1 | IFT adjoint (Bai / DEQ) | Eq. 2.6 is in. IFT is extra. | \(O(1)\) in solver depth. | **Low** until 2.6 is the bottleneck |
| 2 | HF / SlimPajama / 125M | Empty `PreTrainedModel` is worse than none. Not 2080 Ti. | Paper-scale claim. | **Low** until a real train loop |

Do **not** bake `torch.compile` into `src/` (Dynamo 4–113 s per new \(T\)).
Do **not** pick a GPU inside `ParaRNN.forward`. The module follows the tensor / `.to(device)`.
Do **not** treat `fused` as “any \(f\)”.

## Done

| Pri | Change | Why this was first | Result |
|---|---|---|---|
| 0 | ``scan_backend="auto"`` by tensor (fused if CUDA GRU/LSTM/sLSTM-diag and ``is_fused_dtype_supported``: fp16/fp32 any CUDA, bf16 if CC ≥ 8.0). Log the choice. Fused kernels prepend ``h0``. Early-stop + ``NewtonStats``. Layer: list of cells, ``return_hidden``, default hidden-slot output, ``solver`` / ``batch_first`` / ``hidden_layout``. | Default eager hid fused; fused ignored nonzero ``h0``. | Default is ``auto``. Fused+``h0`` matches sequential. Residual is an API. |
| 0c | Hybrid Picard \(H^{(0)}\) + K=1 vs library P(T)+K=3. Untrained S4D-Real SSM guess. | DEER: Newton wants a close guess. | **24.2 → 20.1 ms** fused fwd at T=2048 (P=5 K=1 vs P=3 K=3, same snap). P=3 K=1 misses (err 0.15). S4D-Real guess diverges. Parked (library still auto-P+K=3). Dyck train T=64 not faster. `scripts/bench_hybrid_pk.py`. |
| 1 | Any ``step(h, x)`` via Autograd Jacobian (DEER / Lim et al.). ``jacobian="auto"``: analytic if the cell has ``step_with_jacobian``, else ``torch.func``. | Two hardcoded cells is not a library. | Custom channelwise cell and dense mix (``jac_structure="dense"``) match sequential. Ones-JVP is exact iff ``f`` is channelwise; otherwise set ``dense``. Fused Newton stays ParaGRU/LSTM (not any ``f``). |
| 2 | Eq. 2.6 cell VJP packed on CUDA for ParaGRU/ParaLSTM and ParaSLSTM ``mix='diag'`` (Triton elementwise + ``W_x`` GEMM). Reverse scan already Triton when ``scan_backend`` is ``triton``/``fused``. Head/dense sLSTM stay Autograd on ``step``. | Backward was ``autograd.grad(cell.step)``. | Packed VJP matches Autograd VJP. Existing BPTT tests still pass. Custom cells keep Autograd on ``step``. **sLSTM diag, 2080 Ti, 2026-09-02** (`scripts/bench_packed_vjp.py`, smoke 10/50): isolated VJP min **0.75 vs 2.37 ms** at Dyck B=32 T=64 d_h=32 (**3.2×**); **3.79 vs 8.71 ms** at B=8 T=2048 d_h=256 (**2.3×**). Full Newton bwd **4.04 vs 8.27 ms** (**2.0×**) / **21.6 vs 26.8 ms** (**1.24×**) — reverse scan is shared. Forward unchanged. Dyck 50-step CE identical (max \(\lvert\Delta\mathrm{CE}\rvert=1.2\times10^{-7}\)); train bwd+AdamW mean **8.1 vs 10.7 ms**. Speed, not quality. |
| 3 | fp16 DRAM / fp32 Newton accumulators. Agreement tests **separately** (atol \(2\times10^{-3}\)). bf16 fused is **CC ≥ 8.0**, not a missing kernel; SM 7.x still rejects. | Halves scan DRAM. TC help `W_x` only. | Smoke 10/50, same process as fp32 fused: at \(T=2048\) GRU **1.80 vs 2.61 ms** (**1.45×**), **66 vs 123 MiB** (**1.86×**); LSTM **5.45 vs 6.41 ms** (**1.18×**), **107 vs 205 MiB** (**1.91×**). Short \(T\) is slower. Residual \(\sim10^{-3}\). See [fp16](#fp16). |
| 4 | Triton **fused Newton** (cell + J + scan per Alg. 1 iter). Opt-in `scan_backend="fused"`. GEMM `W_x` stays in PyTorch. Not Apple's kernel. Backward still eq. 2.6. | Newton still launched the eager cell \(K+1\) times after the scans landed. Paper 665× is fused CUDA, not this. | Smoke 10/50, \(T=2048\): GRU fused **2.4 ms** vs naive ParaRNN 27 ms (**11×**) vs naive RNN 959 ms (**400×**); LSTM **6.5 ms** vs 62 ms (**9.6×**) vs 1338 ms (**207×**). Peak mem vs naive ParaRNN **~3–4×** lower. ParaSLSTM diag fused **P=0 30 ms** is a **diverged** Newton; sequential-matched auto-P after Triton Picard is **23.6 ms** vs compiled seq 814 ms (**34×**) vs FlashRNN 5.8 ms (**0.25×**). See [Fused Newton](#fused-newton) and [Fused sLSTM](#fused-slstm). |
| 5 | LSTM 2×2 as four muls, not `einsum`→`bmm`; Blelloch scan (not Hillis–Steele). No `torch.associative_scan` (CUDA/compile prototype, no CPU, no autograd). | **86% of LSTM CUDA** was tiny GEMVs. HS was \(O(T\log T)\) traffic. | Smoke: LSTM \(T=512\) 113→**46 ms**; \(T=2048\) 537→**82 ms**, 7.5 GiB→**719 MiB**; \(T=4096\) **fits, 1.4 GiB** (was OOM). GRU \(T=2048\) 29→**36 ms** (gather tax). |
| 6 | Call-site `torch.compile(newton_apply)` (`reduce-overhead`). **Not** in `src/`. App. B `min_ms` is after warmup and **hides** Dynamo. | Launch tax on small \(T\). | Steady-state GRU \(T\le 256\) **~20–40×**; LSTM \(T=64\) **~16×**. Compile itself is **4–113 s** per new \(T\). LSTM at LM length needs **~2k forwards** to break even. See [Compile Newton](#compile-newton). |
| 7 | Triton **diag** + **2×2** scan (`tl.associative_scan`, two-level tiles). Opt-in `scan_backend="triton"`. Not Apple's kernel. | Eager Blelloch gather tax. | Scan-only smoke (B=8, \(d_h=256\), 10/50, not App. B): diag **~16–34×**; 2×2 \(T=64\ldots4096\) eager 13→33 ms vs Triton 0.18→3.5 ms (**~10–74×**). Newton still pays the eager cell. |
| 8 | Honest sequential: `sequential_apply_compiled` | Speedup vs a Python loop that computed unused J is not a result. | Smoke GRU \(T=64\): eager 29, compiled 24, Newton ~15 ms now. |
| 9 | One `W_x(x)` per Newton forward (eq. 3.1: \(Bx\) independent of \(h\)). Optional `wx=` on `step` / `step_with_jacobian`. Eager sequential also precomputes `(B,T,*)`. Backward VJP still runs `W_x` with grad. | Four identical GEMMs (init + \(K=3\)). | Same numerics. Small \(T\) kernel-count; large \(T\) scan still dominates. |
| 10 | Backward = eq. 2.6 reverse scan + cell VJP | Training would OOM on \(O(KT)\) Newton iterates. | Grads match sequential BPTT. |
| 11 | Preallocated scan buffers (no per-round `cat`) | Allocator high-water. HS itself is gone; the shift helper is obsolete. | GRU \(T=2048\) peak ~330 MiB (then Blelloch). |
| 12 | `step` without Jacobian | Sequential paid for unused \(J\). | Sequential **1.2–1.6×**. |
| 13 | CUPTI profile | Otherwise GEMM/tensor-core story leads. | See [Profile](#profile). |

## Sources

- Bench: App. B (CUDA events, 20 warmup, 100 runs, **min** ms). [`configs/bench/cell_forward.yaml`](../configs/bench/cell_forward.yaml): batch=8, \(d_{\mathrm{in}}=d_h=256\), \(K=3\), float32. CSV: `outputs/bench_cell_forward.csv` (gitignored). **Table below is the first App. B run** (eager sequential with J inside `step`, scan still `cat`). Do not mix it with later smokes.
- Hot path: [`newton.py`](../src/pararnn/solvers/newton.py), [`scan.py`](../src/pararnn/solvers/scan.py), cells.
- Paper App. A/B; Apple modes in [`apple-ml-pararnn.md`](apple-ml-pararnn.md).
- Hardware: Turing 2080 Ti, 11 GiB. FP16 tensor cores. **No bf16 TC**.

## Diagnosis (first App. B, before the Done list)

| Cell | T | Newton min ms | Peak MiB | vs eager sequential (then) |
|---|---|---|---|---|
| ParaGRU | 16 | 6.85 | 24 | 1.90× |
| ParaGRU | 256 | 8.90 | 303 | 23.4× |
| ParaGRU | 1024 | 16.7 | 1323 | 50.8× |
| ParaGRU | 4096 | 58.5 | 5835 | 57.7× |
| ParaLSTM | 16 | 13.0 | 43 | 1.58× |
| ParaLSTM | 128 | 25.2 | 364 | 6.47× |
| ParaLSTM | 1024 | 246 | 3539 | 5.41× |
| ParaLSTM | 2048 | 537 | 7547 | 5.03× |
| ParaLSTM | 4096 | OOM during Newton | — | sequential 5470 ms |

ParaGRU Newton **flat ~7–9 ms to \(T=256\)**, then linear (58 ms / 5.8 GiB at \(T=4096\)). ParaLSTM ~**19× slower than GRU** at \(T=2048\). That table’s sequential column is **inflated** (J inside `step`); use compiled sequential for claims.

```mermaid
flowchart LR
  Wx["W_x of x once"] --> cell["step_with_jacobian times K plus 1"]
  cell --> jac["J diag or 2x2"]
  jac --> hs["Blelloch or Triton scan"]
  hs --> update["states plus delta"]
  Wx --> fused["fused cell plus J plus scan"]
  fused --> update
```

## Compile Newton

Call-site only: `uv run python scripts/bench_time.py --config configs/bench/newton_compile.yaml`. `torch.compile` is **not** in `src/`. Mode `reduce-overhead` (CUDA graphs at static \(T\)). MLflow `newton-compile-bench`. CSV: `outputs/bench_newton_compile.csv`.

App. B `min_ms` is **after** 20 warmup + 100 runs. That number does not include Dynamo / Inductor / graph capture. The first compiled call is timed separately (`compile_s`). Break-even uses **median** save per step (CUDA-graph `min` can be a lucky short capture — GRU \(T=16\) min 0.21 ms vs median 2.11 ms).

\(N = \texttt{compile\_s} / (\texttt{eager\_median} - \texttt{compiled\_median})\). One new \(T\) is one new graph (Inductor warned after 9 sizes).

| Cell | T | Eager median ms | Compiled median ms | Steady-state | Compile s | Break-even forwards |
|---|---|---|---|---|---|---|
| ParaGRU | 16 | 11.1 | 2.11 | 5.3× | 4.2 | 470 |
| ParaGRU | 64 | 14.1 | 0.34 | 42× | 0.5 | 40 |
| ParaGRU | 256 | 18.0 | 0.79 | 23× | 0.7 | 40 |
| ParaGRU | 2048 | 42.9 | 14.9 | 2.9× | 1.4 | 50 |
| ParaGRU | 4096 | 83.1 | 15.1 | 5.5× | 16 | 240 |
| ParaLSTM | 16 | 24.3 | 2.63 | 9.2× | 7.7 | 360 |
| ParaLSTM | 64 | 46.5 | 3.09 | 15× | 7.6 | 180 |
| ParaLSTM | 512 | 45.9 | 6.38 | 7.2× | 88 | 2200 |
| ParaLSTM | 2048 | 59.1 | 23.1 | 2.6× | 104 | 2900 |
| ParaLSTM | 4096 | 104.5 | 45.8 | 2.3× | 113 | 1900 |

GRU \(T\ge 32\) compile is cheap in this process because Inductor already saw nearby shapes. LSTM \(T\ge 128\) is the honest tax: **1–2 minutes** before the first fast step. A 100-run App. B does **not** pay that back. A long training run at **one** padded \(T\) might (~2k–3k steps). A sweep over lengths, or a notebook that compiles once and exits, does not.

Default Dynamo `recompile_limit=8` silently fell back to eager on the 9th \(T\) (GRU \(T=4096\) and all LSTM in the first attempt). The script raises it to 64 for this sweep only.

## Fused Newton

Opt-in: `NewtonConfig(scan_backend="fused")`. Triton cell + Jacobian + scan per Alg. 1 iteration; `W_x(x)` is still one PyTorch GEMM. Not Apple's `parallel_FUSED`. Backward is still eq. 2.6 (J at the fixed point, reverse scan).

`uv run python scripts/bench_time.py --config configs/bench/newton_fused.yaml`. MLflow `newton-fused-bench`. CSV: `outputs/bench_newton_fused.csv`. **10 warmup / 50 runs, min ms, not App. B.** Same process, 2080 Ti, B=8, \(d_h=256\), \(K=3\).

Baselines in this table:

- **Naive RNN** — `sequential_apply` (Python unroll of `cell.step`, no Jacobian).
- **Naive ParaRNN** — eager Newton + Blelloch (`scan_backend="eager"`).
- **Fused** — `scan_backend="fused"`.

Compiled sequential (`sequential_apply_compiled`) is the honest RNN timing baseline; it is **not** the paper's naive sequential. Do not quote the vs-RNN column as fig. 2/5 (A100 fused CUDA, 665×).

| Cell | T | Naive RNN | Naive ParaRNN | Fused | vs naive RNN | vs naive ParaRNN |
|---|---|---|---|---|---|---|
| ParaGRU | 64 | 27.5 | 16.0 | **0.72** | 38× | 22× |
| ParaGRU | 256 | 113 | 20.6 | **0.80** | 140× | 26× |
| ParaGRU | 512 | 234 | 29.6 | **0.90** | 261× | 33× |
| ParaGRU | 2048 | 959 | 27.0 | **2.40** | 400× | 11× |
| ParaGRU | 4096 | 1925 | 38.6 | **4.73** | 407× | 8.2× |
| ParaLSTM | 64 | 58.2 | 34.5 | **0.78** | 75× | 44× |
| ParaLSTM | 256 | 172 | 43.7 | **0.88** | 195× | 50× |
| ParaLSTM | 512 | 352 | 50.7 | **1.57** | 225× | 32× |
| ParaLSTM | 2048 | 1338 | 62.2 | **6.45** | 207× | 9.6× |
| ParaLSTM | 4096 | 2760 | 104.3 | **12.3** | 224× | 8.4× |

Times in ms (min). Peak memory is the next table.

**Honest sequential** — `sequential_apply_compiled` (`reduce-overhead`, CUDA graphs, after warmup). Same smoke. This is the RNN baseline to quote, not the Python loop.

| Cell | T | Compiled seq | Fused | vs compiled seq |
|---|---|---|---|---|
| ParaGRU | 64 | 23.8 | **0.72** | 33× |
| ParaGRU | 256 | 98.6 | **0.80** | 123× |
| ParaGRU | 512 | 224 | **0.90** | 250× |
| ParaGRU | 2048 | 812 | **2.40** | 339× |
| ParaGRU | 4096 | 1569 | **4.73** | 332× |
| ParaLSTM | 64 | 30.0 | **0.78** | 39× |
| ParaLSTM | 256 | 104 | **0.88** | 118× |
| ParaLSTM | 512 | 219 | **1.57** | 140× |
| ParaLSTM | 2048 | 865 | **6.45** | 134× |
| ParaLSTM | 4096 | 1767 | **12.3** | 143× |

At \(T=64\) LSTM, eager Newton is *slower* than compiled sequential (34.5 vs 30.0 ms); fused is not.

Peak allocated MiB (`torch.cuda.max_memory_allocated`, same smoke). Ratio is naive ParaRNN / fused.

| Cell | T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs naive ParaRNN |
|---|---|---|---|---|---|---|
| ParaGRU | 64 | 16 | 12 | 22 | **15** | 1.5× |
| ParaGRU | 256 | 34 | 33 | 57 | **25** | 2.3× |
| ParaGRU | 512 | 56 | 35 | 103 | **39** | 2.7× |
| ParaGRU | 2048 | 193 | 50 | 379 | **123** | 3.1× |
| ParaGRU | 4096 | 375 | 98 | 747 | **235** | 3.2× |
| ParaLSTM | 64 | 18 | 59 | 34 | **18** | 1.9× |
| ParaLSTM | 256 | 40 | 43 | 103 | **35** | 3.0× |
| ParaLSTM | 512 | 69 | 46 | 195 | **59** | 3.3× |
| ParaLSTM | 2048 | 245 | 82 | 747 | **205** | 3.6× |
| ParaLSTM | 4096 | 479 | 162 | 1483 | **399** | 3.7× |

Fused drops **~3–4×** vs eager Newton at LM length because \(J\) never becomes a full PyTorch tensor. It still uses **more** than compiled sequential (scan tiles + \(K\) residuals): GRU \(T=4096\) 98→235 MiB, LSTM 162→399. Compiled LSTM \(T=64\) at 59 MiB is the CUDA-graph capture, not the unroll.

Short \(T\) is launch-bound. At training length: **~10× vs naive ParaRNN**, **~130–340× vs compiled sequential**, **~200–400× vs the Python loop**. None of these is 665×. Agreement vs sequential stayed \(\sim 10^{-7}\).

## Fused sLSTM

Diag mix only (`newton_slstm.py`, 4×4 SRAM). Head/dense raise. Zero-hidden
init, not App. A. Same smoke protocol as the GRU table, 2080 Ti, B=8,
\(d_h=256\), \(K=3\), 10/50, min ms. Details and the K-curve:
[`para-slstm.md`](para-slstm.md).

**P=0** ([`newton_slstm.yaml`](../configs/bench/newton_slstm.yaml)): fused
\(T=2048\) **30.3 ms** times a **diverged** Newton (max |par − seq| is 17 …
\(10^{14}\)). Do not quote GRU's 11×/400× from that table.

**P=3** ([`newton_slstm_picard.yaml`](../configs/bench/newton_slstm_picard.yaml)),
sequential-matched. Sequential / eager columns are the first P=3 smoke;
fused \(T\ge 256\) is after Triton Picard (same 10/50, 2080 Ti, B=8,
\(d_h=256\)). Max |fused − seq| ≤ **3.6e-4**.

| T | Naive RNN | Compiled seq | Naive ParaRNN | Fused | vs RNN | vs compiled | vs eager N |
|---|---|---|---|---|---|---|---|
| 64 | 43.0 | 23.9 | 55.1 | 37.7† | 1.1× | 0.63× | 1.5× |
| 256 | 179 | 100 | 73.9 | **5.10** | 35× | 20× | 14× |
| 512 | 355 | 201 | 92.1 | **7.77** | 46× | 26× | 12× |
| 1024 | 726 | 395 | 121 | **12.1** | 60× | 33× | 10× |
| 2048 | 1484 | 814 | 207 | **23.6** | 63× | **34×** | 8.8× |

† Eager-Blelloch Picard at pinned P=3. Auto-P at T=64 is P=1:
**2.12 ms** fused.

Triton Picard (1D max-plus + two `ax+b` scans) dropped matched fused
\(T=2048\) **77 → 23.6 ms**. Peak MiB fused 27 → 550 vs eager Newton
71 → 1946.

FlashRNN (NX-AI, ICLR 2025) is the sequential competitor. `cuda_fused`
needs `nvcc` and CC **8.0+**; this 2080 Ti is **7.5**, so the harness
uses `triton_fused`. Import always hits `torch.utils.cpp_extension`,
which requires `CUDA_HOME`. This box has no system toolkit; the bench
sets `CUDA_HOME` to the pip `nvidia-cuda-runtime` wheel
(`include/cuda.h`). That is enough to import, not enough to compile
`cuda_fused` (`nvcc` is missing, and the kernel is `sm_80`).

Heads on Turing: **8×32** at \(d_h=256\). D=64 overflows 64 KiB smem;
D=1 (diag) does not compile. This is **not** `mix='diag'`. Optional
extra: `uv sync --extra flashrnn` (NXAI Community License, not MIT).

[`newton_slstm_flashrnn.yaml`](../configs/bench/newton_slstm_flashrnn.yaml),
same 10/50 min ms, B=8, \(d_h=256\), K=3, auto P, `require_agreement`
atol \(10^{-3}\). Fused vs FlashRNN (`>1` = fused faster):

| T | fused (auto P) | flashrnn `triton_fused` | fused / FR |
|---|---|---|---|
| 64 | 2.12 (P=1) | **0.81** | 0.38× |
| 256 | 5.10 (P=3) | **1.18** | 0.23× |
| 512 | 7.77 | **1.88** | 0.24× |
| 1024 | 12.1 | **3.11** | 0.26× |
| 2048 | 23.6 | **5.82** | 0.25× |

T=2048 fused matches sequential to \(3.6\times10^{-4}\). FlashRNN is
**~4×** faster (was ~13× while Picard was eager Blelloch). Do not quote
a 20–50× vs FlashRNN; that needs Ampere `cuda_fused`. The 34×
vs compiled PyTorch unroll is not vs FlashRNN.

Four Flash-style kernel ideas, measured:

1. **1D scans instead of 4×4** — true for **Picard** (this Triton
   kernel). False for Newton: `R h` couples the next gates, so Alg. 1
   still scans a 4×4 `J` of `(c, n, m, h)`. Dropping that is inexact
   Newton, not "the 4×4 was never there".
2. **`h` as epilogue** — already true in Picard (`h = o ⊙ c / n`).
   Live mixing still needs `h` in the Newton state.
3. **Single-kernel P=3+K=3 in SRAM** — 64 KiB Turing smem cannot hold
   \(T=2048\) states. Intercept fell because Picard left eager
   Blelloch, not because P and K share one launch. Newton still dumps
   16 `J` lanes to DRAM per iter.
4. **Tiled scan** — already two-level (`BLOCK_T` then chunk scan),
   span \(O(\mathrm{tile} + \log n_{\mathrm{tiles}})\). Serial
   `chunk_len` Newton windows are the other "chunking"; do not add
   those for speed.

Forecast of 2.5–3.5 ms at T=2048 beating FlashRNN did **not** happen.
Remaining slope is K=3 of 4×4 fused Newton vs a sequential sLSTM kernel.

## Long T vs FlashRNN (no crossover)

[`newton_slstm_flashrnn_long.yaml`](../configs/bench/newton_slstm_flashrnn_long.yaml):
T=2048…16384, B=8, \(d_h=256\), 10/50 min ms, 2080 Ti. Fused T cap is
16384 (`CHUNK_PAD=512`, `CHUNK_D=1`, 40 KiB). Auto P=5 for T>2048.
`residual_atol=1e-5` (library): forced K=3 after Picard overshoots some
T=4096 draws. Agreement only to T=4096 (atol \(2\times10^{-3}\));
T=8192/16384 are timed, not sequential-checked.

FlashRNN is **exactly linear** (Gemini's 11.6 / 23.2 / 46.4 ms was right).
Fused Newton is also linear (DRAM \(O(KT)\), not a wall-clock
\(O(\log T)\) plateau). Ratio stuck at **~0.19×**.

| T | fused (auto P) | flashrnn | fused / FR | fused peak MiB |
|---|---|---|---|---|
| 2048 | 24.3 (P=3) | **5.77** | 0.24× | 550 |
| 4096 | 59.5 (P=5) | **11.3** | 0.19× | 1090 |
| 8192 | 120 | **22.5** | 0.19× | 2170 |
| 16384 | 246 | **46.6** | 0.19× | 4330 |

No overtake at 4096, 8192, or 16384. A later crossover would need the
per-step slope (~15 µs fused vs ~2.8 µs FlashRNN), not more T.

## Shamanskii, tile scan, R mix (measured)

`uv run python scripts/bench_slstm_tiled.py`. Same 10/50, 2080 Ti, B=8,
\(d_h=256\), K=3, auto P, `residual_atol=None`. Defaults stay off.

| variant | T=256 ms | T=2048 ms | vs baseline T=2048 | peak T=2048 |
|---|---|---|---|---|
| baseline (diag, assoc, full J) | **5.42** | **24.3** | 1.0× | 944 MiB |
| scan_tile=seq (serial `tl.range`) | 10.1 | 60.0 | 0.41× | 944 MiB |
| FlashRNN 8×32 | **1.15** | **5.75** | — | 997 MiB |

1. **True frozen scan was tried and removed.** Gemini: iter 0 is a full 20-lane
   scan and stores \(P_t=\prod_k J_k\); later iters skip scanning \(J\) and
   do \(\Delta_t=P_t\sum_{i\le t}P_i^{-1}r_i\) (4-vector prefix-sum). On
   well-conditioned \(J=I+0.05\mathcal{N}\) that matches the affine scan. On
   **real sLSTM** \(P_t\) is singular at T=12 already (`svd_min=0`, forget-gate
   products). Then \(|P\sum P^{+}r-\mathrm{scan}|\) is \(28\) / \(2\cdot10^{2}\)
   / \(2\cdot10^{3}\) at T=12/64/256. Short T still snapped because iter 0 is
   full Newton (T=256 B=8 \(d_h=256\): |par−seq| \(3\cdot10^{-5}\)). T=2048
   same shape: \(P^{+}r\) nonfinite. Wall clock T=256: **5.36 → 282 ms** —
   batched `pinv` of every \(P_t\), not the 4-sum. Reloading raw \(J_t\) and
   still scanning 20 lanes was 26.9 ms at T=2048 (also slower). Not in the tree.
2. **Handwritten warp-shuffle was tried and removed.** `shfl.sync.up` on
   one warp per \(d\) (`BLOCK_D=1`) matched assoc numerically and
   dropped local `bar.sync`, but T=2048 went 24.3→177 ms (16× more CTAs,
   no 16-wide \(d\) vectorization). `tl.associative_scan` on the
   `(32,16)` tile already emits `shfl` plus smem. `scan_tile=seq` is a
   serial `tl.range` prefix, **2.5× slower** — not a shuffle.
3. **R mix:** fused is already `mix='diag'` (\(R\odot h\)). `mix='head'`
   eager Newton at T=16 B=2 \(d_h=256\) 8 heads: **19.7 ms** — not a
   fused path. FlashRNN's 8×32 is a sequential kernel, not our Newton.

## fp16

Turing (SM 7.5): fp16 tensor cores exist; **no bf16 TC**. Ampere+ (SM 8.0+): fused/Triton **bf16** uses the same fp32 algebra + narrow DRAM (`is_fused_dtype_supported` on the **tensor** device). Activations, \(J\) tiles, and residuals stay in the tensor dtype in DRAM; Newton/scan algebra is fp32 inside Triton (`load_acc` / `store_acc`) and in eager Blelloch. `W_x` is still a PyTorch GEMM — that is where TC help. Explicit `scan_backend='fused'` + bf16 on SM 7.x raises (compute capability, not a GPU name). Agreement is vs sequential in the **same** dtype (atol \(2\times10^{-3}\)), not vs fp32 sequential.

`uv run python scripts/bench_time.py --config configs/bench/newton_fp16.yaml`. MLflow `newton-fp16-bench`. CSV: `outputs/bench_newton_fp16.csv`. **10 warmup / 50 runs, min ms, not App. B.** Same process, 2080 Ti, B=8, \(d_h=256\), \(K=3\), modes fused only. fp32 numbers here are the paired run, not the earlier fused-vs-naive table (GRU \(T=2048\) was 2.40 ms there, **2.61 ms** here).

Times in ms (min). Speedup is fp32 / fp16 (>1 is faster).

| Cell | T | fp32 | fp16 | vs fp32 |
|---|---|---|---|---|
| ParaGRU | 64 | 0.92 | 1.15 | 0.80× |
| ParaGRU | 256 | 0.72 | 0.84 | 0.86× |
| ParaGRU | 512 | 0.70 | 0.84 | 0.84× |
| ParaGRU | 2048 | 2.61 | **1.80** | **1.45×** |
| ParaGRU | 4096 | 4.68 | **3.48** | **1.35×** |
| ParaLSTM | 64 | 0.81 | 1.19 | 0.68× |
| ParaLSTM | 256 | 0.83 | 0.84 | 0.99× |
| ParaLSTM | 512 | 1.50 | **1.08** | **1.38×** |
| ParaLSTM | 2048 | 6.41 | **5.45** | **1.18×** |
| ParaLSTM | 4096 | 12.4 | **10.1** | **1.22×** |

Peak allocated MiB. Ratio is fp32 / fp16.

| Cell | T | fp32 | fp16 | vs fp32 |
|---|---|---|---|---|
| ParaGRU | 64 | 15 | 13 | 1.2× |
| ParaGRU | 256 | 25 | 17 | 1.5× |
| ParaGRU | 512 | 39 | 24 | 1.6× |
| ParaGRU | 2048 | 123 | **66** | **1.86×** |
| ParaGRU | 4096 | 235 | **122** | **1.92×** |
| ParaLSTM | 64 | 19 | 16 | 1.2× |
| ParaLSTM | 256 | 35 | 22 | 1.6× |
| ParaLSTM | 512 | 59 | 34 | 1.7× |
| ParaLSTM | 2048 | 205 | **107** | **1.91×** |
| ParaLSTM | 4096 | 399 | **204** | **1.96×** |

Memory at LM length is **~2×** as expected (half DRAM). Time is **1.2–1.45×** at \(T\ge 2048\), not 2×: the scan is still math-heavy in fp32, and `W_x` is one GEMM. Short \(T\) is **slower** (fp16↔fp32 convert + launch; GEMM too small for TC). Max |fused−seq| was \(9.8\times10^{-4}\) (LSTM \(T=4096\): \(1.2\times10^{-3}\)). This is not AMP-like-A100 and not 665×.

## Gemini vs measurement

| Suggestion | Keep? | Correction |
|---|---|---|
| compile / CUDA Graphs | Yes as **opt-in at fixed \(T\)**, not a library default | Steady-state is real. Compile tax is 4–113 s. Do not quote App. B `min_ms` as if Dynamo were free. |
| Fused CUDA/Triton PCR | Yes, **fused Newton is in** (`scan_backend="fused"`) | Still not Apple's kernel. Vs naive ParaRNN **~8–11×** at LM length; vs naive RNN **~200–400×**. Not 665×. |
| FP16/BF16 + tensor cores | fp16 DRAM + fp32 accum **is in**; bf16 fused on CC ≥ 8.0 | SM 7.x has no bf16 TC. TC help `W_x` only. Scan is not 2× at short \(T\) (slower). Residual \(\sim10^{-3}\). |

## Profile

`uv run python scripts/profile_hotpath.py`. Traces: `outputs/profile/`. MLflow `hotpath-profile`. 10/20 runs, not App. B.

**Sequential without J** (before compiled baseline):

| Case | App. B (J in `step`) | `step` only | Ratio |
|---|---|---|---|
| ParaGRU \(T=64\) | 49.8 ms | 41.9 | 1.19× |
| ParaGRU \(T=2048\) | 1654 | 1180 | 1.40× |
| ParaLSTM \(T=512\) | 671 | 412 | 1.63× |

**Newton CUDA** (ATen `self_cuda`; ignore `record_function` %):

- GRU \(T=64\): CUDA 7.9 ms, CPU 167 ms → launches. Compile-Newton cuts this **after** Dynamo pays.
- GRU \(T=2048\): `mul` 40%, `add` 21%, `cat` 16% (cat since removed), `W_x` 8%.
- LSTM \(T=512\): **`aten::bmm` was 86%** (fixed: elementwise 2×2 + Blelloch).

## Do not

- 3060 as a memory overflow. Stay on 2080 Ti by **name**.
- Copy Apple `csrc/`.
- Bake `torch.compile` into `src/`. Opt-in at the call site; log `compile_s`.
- Quote compiled `min_ms` without the compile wall time.
- Quote fused × as fig. 2/5 or 665×. The smoke table is 2080 Ti, 10/50, vs *our* naive RNN / naive ParaRNN.
- bf16 as a scan speedup, or GPU **names** in `src/` (capability only). Lab benches still pin the 2080 Ti in `scripts/gpu.py`.
- Quote the first App. B 50× as if it were vs compiled sequential.
- Compare our ms to fig. 2/5.

MLflow: App. B → `cell-forward-bench`. Compile-Newton → `newton-compile-bench`. Fused Newton → `newton-fused-bench`. fp16 fused → `newton-fp16-bench`. Profiler → `hotpath-profile`. Tag `gpu=NVIDIA GeForce RTX 2080 Ti`.
