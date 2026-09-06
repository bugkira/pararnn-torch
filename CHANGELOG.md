# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use [SemVer](https://semver.org/). This is 0.x: a **minor** may include a break; **1.0.0** waits until the public API is a contract.

## [Unreleased]

## [0.17.1] - 2026-09-06

``NewtonConfig(max_iters=None)`` fills Newton depth from a measured
``K*(T)`` envelope. Triton launches pin the tensor's CUDA device.

### Added

- `NewtonConfig(max_iters=None)` resolves `K` via `auto_newton_iters`
  (cell tables: CfC / Hopfield / Titans; RWKV-7 returns 0; M²RNN log-ramp).
  Pin `max_iters=int` or pass `newton_iters_by_t={T: K, …}` (left-step).
  Envelopes: CfC/Titans ceiling 3; Hopfield `{1: 2, 64: 3}`; τ≈1e-4 through
  T=131072.

### Changed

- README capability-first (News, Models, Install, Quickstart); `pyproject.toml`
  description matches. Colab demo is a short API poke (no in-notebook train).
- Hopfield / CfC / Titans recipes use `max_iters=None`. App. A GRU/LSTM keep
  default `K=3`. Numerics pins for Hopfield drop from 6 to 3.

### Fixed

- Fused/Triton dtype gate calls `torch.cuda.set_device` so the launch matches
  the tensors' card.

## [0.17.0] - 2026-09-06

`ParaTitans`: shallow L=1 Titans-inspired neural memory (vector state;
channelwise-diagonal Newton Jacobian). Deep multi-layer MLP memory is parked.

### Added

- `ParaTitans`: surprise GD on elementwise associative loss
  $`\ell=\tfrac12\|h\odot k-v\|^2`$ plus diagonal tanh polish
  $`u\odot\tanh(W_n x+r\odot h)`$; gates from `Linear(d_in → 5 d_h)`;
  analytic diag `step_with_jacobian`; eager / Triton diag scan; fused Alg. 1
  (`pararnn::newton_titans_fused`); Autograd eq. 2.6 VJP. Adoption slot in
  [`docs/adoption.md`](docs/adoption.md).

## [0.16.0] - 2026-09-06

`ParaRWKV7`: RWKV-7 Goose matrix-state delta monoid (linear in $S$;
factorized apply + associative scan; Newton redirects).

### Added

- `ParaRWKV7`: per-head state $`S\in\mathbb{R}^{d_{\mathrm{head}}\times d_{\mathrm{head}}}`$
  with Goose transition
  $`S_t = S_{t-1}G_t + v_t^\top k_t`$,
  $`G_t=\mathrm{diag}(w_t)-\hat\kappa_t^\top(a_t\odot\hat\kappa_t)`$
  (Peng et al., arXiv:2503.14456). Gates from `Linear(d_in → 6 n_heads d_head)`;
  factorized `step` (no dense $G$); receptance readout
  $`y=\mathrm{flatten}(S@r)`$; Hillis–Steele associative scan of the
  $(G,U)$ monoid (`scan_apply` / `newton_apply` default).
  `newton_apply` redirects to the linear scan (or `sequential_apply` when
  `scan_backend='eager'`). Adoption slot in
  [`docs/adoption.md`](docs/adoption.md).

## [0.15.0] - 2026-09-06

`ParaHopfield`: recurrent Modern-Hopfield slot
$`h_t = V_t\,\mathrm{softmax}(\beta K_t h_{t-1})`$ with dense Newton Jacobian
and `scan_dense` (small $`d_h`$).

### Added

- `ParaHopfield`: $`K_t,V_t\in\mathbb{R}^{d_h\times d_h}`$ from
  `Linear(d_in → 2 d_h²)`; default $`\beta=1/\sqrt{d_h}`$; analytic dense
  `step_with_jacobian`; eager / Triton `scan_dense` (no fused cell+scan yet);
  Autograd eq. 2.6 VJP (`uses_packed_vjp` false). Docs/tests cap d_h ≤ 32
  (warn above). Adoption slot in [`docs/adoption.md`](docs/adoption.md).
  Honest smoke (`scripts/bench_hopfield.py`, CUDA events min, B=8,
  T=2048, d_h=16): measured K*=2 (~1e-7 vs sequential);
  recipe `max_iters=3`. RTX 3060 Triton `scan_dense` ~12.5 ms vs sequential
  ~504 ms (~40×) at K=3; RTX 2080 Ti at K=6 was ~16.1 ms vs ~525 ms
  (~33×). Eager Newton loses to sequential at short T (`T≲64`); `auto`
  picks Triton. Fused cell+scan / packed VJP parked. Peak mem at
  $`d_h=32,T=2048`$ ~617 MiB (dense $J$).
- Lab: [`scripts/bench_hopfield.py`](scripts/bench_hopfield.py).

## [0.14.0] - 2026-09-06

`ParaCfC`: Liquid-style closed-form continuous-time cell with irregular Δt
and diagonal nonlinear mix (Newton Jacobian stays channelwise diagonal).

### Added

- `ParaCfC`: features in `x[..., :-1]`, Δt in `x[..., -1]` (`d_in >= 2`);
  gate `a = σ(-softplus(f)·Δt)` and candidate `tanh(c + u⊙h)` with App. C.1
  clip on `u`. Eager Newton, fused Triton Alg. 1 (`pararnn::newton_cfc_fused`,
  3-wide `project_wx`), packed VJP (softplus·Δt chain; tile `tl.sum`, no
  atomics), compile-safe fullgraph path. `scan_backend='auto'` picks fused on
  CUDA. Honest smoke (`scripts/bench_cfc.py`, CUDA events min, K=3,
  B=8, T=2048, d_h=256): RTX 3060 fused ~2.79 ms vs sequential
  ~643 ms (~230×), max \|err\| ~2e-7; RTX 2080 Ti fused ~2.86 ms vs ~656 ms.
  K=2 already reaches ~1e-7 agreement; packed VJP ~8× vs eager formula on
  3060. At short T (`T≤8`) fused still beats sequential; eager Newton loses
  to sequential below ~T=32.
- Adoption: Liquid / irregular-Δt slot in [`docs/adoption.md`](docs/adoption.md).
- Lab: [`scripts/bench_cfc.py`](scripts/bench_cfc.py) (seq / eager / triton /
  fused + residual-vs-K + optional fwd+bwd / packed VJP).

## [0.13.0] - 2026-09-06

Product entry (block / CausalLM) plus `ParaNLRU`: a nonlinear RG-LRU-style
diag cell for Griffin / RecurrentGemma slots.

### Added

- `ParaNLRU`: nonlinear RG-LRU-style cell (input-only gate + diagonal `u` +
  `tanh` in the step) with eager Newton, fused Triton Alg. 1
  (`pararnn::newton_nlru_fused`), packed VJP (no atomics), compile-safe
  fullgraph path. Smoke (RTX 3060, B=8, T=2048, d_h=256,
  K=3): fused ~2.7 ms vs sequential ~549 ms.
- Product entry path: README Quickstart leads with `ParaSLSTMBlock` /
  `ParaSLSTMForCausalLM`; [`docs/adoption.md`](docs/adoption.md) covers
  Attention swap, CausalLM, Dreamer RSSM slot, and the Griffin / NLRU slot.
- `ParaSLSTMForCausalLM.forward(..., labels=)` returns shifted CE loss;
  `save_pretrained` / `from_pretrained` write and prefer `model.safetensors`
  (dependency `safetensors`), with `pytorch_model.bin` fallback.
- Examples: `examples/causal_lm_smoke.py`, `examples/rssm_recurrent.py`.

### Changed

- CI: merge gate is lint + CPU tests only; self-hosted CUDA moved to
  `gpu.yml` (offline lab runners no longer leave the `ci` badge queued/red).
  Release: GitHub Release always publishes on tag; PyPI Trusted Publishing is
  best-effort until the PyPI publisher is linked.

## [0.12.0] - 2026-09-06

Research cell `ParaM2RNN`: factorized parallel Newton for matrix-state
M²RNN (Mishra et al. arXiv:2603.14360), answering the dense-Jacobian /
unknown-K objections in their §2.5.3 with structured $J$ and measured
$`K^{*}(T)=\Theta(\log T)`$.

### Added

- Research cell `ParaM2RNN` (Mishra et al. arXiv:2603.14360): matrix state
  `H∈R^{K×V}`, factorized Newton (`m2rnn_jvp` / `newton_m2rnn_factorized`)
  without dense `(KV)²`, wired through `newton_apply` + eq. 2.6 reverse
  (autograd cell VJP). CUDA Triton fused: SRAM `K,V≤64`; hybrid tiled scan for
  larger `K,V` with `residual_atol` early-stop (avoids losing to sequential when
  fixed `K` overshoots `K*`); packed VJP (`m2rnn_recurrence_vjp`). `auto` picks
  fused on CUDA. Optional frozen-`W` warm-start via `picard_iters=1`
  (`m2rnn_frozen_w_scan`); under default `W=I` measured $`\Delta K^*\approx 0`$,
  class still $`\Theta(\log T)`$. Compile-safe + fullgraph CUDA fused and
  deterministic bitmatch covered in `tests/numerics/test_compile.py` /
  `test_vjp_determinism.py`. Recipe: `scripts/bench_m2rnn_k_scale.py --init both`.

### Changed

- Literature PDF cache path: `docs/papers/` → `docs/sources/`
  (`scripts/fetch_papers.sh`, gitignore).

## [0.11.0] - 2026-09-06

Beck-style `ParaSLSTM(mix='head')`: factorized CUDA Newton, reverse, and VJP.

### Added

- Factorized fused Newton for `ParaSLSTM(mix='head')`: `slstm_head_jvp` /
  `slstm_head_jt_mvp`, `pararnn::newton_slstm_head_fused` /
  `pararnn::reverse_slstm_head_factor` (no dense `(4d)×(4d)`). CUDA tiers:
  `d_head ≤ 32` fused (all four `R_g` in SRAM), `32 < d_head ≤ 128`
  streamed-`R`, larger / CPU factorized eager. Picard / zero-hidden warm
  start unchanged. Recipe from measured snaps: `K=4`, `omega=1`.
  `scan_backend='eager'` keeps the dense-J oracle. Rectangular batches;
  `cu_seqlens` + explicit `fused` raises; `auto` remaps to `eager`.
- Packed closed-form VJP for `ParaSLSTM(mix='head')`:
  `slstm_head_recurrence_vjp` (SRAM Triton `d_head ≤ 32`, eager else) wired
  through `uses_packed_vjp` / eq. 2.6 (`∇R_head` via outer `h⊗d_z`).
- Bench [`scripts/bench_slstm_head.py`](scripts/bench_slstm_head.py)
  (`--scan-backend compare|fused`, `--d-head-grid`). RTX 2080 Ti float32
  medians — `B=4 K=4`: Newton `d_head=32` ~20 ms / seq ~880 ms at `T=1024`;
  stream `d_head=64` ~31 ms; `96/128` ~174–180 ms. Reproduce:
  `uv run python scripts/bench_slstm_head.py --device cuda`.

### Changed

- `ParaSLSTM(mix='head')` warning: factorized CUDA path (diag remains the
  default 4×4 SRAM product cell).
- Hot-path `log.debug` in Picard / fused sLSTM gated for `torch.compile`
  (`fullgraph=True` head-sLSTM infer/train).

## [0.10.0] - 2026-09-06

Dreamer-style `ParaGRU(mix='head')`: factorized CUDA Newton, reverse, and VJP.

### Added

- `ParaGRU(mix='head', n_heads=…)`: block-diagonal recurrent `A_z,A_r,A_n`
  (Dreamer-style block recurrence) with full `W_x` input mix. Cho gates only —
  Dreamer LN-GRU keeps LayerNorm outside the cell. Default remains `mix='diag'`.
  Head default `max_recurrent_norm=None` (no silent clamp on dense `A_*`).
- Factorized head Jacobian matvecs (`gru_head_jvp` / `gru_head_jt_mvp`) and
  CUDA fused Newton for `ParaGRU(mix='head')` via
  `pararnn::newton_gru_head_fused` / `pararnn::reverse_gru_head_factor`
  custom ops (Dynamo-opaque; `register_fake`): one Triton program per
  `(batch, head)` does gates + residual + factorized `J δ`; `d_head ≤ 64`
  full SRAM, `64 < d_head ≤ 128` streamed-`A` SRAM, larger heads PyTorch
  gates + tiled Triton factor scan. `scan_backend` in
  `{auto,triton,fused}` on CUDA; `eager` keeps the dense-J oracle.
- Packed closed-form VJP for `ParaGRU(mix='head')`: CUDA Triton
  (`gru_head_recurrence_vjp`; SRAM for `d_head≤64`, T-parallel tiled pre +
  outer kernels for larger) with eager oracle (`gru_head_recurrence_vjp_eager`).
  Deterministic `∇A` via per-batch fp32 tiles + `.sum` (outer: one write per
  `(B,head,i_tile,o_tile)`, no atomics).
- Factorized reverse for `d_head>64`: streamed-`A` SRAM path through
  `d_head≤128` (gates+`J^T` in one kernel); above that, tiled Triton `J^T`
  with fused `A_z`/`A_n` tile pass.
- CUDA `scan_dense` / `reverse_scan_dense` (`backend='triton'`): tiled row-wise
  Triton inclusive / reverse scans (any `d_head`; fp32 algebra). Bench
  [`scripts/bench_gru_head.py`](scripts/bench_gru_head.py) (`--scan-backend compare`,
  `--d-head-grid`).
- Head long-T (checked through `T=4096`): no hard pad on fused/stream/hybrid.
  RTX 2080 Ti float32 medians — `B=4 T=128 K=3`: Newton `d_head=64` ~2.4 ms,
  `96/128` ~9–10 ms, `192` ~30 ms; `B=1 T=4096`: Newton `64` ~51 ms / `96`
  ~220 ms, reverse ~63–86 ms, VJP ~4–14 ms, fwd+bwd ~83 / ~293 ms. Reproduce:
  `uv run python scripts/bench_gru_head.py --device cuda`.
- `NewtonConfig(recompute=True)` covered for `ParaGRU(mix='head')` fused
  (Level-2 rematerialize for long-T VRAM; same knob as diag).

### Changed

- Head-cell honesty: `UserWarning` on `ParaGRU(mix='head')`; docs state
  factorized CUDA path, eager dense oracle, and LN mismatch.
- Head + `cu_seqlens`: explicit `scan_backend='fused'` raises `TypeError`
  (no silent fused→eager); `auto` remaps to `eager` with `UserWarning`.
  Ragged packs use `scan_backend='eager'`.

## [0.9.0] - 2026-09-06

LM trunk, continuous-batch serve, vLLM plugin, and public API docs.

### Added

- `ParaSLSTMBlock` / `SwiGLU`: pre-norm RMSNorm + fused-ready `ParaSLSTM` +
  SwiGLU residual trunk (`layers/para_slstm_block.py`).
- Standalone `ParaSLSTMForCausalLM` + `ParaSLSTMConfig` (`pararnn.models`):
  HF-shaped `config.json`, tied LM head, `generate()` via `decode_step`.
- Continuous-batch serve path: `BlockStackPool` / `forward_continuous`
  (packed `cu_seqlens` + shared `slot_ids` → `paged_apply` /
  `decode_step(..., block_table=)`). Example
  [`examples/continuous_batch.py`](examples/continuous_batch.py).
- vLLM engine path: `ParaSLSTMRecurrentLayer` (`MambaBase`, `mamba_type=MAMBA1`)
  consumes worker cache pages + `Mamba1AttentionMetadata` indices; decoder
  stack + `get_mamba_state_*` / copy funcs (`docs/vllm.md`).
- vLLM out-of-tree plugin: `vllm.general_plugins` entry
  `pararnn_paraslstm` → `ModelRegistry.register_model("ParaSLSTMForCausalLM", …)`
  (lazy string). Optional extra `vllm`.
- `pararnn.determinism`: under `torch.use_deterministic_algorithms(True)`, one-shot
  warnings for missing `CUBLAS_WORKSPACE_CONFIG` and packed eq. 2.6 param-grad
  drift; wired from `cell_vjp` (`tests/numerics/test_vjp_determinism.py`).
- Numpydoc-style docstrings on root `__all__` (cells, Newton config/apply,
  paged/decode, CausalLM); Triton `@jit` kernels stay module-level
  (`docs/README.md`).

## [0.8.0] - 2026-09-06

Solver coverage, long-T memory knobs, and the Colab API demo.

### Added

- Colab / Jupyter demo [`notebooks/paraslstm_demo.ipynb`](notebooks/paraslstm_demo.ipynb):
  short API tour (forward, trust, latency, decode) plus citation links
  ([`notebooks/README.md`](notebooks/README.md)).
- Fused CUDA numerics for odd $`d_h`$ / odd T (GRU/LSTM/sLSTM, incl. T=127).
- Overflow-stress tests: NaN / exploded sLSTM → `NewtonDivergenceError`;
  log-decode huge $n$ stays finite (`tests/numerics/test_overflow_stress.py`).
- `NewtonConfig(recompute=True)`: Level-2 selective activation checkpointing —
  rematerialize $`H^\star`$ in eq. 2.6 backward for ultra-long train T
  (`tests/numerics/test_recompute.py`).
- Experimental `NewtonConfig(fused_early_exit=True)`: host ``max|F|`` stop
  between fused Newton steps (fixed K remains the train default).

### Changed

- Direct Triton pin on Linux: `triton>=3.6.0,<3.7` (matches torch 2.11 cu128).
- Parity demo / [`examples/parity.py`](examples/parity.py): explicit `picard_iters=3`
  at T=16 (library auto is P=1) and quiet `pararnn.solvers.newton` WARNING
  so `newton_residual_high` does not flood Colab / stdout; divergence still raises.
- Demo notebook is an API poke only; $`\mathbb{Z}_2`$ training stays in
  [`examples/parity.py`](examples/parity.py).

## [0.7.0] - 2026-09-05

Compile-safe Newton, paged decode path, and OSS front-door polish.

### Added

- Compatibility tests: `torch._dynamo.explain` reports **0** graph breaks under
  the compile-safe preset; `fullgraph=True` inference/training match eager;
  `torch.autocast` smoke; non-reentrant checkpoint + `state_dict`
  (`tests/numerics/test_{compile,autocast,checkpoint}.py`).
- Eager Newton / Blelloch: fp32 accumulators for bf16 DRAM (same path as fp16).
- Triton scans as `torch.library.custom_op` (`pararnn::scan_diag` /
  `scan_block2` / `scan_block4`) with `register_fake` for Dynamo meta
  (`kernels/custom_ops.py`, `tests/numerics/test_custom_ops.py`).
- `newton_apply`: top-level `_NewtonFixedPoint`; non-tensor backward state on
  `ctx` (no TLS — autograd may run backward on a worker thread). Pure fixed-K
  loop (`_newton_forward_pure`); stats / residual early-stop / logging only
  outside `torch.compiler.is_compiling()`.
- Fused Alg. 1 as `pararnn::newton_{gru,lstm,slstm}_fused` custom ops with
  `register_fake` (Dynamo-opaque Triton). Eq. 2.6 remains on
  `_NewtonFixedPoint` (fused inputs lack `W_x`).
- `PagedStatePool` / `paged_apply`: O(1) GPU slot per request (sLSTM
  `(c,n,m,h)`, LSTM `(c,h)`, GRU `h`). Host free-list, `index_select` /
  `index_copy_`, mixed packed prefill+decode via `cu_seqlens`. Sequential
  CUDA writes slots in-kernel through `block_table` for every T. Fused Newton
  loads pool `h0` via the same ids (`newton_apply(..., block_table=)`). `offload` /
  `reload` copy a slot to pinned host RAM and back (`host_capacity` defaults to
  GPU `capacity`). Torchrun demo: `examples/paged_cache.py` on
  [`archive/distributed-demos`](https://github.com/bugkira/pararnn-torch/tree/archive/distributed-demos).
- `decode_step`: T=1 Triton kernel for the recurrent step (gates + mix in one
  SRAM trip). `decode_wx` fills `W_x(x)` into a buffer. `out=` reuses storage;
  `block_table` indexes a pool `(C, …)`. App. C.1 clip is in-kernel so a CUDA
  graph of GEMM → step does not allocate. `sequential_apply` / `ParaRNN.eval()`
  take this path on CUDA when `T=1` and gradients are off
  (`examples/decode_step.py`).
- `SECURITY.md`, bug-report / PR templates, optional `.pre-commit-config.yaml`.
- [`scripts/README.md`](scripts/README.md) index of benches vs diagnostics.
- README Results table and Reproduce commands (2080 Ti medians; Z₂ parity).

### Changed

- `newton_apply` disables outer CUDA/CPU autocast for the solve and eq. 2.6
  backward so `W_x` and states share one dtype. Half-precision training still
  uses explicit `.to(dtype)`.
- README Compatibility blurb: compile, AMP policy, DDP/FSDP / checkpoint
  pointers; Zenodo + ParaRNN ICLR badges. Install path is git until the first
  PyPI Trusted Publishing upload.
- FlashRNN Dyck comparison lives under `scripts/slstm_vs_flashrnn.py`; examples
  stay onboarding.
- Two-card TP / CP / paged-pool demos removed from `examples/` on `main`
  (history on `archive/distributed-demos`). API + numerics tests stay;
  `docs/distributed.md` keeps the architecture.
- `examples/toy_copy.py` → `train_smoke.py`: standalone script (inlined knobs,
  no YAML / repo `sys.path`); stderr only; per-step identity CE. Same style
  across `examples/` (`print`, no MLflow; dyck/parity knobs in-file).
- CI: self-hosted GPU jobs use `continue-on-error` so an offline runner leaves
  lint + CPU tests as the merge gate.
- Docs honesty: CHANGELOG / `docs/structure.md` point archived torchrun demos
  at `archive/distributed-demos`; surface version matches the package.

## [0.6.0] - 2026-09-04

Ragged time, data/tensor/context parallel, and greedy linear-draft verify.
Patch-scale work from the same day (CI pin, ruff format, README onboarding)
sits here too.

### Added

- Packed ragged sequences: `cu_seqlens` on `ParaRNN` / Newton / sequential,
  `h0` length $S$. Segmented scan (eager Hillis–Steele; Triton `scan_diag`
  vs `offs_t`; block2/4 `J=0` at heads). Fused in-kernel packing is ParaGRU.
- `pararnn.distributed`: `warmup_scan_kernels` and `last_newton_residuals`.
  `ParaRNN` wraps with DDP or FSDP2 `fully_shard` (`examples/ddp_fsdp.py`).
- Tensor parallel along $`d_h`$ for channelwise-diagonal cells:
  `tensor_parallel_diag_block`, one AllReduce on the output projection.
  Torchrun demo: `examples/tensor_parallel.py` on `archive/distributed-demos`.
- Context-parallel diag scan: `scan_diag_context_parallel` AllGathers the
  tile monoid $`(P_{\mathrm{end}}, \delta_{\mathrm{end}})`$. Rank 1 applying
  that carry is the numeric check. `NewtonConfig(scan_backend="context_parallel")`
  shards scan work $T/N$ on a replicated Newton trajectory. Torchrun demo:
  `examples/context_parallel.py` on `archive/distributed-demos`.
- `verify_linear_draft`: greedy speculative verify of a K-token chain. One
  Newton (or sequential) unroll from `h0`, first mismatch $`k^\star`$, state
  truncated to $`h_{k^\star}`$, bonus token from the leftover logit
  (`examples/speculative_draft.py`).
- Long-T scan past SRAM pads: chunked adjoint; hierarchical / eager-aggregate
  tile scan (`docs/backward-scan-cap.md`).
- CUDA tests that fused Newton and eq. 2.6 agree with sequential BPTT on
  time-strided, feature-strided, and permute-roundtrip `x`
  (`tests/numerics/test_noncontiguous.py`).

### Changed

- Packed `_linear_vjp` uses `.reshape(x.shape)` on the input-map VJP so a
  strided `(B, T, d_in)` view keeps the caller layout.
- Fused sLSTM Newton keeps the guess, J tiles, and scan residual in fp32 when
  DRAM is fp16/bf16 (`fp32_newton_work`). `W_x` stays a GEMM in the tensor
  dtype (App. B algebra).
- CI: `setup-uv` pinned to v10.0.1.

## [0.5.0] - 2026-09-03

Windowed / Thomas fused Newton, and the library `xLSTMBlock` comes off the
installable surface.

### Removed

- Public `xLSTMBlock` (LayerNorm + residual around `ParaRNN(ParaSLSTM)`).
  Use `ParaRNN(ParaSLSTM(...))`; LN/FFN stacking is the caller's, or
  `examples/xlstm_hybrid.py` for an NX-AI `sLSTMBlock` around the same cell.

### Added

- `NewtonConfig.scan_tile`: `thomas` / `thomas4` (C=4 sequential compose then
  PCR of T/C) and `thomas2` (C=2). Default `assoc` until a bench on the target
  GPU prefers Thomas.
- `NewtonConfig.fused_time_loop`: one fused launch walks `fused_window_len`
  (default 64) with solved-state carry (same residual as `chunk_len`).
- `NewtonConfig.fused_window_len`: `32` / `64` / `128`. `64` sat at ~2e-4 vs
  sequential at T=1024 `d_h=256` P=3 in this repo; `32` trips
  `newton_residual_high`.
- `examples/xlstm_hybrid.py`: NX-AI pre-norm / skip / FFN with
  `ParaRNN(ParaSLSTM mix='diag')` in the recurrent slot (`uv add xlstm`;
  Python 3.11+).
- Zenodo DOI for the ParaSLSTM preprint
  ([10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587)).

### Changed

- `ParaSLSTM`: `mix='diag'` remains the fused default. `mix='head'` warns as
  an unfused ablation (`scan_dense`). `mix='dense'` raises if
  `hidden_size > 8` (Jacobian oracle for tests).
- README quickstart runs `.backward()` on the Newton path.

### Breaking

- Importing `xLSTMBlock` from `pararnn` fails. Callers that wrapped
  `ParaRNN(ParaSLSTM)` themselves keep working.

## [0.4.0] - 2026-09-03

### Added

- `ParaSLSTM` with 4×4 Newton scan (channelwise block Jacobian per feature).
- Analytic sLSTM Jacobian; Newton auto path selects it.
- Triton 4×4 scan and fused diagonal-mix sLSTM kernel.
- Zero-hidden Newton init for sLSTM (running max-plus `m` / linear `n` from the `h=0` unroll); App. A `f(0, x_t)` alone needs many more Newton steps at long `T`.
- xLSTM head-block mixing (`mix='head'`): dense `d_head×d_head` Jacobian per head in the scan.
- Auto Picard iteration count for sLSTM from sequence length (`P=1` for `T≤64`, else `P=3`).
- sLSTM Picard and log-coordinate solvers split into `solvers/slstm_picard.py` and `solvers/slstm_log.py`.
- `xLSTMBlock` (pre-norm residual sLSTM layer).
- Dyck-language smoke test for parallel training with eq. 2.6 backward.
- `device=` / `dtype=` / `reset_parameters()` on `ParaRNN` and `xLSTMBlock`; `extra_repr()` on cells; `dropout` between layers; input validation in `forward`.
- Top-level re-exports: `newton_apply`, `sequential_apply`, `NewtonStats`, `NewtonDivergenceError`, `RNNCell`.
- `gradcheck` coverage for the hand-written eq. 2.6 adjoint; `torch.compile` limits pinned by tests.
- CUDA test marker (`@pytest.mark.cuda`), `tests/conftest.py` `cuda_device` fixture, and `--strict-markers` so CPU runs report skipped GPU coverage instead of silently passing.
- CI: ruff lint job; Python 3.10/3.11/3.12 matrix; self-hosted GPU matrix on Ampere (RTX 3060) and Turing (RTX 2080 Ti) with `CUDA_DEVICE_ORDER=PCI_BUS_ID` pinning.
- Ruff adopted with ruleset `E,F,W,I,UP,B,SIM,RUF,C4,PT` and 100-char line length.

### Changed

- Library surface trimmed to cells, solvers, and `xLSTMBlock`; training smokes and MLflow helpers moved under `examples/` and `scripts/`.
- Fused/Triton bf16 gated by compute capability ≥ 8.0 on the tensor device; `scan_backend="auto"` falls back on Turing-class GPUs.
- Newton fails loud if residual stays huge after `K=3` on sLSTM.
- Packed VJP: single formula source per cell; deterministic tile-sum reduction replaces fp32 atomics (GRU parameter gradients were previously not reproducible run to run).
- Picard buffer reuse and in-kernel GRU `a_*` reduction to cut VRAM.
- bf16 numerics tolerance calibrated to bf16 unit roundoff (1 ULP = 2⁻⁸) instead of reusing the fp16 constant.
- Fused Newton and block-scan scaffolding shared between cells (`kernels/_fused_common.py`, `kernels/_scan_common.py`).
- `solvers/newton.py` split into a package (`config`, `dispatch`, `picard`, public API).

### Breaking

- `ParaRNN` with `hidden_layout="pytorch"` now returns `(output, (h_n, c_n))` like `nn.LSTM`, and final states stack as `(num_layers, B, H)`; `hidden_layout="paper"` is unchanged.

## [0.3.0] - 2026-09-01

### Added

- `NewtonStats` (residual history, iteration count) on the Newton API.
- `scan_backend="auto"`: fused Triton on CUDA when supported, else Triton scan + `step`, else eager Blelloch.
- Fused kernels prepend `h0` (App. A init inside the kernel).
- `ParaRNN` accepts a cell list; `return_hidden` / `output_hidden` flags.

### Changed

- Default `scan_backend` follows the input tensor device instead of a silent fused default.

## [0.2.0] - 2026-08-31

### Added

- `ParaRNN` as a trainable `nn.Module`: `train()` uses Newton+scan, `eval()` sequential unroll.
- `ParaGRU`, `ParaLSTM` cells with Newton + associative scan solvers.
- Fused Triton Newton kernels for GRU/LSTM (diagonal Jacobian blocks).
- Generic-cell autograd path and sequential reference solver.
- Numerics tests: parallel vs sequential agreement, layer forward/backward.

[Unreleased]: https://github.com/bugkira/pararnn-torch/compare/v0.17.1...HEAD
[0.17.1]: https://github.com/bugkira/pararnn-torch/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/bugkira/pararnn-torch/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/bugkira/pararnn-torch/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/bugkira/pararnn-torch/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/bugkira/pararnn-torch/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/bugkira/pararnn-torch/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/bugkira/pararnn-torch/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/bugkira/pararnn-torch/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/bugkira/pararnn-torch/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/bugkira/pararnn-torch/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bugkira/pararnn-torch/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bugkira/pararnn-torch/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bugkira/pararnn-torch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bugkira/pararnn-torch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bugkira/pararnn-torch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bugkira/pararnn-torch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bugkira/pararnn-torch/releases/tag/v0.2.0
