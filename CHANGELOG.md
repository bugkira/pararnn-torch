# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use [SemVer](https://semver.org/). This is 0.x: a **minor** may include a break; **1.0.0** waits until the public API is a contract.

## [Unreleased]

### Added

- Colab / Jupyter demo [`notebooks/paraslstm_demo.ipynb`](notebooks/paraslstm_demo.ipynb):
  drop-in API, sequential↔Newton trust table, latency vs \(T\), short
  \(\mathbb{Z}_2\) parity vs S4D-Real SSM, T=1 `decode_step`, citation
  ([`notebooks/README.md`](notebooks/README.md)).

### Changed

- Parity demo / [`examples/parity.py`](examples/parity.py): explicit `picard_iters=3`
  at \(T{=}16\) (library auto is P=1) and quiet `pararnn.solvers.newton` WARNING
  so `newton_residual_high` does not flood Colab / stdout; divergence still raises.
- Expressivity section split into models / train / plot cells (Colab-sized chunks).

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
  `h0` length \(S\). Segmented scan (eager Hillis–Steele; Triton `scan_diag`
  vs `offs_t`; block2/4 `J=0` at heads). Fused in-kernel packing is ParaGRU.
- `pararnn.distributed`: `warmup_scan_kernels` and `last_newton_residuals`.
  `ParaRNN` wraps with DDP or FSDP2 `fully_shard` (`examples/ddp_fsdp.py`).
- Tensor parallel along \(d_h\) for channelwise-diagonal cells:
  `tensor_parallel_diag_block`, one AllReduce on the output projection.
  Torchrun demo: `examples/tensor_parallel.py` on `archive/distributed-demos`.
- Context-parallel diag scan: `scan_diag_context_parallel` AllGathers the
  tile monoid \((P_{\mathrm{end}}, \delta_{\mathrm{end}})\). Rank 1 applying
  that carry is the numeric check. `NewtonConfig(scan_backend="context_parallel")`
  shards scan work \(T/N\) on a replicated Newton trajectory. Torchrun demo:
  `examples/context_parallel.py` on `archive/distributed-demos`.
- `verify_linear_draft`: greedy speculative verify of a K-token chain. One
  Newton (or sequential) unroll from `h0`, first mismatch \(k^\star\), state
  truncated to \(h_{k^\star}\), bonus token from the leftover logit
  (`examples/speculative_draft.py`).
- Long-\(T\) scan past SRAM pads: chunked adjoint; hierarchical / eager-aggregate
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

[Unreleased]: https://github.com/bugkira/pararnn-torch/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/bugkira/pararnn-torch/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bugkira/pararnn-torch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bugkira/pararnn-torch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/bugkira/pararnn-torch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bugkira/pararnn-torch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bugkira/pararnn-torch/releases/tag/v0.2.0
