# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use [SemVer](https://semver.org/).

## [Unreleased]

### Removed

- Public `xLSTMBlock` (it was LayerNorm + residual around `ParaRNN(ParaSLSTM)`, not a second sLSTM cell). Use `ParaRNN(ParaSLSTM(...))` in the library; LN/FFN stacking is the caller's, or `examples/xlstm_hybrid.py` for an NX-AI `sLSTMBlock` around the same cell.

### Added

- CUDA tests that fused Newton and eq. 2.6 agree with sequential BPTT on time-strided, feature-strided, and permute-roundtrip `x` (`tests/numerics/test_noncontiguous.py`). CPU eager GRU covers the packed-VJP reshape on the same layouts.
- Zenodo DOI for the ParaSLSTM preprint ([10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587)) on the README.
- `examples/xlstm_hybrid.py`: NX-AI pre-norm / skip / FFN with `ParaRNN(ParaSLSTM mix='diag')` in the recurrent slot. Install NX-AI `xlstm` separately (`uv add xlstm`; Python 3.11+).
- `NewtonConfig.scan_tile`: `thomas` / `thomas4` (C=4 sequential compose then PCR of T/C) and `thomas2` (C=2). Default stays `assoc` until a bench on the target GPU prefers Thomas.
- `NewtonConfig.fused_time_loop`: one fused launch walks `fused_window_len` (default 64) with solved-state carry (same residual as `chunk_len`, not global Newton).
- `NewtonConfig.fused_window_len`: `32` / `64` / `128` for that walk. `64` is the `chunk_len` that sat at ~2e-4 vs sequential at T=1024 `d_h=256` P=3 in this repo; `32` trips `newton_residual_high`.

### Changed

- README quickstart runs `.backward()` on the Newton path.
- Packed `_linear_vjp` uses `.reshape(x.shape)` on the input-map VJP so a strided `(B, T, d_in)` view keeps the caller layout.
- Fused sLSTM Newton keeps the guess, J tiles, and scan residual in fp32 when DRAM is fp16/bf16 (`fp32_newton_work`). `W_x` stays a GEMM in the tensor dtype. This is the App. B algebra contract. It tightens P=3 bf16 residual; it does not make Ampere bf16 fused faster than fp32 (the hot path is still fp32 4×4 PCR).
- `ParaSLSTM`: `mix='diag'` remains the fused default. `mix='head'` warns as an unfused ablation (`scan_dense`). `mix='dense'` raises if `hidden_size > 8` (Jacobian oracle for tests).

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

[Unreleased]: https://github.com/bugkira/pararnn-torch/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/bugkira/pararnn-torch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bugkira/pararnn-torch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bugkira/pararnn-torch/releases/tag/v0.2.0
