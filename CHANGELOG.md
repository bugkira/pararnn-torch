# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions use [SemVer](https://semver.org/).

## [Unreleased]

### Added

- `ParaSLSTM` with 4×4 Newton scan (channelwise block Jacobian per feature).
- Analytic sLSTM Jacobian; Newton auto path selects it.
- Triton 4×4 scan and fused diagonal-mix sLSTM kernel.
- Zero-hidden Newton init for sLSTM (running max-plus `m` / linear `n` from the `h=0` unroll); App. A `f(0, x_t)` alone needs many more Newton steps at long `T`.
- xLSTM head-block mixing (`mix='head'`): dense `d_head×d_head` Jacobian per head in the scan.
- Auto Picard iteration count for sLSTM from sequence length (`P=1` for `T≤64`, else `P=3`).
- `xLSTMBlock` (pre-norm residual sLSTM layer).
- Dyck-language smoke test for parallel training with eq. 2.6 backward.

### Changed

- Library surface trimmed to cells, solvers, and `xLSTMBlock`; training smokes and MLflow helpers moved under `examples/` and `scripts/`.
- Fused/Triton bf16 gated by compute capability ≥ 8.0 on the tensor device; `scan_backend="auto"` falls back on Turing-class GPUs.
- Newton fails loud if residual stays huge after `K=3` on sLSTM.
- Packed VJP, Picard buffer reuse, and in-kernel GRU `a_*` reduction to cut VRAM.

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

[Unreleased]: https://github.com/bugkira/pararnn-torch/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/bugkira/pararnn-torch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bugkira/pararnn-torch/releases/tag/v0.2.0
