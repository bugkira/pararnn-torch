# Install

Package **`pararnn-torch`**, import **`pararnn`**. Python 3.10+.

## Users

```bash
# when published on PyPI:
pip install pararnn-torch

# bleeding edge / until first PyPI release:
pip install "pararnn-torch @ git+https://github.com/bugkira/pararnn-torch"
```

Bring your own PyTorch (CPU or CUDA). Optional extras:

| Extra | Purpose |
|---|---|
| `train` | MLflow |
| `lm` | datasets + tokenizers (BabyLM scripts) |
| `vllm` | vLLM plugin registration |
| `flashrnn` | FlashRNN comparison benches |

```bash
pip install "pararnn-torch[train,lm]"
```

## Contributors

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
uv run pytest -q -m "not cuda"
uvx pre-commit install   # optional; same ruff gate as CI
```

Dev PyTorch is pinned via `pyproject.toml` (cu128 index). See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Hardware

**Fused kernels need Linux + NVIDIA CUDA.** Lab cards include RTX 2080 Ti
(Turing) and RTX 3060 (Ampere): fp32 / fp16 fused Newton runs on both.
**bf16 fused** needs compute capability ≥ 8.0 (Ampere+ tensor cores).

On CPU, Mac, or Windows without a usable CUDA build,
`NewtonConfig(scan_backend="auto")` uses the eager PyTorch Newton+scan path —
same numerics contract, slower wall-clock.

| Path | Where it runs |
|---|---|
| Fused Newton (fp32 / fp16) | Linux + NVIDIA CUDA (Turing and newer; lab: 2080 Ti, 3060) |
| Fused Newton (bf16) | Linux + NVIDIA CC ≥ 8.0 (Ampere+) |
| Eager / Triton scan | `scan_backend="auto"` when fused is unavailable |
| CPU | sequential + eager Newton |

Place modules with `.to(device)` like any `nn.Module`.

## Troubleshooting

- **Import / Triton dialect** — Linux pin is `triton>=3.6,<3.7` matching Torch
  2.11 cu128. The first fused / Triton scan launch runs
  `require_fused_triton()`: pin + a tiny JIT smoke. A bad dialect raises
  `[ParaRNN] Triton/CUDA environment failed…` on that first launch.
  Optional early check: `from pararnn.kernels import check_triton_environment`.
  Mismatch → reinstall from the same index as torch, or
  `NewtonConfig(scan_backend="eager")`. Loads use `precision.load_acc` /
  `store_acc` (masked `tl.load`).
- **bf16 on pre-Ampere** — use fp16 or fp32; fused bf16 needs CC ≥ 8.0. See
  [`FAQs.md`](FAQs.md).
- **Release wheels** — Trusted Publishing:
  [`.github/workflows/release.yml`](.github/workflows/release.yml).
