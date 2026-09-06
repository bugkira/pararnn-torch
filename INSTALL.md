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

| Path | Requirement |
|---|---|
| Fused Triton (bf16 / fused Newton) | CUDA compute capability ≥ 8.0 |
| Eager / Triton scan fallback | `NewtonConfig(scan_backend="auto")` |
| CPU | sequential + eager Newton; no fused Triton |

Place modules with `.to(device)` like any `nn.Module`.

## Troubleshooting

- **Import / Triton dialect** — Linux pin is `triton>=3.6,<3.7` matching Torch
  2.11 cu128. Mismatch → reinstall from the same index as torch.
- **CC below 8.0** — fused kernels skip; set `scan_backend="auto"` and keep
  `max_iters` modest. See [`FAQs.md`](FAQs.md).
- **Release wheels** — Trusted Publishing:
  [`.github/workflows/release.yml`](.github/workflows/release.yml).
