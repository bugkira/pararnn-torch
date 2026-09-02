# Contributing

## Setup

This project uses [uv](https://docs.astral.sh/uv/) exclusively — no pip/venv/conda.

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
```

## Tests and lint

```bash
uv run pytest
uv run ruff check
```

CUDA-marked tests (`-m cuda`) exercise fused/Triton paths and require a GPU. A green CPU run is necessary but not sufficient: run `uv run pytest -m cuda` on hardware with CUDA before merging kernel or solver changes.

## Benchmarks and GPU pinning

Any command that records timings or memory must set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The CUDA runtime defaults to `FASTEST_FIRST`, so device index 0 in PyTorch may not match `nvidia-smi` index 0.

## Apple reference code

`third_party/ml-pararnn` is Apple's read-only reference ([custom license](https://github.com/apple/ml-pararnn)). Clone it per [`third_party/README.md`](third_party/README.md) for comparison only. Do not copy, translate, or paste Apple code into `src/`.

## Numerics

Changes to solvers, Jacobians, or scan kernels must keep sequential-vs-parallel agreement tests green. Document tolerances per dtype (float32 vs bfloat16). Justify limits from unit roundoff — do not loosen tolerances until tests pass.
