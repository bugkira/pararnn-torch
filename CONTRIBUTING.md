# Contributing

## Setup

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

`uv run pytest -m cuda` runs the fused Triton path and needs a GPU.

## Apple reference

This library reimplements [Danieli et al.](https://arxiv.org/abs/2510.21450). [`apple/ml-pararnn`](https://github.com/apple/ml-pararnn) is separately licensed (custom, not MIT).

## Numerics

Solver, Jacobian, and scan changes keep sequential-vs-parallel agreement tests green. Document tolerances per dtype (float32 vs bfloat16) from unit roundoff.
