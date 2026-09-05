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

## Numerics

Solver, Jacobian, and scan changes keep sequential-vs-parallel agreement tests green. Document tolerances per dtype (float32 vs bfloat16) from unit roundoff.
