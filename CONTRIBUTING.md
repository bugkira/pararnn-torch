# Contributing

## Setup

See [`INSTALL.md`](INSTALL.md) for user vs contributor paths. Short form:

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
```

Optional local hooks (same ruff gate as CI):

```bash
uvx pre-commit install
```

## Lint and smoke

```bash
uv run ruff check
uv run ruff format --check
uv run python -c "import pararnn; print(pararnn.__version__)"
```

Full pytest (unit + numerics, including CUDA) runs in the maintainer
`pararnn-lab` tree against an editable checkout of this library.

## Numerics

Solver, Jacobian, and scan changes keep sequential-vs-parallel agreement tests
green in lab. Document tolerances per dtype (float32 vs bfloat16) from unit
roundoff.

## Changelog and PRs

User-visible changes go under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md).
Use the PR template checklist.

Before a PyPI / tag cut: `uv build` and confirm the wheel contains only the
`pararnn` package (CI release job also greps for lab paths).

## Examples

Keep `examples/` as short onboarding scripts.
