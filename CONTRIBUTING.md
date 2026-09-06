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

## Tests and lint

```bash
uv run ruff check
uv run ruff format --check
uv run pytest -q -m "not cuda"
uv run pytest -q -m cuda          # fused Triton; needs a GPU
```

## Numerics

Solver, Jacobian, and scan changes keep sequential-vs-parallel agreement tests
green. Document tolerances per dtype (float32 vs bfloat16) from unit roundoff.

## Changelog and PRs

User-visible changes go under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md).
Use the PR template checklist. Lab notes under `docs/internal/` stay local
(gitignored). Apple trees under `third_party/` stay local reference only.

Before a PyPI / tag cut: `bash scripts/check_wheel.sh` (build + twine +
clean-venv import outside the repo).

## Scripts

Benches and training entrypoints: [`scripts/README.md`](scripts/README.md).
Keep `examples/` as short onboarding scripts.
