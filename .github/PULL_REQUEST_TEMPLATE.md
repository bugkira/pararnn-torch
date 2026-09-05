## Summary
<!-- What does this PR change, and why? -->

## Checklist
- [ ] `uv run ruff check && uv run ruff format --check`
- [ ] `uv run pytest -q -m "not cuda"` (and CUDA tests if kernels / solvers change)
- [ ] Sequential ↔ parallel agreement tolerances documented if numerics moved
- [ ] `CHANGELOG.md` `[Unreleased]` updated for user-visible changes
