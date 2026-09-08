# pararnn-torch documentation

**Train nonlinear RNNs in parallel over the sequence; decode one step at a time.**

Package [`pararnn-torch`](https://pypi.org/project/pararnn-torch/) · import `pararnn` · paper [ParaRNN (arXiv:2510.21450)](https://arxiv.org/abs/2510.21450).

## Start here

| Section | Purpose |
|---------|---------|
| [Quickstart](getting_started/quickstart.md) | Install, mental model, first `verify_agreement` |
| [Cell catalog](cells/index.md) | Zoo by state / Jacobian (Classic gated · Normalized · Matrix · Continuous) |
| [Newton + scan](core/newton_scan.md) | Parallel recurrence foundation |
| [Numerics contract](core/numerics_contract.md) | Agreement τ vs Newton residual |
| [Inference](systems/inference.md) | Decode / serve |

## Install

```bash
pip install pararnn-torch
# or
uv add pararnn-torch
```

Fused Triton kernels: Linux + NVIDIA CUDA. Elsewhere `NewtonConfig(scan_backend="auto")` selects the eager Newton+scan path.

## Doc map

1. **Getting Started** — quickstart, adoption, shapes  
2. **Core Architecture** — Newton, Jacobians, numerics  
3. **Cell Catalog** — Diff + YAML per cell, grouped by family  
4. **Systems & Scaling** — inference, vLLM, DDP, compile, OOM  
5. **Audit & Verification** — oracles, xLSTM notes, repo layout  

Architecture pages stay at the algorithm level. Kernel / Triton notes live in the repo and maintainer lab.
