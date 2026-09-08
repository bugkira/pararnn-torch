# pararnn-torch documentation

**Train nonlinear RNNs in parallel over the sequence; decode one step at a time.**

Package [`pararnn-torch`](https://pypi.org/project/pararnn-torch/) · import `pararnn` · paper [ParaRNN (arXiv:2510.21450)](https://arxiv.org/abs/2510.21450).

## Start here

| Page | Purpose |
|------|---------|
| [Architecture fidelity](architecture/index.md) | Which cells match which paper equations, and where we diverge |
| [Spec template](architecture/TEMPLATE.md) | Contract for a cell write-up |
| [Cell catalog](cells.md) | Call-site snippets |
| [Numerics contract](numerics-contract.md) | Agreement τ vs Newton residual |

## Install

```bash
pip install pararnn-torch
# or
uv add pararnn-torch
```

Fused Triton kernels: Linux + NVIDIA CUDA. Elsewhere `NewtonConfig(scan_backend="auto")` selects the eager Newton+scan path.

## Quick mental model

1. A **cell** defines \(h_t = f(h_{t-1}, x_t)\) and a Jacobian structure.
2. **Train:** a few Newton iterations + parallel scan over \(T\) (paper Alg. 1).
3. **Decode:** sequential `step`; CUDA \(T{=}1\) uses `decode_step`.

Architecture pages stay at the algorithm level. Kernel / Triton notes live in the repo and maintainer lab.
