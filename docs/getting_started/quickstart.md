# Quickstart

**Train nonlinear RNNs in parallel over \(T\); decode one step at a time.**

Package [`pararnn-torch`](https://pypi.org/project/pararnn-torch/) · import `pararnn` ·
paper [ParaRNN (arXiv:2510.21450)](https://arxiv.org/abs/2510.21450).

## Install

```bash
pip install pararnn-torch
# or
uv add pararnn-torch
```

Fused Triton: Linux + NVIDIA CUDA. Elsewhere `NewtonConfig(scan_backend="auto")`
selects the eager Newton+scan path.

## Mental model

1. A **cell** defines \(h_t = f(h_{t-1}, x_t)\) and a Jacobian structure.
2. **Train:** a few Newton iterations + parallel scan over \(T\) (paper Alg. 1).
3. **Decode:** sequential `step`; CUDA \(T{=}1\) uses `decode_step`.

```python
import torch
from pararnn import ParaGRU, ParaRNN, verify_agreement

m = ParaRNN(ParaGRU(32, 32))
x = torch.randn(2, 64, 32)
y = m(x)                          # .train() → Newton; .eval() → sequential
print(verify_agreement(m, x).to_dict())
```

## Next

| Goal | Page |
|------|------|
| Pick a cell | [Cell catalog](../cells/index.md) |
| Swap into a stack | [Adoption](adoption.md) |
| Shapes / `cu_seqlens` | [Shapes & layout](shapes_layout.md) |
| Agreement τ | [Numerics contract](../core/numerics_contract.md) |
| Serve / decode | [Inference](../systems/inference.md) |
