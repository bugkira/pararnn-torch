"""fp16 DRAM + fp32 Newton accumulators (Turing). Not bf16.

Paper App. B cell plots are float32; LM training on A100 used bf16 weights.
This box is Turing: fp16 tensor cores exist, bf16 TC do not. Scan/cell math
runs in fp32 inside the kernel; activations and J tiles stay fp16 in DRAM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_ALLOWED = (torch.float16, torch.float32)


def check_cuda_real(*tensors: Tensor, name: str) -> bool:
    """Validate CUDA fp16/fp32. Return True if the storage dtype is fp16."""
    if not tensors:
        raise ValueError(f"{name}: no tensors")
    dtype = tensors[0].dtype
    for t in tensors:
        if t.dtype != dtype:
            raise TypeError(f"{name}: dtype mismatch {t.dtype} vs {dtype}")
        if not t.is_cuda:
            raise RuntimeError(f"{name} requires CUDA")
    if dtype is torch.bfloat16:
        raise TypeError(
            f"{name}: bfloat16 is not used on Turing (no bf16 tensor cores). "
            "Use float16 (fp32 accumulators inside the kernel)."
        )
    if dtype not in _ALLOWED:
        raise TypeError(f"{name} supports float16/float32, got {dtype}")
    return dtype is torch.float16


@triton.jit
def load_acc(ptr, mask, other, FP16: tl.constexpr):
    """Load fp16/fp32, return fp32 for the Newton/scan algebra."""
    x = tl.load(ptr, mask=mask, other=other)
    if FP16:
        x = x.to(tl.float32)
    return x


@triton.jit
def store_acc(ptr, val, mask, FP16: tl.constexpr):
    """Store an fp32 accumulator; downcast when DRAM is fp16."""
    if FP16:
        tl.store(ptr, val.to(tl.float16), mask=mask)
    else:
        tl.store(ptr, val, mask=mask)
