"""Narrow DRAM (fp16/bf16) + fp32 Newton/scan accumulators.

Paper App. B cell plots are float32; LM training used bf16 weights on Ampere.
Algebra is fp32 inside Triton (`load_acc` / `store_acc`). Activations, J
tiles, and residuals stay in the tensor dtype in DRAM. ``W_x`` is a PyTorch
GEMM.

bf16 fused/Triton is **compute capability ≥ 8.0** (Ampere+ tensor cores).
CC 7.x has no bf16 TC: ``auto`` falls back; explicit ``fused`` raises.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_NARROW_DTYPES = (torch.float16, torch.bfloat16)
_BF16_MIN_MAJOR = 8  # Ampere / Ada / Hopper / Blackwell TC


def is_fused_dtype_supported(dtype: torch.dtype, device: torch.device | str | int) -> bool:
    """Whether fused/Triton scan may run this dtype on this **tensor** device.

    fp32 and fp16: any CUDA. bf16: SM 8.0+. Query the tensor's ``device``
    (multi-GPU boxes).
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    if dtype in (torch.float32, torch.float16):
        return True
    if dtype == torch.bfloat16:
        major, _minor = torch.cuda.get_device_capability(dev)
        return major >= _BF16_MIN_MAJOR
    return False


def validate_cuda_tensors(*tensors: Tensor, name: str) -> bool:
    """CUDA, one device, one fused dtype. Return True if DRAM is fp16/bf16."""
    if not tensors:
        raise ValueError(f"{name}: no tensors")
    ref = tensors[0]
    dtype = ref.dtype
    device = ref.device
    for t in tensors:
        if not t.is_cuda:
            raise RuntimeError(f"{name} requires CUDA, got {t.device}")
        if t.device != device:
            raise RuntimeError(f"{name}: tensors on different devices ({t.device} vs {device})")
        if t.dtype != dtype:
            raise TypeError(f"{name}: dtype mismatch {t.dtype} vs {dtype}")
    if not is_fused_dtype_supported(dtype, device):
        if dtype == torch.bfloat16:
            major, minor = torch.cuda.get_device_capability(device)
            raise TypeError(
                f"{name}: bfloat16 needs CUDA compute capability >= 8.0 "
                f"(Ampere+ tensor cores); got sm_{major}{minor} on {device}. "
                "Use float16 or float32; cell+scan algebra stays fp32."
            )
        raise TypeError(f"{name} supports float16/float32/bfloat16, got {dtype}")
    return dtype in _NARROW_DTYPES


@triton.jit
def load_acc(ptr, mask, other):
    """Load DRAM (fp16, bf16, or fp32) and return fp32 for Newton/scan algebra.

    ``.to(tl.float32)`` is a no-op when the pointer is already fp32.
    """
    return tl.load(ptr, mask=mask, other=other).to(tl.float32)


@triton.jit
def store_acc(ptr, val, mask):
    """Store an fp32 accumulator; cast to the pointer's element type."""
    tl.store(ptr, val.to(ptr.dtype.element_ty), mask=mask)
