"""Dense ``d×d`` Newton scan on CUDA (head GRU / head sLSTM / DEER).

``backend='triton'``: one program per batch row walks time. The matvec
``J δ`` is tiled in ``d`` so ``d_head`` may be 8, 64, or 256+ (each program
holds a ``BLOCK_D×BLOCK_D`` tile, not the full Jacobian). Ragged
``cu_seqlens`` uses the eager segmented path in ``scan.py``.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels.precision import load_acc, store_acc

log = logging.getLogger(__name__)

# Tile for J[i,j] loads. Fixed 32 fits common GPUs for any d_head.
_BLOCK_D = 32


@triton.jit
def _dense_row_scan_kernel(
    jac_ptr,
    res_ptr,
    out_ptr,
    time,
    d,
    stride_jb,
    stride_jt,
    stride_jrow,
    stride_jcol,
    stride_rb,
    stride_rt,
    stride_rd,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """Inclusive dense scan: ``δ_t = J_t δ_{t-1} + r_t``, tiled over ``d``."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)

    for t in range(0, time):
        base_j = jac_ptr + pid * stride_jb + t * stride_jt
        base_r = res_ptr + pid * stride_rb + t * stride_rt
        base_o = out_ptr + pid * stride_ob + t * stride_ot
        for i0 in range(0, d, BLOCK_D):
            i = i0 + offs
            mask_i = i < d
            acc = load_acc(base_r + i * stride_rd, mask_i, 0.0)
            for j0 in range(0, d, BLOCK_D):
                j = j0 + offs
                mask_j = j < d
                mask_ij = mask_i[:, None] & mask_j[None, :]
                jtile = load_acc(
                    base_j + i[:, None] * stride_jrow + j[None, :] * stride_jcol,
                    mask_ij,
                    0.0,
                )
                if t == 0:
                    dprev = tl.zeros((BLOCK_D,), dtype=tl.float32)
                else:
                    dprev = load_acc(
                        out_ptr + pid * stride_ob + (t - 1) * stride_ot + j * stride_od,
                        mask_j,
                        0.0,
                    )
                acc += tl.sum(jtile * dprev[None, :], axis=1)
            store_acc(base_o + i * stride_od, acc, mask_i)


@triton.jit
def _dense_row_reverse_kernel(
    jac_ptr,
    part_ptr,
    out_ptr,
    time,
    d,
    stride_jb,
    stride_jt,
    stride_jrow,
    stride_jcol,
    stride_pb,
    stride_pt,
    stride_pd,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """Eq. 2.6 reverse: ``μ_t = J_{t+1}^T μ_{t+1} + g_t``, tiled over ``d``."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)

    for t_rev in range(0, time):
        t = time - 1 - t_rev
        for i0 in range(0, d, BLOCK_D):
            i = i0 + offs
            mask_i = i < d
            acc = load_acc(
                part_ptr + pid * stride_pb + t * stride_pt + i * stride_pd,
                mask_i,
                0.0,
            )
            if t_rev > 0:
                # J_{t+1}^T @ μ_{t+1}: rows of J^T are columns of J_{t+1}
                base_j = jac_ptr + pid * stride_jb + (t + 1) * stride_jt
                for j0 in range(0, d, BLOCK_D):
                    j = j0 + offs
                    mask_j = j < d
                    mask_ij = mask_i[:, None] & mask_j[None, :]
                    # J[out=j, in=i] → (J^T μ)_i += J[j,i] * μ_j
                    jtile = load_acc(
                        base_j + j[:, None] * stride_jrow + i[None, :] * stride_jcol,
                        mask_ij,
                        0.0,
                    )
                    mu_next = load_acc(
                        out_ptr + pid * stride_ob + (t + 1) * stride_ot + j * stride_od,
                        mask_j,
                        0.0,
                    )
                    acc += tl.sum(jtile * mu_next[:, None], axis=0)
            store_acc(
                out_ptr + pid * stride_ob + t * stride_ot + i * stride_od,
                acc,
                mask_i,
            )


def _reverse_dense_row_triton(jac: Tensor, partial: Tensor) -> Tensor:
    """Tiled reverse dense scan (fp32 algebra, any ``d``)."""
    batch, time, d, _ = jac.shape
    dt = jac.dtype
    narrow = dt in (torch.float16, torch.bfloat16)
    jac_w = jac.float().contiguous() if narrow else jac.contiguous()
    part_w = partial.float().contiguous() if narrow else partial.contiguous()
    out = torch.empty_like(part_w)
    _dense_row_reverse_kernel[(batch,)](
        jac_w,
        part_w,
        out,
        time,
        d,
        *jac_w.stride(),
        *part_w.stride(),
        *out.stride(),
        BLOCK_D=_BLOCK_D,
    )
    return out.to(dtype=dt) if narrow else out


def _scan_dense_row_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """One-launch-per-batch dense inclusive scan (fp32 algebra, any ``d``)."""
    batch, time, d, _ = jac.shape
    dt = jac.dtype
    narrow = dt in (torch.float16, torch.bfloat16)
    jac_w = jac.float().contiguous() if narrow else jac.contiguous()
    res_w = residual.float().contiguous() if narrow else residual.contiguous()
    out = torch.empty_like(res_w)
    _dense_row_scan_kernel[(batch,)](
        jac_w,
        res_w,
        out,
        time,
        d,
        *jac_w.stride(),
        *res_w.stride(),
        *out.stride(),
        BLOCK_D=_BLOCK_D,
    )
    return out.to(dtype=dt) if narrow else out


def _reverse_dense_triton_impl(
    jac: Tensor,
    partial: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Validate shapes and run the dense reverse scan (CUDA or eager pack).

    Parameters
    ----------
    jac : Tensor
        Per-step Jacobians. Tensor of shape ``(B, T, d, d)``.
    partial : Tensor
        Incoming reverse partial. Tensor of shape ``(B, T, d)``.
    cu_seqlens : Tensor or None, default=None
        When set, delegates to the eager segmented reverse.

    Returns
    -------
    mu : Tensor
        Reverse adjoint. Tensor of shape ``(B, T, d)``.
    """
    if jac.dim() != 4 or partial.dim() != 3:
        raise ValueError(
            f"reverse_scan_dense_triton needs jac (B,T,d,d) partial (B,T,d); "
            f"got {tuple(jac.shape)} {tuple(partial.shape)}"
        )
    if jac.shape[-1] != partial.shape[-1] or jac.shape[-2] != partial.shape[-1]:
        raise ValueError(
            f"reverse_scan_dense_triton shape mismatch jac {tuple(jac.shape)} "
            f"partial {tuple(partial.shape)}"
        )
    if not jac.is_cuda or not partial.is_cuda:
        raise RuntimeError("reverse_scan_dense_triton requires CUDA")
    if jac.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"reverse_scan_dense_triton supports fp16/fp32/bf16, got {jac.dtype}")
    if cu_seqlens is not None:
        from pararnn.solvers.scan import reverse_scan_dense

        return reverse_scan_dense(jac, partial, backend="eager", cu_seqlens=cu_seqlens)
    out = _reverse_dense_row_triton(jac, partial)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "reverse_scan_dense_triton",
            extra={
                "batch": jac.shape[0],
                "seq_len": jac.shape[1],
                "d": jac.shape[-1],
                "dtype": str(jac.dtype),
            },
        )
    return out


def _scan_dense_triton_impl(
    jac: Tensor,
    residual: Tensor,
    *,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Validate shapes and run the dense inclusive scan (CUDA or eager pack).

    Parameters
    ----------
    jac : Tensor
        Per-step Jacobians. Tensor of shape ``(B, T, d, d)``.
    residual : Tensor
        Newton residual. Tensor of shape ``(B, T, d)``.
    cu_seqlens : Tensor or None, default=None
        When set, delegates to the eager segmented scan.

    Returns
    -------
    delta : Tensor
        Inclusive scan result. Tensor of shape ``(B, T, d)``.
    """
    if jac.dim() != 4 or residual.dim() != 3:
        raise ValueError(
            f"scan_dense_triton needs jac (B,T,d,d) residual (B,T,d); "
            f"got {tuple(jac.shape)} {tuple(residual.shape)}"
        )
    if jac.shape[-1] != residual.shape[-1] or jac.shape[-2] != residual.shape[-1]:
        raise ValueError(
            f"scan_dense_triton shape mismatch jac {tuple(jac.shape)} residual {tuple(residual.shape)}"
        )
    if not jac.is_cuda or not residual.is_cuda:
        raise RuntimeError("scan_dense_triton requires CUDA")
    if jac.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise TypeError(f"scan_dense_triton supports fp16/fp32/bf16, got {jac.dtype}")
    if cu_seqlens is not None:
        from pararnn.solvers.scan import _compose_dense, _fill_ident_dense, _scan_acc

        return _scan_acc(jac, residual, _compose_dense, _fill_ident_dense, cu_seqlens=cu_seqlens)
    out = _scan_dense_row_triton(jac, residual)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "scan_dense_triton",
            extra={
                "batch": jac.shape[0],
                "seq_len": jac.shape[1],
                "d": jac.shape[-1],
                "dtype": str(jac.dtype),
            },
        )
    return out
