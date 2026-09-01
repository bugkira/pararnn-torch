"""Triton diagonal monoid scan (paper eq. 2.4).

Optional CUDA backend for ``scan_diag``. Eager PyTorch remains the default.
Not Apple's PCR: same monoid as ``pararnn.solvers.scan``, via
``tl.associative_scan`` (Triton 3.6, bundled with this PyTorch).

``(J_r, r_r) ⊕ (J_l, r_l) = (J_r J_l, J_r r_l + r_r)``.
Tile time; pad with identity ``(1, 0)``. CUDA float16/float32, and bf16 on
compute capability ≥ 8.0; algebra in fp32. DRAM is the tensor dtype.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors

log = logging.getLogger(__name__)

# 2080 Ti default shared mem ~64 KiB. 128 × 32 × 2 × 4 B = 32 KiB plus scan temps.
_BLOCK_T = 128
_BLOCK_D = 32
_CHUNK_PAD = 64  # 64 * 128 = 8192. Raise BLOCK_T before lengthening the pad.


@triton.jit
def _compose_diag(j_left, r_left, j_right, r_right):
    """combine(earlier, later) along time — eq. 2.4."""
    return j_right * j_left, j_right * r_left + r_right


@triton.jit
def _local_scan_kernel(
    j_ptr,
    r_ptr,
    j_out_ptr,
    r_out_ptr,
    agg_j_ptr,
    agg_r_ptr,
    time,
    d_h,
    stride_jb,
    stride_jt,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rd,
    stride_ojb,
    stride_ojt,
    stride_ojd,
    stride_orb,
    stride_ort,
    stride_ord,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    j = load_acc(
        j_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    r = load_acc(
        r_ptr + pid_b * stride_rb + offs_t[:, None] * stride_rt + offs_d[None, :] * stride_rd,
        mask,
        0.0,
    )
    j_s, r_s = tl.associative_scan((j, r), 0, _compose_diag)
    store_acc(
        j_out_ptr
        + pid_b * stride_ojb
        + offs_t[:, None] * stride_ojt
        + offs_d[None, :] * stride_ojd,
        j_s,
        mask,
    )
    store_acc(
        r_out_ptr
        + pid_b * stride_orb
        + offs_t[:, None] * stride_ort
        + offs_d[None, :] * stride_ord,
        r_s,
        mask,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    agg_j = tl.sum(tl.where(last, j_s, 0.0), axis=0)
    agg_r = tl.sum(tl.where(last, r_s, 0.0), axis=0)
    dmask = offs_d < d_h
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + offs_d * stride_ajd,
        agg_j,
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + offs_d * stride_ard,
        agg_r,
        dmask,
    )


@triton.jit
def _chunk_incl_kernel(
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j = load_acc(
        agg_j_ptr
        + pid_b * stride_ajb
        + offs_c[:, None] * stride_ajc
        + offs_d[None, :] * stride_ajd,
        mask,
        1.0,
    )
    r = load_acc(
        agg_r_ptr
        + pid_b * stride_arb
        + offs_c[:, None] * stride_arc
        + offs_d[None, :] * stride_ard,
        mask,
        0.0,
    )
    _, r_s = tl.associative_scan((j, r), 0, _compose_diag)
    store_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + offs_c[:, None] * stride_ic
        + offs_d[None, :] * stride_id,
        r_s,
        mask,
    )


@triton.jit
def _apply_carry_kernel(
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    out_ptr,
    time,
    d_h,
    stride_jb,
    stride_jt,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rd,
    stride_ib,
    stride_ic,
    stride_id,
    stride_ob,
    stride_ot,
    stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    j_loc = load_acc(
        j_loc_ptr
        + pid_b * stride_jb
        + offs_t[:, None] * stride_jt
        + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    r_loc = load_acc(
        r_loc_ptr
        + pid_b * stride_rb
        + offs_t[:, None] * stride_rt
        + offs_d[None, :] * stride_rd,
        mask,
        0.0,
    )
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    carry = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + offs_d * stride_id,
        offs_d < d_h,
        0.0,
    )
    carry = tl.where(pid_c > 0, carry, 0.0)
    out = j_loc * carry[None, :] + r_loc
    store_acc(
        out_ptr
        + pid_b * stride_ob
        + offs_t[:, None] * stride_ot
        + offs_d[None, :] * stride_od,
        out,
        mask,
    )


def scan_diag_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """Same contract as ``scan_diag``: ``δ_t = jac_t * δ_{t-1} + residual_t``."""
    if jac.shape != residual.shape or jac.dim() != 3:
        raise ValueError("jac and residual must be (batch, time, d)")
    validate_cuda_tensors(jac, residual, name="scan_diag_triton")
    batch, time, d_h = residual.shape
    if time <= 1:
        return residual.clone()

    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD}. "
            "Increase BLOCK_T rather than copying a longer Apple kernel."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D

    j_loc = torch.empty_like(jac)
    r_loc = torch.empty_like(residual)
    agg_j = jac.new_empty(batch, n_chunks, d_h)
    agg_r = residual.new_empty(batch, n_chunks, d_h)

    _local_scan_kernel[(batch, n_chunks, n_dtiles)](
        jac,
        residual,
        j_loc,
        r_loc,
        agg_j,
        agg_r,
        time,
        d_h,
        jac.stride(0),
        jac.stride(1),
        jac.stride(2),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        j_loc.stride(0),
        j_loc.stride(1),
        j_loc.stride(2),
        r_loc.stride(0),
        r_loc.stride(1),
        r_loc.stride(2),
        agg_j.stride(0),
        agg_j.stride(1),
        agg_j.stride(2),
        agg_r.stride(0),
        agg_r.stride(1),
        agg_r.stride(2),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )

    if n_chunks == 1:
        log.debug(
            "scan_diag_triton",
            extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": 1},
        )
        return r_loc

    incl_r = residual.new_empty(batch, n_chunks, d_h)
    _chunk_incl_kernel[(batch, n_dtiles)](
        agg_j,
        agg_r,
        incl_r,
        n_chunks,
        d_h,
        agg_j.stride(0),
        agg_j.stride(1),
        agg_j.stride(2),
        agg_r.stride(0),
        agg_r.stride(1),
        agg_r.stride(2),
        incl_r.stride(0),
        incl_r.stride(1),
        incl_r.stride(2),
        CHUNK_PAD=_CHUNK_PAD,
        BLOCK_D=_BLOCK_D,
    )
    out = torch.empty_like(residual)
    _apply_carry_kernel[(batch, n_chunks, n_dtiles)](
        j_loc,
        r_loc,
        incl_r,
        out,
        time,
        d_h,
        j_loc.stride(0),
        j_loc.stride(1),
        j_loc.stride(2),
        r_loc.stride(0),
        r_loc.stride(1),
        r_loc.stride(2),
        incl_r.stride(0),
        incl_r.stride(1),
        incl_r.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
    )
    log.debug(
        "scan_diag_triton",
        extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": n_chunks},
    )
    return out
