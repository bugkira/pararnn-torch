"""Triton diagonal monoid scan (paper eq. 2.4).

Optional CUDA backend for ``scan_diag``. Same monoid as
``pararnn.solvers.scan``, via ``tl.associative_scan``.

``(J_r, r_r) ⊕ (J_l, r_l) = (J_r J_l, J_r r_l + r_r)``.
Tile time; pad with identity ``(1, 0)``. CUDA float16/float32, and bf16 on
compute capability ≥ 8.0; algebra in fp32. DRAM is the tensor dtype.

Hierarchy (Blelloch / PCR on this monoid):

1. Local ``tl.associative_scan`` inside tiles of ``BLOCK_T=128``.
2. Inclusive scan of up to ``CHUNK_PAD=64`` tile reductions (compile-time
   pad: ``tl.arange(0, CHUNK_PAD)`` has to fit SRAM with scan temps).
3. Superchunks of 64 tiles when ``n_chunks > 64``. Each superchunk runs
   level 2; superchunk reductions are scanned with the same helper
   (recurses if ``n_super > 64``). Exclusive super-carry uses the monoid
   ``r' = J_loc r_carry + r_loc``.

Two levels hold ``T ≤ 64 × 128 = 8192``. One extra rank holds
``T ≤ 64² × 128 = 524288``. Recursion continues past that.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors

log = logging.getLogger(__name__)

# Tile SRAM: 128 × 32 × 2 × 4 B = 32 KiB plus scan temps (~64 KiB shared).
_BLOCK_T = 128
_BLOCK_D = 32
# Leaf pad: 64 × BLOCK_D × 2 × 4 B = 16 KiB for (J, r) before scan temps.
# Danieli et al. do not pick this pad; it is the constexpr SRAM bound for
# ``_chunk_incl_kernel``. Raise BLOCK_T before lengthening the pad.
_CHUNK_PAD = 64


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
    incl_j_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb,
    stride_ajc,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ard,
    stride_ijb,
    stride_ijc,
    stride_ijd,
    stride_irb,
    stride_irc,
    stride_ird,
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
    j_s, r_s = tl.associative_scan((j, r), 0, _compose_diag)
    store_acc(
        incl_j_ptr
        + pid_b * stride_ijb
        + offs_c[:, None] * stride_ijc
        + offs_d[None, :] * stride_ijd,
        j_s,
        mask,
    )
    store_acc(
        incl_r_ptr
        + pid_b * stride_irb
        + offs_c[:, None] * stride_irc
        + offs_d[None, :] * stride_ird,
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
        j_loc_ptr + pid_b * stride_jb + offs_t[:, None] * stride_jt + offs_d[None, :] * stride_jd,
        mask,
        1.0,
    )
    r_loc = load_acc(
        r_loc_ptr + pid_b * stride_rb + offs_t[:, None] * stride_rt + offs_d[None, :] * stride_rd,
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
        out_ptr + pid_b * stride_ob + offs_t[:, None] * stride_ot + offs_d[None, :] * stride_od,
        out,
        mask,
    )


def _incl_pad64(agg_j: Tensor, agg_r: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive scan of ``n_chunks ≤ CHUNK_PAD`` tile aggregates."""
    j = agg_j.contiguous()
    r = agg_r.contiguous()
    batch, n_chunks, d_h = r.shape
    if n_chunks > _CHUNK_PAD:
        raise ValueError(f"leaf chunk scan n_chunks={n_chunks} exceeds CHUNK_PAD={_CHUNK_PAD}")
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    incl_j = j.new_empty(batch, n_chunks, d_h)
    incl_r = r.new_empty(batch, n_chunks, d_h)
    _chunk_incl_kernel[(batch, n_dtiles)](
        j,
        r,
        incl_j,
        incl_r,
        n_chunks,
        d_h,
        *j.stride(),
        *r.stride(),
        *incl_j.stride(),
        *incl_r.stride(),
        CHUNK_PAD=_CHUNK_PAD,
        BLOCK_D=_BLOCK_D,
    )
    return incl_j, incl_r


def scan_chunk_aggregates(agg_j: Tensor, agg_r: Tensor) -> Tensor:
    """Inclusive scan of diagonal tile aggregates along dim=1.

    ``agg_j`` / ``agg_r``: ``(batch, n_chunks, d_h)``. Returns ``incl_r``
    of the same shape (the r-component of the inclusive prefix).
    """
    if agg_j.shape != agg_r.shape or agg_j.dim() != 3:
        raise ValueError("agg_j and agg_r must be (batch, n_chunks, d_h)")
    batch, n_chunks, d_h = agg_r.shape
    if n_chunks <= _CHUNK_PAD:
        _, incl_r = _incl_pad64(agg_j, agg_r)
        return incl_r

    pad = _CHUNK_PAD
    n_super = (n_chunks + pad - 1) // pad
    incl_j = agg_j.new_empty(batch, n_chunks, d_h)
    incl_r = agg_r.new_empty(batch, n_chunks, d_h)
    super_j = agg_j.new_empty(batch, n_super, d_h)
    super_r = agg_r.new_empty(batch, n_super, d_h)
    for s in range(n_super):
        t0 = s * pad
        t1 = min(t0 + pad, n_chunks)
        loc_j, loc_r = _incl_pad64(agg_j[:, t0:t1], agg_r[:, t0:t1])
        incl_j[:, t0:t1] = loc_j
        incl_r[:, t0:t1] = loc_r
        super_j[:, s] = loc_j[:, -1]
        super_r[:, s] = loc_r[:, -1]
    super_incl_r = scan_chunk_aggregates(super_j, super_r)
    for s in range(1, n_super):
        t0 = s * pad
        t1 = min(t0 + pad, n_chunks)
        carry = super_incl_r[:, s - 1].float().unsqueeze(1)
        loc_j = incl_j[:, t0:t1].float()
        loc_r = incl_r[:, t0:t1].float()
        incl_r[:, t0:t1] = (loc_j * carry + loc_r).to(dtype=incl_r.dtype)
    log.debug(
        "scan_chunk_aggregates_l3",
        extra={
            "batch": batch,
            "n_chunks": n_chunks,
            "n_super": n_super,
            "d_h": d_h,
        },
    )
    return incl_r


def scan_diag_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """Same contract as ``scan_diag``: ``δ_t = jac_t * δ_{t-1} + residual_t``."""
    if jac.shape != residual.shape or jac.dim() != 3:
        raise ValueError("jac and residual must be (batch, time, d)")
    validate_cuda_tensors(jac, residual, name="scan_diag_triton")
    batch, time, d_h = residual.shape
    if time <= 1:
        return residual.clone()

    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
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

    incl_r = scan_chunk_aggregates(agg_j, agg_r)
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
