"""Triton 2×2 block-diagonal monoid scan (paper eq. 2.4 / 3.2b).

Optional CUDA backend for ``scan_block2``. Same monoid as eager four-mul
``_mm2``/``_mv2`` — not Apple's PCR. CUDA float16/float32; algebra in fp32.
Not bf16 (Turing has no bf16 tensor cores).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels.precision import check_cuda_real, load_acc, store_acc

log = logging.getLogger(__name__)

# 6 scan lanes × 64 × 16 × 4 B = 24 KiB before scan temps (2080 Ti ~64 KiB).
_BLOCK_T = 64
_BLOCK_D = 16
_CHUNK_PAD = 64  # 64 * 64 = 4096. Raise BLOCK_T before lengthening the pad.


@triton.jit
def _compose_block2(
    a00,
    a01,
    a10,
    a11,
    u0,
    u1,
    b00,
    b01,
    b10,
    b11,
    v0,
    v1,
):
    """combine(earlier=a, later=b): J_b @ J_a, J_b @ r_a + r_b."""
    o00 = b00 * a00 + b01 * a10
    o01 = b00 * a01 + b01 * a11
    o10 = b10 * a00 + b11 * a10
    o11 = b10 * a01 + b11 * a11
    w0 = b00 * u0 + b01 * u1 + v0
    w1 = b10 * u0 + b11 * u1 + v1
    return o00, o01, o10, o11, w0, w1


@triton.jit
def _load_j(ptr, pid_b, offs_t, offs_d, k, ident, mask, sb, st, sk, sd, FP16: tl.constexpr):
    return load_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        mask,
        ident,
        FP16,
    )


@triton.jit
def _load_r(ptr, pid_b, offs_t, offs_d, s, mask, sb, st, ss, sd, FP16: tl.constexpr):
    return load_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + s * ss + offs_d[None, :] * sd,
        mask,
        0.0,
        FP16,
    )


@triton.jit
def _store_j(ptr, val, pid_b, offs_t, offs_d, k, mask, sb, st, sk, sd, FP16: tl.constexpr):
    store_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        val,
        mask,
        FP16,
    )


@triton.jit
def _store_r(ptr, val, pid_b, offs_t, offs_d, s, mask, sb, st, ss, sd, FP16: tl.constexpr):
    store_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + s * ss + offs_d[None, :] * sd,
        val,
        mask,
        FP16,
    )


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
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd,
    stride_ojb,
    stride_ojt,
    stride_ojk,
    stride_ojd,
    stride_orb,
    stride_ort,
    stride_ors,
    stride_ord,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    j00 = _load_j(j_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j01 = _load_j(j_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j10 = _load_j(j_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j11 = _load_j(j_ptr, pid_b, offs_t, offs_d, 3, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    r0 = _load_r(r_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    r1 = _load_r(r_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    s00, s01, s10, s11, u0, u1 = tl.associative_scan(
        (j00, j01, j10, j11, r0, r1), 0, _compose_block2
    )
    _store_j(j_out_ptr, s00, pid_b, offs_t, offs_d, 0, mask, stride_ojb, stride_ojt, stride_ojk, stride_ojd, FP16)
    _store_j(j_out_ptr, s01, pid_b, offs_t, offs_d, 1, mask, stride_ojb, stride_ojt, stride_ojk, stride_ojd, FP16)
    _store_j(j_out_ptr, s10, pid_b, offs_t, offs_d, 2, mask, stride_ojb, stride_ojt, stride_ojk, stride_ojd, FP16)
    _store_j(j_out_ptr, s11, pid_b, offs_t, offs_d, 3, mask, stride_ojb, stride_ojt, stride_ojk, stride_ojd, FP16)
    _store_r(r_out_ptr, u0, pid_b, offs_t, offs_d, 0, mask, stride_orb, stride_ort, stride_ors, stride_ord, FP16)
    _store_r(r_out_ptr, u1, pid_b, offs_t, offs_d, 1, mask, stride_orb, stride_ort, stride_ors, stride_ord, FP16)
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    dmask = offs_d < d_h
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 0 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s00, 0.0), axis=0),
        dmask,
        FP16,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 1 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s01, 0.0), axis=0),
        dmask,
        FP16,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 2 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s10, 0.0), axis=0),
        dmask,
        FP16,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 3 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s11, 0.0), axis=0),
        dmask,
        FP16,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 0 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u0, 0.0), axis=0),
        dmask,
        FP16,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 1 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u1, 0.0), axis=0),
        dmask,
        FP16,
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
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    a00 = _load_j(agg_j_ptr, pid_b, offs_c, offs_d, 0, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    a01 = _load_j(agg_j_ptr, pid_b, offs_c, offs_d, 1, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    a10 = _load_j(agg_j_ptr, pid_b, offs_c, offs_d, 2, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    a11 = _load_j(agg_j_ptr, pid_b, offs_c, offs_d, 3, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    u0 = _load_r(agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    u1 = _load_r(agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    _, _, _, _, s0, s1 = tl.associative_scan((a00, a01, a10, a11, u0, u1), 0, _compose_block2)
    _store_r(incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)
    _store_r(incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)


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
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd,
    stride_ib,
    stride_ic,
    stride_is,
    stride_id,
    stride_ob,
    stride_ot,
    stride_os,
    stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    dmask = offs_d < d_h
    j00 = _load_j(j_loc_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j01 = _load_j(j_loc_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j10 = _load_j(j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j11 = _load_j(j_loc_ptr, pid_b, offs_t, offs_d, 3, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    r0 = _load_r(r_loc_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    r1 = _load_r(r_loc_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    c0 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 0 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
        FP16,
    )
    c1 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 1 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
        FP16,
    )
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    out0 = j00 * c0[None, :] + j01 * c1[None, :] + r0
    out1 = j10 * c0[None, :] + j11 * c1[None, :] + r1
    _store_r(out_ptr, out0, pid_b, offs_t, offs_d, 0, mask, stride_ob, stride_ot, stride_os, stride_od, FP16)
    _store_r(out_ptr, out1, pid_b, offs_t, offs_d, 1, mask, stride_ob, stride_ot, stride_os, stride_od, FP16)


def scan_block2_triton(jac: Tensor, residual: Tensor) -> Tensor:
    """Same contract as ``scan_block2``. ``jac`` is ``(B, T, 2, 2, d)``."""
    if residual.dim() != 4 or residual.shape[2] != 2:
        raise ValueError("residual must be (batch, time, 2, d)")
    if jac.shape[:2] != residual.shape[:2] or jac.shape[-1] != residual.shape[-1]:
        raise ValueError("jac/residual batch, time, d mismatch")
    if jac.shape[2:4] != (2, 2):
        raise ValueError("jac must be (batch, time, 2, 2, d)")
    fp16 = check_cuda_real(jac, residual, name="scan_block2_triton")
    jac = jac.contiguous()
    residual = residual.contiguous()
    batch, time, _, d_h = residual.shape
    if time <= 1:
        return residual.clone()

    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD}. "
            "Increase BLOCK_T rather than copying a longer Apple kernel."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    j_flat = jac.view(batch, time, 4, d_h)
    r_flat = residual
    j_loc = torch.empty_like(j_flat)
    r_loc = torch.empty_like(r_flat)
    agg_j = j_flat.new_empty(batch, n_chunks, 4, d_h)
    agg_r = r_flat.new_empty(batch, n_chunks, 2, d_h)

    _local_scan_kernel[(batch, n_chunks, n_dtiles)](
        j_flat,
        r_flat,
        j_loc,
        r_loc,
        agg_j,
        agg_r,
        time,
        d_h,
        *j_flat.stride(),
        *r_flat.stride(),
        *j_loc.stride(),
        *r_loc.stride(),
        *agg_j.stride(),
        *agg_r.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        FP16=fp16,
    )
    if n_chunks == 1:
        log.debug(
            "scan_block2_triton",
            extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": 1},
        )
        return r_loc

    incl_r = r_flat.new_empty(batch, n_chunks, 2, d_h)
    _chunk_incl_kernel[(batch, n_dtiles)](
        agg_j,
        agg_r,
        incl_r,
        n_chunks,
        d_h,
        *agg_j.stride(),
        *agg_r.stride(),
        *incl_r.stride(),
        CHUNK_PAD=_CHUNK_PAD,
        BLOCK_D=_BLOCK_D,
        FP16=fp16,
    )
    out = torch.empty_like(r_flat)
    _apply_carry_kernel[(batch, n_chunks, n_dtiles)](
        j_loc,
        r_loc,
        incl_r,
        out,
        time,
        d_h,
        *j_loc.stride(),
        *r_loc.stride(),
        *incl_r.stride(),
        *out.stride(),
        BLOCK_T=_BLOCK_T,
        BLOCK_D=_BLOCK_D,
        FP16=fp16,
    )
    log.debug(
        "scan_block2_triton",
        extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": n_chunks},
    )
    return out
