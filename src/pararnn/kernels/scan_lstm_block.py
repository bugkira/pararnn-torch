"""Triton 2×2 block-diagonal monoid scan (paper eq. 2.4 / 3.2b).

Optional CUDA backend for ``scan_block2``. Same monoid as eager four-mul
``_mm2``/``_mv2``. CUDA float16/float32, and bf16 on compute capability
≥ 8.0; algebra in fp32. DRAM is the tensor dtype.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import _load_state as _load_r
from pararnn.kernels._fused_common import _store_state as _store_r
from pararnn.kernels._scan_common import _load_j, _store_j, run_block_scan_triton
from pararnn.kernels.precision import load_acc, store_acc

log = logging.getLogger(__name__)

# 6 scan lanes × 64 × 16 × 4 B = 24 KiB before scan temps (~64 KiB shared).
_BLOCK_T = 64
_BLOCK_D = 16
_CHUNK_PAD = 64  # leaf: 64 × 64 = 4096. Past that, eager Blelloch on aggregates.


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
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    j00 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j01 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 3, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_r(r_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    r1 = _load_r(r_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    s00, s01, s10, s11, u0, u1 = tl.associative_scan(
        (j00, j01, j10, j11, r0, r1), 0, _compose_block2
    )
    _store_j(
        j_out_ptr,
        s00,
        pid_b,
        offs_t,
        offs_d,
        0,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s01,
        pid_b,
        offs_t,
        offs_d,
        1,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s10,
        pid_b,
        offs_t,
        offs_d,
        2,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s11,
        pid_b,
        offs_t,
        offs_d,
        3,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_r(
        r_out_ptr,
        u0,
        pid_b,
        offs_t,
        offs_d,
        0,
        mask,
        stride_orb,
        stride_ort,
        stride_ors,
        stride_ord,
    )
    _store_r(
        r_out_ptr,
        u1,
        pid_b,
        offs_t,
        offs_d,
        1,
        mask,
        stride_orb,
        stride_ort,
        stride_ors,
        stride_ord,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    dmask = offs_d < d_h
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 0 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s00, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 1 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s01, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 2 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s10, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_j_ptr + pid_b * stride_ajb + pid_c * stride_ajc + 3 * stride_ajk + offs_d * stride_ajd,
        tl.sum(tl.where(last, s11, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 0 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u0, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 1 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u1, 0.0), axis=0),
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
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    a00 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        0,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    a01 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        1,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    a10 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        2,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    a11 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        3,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    u0 = _load_r(
        agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u1 = _load_r(
        agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    _, _, _, _, s0, s1 = tl.associative_scan((a00, a01, a10, a11, u0, u1), 0, _compose_block2)
    _store_r(
        incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_r(
        incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id
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
    j00 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j01 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 3, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r1 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    c0 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 0 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
    )
    c1 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 1 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
    )
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    out0 = j00 * c0[None, :] + j01 * c1[None, :] + r0
    out1 = j10 * c0[None, :] + j11 * c1[None, :] + r1
    _store_r(
        out_ptr, out0, pid_b, offs_t, offs_d, 0, mask, stride_ob, stride_ot, stride_os, stride_od
    )
    _store_r(
        out_ptr, out1, pid_b, offs_t, offs_d, 1, mask, stride_ob, stride_ot, stride_os, stride_od
    )


def _scan_block2_triton_impl(jac: Tensor, residual: Tensor) -> Tensor:
    """Same contract as ``scan_block2``. ``jac`` is ``(B, T, 2, 2, d)``.

    Public entry is ``pararnn::scan_block2`` in ``custom_ops``.
    """
    return run_block_scan_triton(
        jac,
        residual,
        n_state=2,
        local_kernel=_local_scan_kernel,
        chunk_incl_kernel=_chunk_incl_kernel,
        apply_carry_kernel=_apply_carry_kernel,
        block_t=_BLOCK_T,
        block_d=_BLOCK_D,
        chunk_pad=_CHUNK_PAD,
        name="scan_block2_triton",
        logger=log,
    )
