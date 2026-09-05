"""Triton 4×4 block-diagonal monoid scan (sLSTM channelwise).

Optional CUDA backend for ``scan_block4``. Same monoid as eager ``_mm4``/``_mv4``.
CUDA float16/float32, and bf16 on compute capability ≥ 8.0; algebra in fp32.
DRAM is the tensor dtype.

20 scan lanes need smaller tiles than 2×2: 20 × 32 × 16 × 4 B = 40 KiB
before scan temps (~64 KiB shared). Chunk scan uses BLOCK_D=8.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import _load_state as _load_r
from pararnn.kernels._fused_common import _store_state as _store_r
from pararnn.kernels._scan_common import _load_j, _store_agg_j, _store_j, run_block_scan_triton
from pararnn.kernels.precision import load_acc, store_acc

log = logging.getLogger(__name__)

_BLOCK_T = 32
_BLOCK_D = 16
_CHUNK_D = 8
_CHUNK_PAD = 64  # leaf: 64 × 32 = 2048. Past that, eager Blelloch on aggregates.


@triton.jit
def _compose_block4(
    a00,
    a01,
    a02,
    a03,
    a10,
    a11,
    a12,
    a13,
    a20,
    a21,
    a22,
    a23,
    a30,
    a31,
    a32,
    a33,
    u0,
    u1,
    u2,
    u3,
    b00,
    b01,
    b02,
    b03,
    b10,
    b11,
    b12,
    b13,
    b20,
    b21,
    b22,
    b23,
    b30,
    b31,
    b32,
    b33,
    v0,
    v1,
    v2,
    v3,
):
    """combine(earlier=a, later=b): J_b @ J_a, J_b @ r_a + r_b."""
    o00 = b00 * a00 + b01 * a10 + b02 * a20 + b03 * a30
    o01 = b00 * a01 + b01 * a11 + b02 * a21 + b03 * a31
    o02 = b00 * a02 + b01 * a12 + b02 * a22 + b03 * a32
    o03 = b00 * a03 + b01 * a13 + b02 * a23 + b03 * a33
    o10 = b10 * a00 + b11 * a10 + b12 * a20 + b13 * a30
    o11 = b10 * a01 + b11 * a11 + b12 * a21 + b13 * a31
    o12 = b10 * a02 + b11 * a12 + b12 * a22 + b13 * a32
    o13 = b10 * a03 + b11 * a13 + b12 * a23 + b13 * a33
    o20 = b20 * a00 + b21 * a10 + b22 * a20 + b23 * a30
    o21 = b20 * a01 + b21 * a11 + b22 * a21 + b23 * a31
    o22 = b20 * a02 + b21 * a12 + b22 * a22 + b23 * a32
    o23 = b20 * a03 + b21 * a13 + b22 * a23 + b23 * a33
    o30 = b30 * a00 + b31 * a10 + b32 * a20 + b33 * a30
    o31 = b30 * a01 + b31 * a11 + b32 * a21 + b33 * a31
    o32 = b30 * a02 + b31 * a12 + b32 * a22 + b33 * a32
    o33 = b30 * a03 + b31 * a13 + b32 * a23 + b33 * a33
    w0 = b00 * u0 + b01 * u1 + b02 * u2 + b03 * u3 + v0
    w1 = b10 * u0 + b11 * u1 + b12 * u2 + b13 * u3 + v1
    w2 = b20 * u0 + b21 * u1 + b22 * u2 + b23 * u3 + v2
    w3 = b30 * u0 + b31 * u1 + b32 * u2 + b33 * u3 + v3
    return (
        o00,
        o01,
        o02,
        o03,
        o10,
        o11,
        o12,
        o13,
        o20,
        o21,
        o22,
        o23,
        o30,
        o31,
        o32,
        o33,
        w0,
        w1,
        w2,
        w3,
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
    j02 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j03 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 3, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 4, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 5, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j12 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 6, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j13 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 7, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j20 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 8, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j21 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 9, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j22 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 10, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j23 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 11, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j30 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 12, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j31 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 13, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j32 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 14, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j33 = _load_j(
        j_ptr, pid_b, offs_t, offs_d, 15, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_r(r_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    r1 = _load_r(r_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    r2 = _load_r(r_ptr, pid_b, offs_t, offs_d, 2, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    r3 = _load_r(r_ptr, pid_b, offs_t, offs_d, 3, mask, stride_rb, stride_rt, stride_rs, stride_rd)
    (
        s00,
        s01,
        s02,
        s03,
        s10,
        s11,
        s12,
        s13,
        s20,
        s21,
        s22,
        s23,
        s30,
        s31,
        s32,
        s33,
        u0,
        u1,
        u2,
        u3,
    ) = tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            r0,
            r1,
            r2,
            r3,
        ),
        0,
        _compose_block4,
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
        s02,
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
        s03,
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
    _store_j(
        j_out_ptr,
        s10,
        pid_b,
        offs_t,
        offs_d,
        4,
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
        5,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s12,
        pid_b,
        offs_t,
        offs_d,
        6,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s13,
        pid_b,
        offs_t,
        offs_d,
        7,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s20,
        pid_b,
        offs_t,
        offs_d,
        8,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s21,
        pid_b,
        offs_t,
        offs_d,
        9,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s22,
        pid_b,
        offs_t,
        offs_d,
        10,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s23,
        pid_b,
        offs_t,
        offs_d,
        11,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s30,
        pid_b,
        offs_t,
        offs_d,
        12,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s31,
        pid_b,
        offs_t,
        offs_d,
        13,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s32,
        pid_b,
        offs_t,
        offs_d,
        14,
        mask,
        stride_ojb,
        stride_ojt,
        stride_ojk,
        stride_ojd,
    )
    _store_j(
        j_out_ptr,
        s33,
        pid_b,
        offs_t,
        offs_d,
        15,
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
    _store_r(
        r_out_ptr,
        u2,
        pid_b,
        offs_t,
        offs_d,
        2,
        mask,
        stride_orb,
        stride_ort,
        stride_ors,
        stride_ord,
    )
    _store_r(
        r_out_ptr,
        u3,
        pid_b,
        offs_t,
        offs_d,
        3,
        mask,
        stride_orb,
        stride_ort,
        stride_ors,
        stride_ord,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    dmask = offs_d < d_h
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s00, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        0,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s01, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        1,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s02, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        2,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s03, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        3,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s10, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        4,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s11, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        5,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s12, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        6,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s13, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        7,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s20, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        8,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s21, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        9,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s22, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        10,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s23, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        11,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s30, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        12,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s31, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        13,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s32, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        14,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    _store_agg_j(
        agg_j_ptr,
        tl.sum(tl.where(last, s33, 0.0), axis=0),
        pid_b,
        pid_c,
        offs_d,
        15,
        dmask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
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
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 2 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u2, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + 3 * stride_ars + offs_d * stride_ard,
        tl.sum(tl.where(last, u3, 0.0), axis=0),
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
    j00 = _load_j(
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
    j01 = _load_j(
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
    j02 = _load_j(
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
    j03 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        3,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j10 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        4,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j11 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        5,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j12 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        6,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j13 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        7,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j20 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        8,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j21 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        9,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j22 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        10,
        1.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j23 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        11,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j30 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        12,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j31 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        13,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j32 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        14,
        0.0,
        mask,
        stride_ajb,
        stride_ajc,
        stride_ajk,
        stride_ajd,
    )
    j33 = _load_j(
        agg_j_ptr,
        pid_b,
        offs_c,
        offs_d,
        15,
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
    u2 = _load_r(
        agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u3 = _load_r(
        agg_r_ptr, pid_b, offs_c, offs_d, 3, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        s0,
        s1,
        s2,
        s3,
    ) = tl.associative_scan(
        (
            j00,
            j01,
            j02,
            j03,
            j10,
            j11,
            j12,
            j13,
            j20,
            j21,
            j22,
            j23,
            j30,
            j31,
            j32,
            j33,
            u0,
            u1,
            u2,
            u3,
        ),
        0,
        _compose_block4,
    )
    _store_r(
        incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_r(
        incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_r(
        incl_r_ptr, s2, pid_b, offs_c, offs_d, 2, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_r(
        incl_r_ptr, s3, pid_b, offs_c, offs_d, 3, mask, stride_ib, stride_ic, stride_is, stride_id
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
    j02 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j03 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 3, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 4, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 5, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j12 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 6, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j13 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 7, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j20 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 8, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j21 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 9, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j22 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 10, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j23 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 11, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j30 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 12, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j31 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 13, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j32 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 14, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j33 = _load_j(
        j_loc_ptr, pid_b, offs_t, offs_d, 15, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 0, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r1 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 1, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r2 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 2, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r3 = _load_r(
        r_loc_ptr, pid_b, offs_t, offs_d, 3, mask, stride_rb, stride_rt, stride_rs, stride_rd
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
    c2 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 2 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
    )
    c3 = load_acc(
        incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + 3 * stride_is + offs_d * stride_id,
        dmask,
        0.0,
    )
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    c2 = tl.where(pid_c > 0, c2, 0.0)
    c3 = tl.where(pid_c > 0, c3, 0.0)
    out0 = j00 * c0[None, :] + j01 * c1[None, :] + j02 * c2[None, :] + j03 * c3[None, :] + r0
    out1 = j10 * c0[None, :] + j11 * c1[None, :] + j12 * c2[None, :] + j13 * c3[None, :] + r1
    out2 = j20 * c0[None, :] + j21 * c1[None, :] + j22 * c2[None, :] + j23 * c3[None, :] + r2
    out3 = j30 * c0[None, :] + j31 * c1[None, :] + j32 * c2[None, :] + j33 * c3[None, :] + r3
    _store_r(
        out_ptr, out0, pid_b, offs_t, offs_d, 0, mask, stride_ob, stride_ot, stride_os, stride_od
    )
    _store_r(
        out_ptr, out1, pid_b, offs_t, offs_d, 1, mask, stride_ob, stride_ot, stride_os, stride_od
    )
    _store_r(
        out_ptr, out2, pid_b, offs_t, offs_d, 2, mask, stride_ob, stride_ot, stride_os, stride_od
    )
    _store_r(
        out_ptr, out3, pid_b, offs_t, offs_d, 3, mask, stride_ob, stride_ot, stride_os, stride_od
    )


def _scan_block4_triton_impl(jac: Tensor, residual: Tensor) -> Tensor:
    """Same contract as ``scan_block4``. ``jac`` is ``(B, T, 4, 4, d)``.

    Public entry is ``pararnn::scan_block4`` in ``custom_ops``.
    """
    return run_block_scan_triton(
        jac,
        residual,
        n_state=4,
        local_kernel=_local_scan_kernel,
        chunk_incl_kernel=_chunk_incl_kernel,
        apply_carry_kernel=_apply_carry_kernel,
        block_t=_BLOCK_T,
        block_d=_BLOCK_D,
        chunk_pad=_CHUNK_PAD,
        chunk_d=_CHUNK_D,
        name="scan_block4_triton",
        logger=log,
    )
