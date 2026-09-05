"""Fused ParaSLSTM Newton: diag mix, 4×4 J + scan (Beck sLSTM + paper Alg. 1).

``W_x(x)`` stays a cuBLAS GEMM. This kernel is diag mix, 4×4 SRAM.
CUDA float16/float32, and bf16 on compute capability ≥ 8.0; cell+scan
algebra in fp32. DRAM is the tensor dtype.

The 4×4 linearizes ``R h`` into the next gates. Picard (frozen ``R h``)
is the 1D max-plus + two ``ax+b`` scans in ``picard_slstm.py``.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl
from torch import Tensor

from pararnn.kernels._fused_common import (
    _load_h0,
    _load_state,
    _store_state,
    _tanh,
    alloc_fp32_update,
    fp32_newton_work,
    fp32_omega_add,
    fused_early_exit_hit,
    log_fused_done,
    log_fused_iter,
    mark_fused_iters_done,
    prepare_h0_block_table,
    time_tiles,
)
from pararnn.kernels._scan_common import (
    _load_j as _load_j_lane,
)
from pararnn.kernels._scan_common import (
    _store_agg_j,
    incl_block_aggregates,
)
from pararnn.kernels._scan_common import (
    _store_j as _store_j_lane,
)
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)
from pararnn.solvers.newton.config import FUSED_WINDOW_DEFAULT, FUSED_WINDOW_LENS

log = logging.getLogger(__name__)

# Same tiles as scan_block4 (20 scan lanes). 20 x 32 x 16 x 4 B = 40 KiB.
_BLOCK_T = 32
_BLOCK_D = 16
# Chunk scan: 20 × CHUNK_PAD × CHUNK_D × 4 B ≤ 64 KiB. 512 × 1 → 40 KiB.
# Two-level T = 32 × 512 = 16384. Past that, eager Blelloch on aggregates.
_CHUNK_D = 1
_CHUNK_PAD = 512


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
def _seq_scan_block4(
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
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Inclusive serial prefix along time. Ablation vs ``tl.associative_scan``."""
    a00 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a01 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a02 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a03 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a10 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a11 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a12 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a13 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a20 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a21 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a22 = tl.full((BLOCK_D,), 1.0, tl.float32)
    a23 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a30 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a31 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a32 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    a33 = tl.full((BLOCK_D,), 1.0, tl.float32)
    w0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    o00 = tl.zeros_like(j00)
    o01 = tl.zeros_like(j00)
    o02 = tl.zeros_like(j00)
    o03 = tl.zeros_like(j00)
    o10 = tl.zeros_like(j00)
    o11 = tl.zeros_like(j00)
    o12 = tl.zeros_like(j00)
    o13 = tl.zeros_like(j00)
    o20 = tl.zeros_like(j00)
    o21 = tl.zeros_like(j00)
    o22 = tl.zeros_like(j00)
    o23 = tl.zeros_like(j00)
    o30 = tl.zeros_like(j00)
    o31 = tl.zeros_like(j00)
    o32 = tl.zeros_like(j00)
    o33 = tl.zeros_like(j00)
    u0 = tl.zeros_like(r0)
    u1 = tl.zeros_like(r0)
    u2 = tl.zeros_like(r0)
    u3 = tl.zeros_like(r0)
    offs_t = tl.arange(0, BLOCK_T)
    for t in tl.range(BLOCK_T):
        sel = (offs_t == t)[:, None]
        e00 = tl.sum(tl.where(sel, j00, 0.0), 0)
        e01 = tl.sum(tl.where(sel, j01, 0.0), 0)
        e02 = tl.sum(tl.where(sel, j02, 0.0), 0)
        e03 = tl.sum(tl.where(sel, j03, 0.0), 0)
        e10 = tl.sum(tl.where(sel, j10, 0.0), 0)
        e11 = tl.sum(tl.where(sel, j11, 0.0), 0)
        e12 = tl.sum(tl.where(sel, j12, 0.0), 0)
        e13 = tl.sum(tl.where(sel, j13, 0.0), 0)
        e20 = tl.sum(tl.where(sel, j20, 0.0), 0)
        e21 = tl.sum(tl.where(sel, j21, 0.0), 0)
        e22 = tl.sum(tl.where(sel, j22, 0.0), 0)
        e23 = tl.sum(tl.where(sel, j23, 0.0), 0)
        e30 = tl.sum(tl.where(sel, j30, 0.0), 0)
        e31 = tl.sum(tl.where(sel, j31, 0.0), 0)
        e32 = tl.sum(tl.where(sel, j32, 0.0), 0)
        e33 = tl.sum(tl.where(sel, j33, 0.0), 0)
        er0 = tl.sum(tl.where(sel, r0, 0.0), 0)
        er1 = tl.sum(tl.where(sel, r1, 0.0), 0)
        er2 = tl.sum(tl.where(sel, r2, 0.0), 0)
        er3 = tl.sum(tl.where(sel, r3, 0.0), 0)
        (
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
            w0,
            w1,
            w2,
            w3,
        ) = _compose_block4(
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
            w0,
            w1,
            w2,
            w3,
            e00,
            e01,
            e02,
            e03,
            e10,
            e11,
            e12,
            e13,
            e20,
            e21,
            e22,
            e23,
            e30,
            e31,
            e32,
            e33,
            er0,
            er1,
            er2,
            er3,
        )
        o00 = tl.where(sel, a00[None, :], o00)
        o01 = tl.where(sel, a01[None, :], o01)
        o02 = tl.where(sel, a02[None, :], o02)
        o03 = tl.where(sel, a03[None, :], o03)
        o10 = tl.where(sel, a10[None, :], o10)
        o11 = tl.where(sel, a11[None, :], o11)
        o12 = tl.where(sel, a12[None, :], o12)
        o13 = tl.where(sel, a13[None, :], o13)
        o20 = tl.where(sel, a20[None, :], o20)
        o21 = tl.where(sel, a21[None, :], o21)
        o22 = tl.where(sel, a22[None, :], o22)
        o23 = tl.where(sel, a23[None, :], o23)
        o30 = tl.where(sel, a30[None, :], o30)
        o31 = tl.where(sel, a31[None, :], o31)
        o32 = tl.where(sel, a32[None, :], o32)
        o33 = tl.where(sel, a33[None, :], o33)
        u0 = tl.where(sel, w0[None, :], u0)
        u1 = tl.where(sel, w1[None, :], u1)
        u2 = tl.where(sel, w2[None, :], u2)
        u3 = tl.where(sel, w3[None, :], u3)
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
        u0,
        u1,
        u2,
        u3,
    )


@triton.jit
def _at_k(x3, k, C: tl.constexpr):
    """Slice ``x3[:, k, :]`` without integer indexing (Triton forbids it)."""
    sel = tl.arange(0, C)[None, :, None] == k
    return tl.sum(tl.where(sel, x3, 0.0), 1)


@triton.jit
def _put_k(dst3, val2, k, C: tl.constexpr):
    """Scatter ``val2`` into ``dst3[:, k, :]``."""
    sel = tl.arange(0, C)[None, :, None] == k
    return tl.where(sel, val2[:, None, :], dst3)


@triton.jit
def _thomas_pcr_scan_block4(
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
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    THOMAS_C: tl.constexpr,
):
    """Blocked 4×4 scan: sequential compose of ``THOMAS_C`` steps, then PCR.

    ``THOMAS_C`` must divide ``BLOCK_T``. Intra-group Thomas is a loop of
    length ``C`` on shape ``(T/C, D)``, not ``tl.range(BLOCK_T)``. Coarse
    inclusive scan is ``tl.associative_scan`` of ``T/C`` monoid elements.
    """
    n_coarse: tl.constexpr = BLOCK_T // THOMAS_C
    j00g = tl.reshape(j00, [n_coarse, THOMAS_C, BLOCK_D])
    j01g = tl.reshape(j01, [n_coarse, THOMAS_C, BLOCK_D])
    j02g = tl.reshape(j02, [n_coarse, THOMAS_C, BLOCK_D])
    j03g = tl.reshape(j03, [n_coarse, THOMAS_C, BLOCK_D])
    j10g = tl.reshape(j10, [n_coarse, THOMAS_C, BLOCK_D])
    j11g = tl.reshape(j11, [n_coarse, THOMAS_C, BLOCK_D])
    j12g = tl.reshape(j12, [n_coarse, THOMAS_C, BLOCK_D])
    j13g = tl.reshape(j13, [n_coarse, THOMAS_C, BLOCK_D])
    j20g = tl.reshape(j20, [n_coarse, THOMAS_C, BLOCK_D])
    j21g = tl.reshape(j21, [n_coarse, THOMAS_C, BLOCK_D])
    j22g = tl.reshape(j22, [n_coarse, THOMAS_C, BLOCK_D])
    j23g = tl.reshape(j23, [n_coarse, THOMAS_C, BLOCK_D])
    j30g = tl.reshape(j30, [n_coarse, THOMAS_C, BLOCK_D])
    j31g = tl.reshape(j31, [n_coarse, THOMAS_C, BLOCK_D])
    j32g = tl.reshape(j32, [n_coarse, THOMAS_C, BLOCK_D])
    j33g = tl.reshape(j33, [n_coarse, THOMAS_C, BLOCK_D])
    r0g = tl.reshape(r0, [n_coarse, THOMAS_C, BLOCK_D])
    r1g = tl.reshape(r1, [n_coarse, THOMAS_C, BLOCK_D])
    r2g = tl.reshape(r2, [n_coarse, THOMAS_C, BLOCK_D])
    r3g = tl.reshape(r3, [n_coarse, THOMAS_C, BLOCK_D])
    p00 = _at_k(j00g, 0, THOMAS_C)
    p01 = _at_k(j01g, 0, THOMAS_C)
    p02 = _at_k(j02g, 0, THOMAS_C)
    p03 = _at_k(j03g, 0, THOMAS_C)
    p10 = _at_k(j10g, 0, THOMAS_C)
    p11 = _at_k(j11g, 0, THOMAS_C)
    p12 = _at_k(j12g, 0, THOMAS_C)
    p13 = _at_k(j13g, 0, THOMAS_C)
    p20 = _at_k(j20g, 0, THOMAS_C)
    p21 = _at_k(j21g, 0, THOMAS_C)
    p22 = _at_k(j22g, 0, THOMAS_C)
    p23 = _at_k(j23g, 0, THOMAS_C)
    p30 = _at_k(j30g, 0, THOMAS_C)
    p31 = _at_k(j31g, 0, THOMAS_C)
    p32 = _at_k(j32g, 0, THOMAS_C)
    p33 = _at_k(j33g, 0, THOMAS_C)
    pu0 = _at_k(r0g, 0, THOMAS_C)
    pu1 = _at_k(r1g, 0, THOMAS_C)
    pu2 = _at_k(r2g, 0, THOMAS_C)
    pu3 = _at_k(r3g, 0, THOMAS_C)
    i00 = j00g * 0.0
    i01 = j01g * 0.0
    i02 = j02g * 0.0
    i03 = j03g * 0.0
    i10 = j10g * 0.0
    i11 = j11g * 0.0
    i12 = j12g * 0.0
    i13 = j13g * 0.0
    i20 = j20g * 0.0
    i21 = j21g * 0.0
    i22 = j22g * 0.0
    i23 = j23g * 0.0
    i30 = j30g * 0.0
    i31 = j31g * 0.0
    i32 = j32g * 0.0
    i33 = j33g * 0.0
    iu0 = r0g * 0.0
    iu1 = r1g * 0.0
    iu2 = r2g * 0.0
    iu3 = r3g * 0.0
    i00 = _put_k(i00, p00, 0, THOMAS_C)
    i01 = _put_k(i01, p01, 0, THOMAS_C)
    i02 = _put_k(i02, p02, 0, THOMAS_C)
    i03 = _put_k(i03, p03, 0, THOMAS_C)
    i10 = _put_k(i10, p10, 0, THOMAS_C)
    i11 = _put_k(i11, p11, 0, THOMAS_C)
    i12 = _put_k(i12, p12, 0, THOMAS_C)
    i13 = _put_k(i13, p13, 0, THOMAS_C)
    i20 = _put_k(i20, p20, 0, THOMAS_C)
    i21 = _put_k(i21, p21, 0, THOMAS_C)
    i22 = _put_k(i22, p22, 0, THOMAS_C)
    i23 = _put_k(i23, p23, 0, THOMAS_C)
    i30 = _put_k(i30, p30, 0, THOMAS_C)
    i31 = _put_k(i31, p31, 0, THOMAS_C)
    i32 = _put_k(i32, p32, 0, THOMAS_C)
    i33 = _put_k(i33, p33, 0, THOMAS_C)
    iu0 = _put_k(iu0, pu0, 0, THOMAS_C)
    iu1 = _put_k(iu1, pu1, 0, THOMAS_C)
    iu2 = _put_k(iu2, pu2, 0, THOMAS_C)
    iu3 = _put_k(iu3, pu3, 0, THOMAS_C)
    for k in tl.static_range(1, THOMAS_C):
        e00 = _at_k(j00g, k, THOMAS_C)
        e01 = _at_k(j01g, k, THOMAS_C)
        e02 = _at_k(j02g, k, THOMAS_C)
        e03 = _at_k(j03g, k, THOMAS_C)
        e10 = _at_k(j10g, k, THOMAS_C)
        e11 = _at_k(j11g, k, THOMAS_C)
        e12 = _at_k(j12g, k, THOMAS_C)
        e13 = _at_k(j13g, k, THOMAS_C)
        e20 = _at_k(j20g, k, THOMAS_C)
        e21 = _at_k(j21g, k, THOMAS_C)
        e22 = _at_k(j22g, k, THOMAS_C)
        e23 = _at_k(j23g, k, THOMAS_C)
        e30 = _at_k(j30g, k, THOMAS_C)
        e31 = _at_k(j31g, k, THOMAS_C)
        e32 = _at_k(j32g, k, THOMAS_C)
        e33 = _at_k(j33g, k, THOMAS_C)
        er0 = _at_k(r0g, k, THOMAS_C)
        er1 = _at_k(r1g, k, THOMAS_C)
        er2 = _at_k(r2g, k, THOMAS_C)
        er3 = _at_k(r3g, k, THOMAS_C)
        (
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
        ) = _compose_block4(
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
            e00,
            e01,
            e02,
            e03,
            e10,
            e11,
            e12,
            e13,
            e20,
            e21,
            e22,
            e23,
            e30,
            e31,
            e32,
            e33,
            er0,
            er1,
            er2,
            er3,
        )
        i00 = _put_k(i00, p00, k, THOMAS_C)
        i01 = _put_k(i01, p01, k, THOMAS_C)
        i02 = _put_k(i02, p02, k, THOMAS_C)
        i03 = _put_k(i03, p03, k, THOMAS_C)
        i10 = _put_k(i10, p10, k, THOMAS_C)
        i11 = _put_k(i11, p11, k, THOMAS_C)
        i12 = _put_k(i12, p12, k, THOMAS_C)
        i13 = _put_k(i13, p13, k, THOMAS_C)
        i20 = _put_k(i20, p20, k, THOMAS_C)
        i21 = _put_k(i21, p21, k, THOMAS_C)
        i22 = _put_k(i22, p22, k, THOMAS_C)
        i23 = _put_k(i23, p23, k, THOMAS_C)
        i30 = _put_k(i30, p30, k, THOMAS_C)
        i31 = _put_k(i31, p31, k, THOMAS_C)
        i32 = _put_k(i32, p32, k, THOMAS_C)
        i33 = _put_k(i33, p33, k, THOMAS_C)
        iu0 = _put_k(iu0, pu0, k, THOMAS_C)
        iu1 = _put_k(iu1, pu1, k, THOMAS_C)
        iu2 = _put_k(iu2, pu2, k, THOMAS_C)
        iu3 = _put_k(iu3, pu3, k, THOMAS_C)
    (
        g00,
        g01,
        g02,
        g03,
        g10,
        g11,
        g12,
        g13,
        g20,
        g21,
        g22,
        g23,
        g30,
        g31,
        g32,
        g33,
        gu0,
        gu1,
        gu2,
        gu3,
    ) = tl.associative_scan(
        (
            p00,
            p01,
            p02,
            p03,
            p10,
            p11,
            p12,
            p13,
            p20,
            p21,
            p22,
            p23,
            p30,
            p31,
            p32,
            p33,
            pu0,
            pu1,
            pu2,
            pu3,
        ),
        0,
        _compose_block4,
    )
    offs_n = tl.arange(0, n_coarse)
    first = offs_n[:, None] == 0
    x00 = tl.where(first, 1.0, 0.0)
    x01 = tl.zeros_like(p00)
    x02 = tl.zeros_like(p00)
    x03 = tl.zeros_like(p00)
    x10 = tl.zeros_like(p00)
    x11 = tl.where(first, 1.0, 0.0)
    x12 = tl.zeros_like(p00)
    x13 = tl.zeros_like(p00)
    x20 = tl.zeros_like(p00)
    x21 = tl.zeros_like(p00)
    x22 = tl.where(first, 1.0, 0.0)
    x23 = tl.zeros_like(p00)
    x30 = tl.zeros_like(p00)
    x31 = tl.zeros_like(p00)
    x32 = tl.zeros_like(p00)
    x33 = tl.where(first, 1.0, 0.0)
    xu0 = tl.zeros_like(p00)
    xu1 = tl.zeros_like(p00)
    xu2 = tl.zeros_like(p00)
    xu3 = tl.zeros_like(p00)
    for n in tl.static_range(1, n_coarse):
        sel_d = offs_n[:, None] == n
        sel_s = offs_n[:, None] == (n - 1)
        x00 = tl.where(sel_d, tl.sum(tl.where(sel_s, g00, 0.0), 0)[None, :], x00)
        x01 = tl.where(sel_d, tl.sum(tl.where(sel_s, g01, 0.0), 0)[None, :], x01)
        x02 = tl.where(sel_d, tl.sum(tl.where(sel_s, g02, 0.0), 0)[None, :], x02)
        x03 = tl.where(sel_d, tl.sum(tl.where(sel_s, g03, 0.0), 0)[None, :], x03)
        x10 = tl.where(sel_d, tl.sum(tl.where(sel_s, g10, 0.0), 0)[None, :], x10)
        x11 = tl.where(sel_d, tl.sum(tl.where(sel_s, g11, 0.0), 0)[None, :], x11)
        x12 = tl.where(sel_d, tl.sum(tl.where(sel_s, g12, 0.0), 0)[None, :], x12)
        x13 = tl.where(sel_d, tl.sum(tl.where(sel_s, g13, 0.0), 0)[None, :], x13)
        x20 = tl.where(sel_d, tl.sum(tl.where(sel_s, g20, 0.0), 0)[None, :], x20)
        x21 = tl.where(sel_d, tl.sum(tl.where(sel_s, g21, 0.0), 0)[None, :], x21)
        x22 = tl.where(sel_d, tl.sum(tl.where(sel_s, g22, 0.0), 0)[None, :], x22)
        x23 = tl.where(sel_d, tl.sum(tl.where(sel_s, g23, 0.0), 0)[None, :], x23)
        x30 = tl.where(sel_d, tl.sum(tl.where(sel_s, g30, 0.0), 0)[None, :], x30)
        x31 = tl.where(sel_d, tl.sum(tl.where(sel_s, g31, 0.0), 0)[None, :], x31)
        x32 = tl.where(sel_d, tl.sum(tl.where(sel_s, g32, 0.0), 0)[None, :], x32)
        x33 = tl.where(sel_d, tl.sum(tl.where(sel_s, g33, 0.0), 0)[None, :], x33)
        xu0 = tl.where(sel_d, tl.sum(tl.where(sel_s, gu0, 0.0), 0)[None, :], xu0)
        xu1 = tl.where(sel_d, tl.sum(tl.where(sel_s, gu1, 0.0), 0)[None, :], xu1)
        xu2 = tl.where(sel_d, tl.sum(tl.where(sel_s, gu2, 0.0), 0)[None, :], xu2)
        xu3 = tl.where(sel_d, tl.sum(tl.where(sel_s, gu3, 0.0), 0)[None, :], xu3)
    xb00 = x00[:, None, :]
    xb01 = x01[:, None, :]
    xb02 = x02[:, None, :]
    xb03 = x03[:, None, :]
    xb10 = x10[:, None, :]
    xb11 = x11[:, None, :]
    xb12 = x12[:, None, :]
    xb13 = x13[:, None, :]
    xb20 = x20[:, None, :]
    xb21 = x21[:, None, :]
    xb22 = x22[:, None, :]
    xb23 = x23[:, None, :]
    xb30 = x30[:, None, :]
    xb31 = x31[:, None, :]
    xb32 = x32[:, None, :]
    xb33 = x33[:, None, :]
    xbu0 = xu0[:, None, :]
    xbu1 = xu1[:, None, :]
    xbu2 = xu2[:, None, :]
    xbu3 = xu3[:, None, :]
    (
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
        ou0,
        ou1,
        ou2,
        ou3,
    ) = _compose_block4(
        xb00,
        xb01,
        xb02,
        xb03,
        xb10,
        xb11,
        xb12,
        xb13,
        xb20,
        xb21,
        xb22,
        xb23,
        xb30,
        xb31,
        xb32,
        xb33,
        xbu0,
        xbu1,
        xbu2,
        xbu3,
        i00,
        i01,
        i02,
        i03,
        i10,
        i11,
        i12,
        i13,
        i20,
        i21,
        i22,
        i23,
        i30,
        i31,
        i32,
        i33,
        iu0,
        iu1,
        iu2,
        iu3,
    )
    return (
        tl.reshape(o00, [BLOCK_T, BLOCK_D]),
        tl.reshape(o01, [BLOCK_T, BLOCK_D]),
        tl.reshape(o02, [BLOCK_T, BLOCK_D]),
        tl.reshape(o03, [BLOCK_T, BLOCK_D]),
        tl.reshape(o10, [BLOCK_T, BLOCK_D]),
        tl.reshape(o11, [BLOCK_T, BLOCK_D]),
        tl.reshape(o12, [BLOCK_T, BLOCK_D]),
        tl.reshape(o13, [BLOCK_T, BLOCK_D]),
        tl.reshape(o20, [BLOCK_T, BLOCK_D]),
        tl.reshape(o21, [BLOCK_T, BLOCK_D]),
        tl.reshape(o22, [BLOCK_T, BLOCK_D]),
        tl.reshape(o23, [BLOCK_T, BLOCK_D]),
        tl.reshape(o30, [BLOCK_T, BLOCK_D]),
        tl.reshape(o31, [BLOCK_T, BLOCK_D]),
        tl.reshape(o32, [BLOCK_T, BLOCK_D]),
        tl.reshape(o33, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou0, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou1, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou2, [BLOCK_T, BLOCK_D]),
        tl.reshape(ou3, [BLOCK_T, BLOCK_D]),
    )


@triton.jit
def _scan_select_block4(
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
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
):
    """Tile scan: Thomas+PCR, serial prefix, or ``tl.associative_scan``."""
    if THOMAS_C >= 2:
        return _thomas_pcr_scan_block4(
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
            BLOCK_T,
            BLOCK_D,
            THOMAS_C,
        )
    if SEQ:
        return _seq_scan_block4(
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
            BLOCK_T,
            BLOCK_D,
        )
    return tl.associative_scan(
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


@triton.jit
def _slstm_pred_j(
    c_prev,
    n_prev,
    m_prev,
    h_prev,
    zi_x,
    zf_x,
    zz_x,
    zo_x,
    r_i,
    r_f,
    r_z,
    r_o,
    eps,
):
    """sLSTM step + 4x4 channelwise J. Ties on max split 0.5/0.5 (PyTorch)."""
    z_i = r_i * h_prev + zi_x
    z_f = r_f * h_prev + zf_x
    z_z = r_z * h_prev + zz_x
    z_o = r_o * h_prev + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    alpha = tl.where(left > z_i, 1.0, 0.0) + tl.where(left == z_i, 0.5, 0.0)
    beta = 1.0 - alpha
    i_t = tl.exp(z_i - m_new)
    f_t = tl.exp(z_f + m_prev - m_new)
    z = _tanh(z_z)
    n_new = f_t * n_prev + i_t
    c_new = f_t * c_prev + i_t * z
    o = tl.sigmoid(z_o)
    denom = n_new + eps
    h_new = o * (c_new / denom)
    dm_dh = alpha * r_f + beta * r_i
    di_dm = -i_t * alpha
    df_dm = f_t * beta
    di_dh = i_t * (r_i - dm_dh)
    df_dh = f_t * (r_f - dm_dh)
    dz_dh = (1.0 - z * z) * r_z
    do_dh = o * (1.0 - o) * r_o
    j_cc = f_t
    j_cn = 0.0
    j_cm = df_dm * c_prev + di_dm * z
    j_ch = df_dh * c_prev + di_dh * z + i_t * dz_dh
    j_nc = 0.0
    j_nn = f_t
    j_nm = df_dm * n_prev + di_dm
    j_nh = df_dh * n_prev + di_dh
    j_mc = 0.0
    j_mn = 0.0
    j_mm = alpha
    j_mh = dm_dh
    inv = o / denom
    dn = -o * c_new / (denom * denom)
    du = c_new / denom
    j_hc = inv * j_cc
    j_hn = dn * j_nn
    j_hm = inv * j_cm + dn * j_nm
    j_hh = inv * j_ch + dn * j_nh + du * do_dh
    return (
        c_new,
        n_new,
        m_new,
        h_new,
        j_cc,
        j_cn,
        j_cm,
        j_ch,
        j_nc,
        j_nn,
        j_nm,
        j_nh,
        j_mc,
        j_mn,
        j_mm,
        j_mh,
        j_hc,
        j_hn,
        j_hm,
        j_hh,
    )


@triton.jit
def _slstm_log_pred_j(
    u_prev,
    ln_prev,
    m_prev,
    h_prev,
    zi_x,
    zf_x,
    zz_x,
    zo_x,
    r_i,
    r_f,
    r_z,
    r_o,
):
    """Convex combo + LSE step and 4x4 J. Slots are ``(u, log n, m, h)``."""
    z_i = r_i * h_prev + zi_x
    z_f = r_f * h_prev + zf_x
    z_z = r_z * h_prev + zz_x
    z_o = r_o * h_prev + zo_x
    left = z_f + m_prev
    m_new = tl.where(left > z_i, left, z_i)
    alpha = tl.where(left > z_i, 1.0, 0.0) + tl.where(left == z_i, 0.5, 0.0)
    beta = 1.0 - alpha
    a = z_f + m_prev - m_new + ln_prev
    b = z_i - m_new
    mx = tl.maximum(a, b)
    ln_new = mx + tl.log(tl.exp(a - mx) + tl.exp(b - mx))
    gamma = tl.exp(b - ln_new)
    z = _tanh(z_z)
    omg = 1.0 - gamma
    u_new = omg * u_prev + gamma * z
    o = tl.sigmoid(z_o)
    h_new = o * u_new
    dm_dh = alpha * r_f + beta * r_i
    da_dh = r_f - dm_dh
    db_dh = r_i - dm_dh
    da_dm = beta
    db_dm = -alpha
    dln_dln = omg
    dln_dm = omg * da_dm + gamma * db_dm
    dln_dh = omg * da_dh + gamma * db_dh
    dgamma_dln = gamma * (0.0 - dln_dln)
    dgamma_dm = gamma * (db_dm - dln_dm)
    dgamma_dh = gamma * (db_dh - dln_dh)
    dz_dh = (1.0 - z * z) * r_z
    uz = z - u_prev
    du_du = omg
    du_dln = uz * dgamma_dln
    du_dm = uz * dgamma_dm
    du_dh = uz * dgamma_dh + gamma * dz_dh
    do_dh = o * (1.0 - o) * r_o
    j_uu = du_du
    j_uln = du_dln
    j_um = du_dm
    j_uh = du_dh
    j_lnu = 0.0
    j_lnln = dln_dln
    j_lnm = dln_dm
    j_lnh = dln_dh
    j_mu = 0.0
    j_mn = 0.0
    j_mm = alpha
    j_mh = dm_dh
    j_hu = o * du_du
    j_hln = o * du_dln
    j_hm = o * du_dm
    j_hh = o * du_dh + u_new * do_dh
    return (
        u_new,
        ln_new,
        m_new,
        h_new,
        j_uu,
        j_uln,
        j_um,
        j_uh,
        j_lnu,
        j_lnln,
        j_lnm,
        j_lnh,
        j_mu,
        j_mn,
        j_mm,
        j_mh,
        j_hu,
        j_hln,
        j_hm,
        j_hh,
    )


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    zi = load_acc(base + offs_d[None, :] * sd, mask, 0.0)
    zf = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0)
    zz = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0)
    zo = load_acc(base + (offs_d[None, :] + 3 * d_h) * sd, mask, 0.0)
    return zi, zf, zz, zo


@triton.jit
def _load_r_gate(r_ptr, gate, offs_d, dmask, sg, sd):
    return load_acc(r_ptr + gate * sg + offs_d * sd, dmask, 0.0)


@triton.jit
def _slstm_init_kernel(
    wx_ptr,
    s_ptr,
    r_ptr,
    h0_ptr,
    d_h,
    time,
    eps,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_rg,
    stride_rd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """App. A: only t=0 sees ``h0``; later t still ``f(0, x_t)``."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    c0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    n0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    m0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    h0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], 0.0)
    n_prev = tl.where(is_t0, n0[None, :], 0.0)
    m_prev = tl.where(is_t0, m0[None, :], 0.0)
    h_prev = tl.where(is_t0, h0[None, :], 0.0)
    c, n, m, h, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _slstm_pred_j(
        c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
    )
    _store_state(
        s_ptr, c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, n, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, m, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr, h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )


@triton.jit
def _slstm_cell_local_scan_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    h0_ptr,
    j_loc_ptr,
    r_loc_ptr,
    agg_j_ptr,
    agg_r_ptr,
    time,
    d_h,
    eps,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rg,
    stride_rd,
    stride_jb,
    stride_jt,
    stride_jk,
    stride_jd,
    stride_rb,
    stride_rt,
    stride_rs,
    stride_rd_s,
    stride_ajb,
    stride_ajc,
    stride_ajk,
    stride_ajd,
    stride_arb,
    stride_arc,
    stride_ars,
    stride_ard,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOG: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    t0 = pid_c * BLOCK_T
    d0 = pid_d * BLOCK_D
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
    dmask = offs_d < d_h
    c = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    c_prev = _load_state(
        s_ptr,
        pid_b,
        offs_tm1,
        offs_d,
        SLOT_C,
        mask_prev,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    n_prev = _load_state(
        s_ptr,
        pid_b,
        offs_tm1,
        offs_d,
        SLOT_N,
        mask_prev,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    m_prev = _load_state(
        s_ptr,
        pid_b,
        offs_tm1,
        offs_d,
        SLOT_M,
        mask_prev,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    h_prev = _load_state(
        s_ptr,
        pid_b,
        offs_tm1,
        offs_d,
        SLOT_H,
        mask_prev,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    c0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    n0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    m0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    h0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], c_prev)
    n_prev = tl.where(is_t0, n0[None, :], n_prev)
    m_prev = tl.where(is_t0, m0[None, :], m_prev)
    h_prev = tl.where(is_t0, h0[None, :], h_prev)
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    if LOG:
        (
            c_new,
            n_new,
            m_new,
            h_new,
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
        ) = _slstm_log_pred_j(c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o)
    else:
        (
            c_new,
            n_new,
            m_new,
            h_new,
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
        ) = _slstm_pred_j(c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps)
    r0 = tl.where(mask, c_new - c, 0.0)
    r1 = tl.where(mask, n_new - n, 0.0)
    r2 = tl.where(mask, m_new - m, 0.0)
    r3 = tl.where(mask, h_new - h, 0.0)
    j00 = tl.where(mask, j00, 1.0)
    j01 = tl.where(mask, j01, 0.0)
    j02 = tl.where(mask, j02, 0.0)
    j03 = tl.where(mask, j03, 0.0)
    j10 = tl.where(mask, j10, 0.0)
    j11 = tl.where(mask, j11, 1.0)
    j12 = tl.where(mask, j12, 0.0)
    j13 = tl.where(mask, j13, 0.0)
    j20 = tl.where(mask, j20, 0.0)
    j21 = tl.where(mask, j21, 0.0)
    j22 = tl.where(mask, j22, 1.0)
    j23 = tl.where(mask, j23, 0.0)
    j30 = tl.where(mask, j30, 0.0)
    j31 = tl.where(mask, j31, 0.0)
    j32 = tl.where(mask, j32, 0.0)
    j33 = tl.where(mask, j33, 1.0)
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
    ) = _scan_select_block4(
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
        BLOCK_T,
        BLOCK_D,
        SEQ,
        THOMAS_C,
    )
    _store_j_lane(
        j_loc_ptr, s00, pid_b, offs_t, offs_d, 0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s01, pid_b, offs_t, offs_d, 1, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s02, pid_b, offs_t, offs_d, 2, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s03, pid_b, offs_t, offs_d, 3, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s10, pid_b, offs_t, offs_d, 4, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s11, pid_b, offs_t, offs_d, 5, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s12, pid_b, offs_t, offs_d, 6, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s13, pid_b, offs_t, offs_d, 7, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s20, pid_b, offs_t, offs_d, 8, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s21, pid_b, offs_t, offs_d, 9, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s22, pid_b, offs_t, offs_d, 10, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s23, pid_b, offs_t, offs_d, 11, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s30, pid_b, offs_t, offs_d, 12, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s31, pid_b, offs_t, offs_d, 13, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s32, pid_b, offs_t, offs_d, 14, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_j_lane(
        j_loc_ptr, s33, pid_b, offs_t, offs_d, 15, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    _store_state(
        r_loc_ptr,
        u0,
        pid_b,
        offs_t,
        offs_d,
        SLOT_C,
        mask,
        stride_rb,
        stride_rt,
        stride_rs,
        stride_rd_s,
    )
    _store_state(
        r_loc_ptr,
        u1,
        pid_b,
        offs_t,
        offs_d,
        SLOT_N,
        mask,
        stride_rb,
        stride_rt,
        stride_rs,
        stride_rd_s,
    )
    _store_state(
        r_loc_ptr,
        u2,
        pid_b,
        offs_t,
        offs_d,
        SLOT_M,
        mask,
        stride_rb,
        stride_rt,
        stride_rs,
        stride_rd_s,
    )
    _store_state(
        r_loc_ptr,
        u3,
        pid_b,
        offs_t,
        offs_d,
        SLOT_H,
        mask,
        stride_rb,
        stride_rt,
        stride_rs,
        stride_rd_s,
    )
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
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
        agg_r_ptr
        + pid_b * stride_arb
        + pid_c * stride_arc
        + SLOT_C * stride_ars
        + offs_d * stride_ard,
        tl.sum(tl.where(last, u0, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr
        + pid_b * stride_arb
        + pid_c * stride_arc
        + SLOT_N * stride_ars
        + offs_d * stride_ard,
        tl.sum(tl.where(last, u1, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr
        + pid_b * stride_arb
        + pid_c * stride_arc
        + SLOT_M * stride_ars
        + offs_d * stride_ard,
        tl.sum(tl.where(last, u2, 0.0), axis=0),
        dmask,
    )
    store_acc(
        agg_r_ptr
        + pid_b * stride_arb
        + pid_c * stride_arc
        + SLOT_H * stride_ars
        + offs_d * stride_ard,
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
    j00 = _load_j_lane(
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
    j01 = _load_j_lane(
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
    j02 = _load_j_lane(
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
    j03 = _load_j_lane(
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
    j10 = _load_j_lane(
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
    j11 = _load_j_lane(
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
    j12 = _load_j_lane(
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
    j13 = _load_j_lane(
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
    j20 = _load_j_lane(
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
    j21 = _load_j_lane(
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
    j22 = _load_j_lane(
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
    j23 = _load_j_lane(
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
    j30 = _load_j_lane(
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
    j31 = _load_j_lane(
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
    j32 = _load_j_lane(
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
    j33 = _load_j_lane(
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
    u0 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u1 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u2 = _load_state(
        agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard
    )
    u3 = _load_state(
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
    _store_state(
        incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s2, pid_b, offs_c, offs_d, 2, mask, stride_ib, stride_ic, stride_is, stride_id
    )
    _store_state(
        incl_r_ptr, s3, pid_b, offs_c, offs_d, 3, mask, stride_ib, stride_ic, stride_is, stride_id
    )


@triton.jit
def _slstm_apply_update_kernel(
    s_ptr,
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
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
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
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
    dmask = offs_d < d_h
    j00 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j01 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j02 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j03 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 3, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j10 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 4, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j11 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 5, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j12 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 6, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j13 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 7, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j20 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 8, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j21 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 9, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j22 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 10, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j23 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 11, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j30 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 12, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j31 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 13, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j32 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 14, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    j33 = _load_j_lane(
        j_loc_ptr, pid_b, offs_t, offs_d, 15, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd
    )
    r0 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r1 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r2 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    r3 = _load_state(
        r_loc_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd
    )
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    c0 = load_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + idx_c * stride_ic
        + SLOT_C * stride_is
        + offs_d * stride_id,
        dmask,
        0.0,
    )
    c1 = load_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + idx_c * stride_ic
        + SLOT_N * stride_is
        + offs_d * stride_id,
        dmask,
        0.0,
    )
    c2 = load_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + idx_c * stride_ic
        + SLOT_M * stride_is
        + offs_d * stride_id,
        dmask,
        0.0,
    )
    c3 = load_acc(
        incl_r_ptr
        + pid_b * stride_ib
        + idx_c * stride_ic
        + SLOT_H * stride_is
        + offs_d * stride_id,
        dmask,
        0.0,
    )
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    c2 = tl.where(pid_c > 0, c2, 0.0)
    c3 = tl.where(pid_c > 0, c3, 0.0)
    delta_c = j00 * c0[None, :] + j01 * c1[None, :] + j02 * c2[None, :] + j03 * c3[None, :] + r0
    delta_n = j10 * c0[None, :] + j11 * c1[None, :] + j12 * c2[None, :] + j13 * c3[None, :] + r1
    delta_m = j20 * c0[None, :] + j21 * c1[None, :] + j22 * c2[None, :] + j23 * c3[None, :] + r2
    delta_h = j30 * c0[None, :] + j31 * c1[None, :] + j32 * c2[None, :] + j33 * c3[None, :] + r3
    c = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    n = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    m = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    h = _load_state(
        s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd
    )
    _store_state(
        s_ptr,
        c + omega * delta_c,
        pid_b,
        offs_t,
        offs_d,
        SLOT_C,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        n + omega * delta_n,
        pid_b,
        offs_t,
        offs_d,
        SLOT_N,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        m + omega * delta_m,
        pid_b,
        offs_t,
        offs_d,
        SLOT_M,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )
    _store_state(
        s_ptr,
        h + omega * delta_h,
        pid_b,
        offs_t,
        offs_d,
        SLOT_H,
        mask,
        stride_sb,
        stride_st,
        stride_ss,
        stride_sd,
    )


@triton.jit
def _slstm_window_walk_kernel(
    s_ptr,
    wx_ptr,
    r_ptr,
    h0_ptr,
    time,
    d_h,
    eps,
    omega,
    n_tiles,
    stride_sb,
    stride_st,
    stride_ss,
    stride_sd,
    stride_h0b,
    stride_h0s,
    stride_h0d,
    stride_wb,
    stride_wt,
    stride_wd,
    stride_rg,
    stride_rd,
    bt_ptr,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SEQ: tl.constexpr,
    THOMAS_C: tl.constexpr,
    MAX_ITERS: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """Windowed Newton: sequential ``BLOCK_T`` tiles, solved-state carry in DRAM.

    Grid is ``(B, n_dtiles)``. Each program walks time; diag mix is per-feature
    so tiles of different ``d`` do not share a carry. Same contract as
    ``chunk_len=BLOCK_T``: K Newton steps on a tile, then the next tile reads
    the solved last state as ``h_{t-1}``.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    d0 = pid_d * BLOCK_D
    offs_d = d0 + tl.arange(0, BLOCK_D)
    dmask = offs_d < d_h
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd)
    c0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    n0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    m0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    h0 = _load_h0(
        h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, bt_ptr, HAS_BT
    )
    for pid_c in tl.range(n_tiles):
        t0 = pid_c * BLOCK_T
        offs_t = t0 + tl.arange(0, BLOCK_T)
        mask = (offs_t[:, None] < time) & (offs_d[None, :] < d_h)
        zi, zf, zz, zo = _load_wx(
            wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd
        )
        for _it in tl.static_range(MAX_ITERS):
            c = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_C,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            n = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_N,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            m = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_M,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            h = _load_state(
                s_ptr,
                pid_b,
                offs_t,
                offs_d,
                SLOT_H,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            offs_tm1 = offs_t - 1
            mask_prev = (
                (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
            )
            c_prev = _load_state(
                s_ptr,
                pid_b,
                offs_tm1,
                offs_d,
                SLOT_C,
                mask_prev,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            n_prev = _load_state(
                s_ptr,
                pid_b,
                offs_tm1,
                offs_d,
                SLOT_N,
                mask_prev,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            m_prev = _load_state(
                s_ptr,
                pid_b,
                offs_tm1,
                offs_d,
                SLOT_M,
                mask_prev,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            h_prev = _load_state(
                s_ptr,
                pid_b,
                offs_tm1,
                offs_d,
                SLOT_H,
                mask_prev,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            is_t0 = (offs_t == 0)[:, None]
            c_prev = tl.where(is_t0, c0[None, :], c_prev)
            n_prev = tl.where(is_t0, n0[None, :], n_prev)
            m_prev = tl.where(is_t0, m0[None, :], m_prev)
            h_prev = tl.where(is_t0, h0[None, :], h_prev)
            (
                c_new,
                n_new,
                m_new,
                h_new,
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
            ) = _slstm_pred_j(
                c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
            )
            r0 = tl.where(mask, c_new - c, 0.0)
            r1 = tl.where(mask, n_new - n, 0.0)
            r2 = tl.where(mask, m_new - m, 0.0)
            r3 = tl.where(mask, h_new - h, 0.0)
            j00 = tl.where(mask, j00, 1.0)
            j01 = tl.where(mask, j01, 0.0)
            j02 = tl.where(mask, j02, 0.0)
            j03 = tl.where(mask, j03, 0.0)
            j10 = tl.where(mask, j10, 0.0)
            j11 = tl.where(mask, j11, 1.0)
            j12 = tl.where(mask, j12, 0.0)
            j13 = tl.where(mask, j13, 0.0)
            j20 = tl.where(mask, j20, 0.0)
            j21 = tl.where(mask, j21, 0.0)
            j22 = tl.where(mask, j22, 1.0)
            j23 = tl.where(mask, j23, 0.0)
            j30 = tl.where(mask, j30, 0.0)
            j31 = tl.where(mask, j31, 0.0)
            j32 = tl.where(mask, j32, 0.0)
            j33 = tl.where(mask, j33, 1.0)
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
                u0,
                u1,
                u2,
                u3,
            ) = _scan_select_block4(
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
                BLOCK_T,
                BLOCK_D,
                SEQ,
                THOMAS_C,
            )
            _store_state(
                s_ptr,
                c + omega * u0,
                pid_b,
                offs_t,
                offs_d,
                SLOT_C,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            _store_state(
                s_ptr,
                n + omega * u1,
                pid_b,
                offs_t,
                offs_d,
                SLOT_N,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            _store_state(
                s_ptr,
                m + omega * u2,
                pid_b,
                offs_t,
                offs_d,
                SLOT_M,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )
            _store_state(
                s_ptr,
                h + omega * u3,
                pid_b,
                offs_t,
                offs_d,
                SLOT_H,
                mask,
                stride_sb,
                stride_st,
                stride_ss,
                stride_sd,
            )


def _parse_scan_tile(scan_tile: str) -> tuple[int, bool]:
    """Return ``(thomas_c, seq)``. ``thomas_c=0`` is assoc or seq."""
    if scan_tile == "assoc":
        return 0, False
    if scan_tile == "seq":
        return 0, True
    if scan_tile in ("thomas", "thomas4"):
        return 4, False
    if scan_tile == "thomas2":
        return 2, False
    raise ValueError(f"unknown scan_tile {scan_tile!r}; use assoc, seq, thomas, thomas2, thomas4")


def _slstm_fused_windows(
    wx: Tensor,
    r: Tensor,
    states: Tensor,
    h0: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    scan_tile: str,
    window_len: int,
    bt: Tensor,
    has_bt: bool,
) -> Tensor:
    """In-kernel windowed Newton: ``window_len`` tiles, solved-state DRAM carry.

    Same contract as ``chunk_len=window_len``: each window is a full K-step
    Newton; the next tile reads the solved last state as ``h_{t-1}``. One
    launch walks T. ``window_len`` is Triton ``BLOCK_T`` (32 / 64 / 128).
    """
    if window_len not in FUSED_WINDOW_LENS:
        raise ValueError(f"window_len must be one of {FUSED_WINDOW_LENS}, got {window_len}")
    batch, time, _four = wx.shape
    d_h = r.shape[-1]
    n_chunks, n_dtiles = time_tiles(
        time,
        d_h,
        window_len,
        _BLOCK_D,
        _CHUNK_PAD,
        cap_suffix=(f" (T≤{window_len * _CHUNK_PAD}). Shrink CHUNK_D or raise CHUNK_PAD."),
    )
    thomas_c, seq = _parse_scan_tile(scan_tile)
    log.debug(
        "newton_slstm_fused_windows",
        extra={
            "seq_len": time,
            "batch": batch,
            "d_h": d_h,
            "n_tiles": n_chunks,
            "window_len": window_len,
            "newton_iters": max_iters,
            "scan_tile": scan_tile,
            "device": str(wx.device),
            "dtype": str(wx.dtype),
        },
    )
    _slstm_window_walk_kernel[(batch, n_dtiles)](
        states,
        wx,
        r,
        h0,
        time,
        d_h,
        float(eps),
        float(omega),
        n_chunks,
        *states.stride(),
        *h0.stride(),
        *wx.stride(),
        *r.stride(),
        bt,
        SLOT_C=SLSTM_CELL,
        SLOT_N=SLSTM_NORMALIZER,
        SLOT_M=SLSTM_STABILIZER,
        SLOT_H=SLSTM_HIDDEN,
        BLOCK_T=window_len,
        BLOCK_D=_BLOCK_D,
        SEQ=seq,
        THOMAS_C=thomas_c,
        MAX_ITERS=max_iters,
        HAS_BT=has_bt,
    )
    return states


def _newton_slstm_fused_impl(
    wx: Tensor,
    r: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    h0: Tensor | None = None,
    states: Tensor | None = None,
    log_coords: bool = False,
    scan_tile: str = "assoc",
    time_loop: bool = False,
    window_len: int | None = None,
    block_table: Tensor | None = None,
    early_exit_atol: float | None = None,
    residual_fn=None,
    iters_done_out: list[int] | None = None,
) -> Tensor:
    """Alg. 1 for diag-mix ParaSLSTM. Public entry: ``pararnn::newton_slstm_fused``."""
    from pararnn.solvers.slstm_log import (
        slstm_clamp_log_coords,
        slstm_decode_log,
        slstm_encode_log,
    )

    wx = wx.contiguous()
    r = r.contiguous()
    batch, time, four_d = wx.shape
    d_h = r.shape[-1]
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    if four_d != 4 * d_h:
        raise ValueError(f"wx last dim {four_d} != 4 * d_h={4 * d_h}")
    if log_coords and block_table is not None:
        raise TypeError("block_table fused sLSTM is native coords only")
    h0, bt, has_bt = prepare_h0_block_table(wx, h0, batch, (SLSTM_SLOTS, d_h), block_table)
    validate_cuda_tensors(wx, r, h0, name="newton_slstm_fused")
    if states is not None:
        validate_cuda_tensors(wx, states, name="newton_slstm_fused")
    n_chunks, n_dtiles = time_tiles(
        time,
        d_h,
        _BLOCK_T,
        _BLOCK_D,
        _CHUNK_PAD,
        cap=False,
        cap_suffix=(f" (T≤{_BLOCK_T * _CHUNK_PAD}). Shrink CHUNK_D or raise CHUNK_PAD."),
    )
    grid_td = (batch, n_chunks, n_dtiles)
    if states is None:
        states = wx.new_empty(batch, time, SLSTM_SLOTS, d_h)
        _slstm_init_kernel[grid_td](
            wx,
            states,
            r,
            h0,
            d_h,
            time,
            float(eps),
            *wx.stride(),
            *states.stride(),
            *r.stride(),
            *h0.stride(),
            bt,
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            HAS_BT=has_bt,
        )
    else:
        if states.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"states shape {tuple(states.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if states.dtype != wx.dtype:
            states = states.to(dtype=wx.dtype)
        states = states.contiguous().clone()
    if log_coords:
        states = slstm_encode_log(states, eps=float(eps))
        h0 = slstm_encode_log(h0, eps=float(eps))
    if max_iters <= 0:
        if log_coords:
            return slstm_decode_log(states, eps=float(eps))
        return states
    work, h0_work = fp32_newton_work(states, h0)
    if time_loop:
        if log_coords:
            raise TypeError("fused_time_loop is native coords only")
        work = _slstm_fused_windows(
            wx,
            r,
            work,
            h0_work,
            max_iters=max_iters,
            omega=omega,
            eps=eps,
            scan_tile=scan_tile,
            window_len=int(window_len) if window_len is not None else FUSED_WINDOW_DEFAULT,
            bt=bt,
            has_bt=has_bt,
        )
        if work.dtype != wx.dtype:
            return work.to(dtype=wx.dtype)
        return work

    j_loc = work.new_empty(batch, time, 16, d_h)
    r_loc = work.new_empty(batch, time, SLSTM_SLOTS, d_h)
    agg_j = j_loc.new_empty(batch, n_chunks, 16, d_h)
    agg_r = r_loc.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h)
    incl_r = r_loc.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h) if n_chunks > 1 else None
    omega_f = float(omega)
    states32, r32 = alloc_fp32_update(work, r_loc, n_chunks)
    thomas_c, seq = _parse_scan_tile(scan_tile)

    for it in range(max_iters):
        _slstm_cell_local_scan_kernel[grid_td](
            work,
            wx,
            r,
            h0_work,
            j_loc,
            r_loc,
            agg_j,
            agg_r,
            time,
            d_h,
            float(eps),
            *work.stride(),
            *h0_work.stride(),
            *wx.stride(),
            *r.stride(),
            *j_loc.stride(),
            *r_loc.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            bt,
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            LOG=log_coords,
            SEQ=seq,
            THOMAS_C=thomas_c,
            HAS_BT=has_bt,
        )
        if states32 is not None and r32 is not None:
            fp32_omega_add(work, r_loc, omega_f, states32, r32)
        else:
            if incl_r is None:
                raise RuntimeError("fused sLSTM scan buffer missing for n_chunks>1")
            incl_r.copy_(
                incl_block_aggregates(
                    agg_j,
                    agg_r,
                    n_state=SLSTM_SLOTS,
                    chunk_incl_kernel=_chunk_incl_kernel,
                    chunk_pad=_CHUNK_PAD,
                    block_d=_CHUNK_D,
                    logger=log,
                )
            )
            _slstm_apply_update_kernel[grid_td](
                work,
                j_loc,
                r_loc,
                incl_r,
                time,
                d_h,
                float(omega),
                *work.stride(),
                *j_loc.stride(),
                *r_loc.stride(),
                *incl_r.stride(),
                SLOT_C=SLSTM_CELL,
                SLOT_N=SLSTM_NORMALIZER,
                SLOT_M=SLSTM_STABILIZER,
                SLOT_H=SLSTM_HIDDEN,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
            )
        if log_coords:
            work = slstm_clamp_log_coords(work)
        log_fused_iter(
            log,
            "newton_slstm_fused_iter",
            it=it,
            time=time,
            batch=batch,
            d_h=d_h,
            n_chunks=n_chunks,
        )
        if residual_fn is not None and early_exit_atol is not None:
            check = work
            if log_coords:
                check = slstm_decode_log(work, eps=float(eps))
            if check.dtype != wx.dtype:
                check = check.to(dtype=wx.dtype)
            if fused_early_exit_hit(
                residual_fn,
                early_exit_atol,
                check,
                iters_done_out=iters_done_out,
                it=it,
            ):
                out = check if log_coords else work
                if out.dtype != wx.dtype:
                    out = out.to(dtype=wx.dtype)
                log_fused_done(
                    log,
                    "newton_slstm_fused",
                    time=time,
                    batch=batch,
                    d_h=d_h,
                    max_iters=it + 1,
                    n_chunks=n_chunks,
                    log_coords=log_coords,
                    scan_tile=scan_tile,
                    acc_dtype=str(work.dtype),
                    early_exit=True,
                )
                return out
    mark_fused_iters_done(iters_done_out, max_iters)
    log_fused_done(
        log,
        "newton_slstm_fused",
        time=time,
        batch=batch,
        d_h=d_h,
        max_iters=max_iters,
        n_chunks=n_chunks,
        log_coords=log_coords,
        scan_tile=scan_tile,
        acc_dtype=str(work.dtype),
    )
    if log_coords:
        work = slstm_decode_log(work, eps=float(eps))
    if work.dtype != wx.dtype:
        return work.to(dtype=wx.dtype)
    return work
