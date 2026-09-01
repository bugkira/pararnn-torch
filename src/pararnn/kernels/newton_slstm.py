"""Fused ParaSLSTM Newton: diag mix, 4x4 J + scan (Beck sLSTM + paper Alg. 1).

``W_x(x)`` stays a cuBLAS GEMM. Not Apple's fused CUDA. Not FlashRNN.
``mix='head'`` / ``'dense'`` are not this kernel (not 4x4 SRAM).
CUDA float16/float32; cell+scan algebra in fp32. Not bf16 (Turing).
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.prec import check_cuda_real, load_acc, store_acc
from pararnn.layout import (
    SLSTM_CELL,
    SLSTM_HIDDEN,
    SLSTM_NORMALIZER,
    SLSTM_SLOTS,
    SLSTM_STABILIZER,
)

log = logging.getLogger(__name__)

# Same tiles as scan_block4 (20 scan lanes). 20 x 32 x 16 x 4 B = 40 KiB.
_BLOCK_T = 32
_BLOCK_D = 16
_CHUNK_D = 8
_CHUNK_PAD = 64  # 64 * 32 = 2048.


@triton.jit
def _tanh(x):
    return _nv_tanh(x)


@triton.jit
def _compose_block4(
    a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33,
    u0, u1, u2, u3,
    b00, b01, b02, b03, b10, b11, b12, b13, b20, b21, b22, b23, b30, b31, b32, b33,
    v0, v1, v2, v3,
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
        o00, o01, o02, o03, o10, o11, o12, o13, o20, o21, o22, o23, o30, o31, o32, o33,
        w0, w1, w2, w3,
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
        c_new, n_new, m_new, h_new,
        j_cc, j_cn, j_cm, j_ch,
        j_nc, j_nn, j_nm, j_nh,
        j_mc, j_mn, j_mm, j_mh,
        j_hc, j_hn, j_hm, j_hh,
    )


@triton.jit
def _load_wx(wx_ptr, pid_b, offs_t, offs_d, d_h, mask, sb, st, sd, FP16: tl.constexpr):
    base = wx_ptr + pid_b * sb + offs_t[:, None] * st
    zi = load_acc(base + offs_d[None, :] * sd, mask, 0.0, FP16)
    zf = load_acc(base + (offs_d[None, :] + d_h) * sd, mask, 0.0, FP16)
    zz = load_acc(base + (offs_d[None, :] + 2 * d_h) * sd, mask, 0.0, FP16)
    zo = load_acc(base + (offs_d[None, :] + 3 * d_h) * sd, mask, 0.0, FP16)
    return zi, zf, zz, zo


@triton.jit
def _load_r_gate(r_ptr, gate, offs_d, dmask, sg, sd, FP16: tl.constexpr):
    return load_acc(r_ptr + gate * sg + offs_d * sd, dmask, 0.0, FP16)


@triton.jit
def _load_state(s_ptr, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd, FP16: tl.constexpr):
    return load_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        mask,
        0.0,
        FP16,
    )


@triton.jit
def _store_state(s_ptr, val, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd, FP16: tl.constexpr):
    store_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        val,
        mask,
        FP16,
    )


@triton.jit
def _load_h0(h0_ptr, pid_b, offs_d, slot, dmask, sb, ss, sd, FP16: tl.constexpr):
    return load_acc(
        h0_ptr + pid_b * sb + slot * ss + offs_d * sd,
        dmask,
        0.0,
        FP16,
    )


@triton.jit
def _store_j_lane(ptr, val, pid_b, offs_t, offs_d, k, mask, sb, st, sk, sd, FP16: tl.constexpr):
    store_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        val,
        mask,
        FP16,
    )


@triton.jit
def _load_j_lane(ptr, pid_b, offs_t, offs_d, k, ident, mask, sb, st, sk, sd, FP16: tl.constexpr):
    return load_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        mask,
        ident,
        FP16,
    )


@triton.jit
def _store_agg_j(ptr, val, pid_b, pid_c, offs_d, k, dmask, sb, sc, sk, sd, FP16: tl.constexpr):
    store_acc(ptr + pid_b * sb + pid_c * sc + k * sk + offs_d * sd, val, dmask, FP16)


@triton.jit
def _slstm_init_kernel(
    wx_ptr,
    s_ptr,
    r_ptr,
    h0_ptr,
    d_h,
    time,
    eps,
    stride_wb, stride_wt, stride_wd,
    stride_sb, stride_st, stride_ss, stride_sd,
    stride_rg, stride_rd,
    stride_h0b, stride_h0s, stride_h0d,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
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
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd, FP16
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd, FP16)
    c0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    n0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    m0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    h0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], 0.0)
    n_prev = tl.where(is_t0, n0[None, :], 0.0)
    m_prev = tl.where(is_t0, m0[None, :], 0.0)
    h_prev = tl.where(is_t0, h0[None, :], 0.0)
    c, n, m, h, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = _slstm_pred_j(
        c_prev, n_prev, m_prev, h_prev, zi, zf, zz, zo, r_i, r_f, r_z, r_o, eps
    )
    _store_state(s_ptr, c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, n, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, m, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)


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
    stride_sb, stride_st, stride_ss, stride_sd,
    stride_h0b, stride_h0s, stride_h0d,
    stride_wb, stride_wt, stride_wd,
    stride_rg, stride_rd,
    stride_jb, stride_jt, stride_jk, stride_jd,
    stride_rb, stride_rt, stride_rs, stride_rd_s,
    stride_ajb, stride_ajc, stride_ajk, stride_ajd,
    stride_arb, stride_arc, stride_ars, stride_ard,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
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
    dmask = offs_d < d_h
    c = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    n = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    m = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    h = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    offs_tm1 = offs_t - 1
    mask_prev = (offs_tm1[:, None] >= 0) & (offs_tm1[:, None] < time) & (offs_d[None, :] < d_h)
    c_prev = _load_state(s_ptr, pid_b, offs_tm1, offs_d, SLOT_C, mask_prev, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    n_prev = _load_state(s_ptr, pid_b, offs_tm1, offs_d, SLOT_N, mask_prev, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    m_prev = _load_state(s_ptr, pid_b, offs_tm1, offs_d, SLOT_M, mask_prev, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    h_prev = _load_state(s_ptr, pid_b, offs_tm1, offs_d, SLOT_H, mask_prev, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    c0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_C, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    n0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_N, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    m0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_M, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    h0 = _load_h0(h0_ptr, pid_b, offs_d, SLOT_H, dmask, stride_h0b, stride_h0s, stride_h0d, FP16)
    is_t0 = (offs_t == 0)[:, None]
    c_prev = tl.where(is_t0, c0[None, :], c_prev)
    n_prev = tl.where(is_t0, n0[None, :], n_prev)
    m_prev = tl.where(is_t0, m0[None, :], m_prev)
    h_prev = tl.where(is_t0, h0[None, :], h_prev)
    zi, zf, zz, zo = _load_wx(
        wx_ptr, pid_b, offs_t, offs_d, d_h, mask, stride_wb, stride_wt, stride_wd, FP16
    )
    r_i = _load_r_gate(r_ptr, 0, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_f = _load_r_gate(r_ptr, 1, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_z = _load_r_gate(r_ptr, 2, offs_d, dmask, stride_rg, stride_rd, FP16)
    r_o = _load_r_gate(r_ptr, 3, offs_d, dmask, stride_rg, stride_rd, FP16)
    (
        c_new, n_new, m_new, h_new,
        j00, j01, j02, j03, j10, j11, j12, j13, j20, j21, j22, j23, j30, j31, j32, j33,
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
        s00, s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23, s30, s31, s32, s33,
        u0, u1, u2, u3,
    ) = tl.associative_scan(
        (
            j00, j01, j02, j03, j10, j11, j12, j13, j20, j21, j22, j23, j30, j31, j32, j33,
            r0, r1, r2, r3,
        ),
        0,
        _compose_block4,
    )
    _store_j_lane(j_loc_ptr, s00, pid_b, offs_t, offs_d, 0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s01, pid_b, offs_t, offs_d, 1, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s02, pid_b, offs_t, offs_d, 2, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s03, pid_b, offs_t, offs_d, 3, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s10, pid_b, offs_t, offs_d, 4, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s11, pid_b, offs_t, offs_d, 5, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s12, pid_b, offs_t, offs_d, 6, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s13, pid_b, offs_t, offs_d, 7, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s20, pid_b, offs_t, offs_d, 8, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s21, pid_b, offs_t, offs_d, 9, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s22, pid_b, offs_t, offs_d, 10, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s23, pid_b, offs_t, offs_d, 11, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s30, pid_b, offs_t, offs_d, 12, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s31, pid_b, offs_t, offs_d, 13, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s32, pid_b, offs_t, offs_d, 14, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_j_lane(j_loc_ptr, s33, pid_b, offs_t, offs_d, 15, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    _store_state(r_loc_ptr, u0, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd_s, FP16)
    _store_state(r_loc_ptr, u1, pid_b, offs_t, offs_d, SLOT_N, mask, stride_rb, stride_rt, stride_rs, stride_rd_s, FP16)
    _store_state(r_loc_ptr, u2, pid_b, offs_t, offs_d, SLOT_M, mask, stride_rb, stride_rt, stride_rs, stride_rd_s, FP16)
    _store_state(r_loc_ptr, u3, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd_s, FP16)
    last = (tl.arange(0, BLOCK_T) == (BLOCK_T - 1))[:, None]
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s00, 0.0), axis=0), pid_b, pid_c, offs_d, 0, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s01, 0.0), axis=0), pid_b, pid_c, offs_d, 1, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s02, 0.0), axis=0), pid_b, pid_c, offs_d, 2, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s03, 0.0), axis=0), pid_b, pid_c, offs_d, 3, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s10, 0.0), axis=0), pid_b, pid_c, offs_d, 4, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s11, 0.0), axis=0), pid_b, pid_c, offs_d, 5, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s12, 0.0), axis=0), pid_b, pid_c, offs_d, 6, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s13, 0.0), axis=0), pid_b, pid_c, offs_d, 7, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s20, 0.0), axis=0), pid_b, pid_c, offs_d, 8, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s21, 0.0), axis=0), pid_b, pid_c, offs_d, 9, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s22, 0.0), axis=0), pid_b, pid_c, offs_d, 10, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s23, 0.0), axis=0), pid_b, pid_c, offs_d, 11, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s30, 0.0), axis=0), pid_b, pid_c, offs_d, 12, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s31, 0.0), axis=0), pid_b, pid_c, offs_d, 13, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s32, 0.0), axis=0), pid_b, pid_c, offs_d, 14, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    _store_agg_j(agg_j_ptr, tl.sum(tl.where(last, s33, 0.0), axis=0), pid_b, pid_c, offs_d, 15, dmask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_C * stride_ars + offs_d * stride_ard, tl.sum(tl.where(last, u0, 0.0), axis=0), dmask, FP16)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_N * stride_ars + offs_d * stride_ard, tl.sum(tl.where(last, u1, 0.0), axis=0), dmask, FP16)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_M * stride_ars + offs_d * stride_ard, tl.sum(tl.where(last, u2, 0.0), axis=0), dmask, FP16)
    store_acc(agg_r_ptr + pid_b * stride_arb + pid_c * stride_arc + SLOT_H * stride_ars + offs_d * stride_ard, tl.sum(tl.where(last, u3, 0.0), axis=0), dmask, FP16)


@triton.jit
def _chunk_incl_kernel(
    agg_j_ptr,
    agg_r_ptr,
    incl_r_ptr,
    n_chunks,
    d_h,
    stride_ajb, stride_ajc, stride_ajk, stride_ajd,
    stride_arb, stride_arc, stride_ars, stride_ard,
    stride_ib, stride_ic, stride_is, stride_id,
    CHUNK_PAD: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    d0 = tl.program_id(1) * BLOCK_D
    offs_c = tl.arange(0, CHUNK_PAD)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    mask = (offs_c[:, None] < n_chunks) & (offs_d[None, :] < d_h)
    j00 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 0, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j01 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 1, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j02 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 2, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j03 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 3, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j10 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 4, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j11 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 5, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j12 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 6, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j13 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 7, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j20 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 8, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j21 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 9, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j22 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 10, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j23 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 11, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j30 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 12, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j31 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 13, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j32 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 14, 0.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    j33 = _load_j_lane(agg_j_ptr, pid_b, offs_c, offs_d, 15, 1.0, mask, stride_ajb, stride_ajc, stride_ajk, stride_ajd, FP16)
    u0 = _load_state(agg_r_ptr, pid_b, offs_c, offs_d, 0, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    u1 = _load_state(agg_r_ptr, pid_b, offs_c, offs_d, 1, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    u2 = _load_state(agg_r_ptr, pid_b, offs_c, offs_d, 2, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    u3 = _load_state(agg_r_ptr, pid_b, offs_c, offs_d, 3, mask, stride_arb, stride_arc, stride_ars, stride_ard, FP16)
    (
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
        s0, s1, s2, s3,
    ) = tl.associative_scan(
        (
            j00, j01, j02, j03, j10, j11, j12, j13, j20, j21, j22, j23, j30, j31, j32, j33,
            u0, u1, u2, u3,
        ),
        0,
        _compose_block4,
    )
    _store_state(incl_r_ptr, s0, pid_b, offs_c, offs_d, 0, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)
    _store_state(incl_r_ptr, s1, pid_b, offs_c, offs_d, 1, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)
    _store_state(incl_r_ptr, s2, pid_b, offs_c, offs_d, 2, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)
    _store_state(incl_r_ptr, s3, pid_b, offs_c, offs_d, 3, mask, stride_ib, stride_ic, stride_is, stride_id, FP16)


@triton.jit
def _slstm_apply_update_kernel(
    s_ptr,
    j_loc_ptr,
    r_loc_ptr,
    incl_r_ptr,
    time,
    d_h,
    omega,
    stride_sb, stride_st, stride_ss, stride_sd,
    stride_jb, stride_jt, stride_jk, stride_jd,
    stride_rb, stride_rt, stride_rs, stride_rd,
    stride_ib, stride_ic, stride_is, stride_id,
    SLOT_C: tl.constexpr,
    SLOT_N: tl.constexpr,
    SLOT_M: tl.constexpr,
    SLOT_H: tl.constexpr,
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
    dmask = offs_d < d_h
    j00 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 0, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j01 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 1, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j02 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 2, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j03 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 3, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j10 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 4, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j11 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 5, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j12 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 6, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j13 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 7, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j20 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 8, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j21 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 9, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j22 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 10, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j23 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 11, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j30 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 12, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j31 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 13, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j32 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 14, 0.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    j33 = _load_j_lane(j_loc_ptr, pid_b, offs_t, offs_d, 15, 1.0, mask, stride_jb, stride_jt, stride_jk, stride_jd, FP16)
    r0 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    r1 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    r2 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    r3 = _load_state(r_loc_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_rb, stride_rt, stride_rs, stride_rd, FP16)
    idx_c = tl.where(pid_c > 0, pid_c - 1, 0)
    c0 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_C * stride_is + offs_d * stride_id, dmask, 0.0, FP16)
    c1 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_N * stride_is + offs_d * stride_id, dmask, 0.0, FP16)
    c2 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_M * stride_is + offs_d * stride_id, dmask, 0.0, FP16)
    c3 = load_acc(incl_r_ptr + pid_b * stride_ib + idx_c * stride_ic + SLOT_H * stride_is + offs_d * stride_id, dmask, 0.0, FP16)
    c0 = tl.where(pid_c > 0, c0, 0.0)
    c1 = tl.where(pid_c > 0, c1, 0.0)
    c2 = tl.where(pid_c > 0, c2, 0.0)
    c3 = tl.where(pid_c > 0, c3, 0.0)
    delta_c = j00 * c0[None, :] + j01 * c1[None, :] + j02 * c2[None, :] + j03 * c3[None, :] + r0
    delta_n = j10 * c0[None, :] + j11 * c1[None, :] + j12 * c2[None, :] + j13 * c3[None, :] + r1
    delta_m = j20 * c0[None, :] + j21 * c1[None, :] + j22 * c2[None, :] + j23 * c3[None, :] + r2
    delta_h = j30 * c0[None, :] + j31 * c1[None, :] + j32 * c2[None, :] + j33 * c3[None, :] + r3
    c = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    n = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    m = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    h = _load_state(s_ptr, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, c + omega * delta_c, pid_b, offs_t, offs_d, SLOT_C, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, n + omega * delta_n, pid_b, offs_t, offs_d, SLOT_N, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, m + omega * delta_m, pid_b, offs_t, offs_d, SLOT_M, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)
    _store_state(s_ptr, h + omega * delta_h, pid_b, offs_t, offs_d, SLOT_H, mask, stride_sb, stride_st, stride_ss, stride_sd, FP16)


def newton_slstm_fused(
    wx: Tensor,
    r: Tensor,
    *,
    max_iters: int,
    omega: float,
    eps: float,
    h0: Tensor | None = None,
    states: Tensor | None = None,
) -> Tensor:
    """Alg. 1 for diag-mix ParaSLSTM. ``wx`` is ``W_x(x)`` with shape ``(B, T, 4 d_h)``.

    ``r`` is clipped ``R`` ``(4, d_h)``. ``h0`` is paper ``h_0`` (default zeros),
    shape ``(B, 4, d_h)``. ``states`` is the Newton guess (zero-hidden init);
    if omitted, App. A ``f(0, x_t)`` is used.
    """
    wx = wx.contiguous()
    r = r.contiguous()
    batch, time, four_d = wx.shape
    d_h = r.shape[-1]
    if r.shape != (4, d_h):
        raise ValueError(f"r shape {tuple(r.shape)} != {(4, d_h)}")
    if four_d != 4 * d_h:
        raise ValueError(f"wx last dim {four_d} != 4 * d_h={4 * d_h}")
    if h0 is None:
        h0 = wx.new_zeros(batch, SLSTM_SLOTS, d_h)
    else:
        h0 = h0.contiguous()
        if h0.shape != (batch, SLSTM_SLOTS, d_h):
            raise ValueError(f"h0 shape {tuple(h0.shape)} != {(batch, SLSTM_SLOTS, d_h)}")
        if h0.dtype != wx.dtype:
            h0 = h0.to(dtype=wx.dtype)
    fp16 = check_cuda_real(wx, r, h0, name="newton_slstm_fused")
    if states is not None:
        check_cuda_real(wx, states, name="newton_slstm_fused")
    n_chunks = (time + _BLOCK_T - 1) // _BLOCK_T
    if n_chunks > _CHUNK_PAD:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {_BLOCK_T}; cap is {_CHUNK_PAD}. "
            "Increase BLOCK_T rather than copying a longer Apple kernel."
        )
    n_dtiles = (d_h + _BLOCK_D - 1) // _BLOCK_D
    n_dtiles_chunk = (d_h + _CHUNK_D - 1) // _CHUNK_D
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
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            FP16=fp16,
        )
    else:
        if states.shape != (batch, time, SLSTM_SLOTS, d_h):
            raise ValueError(
                f"states shape {tuple(states.shape)} != {(batch, time, SLSTM_SLOTS, d_h)}"
            )
        if states.dtype != wx.dtype:
            states = states.to(dtype=wx.dtype)
        states = states.contiguous().clone()
    if max_iters <= 0:
        return states

    j_loc = states.new_empty(batch, time, 16, d_h)
    r_loc = states.new_empty(batch, time, SLSTM_SLOTS, d_h)
    agg_j = j_loc.new_empty(batch, n_chunks, 16, d_h)
    agg_r = r_loc.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h)
    incl_r = r_loc.new_empty(batch, n_chunks, SLSTM_SLOTS, d_h) if n_chunks > 1 else None

    for it in range(max_iters):
        _slstm_cell_local_scan_kernel[grid_td](
            states,
            wx,
            r,
            h0,
            j_loc,
            r_loc,
            agg_j,
            agg_r,
            time,
            d_h,
            float(eps),
            *states.stride(),
            *h0.stride(),
            *wx.stride(),
            *r.stride(),
            *j_loc.stride(),
            *r_loc.stride(),
            *agg_j.stride(),
            *agg_r.stride(),
            SLOT_C=SLSTM_CELL,
            SLOT_N=SLSTM_NORMALIZER,
            SLOT_M=SLSTM_STABILIZER,
            SLOT_H=SLSTM_HIDDEN,
            BLOCK_T=_BLOCK_T,
            BLOCK_D=_BLOCK_D,
            FP16=fp16,
        )
        if n_chunks == 1:
            states.copy_((states.float() + float(omega) * r_loc.float()).to(states.dtype))
        else:
            _chunk_incl_kernel[(batch, n_dtiles_chunk)](
                agg_j,
                agg_r,
                incl_r,
                n_chunks,
                d_h,
                *agg_j.stride(),
                *agg_r.stride(),
                *incl_r.stride(),
                CHUNK_PAD=_CHUNK_PAD,
                BLOCK_D=_CHUNK_D,
                FP16=fp16,
            )
            _slstm_apply_update_kernel[grid_td](
                states,
                j_loc,
                r_loc,
                incl_r,
                time,
                d_h,
                float(omega),
                *states.stride(),
                *j_loc.stride(),
                *r_loc.stride(),
                *incl_r.stride(),
                SLOT_C=SLSTM_CELL,
                SLOT_N=SLSTM_NORMALIZER,
                SLOT_M=SLSTM_STABILIZER,
                SLOT_H=SLSTM_HIDDEN,
                BLOCK_T=_BLOCK_T,
                BLOCK_D=_BLOCK_D,
                FP16=fp16,
            )
        if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
            log.debug(
                "newton_slstm_fused_iter",
                extra={
                    "iter": it,
                    "seq_len": time,
                    "batch": batch,
                    "d_h": d_h,
                    "n_chunks": n_chunks,
                },
            )
    log.debug(
        "newton_slstm_fused",
        extra={
            "seq_len": time,
            "batch": batch,
            "d_h": d_h,
            "max_iters": max_iters,
            "n_chunks": n_chunks,
        },
    )
    return states
