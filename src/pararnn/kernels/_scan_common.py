"""Shared block-scan DRAM helpers and host local/chunk/apply orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import triton
from torch import Tensor

from pararnn.kernels._fused_common import time_tiles
from pararnn.kernels.precision import load_acc, store_acc, validate_cuda_tensors


@triton.jit
def _load_j(ptr, pid_b, offs_t, offs_d, k, ident, mask, sb, st, sk, sd):
    return load_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        mask,
        ident,
    )


@triton.jit
def _store_j(ptr, val, pid_b, offs_t, offs_d, k, mask, sb, st, sk, sd):
    store_acc(
        ptr + pid_b * sb + offs_t[:, None] * st + k * sk + offs_d[None, :] * sd,
        val,
        mask,
    )


@triton.jit
def _store_agg_j(ptr, val, pid_b, pid_c, offs_d, k, dmask, sb, sc, sk, sd):
    store_acc(ptr + pid_b * sb + pid_c * sc + k * sk + offs_d * sd, val, dmask)


def run_block_scan_triton(
    jac: Tensor,
    residual: Tensor,
    *,
    n_state: int,
    local_kernel: Callable[..., object],
    chunk_incl_kernel: Callable[..., object],
    apply_carry_kernel: Callable[..., object],
    block_t: int,
    block_d: int,
    chunk_pad: int,
    name: str,
    logger: logging.Logger,
    chunk_d: int | None = None,
) -> Tensor:
    """Local tile scan, then chunk-inclusive carry, then apply. ``n_state`` is 2 or 4."""
    if residual.dim() != 4 or residual.shape[2] != n_state:
        raise ValueError(f"residual must be (batch, time, {n_state}, d)")
    if jac.shape[:2] != residual.shape[:2] or jac.shape[-1] != residual.shape[-1]:
        raise ValueError("jac/residual batch, time, d mismatch")
    if jac.shape[2:4] != (n_state, n_state):
        raise ValueError(f"jac must be (batch, time, {n_state}, {n_state}, d)")
    validate_cuda_tensors(jac, residual, name=name)
    jac = jac.contiguous()
    residual = residual.contiguous()
    batch, time, _, d_h = residual.shape
    if time <= 1:
        return residual.clone()

    n_chunks, n_dtiles = time_tiles(time, d_h, block_t, block_d, chunk_pad)
    n_jac = n_state * n_state
    j_flat = jac.view(batch, time, n_jac, d_h)
    r_flat = residual
    j_loc = torch.empty_like(j_flat)
    r_loc = torch.empty_like(r_flat)
    agg_j = j_flat.new_empty(batch, n_chunks, n_jac, d_h)
    agg_r = r_flat.new_empty(batch, n_chunks, n_state, d_h)

    local_kernel[(batch, n_chunks, n_dtiles)](
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
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    if n_chunks == 1:
        logger.debug(
            name,
            extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": 1},
        )
        return r_loc

    incl_r = r_flat.new_empty(batch, n_chunks, n_state, d_h)
    cd = block_d if chunk_d is None else chunk_d
    n_dtiles_chunk = (d_h + cd - 1) // cd
    chunk_incl_kernel[(batch, n_dtiles_chunk)](
        agg_j,
        agg_r,
        incl_r,
        n_chunks,
        d_h,
        *agg_j.stride(),
        *agg_r.stride(),
        *incl_r.stride(),
        CHUNK_PAD=chunk_pad,
        BLOCK_D=cd,
    )
    out = torch.empty_like(r_flat)
    apply_carry_kernel[(batch, n_chunks, n_dtiles)](
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
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    logger.debug(
        name,
        extra={"batch": batch, "seq_len": time, "d_h": d_h, "n_chunks": n_chunks},
    )
    return out
