"""Shared fused-Newton host scaffold and LSTM/sLSTM state DRAM helpers."""

from __future__ import annotations

import logging

import torch
import triton
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc


@triton.jit
def _tanh(x):
    """CUDA libdevice tanh — same family as torch.tanh."""
    return _nv_tanh(x)


@triton.jit
def _load_state(s_ptr, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    return load_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        mask,
        0.0,
    )


@triton.jit
def _store_state(s_ptr, val, pid_b, offs_t, offs_d, slot, mask, sb, st, ss, sd):
    store_acc(
        s_ptr + pid_b * sb + offs_t[:, None] * st + slot * ss + offs_d[None, :] * sd,
        val,
        mask,
    )


@triton.jit
def _load_h0(h0_ptr, pid_b, offs_d, slot, dmask, sb, ss, sd):
    return load_acc(
        h0_ptr + pid_b * sb + slot * ss + offs_d * sd,
        dmask,
        0.0,
    )


def prepare_h0(wx: Tensor, h0: Tensor | None, shape: tuple[int, ...]) -> Tensor:
    if h0 is None:
        return wx.new_zeros(*shape)
    h0 = h0.contiguous()
    if h0.shape != shape:
        raise ValueError(f"h0 shape {tuple(h0.shape)} != {shape}")
    if h0.dtype != wx.dtype:
        h0 = h0.to(dtype=wx.dtype)
    return h0


def time_tiles(
    time: int,
    d_h: int,
    block_t: int,
    block_d: int,
    chunk_pad: int,
    *,
    cap: bool = True,
    cap_suffix: str = ". Increase BLOCK_T or CHUNK_PAD.",
) -> tuple[int, int]:
    n_chunks = (time + block_t - 1) // block_t
    if cap and n_chunks > chunk_pad:
        raise ValueError(
            f"T={time} needs {n_chunks} tiles of {block_t}; cap is {chunk_pad}{cap_suffix}"
        )
    return n_chunks, (d_h + block_d - 1) // block_d


def alloc_fp32_update(
    states: Tensor, r_loc: Tensor, n_chunks: int
) -> tuple[Tensor | None, Tensor | None]:
    if n_chunks != 1:
        return None, None
    return (
        states.new_empty(states.shape, dtype=torch.float32),
        r_loc.new_empty(r_loc.shape, dtype=torch.float32),
    )


def fp32_newton_work(states: Tensor, h0: Tensor) -> tuple[Tensor, Tensor]:
    """fp32 copies of the guess and ``h0`` when DRAM is fp16/bf16.

    ``load_acc`` already widens a single load, but J tiles and the Newton
    update were stored back in the tensor dtype. That truncation is why
    P=3 bf16 residual blew up (~0.5 vs ~6e-6 in fp32). Ampere bf16 fused
    is still slower than fp32 (hot path is fp32 4×4 PCR, not TC GEMM);
    this helper does not change that. ``W_x`` stays a GEMM in the tensor
    dtype.
    """
    if states.dtype == torch.float32:
        return states, h0
    return states.float(), h0.float()


def fp32_omega_add(
    states: Tensor, r_loc: Tensor, omega: float, states32: Tensor, r32: Tensor
) -> None:
    states32.copy_(states)
    r32.copy_(r_loc)
    states32.add_(r32, alpha=omega)
    states.copy_(states32)


def log_fused_iter(
    log: logging.Logger,
    event: str,
    *,
    it: int,
    time: int,
    batch: int,
    d_h: int,
    n_chunks: int,
) -> None:
    if log.isEnabledFor(logging.DEBUG) and not torch.compiler.is_compiling():
        log.debug(
            event,
            extra={
                "iter": it,
                "seq_len": time,
                "batch": batch,
                "d_h": d_h,
                "n_chunks": n_chunks,
            },
        )


def log_fused_done(
    log: logging.Logger,
    event: str,
    *,
    time: int,
    batch: int,
    d_h: int,
    max_iters: int,
    n_chunks: int,
    **extra: object,
) -> None:
    payload: dict[str, object] = {
        "seq_len": time,
        "batch": batch,
        "d_h": d_h,
        "max_iters": max_iters,
        "n_chunks": n_chunks,
    }
    payload.update(extra)
    log.debug(event, extra=payload)
