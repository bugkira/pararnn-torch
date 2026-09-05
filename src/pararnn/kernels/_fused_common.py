"""Shared fused-Newton host scaffold and LSTM/sLSTM state DRAM helpers."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra.cuda.libdevice import tanh as _nv_tanh

from pararnn.kernels.precision import load_acc, store_acc


@triton.jit
def _tanh(x):
    """CUDA libdevice tanh — same family as torch.tanh."""
    return _nv_tanh(x)


@triton.jit
def _bt_row(pid, bt_ptr, HAS_BT: tl.constexpr):
    if HAS_BT:
        return tl.load(bt_ptr + pid)
    return pid


@triton.jit
def is_seg_head(offs_t, cs_ptr, n_seq):
    """True where ``offs_t`` equals a packed start (``cu_seqlens[:-1]``)."""
    head = (offs_t * 0) != 0
    for s in range(n_seq):
        head = head | (offs_t == tl.load(cs_ptr + s))
    return head


@triton.jit
def gather_h0_heads(
    h0_ptr,
    offs_t,
    offs_d,
    dmask,
    cs_ptr,
    n_seq,
    stride_h0b,
    stride_h0d,
    bt_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_BT: tl.constexpr,
):
    """``h0[s]`` at packed heads, zeros elsewhere. Shape ``(BLOCK_T, BLOCK_D)``.

    ``HAS_BT``: packed sequence ``s`` reads pool row ``block_table[s]``.
    """
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    for s in range(n_seq):
        start = tl.load(cs_ptr + s)
        row = _bt_row(s, bt_ptr, HAS_BT)
        h0s = load_acc(h0_ptr + row * stride_h0b + offs_d * stride_h0d, dmask, 0.0)
        acc = tl.where((offs_t == start)[:, None], h0s[None, :], acc)
    return acc


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
def _load_h0(h0_ptr, pid_b, offs_d, slot, dmask, sb, ss, sd, bt_ptr, HAS_BT: tl.constexpr):
    row = _bt_row(pid_b, bt_ptr, HAS_BT)
    return load_acc(
        h0_ptr + row * sb + slot * ss + offs_d * sd,
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


def prepare_h0_block_table(
    wx: Tensor,
    h0: Tensor | None,
    dense_batch: int,
    tail: tuple[int, ...],
    block_table: Tensor | None,
) -> tuple[Tensor, Tensor, bool]:
    """Dense ``h0`` of batch ``dense_batch``, or a pool plus ``(dense_batch,)`` ids.

    Returns ``(h0, bt, has_bt)``. When ``has_bt`` is false, ``bt`` is a dummy
    pointer (``h0``) so Triton still gets a tensor.
    """
    if block_table is None:
        h0 = prepare_h0(wx, h0, (dense_batch, *tail))
        return h0, h0, False
    if h0 is None:
        raise ValueError("block_table needs h0 as a pool (C, …)")
    bt = block_table.to(device=wx.device, dtype=torch.int32).contiguous()
    if bt.dim() != 1:
        raise ValueError(f"block_table must be 1-D (B,), got {tuple(bt.shape)}")
    if int(bt.numel()) != dense_batch:
        raise ValueError(f"block_table B={int(bt.numel())} != {dense_batch}")
    h0 = h0.contiguous()
    if h0.dtype != wx.dtype:
        h0 = h0.to(dtype=wx.dtype)
    if tuple(h0.shape[1:]) != tail:
        raise ValueError(f"pool h0 tail {tuple(h0.shape[1:])} != {tail}")
    if int(h0.shape[0]) < 1:
        raise ValueError("pool h0 is empty")
    mx = int(bt.max().item()) if bt.numel() else -1
    mn = int(bt.min().item()) if bt.numel() else 0
    if mn < 0 or mx >= int(h0.shape[0]):
        raise ValueError(f"block_table slots [{mn}, {mx}] outside pool capacity {int(h0.shape[0])}")
    return h0, bt, True


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
