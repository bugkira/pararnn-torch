"""Two-rank row-parallel AllReduce reconstructs a full ``Linear(d_h, d_out)``."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pararnn import NewtonConfig, ParaGRU, newton_apply
from pararnn.tensor_parallel import (
    RowParallelLinear,
    copy_diag_cell_shard,
    copy_row_linear_shard,
    hidden_shard_slice,
    local_hidden_size,
    tensor_parallel_diag_block,
)

_D_IN = 8
_D_H = 8
_D_OUT = 6
_T = 12
_BATCH = 2
_ATOL = 1e-5
_RTOL = 1e-5
_CFG = NewtonConfig(
    max_iters=3,
    scan_backend="eager",
    residual_atol=None,
    residual_fail=None,
    picard_adapt=False,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _init_group(rank: int, world: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    dist.init_process_group(backend, rank=rank, world_size=world)


def _cpu_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(0)
    full_cell = ParaGRU(_D_IN, _D_H)
    full_out = nn.Linear(_D_H, _D_OUT)
    loc = local_hidden_size(_D_H, world)
    sl = hidden_shard_slice(_D_H, rank, world)
    cell = ParaGRU(_D_IN, loc)
    copy_diag_cell_shard(full_cell, cell, rank, world)
    row = RowParallelLinear(loc, _D_OUT)
    copy_row_linear_shard(row, full_out, sl)

    torch.manual_seed(4)
    x = torch.randn(_BATCH, _T, _D_IN)
    w = torch.randn(_BATCH, _T, _D_OUT)
    y_local = newton_apply(cell, x, _CFG)
    y = row(y_local)
    y_ref = full_out(newton_apply(full_cell, x, _CFG))
    torch.testing.assert_close(y, y_ref, atol=_ATOL, rtol=_RTOL)

    (y * w).sum().backward()
    (y_ref * w).sum().backward()
    torch.testing.assert_close(cell.a_z.grad, full_cell.a_z.grad[sl], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(row.weight.grad, full_out.weight.grad[:, sl], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(row.bias.grad, full_out.bias.grad, atol=_ATOL, rtol=_RTOL)
    dist.destroy_process_group()


def _cuda_worker(rank: int, world: int, port: int) -> None:
    torch.cuda.set_device(rank)
    _init_group(rank, world, port, "nccl")
    device = torch.device(f"cuda:{rank}")
    from pararnn.distributed import last_newton_residuals, warmup_scan_kernels

    torch.manual_seed(0)
    block = tensor_parallel_diag_block(
        "gru",
        _D_IN,
        _D_H,
        _D_OUT,
        config=NewtonConfig(max_iters=3, scan_backend="auto"),
        device=device,
    )
    dummy = torch.randn(_BATCH, _T, _D_IN, device=device)
    warmup_scan_kernels(block, dummy)
    torch.manual_seed(5)
    x = torch.randn(_BATCH, _T, _D_IN, device=device)
    block.train()
    y = block(x)
    loss = y.square().mean()
    loss.backward()
    gathered = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(gathered, y.contiguous())
    if rank == 0:
        torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
        assert torch.isfinite(loss.detach())
        res = last_newton_residuals(block)
        assert res
    dist.destroy_process_group()


def test_tp_cpu_matches_full_linear() -> None:
    port = _free_port()
    mp.spawn(_cpu_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.cuda
def test_tp_cuda_allreduce_agrees() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    port = _free_port()
    mp.spawn(_cuda_worker, args=(2, port), nprocs=2, join=True)
