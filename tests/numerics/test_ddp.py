"""Two-rank DDP / FSDP around ``ParaRNN``.

CPU gloo: DDP grads match the mean of per-rank losses (custom Newton
backward participates in the reducer). CUDA NCCL: skip unless two devices
are visible; after one SGD step parameters match across ranks. FSDP2
``fully_shard`` runs one backward on CPU and CUDA.
"""

from __future__ import annotations

import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from pararnn import NewtonConfig, ParaGRU, ParaRNN
from pararnn.distributed import last_newton_residuals, warmup_scan_kernels

# App. A: K=3. Wiring sizes — reducer/NCCL path, not a quality run.
_D_IN = 8
_T = 16
_BATCH = 2
_LR = 0.1  # one SGD step so ranks share an updated weight; not a train lr.
_ATOL = 1e-5
_RTOL = 1e-5


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _newton_cfg(*, scan_backend: str) -> NewtonConfig:
    return NewtonConfig(
        max_iters=3,
        scan_backend=scan_backend,
        residual_atol=None,
        residual_fail=None,
        picard_adapt=False,
    )


class _Toy(nn.Module):
    def __init__(self, *, scan_backend: str) -> None:
        super().__init__()
        self.rnn = ParaRNN(ParaGRU(_D_IN, _D_IN), config=_newton_cfg(scan_backend=scan_backend))
        self.head = nn.Linear(_D_IN, _D_IN)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.rnn(x))


def _init_group(rank: int, world: int, port: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    dist.init_process_group(backend, rank=rank, world_size=world)


def _batch(rank: int, device: torch.device) -> Tensor:
    torch.manual_seed(100 + rank)
    return torch.randn(_BATCH, _T, _D_IN, device=device)


def _ddp_cpu_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(0)
    model = _Toy(scan_backend="eager")
    ddp = DDP(model)
    x = _batch(rank, torch.device("cpu"))
    ddp.train()
    loss = ddp(x).square().mean()
    loss.backward()
    if rank == 0:
        ddp_grads = [p.grad.detach().clone() for p in ddp.parameters()]
        torch.manual_seed(0)
        ref = _Toy(scan_backend="eager")
        xs = [_batch(r, torch.device("cpu")) for r in range(world)]
        ref_loss = sum(ref(x_r).square().mean() for x_r in xs) / float(world)
        ref_loss.backward()
        for got, exp in zip(ddp_grads, (p.grad for p in ref.parameters()), strict=True):
            torch.testing.assert_close(got, exp, atol=_ATOL, rtol=_RTOL)
        res = last_newton_residuals(ddp)
        assert res, "Newton stats missing on DDP-wrapped ParaRNN"
        assert math.isfinite(res[0])
    dist.destroy_process_group()


def _fsdp_cpu_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = _Toy(scan_backend="eager")
    mesh = init_device_mesh("cpu", (world,))
    fully_shard(model, mesh=mesh)
    x = _batch(rank, torch.device("cpu"))
    model.train()
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss.detach())
    res = last_newton_residuals(model)
    assert res, "Newton stats missing on fully_shard ParaRNN"
    assert math.isfinite(res[0])
    dist.destroy_process_group()


def _ddp_cuda_worker(rank: int, world: int, port: int) -> None:
    torch.cuda.set_device(rank)
    _init_group(rank, world, port, "nccl")
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(0)
    model = _Toy(scan_backend="auto").to(device)
    warmup_scan_kernels(model, torch.randn(_BATCH, _T, _D_IN, device=device))
    ddp = DDP(model, device_ids=[rank], output_device=rank)
    ddp.train()
    x = _batch(rank, device)
    loss = ddp(x).square().mean()
    loss.backward()
    opt = torch.optim.SGD(ddp.parameters(), lr=_LR)
    opt.step()
    p = next(ddp.parameters()).detach().contiguous().flatten()
    gathered = [torch.empty_like(p) for _ in range(world)]
    dist.all_gather(gathered, p)
    if rank == 0:
        delta = (gathered[0] - gathered[1]).abs().amax()
        assert float(delta) == 0.0, float(delta)
        assert torch.isfinite(loss.detach())
        res = last_newton_residuals(ddp)
        assert res, "Newton stats missing after CUDA DDP step"
        assert math.isfinite(res[0])
    dist.destroy_process_group()


def _fsdp_cuda_worker(rank: int, world: int, port: int) -> None:
    torch.cuda.set_device(rank)
    _init_group(rank, world, port, "nccl")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(0)
    model = _Toy(scan_backend="auto").to(device)
    warmup_scan_kernels(model, torch.randn(_BATCH, _T, _D_IN, device=device))
    mesh = init_device_mesh("cuda", (world,))
    fully_shard(model, mesh=mesh)
    model.train()
    x = _batch(rank, device)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss.detach())
    dist.destroy_process_group()


def _spawn(worker, world: int = 2) -> None:
    port = _free_port()
    mp.spawn(worker, args=(world, port), nprocs=world, join=True)


def test_ddp_cpu_grads_match_mean_of_ranks() -> None:
    _spawn(_ddp_cpu_worker)


def test_fsdp_cpu_one_step() -> None:
    _spawn(_fsdp_cpu_worker)


@pytest.mark.cuda
def test_ddp_cuda_params_match_after_step() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    _spawn(_ddp_cuda_worker)


@pytest.mark.cuda
def test_fsdp_cuda_one_step() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    _spawn(_fsdp_cuda_worker)
