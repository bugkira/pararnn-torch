"""Context-parallel diag scan: Rank 1 carry, VJP, Newton hook, concat vs full scan."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pararnn import NewtonConfig, ParaGRU, newton_apply
from pararnn.solvers.scan import reverse_scan_diag, scan_diag
from pararnn.solvers.seq_parallel import (
    all_gather_time_tiles,
    reverse_scan_diag_context_parallel,
    scan_diag_context_parallel,
    time_shard_bounds,
)

_ATOL = 1e-5
_RTOL = 1e-5
_GRAD_ATOL = 2e-4


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


def _cpu_fwd_worker(rank: int, world: int, port: int, time: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(time)
    batch, dim = 3, 5
    jac = torch.randn(batch, time, dim) * 0.3
    residual = torch.randn(batch, time, dim)
    start, end = time_shard_bounds(time, rank, world)
    got = scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        residual[:, start:end].contiguous(),
        backend="eager",
    )
    ref = scan_diag(jac, residual, backend="eager")
    err = jac.new_zeros(()) if got.numel() == 0 else (got - ref[:, start:end]).abs().amax()
    torch.testing.assert_close(got, ref[:, start:end], atol=_ATOL, rtol=_RTOL)
    gathered = all_gather_time_tiles(got, time, rank, world, None)
    torch.testing.assert_close(gathered, ref, atol=_ATOL, rtol=_RTOL)
    gathered_err = [torch.zeros((), device=jac.device) for _ in range(world)]
    dist.all_gather(gathered_err, err)
    if rank == 0:
        e0, e1 = float(gathered_err[0]), float(gathered_err[1])
        assert e0 < 1e-4, f"rank0 max_abs={e0}"
        assert e1 < 1e-4, f"rank1 max_abs={e1} (carry path)"
    dist.destroy_process_group()


def _cpu_grad_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(3)
    time, batch, dim = 16, 2, 4
    jac = (torch.randn(batch, time, dim) * 0.3).requires_grad_(True)
    residual = torch.randn(batch, time, dim, requires_grad=True)
    w = torch.randn(batch, time, dim)
    start, end = time_shard_bounds(time, rank, world)
    y = scan_diag_context_parallel(
        jac[:, start:end],
        residual[:, start:end],
        backend="eager",
    )
    (y * w[:, start:end]).sum().backward()
    ref_j = jac.detach().clone().requires_grad_(True)
    ref_r = residual.detach().clone().requires_grad_(True)
    y_ref = scan_diag(ref_j, ref_r, backend="eager")
    (y_ref * w).sum().backward()
    torch.testing.assert_close(
        jac.grad[:, start:end], ref_j.grad[:, start:end], atol=_GRAD_ATOL, rtol=_GRAD_ATOL
    )
    torch.testing.assert_close(
        residual.grad[:, start:end],
        ref_r.grad[:, start:end],
        atol=_GRAD_ATOL,
        rtol=_GRAD_ATOL,
    )
    dist.destroy_process_group()


def _cpu_reverse_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(9)
    time, batch, dim = 17, 2, 5
    jac = torch.randn(batch, time, dim) * 0.3
    partial = torch.randn(batch, time, dim)
    start, end = time_shard_bounds(time, rank, world)
    got = reverse_scan_diag_context_parallel(
        jac[:, start:end].contiguous(),
        partial[:, start:end].contiguous(),
        backend="eager",
    )
    ref = reverse_scan_diag(jac, partial, backend="eager")
    torch.testing.assert_close(got, ref[:, start:end], atol=_ATOL, rtol=_RTOL)
    dist.destroy_process_group()


def _cpu_newton_worker(rank: int, world: int, port: int) -> None:
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(0)
    cell = ParaGRU(6, 6)
    x = torch.randn(2, 24, 6)
    cfg_cp = NewtonConfig(
        max_iters=3,
        scan_backend="context_parallel",
        residual_atol=None,
        residual_fail=None,
    )
    cfg_e = NewtonConfig(
        max_iters=3, scan_backend="eager", residual_atol=None, residual_fail=None
    )
    y_cp = newton_apply(cell, x, cfg_cp)
    torch.manual_seed(0)
    cell_e = ParaGRU(6, 6)
    y_e = newton_apply(cell_e, x, cfg_e)
    torch.testing.assert_close(y_cp, y_e, atol=1e-4, rtol=1e-4)
    dist.destroy_process_group()


def _cpu_newton_grad_worker(rank: int, world: int, port: int) -> None:
    """Eq. 2.6 reverse scan under CP: Rank 1 sends μ carry to Rank 0."""
    _init_group(rank, world, port, "gloo")
    torch.manual_seed(4)
    cell = ParaGRU(6, 6)
    x = torch.randn(2, 24, 6, requires_grad=True)
    w = torch.randn_like(x)
    cfg_cp = NewtonConfig(
        max_iters=3,
        scan_backend="context_parallel",
        residual_atol=None,
        residual_fail=None,
    )
    y = newton_apply(cell, x, cfg_cp)
    (y * w).sum().backward()
    torch.manual_seed(4)
    cell_e = ParaGRU(6, 6)
    x_e = x.detach().clone().requires_grad_(True)
    cfg_e = NewtonConfig(
        max_iters=3, scan_backend="eager", residual_atol=None, residual_fail=None
    )
    y_e = newton_apply(cell_e, x_e, cfg_e)
    (y_e * w).sum().backward()
    torch.testing.assert_close(x.grad, x_e.grad, atol=_GRAD_ATOL, rtol=_GRAD_ATOL)
    for p, p_e in zip(cell.parameters(), cell_e.parameters(), strict=True):
        torch.testing.assert_close(p.grad, p_e.grad, atol=_GRAD_ATOL, rtol=_GRAD_ATOL)
    dist.destroy_process_group()


def _cuda_concat_worker(rank: int, world: int, port: int) -> None:
    """Eager CP on each GPU, concat on CPU vs CPU ``scan_diag``. Rank 1 is the carry."""
    torch.cuda.set_device(rank)
    _init_group(rank, world, port, "nccl")
    torch.manual_seed(11)
    time, batch, dim = 32, 2, 8
    jac_cpu = torch.randn(batch, time, dim) * 0.3
    residual_cpu = torch.randn(batch, time, dim)
    ref = scan_diag(jac_cpu, residual_cpu, backend="eager")
    device = torch.device(f"cuda:{rank}")
    start, end = time_shard_bounds(time, rank, world)
    got = scan_diag_context_parallel(
        jac_cpu[:, start:end].to(device).contiguous(),
        residual_cpu[:, start:end].to(device).contiguous(),
        backend="eager",
    )
    got_cpu = got.detach().cpu()
    err = (got_cpu - ref[:, start:end]).abs().amax()
    err_t = err.reshape(1).to(device)
    errs = [torch.zeros(1, device=device, dtype=err_t.dtype) for _ in range(world)]
    dist.all_gather(errs, err_t)
    gathered = all_gather_time_tiles(got, time, rank, world, None)
    if rank == 0:
        e0, e1 = float(errs[0].cpu()), float(errs[1].cpu())
        torch.testing.assert_close(gathered.cpu(), ref, atol=_ATOL, rtol=_RTOL)
        assert e0 < 1e-4, f"rank0 max_abs={e0}"
        assert e1 < 1e-4, f"rank1 max_abs={e1} (carry path)"
    dist.destroy_process_group()


@pytest.mark.parametrize("time", [1, 2, 8, 17, 64])
def test_context_parallel_cpu_rank1_matches_full_scan(time: int) -> None:
    port = _free_port()
    mp.spawn(_cpu_fwd_worker, args=(2, port, time), nprocs=2, join=True)


def test_context_parallel_cpu_grads_match_scan_diag() -> None:
    port = _free_port()
    mp.spawn(_cpu_grad_worker, args=(2, port), nprocs=2, join=True)


def test_reverse_context_parallel_cpu_matches() -> None:
    port = _free_port()
    mp.spawn(_cpu_reverse_worker, args=(2, port), nprocs=2, join=True)


def test_newton_context_parallel_cpu_matches_eager() -> None:
    port = _free_port()
    mp.spawn(_cpu_newton_worker, args=(2, port), nprocs=2, join=True)


def test_newton_context_parallel_cpu_grads_match_eager() -> None:
    port = _free_port()
    mp.spawn(_cpu_newton_grad_worker, args=(2, port), nprocs=2, join=True)


@pytest.mark.cuda
def test_context_parallel_cuda_rank1_concat_vs_cpu_scan() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    port = _free_port()
    mp.spawn(_cuda_concat_worker, args=(2, port), nprocs=2, join=True)
