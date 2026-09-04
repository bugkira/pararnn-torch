"""Context-parallel diag scan: Rank 1 carry check, then T=8192/32768 timing.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      uv run torchrun --nproc_per_node=2 examples/context_parallel.py

Agreement uses **eager** local scans so Rank 1's AllGather carry is compared
to a CPU ``scan_diag`` of the full system (Rank 0's carry is identically 0).
Timing uses ``backend=auto`` (Triton on CUDA). Single-GPU baseline runs on
rank 0 only. ``B=8``, ``d=256``. Local smoke: no MLflow.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.solvers.scan import scan_diag
from pararnn.solvers.seq_parallel import scan_diag_context_parallel, time_shard_bounds

log = logging.getLogger("context_parallel")

_AGREE_T = 2048
_BENCH_TS = (8192, 32768)
_BATCH = 8
_DIM = 256
_WARMUP = 3
_RUNS = 10


def _tile_backend(x: torch.Tensor) -> str:
    return "triton" if is_fused_dtype_supported(x.dtype, x.device) else "eager"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _minmax_ms(fn, device: torch.device, *, warmup: int, n_runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), float(sum(samples) / len(samples))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    if "RANK" not in os.environ:
        raise SystemExit(
            "Launch with torchrun, e.g. "
            "uv run torchrun --nproc_per_node=2 examples/context_parallel.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl", device_id=device)
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")

    gpu_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    torch.manual_seed(0)
    jac_cpu = torch.randn(_BATCH, _AGREE_T, _DIM) * 0.3
    residual_cpu = torch.randn(_BATCH, _AGREE_T, _DIM)
    ref = scan_diag(jac_cpu, residual_cpu, backend="eager")
    start, end = time_shard_bounds(_AGREE_T, rank, world)
    got = scan_diag_context_parallel(
        jac_cpu[:, start:end].to(device).contiguous(),
        residual_cpu[:, start:end].to(device).contiguous(),
        backend="eager",
    )
    err = float((got.detach().cpu() - ref[:, start:end]).abs().amax())
    err_t = torch.tensor([err], device=device)
    errs = [torch.zeros(1, device=device) for _ in range(world)]
    dist.all_gather(errs, err_t)
    if rank == 0:
        log.info(
            "agree T=%s B=%s d=%s rank0_maxabs=%s rank1_maxabs=%s (rank1 is the carry path)",
            _AGREE_T,
            _BATCH,
            _DIM,
            float(errs[0].cpu()),
            float(errs[1].cpu()) if world > 1 else float("nan"),
        )

    for seq_len in _BENCH_TS:
        s, e = time_shard_bounds(seq_len, rank, world)
        local_t = e - s
        torch.manual_seed(seq_len * 17 + rank)
        jac_loc = (torch.randn(_BATCH, local_t, _DIM, device=device) * 0.3).contiguous()
        res_loc = torch.randn(_BATCH, local_t, _DIM, device=device).contiguous()
        tile_backend = _tile_backend(jac_loc)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def _cp(
            j: torch.Tensor = jac_loc,
            r: torch.Tensor = res_loc,
            b: str = tile_backend,
        ) -> None:
            scan_diag_context_parallel(j, r, backend=b)

        cp_min, cp_mean = _minmax_ms(_cp, device, warmup=_WARMUP, n_runs=_RUNS)
        peak_mib = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        )
        log.info(
            "bench_cp T=%s rank=%s gpu=%s backend=%s local_t=%s "
            "min_ms=%.3f mean_ms=%.3f peak_MiB=%.1f",
            seq_len,
            rank,
            gpu_name,
            tile_backend,
            local_t,
            cp_min,
            cp_mean,
            peak_mib,
        )
        dist.barrier()
        if rank == 0:
            torch.manual_seed(seq_len)
            jac_full = (torch.randn(_BATCH, seq_len, _DIM, device=device) * 0.3).contiguous()
            res_full = torch.randn(_BATCH, seq_len, _DIM, device=device).contiguous()
            full_backend = _tile_backend(jac_full)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def _full(
                j: torch.Tensor = jac_full,
                r: torch.Tensor = res_full,
                b: str = full_backend,
            ) -> None:
                scan_diag(j, r, backend=b)

            full_min, full_mean = _minmax_ms(_full, device, warmup=_WARMUP, n_runs=_RUNS)
            full_peak = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            log.info(
                "bench_single T=%s gpu=%s backend=%s min_ms=%.3f mean_ms=%.3f peak_MiB=%.1f",
                seq_len,
                gpu_name,
                full_backend,
                full_min,
                full_mean,
                full_peak,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
