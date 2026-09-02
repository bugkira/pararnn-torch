"""CUDA event timing for lab benches. Not shipped."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)


def cuda_minmax(fn: Callable[[], None], *, warmup: int, n_runs: int) -> tuple[float, float, float]:
    """Return (min, median, mean) milliseconds from CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    mean = sum(samples) / len(samples)
    median = samples[len(samples) // 2]
    return samples[0], median, mean


def cuda_min_ms(fn: Callable[[], None], *, warmup: int, n_runs: int) -> float:
    return cuda_minmax(fn, warmup=warmup, n_runs=n_runs)[0]


def peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_cuda_peak() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def time_forward(
    name: str,
    fn: Callable[[], None],
    *,
    warmup: int,
    n_runs: int,
    seq_len: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    reset_cuda_peak()
    fn()
    torch.cuda.synchronize()
    tmin, tmed, tmean = cuda_minmax(fn, warmup=warmup, n_runs=n_runs)
    mem = peak_mib()
    lg = logger or log
    if seq_len is not None:
        lg.info(
            "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB  T=%d",
            name,
            tmin,
            tmed,
            tmean,
            mem,
            seq_len,
        )
    else:
        lg.info(
            "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB",
            name,
            tmin,
            tmed,
            tmean,
            mem,
        )
    return {"min_ms": tmin, "median_ms": tmed, "mean_ms": tmean, "peak_mib": mem}
