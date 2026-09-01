"""Ablate serial tile scan and R mixing vs FlashRNN.

  uv run python scripts/bench_slstm_tiled.py

2080 Ti by name. 10 warmup / 50 runs, min ms. MLflow newton-slstm-bench.
mix='diag' is already the fused R (not a new kernel). Head mix is eager.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
from torch import Tensor

from pararnn.cells import ParaSLSTM
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, setup_logging, wait_until_free

log = logging.getLogger("bench")

WARMUP = 10
N_RUNS = 50
BATCH = 8
D_H = 256


def _cuda_minmax(fn, *, warmup: int, n_runs: int) -> tuple[float, float, float]:
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


def _time(name: str, fn, T: int) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    tmin, tmed, tmean = _cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    log.info(
        "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB  T=%d",
        name,
        tmin,
        tmed,
        tmean,
        mem,
        T,
    )
    return tmin


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    if device.type != "cuda":
        raise RuntimeError("needs the 2080 Ti")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=8.0, poll_s=30.0)
    log.info("bench_slstm_tiled gpu=%s", torch.cuda.get_device_name(device))

    import mlflow

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name="slstm-tiled-ablation"):
        mlflow.set_tags({"gpu": torch.cuda.get_device_name(device), "protocol": "smoke-10-50"})
        variants = (
            ("baseline", NewtonConfig(scan_backend="fused", residual_atol=None)),
            (
                "scan_seq",
                NewtonConfig(
                    scan_backend="fused", residual_atol=None, scan_tile="seq"
                ),
            ),
        )
        for T in (256, 2048):
            torch.manual_seed(0)
            cell = ParaSLSTM(D_H, D_H, mix="diag").to(device).eval()
            x = torch.randn(BATCH, T, D_H, device=device)
            with torch.no_grad():
                seq = sequential_apply(cell, x)
            for name, cfg in variants:
                cfg = replace(cfg, max_iters=3)
                with torch.no_grad():
                    par = newton_apply(cell, x, cfg)
                    err = float((par - seq).abs().amax())
                log.info("%s T=%d max|par-seq|=%.3e", name, T, err)
                mlflow.log_metric(f"{name}_err", err, step=T)
                tmin = _time(
                    f"ParaSLSTM {name}",
                    lambda c=cell, xx=x, k=cfg: newton_apply(c, xx, k),
                    T,
                )
                mlflow.log_metric(f"{name}_min_ms", tmin, step=T)

            if T == 256:
                torch.manual_seed(0)
                head = ParaSLSTM(D_H, D_H, mix="head", n_heads=8).to(device).eval()
                xh = torch.randn(2, 16, D_H, device=device)
                cfg_h = NewtonConfig(
                    max_iters=3,
                    scan_backend="eager",
                    residual_atol=None,
                    picard_iters=0,
                    residual_fail=None,
                )
                tmin = _time(
                    "ParaSLSTM mix=head eager T=16 B=2",
                    lambda c=head, xx=xh, k=cfg_h: newton_apply(c, xx, k),
                    16,
                )
                mlflow.log_metric("head_eager_min_ms", tmin, step=16)

            try:
                import os
                from pathlib import Path

                if not os.environ.get("CUDA_HOME"):
                    nvidia = Path(torch.__file__).resolve().parent.parent / "nvidia"
                    runtime = nvidia / "cuda_runtime"
                    if (runtime / "include" / "cuda.h").is_file():
                        os.environ["CUDA_HOME"] = str(runtime)
                from flashrnn import flashrnn

                n_heads, d_head = 8, 32

                def _fr(
                    xx: Tensor = x,
                    seq_len: int = T,
                    nh: int = n_heads,
                    dh: int = d_head,
                ) -> None:
                    wx5 = xx.new_empty(BATCH, seq_len, 4, nh, dh).normal_()
                    rec = xx.new_empty(4, nh, dh, dh).normal_().mul_(0.1)
                    bias = xx.new_zeros(4, nh, dh)
                    s0 = xx.new_zeros(4, BATCH, 1, nh, dh)
                    flashrnn(
                        wx5,
                        rec,
                        bias,
                        states=s0,
                        function="slstm",
                        backend="triton_fused",
                        dtype="float32",
                    )

                tmin = _time("flashrnn_triton_fused", _fr, T)
                mlflow.log_metric("flashrnn_min_ms", tmin, step=T)
            except (ImportError, RuntimeError, OSError) as exc:
                log.warning("flashrnn skip: %s", exc)


if __name__ == "__main__":
    main()
