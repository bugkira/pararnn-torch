#!/usr/bin/env python3
"""Latency: ParaGRU mix='head' Newton vs sequential unroll.

Dreamer-style geometry: d_h=512, n_heads=8 (d_head=64). Grid over T and
optional d_head. On CUDA, compares fused (factorized J) vs eager (dense J).

  uv run python scripts/bench_gru_head.py
  uv run python scripts/bench_gru_head.py --device cuda --T 64,128,256
  uv run python scripts/bench_gru_head.py --d-head-grid 8,64,256
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch

from pararnn import NewtonConfig, ParaGRU, newton_apply, sequential_apply

log = logging.getLogger("bench_gru_head")


def _median_ms(fn, *, warmup: int, repeats: int, sync_cuda: bool) -> float:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _run_row(
    *,
    cell: ParaGRU,
    x: torch.Tensor,
    backends: list[str],
    max_iters: int,
    warmup: int,
    repeats: int,
    sync: bool,
) -> None:
    ms_s = _median_ms(
        lambda: sequential_apply(cell, x),
        warmup=warmup,
        repeats=repeats,
        sync_cuda=sync,
    )
    ref = sequential_apply(cell, x)
    d_head = cell.d_head
    for backend in backends:
        cfg = NewtonConfig(max_iters=max_iters, scan_backend=backend, residual_atol=None)
        ms_n = _median_ms(
            lambda cfg=cfg: newton_apply(cell, x, cfg),
            warmup=warmup,
            repeats=repeats,
            sync_cuda=sync,
        )
        with torch.no_grad():
            err = float((newton_apply(cell, x, cfg) - ref).abs().amax())
        log.info(
            "T=%d d_head=%d backend=%-6s newton=%.2fms seq=%.2fms ratio=%.2f max|err|=%.2e",
            x.shape[1],
            d_head,
            backend,
            ms_n,
            ms_s,
            ms_n / ms_s,
            err,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-h", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument(
        "--d-head-grid",
        default="",
        help="Comma list of d_head; overrides --d-h/--n-heads (n_heads=2 each)",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--T", default="32,64,128,256")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument(
        "--scan-backend",
        default="auto",
        help="Single backend, or 'compare' for fused+eager on CUDA",
    )
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(args.device)
    lengths = [int(t) for t in args.T.split(",") if t.strip()]
    sync = device.type == "cuda"

    if args.scan_backend == "compare":
        backends = ["fused", "eager"] if device.type == "cuda" else ["eager"]
    else:
        backends = [args.scan_backend]

    geometries: list[tuple[int, int]]
    if args.d_head_grid.strip():
        geometries = [(2, int(d)) for d in args.d_head_grid.split(",") if d.strip()]
    else:
        geometries = [(args.n_heads, args.d_h // args.n_heads)]
        if args.d_h % args.n_heads != 0:
            raise SystemExit(f"d_h={args.d_h} not divisible by n_heads={args.n_heads}")

    # Paper K=3 (App. A). Factorized head path avoids dense J; measure residual
    # vs sequential and raise K only if max|err| stays above 1e-3 on this grid.
    for n_heads, d_head in geometries:
        d_h = n_heads * d_head
        cell = ParaGRU(d_h, d_h, mix="head", n_heads=n_heads, device=device).eval()
        for t in lengths:
            x = 0.3 * torch.randn(args.batch, t, d_h, device=device)
            _run_row(
                cell=cell,
                x=x,
                backends=backends,
                max_iters=args.max_iters,
                warmup=args.warmup,
                repeats=args.repeats,
                sync=sync,
            )

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("bench_gru_head")
        with mlflow.start_run(run_name=f"head_compare_{device.type}"):
            mlflow.log_params(
                {
                    "batch": args.batch,
                    "max_iters": args.max_iters,
                    "scan_backend": args.scan_backend,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "T": args.T,
                    "d_head_grid": args.d_head_grid or f"{args.d_h // args.n_heads}",
                }
            )


if __name__ == "__main__":
    main()
