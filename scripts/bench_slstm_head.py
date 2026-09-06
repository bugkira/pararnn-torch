#!/usr/bin/env python3
"""Latency + asymptotics: ParaSLSTM mix='head' Newton vs sequential.

Tiers: ``d_head ≤ 32`` fused SRAM; ``32 < d ≤ 128`` streamed-``R``; larger
factorized eager. Head recipe often ``K=4`` (measured snaps).

  uv run python scripts/bench_slstm_head.py
  uv run python scripts/bench_slstm_head.py --device cuda --T 64,128,256,512
  uv run python scripts/bench_slstm_head.py --d-head-grid 16,32,48,64,96,128
"""

from __future__ import annotations

import argparse
import itertools
import logging
import statistics
import time

import torch

from pararnn import NewtonConfig, ParaSLSTM, newton_apply, sequential_apply

log = logging.getLogger("bench_slstm_head")


def _median_ms(fn, *, warmup: int, repeats: int, sync_cuda: bool) -> float:
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _tier(d_head: int) -> str:
    if d_head <= 32:
        return "fused"
    if d_head <= 128:
        return "stream"
    return "eager_factor"


def _dense_j_bytes(batch: int, time: int, d_h: int, dtype: torch.dtype) -> int:
    """Rough ``(B, T, 4 d_h, 4 d_h)`` Jacobian footprint for eager oracle."""
    s = 4 * d_h
    return batch * time * s * s * dtype.itemsize


def _run_row(
    *,
    cell: ParaSLSTM,
    x: torch.Tensor,
    backends: list[str],
    max_iters: int,
    warmup: int,
    repeats: int,
    sync: bool,
) -> dict[str, float]:
    ms_s = _median_ms(
        lambda: sequential_apply(cell, x),
        warmup=warmup,
        repeats=repeats,
        sync_cuda=sync,
    )
    ref = sequential_apply(cell, x)
    d_head = int(cell.d_head or 0)
    d_h = int(cell.d_h)
    out: dict[str, float] = {"seq_ms": ms_s}
    for backend in backends:
        if backend == "eager" and x.is_cuda:
            need = _dense_j_bytes(x.shape[0], x.shape[1], d_h, torch.float32)
            free, _total = torch.cuda.mem_get_info()
            if need > 0.45 * free:
                log.warning(
                    "T=%4d d_head=%3d skip eager (dense J ~%.2f GiB, free %.2f GiB)",
                    x.shape[1],
                    d_head,
                    need / (1024**3),
                    free / (1024**3),
                )
                continue
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
            "T=%4d d_head=%3d tier=%-12s backend=%-6s newton=%7.2fms seq=%7.2fms "
            "ratio=%.2f max|err|=%.2e",
            x.shape[1],
            d_head,
            _tier(d_head),
            backend,
            ms_n,
            ms_s,
            ms_n / ms_s,
            err,
        )
        out[f"{backend}_ms"] = ms_n
        out[f"{backend}_err"] = err
    return out


def _log_t_scaling(rows: list[tuple[int, float]], label: str) -> None:
    """Print pairwise T-doubling factors (expect ~2 for O(T))."""
    if len(rows) < 2:
        return
    parts = []
    for (t0, m0), (t1, m1) in itertools.pairwise(rows):
        if m0 <= 0:
            continue
        parts.append(f"T{t0}->{t1}: ×{m1 / m0:.2f} (T×{t1 / t0:.1f})")
    log.info("asymptotics %s  %s", label, " | ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-h", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument(
        "--d-head-grid",
        default="",
        help="Comma list of d_head; overrides --d-h/--n-heads (n_heads=2 each)",
    )
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--T", default="64,128,256,512,1024")
    p.add_argument(
        "--max-iters",
        type=int,
        default=4,
        help="Head snaps often K=4 (docs/internal/para-slstm.md)",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument(
        "--scan-backend",
        default="compare",
        help="Single backend, or 'compare' for fused+eager on CUDA",
    )
    p.add_argument("--mlflow", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = torch.device(args.device)
    lengths = [int(t) for t in args.T.split(",") if t.strip()]
    sync = device.type == "cuda"
    if device.type == "cuda":
        log.info("gpu=%s", torch.cuda.get_device_name(0))

    if args.scan_backend == "compare":
        backends = ["fused", "eager"] if device.type == "cuda" else ["eager"]
    else:
        backends = [args.scan_backend]

    if args.d_head_grid.strip():
        geometries = [(2, int(d)) for d in args.d_head_grid.split(",") if d.strip()]
    else:
        if args.d_h % args.n_heads != 0:
            raise SystemExit(f"d_h={args.d_h} not divisible by n_heads={args.n_heads}")
        geometries = [(args.n_heads, args.d_h // args.n_heads)]

    # K=4: head snaps (para-slstm.md). Raise K only if max|err| > 1e-3 on this grid.
    scaling: dict[str, list[tuple[int, float]]] = {}
    for n_heads, d_head in geometries:
        d_h = n_heads * d_head
        d_in = min(d_h, 32)
        cell = ParaSLSTM(d_in, d_h, mix="head", n_heads=n_heads, device=device).eval()
        for t in lengths:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            x = 0.2 * torch.randn(args.batch, t, d_in, device=device)
            row = _run_row(
                cell=cell,
                x=x,
                backends=backends,
                max_iters=args.max_iters,
                warmup=args.warmup,
                repeats=args.repeats,
                sync=sync,
            )
            primary = next((b for b in backends if f"{b}_ms" in row), None)
            if primary is not None:
                key = f"d{d_head}_{primary}"
                scaling.setdefault(key, []).append((t, row[f"{primary}_ms"]))
            scaling.setdefault(f"d{d_head}_seq", []).append((t, row["seq_ms"]))
        del cell
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for label, series in scaling.items():
        _log_t_scaling(series, label)

    if args.mlflow:
        import mlflow

        mlflow.set_experiment("bench_slstm_head")
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
