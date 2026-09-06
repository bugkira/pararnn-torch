#!/usr/bin/env python3
"""M²RNN factorized Newton: residual vs K and wall time vs sequential.

Recipe start from measured snaps (docs/internal/m2rnn-jacobian.md):
``max_iters=4``, ``omega=1``. Raise K only if residual plateaus above 1e-4.

  uv run python scripts/bench_m2rnn.py
  uv run python scripts/bench_m2rnn.py --device cuda --shapes 4x4,8x8,16x16
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch

from pararnn import NewtonConfig, NewtonStats, ParaM2RNN, newton_apply, sequential_apply

log = logging.getLogger("bench_m2rnn")


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


def _parse_shapes(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip().lower()
        if "x" not in part:
            raise ValueError(f"shape must look like 8x8, got {part!r}")
        a, b = part.split("x", 1)
        out.append((int(a), int(b)))
    return out


def residual_curve(
    cell: ParaM2RNN,
    x: torch.Tensor,
    *,
    ks: list[int],
) -> list[tuple[int, float]]:
    ref = sequential_apply(cell, x)
    rows = []
    with torch.no_grad():
        for k in ks:
            cfg = NewtonConfig(max_iters=k, omega=1.0, residual_atol=None)
            par = newton_apply(cell, x, cfg)
            err = float((par - ref).abs().amax())
            rows.append((k, err))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--shapes", default="4x4,8x8,16x16")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", default="32,64")
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--ks", default="0,1,2,3,4,6,8")
    p.add_argument("--max-iters", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scan-backend",
        default="auto",
        help="'auto'/'fused'/'eager', or 'compare' for fused+eager on CUDA",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(args.device)
    sync = device.type == "cuda"
    shapes = _parse_shapes(args.shapes)
    ts = [int(t) for t in args.T.split(",") if t.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    if args.scan_backend == "compare":
        backends = ["fused", "eager"] if sync else ["eager"]
    else:
        backends = [args.scan_backend]

    log.info(
        "device=%s dtype=float32 seed=%d shapes=%s T=%s backends=%s",
        device,
        args.seed,
        shapes,
        ts,
        backends,
    )
    if sync:
        log.info("gpu=%s", torch.cuda.get_device_name(device))

    for k_dim, v_dim in shapes:
        for t in ts:
            torch.manual_seed(args.seed)
            cell = ParaM2RNN(d_in=args.d_in, k_dim=k_dim, v_dim=v_dim).to(device).eval()
            x = (0.15 * torch.randn(args.batch, t, args.d_in, device=device)).detach()
            curve = residual_curve(cell, x, ks=ks)
            curve_s = " ".join(f"K={k}:{e:.2e}" for k, e in curve)
            log.info("residual KxV=%dx%d T=%d | %s", k_dim, v_dim, t, curve_s)

            ms_s = _median_ms(
                lambda: sequential_apply(cell, x),
                warmup=args.warmup,
                repeats=args.repeats,
                sync_cuda=sync,
            )
            ref = sequential_apply(cell, x)
            for backend in backends:
                cfg = NewtonConfig(
                    max_iters=args.max_iters,
                    omega=1.0,
                    scan_backend=backend,
                    residual_atol=None,
                    residual_fail=None,
                )
                st = NewtonStats()
                ms_n = _median_ms(
                    lambda cfg=cfg: newton_apply(cell, x, cfg, stats=st),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    sync_cuda=sync,
                )
                with torch.no_grad():
                    err = float((newton_apply(cell, x, cfg) - ref).abs().amax())
                log.info(
                    "time    KxV=%dx%d T=%d backend=%-6s newton=%.2fms seq=%.2fms "
                    "ratio=%.2f max|err|=%.2e residual=%.2e tag=%s",
                    k_dim,
                    v_dim,
                    t,
                    backend,
                    ms_n,
                    ms_s,
                    ms_n / ms_s,
                    err,
                    st.max_residual,
                    st.scan_backend,
                )


if __name__ == "__main__":
    main()
