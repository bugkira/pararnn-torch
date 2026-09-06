#!/usr/bin/env python3
"""Measure M²RNN Newton K*(T) and fit asymptotics.

Critical depth ``K* = min{K : max|H_par − H_seq|_∞ < τ}``.

Inits:
  app_a     — Danieli App. A parallel guess (zero prev, full W).
  frozen_w  — ``W=0`` linear scan warm-start (``picard_iters=1``), then Newton.

Hypotheses (literature):
  H1  K* = O(1)     — Danieli et al. ParaRNN: GRU/LSTM snap at K≈3.
  H2  K* = Θ(log T) — Gonzalez thesis / PL rate.
  H3  K* = Θ(√T)    — empirical rival on moderate T grids.
  H0  K* = Θ(T)     — DEER global bound.

  uv run python scripts/bench_m2rnn_k_scale.py
  uv run python scripts/bench_m2rnn_k_scale.py --init frozen_w --out outputs/m2rnn_k_scale_fw.csv
  uv run python scripts/bench_m2rnn_k_scale.py --init both --time
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from pathlib import Path

import torch

from pararnn import NewtonConfig, ParaM2RNN, newton_apply, sequential_apply
from pararnn.kernels.m2rnn_factor import m2rnn_frozen_w_scan
from pararnn.kernels.newton_m2rnn import newton_m2rnn_factorized

log = logging.getLogger("bench_m2rnn_k_scale")
ROOT = Path(__file__).resolve().parents[1]


def _cfg(k: int, *, frozen_w: bool) -> NewtonConfig:
    return NewtonConfig(
        max_iters=k,
        omega=1.0,
        scan_backend="fused",
        residual_atol=None,
        residual_fail=None,
        picard_iters=1 if frozen_w else 0,
    )


def _k_star(
    cell: ParaM2RNN,
    x: torch.Tensor,
    *,
    tau: float,
    k_max: int,
    frozen_w: bool,
) -> tuple[int | None, float]:
    """Binary search smallest K with max|par−seq| < tau. None if censored."""
    ref = sequential_apply(cell, x)
    lo, hi = 0, k_max
    best: int | None = None
    last_err = float("nan")
    k = 1
    while k <= k_max:
        with torch.no_grad():
            err = float((newton_apply(cell, x, _cfg(k, frozen_w=frozen_w)) - ref).abs().amax())
        last_err = err
        if err < tau:
            best = k
            hi = k
            break
        k = k * 2 if k < k_max else k_max + 1
        if k > k_max:
            break
    if best is None:
        return None, last_err
    lo = best // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        with torch.no_grad():
            err = float((newton_apply(cell, x, _cfg(mid, frozen_w=frozen_w)) - ref).abs().amax())
        if err < tau:
            hi = mid
            best = mid
            last_err = err
        else:
            lo = mid
    return best, last_err


def _init_err(cell: ParaM2RNN, x: torch.Tensor, *, frozen_w: bool) -> float:
    """max|H_guess − H_seq| before any Newton step."""
    with torch.no_grad():
        ref = sequential_apply(cell, x)
        if frozen_w:
            k_dim, v_dim = cell.k_dim, cell.v_dim
            wx = cell.W_x(x)
            k = wx[..., :k_dim]
            v = wx[..., k_dim : k_dim + v_dim]
            f = torch.sigmoid(wx[..., -1])
            guess = m2rnn_frozen_w_scan(k, v, f, h0=None)
        else:
            guess = newton_m2rnn_factorized(cell, x, max_iters=0, omega=1.0)
        return float((guess - ref).detach().abs().amax())


def _ols(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return a, b, R² for y ≈ a + b x."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=False))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y, strict=False))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def _fit_report(rows: list[dict], *, label: str) -> None:
    finite = [r for r in rows if r["Kstar"] is not None]
    if len(finite) < 4:
        log.warning("[%s] too few finite K* points (%d) for fits", label, len(finite))
        return
    by_t: dict[int, list[int]] = {}
    for r in finite:
        by_t.setdefault(int(r["T"]), []).append(int(r["Kstar"]))
    ts = sorted(by_t)
    means = [sum(by_t[t]) / len(by_t[t]) for t in ts]
    log.info(
        "[%s] mean K*(T): %s",
        label,
        " ".join(f"T={t}:{m:.2f}" for t, m in zip(ts, means, strict=False)),
    )

    my = sum(means) / len(means)
    mae_const = sum(abs(m - my) for m in means) / len(means)
    log.info("[%s] H1 O(1): mean K*=%.3f  MAE_vs_means=%.3f", label, my, mae_const)

    feats = {
        "log2T": [math.log2(t) for t in ts],
        "sqrtT": [math.sqrt(t) for t in ts],
        "T": [float(t) for t in ts],
    }
    for name, xs in feats.items():
        a, b, r2 = _ols(xs, means)
        mae = sum(abs(means[i] - (a + b * xs[i])) for i in range(len(ts))) / len(ts)
        log.info(
            "[%s] fit %s: K*≈%.3f%+.4f*%s  R²=%.3f  MAE=%.3f",
            label,
            name,
            a,
            b,
            name,
            r2,
            mae,
        )

    ys = [float(r["Kstar"]) for r in finite]
    for name, xf in (
        ("log2T", lambda r: math.log2(int(r["T"]))),
        ("sqrtT", lambda r: math.sqrt(int(r["T"]))),
        ("T", lambda r: float(r["T"])),
    ):
        xs = [xf(r) for r in finite]
        a, b, r2 = _ols(xs, ys)
        log.info(
            "[%s] seed-level %s: a=%.3f b=%.4f R²=%.3f N=%d",
            label,
            name,
            a,
            b,
            r2,
            len(ys),
        )


def _time_at_kstar(
    cell: ParaM2RNN,
    x: torch.Tensor,
    *,
    kstar: int,
    frozen_w: bool,
    repeats: int = 20,
) -> tuple[float, float]:
    """Median ms for Newton at K* (early-stop off) and sequential."""
    device = x.device
    cfg = _cfg(kstar, frozen_w=frozen_w)
    # warmup
    with torch.no_grad():
        newton_apply(cell, x, cfg)
        sequential_apply(cell, x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    def _med(fn) -> float:
        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - t0) * 1e3)
        samples.sort()
        return samples[len(samples) // 2]

    ms_par = _med(lambda: newton_apply(cell, x, cfg))
    ms_seq = _med(lambda: sequential_apply(cell, x))
    return ms_par, ms_seq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k-dim", type=int, default=16)
    p.add_argument("--v-dim", type=int, default=16)
    p.add_argument("--d-in", type=int, default=32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--T", default="32,64,128,256,512,1024,2048,4096")
    p.add_argument("--seeds", default="0,1,2,3")
    p.add_argument("--tau", type=float, default=1e-4)
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--x-scale", type=float, default=0.15)
    p.add_argument(
        "--init",
        choices=("app_a", "frozen_w", "both"),
        default="both",
        help="Warm-start: App. A, frozen-W, or both (side-by-side).",
    )
    p.add_argument(
        "--time",
        action="store_true",
        help="Also time Newton@K* vs sequential (median ms).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV path (default outputs/m2rnn_k_scale[_fw].csv).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    ts = [int(t) for t in args.T.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    inits = ("app_a", "frozen_w") if args.init == "both" else (args.init,)
    if args.out is None:
        suffix = {"app_a": "", "frozen_w": "_fw", "both": "_compare"}[args.init]
        args.out = ROOT / "outputs" / f"m2rnn_k_scale{suffix}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "M²RNN K*(T)  device=%s kv=%dx%d tau=%.1e k_max=%d T=%s seeds=%s init=%s",
        device,
        args.k_dim,
        args.v_dim,
        args.tau,
        args.k_max,
        ts,
        seeds,
        args.init,
    )
    if device.type == "cuda":
        log.info("gpu=%s", torch.cuda.get_device_name(device))

    rows_by_init: dict[str, list[dict]] = {i: [] for i in inits}
    fields = [
        "init",
        "seed",
        "T",
        "k_dim",
        "v_dim",
        "tau",
        "Kstar",
        "censored",
        "err_at_kmax",
        "init_err",
        "ms_par",
        "ms_seq",
        "par_over_seq",
    ]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for seed in seeds:
            for t in ts:
                torch.manual_seed(seed)
                cell = (
                    ParaM2RNN(d_in=args.d_in, k_dim=args.k_dim, v_dim=args.v_dim).to(device).eval()
                )
                x = (args.x_scale * torch.randn(args.batch, t, args.d_in, device=device)).detach()
                for init in inits:
                    frozen = init == "frozen_w"
                    kstar, err = _k_star(cell, x, tau=args.tau, k_max=args.k_max, frozen_w=frozen)
                    init_err = _init_err(cell, x, frozen_w=frozen)
                    ms_par = ms_seq = par_over = ""
                    if args.time and kstar is not None:
                        ms_p, ms_s = _time_at_kstar(cell, x, kstar=kstar, frozen_w=frozen)
                        ms_par = f"{ms_p:.3f}"
                        ms_seq = f"{ms_s:.3f}"
                        par_over = f"{ms_p / ms_s:.3f}"
                    censored = 0 if kstar is not None else 1
                    row = {
                        "init": init,
                        "seed": seed,
                        "T": t,
                        "k_dim": args.k_dim,
                        "v_dim": args.v_dim,
                        "tau": args.tau,
                        "Kstar": kstar if kstar is not None else "",
                        "censored": censored,
                        "err_at_kmax": f"{err:.6e}",
                        "init_err": f"{init_err:.6e}",
                        "ms_par": ms_par,
                        "ms_seq": ms_seq,
                        "par_over_seq": par_over,
                    }
                    w.writerow(row)
                    rows_by_init[init].append({**row, "Kstar": kstar})
                    log.info(
                        "init=%s seed=%d T=%d K*=%s init_err=%.3e censored=%d%s",
                        init,
                        seed,
                        t,
                        kstar if kstar is not None else f">{args.k_max}",
                        init_err,
                        censored,
                        (f"  par={ms_par}ms seq={ms_seq}ms ({par_over}x)" if ms_par else ""),
                    )

    for init, rows in rows_by_init.items():
        _fit_report(rows, label=init)

    if len(inits) == 2:
        # Pairwise ΔK* on matching (seed,T)
        a_map = {
            (int(r["seed"]), int(r["T"])): r["Kstar"]
            for r in rows_by_init["app_a"]
            if r["Kstar"] is not None
        }
        deltas = []
        for r in rows_by_init["frozen_w"]:
            if r["Kstar"] is None:
                continue
            key = (int(r["seed"]), int(r["T"]))
            if key in a_map:
                deltas.append(int(a_map[key]) - int(r["Kstar"]))
        if deltas:
            log.info(
                "ΔK* (app_a − frozen_w): mean=%.2f  median=%.1f  min=%d  max=%d  N=%d",
                sum(deltas) / len(deltas),
                sorted(deltas)[len(deltas) // 2],
                min(deltas),
                max(deltas),
                len(deltas),
            )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
