"""Forward timing: naive sequential RNN vs Newton+scan ParaRNN.

Protocol: Danieli et al. 2025 App. B (CUDA Events, 20 warmup, 100 runs, min).
GPU: 2080 Ti by name. Config: --config (default configs/bench/cell_forward.yaml).

  uv run python scripts/bench_time.py
  uv run python scripts/bench_time.py --config configs/bench/newton_compile.yaml
  uv run python scripts/bench_time.py --config configs/bench/newton_fused.yaml
  uv run python scripts/bench_time.py --config configs/bench/newton_slstm.yaml
  uv run python scripts/bench_time.py --config configs/bench/newton_fp16.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from pararnn import device, wait_until_free
from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.logconf import setup_logging
from pararnn.solvers import (
    NewtonConfig,
    newton_apply,
    sequential_apply,
    sequential_apply_compiled,
)

log = logging.getLogger("bench")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bench" / "cell_forward.yaml"

# torch.compile of newton_apply at the call site — not in src/. Keyed by
# cell identity + T because reduce-overhead CUDA graphs are shape-static.
_compiled_newton: dict[tuple[int, str, int], Callable[[Tensor], Tensor]] = {}


def _cuda_minmax(fn, *, warmup: int, n_runs: int) -> tuple[float, float, float]:
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


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


def _lock_hash() -> str:
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return "none"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def _torch_dtype(name: str) -> torch.dtype:
    """YAML dtype. fp16 is Turing TC; bf16 is rejected (no bf16 TC)."""
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        raise ValueError(
            "bfloat16 is not used on Turing (no bf16 tensor cores). Use float16."
        )
    raise ValueError(f"unsupported dtype {name!r}")


def _agree_tol(dtype: torch.dtype, spec: dict) -> float:
    if "agree_tol" in spec:
        return float(spec["agree_tol"])
    # fp16 vs sequential fp16: residual ~1e-3 (docs/bottlenecks.md). Not 1e-4 vs fp32.
    return 2e-3 if dtype is torch.float16 else 1e-4


def _build_cells(spec: dict, d_in: int, d_h: int, *, device, dtype: torch.dtype) -> dict[str, nn.Module]:
    """Default: ParaGRU + ParaLSTM. YAML ``cells`` can pin a subset (sLSTM)."""
    raw = spec.get("cells")
    if not raw:
        raw = [{"name": "ParaGRU"}, {"name": "ParaLSTM"}]
    out: dict[str, nn.Module] = {}
    for item in raw:
        if isinstance(item, str):
            item = {"name": item}
        name = str(item["name"])
        if name == "ParaGRU":
            out[name] = ParaGRU(d_in, d_h).to(device=device, dtype=dtype).eval()
        elif name == "ParaLSTM":
            out[name] = ParaLSTM(d_in, d_h).to(device=device, dtype=dtype).eval()
        elif name == "ParaSLSTM":
            mix = str(item.get("mix", "diag"))
            n_heads = item.get("n_heads")
            key = name if mix == "diag" else f"{name}_{mix}"
            kw = {"mix": mix}
            if n_heads is not None:
                kw["n_heads"] = int(n_heads)
            out[key] = ParaSLSTM(d_in, d_h, **kw).to(device=device, dtype=dtype).eval()
        else:
            raise ValueError(f"unknown bench cell {name!r}")
    return out


@torch.no_grad()
def _agree(cell: nn.Module, x: Tensor, cfg: NewtonConfig) -> float:
    naive = sequential_apply(cell, x)
    par = newton_apply(cell, x, cfg)
    return float((par.float() - naive.float()).abs().amax())


def _newton_eager(cell: nn.Module, x: Tensor, cfg: NewtonConfig) -> Tensor:
    """Forward-only Newton. App. B is a forward protocol."""
    with torch.no_grad():
        return newton_apply(cell, x, cfg)


def _newton_fused(cell: nn.Module, x: Tensor, cfg: NewtonConfig) -> Tensor:
    """Cell+J+scan Triton Newton. Opt-in; not the library default."""
    fused = NewtonConfig(
        max_iters=cfg.max_iters,
        omega=cfg.omega,
        scan_backend="fused",
        residual_atol=None,
    )
    with torch.no_grad():
        return newton_apply(cell, x, fused)


def _newton_compiled(
    cell: nn.Module,
    x: Tensor,
    cfg: NewtonConfig,
    *,
    mode: str,
) -> Tensor:
    """``torch.compile(newton_apply)`` for this cell and sequence length.

    Not a library API. Forward-only (``no_grad``): App. B is a forward
    protocol, and Dynamo does not compile the eq. 2.6 ``autograd.Function``.
    """
    key = (id(cell), mode, int(x.shape[1]))
    fn = _compiled_newton.get(key)
    if fn is None:

        def _fwd(
            xx: Tensor, c: nn.Module = cell, conf: NewtonConfig = cfg
        ) -> Tensor:
            return newton_apply(c, xx, conf)

        fn = torch.compile(_fwd, mode=mode)
        _compiled_newton[key] = fn
    torch.compiler.cudagraph_mark_step_begin()
    with torch.no_grad():
        return fn(x)


def _time_one(
    name: str,
    fn,
    x: Tensor,
    *,
    warmup: int,
    n_runs: int,
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    tmin, tmed, tmean = _cuda_minmax(fn, warmup=warmup, n_runs=n_runs)
    mem = _peak_mib()
    log.info(
        "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB  T=%d",
        name,
        tmin,
        tmed,
        tmean,
        mem,
        x.shape[1],
    )
    return {
        "min_ms": tmin,
        "median_ms": tmed,
        "mean_ms": tmean,
        "peak_mib": mem,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


_DEFAULT_MODES = ("sequential_eager", "sequential_compiled", "newton")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="App. B cell-forward timing")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config (default: configs/bench/cell_forward.yaml)",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    spec = yaml.safe_load(config_path.read_text())
    if device.type != "cuda":
        raise RuntimeError("App. B needs the 2080 Ti (PARARNN_DEVICE to override)")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=8.0, poll_s=30.0)
    newton_cfg = NewtonConfig(
        max_iters=int(spec["newton_iters"]),
        scan_backend="eager",
        residual_atol=None,
    )
    warmup, n_runs = int(spec["warmup"]), int(spec["n_runs"])
    batch, d_in, d_h = int(spec["batch"]), int(spec["d_in"]), int(spec["d_h"])
    seq_max_seq = int(spec["seq_lens_sequential_max"])
    newton_max_t = {k: int(v) for k, v in spec.get("newton_max_t", {}).items()}
    seq_compile_mode = str(spec.get("sequential_compile_mode", "reduce-overhead"))
    newton_compile_mode = str(spec.get("newton_compile_mode", "reduce-overhead"))
    modes = tuple(spec.get("modes", _DEFAULT_MODES))
    experiment = str(spec.get("mlflow_experiment", "cell-forward-bench"))
    run_name = str(spec.get("mlflow_run_name", "pararnn-vs-naive-rnn"))
    csv_rel = spec.get("csv", "outputs/bench_cell_forward.csv")
    if "newton_compiled" in modes or "sequential_compiled" in modes:
        # Each T is a new static shape. Dynamo's default recompile_limit=8
        # is exhausted by 9 seq_lens; _scan then silently runs eager
        # (GRU T=4096 and all of ParaLSTM in the first compile run).
        # 9 lengths × 2 cells × a few inner frames; 64 is the fallback
        # if we add lengths, not a paper hyperparameter.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 256

    dtype_names = [str(n) for n in spec.get("dtypes", [spec["dtype"]])]
    multi_dtype = len(dtype_names) > 1

    log.info(
        "bench_start gpu=%s torch=%s batch=%d d_h=%d K=%d dtypes=%s modes=%s",
        torch.cuda.get_device_name(device),
        torch.__version__,
        batch,
        d_h,
        newton_cfg.max_iters,
        ",".join(dtype_names),
        ",".join(modes),
    )

    import mlflow

    mlflow.set_experiment(experiment)
    rows: list[dict] = []
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "gpu": torch.cuda.get_device_name(device),
                "dtype": ",".join(dtype_names),
                "protocol": "smoke-10-50" if int(spec["warmup"]) != 20 else "danieli2025-appB",
                "modes": ",".join(modes),
            }
        )
        mlflow.log_params(
            {
                "newton_iters": newton_cfg.max_iters,
                "batch": batch,
                "d_in": d_in,
                "d_h": d_h,
                "warmup": warmup,
                "n_runs": n_runs,
                "git": _git_commit(),
                "uv_lock": _lock_hash(),
                "config": str(config_path.relative_to(ROOT)),
                "sequential_compile_mode": seq_compile_mode,
                "newton_compile_mode": newton_compile_mode,
                "modes": ",".join(modes),
                "dtypes": ",".join(dtype_names),
                "require_agreement": str(spec.get("require_agreement", True)),
                "x_scale": spec.get("x_scale", 1.0),
                "cells": ",".join(
                    i if isinstance(i, str) else i.get("name", "?")
                    for i in spec.get("cells", ["ParaGRU", "ParaLSTM"])
                ),
            }
        )
        mlflow.log_artifact(str(config_path))
        why = spec.get("why") or (
            "Naive sequential unroll (eager + torch.compile reduce-overhead) vs "
            "Newton+scan on 2080 Ti. Speedup is vs compiled sequential (#12). "
            "Newton backward is eq. 2.6 reverse scan, not through K iterates. "
            f"Protocol and shapes: {config_path.relative_to(ROOT)}."
        )
        mlflow.log_text(str(why), "why.txt")

        csv_path = ROOT / csv_rel
        try:
            for dtype_name in dtype_names:
                dt = _torch_dtype(dtype_name)
                agree_tol = _agree_tol(dt, spec)
                require_agreement = bool(spec.get("require_agreement", True))
                x_scale = float(spec.get("x_scale", 1.0))
                log.info(
                    "bench_dtype=%s agree_tol=%g require_agreement=%s x_scale=%g",
                    dtype_name,
                    agree_tol,
                    require_agreement,
                    x_scale,
                )

                def mkey(cell: str, rest: str, d=dtype_name) -> str:
                    return f"{cell}_{d}_{rest}" if multi_dtype else f"{cell}_{rest}"
                cells = _build_cells(spec, d_in, d_h, device=device, dtype=dt)
                for cell_name, cell in cells.items():
                    for T in spec["seq_lens"]:
                        x = x_scale * torch.randn(
                            batch, int(T), d_in, device=device, dtype=dt
                        )
                        seq_stats = None
                        seq_eager_stats = None
                        if T <= seq_max_seq:
                            if "newton" in modes or "newton_compiled" in modes:
                                err = _agree(cell, x, newton_cfg)
                                log.info("%s T=%d max|par-naive|=%.3e", cell_name, T, err)
                                if err > agree_tol:
                                    msg = f"agreement failed {cell_name} T={T}: {err}"
                                    if require_agreement:
                                        raise RuntimeError(msg)
                                    log.warning("%s (logged, not fatal)", msg)
                                mlflow.log_metric(
                                    mkey(cell_name, "max_abs_err"), err, step=int(T)
                                )

                            if "newton_fused" in modes:
                                fused_cfg = NewtonConfig(
                                    max_iters=newton_cfg.max_iters,
                                    omega=newton_cfg.omega,
                                    scan_backend="fused",
                                    residual_atol=None,
                                )
                                err_f = _agree(cell, x, fused_cfg)
                                log.info(
                                    "%s T=%d max|fused-naive|=%.3e", cell_name, T, err_f
                                )
                                if err_f > agree_tol:
                                    msg = f"fused agreement failed {cell_name} T={T}: {err_f}"
                                    if require_agreement:
                                        raise RuntimeError(msg)
                                    log.warning("%s (logged, not fatal)", msg)
                                mlflow.log_metric(
                                    mkey(cell_name, "fused_max_abs_err"), err_f, step=int(T)
                                )

                            if "sequential_eager" in modes:
                                seq_eager_stats = _time_one(
                                    f"{cell_name} sequential_eager",
                                    lambda c=cell, xx=x: sequential_apply(c, xx),
                                    x,
                                    warmup=warmup,
                                    n_runs=n_runs,
                                )
                                rows.append(
                                    {
                                        "cell": cell_name,
                                        "mode": "sequential_eager",
                                        "T": T,
                                        "dtype": dtype_name,
                                        **seq_eager_stats,
                                    }
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "sequential_eager_min_ms"),
                                    seq_eager_stats["min_ms"],
                                    step=int(T),
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "sequential_peak_mib"),
                                    seq_eager_stats["peak_mib"],
                                    step=int(T),
                                )
                            if "sequential_compiled" in modes:
                                seq_stats = _time_one(
                                    f"{cell_name} sequential_compiled",
                                    lambda c=cell, xx=x: sequential_apply_compiled(
                                        c, xx, mode=seq_compile_mode
                                    ),
                                    x,
                                    warmup=warmup,
                                    n_runs=n_runs,
                                )
                                rows.append(
                                    {
                                        "cell": cell_name,
                                        "mode": "sequential_compiled",
                                        "T": T,
                                        "dtype": dtype_name,
                                        **seq_stats,
                                    }
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "sequential_compiled_min_ms"),
                                    seq_stats["min_ms"],
                                    step=int(T),
                                )

                        newton_cap = newton_max_t.get(cell_name, int(T))
                        if T > newton_cap:
                            log.info(
                                "%s T=%d skip newton (cap=%d; 2080 Ti 11 GiB, not falling back)",
                                cell_name,
                                T,
                                newton_cap,
                            )
                            continue

                        par_stats = None
                        if "newton" in modes:
                            par_stats = _time_one(
                                f"{cell_name} newton",
                                lambda c=cell, xx=x: _newton_eager(c, xx, newton_cfg),
                                x,
                                warmup=warmup,
                                n_runs=n_runs,
                            )
                            rows.append(
                                {"cell": cell_name, "mode": "newton", "T": T, "dtype": dtype_name, **par_stats}
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_min_ms"), par_stats["min_ms"], step=int(T)
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_peak_mib"),
                                par_stats["peak_mib"],
                                step=int(T),
                            )

                        fused_stats = None
                        if "newton_fused" in modes:
                            fused_stats = _time_one(
                                f"{cell_name} newton_fused",
                                lambda c=cell, xx=x: _newton_fused(c, xx, newton_cfg),
                                x,
                                warmup=warmup,
                                n_runs=n_runs,
                            )
                            rows.append(
                                {
                                    "cell": cell_name,
                                    "mode": "newton_fused",
                                    "T": T,
                                    "dtype": dtype_name,
                                    **fused_stats,
                                }
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_fused_min_ms"),
                                fused_stats["min_ms"],
                                step=int(T),
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_fused_peak_mib"),
                                fused_stats["peak_mib"],
                                step=int(T),
                            )
                            if par_stats is not None:
                                vs = par_stats["min_ms"] / fused_stats["min_ms"]
                                log.info(
                                    "%s T=%d fused speedup vs eager newton (min)=%.2fx",
                                    cell_name,
                                    T,
                                    vs,
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "fused_speedup"), vs, step=int(T)
                                )

                        compiled_stats = None
                        compile_s = None
                        if "newton_compiled" in modes:
                            if T <= seq_max_seq:
                                torch.cuda.synchronize()
                                t_compile = time.perf_counter()
                                compiled_h = _newton_compiled(
                                    cell, x, newton_cfg, mode=newton_compile_mode
                                )
                                torch.cuda.synchronize()
                                compile_s = time.perf_counter() - t_compile
                                with torch.no_grad():
                                    seq_h = sequential_apply(cell, x)
                                err_c = float((compiled_h.float() - seq_h.float()).abs().amax())
                                log.info(
                                    "%s T=%d max|compiled-naive|=%.3e compile_s=%.2f",
                                    cell_name,
                                    T,
                                    err_c,
                                    compile_s,
                                )
                                if err_c > agree_tol:
                                    raise RuntimeError(
                                        f"compiled agreement failed {cell_name} T={T}: {err_c}"
                                    )
                                mlflow.log_metric(
                                    mkey(cell_name, "compiled_max_abs_err"),
                                    err_c,
                                    step=int(T),
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "compile_s"), compile_s, step=int(T)
                                )
                            compiled_stats = _time_one(
                                f"{cell_name} newton_compiled",
                                lambda c=cell, xx=x: _newton_compiled(
                                    c, xx, newton_cfg, mode=newton_compile_mode
                                ),
                                x,
                                warmup=warmup,
                                n_runs=n_runs,
                            )
                            rows.append(
                                {
                                    "cell": cell_name,
                                    "mode": "newton_compiled",
                                    "T": T,
                                    "dtype": dtype_name,
                                    "compile_s": compile_s,
                                    **compiled_stats,
                                }
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_compiled_min_ms"),
                                compiled_stats["min_ms"],
                                step=int(T),
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_compiled_peak_mib"),
                                compiled_stats["peak_mib"],
                                step=int(T),
                            )

                        if par_stats is not None and compiled_stats is not None:
                            vs_eager = par_stats["min_ms"] / compiled_stats["min_ms"]
                            log.info(
                                "%s T=%d compile speedup vs eager newton (min)=%.2fx",
                                cell_name,
                                T,
                                vs_eager,
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "compile_speedup"), vs_eager, step=int(T)
                            )
                            # Median, not min: CUDA-graph min can be a lucky short capture.
                            saved_s = (
                                par_stats["median_ms"] - compiled_stats["median_ms"]
                            ) / 1000.0
                            if compile_s is not None and saved_s > 0:
                                break_even = compile_s / saved_s
                                log.info(
                                    "%s T=%d compile_s=%.2f break_even=%.0f forwards "
                                    "(median save %.2f ms/step)",
                                    cell_name,
                                    T,
                                    compile_s,
                                    break_even,
                                    saved_s * 1000.0,
                                )
                                mlflow.log_metric(
                                    mkey(cell_name, "compile_break_even"),
                                    break_even,
                                    step=int(T),
                                )
                                rows[-1]["break_even_forwards"] = break_even
                            elif compile_s is not None:
                                log.info(
                                    "%s T=%d compile_s=%.2f does not beat eager median",
                                    cell_name,
                                    T,
                                    compile_s,
                                )
                        if seq_stats is not None and par_stats is not None:
                            speedup = seq_stats["min_ms"] / par_stats["min_ms"]
                            log.info(
                                "%s T=%d speedup vs compiled sequential (min)=%.2fx",
                                cell_name,
                                T,
                                speedup,
                            )
                            mlflow.log_metric(mkey(cell_name, "speedup"), speedup, step=int(T))
                        if seq_eager_stats is not None and fused_stats is not None:
                            vs_naive_rnn = (
                                seq_eager_stats["min_ms"] / fused_stats["min_ms"]
                            )
                            log.info(
                                "%s T=%d fused vs naive RNN sequential_eager (min)=%.2fx",
                                cell_name,
                                T,
                                vs_naive_rnn,
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "fused_vs_seq_eager"), vs_naive_rnn, step=int(T)
                            )
                        if seq_stats is not None and fused_stats is not None:
                            vs_seq_c = seq_stats["min_ms"] / fused_stats["min_ms"]
                            log.info(
                                "%s T=%d fused vs compiled sequential (min)=%.2fx",
                                cell_name,
                                T,
                                vs_seq_c,
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "fused_vs_seq_compiled"), vs_seq_c, step=int(T)
                            )
                        if seq_eager_stats is not None and par_stats is not None:
                            vs_par_naive = (
                                seq_eager_stats["min_ms"] / par_stats["min_ms"]
                            )
                            log.info(
                                "%s T=%d eager Newton vs naive RNN sequential_eager (min)=%.2fx",
                                cell_name,
                                T,
                                vs_par_naive,
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "newton_vs_seq_eager"), vs_par_naive, step=int(T)
                            )
                        if seq_stats is not None and compiled_stats is not None:
                            speedup_c = seq_stats["min_ms"] / compiled_stats["min_ms"]
                            log.info(
                                "%s T=%d compiled-newton vs compiled-seq (min)=%.2fx",
                                cell_name,
                                T,
                                speedup_c,
                            )
                            mlflow.log_metric(
                                mkey(cell_name, "compiled_speedup"), speedup_c, step=int(T)
                            )
                        _write_csv(rows, csv_path)
                        del x
                        torch.cuda.empty_cache()
        finally:
            if rows:
                _write_csv(rows, csv_path)
                mlflow.log_artifact(str(csv_path))
                log.info("wrote %s", csv_path)


if __name__ == "__main__":
    main()
