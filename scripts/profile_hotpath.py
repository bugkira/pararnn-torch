"""PyTorch profiler (+ optional nsys) on Newton vs sequential.

Shapes: configs/bench/profile_hotpath.yaml. GPU: 2080 Ti by name.

  uv run python scripts/profile_hotpath.py
  nsys profile -o outputs/profile/nsys_hotpath --stats=true \\
      uv run python scripts/profile_hotpath.py
"""

from __future__ import annotations

import csv
import hashlib
import logging
import subprocess
from pathlib import Path

import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, setup_logging, wait_until_free

log = logging.getLogger("profile")
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "bench" / "profile_hotpath.yaml"
CELLS = {"ParaGRU": ParaGRU, "ParaLSTM": ParaLSTM}


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


def _cuda_min_ms(fn, *, warmup: int, n_runs: int) -> float:
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
    return min(samples)


def _top_cuda_rows(prof: profile, n: int = 15) -> list[dict[str, str | float | int]]:
    key_avgs = sorted(
        prof.key_averages(),
        key=lambda e: e.self_device_time_total,
        reverse=True,
    )
    rows: list[dict[str, str | float | int]] = []
    for ev in key_avgs[:n]:
        if ev.self_device_time_total <= 0:
            continue
        rows.append(
            {
                "name": ev.key,
                "self_cuda_us": float(ev.self_device_time_total),
                "cpu_us": float(ev.self_cpu_time_total),
                "calls": int(ev.count),
            }
        )
    return rows


def main() -> None:
    setup_logging()
    spec = yaml.safe_load(CONFIG_PATH.read_text())
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    if device.type != "cuda":
        raise RuntimeError("profile needs the 2080 Ti (CUDA_VISIBLE_DEVICES to restrict)")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=4.0, poll_s=30.0)
    newton_cfg = NewtonConfig(
        max_iters=int(spec["newton_iters"]),
        scan_backend="eager",
        residual_atol=None,
    )
    out_dir = ROOT / "outputs" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    import mlflow

    mlflow.set_experiment("hotpath-profile")
    with mlflow.start_run(run_name="eager-newton-scan"):
        mlflow.set_tags(
            {
                "gpu": torch.cuda.get_device_name(device),
                "dtype": spec["dtype"],
                "purpose": "bottlenecks.md #1 profile after #2 step-without-J",
            }
        )
        mlflow.log_params(
            {
                "newton_iters": newton_cfg.max_iters,
                "batch": spec["batch"],
                "d_in": spec["d_in"],
                "d_h": spec["d_h"],
                "git": _git_commit(),
                "uv_lock": _lock_hash(),
                "config": str(CONFIG_PATH.relative_to(ROOT)),
            }
        )
        mlflow.log_artifact(str(CONFIG_PATH))
        mlflow.log_text(
            "CUPTI profile of sequential vs Newton on 2080 Ti. "
            "Shapes justified in configs/bench/profile_hotpath.yaml. "
            "step() no longer builds J (bottlenecks.md #2).",
            "why.txt",
        )

        summary_rows: list[dict] = []
        for case in spec["cases"]:
            cell_name = case["cell"]
            T = int(case["T"])
            cell = CELLS[cell_name](int(spec["d_in"]), int(spec["d_h"])).to(device).eval()
            x = torch.randn(int(spec["batch"]), T, int(spec["d_in"]), device=device)
            modes = {
                "sequential": lambda c=cell, xx=x: sequential_apply(c, xx),
                "newton": lambda c=cell, xx=x: newton_apply(c, xx, newton_cfg),
            }
            for mode, fn in modes.items():
                tag = f"{cell_name}_T{T}_{mode}"
                log.info("timing %s", tag)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                min_ms = _cuda_min_ms(
                    fn,
                    warmup=int(spec["timing_warmup"]),
                    n_runs=int(spec["timing_runs"]),
                )
                peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
                log.info("%s min=%.3f ms peak=%.1f MiB", tag, min_ms, peak_mib)
                mlflow.log_metric(f"{tag}_min_ms", min_ms)
                mlflow.log_metric(f"{tag}_peak_mib", peak_mib)
                summary_rows.append(
                    {
                        "cell": cell_name,
                        "mode": mode,
                        "T": T,
                        "min_ms": min_ms,
                        "peak_mib": peak_mib,
                    }
                )

                log.info("profiling %s", tag)
                activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                with profile(
                    activities=activities,
                    record_shapes=True,
                    with_stack=False,
                    schedule=torch.profiler.schedule(
                        wait=int(spec["profiler_wait"]),
                        warmup=int(spec["profiler_warmup"]),
                        active=int(spec["profiler_active"]),
                        repeat=1,
                    ),
                    on_trace_ready=None,
                ) as prof:
                    for _ in range(
                        int(spec["profiler_wait"])
                        + int(spec["profiler_warmup"])
                        + int(spec["profiler_active"])
                    ):
                        with record_function(tag):
                            fn()
                        prof.step()
                table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=20)
                table_path = out_dir / f"{tag}_kernels.txt"
                table_path.write_text(table)
                trace_path = out_dir / f"{tag}.json"
                prof.export_chrome_trace(str(trace_path))
                log.info("wrote %s and %s", table_path, trace_path)
                mlflow.log_artifact(str(table_path))
                mlflow.log_artifact(str(trace_path))
                top = _top_cuda_rows(prof)
                for i, row in enumerate(top[:8]):
                    mlflow.log_metric(
                        f"{tag}_kernel{i}_us",
                        row["self_cuda_us"],
                    )
                    log.info(
                        "  kernel[%d] %s  self_cuda=%.0f us  calls=%s",
                        i,
                        row["name"],
                        row["self_cuda_us"],
                        row["calls"],
                    )
            del cell, x
            torch.cuda.empty_cache()

        csv_path = out_dir / "profile_min_ms.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        mlflow.log_artifact(str(csv_path))
        log.info("wrote %s", csv_path)


if __name__ == "__main__":
    main()
