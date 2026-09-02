"""A/B: HEAD vs working-tree Newton + fused host. Smoke 10/50, not App. B.

  uv run python scripts/bench_newton_patch.py

2080 Ti by name. K=3 App. A. B=8, d_h=256: paper Table 3 per-head width.
Library default is residual_atol=1e-5 / residual_fail=1.0; benches often set
both None. This run times both, plus eq. 2.6 backward with/without h0.
T=64 is n_chunks==1 (GRU tile 128, LSTM tile 64) — fused fp32 scratch.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

from gpu import (
    DEFAULT_EXPERIMENT_GPU_NAME,
    select_device,
    setup_logging,
    wait_until_free,
)

log = logging.getLogger("bench")
ROOT = Path(__file__).resolve().parents[1]
HOT_FILES = (
    ROOT / "src" / "pararnn" / "solvers" / "newton.py",
    ROOT / "src" / "pararnn" / "kernels" / "newton_gru.py",
    ROOT / "src" / "pararnn" / "kernels" / "newton_lstm.py",
    ROOT / "src" / "pararnn" / "kernels" / "newton_slstm.py",
)
_PYC_DIRS = (
    ROOT / "src" / "pararnn" / "solvers" / "__pycache__",
    ROOT / "src" / "pararnn" / "kernels" / "__pycache__",
)
WARMUP = 10
N_RUNS = 50
BATCH = 8
D_H = 256
# K=3: App. A. T=64 is one fused tile; T=2048 is the documented smoke length.
SEQ_LENS = (64, 2048)


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


def _cfg(*, backend: str, library_default: bool):
    from pararnn import NewtonConfig

    if library_default:
        return NewtonConfig(max_iters=3, scan_backend=backend)
    return NewtonConfig(
        max_iters=3,
        scan_backend=backend,
        residual_atol=None,
        residual_fail=None,
    )


def _time_forward(cell: nn.Module, x: Tensor, cfg, name: str) -> dict:
    from pararnn.solvers import NewtonStats, newton_apply

    def fn() -> None:
        with torch.no_grad():
            newton_apply(cell, x, cfg)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    tmin, tmed, tmean = _cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    st = NewtonStats()
    with torch.no_grad():
        newton_apply(cell, x, cfg, stats=st)
    row = {
        "name": name,
        "min_ms": tmin,
        "median_ms": tmed,
        "mean_ms": tmean,
        "peak_mib": mem,
        "iters": st.iters,
        "max_residual": st.max_residual,
        "scan_backend": st.scan_backend,
    }
    log.info(
        "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB  iters=%s  res=%.3e",
        name,
        tmin,
        tmed,
        tmean,
        mem,
        st.iters,
        st.max_residual,
    )
    return row


def _time_backward(cell: nn.Module, x: Tensor, cfg, h0: Tensor | None, name: str) -> dict:
    from pararnn.solvers import newton_apply

    x_in = x.detach().requires_grad_(True)
    h0_in = None if h0 is None else h0.detach().requires_grad_(True)

    def fn() -> None:
        if x_in.grad is not None:
            x_in.grad = None
        if h0_in is not None and h0_in.grad is not None:
            h0_in.grad = None
        cell.zero_grad(set_to_none=True)
        y = newton_apply(cell, x_in, cfg, h0=h0_in)
        y.square().sum().backward()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    tmin, tmed, tmean = _cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    log.info(
        "%s  min=%.3f ms  median=%.3f  mean=%.3f  peak=%.1f MiB",
        name,
        tmin,
        tmed,
        tmean,
        mem,
    )
    return {
        "name": name,
        "min_ms": tmin,
        "median_ms": tmed,
        "mean_ms": tmean,
        "peak_mib": mem,
    }


def _worker(label: str) -> list[dict]:
    from pararnn.cells import ParaGRU, ParaLSTM
    from pararnn.solvers.newton import newton_apply as _

    del _
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    rows: list[dict] = []
    torch.manual_seed(0)
    for T in SEQ_LENS:
        x = torch.randn(BATCH, T, D_H, device=device)
        gru = ParaGRU(D_H, D_H).to(device).eval()
        lstm = ParaLSTM(D_H, D_H).to(device).eval()
        for backend, cell, cname in (
            ("fused", gru, "gru"),
            ("eager", gru, "gru"),
            ("fused", lstm, "lstm"),
        ):
            if backend == "fused" and device.type != "cuda":
                continue
            for lib in (True, False):
                cfg = _cfg(backend=backend, library_default=lib)
                path = "library" if lib else "bench"
                name = f"{label}/{cname}/{backend}/{path}/T{T}/fwd"
                row = _time_forward(cell, x, cfg, name)
                row.update(label=label, cell=cname, backend=backend, path=path, T=T, kind="fwd")
                rows.append(row)
        cfg_b = _cfg(backend="eager", library_default=True)
        h0 = 0.3 * torch.randn(BATCH, D_H, device=device)
        for use_h0, tag in ((False, "noh0"), (True, "h0")):
            name = f"{label}/gru/eager/library/T{T}/bwd-{tag}"
            row = _time_backward(gru, x, cfg_b, h0 if use_h0 else None, name)
            row.update(
                label=label,
                cell="gru",
                backend="eager",
                path="library",
                T=T,
                kind=f"bwd-{tag}",
            )
            rows.append(row)
    return rows


def _purge_pyc() -> None:
    for pyc in _PYC_DIRS:
        if pyc.is_dir():
            shutil.rmtree(pyc)


def _write_head_hot_files() -> dict[Path, bytes]:
    backups = {path: path.read_bytes() for path in HOT_FILES}
    for path in HOT_FILES:
        rel = path.relative_to(ROOT).as_posix()
        path.write_bytes(subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT))
    _purge_pyc()
    return backups


def _restore_hot_files(backups: dict[Path, bytes]) -> None:
    for path, blob in backups.items():
        path.write_bytes(blob)
    _purge_pyc()


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    if device.type != "cuda":
        raise RuntimeError("needs the 2080 Ti")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=8.0, poll_s=30.0)
    log.info("gpu=%s", torch.cuda.get_device_name(device))

    import mlflow

    mlflow.set_experiment("newton-patch-bench")
    with mlflow.start_run(run_name="accelerator-review-d2h-scratch"):
        sys.path.insert(0, str(ROOT / "scripts"))
        from utils.mlflow_helper import git_commit, lock_hash, uv_export_hash

        mlflow.set_tags(
            {
                "gpu": torch.cuda.get_device_name(device),
                "protocol": "smoke-10-50",
                "why": (
                    "A/B HEAD vs accelerator-review: D2H only if residual_atol, "
                    "fused n_chunks==1 fp32 scratch, head-pack contiguous"
                ),
                "cell": "para_gru_lstm",
                "mode": "parallel",
                "dtype": "fp32",
            }
        )
        mlflow.log_params(
            {
                "warmup": WARMUP,
                "n_runs": N_RUNS,
                "batch": BATCH,
                "d_h": D_H,
                "newton_iters": 3,
                "head": subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_commit": git_commit(),
                "uv_lock_hash": lock_hash(),
                "uv_export_hash": uv_export_hash(),
            }
        )
        note = ROOT / "outputs" / "bench_newton_patch_why.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "Smoke 10/50 after accelerator-review. Compare git HEAD hot path "
            "(newton.py + fused GRU/LSTM/sLSTM host) to the working tree: "
            "float(amax) only if residual_atol; fused n_chunks==1 scratch; "
            "head-pack contiguous. Fail-loud cell.step after K is unchanged.\n"
        )
        mlflow.log_artifact(str(note))
        backups = {path: path.read_bytes() for path in HOT_FILES}
        all_rows: list[dict] = []
        try:
            _write_head_hot_files()
            log.info("timing HEAD hot path (newton.py + fused kernels)")
            before = subprocess.check_output(
                [sys.executable, str(Path(__file__).resolve()), "--worker", "before"],
                cwd=ROOT,
            )
            all_rows.extend(json.loads(before.decode()))
        finally:
            _restore_hot_files(backups)
        log.info("timing working-tree hot path")
        after = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "after"],
            cwd=ROOT,
        )
        all_rows.extend(json.loads(after.decode()))

        by_name = {r["name"]: r for r in all_rows}
        for row in all_rows:
            if row["label"] != "after":
                continue
            key = row["name"].replace("after/", "before/", 1)
            old = by_name.get(key)
            if old is None:
                continue
            ratio = row["min_ms"] / old["min_ms"] if old["min_ms"] else float("nan")
            delta = row["min_ms"] - old["min_ms"]
            log.info(
                "delta %s  before=%.3f after=%.3f  d=%+.3f ms  ratio=%.3f",
                row["name"].removeprefix("after/"),
                old["min_ms"],
                row["min_ms"],
                delta,
                ratio,
            )
            step = int(row["T"])
            metric = f"{row['cell']}_{row['backend']}_{row['path']}_{row['kind']}"
            mlflow.log_metric(f"{metric}_before_ms", old["min_ms"], step=step)
            mlflow.log_metric(f"{metric}_after_ms", row["min_ms"], step=step)
            mlflow.log_metric(f"{metric}_ratio", ratio, step=step)
        out = ROOT / "outputs" / "bench_newton_patch.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_rows, indent=2))
        mlflow.log_artifact(str(out))
        log.info("wrote %s", out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        setup_logging()
        print(json.dumps(_worker(sys.argv[2])))
    else:
        main()
