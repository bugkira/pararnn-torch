"""Virtual two-rank diag scan on one 2080 Ti (CUDA streams).

  uv run python scripts/seq_parallel_ranks.py

Not NCCL. Not a speedup claim: one saturated card. Numerics vs ``scan_diag``,
timing vs sequential prefix (FlashRNN-style wait on the left tile).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from pararnn.solvers.scan import scan_diag
from pararnn.solvers.seq_parallel import (
    scan_diag_two_ranks,
    sequential_prefix_two_ranks,
)
from utils.cuda_timing import cuda_minmax
from utils.mlflow_helper import git_commit, lock_hash, setup_logging, uv_export_hash

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, wait_until_free

log = logging.getLogger("seq_parallel")

WARMUP = 10
N_RUNS = 50
BATCH = 8
DIM = 256
# Long enough that a Blelloch pass is visible; still one card.
SEQ_LENS = (512, 2048, 8192)


def _system(batch: int, time: int, dim: int, device: torch.device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    jac = (torch.randn(batch, time, dim, generator=g) * 0.3).to(device)
    residual = torch.randn(batch, time, dim, generator=g).to(device)
    return jac, residual


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=8.0, poll_s=30.0)
    gpu_name = torch.cuda.get_device_name(device)
    log.info("seq_parallel_ranks gpu=%s", gpu_name)

    import mlflow

    mlflow.set_experiment("seq-parallel-scan")
    with mlflow.start_run(run_name="two-stream-virtual-ranks"):
        mlflow.set_tags(
            {
                "gpu": gpu_name,
                "mode": "virtual_two_rank",
                "cell": "scan_diag",
            }
        )
        mlflow.log_params(
            {
                "batch": BATCH,
                "d": DIM,
                "warmup": WARMUP,
                "n_runs": N_RUNS,
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "why": (
                    "Correctness of two-tile carry on one GPU. Not multi-node. "
                    "Do not read stream overlap as cluster speedup."
                ),
            }
        )

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        max_abs = 0.0
        for t in SEQ_LENS:
            jac, residual = _system(BATCH, t, DIM, device, seed=t)
            ref = scan_diag(jac, residual)
            two = scan_diag_two_ranks(jac, residual, streams=(s0, s1))
            pref = sequential_prefix_two_ranks(jac, residual)
            torch.cuda.synchronize()
            err_two = (two - ref).abs().amax().item()
            err_pref = (pref - ref).abs().amax().item()
            max_abs = max(max_abs, err_two, err_pref)
            log.info(
                "agree T=%s maxabs_two=%.3e maxabs_prefix=%.3e",
                t,
                err_two,
                err_pref,
            )
            if err_two > 1e-4 or err_pref > 1e-4:
                raise RuntimeError(f"scan mismatch T={t} two={err_two:.3e} prefix={err_pref:.3e}")
            mlflow.log_metric("maxabs_two_rank", err_two, step=t)
            mlflow.log_metric("maxabs_seq_prefix", err_pref, step=t)

            def _ref(j: torch.Tensor = jac, r: torch.Tensor = residual) -> None:
                scan_diag(j, r)

            def _two(j: torch.Tensor = jac, r: torch.Tensor = residual) -> None:
                scan_diag_two_ranks(j, r, streams=(s0, s1))

            def _pref(j: torch.Tensor = jac, r: torch.Tensor = residual) -> None:
                sequential_prefix_two_ranks(j, r)

            rows = []
            for name, fn in (
                ("scan_diag", _ref),
                ("two_rank_streams", _two),
                ("seq_prefix", _pref),
            ):
                tmin, tmed, tmean = cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
                mem = torch.cuda.max_memory_allocated(device) / (1024**2)
                log.info(
                    "%s T=%s min=%.3f median=%.3f mean=%.3f ms peak=%.1f MiB",
                    name,
                    t,
                    tmin,
                    tmed,
                    tmean,
                    mem,
                )
                mlflow.log_metric(f"{name}/min_ms", tmin, step=t)
                mlflow.log_metric(f"{name}/median_ms", tmed, step=t)
                mlflow.log_metric(f"{name}/mean_ms", tmean, step=t)
                rows.append((name, tmin, tmed, tmean))

            two_min = next(r[1] for r in rows if r[0] == "two_rank_streams")
            ref_min = next(r[1] for r in rows if r[0] == "scan_diag")
            mlflow.log_metric("two_over_ref_min", two_min / ref_min, step=t)

        mlflow.log_metric("maxabs_all", max_abs)
        log.info("done maxabs_all=%.3e", max_abs)


if __name__ == "__main__":
    logging.getLogger("gpu").setLevel(logging.INFO)
    main()
