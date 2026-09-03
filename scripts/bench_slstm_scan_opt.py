"""Fused sLSTM scan ablations: assoc vs Thomas+PCR vs chunk windows.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        uv run python scripts/bench_slstm_scan_opt.py
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
        uv run python scripts/bench_slstm_scan_opt.py

Protocol: 10 warmup / 20 runs, min CUDA-event ms. Not App. B (20/100).
Shapes from configs/bench/newton_slstm.yaml: B=8, d_h=256, K=3.
picard_iters=0 pins the P=0 fused table (same as that YAML).
GPU is whatever card CUDA_VISIBLE_DEVICES leaves (3060 or 2080 Ti).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# CUDA_DEVICE_ORDER is set in gpu.py; that import must precede torch.
from gpu import select_device, wait_until_free  # noqa: I001

import torch
from torch import Tensor

from pararnn import NewtonConfig, NewtonStats, ParaSLSTM
from pararnn.kernels.precision import is_fused_dtype_supported
from pararnn.solvers import newton_apply
from utils.cuda_timing import time_forward
from utils.mlflow_helper import git_commit, lock_hash, setup_logging

log = logging.getLogger("bench")

# 10/20: enough to rank kernels, not the 10/50 smoke in newton_slstm.yaml.
WARMUP = 10
N_RUNS = 20
BATCH = 8
D_H = 256
# App. A K=3. Width 256: P=0 fused does not snap (YAML require_agreement false).
NEWTON_ITERS = 3
SEQ_LENS = (256, 1024)


def _cfg(picard: int, **kw) -> NewtonConfig:
    # P=0: same table as configs/bench/newton_slstm.yaml (diverged at d_h=256).
    # P=3: library auto for T≤2048 (slstm_auto_picard / para-slstm.md).
    base = {
        "max_iters": NEWTON_ITERS,
        "scan_backend": "fused",
        "residual_atol": None,
        "residual_fail": None,
        "picard_iters": picard,
    }
    base.update(kw)
    return NewtonConfig(**base)


def _variants(picard: int) -> list[tuple[str, NewtonConfig]]:
    return [
        ("assoc", _cfg(picard, scan_tile="assoc")),
        ("seq", _cfg(picard, scan_tile="seq")),
        ("chunk64_python", _cfg(picard, scan_tile="assoc", chunk_len=64)),
        ("thomas2", _cfg(picard, scan_tile="thomas2")),
        ("thomas4", _cfg(picard, scan_tile="thomas4")),
        ("assoc_loop", _cfg(picard, fused_time_loop=True, fused_window_len=64)),
    ]


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", default="")
    parser.add_argument(
        "--picard",
        type=int,
        default=0,
        help="Picard P. 0 = P=0 timing table (diverged at d_h=256). 3 = library T≤2048.",
    )
    args = parser.parse_args()
    if args.gpu_name:
        device = select_device(args.gpu_name)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=4.0)
    gpu = torch.cuda.get_device_name(device)
    cc = torch.cuda.get_device_capability(device)
    log.info("scan_opt_start gpu=%s cc=%s git=%s lock=%s", gpu, cc, git_commit()[:8], lock_hash())

    import mlflow

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name=f"slstm-scan-opt-{gpu.replace(' ', '-')}"):
        mlflow.set_tags(
            {
                "gpu": gpu,
                "cc": f"{cc[0]}.{cc[1]}",
                "cell": "para_slstm",
                "mix": "diag",
                "protocol": "smoke-10-20",
            }
        )
        mlflow.log_params(
            {
                "batch": BATCH,
                "d_h": D_H,
                "newton_iters": NEWTON_ITERS,
                "picard_iters": args.picard,
                "warmup": WARMUP,
                "n_runs": N_RUNS,
                "git": git_commit(),
                "uv_lock": lock_hash(),
            }
        )
        dtypes: list[torch.dtype] = [torch.float32]
        if is_fused_dtype_supported(torch.bfloat16, device):
            dtypes.append(torch.bfloat16)
        variants = _variants(args.picard)
        log.info("variants=%s picard=%d", [n for n, _ in variants], args.picard)
        for dtype in dtypes:
            for T in SEQ_LENS:
                torch.manual_seed(0)
                cell = ParaSLSTM(D_H, D_H, mix="diag").to(device=device, dtype=dtype).eval()
                x = torch.randn(BATCH, T, D_H, device=device, dtype=dtype)
                ref: Tensor | None = None
                for name, cfg in variants:
                    cfg = replace(cfg)
                    st = NewtonStats()
                    try:
                        with torch.no_grad():
                            y = newton_apply(cell, x, cfg, stats=st)
                    except (TypeError, ValueError) as exc:
                        log.info("skip %s T=%d dtype=%s err=%s", name, T, dtype, exc)
                        continue
                    if ref is None:
                        ref = y
                        err_assoc = 0.0
                    else:
                        err_assoc = float((y.float() - ref.float()).abs().amax())
                    row = time_forward(
                        f"{name} {dtype} T={T}",
                        lambda c=cell, xx=x, k=cfg: newton_apply(c, xx, k),
                        warmup=WARMUP,
                        n_runs=N_RUNS,
                        seq_len=T,
                        logger=log,
                    )
                    step = T if dtype == torch.float32 else T + 10_000
                    mlflow.log_metric(f"{name}_{dtype}_min_ms", row["min_ms"], step=step)
                    mlflow.log_metric(f"{name}_{dtype}_peak_mib", row["peak_mib"], step=step)
                    mlflow.log_metric(f"{name}_{dtype}_vs_assoc", err_assoc, step=step)
                    mlflow.log_metric(f"{name}_{dtype}_residual", float(st.max_residual), step=step)
                    log.info(
                        "done name=%s dtype=%s T=%d min_ms=%.3f peak_mib=%.1f "
                        "vs_assoc=%.3e residual=%.3e",
                        name,
                        dtype,
                        T,
                        row["min_ms"],
                        row["peak_mib"],
                        err_assoc,
                        float(st.max_residual),
                    )


if __name__ == "__main__":
    main()
