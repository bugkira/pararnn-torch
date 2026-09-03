"""Windowed fused sLSTM: residual vs sequential, CE/PPL vs assoc, timing.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        uv run python scripts/bench_slstm_window_ce.py

Not TinyStories: untrained next-token CE on the same weights (embed +
ParaSLSTM hidden + linear). If CE matches assoc to ~0.01, windowed Newton
is a usable prefill/inference path. Hidden max-diff is the solver error.

Shapes: B=8, T=1024, d_h=256, K=3, P=3 (library auto for T≤2048).
Windows 32/64/128 from NewtonConfig.fused_window_len (this repo: chunk_len=64
was ~2e-4 vs sequential). 10 warmup / 20 runs, min CUDA-event ms.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from gpu import select_device, wait_until_free  # noqa: I001

import torch
from torch.nn import functional as F

from pararnn import NewtonConfig, NewtonStats, ParaSLSTM, sequential_apply
from pararnn.layout import SLSTM_HIDDEN
from pararnn.solvers import newton_apply
from pararnn.solvers.newton.config import FUSED_WINDOW_LENS
from utils.cuda_timing import time_forward
from utils.mlflow_helper import git_commit, lock_hash, setup_logging

log = logging.getLogger("bench")

WARMUP = 10
N_RUNS = 20
BATCH = 8
T = 1024
D_H = 256
VOCAB = 32
# P=3: slstm_auto_picard for T≤2048. Agreement table, not the P=0 YAML.
PICARD = 3
NEWTON_ITERS = 3


def _cfg(**kw) -> NewtonConfig:
    base = {
        "max_iters": NEWTON_ITERS,
        "scan_backend": "fused",
        "residual_atol": None,
        "residual_fail": None,
        "picard_iters": PICARD,
    }
    base.update(kw)
    return NewtonConfig(**base)


def _ce_ppl(logits: torch.Tensor, tokens: torch.Tensor) -> tuple[float, float]:
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB),
        tokens[:, 1:].reshape(-1),
        reduction="mean",
    )
    val = float(ce)
    return val, math.exp(val)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", default="")
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
    log.info("window_ce_start gpu=%s cc=%s git=%s", gpu, cc, git_commit()[:8])

    import mlflow

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name=f"slstm-window-ce-{gpu.replace(' ', '-')}"):
        mlflow.set_tags(
            {
                "gpu": gpu,
                "cc": f"{cc[0]}.{cc[1]}",
                "cell": "para_slstm",
                "protocol": "window-ce-10-20",
            }
        )
        mlflow.log_params(
            {
                "batch": BATCH,
                "seq_len": T,
                "d_h": D_H,
                "vocab": VOCAB,
                "newton_iters": NEWTON_ITERS,
                "picard_iters": PICARD,
                "git": git_commit(),
                "uv_lock": lock_hash(),
            }
        )
        torch.manual_seed(0)
        cell = ParaSLSTM(D_H, D_H, mix="diag").to(device).eval()
        embed = torch.nn.Embedding(VOCAB, D_H).to(device)
        head = torch.nn.Linear(D_H, VOCAB).to(device)
        tokens = torch.randint(0, VOCAB, (BATCH, T), device=device)
        x = embed(tokens)

        seq_states = sequential_apply(cell, x)
        seq_h = seq_states[:, :, SLSTM_HIDDEN]
        seq_logits = head(seq_h)
        seq_ce, seq_ppl = _ce_ppl(seq_logits, tokens)

        variants: list[tuple[str, NewtonConfig]] = [("assoc", _cfg())]
        for w in FUSED_WINDOW_LENS:
            variants.append(
                (
                    f"loop{w}",
                    _cfg(fused_time_loop=True, fused_window_len=w),
                )
            )

        ref_h: torch.Tensor | None = None
        ref_ce = 0.0
        for name, cfg in variants:
            st = NewtonStats()
            with torch.no_grad():
                y = newton_apply(cell, x, cfg, stats=st)
            h = y[:, :, SLSTM_HIDDEN]
            logits = head(h)
            ce, ppl = _ce_ppl(logits, tokens)
            vs_seq = float((h.float() - seq_h.float()).abs().amax())
            if ref_h is None:
                ref_h = h
                ref_ce = ce
                vs_assoc = 0.0
                dce = 0.0
            else:
                vs_assoc = float((h.float() - ref_h.float()).abs().amax())
                dce = abs(ce - ref_ce)
            row = time_forward(
                f"{name} fp32 T={T}",
                lambda c=cell, xx=x, k=cfg: newton_apply(c, xx, k),
                warmup=WARMUP,
                n_runs=N_RUNS,
                seq_len=T,
                logger=log,
            )
            mlflow.log_metric(f"{name}_min_ms", row["min_ms"])
            mlflow.log_metric(f"{name}_residual", float(st.max_residual))
            mlflow.log_metric(f"{name}_vs_seq", vs_seq)
            mlflow.log_metric(f"{name}_vs_assoc", vs_assoc)
            mlflow.log_metric(f"{name}_ce", ce)
            mlflow.log_metric(f"{name}_ppl", ppl)
            mlflow.log_metric(f"{name}_dce_assoc", dce)
            log.info(
                "done name=%s min_ms=%.3f residual=%.3e vs_seq=%.3e vs_assoc=%.3e "
                "ce=%.4f ppl=%.4f dce_assoc=%.4f seq_ce=%.4f",
                name,
                row["min_ms"],
                float(st.max_residual),
                vs_seq,
                vs_assoc,
                ce,
                ppl,
                dce,
                seq_ce,
            )
        mlflow.log_metric("sequential_ce", seq_ce)
        mlflow.log_metric("sequential_ppl", seq_ppl)


if __name__ == "__main__":
    main()
