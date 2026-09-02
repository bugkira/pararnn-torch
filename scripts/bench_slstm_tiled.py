"""Ablate serial tile scan and R mixing vs FlashRNN.

  uv run python scripts/bench_slstm_tiled.py

2080 Ti by name. 10 warmup / 50 runs, min ms. MLflow newton-slstm-bench.
mix='diag' is already the fused R (not a new kernel). Head mix is eager.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
from torch import Tensor

from pararnn.cells import ParaSLSTM
from pararnn.solvers import NewtonConfig, newton_apply, sequential_apply
from utils.cuda_timing import time_forward
from utils.flashrnn_glue import ensure_cuda_home
from utils.mlflow_helper import setup_logging

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, wait_until_free

log = logging.getLogger("bench")

WARMUP = 10
N_RUNS = 50
BATCH = 8
D_H = 256


def _time(name: str, fn, T: int) -> float:
    return time_forward(
        name,
        fn,
        warmup=WARMUP,
        n_runs=N_RUNS,
        seq_len=T,
        logger=log,
    )["min_ms"]


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    if device.type != "cuda":
        raise RuntimeError("needs the 2080 Ti")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=8.0, poll_s=30.0)
    log.info("bench_slstm_tiled gpu=%s", torch.cuda.get_device_name(device))

    import mlflow

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name="slstm-tiled-ablation"):
        mlflow.set_tags({"gpu": torch.cuda.get_device_name(device), "protocol": "smoke-10-50"})
        variants = (
            ("baseline", NewtonConfig(scan_backend="fused", residual_atol=None)),
            (
                "scan_seq",
                NewtonConfig(scan_backend="fused", residual_atol=None, scan_tile="seq"),
            ),
        )
        for T in (256, 2048):
            torch.manual_seed(0)
            cell = ParaSLSTM(D_H, D_H, mix="diag").to(device).eval()
            x = torch.randn(BATCH, T, D_H, device=device)
            with torch.no_grad():
                seq = sequential_apply(cell, x)
            for name, cfg in variants:
                cfg = replace(cfg, max_iters=3)
                with torch.no_grad():
                    par = newton_apply(cell, x, cfg)
                    err = float((par - seq).abs().amax())
                log.info("%s T=%d max|par-seq|=%.3e", name, T, err)
                mlflow.log_metric(f"{name}_err", err, step=T)
                tmin = _time(
                    f"ParaSLSTM {name}",
                    lambda c=cell, xx=x, k=cfg: newton_apply(c, xx, k),
                    T,
                )
                mlflow.log_metric(f"{name}_min_ms", tmin, step=T)

            if T == 256:
                torch.manual_seed(0)
                head = ParaSLSTM(D_H, D_H, mix="head", n_heads=8).to(device).eval()
                xh = torch.randn(2, 16, D_H, device=device)
                cfg_h = NewtonConfig(
                    max_iters=3,
                    scan_backend="eager",
                    residual_atol=None,
                    picard_iters=0,
                    residual_fail=None,
                )
                tmin = _time(
                    "ParaSLSTM mix=head eager T=16 B=2",
                    lambda c=head, xx=xh, k=cfg_h: newton_apply(c, xx, k),
                    16,
                )
                mlflow.log_metric("head_eager_min_ms", tmin, step=16)

            try:
                ensure_cuda_home(logger=log)
                from flashrnn import flashrnn

                n_heads, d_head = 8, 32

                def _fr(
                    xx: Tensor = x,
                    seq_len: int = T,
                    nh: int = n_heads,
                    dh: int = d_head,
                ) -> None:
                    wx5 = xx.new_empty(BATCH, seq_len, 4, nh, dh).normal_()
                    rec = xx.new_empty(4, nh, dh, dh).normal_().mul_(0.1)
                    bias = xx.new_zeros(4, nh, dh)
                    s0 = xx.new_zeros(4, BATCH, 1, nh, dh)
                    flashrnn(
                        wx5,
                        rec,
                        bias,
                        states=s0,
                        function="slstm",
                        backend="triton_fused",
                        dtype="float32",
                    )

                tmin = _time("flashrnn_triton_fused", _fr, T)
                mlflow.log_metric("flashrnn_min_ms", tmin, step=T)
            except (ImportError, RuntimeError, OSError) as exc:
                log.warning("flashrnn skip: %s", exc)


if __name__ == "__main__":
    main()
