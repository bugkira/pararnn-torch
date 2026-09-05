"""A/B: packed sLSTM-diag VJP vs Autograd on ``step``.

  uv run python scripts/bench_packed_vjp.py

Does not change the cell. Eq. 2.6 reverse scan is shared; only the cell VJP
differs. Quality = max |grad packed − Autograd|. Speed = CUDA events.

Shapes: Dyck train smoke (``configs/train/dyck_vs_flashrnn.yaml``) and fused
Newton smoke (B=8, d_h=256, T=2048, ``configs/bench/newton_slstm.yaml``).
Protocol: 10 warmup / 50 runs — house smoke, not App. B. GPU: 2080 Ti by name.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import torch
from torch import Tensor
from torch.nn import functional as F

from examples.dyck_language import VOCAB, sample_dyck1
from scripts.slstm_vs_flashrnn import _NewtonDyckLM
from pararnn import NewtonConfig, ParaSLSTM
from pararnn.layout import prepend_state
from pararnn.solvers import newton as newton_mod
from pararnn.solvers import newton_apply
from pararnn.solvers.newton import _eq26_vjp
from pararnn.solvers.vjp import cell_vjp
from pararnn.solvers.vjp import uses_packed_vjp as _uses_packed_vjp
from utils.cuda_timing import cuda_minmax
from utils.mlflow_helper import git_commit, lock_hash, setup_logging, uv_export_hash

from gpu import (
    DEFAULT_EXPERIMENT_GPU_NAME,
    select_device,
    wait_until_free,
)

log = logging.getLogger("packed_vjp")

# House smoke (bench_newton_patch / newton_slstm.yaml), not App. B 20/100.
WARMUP = 10
N_RUNS = 50
# Dyck yaml: B=32, T=64, d_h=32, K=3 fused, lr 3e-3, 50 AdamW.
DYCK_BATCH, DYCK_T, DYCK_DH, DYCK_STEPS, DYCK_LR = 32, 64, 32, 50, 3e-3
# Fused sLSTM smoke width (paper Table 3 per-head 256). Auto P=3 at T=2048.
BENCH_BATCH, BENCH_DH, BENCH_T = 8, 256, 2048
# K=3 App. A / library. Explicit P so the bench path skips picard_adapt D2H.
NEWTON_K = 3


def _set_packed(enabled: bool) -> None:
    newton_mod.uses_packed_vjp = _uses_packed_vjp if enabled else lambda _cell: False


def _max_err(a: Tensor | None, b: Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a.detach() - b.detach()).abs().amax())


def _vjp_quality(cell: ParaSLSTM, h_prev: Tensor, x: Tensor, mu: Tensor) -> dict:
    gx_p, gp_p = cell_vjp(cell, h_prev, x, mu, packed=True)
    gx_a, gp_a = cell_vjp(cell, h_prev, x, mu, packed=False)
    err_x = _max_err(gx_p, gx_a)
    err_p = 0.0
    for a, b in zip(gp_p, gp_a, strict=True):
        err_p = max(err_p, _max_err(a, b))
    return {"err_x": err_x, "err_params": err_p}


def _vjp_time(cell: ParaSLSTM, h_prev: Tensor, x: Tensor, mu: Tensor, packed: bool) -> dict:
    def fn() -> None:
        cell_vjp(cell, h_prev, x, mu, packed=packed)

    tmin, tmed, tmean = cuda_minmax(fn, warmup=WARMUP, n_runs=N_RUNS)
    tag = "packed" if packed else "autograd"
    log.info(
        "vjp/%s  min=%.3f ms  median=%.3f  mean=%.3f",
        tag,
        tmin,
        tmed,
        tmean,
    )
    return {"min_ms": tmin, "median_ms": tmed, "mean_ms": tmean}


def _newton_fwd_bwd_time(cell: ParaSLSTM, x: Tensor, cfg: NewtonConfig, packed: bool) -> dict:
    _set_packed(packed)
    x_in = x.detach().requires_grad_(True)

    def fwd_fn() -> None:
        with torch.no_grad():
            newton_apply(cell, x_in.detach(), cfg)

    def rebuild() -> Tensor:
        if x_in.grad is not None:
            x_in.grad = None
        cell.zero_grad(set_to_none=True)
        return newton_apply(cell, x_in, cfg)

    fmin, fmed, fmean = cuda_minmax(fwd_fn, warmup=WARMUP, n_runs=N_RUNS)
    for _ in range(WARMUP):
        rebuild().square().sum().backward()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(N_RUNS):
        y = rebuild()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        y.square().sum().backward()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    bmin = samples[0]
    bmed = samples[len(samples) // 2]
    bmean = sum(samples) / len(samples)
    tag = "packed" if packed else "autograd"
    log.info(
        "newton/%s  fwd min=%.3f med=%.3f  bwd min=%.3f med=%.3f",
        tag,
        fmin,
        fmed,
        bmin,
        bmed,
    )
    return {
        "fwd_min_ms": fmin,
        "fwd_median_ms": fmed,
        "fwd_mean_ms": fmean,
        "bwd_min_ms": bmin,
        "bwd_median_ms": bmed,
        "bwd_mean_ms": bmean,
    }


def _dyck_train(device: torch.device, packed: bool, *, seed: int = 0) -> dict:
    _set_packed(packed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = NewtonConfig(max_iters=NEWTON_K, scan_backend="fused")
    model = _NewtonDyckLM(DYCK_DH, cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=DYCK_LR, weight_decay=0.0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    bwd_ms: list[float] = []
    for step in range(DYCK_STEPS):
        tokens = sample_dyck1(DYCK_BATCH, DYCK_T, generator=gen).to(device)
        logits = model(tokens[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)

        def _bwd(loss_t: Tensor = loss) -> None:
            loss_t.backward()
            opt.step()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        _bwd()
        end.record()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        bwd_ms.append(start.elapsed_time(end))
        log.info(
            "dyck/%s step=%02d ce=%.4f bwd=%.2f ms",
            "packed" if packed else "autograd",
            step,
            losses[-1],
            bwd_ms[-1],
        )
    warm = 5
    steady = bwd_ms[warm:]
    return {
        "ce0": losses[0],
        "ce_final": losses[-1],
        "bwd_min_ms": min(steady),
        "bwd_mean_ms": sum(steady) / len(steady),
        "losses": losses,
    }


def _shape_bundle(
    device: torch.device,
    *,
    batch: int,
    seq_len: int,
    d_h: int,
    picard: int,
) -> tuple[ParaSLSTM, Tensor, Tensor, Tensor, NewtonConfig, Tensor]:
    torch.manual_seed(59)
    cell = ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag").to(device)
    x = torch.randn(batch, seq_len, d_h, device=device)
    cfg = NewtonConfig(
        max_iters=NEWTON_K,
        scan_backend="fused",
        picard_iters=picard,
        picard_adapt=False,
        residual_atol=None,
        residual_fail=None,
    )
    with torch.no_grad():
        states = newton_apply(cell, x, cfg)
    h_prev = prepend_state(states, None)
    mu = torch.randn_like(states)
    return cell, h_prev, x, mu, cfg, states


def main() -> None:
    setup_logging()
    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    wait_until_free(device)
    log.info(
        "packed_vjp_bench gpu=%s warmup=%d n_runs=%d",
        torch.cuda.get_device_name(device),
        WARMUP,
        N_RUNS,
    )

    import mlflow

    mlflow.set_experiment("newton-slstm-bench")
    with mlflow.start_run(run_name="packed-vjp-vs-autograd"):
        mlflow.set_tags(
            {
                "gpu": torch.cuda.get_device_name(device),
                "cell": "para_slstm",
                "mix": "diag",
                "mode": "packed_vjp_ab",
                "dtype": "fp32",
            }
        )
        mlflow.log_params(
            {
                "warmup": WARMUP,
                "n_runs": N_RUNS,
                "newton_k": NEWTON_K,
                "git_commit": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
            }
        )
        mlflow.log_text(
            "Packed VJP is a faster eq. 2.6 cell VJP, not a better cell. "
            "Autograd on step is the quality oracle.\n",
            "why.txt",
        )

        shapes = (
            ("dyck", DYCK_BATCH, DYCK_T, DYCK_DH, 1),
            ("bench", BENCH_BATCH, BENCH_T, BENCH_DH, 3),
        )
        for name, batch, seq_len, d_h, picard in shapes:
            log.info("shape=%s B=%d T=%d d_h=%d P=%d", name, batch, seq_len, d_h, picard)
            cell, h_prev, x, mu, cfg, states = _shape_bundle(
                device, batch=batch, seq_len=seq_len, d_h=d_h, picard=picard
            )
            q = _vjp_quality(cell, h_prev, x, mu)
            log.info(
                "vjp quality %s  max|dx|=%.3e  max|dtheta|=%.3e",
                name,
                q["err_x"],
                q["err_params"],
            )
            mlflow.log_metric(f"{name}/vjp_err_x", q["err_x"])
            mlflow.log_metric(f"{name}/vjp_err_params", q["err_params"])
            for packed, tag in ((True, "packed"), (False, "autograd")):
                row = _vjp_time(cell, h_prev, x, mu, packed)
                mlflow.log_metric(f"{name}/vjp_{tag}_min_ms", row["min_ms"])
                mlflow.log_metric(f"{name}/vjp_{tag}_median_ms", row["median_ms"])

            # Same H: reverse scan is shared; only the cell VJP flag changes.
            # Two independent newton_apply solves are not a VJP A/B (fused
            # P=1/P=3 roots can disagree).
            partial = torch.randn_like(states)
            _set_packed(True)
            gx_p, gp_p, _ = _eq26_vjp(cell, states, x, partial, backend="triton")
            _set_packed(False)
            gx_a, gp_a, _ = _eq26_vjp(cell, states, x, partial, backend="triton")
            err_nx = _max_err(gx_p, gx_a)
            err_np = max(_max_err(a, b) for a, b in zip(gp_p, gp_a, strict=True))
            log.info(
                "eq26 quality %s  max|dx|=%.3e  max|dtheta|=%.3e",
                name,
                err_nx,
                err_np,
            )
            mlflow.log_metric(f"{name}/eq26_err_x", err_nx)
            mlflow.log_metric(f"{name}/eq26_err_params", err_np)

            for packed, tag in ((True, "packed"), (False, "autograd")):
                row = _newton_fwd_bwd_time(cell, x, cfg, packed)
                mlflow.log_metric(f"{name}/fwd_{tag}_min_ms", row["fwd_min_ms"])
                mlflow.log_metric(f"{name}/bwd_{tag}_min_ms", row["bwd_min_ms"])
                mlflow.log_metric(f"{name}/bwd_{tag}_median_ms", row["bwd_median_ms"])
            _set_packed(True)

        log.info("dyck 50-step train A/B (same seed, packed vs autograd VJP)")
        packed_train = _dyck_train(device, True)
        autograd_train = _dyck_train(device, False)
        ce_gap = abs(packed_train["ce_final"] - autograd_train["ce_final"])
        max_ce_gap = max(
            abs(a - b)
            for a, b in zip(packed_train["losses"], autograd_train["losses"], strict=True)
        )
        log.info(
            "dyck CE packed %.4f → %.4f  autograd %.4f → %.4f  "
            "max|ΔCE|=%.3e  bwd min packed=%.2f autograd=%.2f",
            packed_train["ce0"],
            packed_train["ce_final"],
            autograd_train["ce0"],
            autograd_train["ce_final"],
            max_ce_gap,
            packed_train["bwd_min_ms"],
            autograd_train["bwd_min_ms"],
        )
        mlflow.log_metric("dyck/ce0_packed", packed_train["ce0"])
        mlflow.log_metric("dyck/ce_final_packed", packed_train["ce_final"])
        mlflow.log_metric("dyck/ce0_autograd", autograd_train["ce0"])
        mlflow.log_metric("dyck/ce_final_autograd", autograd_train["ce_final"])
        mlflow.log_metric("dyck/max_abs_ce_gap", max_ce_gap)
        mlflow.log_metric("dyck/final_ce_gap", ce_gap)
        mlflow.log_metric("dyck/bwd_min_packed_ms", packed_train["bwd_min_ms"])
        mlflow.log_metric("dyck/bwd_min_autograd_ms", autograd_train["bwd_min_ms"])
        mlflow.log_metric("dyck/bwd_mean_packed_ms", packed_train["bwd_mean_ms"])
        mlflow.log_metric("dyck/bwd_mean_autograd_ms", autograd_train["bwd_mean_ms"])
        _set_packed(True)


if __name__ == "__main__":
    main()
