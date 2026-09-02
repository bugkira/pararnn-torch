"""Shared smoke-train helpers for examples/. Not public API."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from scripts.utils.mlflow_helper import git_commit, lock_hash, uv_export_hash

log = logging.getLogger(__name__)

TrainFn = Callable[..., tuple[list[float], list[float], str]]


def ensure_repo_on_path() -> Path:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    return _REPO


def smoke_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_label(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def configure_cuda_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


def resolve_scan_backend(spec: dict, device: torch.device) -> str:
    scan_backend = str(spec["scan_backend"])
    if device.type != "cuda" and scan_backend == "fused":
        log.warning("fused_requires_cuda_using_eager")
        scan_backend = "eager"
    return scan_backend


def lr_candidates(spec: dict) -> list[float]:
    return [float(spec["lr"]), *[float(x) for x in spec.get("lr_fallback", [])]]


def mlflow_repro_params(config_path: Path) -> dict[str, str]:
    return {
        "git": git_commit(),
        "uv_lock": lock_hash(),
        "uv_export": uv_export_hash(),
        "config": str(config_path),
    }


def run_lr_backend_fallback(
    spec: dict,
    device: torch.device,
    train_fn: TrainFn,
    *,
    scan_backend: str,
    gpu_name: str,
) -> tuple[list[float], list[float], str, float]:
    lrs = lr_candidates(spec)
    last_losses: list[float] | None = None
    last_residuals: list[float] | None = None
    used_lr: float | None = None
    used_backend = scan_backend
    for lr in lrs:
        try:
            losses, residuals, used_backend = train_fn(
                spec, device, lr=lr, scan_backend=used_backend
            )
        except torch.cuda.OutOfMemoryError:
            if used_backend not in ("fused", "auto"):
                raise
            log.warning("fused_oom_fallback_eager gpu=%s (staying on this card)", gpu_name)
            torch.cuda.empty_cache()
            used_backend = "eager"
            losses, residuals, used_backend = train_fn(spec, device, lr=lr, scan_backend="eager")
        last_losses = losses
        last_residuals = residuals
        used_lr = lr
        if losses[-1] < losses[0]:
            break
        log.warning(
            "lr_did_not_drop lr=%s loss0=%s loss_final=%s",
            lr,
            losses[0],
            losses[-1],
        )
    assert last_losses is not None
    assert used_lr is not None
    return last_losses, last_residuals or [], used_backend, used_lr


def log_smoke_train_metrics(
    mlflow_mod,
    losses: list[float],
    residuals: list[float],
    *,
    used_lr: float,
    used_backend: str,
) -> None:
    for step, loss in enumerate(losses):
        mlflow_mod.log_metric("loss", loss, step=step)
        if residuals:
            mlflow_mod.log_metric("newton_residual", residuals[step], step=step)
    mlflow_mod.log_param("lr_used", used_lr)
    mlflow_mod.log_param("scan_backend_used", used_backend)


def newton_residual(rnn) -> tuple[float, str | None]:
    if rnn.last_stats:
        st = rnn.last_stats[0]
        return st.max_residual, st.scan_backend
    return float("nan"), None


def resolved_scan_backend(rnn, scan_backend: str) -> str:
    if rnn.last_stats:
        return rnn.last_stats[0].scan_backend or scan_backend
    return scan_backend
