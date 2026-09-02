"""Toy copy smoke: ParaRNN + linear head + CE + AdamW, MLflow-logged.

    uv run python examples/toy_copy.py --config configs/train/toy.yaml

Device is ``cuda`` if available, else CPU. Lab benches pin a GPU by name in
``scripts/gpu.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import NewtonConfig, ParaGRU, ParaRNN
from scripts.utils.mlflow_helper import (
    ROOT,
    git_commit,
    lock_hash,
    setup_logging,
    uv_export_hash,
)

log = logging.getLogger("toy")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "toy.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CopyLM(nn.Module):
    def __init__(self, vocab: int, d_h: int, newton_cfg: NewtonConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_h)
        self.rnn = ParaRNN(ParaGRU(d_in=d_h, d_h=d_h), config=newton_cfg)
        self.head = nn.Linear(d_h, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.head(self.rnn(self.embed(tokens)))


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    _validate_spec(spec)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "cpu"
    scan_backend = str(spec["scan_backend"])
    if device.type != "cuda" and scan_backend == "fused":
        log.warning("fused_requires_cuda_using_eager")
        scan_backend = "eager"

    lrs = [float(spec["lr"]), *[float(x) for x in spec.get("lr_fallback", [])]]
    log.info(
        "toy_start gpu=%s torch=%s lrs=%s scan_backend=%s",
        gpu_name,
        torch.__version__,
        lrs,
        scan_backend,
    )

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    with mlflow.start_run(run_name=str(spec.get("mlflow_run_name", "copy"))):
        mlflow.set_tags(
            {
                "cell": "para_gru",
                "gpu": gpu_name,
                "dtype": str(spec["dtype"]),
                "task": str(spec["task"]),
            }
        )
        mlflow.log_params(
            {
                "vocab": spec["vocab"],
                "seq_len": spec["seq_len"],
                "batch": spec["batch"],
                "d_h": spec["d_h"],
                "num_layers": spec["num_layers"],
                "steps": spec["steps"],
                "newton_iters": spec["newton_iters"],
                "lr": spec["lr"],
                "weight_decay": spec["weight_decay"],
                "dtype": spec["dtype"],
                "scan_backend": scan_backend,
                "seed": spec["seed"],
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "config": str(args.config),
            }
        )
        mlflow.log_artifact(str(args.config))
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")

        last_losses: list[float] | None = None
        last_residuals: list[float] | None = None
        used_lr: float | None = None
        used_backend = scan_backend
        for lr in lrs:
            try:
                losses, residuals, used_backend = _train(
                    spec, device, lr=lr, scan_backend=used_backend
                )
            except torch.cuda.OutOfMemoryError:
                if used_backend not in ("fused", "auto"):
                    raise
                log.warning(
                    "fused_oom_fallback_eager gpu=%s (staying on this card)",
                    gpu_name,
                )
                torch.cuda.empty_cache()
                used_backend = "eager"
                losses, residuals, used_backend = _train(spec, device, lr=lr, scan_backend="eager")
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
        for step, loss in enumerate(last_losses):
            mlflow.log_metric("loss", loss, step=step)
            if last_residuals is not None:
                mlflow.log_metric("newton_residual", last_residuals[step], step=step)
        mlflow.log_param("lr_used", used_lr)
        mlflow.log_param("scan_backend_used", used_backend)
        if last_losses[-1] >= last_losses[0]:
            raise RuntimeError(
                f"copy smoke failed: loss at step {len(last_losses) - 1} "
                f"({last_losses[-1]:.4f}) >= loss at step 0 ({last_losses[0]:.4f}) "
                f"after lrs {lrs}"
            )
        log.info(
            "toy_ok lr=%s loss0=%.4f loss_final=%.4f scan_backend=%s",
            used_lr,
            last_losses[0],
            last_losses[-1],
            used_backend,
        )


def _train(
    spec: dict,
    device: torch.device,
    *,
    lr: float,
    scan_backend: str,
) -> tuple[list[float], list[float], str]:
    seed = int(spec["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    vocab = int(spec["vocab"])
    seq_len = int(spec["seq_len"])
    batch = int(spec["batch"])
    d_h = int(spec["d_h"])
    steps = int(spec["steps"])
    newton_cfg = NewtonConfig(max_iters=int(spec["newton_iters"]), scan_backend=scan_backend)
    model = _CopyLM(vocab, d_h, newton_cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(spec["weight_decay"]),
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    residuals: list[float] = []
    for step in range(steps + 1):
        tokens = torch.randint(0, vocab, (batch, seq_len), generator=gen, device="cpu").to(device)
        logits = model(tokens)
        residual = float("nan")
        resolved = scan_backend
        if model.rnn.last_stats:
            residual = model.rnn.last_stats[0].max_residual
            resolved = model.rnn.last_stats[0].scan_backend or scan_backend
        loss = F.cross_entropy(logits.reshape(-1, vocab), tokens.reshape(-1))
        loss_f = float(loss.detach())
        losses.append(loss_f)
        residuals.append(residual)
        log.info(
            "train_step step=%d loss=%.4f residual=%.3e lr=%s backend=%s "
            "seq_len=%d batch=%d d_h=%d",
            step,
            loss_f,
            residual,
            lr,
            resolved,
            seq_len,
            batch,
            d_h,
        )
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "newton_residual",
                extra={"step": step, "max_residual": residual, "seq_len": seq_len},
            )
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    used = scan_backend
    if model.rnn.last_stats:
        used = model.rnn.last_stats[0].scan_backend or scan_backend
    return losses, residuals, used


def _validate_spec(spec: dict) -> None:
    if spec.get("dtype") != "float32":
        raise ValueError("toy smoke is float32 (see configs/train/toy.yaml)")
    if spec.get("cell") != "para_gru":
        raise ValueError("toy smoke uses ParaGRU")
    if int(spec.get("num_layers", 1)) != 1:
        raise ValueError("toy smoke is num_layers=1")


if __name__ == "__main__":
    main()
