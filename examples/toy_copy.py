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

from examples._harness import (
    configure_cuda_device,
    gpu_label,
    log_smoke_train_metrics,
    lr_candidates,
    mlflow_repro_params,
    newton_residual,
    resolve_scan_backend,
    resolved_scan_backend,
    run_lr_backend_fallback,
    smoke_device,
)
from pararnn import NewtonConfig, ParaGRU, ParaRNN
from scripts.utils.mlflow_helper import ROOT, setup_logging

log = logging.getLogger("toy")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "toy.yaml"
device = smoke_device()


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

    configure_cuda_device(device)
    gpu_name = gpu_label(device)
    scan_backend = resolve_scan_backend(spec, device)
    lrs = lr_candidates(spec)
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
                **mlflow_repro_params(args.config),
            }
        )
        mlflow.log_artifact(str(args.config))
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")

        last_losses, last_residuals, used_backend, used_lr = run_lr_backend_fallback(
            spec,
            device,
            _train,
            scan_backend=scan_backend,
            gpu_name=gpu_name,
        )
        log_smoke_train_metrics(
            mlflow, last_losses, last_residuals, used_lr=used_lr, used_backend=used_backend
        )
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
        residual, _ = newton_residual(model.rnn)
        resolved = resolved_scan_backend(model.rnn, scan_backend)
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
    return losses, residuals, resolved_scan_backend(model.rnn, scan_backend)


def _validate_spec(spec: dict) -> None:
    if spec.get("dtype") != "float32":
        raise ValueError("toy smoke is float32 (see configs/train/toy.yaml)")
    if spec.get("cell") != "para_gru":
        raise ValueError("toy smoke uses ParaGRU")
    if int(spec.get("num_layers", 1)) != 1:
        raise ValueError("toy smoke is num_layers=1")


if __name__ == "__main__":
    main()
