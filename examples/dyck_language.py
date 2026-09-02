"""Dyck-1 next-token smoke: ParaSLSTM + Newton grads.

    uv run python examples/dyck_language.py --config configs/train/dyck.yaml

Library contract: K=3, Picard P from T (here P=1). Fail-loud if Newton
diverges. Device is ``cuda`` if available, else CPU — not a lab GPU name.
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
from pararnn import NewtonConfig, ParaRNN, ParaSLSTM
from scripts.utils.mlflow_helper import ROOT, setup_logging

log = logging.getLogger("dyck")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "dyck.yaml"
device = smoke_device()
OPEN, CLOSE = 0, 1
VOCAB = 2


def sample_dyck1(batch: int, length: int, *, generator: torch.Generator | None = None) -> Tensor:
    """Even-length Dyck-1 words. Token 0='(', 1=')'."""
    if length < 2 or length % 2:
        raise ValueError(f"Dyck-1 length must be even and >=2, got {length}")
    out = torch.empty(batch, length, dtype=torch.long)
    for b in range(batch):
        depth = 0
        for t in range(length):
            remain = length - t
            if depth == 0:
                tok = OPEN
            elif remain == depth:
                tok = CLOSE
            else:
                tok = int(torch.randint(0, 2, (1,), generator=generator).item())
                if tok == CLOSE and depth == 0:
                    tok = OPEN
            depth += 1 if tok == OPEN else -1
            out[b, t] = tok
        if depth != 0:
            raise RuntimeError(f"Dyck sampler ended at depth={depth}")
    return out


class _DyckLM(nn.Module):
    def __init__(self, d_h: int, newton_cfg: NewtonConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_h)
        self.rnn = ParaRNN(
            ParaSLSTM(d_in=d_h, d_h=d_h, mix="diag"),
            config=newton_cfg,
            output_hidden=True,
        )
        self.head = nn.Linear(d_h, VOCAB)

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
        "dyck_start gpu=%s torch=%s lrs=%s scan_backend=%s",
        gpu_name,
        torch.__version__,
        lrs,
        scan_backend,
    )

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    with mlflow.start_run(run_name=str(spec.get("mlflow_run_name", "dyck1"))):
        mlflow.set_tags(
            {
                "cell": "para_slstm",
                "mix": "diag",
                "gpu": gpu_name,
                "dtype": str(spec["dtype"]),
                "task": str(spec["task"]),
            }
        )
        mlflow.log_params(
            {
                "seq_len": spec["seq_len"],
                "batch": spec["batch"],
                "d_h": spec["d_h"],
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
                f"dyck smoke failed: loss at step {len(last_losses) - 1} "
                f"({last_losses[-1]:.4f}) >= loss at step 0 ({last_losses[0]:.4f}) "
                f"after lrs {lrs}"
            )
        log.info(
            "dyck_ok lr=%s loss0=%.4f loss_final=%.4f scan_backend=%s",
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
    seq_len = int(spec["seq_len"])
    batch = int(spec["batch"])
    d_h = int(spec["d_h"])
    steps = int(spec["steps"])
    newton_cfg = NewtonConfig(
        max_iters=int(spec["newton_iters"]),
        scan_backend=scan_backend,
    )
    model = _DyckLM(d_h, newton_cfg).to(device)
    model.train()
    _check_grads_finite(model, device, seq_len=min(seq_len, 8))
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(spec["weight_decay"]),
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    residuals: list[float] = []
    for step in range(steps + 1):
        tokens = sample_dyck1(batch, seq_len, generator=gen).to(device)
        logits = model(tokens[:, :-1])
        residual, _ = newton_residual(model.rnn)
        resolved = resolved_scan_backend(model.rnn, scan_backend)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
        loss_f = float(loss.detach())
        losses.append(loss_f)
        residuals.append(residual)
        log.info(
            "train_step step=%d loss=%.4f residual=%.3e lr=%s backend=%s seq_len=%d",
            step,
            loss_f,
            residual,
            lr,
            resolved,
            seq_len,
        )
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return losses, residuals, resolved_scan_backend(model.rnn, scan_backend)


def _check_grads_finite(model: _DyckLM, device: torch.device, *, seq_len: int) -> None:
    """One backward: all grads finite. Eq. 2.6, not autograd through K."""
    tokens = sample_dyck1(4, seq_len, generator=torch.Generator().manual_seed(0)).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    loss.backward()
    bad = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    if bad:
        raise RuntimeError(f"non-finite or missing grads: {bad}")
    log.info(
        "dyck_grads_finite n_params=%d loss=%.4f",
        len(list(model.parameters())),
        float(loss.detach()),
    )
    model.zero_grad(set_to_none=True)


def _validate_spec(spec: dict) -> None:
    if spec.get("dtype") != "float32":
        raise ValueError("dyck smoke is float32")
    if spec.get("cell") != "para_slstm":
        raise ValueError("dyck smoke uses ParaSLSTM")
    if int(spec.get("seq_len", 0)) % 2:
        raise ValueError("Dyck-1 seq_len must be even")
    if int(spec.get("newton_iters", 0)) != 3:
        raise ValueError("library contract is newton_iters=3")


if __name__ == "__main__":
    main()
