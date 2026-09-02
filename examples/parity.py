"""Running parity (Z2 tagging): stacked xLSTMBlock vs sequential vs SSM.

    uv run python examples/parity.py --config configs/train/parity_t16.yaml

Merrill et al. 2024 §5: token-tagging, label at t is the prefix product.
Here the group is Z2 (XOR). Not A5. Not FlashRNN. Lab GPU: 2080 Ti by name.
T=32 / 300 steps (`configs/train/parity.yaml`) stays copy-only; use T=16 /
2000 steps for the quality smoke.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pararnn import NewtonConfig, xLSTMBlock
from pararnn.solvers.scan import scan_diag
from scripts.utils.mlflow_helper import (
    ROOT,
    git_commit,
    lock_hash,
    setup_logging,
    uv_export_hash,
)

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, wait_until_free

log = logging.getLogger("parity")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "parity_t16.yaml"
VOCAB = 2


def sample_parity(
    batch: int, length: int, *, generator: torch.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Bits and running XOR labels (Merrill §5 tagging on Z2)."""
    bits = torch.randint(0, 2, (batch, length), generator=generator)
    labels = bits.cumsum(dim=1) % 2
    return bits, labels


def _cosine_lr(step: int, total: int, base: float, warmup_frac: float) -> float:
    """ParaRNN App. C / Bergsma: 10% warmup, cosine to 0."""
    warm = max(int(total * warmup_frac), 1)
    if step < warm:
        return base * float(step + 1) / float(warm)
    t = float(step - warm) / float(max(total - warm, 1))
    return base * 0.5 * (1.0 + math.cos(math.pi * t))


class _S6Block(nn.Module):
    """Pre-norm residual diagonal selective SSM. Not mamba-ssm.

    h_t = exp(-Δ_t A) ⊙ h_{t-1} + (B_t ⊙ x_t). A is S4D-Real (n+1).
    Δ range Gu & Dao 2023 [1e-3, 1e-1]. ``scan_diag`` eager so Autograd
    flows (Triton scan has no backward). Linear SSM: Merrill TC^0 bound.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_model)
        self.C_proj = nn.Linear(d_model, d_model)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_model + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.linspace(math.log(1e-3), math.log(1e-1), d_model))

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        dt = F.softplus(self.dt_proj(z) + self.dt_bias)
        decay = torch.exp(-dt * torch.exp(self.log_A))
        drive = self.B_proj(z) * z
        h = scan_diag(decay, drive, backend="eager")
        return x + self.C_proj(h)


class _ParityNet(nn.Module):
    def __init__(
        self,
        d_h: int,
        num_layers: int,
        arm: str,
        newton_cfg: NewtonConfig,
        max_recurrent_norm: float | None,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.embed = nn.Embedding(VOCAB, d_h)
        if arm == "ssm":
            self.blocks = nn.ModuleList(_S6Block(d_h) for _ in range(num_layers))
        else:
            solver = "auto" if arm == "newton" else "sequential"
            self.blocks = nn.ModuleList(
                xLSTMBlock(
                    d_h,
                    solver=solver,
                    mix="diag",
                    max_recurrent_norm=max_recurrent_norm,
                    config=newton_cfg,
                )
                for _ in range(num_layers)
            )
        self.head = nn.Linear(d_h, VOCAB)

    def forward(self, tokens: Tensor) -> Tensor:
        h = self.embed(tokens)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def newton_residuals(self) -> list[float]:
        out: list[float] = []
        if self.arm != "newton":
            return out
        for block in self.blocks:
            if not isinstance(block, xLSTMBlock):
                continue
            for st in block.rnn.last_stats:
                out.append(st.max_residual)
        return out


@torch.no_grad()
def _eval_acc(model: nn.Module, bits: Tensor, labels: Tensor) -> dict[str, float]:
    """Overall / first-token / last-token / exact-sequence accuracy.

    t=0 is a copy of the bit (trivial). Last token is the full prefix XOR
    (Merrill tagging). Overall ~0.5 + 0.5/T is copy-only.
    """
    was_train = model.training
    model.eval()
    pred = model(bits).argmax(dim=-1)
    hit = pred == labels
    out = {
        "tok": float(hit.float().mean()),
        "exact": float(hit.all(dim=1).float().mean()),
        "tok_t0": float(hit[:, 0].float().mean()),
        "tok_last": float(hit[:, -1].float().mean()),
    }
    model.train(was_train)
    return out


def _train_arm(
    spec: dict,
    device: torch.device,
    *,
    arm: str,
    lr: float,
) -> dict:
    seed = int(spec["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = NewtonConfig(
        max_iters=int(spec["newton_iters"]),
        scan_backend=str(spec["scan_backend"]),
    )
    clip = spec.get("max_recurrent_norm")
    model = _ParityNet(
        int(spec["d_h"]),
        int(spec["num_layers"]),
        arm,
        cfg,
        None if clip is None else float(clip),
    ).to(device)
    model.train()
    betas = tuple(float(x) for x in spec["adam_betas"])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(spec["weight_decay"]),
        betas=betas,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    eval_gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    steps = int(spec["steps"])
    batch = int(spec["batch"])
    seq_len = int(spec["seq_len"])
    eval_len = int(spec["eval_seq_len"])
    warmup_frac = float(spec["warmup_frac"])
    losses: list[float] = []
    token_acc: list[float] = []
    n_params = sum(p.numel() for p in model.parameters())
    log.info("parity_arm=%s n_params=%d lr=%g", arm, n_params, lr)

    eval_bits, eval_lab = sample_parity(batch * 4, seq_len, generator=eval_gen)
    long_bits, long_lab = sample_parity(batch * 4, eval_len, generator=eval_gen)
    eval_bits, eval_lab = eval_bits.to(device), eval_lab.to(device)
    long_bits, long_lab = long_bits.to(device), long_lab.to(device)

    for step in range(steps):
        bits, labels = sample_parity(batch, seq_len, generator=gen)
        bits, labels = bits.to(device), labels.to(device)
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr, warmup_frac)
        logits = model(bits)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            token_acc.append(float((logits.argmax(-1) == labels).float().mean()))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            tr_tok = token_acc[-1]
            res = model.newton_residuals()
            ev = _eval_acc(model, eval_bits, eval_lab)
            lg = _eval_acc(model, long_bits, long_lab)
            log.info(
                "arm=%s step=%03d ce=%.4f train_tok=%.3f "
                "eval_tok=%.3f t0=%.3f last=%.3f exact=%.3f "
                "long_tok=%.3f long_last=%.3f long_exact=%.3f res=%s",
                arm,
                step,
                losses[-1],
                tr_tok,
                ev["tok"],
                ev["tok_t0"],
                ev["tok_last"],
                ev["exact"],
                lg["tok"],
                lg["tok_last"],
                lg["exact"],
                [f"{r:.2e}" for r in res] if res else "-",
            )
    ev = _eval_acc(model, eval_bits, eval_lab)
    lg = _eval_acc(model, long_bits, long_lab)
    return {
        "n_params": n_params,
        "loss0": losses[0],
        "loss_final": losses[-1],
        "train_tok_final": token_acc[-1],
        "eval_tok": ev["tok"],
        "eval_exact": ev["exact"],
        "eval_t0": ev["tok_t0"],
        "eval_last": ev["tok_last"],
        "long_tok": lg["tok"],
        "long_exact": lg["exact"],
        "long_last": lg["tok_last"],
        "losses": losses,
        "token_acc": token_acc,
    }


def _validate_spec(spec: dict) -> None:
    if spec.get("task") != "running_parity":
        raise ValueError(f"expected task=running_parity, got {spec.get('task')!r}")
    if int(spec["seq_len"]) < 2:
        raise ValueError("seq_len must be >= 2")
    if int(spec["eval_seq_len"]) < int(spec["seq_len"]):
        raise ValueError("eval_seq_len must be >= seq_len")
    if int(spec["num_layers"]) < 1:
        raise ValueError("num_layers must be >= 1")
    arms = spec.get("arms") or []
    if not arms:
        raise ValueError("arms must be a non-empty list")
    for a in arms:
        if a not in ("newton", "eager", "ssm"):
            raise ValueError(f"unknown arm {a!r}")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    _validate_spec(spec)

    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=2.0)
    gpu_name = torch.cuda.get_device_name(device)
    scan_backend = str(spec["scan_backend"])
    log.info("parity_start gpu=%s scan_backend=%s", gpu_name, scan_backend)

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    with mlflow.start_run(run_name=str(spec.get("mlflow_run_name", "parity"))):
        mlflow.set_tags(
            {
                "task": "running_parity",
                "mix": str(spec["mix"]),
                "gpu": gpu_name,
                "dtype": str(spec["dtype"]),
            }
        )
        mlflow.log_params(
            {
                "seq_len": spec["seq_len"],
                "eval_seq_len": spec["eval_seq_len"],
                "batch": spec["batch"],
                "d_h": spec["d_h"],
                "num_layers": spec["num_layers"],
                "steps": spec["steps"],
                "newton_iters": spec["newton_iters"],
                "lr": spec["lr"],
                "weight_decay": spec["weight_decay"],
                "scan_backend": scan_backend,
                "seed": spec["seed"],
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
            }
        )
        mlflow.log_artifact(str(args.config))
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")

        lrs = [float(spec["lr"]), *[float(x) for x in spec.get("lr_fallback", [])]]
        for arm in spec["arms"]:
            used: dict | None = None
            used_lr = lrs[0]
            for lr in lrs:
                row = _train_arm(spec, device, arm=arm, lr=lr)
                used = row
                used_lr = lr
                if row["eval_tok"] > 0.6:
                    break
                if lr == lrs[-1]:
                    log.warning(
                        "arm=%s lr=%s eval_tok=%.3f last=%.3f still near chance",
                        arm,
                        lr,
                        row["eval_tok"],
                        row["eval_last"],
                    )
                    break
                log.warning(
                    "arm=%s lr=%s eval_tok=%.3f last=%.3f still near chance; trying fallback",
                    arm,
                    lr,
                    row["eval_tok"],
                    row["eval_last"],
                )
            assert used is not None
            mlflow.log_param(f"{arm}_lr_used", used_lr)
            mlflow.log_param(f"{arm}_n_params", used["n_params"])
            mlflow.log_metric(f"{arm}/loss0", used["loss0"])
            mlflow.log_metric(f"{arm}/loss_final", used["loss_final"])
            mlflow.log_metric(f"{arm}/eval_tok", used["eval_tok"])
            mlflow.log_metric(f"{arm}/eval_exact", used["eval_exact"])
            mlflow.log_metric(f"{arm}/eval_t0", used["eval_t0"])
            mlflow.log_metric(f"{arm}/eval_last", used["eval_last"])
            mlflow.log_metric(f"{arm}/long_tok", used["long_tok"])
            mlflow.log_metric(f"{arm}/long_exact", used["long_exact"])
            mlflow.log_metric(f"{arm}/long_last", used["long_last"])
            for i, (ce, acc) in enumerate(zip(used["losses"], used["token_acc"], strict=True)):
                if i % 10 == 0 or i + 1 == len(used["losses"]):
                    mlflow.log_metric(f"{arm}/loss", ce, step=i)
                    mlflow.log_metric(f"{arm}/train_tok", acc, step=i)
            log.info(
                "arm=%s done eval_tok=%.3f t0=%.3f last=%.3f exact=%.3f "
                "long_tok=%.3f long_last=%.3f long_exact=%.3f",
                arm,
                used["eval_tok"],
                used["eval_t0"],
                used["eval_last"],
                used["eval_exact"],
                used["long_tok"],
                used["long_last"],
                used["long_exact"],
            )


if __name__ == "__main__":
    main()
