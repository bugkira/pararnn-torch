"""Zero-shot TiRex: keep W_x / FFN / patch embed, replace sLSTM R with diag or 0.

NX-AI cell stores ``_recurrent_kernel_`` as ``(n_heads, d_head, n_gates * d_head)``.
No training. Forecast MAE vs a held-out continuation, plus agreement with the
unpatched model.

    CUDA_VISIBLE_DEVICES=0 uv run --with tirex-ts python scripts/eval_tirex_dropin_diag.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import torch
from torch import Tensor

from utils.mlflow_helper import ROOT, git_commit, lock_hash, uv_export_hash

from gpu import select_device, wait_until_free
from gpu import setup_logging as setup_gpu_logging

log = logging.getLogger("eval_tirex_dropin")

CTX = 512
HORIZON = 64
N_SERIES = 48
SEED = 0


def _recurrent_params(model: torch.nn.Module) -> list[tuple[str, Tensor]]:
    out = []
    for name, p in model.named_parameters():
        if name.endswith("slstm_cell._recurrent_kernel_"):
            out.append((name, p))
    if not out:
        raise RuntimeError("no sLSTM _recurrent_kernel_ tensors found")
    return out


def _as_blocks(kernel: Tensor) -> Tensor:
    """``(H, d_head, G * d_head)`` → ``(H, d_head, G, d_head)``."""
    heads, d_head, rest = kernel.shape
    if rest % d_head != 0:
        raise ValueError(f"kernel {tuple(kernel.shape)} is not (H, d, G*d)")
    n_gates = rest // d_head
    return kernel.view(heads, d_head, n_gates, d_head)


def _frob_energy(kernel: Tensor) -> dict[str, float]:
    blocks = _as_blocks(kernel.detach().float())
    diag = torch.diagonal(blocks, dim1=1, dim2=3)
    full = float(torch.linalg.vector_norm(blocks))
    dnorm = float(torch.linalg.vector_norm(diag))
    off = blocks.clone()
    off.diagonal(dim1=1, dim2=3).zero_()
    return {
        "frob": full,
        "frob_diag": dnorm,
        "frac_diag_sq": (dnorm * dnorm) / max(full * full, 1e-12),
        "frob_off": float(torch.linalg.vector_norm(off)),
    }


def _apply_diag_(kernel: Tensor) -> None:
    blocks = _as_blocks(kernel.data)
    keep = torch.zeros_like(blocks)
    keep.diagonal(dim1=1, dim2=3).copy_(blocks.diagonal(dim1=1, dim2=3))
    kernel.data.copy_(keep.reshape_as(kernel.data))


def _make_series(n: int, length: int, *, seed: int, device: torch.device) -> Tensor:
    """Deterministic mix of trend / season / AR — held-out tail is the target."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    series = []
    for _i in range(n):
        amp = torch.rand((), generator=g).item() * 2.0 + 0.5
        freq = math.pi * (0.01 + 0.04 * torch.rand((), generator=g).item())
        phase = torch.rand((), generator=g).item() * 2 * math.pi
        slope = (torch.rand((), generator=g).item() - 0.5) * 0.002
        noise_s = 0.05 + 0.1 * torch.rand((), generator=g).item()
        ar = 0.4 + 0.5 * torch.rand((), generator=g).item()
        eps = torch.randn(length, generator=g)
        ar_path = torch.zeros(length)
        for k in range(1, length):
            ar_path[k] = ar * ar_path[k - 1] + eps[k]
        y = amp * torch.sin(freq * t[0] + phase) + slope * t[0] + 0.3 * ar_path + noise_s * eps
        series.append(y)
    return torch.stack(series, dim=0).to(device)


def _mae(a: Tensor, b: Tensor) -> float:
    return float((a - b).abs().mean())


def _mse(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).mean())


def _smape(pred: Tensor, truth: Tensor) -> float:
    denom = pred.abs() + truth.abs()
    return float((2.0 * (pred - truth).abs() / denom.clamp_min(1e-6)).mean())


@torch.no_grad()
def _forecast_mean(model, context: Tensor, horizon: int) -> Tensor:
    _q, mean = model.forecast(context=context, prediction_length=horizon)
    return mean.detach().float().cpu()


def main() -> None:
    setup_gpu_logging()
    device = select_device("3060")
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=4.0, poll_s=30.0)

    from tirex import load_model

    log.info("tirex_dropin_load hub=NX-AI/TiRex backend=torch ctx=%d horizon=%d", CTX, HORIZON)
    model = load_model("NX-AI/TiRex", device=str(device), backend="torch", compile=False)
    rec = _recurrent_params(model)
    energies = [_frob_energy(p) for _, p in rec]
    frac = sum(e["frac_diag_sq"] for e in energies) / len(energies)
    log.info(
        "tirex_dropin_R n_layers=%d mean_frac_diag_sq=%.4f frob0=%.3f",
        len(rec),
        frac,
        energies[0]["frob"],
    )
    snapshots = [p.detach().clone() for _, p in rec]

    full = CTX + HORIZON
    series = _make_series(N_SERIES, full, seed=SEED, device=device)
    ctx, truth = series[:, :CTX], series[:, CTX:].cpu()

    mean_orig = _forecast_mean(model, ctx, HORIZON)
    log.info(
        "tirex_dropin arm=orig mae=%.5f mse=%.5f smape=%.5f",
        _mae(mean_orig, truth),
        _mse(mean_orig, truth),
        _smape(mean_orig, truth),
    )

    def restore() -> None:
        for (_, p), snap in zip(rec, snapshots, strict=True):
            p.data.copy_(snap)

    arms: dict[str, dict] = {}

    def eval_arm(tag: str, patch) -> None:
        restore()
        if patch is not None:
            for _, p in rec:
                patch(p)
        mean = _forecast_mean(model, ctx, HORIZON)
        row = {
            "mae_truth": _mae(mean, truth),
            "mse_truth": _mse(mean, truth),
            "smape_truth": _smape(mean, truth),
            "mae_vs_orig": _mae(mean, mean_orig),
            "mse_vs_orig": _mse(mean, mean_orig),
        }
        arms[tag] = row
        log.info(
            "tirex_dropin arm=%s mae=%.5f smape=%.5f mae_vs_orig=%.5f",
            tag,
            row["mae_truth"],
            row["smape_truth"],
            row["mae_vs_orig"],
        )

    eval_arm("orig", None)
    eval_arm("diag", _apply_diag_)
    eval_arm("zero", lambda p: p.data.zero_())

    result = {
        "task": "tirex_dropin_diag",
        "hub": "NX-AI/TiRex",
        "backend": "torch",
        "n_series": N_SERIES,
        "ctx": CTX,
        "horizon": HORIZON,
        "gpu": torch.cuda.get_device_name(device),
        "n_slstm": len(rec),
        "mean_frac_diag_sq": frac,
        "kernel0": energies[0],
        "arms": arms,
        "delta_mae_diag": arms["diag"]["mae_truth"] - arms["orig"]["mae_truth"],
        "delta_mae_zero": arms["zero"]["mae_truth"] - arms["orig"]["mae_truth"],
        "delta_smape_diag": arms["diag"]["smape_truth"] - arms["orig"]["smape_truth"],
    }
    out = ROOT / "results" / "tirex_dropin_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    import mlflow

    mlflow.set_experiment("tirex-dropin")
    with mlflow.start_run(run_name="tirex-dropin-diag"):
        mlflow.log_params(
            {
                "hub": "NX-AI/TiRex",
                "ctx": CTX,
                "horizon": HORIZON,
                "n_series": N_SERIES,
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "gpu": result["gpu"],
            }
        )
        mlflow.log_metric("mae_orig", arms["orig"]["mae_truth"])
        mlflow.log_metric("mae_diag", arms["diag"]["mae_truth"])
        mlflow.log_metric("mae_zero", arms["zero"]["mae_truth"])
        mlflow.log_metric("smape_orig", arms["orig"]["smape_truth"])
        mlflow.log_metric("smape_diag", arms["diag"]["smape_truth"])
        mlflow.log_metric("mae_diag_vs_orig", arms["diag"]["mae_vs_orig"])
        mlflow.log_metric("mean_frac_diag_sq", frac)
        mlflow.log_artifact(str(out))
        mlflow.log_text(
            "Zero-shot: TiRex sLSTM recurrent_kernel → diag or 0. "
            "Input gates / FFN / patch embed unchanged. Synthetic mix "
            f"N={N_SERIES} ctx={CTX} horizon={HORIZON}.\n",
            "why.txt",
        )
    log.info("tirex_dropin_done wrote=%s delta_mae_diag=%+.5f", out, result["delta_mae_diag"])


if __name__ == "__main__":
    main()
