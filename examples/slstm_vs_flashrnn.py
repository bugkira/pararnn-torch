"""Dyck-1 train: ParaSLSTM Newton vs NX-AI FlashRNN.

    uv sync --extra flashrnn --group dev
    uv run python examples/slstm_vs_flashrnn.py --config configs/train/dyck_vs_flashrnn.yaml
    uv run python examples/slstm_vs_flashrnn.py --config configs/train/dyck_vs_flashrnn_head.yaml

FlashRNN stays in this example / ``scripts/``, not ``xLSTMBlock``. Turing
has no ``cuda_fused`` (CC 8.0); this run uses ``triton_fused``. Head config
is mix=head, n_heads=1, K=4 — same 1×32 mixing as FlashRNN, not fused diag.
"""

from __future__ import annotations

import argparse
import logging
import os
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

from examples.dyck_language import VOCAB, sample_dyck1
from pararnn import NewtonConfig, ParaRNN, ParaSLSTM
from pararnn.weight_init import kaiming_uniform_linear_
from scripts.utils.mlflow_helper import (
    ROOT,
    git_commit,
    lock_hash,
    setup_logging,
    uv_export_hash,
)

from gpu import DEFAULT_EXPERIMENT_GPU_NAME, select_device, wait_until_free

log = logging.getLogger("slstm_vs_fr")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "dyck_vs_flashrnn.yaml"

# FlashRNN vanilla sLSTM unbinds (y, c, n, m); hidden is slot 0 (not our
# layout (c, n, m, h) in pararnn.layout).
_FR_HIDDEN = 0
_FR_GATES = 4
_FR_STATES = 4


class _NewtonDyckLM(nn.Module):
    def __init__(
        self,
        d_h: int,
        newton_cfg: NewtonConfig,
        *,
        mix: str = "diag",
        n_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_h)
        self.rnn = ParaRNN(
            ParaSLSTM(d_in=d_h, d_h=d_h, mix=mix, n_heads=n_heads),
            config=newton_cfg,
            output_hidden=True,
        )
        self.head = nn.Linear(d_h, VOCAB)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.head(self.rnn(self.embed(tokens)))


class _FlashRNNDyckLM(nn.Module):
    """Sequential sLSTM via FlashRNN. Not a library cell. Head mix, not diag."""

    def __init__(self, d_h: int, n_heads: int, d_head: int, backend: str) -> None:
        super().__init__()
        if n_heads * d_head != d_h:
            raise ValueError(f"n_heads*d_head={n_heads * d_head} != d_h={d_h}")
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_head = d_head
        self.backend = backend
        self.embed = nn.Embedding(VOCAB, d_h)
        self.W_x = nn.Linear(d_h, _FR_GATES * d_h, bias=True)
        self.R = nn.Parameter(torch.empty(_FR_GATES, n_heads, d_head, d_head))
        self.b = nn.Parameter(torch.zeros(_FR_GATES, n_heads, d_head))
        self.head = nn.Linear(d_h, VOCAB)
        kaiming_uniform_linear_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        for g in range(_FR_GATES):
            for hd in range(n_heads):
                nn.init.orthogonal_(self.R[g, hd], gain=0.25)

    def forward(self, tokens: Tensor) -> Tensor:
        from flashrnn import flashrnn

        x = self.embed(tokens)
        batch, time, _ = x.shape
        wx = self.W_x(x).reshape(batch, time, _FR_GATES, self.n_heads, self.d_head)
        s0 = x.new_zeros(_FR_STATES, batch, 1, self.n_heads, self.d_head)
        states, _ = flashrnn(
            wx,
            self.R,
            self.b,
            states=s0,
            function="slstm",
            backend=self.backend,
            dtype="float32",
        )
        h = states[_FR_HIDDEN].reshape(batch, time, self.d_h)
        return self.head(h)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    _validate_spec(spec)

    device = select_device(DEFAULT_EXPERIMENT_GPU_NAME)
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=4.0, poll_s=30.0)
    gpu_name = torch.cuda.get_device_name(device)
    scan_backend = str(spec["scan_backend"])
    fr_backend = _flashrnn_backend()
    lrs = [float(spec["lr"]), *[float(x) for x in spec.get("lr_fallback", [])]]
    log.info(
        "start gpu=%s torch=%s lrs=%s scan_backend=%s flashrnn=%s",
        gpu_name,
        torch.__version__,
        lrs,
        scan_backend,
        fr_backend,
    )

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    with mlflow.start_run(run_name=str(spec.get("mlflow_run_name", "dyck-vs-fr"))):
        mlflow.set_tags(
            {
                "task": str(spec["task"]),
                "gpu": gpu_name,
                "dtype": str(spec["dtype"]),
                "cell": "para_slstm",
                "mix": str(spec.get("mix", "diag")),
                "flashrnn_backend": fr_backend,
                "mode": "train",
            }
        )
        mlflow.log_params(
            {
                "seq_len": spec["seq_len"],
                "batch": spec["batch"],
                "d_h": spec["d_h"],
                "steps": spec["steps"],
                "warmup_time_steps": spec["warmup_time_steps"],
                "newton_iters": spec["newton_iters"],
                "scan_backend": scan_backend,
                "seed": spec["seed"],
                "lr": spec["lr"],
                "weight_decay": spec["weight_decay"],
                "mix": str(spec.get("mix", "diag")),
                "n_heads": spec.get("n_heads") if spec.get("n_heads") is not None else "none",
                "picard_iters": spec.get("picard_iters")
                if spec.get("picard_iters") is not None
                else "auto",
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "config": str(args.config),
            }
        )
        mlflow.log_artifact(str(args.config))
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")

        used_lr: float | None = None
        summaries: dict[str, dict] = {}
        for lr in lrs:
            summaries = {}
            ok = True
            for backend in spec["backends"]:
                row = _train(
                    spec,
                    device,
                    lr=lr,
                    backend=str(backend),
                    scan_backend=scan_backend,
                    flashrnn_backend=fr_backend,
                )
                summaries[str(backend)] = row
                if row["losses"][-1] >= row["losses"][0]:
                    ok = False
            used_lr = lr
            if ok:
                break
            log.warning("lr_did_not_drop lr=%s", lr)
        assert used_lr is not None and summaries
        mlflow.log_param("lr_used", used_lr)
        for name, row in summaries.items():
            for step, loss in enumerate(row["losses"]):
                mlflow.log_metric(f"{name}/loss", loss, step=step)
                if row["residuals"][step] == row["residuals"][step]:
                    mlflow.log_metric(
                        f"{name}/newton_residual", row["residuals"][step], step=step
                    )
            for step, dt in enumerate(row["step_ms"]):
                mlflow.log_metric(f"{name}/step_ms", dt, step=step)
            timed = row["step_ms"][int(spec["warmup_time_steps"]) :]
            min_ms = min(timed)
            mean_ms = sum(timed) / len(timed)
            mlflow.log_metric(f"{name}/min_step_ms", min_ms)
            mlflow.log_metric(f"{name}/mean_step_ms", mean_ms)
            mlflow.log_metric(f"{name}/peak_mib", row["peak_mib"])
            mlflow.log_metric(f"{name}/loss0", row["losses"][0])
            mlflow.log_metric(f"{name}/loss_final", row["losses"][-1])
            n_params = row["n_params"]
            mlflow.log_param(f"{name}_n_params", n_params)
            log.info(
                "%s lr=%s loss0=%.4f loss_final=%.4f min_step=%.3f ms "
                "mean_step=%.3f ms peak=%.1f MiB n_params=%d",
                name,
                used_lr,
                row["losses"][0],
                row["losses"][-1],
                min_ms,
                mean_ms,
                row["peak_mib"],
                n_params,
            )
        if "newton" in summaries and "flashrnn" in summaries:
            n_ms = min(summaries["newton"]["step_ms"][int(spec["warmup_time_steps"]) :])
            f_ms = min(summaries["flashrnn"]["step_ms"][int(spec["warmup_time_steps"]) :])
            ratio = n_ms / f_ms if f_ms else float("nan")
            mlflow.log_metric("newton_vs_flashrnn_min_step", ratio)
            log.info("newton/flashrnn min_step ratio=%.3f (>1 = FlashRNN faster)", ratio)
        failed = [
            name
            for name, row in summaries.items()
            if row["losses"][-1] >= row["losses"][0]
        ]
        if failed:
            raise RuntimeError(f"loss did not drop for {failed} after lrs {lrs}")


def _train(
    spec: dict,
    device: torch.device,
    *,
    lr: float,
    backend: str,
    scan_backend: str,
    flashrnn_backend: str,
) -> dict:
    seed = int(spec["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seq_len = int(spec["seq_len"])
    batch = int(spec["batch"])
    d_h = int(spec["d_h"])
    steps = int(spec["steps"])
    if backend == "newton":
        mix = str(spec.get("mix", "diag"))
        n_heads = spec.get("n_heads")
        n_heads_i = None if n_heads is None else int(n_heads)
        cfg = NewtonConfig(
            max_iters=int(spec["newton_iters"]),
            scan_backend=scan_backend,
            picard_iters=None
            if spec.get("picard_iters") is None
            else int(spec["picard_iters"]),
        )
        model: nn.Module = _NewtonDyckLM(
            d_h, cfg, mix=mix, n_heads=n_heads_i
        ).to(device)
    elif backend == "flashrnn":
        n_heads, d_head = _flashrnn_heads(d_h)
        model = _FlashRNNDyckLM(d_h, n_heads, d_head, flashrnn_backend).to(device)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(spec["weight_decay"]),
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    residuals: list[float] = []
    step_ms: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    n_params = sum(p.numel() for p in model.parameters())
    for step in range(steps + 1):
        tokens = sample_dyck1(batch, seq_len, generator=gen).to(device)

        def _fwd_bwd(tok: Tensor = tokens) -> Tensor:
            logits = model(tok[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), tok[:, 1:].reshape(-1))
            return loss

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = _fwd_bwd()
        residual = float("nan")
        if isinstance(model, _NewtonDyckLM) and model.rnn.last_stats:
            residual = model.rnn.last_stats[0].max_residual
        loss_f = float(loss.detach())
        if step < steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        end.record()
        torch.cuda.synchronize()
        dt = start.elapsed_time(end)
        losses.append(loss_f)
        residuals.append(residual)
        if step < steps:
            step_ms.append(dt)
        log.info(
            "train_step backend=%s step=%d loss=%.4f residual=%.3e dt_ms=%.3f",
            backend,
            step,
            loss_f,
            residual,
            dt,
        )
    peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return {
        "losses": losses,
        "residuals": residuals,
        "step_ms": step_ms,
        "peak_mib": peak,
        "n_params": n_params,
    }


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    nvidia = Path(torch.__file__).resolve().parent.parent / "nvidia"
    runtime = nvidia / "cuda_runtime"
    if (runtime / "include" / "cuda.h").is_file():
        os.environ["CUDA_HOME"] = str(runtime)
        log.info("flashrnn CUDA_HOME=%s (pip cuda_runtime)", runtime)


def _flashrnn_backend() -> str:
    _ensure_cuda_home()
    try:
        import flashrnn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "FlashRNN is required for this example. "
            "uv sync --extra flashrnn --group dev"
        ) from exc
    major, _minor = torch.cuda.get_device_capability()
    if major >= 8:
        return "cuda_fused"
    log.warning("flashrnn backend=triton_fused (CC %d < 8; cuda_fused is Ampere+)", major)
    return "triton_fused"


def _flashrnn_heads(d_h: int) -> tuple[int, int]:
    if d_h % 32 == 0:
        return d_h // 32, 32
    if d_h % 16 == 0:
        return d_h // 16, 16
    raise ValueError(f"d_h={d_h} not divisible by 16 for FlashRNN heads")


def _validate_spec(spec: dict) -> None:
    if spec.get("dtype") != "float32":
        raise ValueError("this smoke is float32 (Turing; not bf16)")
    if spec.get("cell") != "para_slstm":
        raise ValueError("this smoke uses ParaSLSTM")
    if int(spec.get("seq_len", 0)) % 2:
        raise ValueError("Dyck-1 seq_len must be even")
    mix = str(spec.get("mix", "diag"))
    k = int(spec.get("newton_iters", 0))
    if mix == "diag":
        if k != 3:
            raise ValueError("diag mix: library contract is newton_iters=3")
    elif mix == "head":
        if k != 4:
            raise ValueError(
                "mix=head snaps at K=4 (para-slstm.md), not the GRU K=3 default"
            )
        heads = spec.get("n_heads")
        if heads is None or int(spec["d_h"]) % int(heads):
            raise ValueError("mix=head needs n_heads dividing d_h")
        # Auto P is 1 at T<=64. Head mix left that basin at step 20 (max|F|=1.9).
        # Library fallback is raise P, not K (slstm_auto_picard).
        p = spec.get("picard_iters")
        if p is None or int(p) < 3:
            raise ValueError("mix=head train needs picard_iters>=3 (P=1 residual_fail)")
    else:
        raise ValueError(f"train smoke mix must be diag or head, got {mix!r}")
    backends = list(spec.get("backends", []))
    if not backends or set(backends) - {"newton", "flashrnn"}:
        raise ValueError("backends must be a non-empty subset of newton, flashrnn")


if __name__ == "__main__":
    main()
