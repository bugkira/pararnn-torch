"""BabyLM Strict-Small mixing ablation (dense head / diag seq / fused diag).

    uv run --extra lm python scripts/train_babylm.py --cell_type diag_fused
    uv run --extra lm python scripts/train_babylm.py --summarize
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from pararnn import NewtonDivergenceError
from scripts.babylm_data import prepare_packed
from scripts.babylm_model import (
    CELL_TYPES,
    BabyLMModel,
    arm_label,
    count_params,
    mixing_label,
    solver_label,
)
from scripts.utils.mlflow_helper import ROOT, git_commit, lock_hash, uv_export_hash

from gpu import select_device, wait_until_free
from gpu import setup_logging as setup_gpu_logging

log = logging.getLogger("train_babylm")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "babylm.yaml"


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)


def _cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * float(step + 1) / float(max(warmup, 1))
    t = float(step - warmup) / float(max(total - warmup, 1))
    return base * 0.5 * (1.0 + math.cos(math.pi * t))


def _dtype(name: str) -> torch.dtype:
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def _batch_rows(tokens: Tensor, step: int, batch: int) -> Tensor:
    n = int(tokens.shape[0])
    start = (step * batch) % n
    idx = (torch.arange(batch) + start) % n
    return tokens[idx]


@torch.no_grad()
def _eval_nll(
    model: nn.Module,
    tokens: Tensor,
    *,
    device: torch.device,
    batch: int,
    max_rows: int,
    amp_dtype: torch.dtype,
) -> float:
    was_train = model.training
    model.eval()
    rows = tokens[: max(1, min(int(max_rows), int(tokens.shape[0])))]
    total = 0.0
    ntok = 0
    for start in range(0, rows.shape[0], batch):
        chunk = rows[start : start + batch].to(device)
        logits = model(chunk)
        logp = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        )
        total += float(logp)
        ntok += int(chunk[:, 1:].numel())
    model.train(was_train)
    return total / max(ntok, 1)


def _watchdog_hits(model: BabyLMModel, threshold: float) -> int:
    hits = 0
    for st in model.newton_stats():
        res = float(st.max_residual)
        if math.isfinite(res) and res > threshold:
            hits += 1
    return hits


def _write_markdown(results_dir: Path) -> Path:
    rows = []
    for path in sorted(results_dir.glob("babylm_*.json")):
        if path.name == "babylm_ablation.json":
            continue
        rows.append(json.loads(path.read_text()))
    order = {name: i for i, name in enumerate(CELL_TYPES)}
    rows.sort(key=lambda r: order.get(r.get("cell_type", ""), 99))
    dtypes = sorted({str(r.get("dtype", "?")) for r in rows})
    dtype_s = dtypes[0] if len(dtypes) == 1 else ", ".join(dtypes)
    gpu = rows[0].get("gpu", "RTX 3060") if rows else "RTX 3060"
    steps = rows[0].get("steps") if rows else None
    seq_len = rows[0].get("seq_len", 512) if rows else 512
    lines = [
        "# BabyLM-10M mixing ablation",
        "",
        f"{gpu}, **{dtype_s}**, T={seq_len}, 6×384, 16k BPE, BabyLM 2026 Strict-Small.",
        f"One epoch = {steps} steps, B=16, grad accum=2." if isinstance(steps, int) else "",
        "",
        "| Model Arm | Mixing R | Solver | Params | Val PPL | tok/s | Peak VRAM |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    lines = [ln for ln in lines if ln is not None]
    for rec in rows:
        ppl = rec.get("val_ppl_final")
        ppl_s = f"{ppl:.2f}" if isinstance(ppl, (int, float)) and math.isfinite(ppl) else "—"
        tok_s = rec.get("tok_per_s")
        tok_str = f"{tok_s:.0f}" if isinstance(tok_s, (int, float)) else "—"
        vram = rec.get("peak_vram_gb")
        vram_s = f"{vram:.2f} GB" if isinstance(vram, (int, float)) else "—"
        params = rec.get("params")
        params_s = f"{params / 1e6:.1f}M" if isinstance(params, int) else "—"
        lines.append(
            f"| {rec.get('arm', rec.get('cell_type'))} | {rec.get('mixing')} | "
            f"{rec.get('solver')} | {params_s} | {ppl_s} | {tok_str} | {vram_s} |"
        )
    lines.append("")
    out = results_dir / "babylm_ablation.md"
    out.write_text("\n".join(lines))
    return out


def train_one(spec: dict, *, cell_type: str, max_steps: int | None) -> dict:
    setup_gpu_logging()
    device = select_device(str(spec["gpu"]))
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=6.0, poll_s=30.0)

    _seed_all(int(spec["seed"]))
    train_tok, val_tok, tokenizer = prepare_packed(spec, ROOT)
    spec = dict(spec)
    spec["vocab_size"] = int(tokenizer.get_vocab_size())

    amp_dtype = _dtype(str(spec["dtype"]))
    if amp_dtype is torch.bfloat16 and torch.cuda.get_device_capability(device)[0] < 8:
        raise RuntimeError("bfloat16 fused/train needs Ampere (sm_80+); this box's 3060 is CC 8.6")
    model = BabyLMModel(spec, cell_type=cell_type).to(device=device, dtype=amp_dtype)
    opt = AdamW(
        model.parameters(),
        lr=float(spec["lr"]),
        betas=tuple(spec["adam_betas"]),
        weight_decay=float(spec["weight_decay"]),
    )
    batch = int(spec["batch"])
    accum = int(spec["grad_accum"])
    seq_per_step = batch * accum
    steps_epoch = math.ceil(int(train_tok.shape[0]) / seq_per_step)
    total = int(max_steps) if max_steps is not None else int(spec["max_steps"] or steps_epoch)
    ppl_steps = {int(s) for s in spec.get("ppl_steps", [])}
    ppl_steps.add(total)
    eval_every = int(spec["eval_every"])
    clip = float(spec["grad_clip"])
    tokens_per_step = seq_per_step * (int(spec["seq_len"]) - 1)

    import mlflow

    run_name = f"{spec.get('mlflow_run_prefix', 'babylm')}-{cell_type}"
    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    result = {
        "cell_type": cell_type,
        "arm": arm_label(cell_type),
        "mixing": mixing_label(cell_type),
        "solver": solver_label(cell_type),
        "params": count_params(model),
        "gpu": torch.cuda.get_device_name(device),
        "dtype": str(spec["dtype"]),
        "steps": total,
        "batch": batch,
        "seq_len": int(spec["seq_len"]),
        "watchdog_hits": 0,
        "skipped_divergent": 0,
    }
    ckpt_dir = ROOT / spec["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / spec["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "babylm_train_start cell=%s gpu=%s params=%d train_rows=%d steps=%d",
        cell_type,
        result["gpu"],
        result["params"],
        int(train_tok.shape[0]),
        total,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "cell": "para_slstm",
                "cell_type": cell_type,
                "mode": "fused" if cell_type == "diag_fused" else "sequential",
                "dtype": str(spec["dtype"]),
                "gpu": result["gpu"],
            }
        )
        mlflow.log_params(
            {
                "cell_type": cell_type,
                "lr": spec["lr"],
                "batch": batch,
                "seq_len": spec["seq_len"],
                "d_model": spec["d_model"],
                "num_layers": spec["num_layers"],
                "newton_iters": spec["newton_iters"],
                "picard_iters": spec["picard_iters"],
                "warmup_steps": spec["warmup_steps"],
                "weight_decay": spec["weight_decay"],
                "seed": spec["seed"],
                "vocab_size": spec["vocab_size"],
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
            }
        )
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")

        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        watchdog = 0
        skipped = 0
        step_ms_sum = 0.0
        timed_steps = 0
        last_ppl = float("nan")

        for step in range(total):
            lr = _cosine_lr(step, total, float(spec["lr"]), int(spec["warmup_steps"]))
            for grp in opt.param_groups:
                grp["lr"] = lr
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            diverged = False
            try:
                for micro in range(accum):
                    rows = _batch_rows(train_tok, step * accum + micro, batch).to(device)
                    logits = model(rows)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                        rows[:, 1:].reshape(-1),
                    )
                    (loss / accum).backward()
                    loss_acc += float(loss.detach()) / accum
            except NewtonDivergenceError as exc:
                diverged = True
                skipped += 1
                opt.zero_grad(set_to_none=True)
                log.error(
                    "newton_diverged cell=%s step=%d err=%s seq_len=%s batch=%s d_model=%s",
                    cell_type,
                    step,
                    exc,
                    spec["seq_len"],
                    batch,
                    spec["d_model"],
                )
            if not diverged:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step >= 2:
                step_ms_sum += dt_ms
                timed_steps += 1
            hits = _watchdog_hits(model, 1e-3)
            watchdog += hits
            residuals = [float(st.max_residual) for st in model.newton_stats()]
            max_res = max(residuals) if residuals else float("nan")
            tok_s = tokens_per_step / (dt_ms / 1000.0) if dt_ms > 0 else float("nan")
            mlflow.log_metric("train_loss", loss_acc, step=step)
            mlflow.log_metric("lr", lr, step=step)
            mlflow.log_metric("step_ms", dt_ms, step=step)
            mlflow.log_metric("tok_per_s", tok_s, step=step)
            if math.isfinite(max_res):
                mlflow.log_metric("newton_residual", max_res, step=step)
            if hits:
                mlflow.log_metric("watchdog_hits", hits, step=step)
                log.info(
                    "watchdog_residual cell=%s step=%d hits=%d max_residual=%.3e",
                    cell_type,
                    step,
                    hits,
                    max_res,
                )

            do_val = (step + 1) % eval_every == 0 or (step + 1) in ppl_steps or step == 0
            if do_val:
                nll = _eval_nll(
                    model,
                    val_tok,
                    device=device,
                    batch=batch,
                    max_rows=int(spec["val_sequences"]),
                    amp_dtype=amp_dtype,
                )
                ppl = math.exp(nll) if nll < 20.0 else float("inf")
                mlflow.log_metric("val_loss", nll, step=step)
                mlflow.log_metric("val_ppl", ppl, step=step)
                log.info(
                    "babylm_step cell=%s step=%d/%d loss=%.4f val_nll=%.4f val_ppl=%.2f "
                    "step_ms=%.1f tok/s=%.0f residual=%.3e",
                    cell_type,
                    step + 1,
                    total,
                    loss_acc,
                    nll,
                    ppl,
                    dt_ms,
                    tok_s,
                    max_res,
                )
            if (step + 1) in ppl_steps:
                nll_r = _eval_nll(
                    model,
                    val_tok,
                    device=device,
                    batch=batch,
                    max_rows=int(spec["val_report_sequences"]),
                    amp_dtype=amp_dtype,
                )
                last_ppl = math.exp(nll_r) if nll_r < 20.0 else float("inf")
                mlflow.log_metric("val_ppl_report", last_ppl, step=step)
                log.info(
                    "babylm_ppl cell=%s step=%d val_ppl=%.2f nll=%.4f",
                    cell_type,
                    step + 1,
                    last_ppl,
                    nll_r,
                )

        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        mean_ms = step_ms_sum / max(timed_steps, 1)
        result.update(
            {
                "val_ppl_final": last_ppl,
                "tok_per_s": tokens_per_step / (mean_ms / 1000.0) if mean_ms > 0 else float("nan"),
                "step_ms": mean_ms,
                "peak_vram_gb": peak,
                "watchdog_hits": watchdog,
                "skipped_divergent": skipped,
            }
        )
        mlflow.log_metric("peak_vram_gb", peak)
        mlflow.log_metric("val_ppl_final", last_ppl)
        ckpt = ckpt_dir / f"{cell_type}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "cell_type": cell_type,
                "spec": {k: spec[k] for k in spec if k != "why"},
                "result": result,
            },
            ckpt,
        )
        out_json = results_dir / f"babylm_{cell_type}.json"
        out_json.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(out_json))
        log.info(
            "babylm_train_done cell=%s ppl=%.2f tok/s=%.0f peak_gb=%.2f watchdog=%d skip=%d",
            cell_type,
            last_ppl,
            result["tok_per_s"],
            peak,
            watchdog,
            skipped,
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cell_type", choices=CELL_TYPES)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    results_dir = ROOT / spec["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        path = _write_markdown(results_dir)
        log_setup = logging.getLogger("train_babylm")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            stream=sys.stderr,
        )
        log_setup.info("wrote %s", path)
        return
    if args.cell_type is None:
        raise SystemExit("need --cell_type or --summarize")
    train_one(spec, cell_type=args.cell_type, max_steps=args.max_steps)
    _write_markdown(results_dir)


if __name__ == "__main__":
    main()
