"""Distill one BabyLM dense sLSTM slot into mix='diag'.

Cache that layer's (RNN-in, RNN-out) once, then two MSE epochs on the student.

    uv run --extra lm python scripts/distill_babylm_layer.py
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

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from pararnn import NewtonConfig, NewtonDivergenceError, ParaRNN, ParaSLSTM
from scripts.babylm_data import prepare_packed
from scripts.babylm_model import BabyLMModel, count_params
from scripts.utils.mlflow_helper import ROOT, git_commit, lock_hash, uv_export_hash

from gpu import select_device, wait_until_free
from gpu import setup_logging as setup_gpu_logging

log = logging.getLogger("distill_babylm")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "babylm_distill.yaml"


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


def _load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    inherit = spec.pop("inherit", None)
    if inherit:
        base = yaml.safe_load((ROOT / inherit).read_text())
        merged = dict(base)
        merged.update(spec)
        return merged
    return spec


def _resolve_layer(spec: dict) -> int:
    n = int(spec["num_layers"])
    raw = int(spec["layer"])
    layer = raw if raw >= 0 else n + raw
    if not 0 <= layer < n:
        raise ValueError(f"layer={raw} resolves to {layer}, stack has {n} blocks")
    return layer


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "x_train": cache_dir / "x_train.f16",
        "y_train": cache_dir / "y_train.f16",
        "x_val": cache_dir / "x_val.f16",
        "y_val": cache_dir / "y_val.f16",
        "meta": cache_dir / "meta.json",
    }


def _diag_of_head_r(r_head: Tensor) -> Tensor:
    return torch.diagonal(r_head.detach(), dim1=-2, dim2=-1).reshape(4, -1)


def _open_memmap(path: Path, shape: tuple[int, ...], *, create: bool) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w+" if create else "r"
    return np.memmap(path, dtype=np.float16, mode=mode, shape=shape)


@torch.no_grad()
def _dump_split(
    teacher: BabyLMModel,
    tokens: Tensor,
    layer: int,
    *,
    x_path: Path,
    y_path: Path,
    device: torch.device,
    batch: int,
) -> tuple[int, int, int]:
    n, t = int(tokens.shape[0]), int(tokens.shape[1])
    d = int(teacher.embed.embedding_dim)
    x_mm = _open_memmap(x_path, (n, t, d), create=True)
    y_mm = _open_memmap(y_path, (n, t, d), create=True)
    teacher.eval()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = tokens[start:end].to(device)
        x_in, y_out = teacher.rnn_io_at_layer(chunk, layer)
        x_mm[start:end] = x_in.float().cpu().numpy().astype(np.float16, copy=False)
        y_mm[start:end] = y_out.float().cpu().numpy().astype(np.float16, copy=False)
        if start == 0 or end == n or (start // batch) % 20 == 0:
            log.info("distill_dump layer=%d rows=%d/%d", layer, end, n)
    x_mm.flush()
    y_mm.flush()
    del x_mm, y_mm
    return n, t, d


def _load_row_batch(mm: np.memmap, idx: Tensor) -> Tensor:
    rows = np.asarray(mm[idx.detach().cpu().numpy()], dtype=np.float32)
    return torch.from_numpy(rows)


@torch.no_grad()
def _eval_mse(
    student: nn.Module,
    x_mm: np.memmap,
    y_mm: np.memmap,
    *,
    device: torch.device,
    batch: int,
    max_rows: int | None = None,
) -> float:
    was_train = student.training
    student.eval()
    n = int(x_mm.shape[0]) if max_rows is None else min(int(max_rows), int(x_mm.shape[0]))
    total = 0.0
    count = 0
    for start in range(0, n, batch):
        end = min(start + batch, n)
        idx = torch.arange(start, end)
        pred = student(_load_row_batch(x_mm, idx).to(device))
        tgt = _load_row_batch(y_mm, idx).to(device)
        total += float(F.mse_loss(pred.float(), tgt, reduction="sum"))
        count += int(pred.numel())
    student.train(was_train)
    return total / max(count, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec = _load_spec(args.config if args.config.is_absolute() else ROOT / args.config)
    setup_gpu_logging()
    device = select_device(str(spec["gpu"]))
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=6.0, poll_s=30.0)
    _seed_all(int(spec["seed"]))

    layer = _resolve_layer(spec)
    train_tok, val_tok, tokenizer = prepare_packed(spec, ROOT)
    spec = dict(spec)
    spec["vocab_size"] = int(tokenizer.get_vocab_size())
    cache_dir = ROOT / spec["activation_cache_dir"]
    paths = _cache_paths(cache_dir)
    meta_ok = False
    if paths["meta"].exists() and paths["x_train"].exists():
        meta = json.loads(paths["meta"].read_text())
        meta_ok = (
            int(meta.get("layer", -1)) == layer
            and int(meta.get("n_train", -1)) == int(train_tok.shape[0])
            and int(meta.get("n_val", -1)) == int(val_tok.shape[0])
        )
        if meta_ok:
            log.info("distill_cache_hit dir=%s layer=%d", cache_dir, layer)

    if not meta_ok:
        ckpt_path = ROOT / spec["teacher_ckpt"]
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        teacher = BabyLMModel(spec, cell_type=str(spec["teacher_cell_type"]))
        teacher.load_state_dict(blob["model"])
        teacher.to(device)
        teacher.eval()
        dump_b = int(spec["dump_batch"])
        log.info(
            "distill_dump_start layer=%d teacher=%s dump_batch=%d gpu=%s",
            layer,
            ckpt_path,
            dump_b,
            torch.cuda.get_device_name(device),
        )
        n_tr, t, d = _dump_split(
            teacher,
            train_tok,
            layer,
            x_path=paths["x_train"],
            y_path=paths["y_train"],
            device=device,
            batch=dump_b,
        )
        n_va, _, _ = _dump_split(
            teacher,
            val_tok,
            layer,
            x_path=paths["x_val"],
            y_path=paths["y_val"],
            device=device,
            batch=dump_b,
        )
        teacher_cell = teacher.blocks[layer].rnn.layers[0]
        init_payload = {
            "W_x": teacher_cell.W_x.state_dict(),
            "R_diag": _diag_of_head_r(teacher_cell.R_head).cpu().clone(),
        }
        torch.save(init_payload, cache_dir / "teacher_cell_init.pt")
        paths["meta"].write_text(
            json.dumps(
                {
                    "layer": layer,
                    "n_train": n_tr,
                    "n_val": n_va,
                    "seq_len": t,
                    "d_model": d,
                    "teacher_ckpt": str(ckpt_path),
                },
                indent=2,
            )
        )
        del teacher
        torch.cuda.empty_cache()
        log.info("distill_dump_done n_train=%d n_val=%d T=%d d=%d", n_tr, n_va, t, d)
    else:
        init_payload = torch.load(cache_dir / "teacher_cell_init.pt", map_location="cpu")

    meta = json.loads(paths["meta"].read_text())
    n_tr, t, d = int(meta["n_train"]), int(meta["seq_len"]), int(meta["d_model"])
    n_va = int(meta["n_val"])
    x_train = _open_memmap(paths["x_train"], (n_tr, t, d), create=False)
    y_train = _open_memmap(paths["y_train"], (n_tr, t, d), create=False)
    x_val = _open_memmap(paths["x_val"], (n_va, t, d), create=False)
    y_val = _open_memmap(paths["y_val"], (n_va, t, d), create=False)

    newton_cfg = NewtonConfig(
        max_iters=int(spec["newton_iters"]),
        scan_backend="fused",
        picard_iters=int(spec["picard_iters"]),
        picard_adapt=bool(spec["picard_adapt"]),
    )
    cell = ParaSLSTM(d, d, mix="diag", max_recurrent_norm=spec.get("max_recurrent_norm", 0.5))
    cell.W_x.load_state_dict(init_payload["W_x"])
    cell.R.data.copy_(init_payload["R_diag"].to(dtype=cell.R.dtype))
    student = ParaRNN(cell, config=newton_cfg, output_hidden=True, solver="auto")
    student.to(device)
    log.info("distill_student params=%d layer=%d mix=diag", count_params(student), layer)

    opt = AdamW(
        student.parameters(),
        lr=float(spec["lr"]),
        betas=tuple(spec["adam_betas"]),
        weight_decay=float(spec["weight_decay"]),
    )
    batch = int(spec["batch"])
    accum = int(spec["grad_accum"])
    seq_per_step = batch * accum
    steps_epoch = math.ceil(n_tr / seq_per_step)
    epochs = int(spec["epochs"])
    total = steps_epoch * epochs
    clip = float(spec["grad_clip"])
    eval_every = int(spec["eval_every"])

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    result = {
        "layer": layer,
        "epochs": epochs,
        "steps": total,
        "n_train": n_tr,
        "gpu": torch.cuda.get_device_name(device),
        "lr": float(spec["lr"]),
        "watchdog_hits": 0,
        "skipped_divergent": 0,
    }
    with mlflow.start_run(run_name=str(spec["mlflow_run_prefix"])):
        mlflow.log_params(
            {
                "layer": layer,
                "epochs": epochs,
                "lr": spec["lr"],
                "batch": batch,
                "d_model": d,
                "seq_len": t,
                "newton_iters": spec["newton_iters"],
                "picard_iters": spec["picard_iters"],
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "gpu": result["gpu"],
            }
        )
        mlflow.log_text(str(spec.get("why", "")).strip() + "\n", "why.txt")
        student.train()
        torch.cuda.reset_peak_memory_stats(device)
        watchdog = 0
        skipped = 0
        last_mse = float("nan")
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
                    start = ((step * accum + micro) * batch) % n_tr
                    idx = (torch.arange(batch) + start) % n_tr
                    xb = _load_row_batch(x_train, idx).to(device)
                    yb = _load_row_batch(y_train, idx).to(device)
                    pred = student(xb)
                    loss = F.mse_loss(pred.float(), yb)
                    (loss / accum).backward()
                    loss_acc += float(loss.detach()) / accum
            except NewtonDivergenceError as exc:
                diverged = True
                skipped += 1
                opt.zero_grad(set_to_none=True)
                log.error(
                    "newton_diverged step=%d err=%s seq_len=%s batch=%s d_model=%s",
                    step,
                    exc,
                    t,
                    batch,
                    d,
                )
            if not diverged:
                nn.utils.clip_grad_norm_(student.parameters(), clip)
                opt.step()
            torch.cuda.synchronize(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            residuals = [float(st.max_residual) for st in student.last_stats]
            max_res = max(residuals) if residuals else float("nan")
            hits = sum(1 for r in residuals if math.isfinite(r) and r > 1e-3)
            watchdog += hits
            mlflow.log_metric("train_mse", loss_acc, step=step)
            mlflow.log_metric("lr", lr, step=step)
            mlflow.log_metric("step_ms", dt_ms, step=step)
            if math.isfinite(max_res):
                mlflow.log_metric("newton_residual", max_res, step=step)
            do_val = (step + 1) % eval_every == 0 or step == 0 or step + 1 == total
            if do_val:
                last_mse = _eval_mse(student, x_val, y_val, device=device, batch=batch)
                mlflow.log_metric("val_mse", last_mse, step=step)
                log.info(
                    "distill_step step=%d/%d train_mse=%.6f val_mse=%.6f residual=%.3e dt_ms=%.1f",
                    step + 1,
                    total,
                    loss_acc,
                    last_mse,
                    max_res,
                    dt_ms,
                )
            elif (step + 1) % 20 == 0:
                log.info(
                    "distill_step step=%d/%d train_mse=%.6f residual=%.3e dt_ms=%.1f",
                    step + 1,
                    total,
                    loss_acc,
                    max_res,
                    dt_ms,
                )
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        result.update(
            {
                "val_mse_final": last_mse,
                "peak_vram_gb": peak,
                "watchdog_hits": watchdog,
                "skipped_divergent": skipped,
            }
        )
        ckpt_dir = ROOT / spec["ckpt_dir"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / f"distill_dense_l{layer}.pt"
        torch.save(
            {
                "student": student.state_dict(),
                "layer": layer,
                "spec": {k: spec[k] for k in spec if k != "why"},
                "result": result,
            },
            ckpt,
        )
        out_json = ROOT / spec["results_dir"] / f"babylm_distill_dense_l{layer}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(out_json))
        mlflow.log_metric("val_mse_final", last_mse)
        mlflow.log_metric("peak_vram_gb", peak)
        log.info(
            "distill_done layer=%d val_mse=%.6f peak_gb=%.2f watchdog=%d skip=%d ckpt=%s",
            layer,
            last_mse,
            peak,
            watchdog,
            skipped,
            ckpt,
        )


if __name__ == "__main__":
    main()
