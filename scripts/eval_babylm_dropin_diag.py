"""Zero-shot: BabyLM-dense weights with every sLSTM slot as mix='diag'.

Copies embeddings, LN, MLP, ``W_x``, and ``diag(R_head)`` (or zeros ``R``).
Val NLL/PPL on the packed holdout. No training.

    uv run --extra lm python scripts/eval_babylm_dropin_diag.py
"""

from __future__ import annotations

import argparse
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
import yaml

from scripts.babylm_data import prepare_packed
from scripts.babylm_model import BabyLMModel, count_params
from scripts.train_babylm import _dtype, _eval_nll
from scripts.utils.mlflow_helper import ROOT, git_commit, lock_hash, uv_export_hash

from gpu import select_device, wait_until_free
from gpu import setup_logging as setup_gpu_logging

log = logging.getLogger("eval_babylm_dropin")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "babylm.yaml"


def _diag_of_head_r(r_head: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(r_head.detach(), dim1=-2, dim2=-1).reshape(4, -1)


def _copy_stem(src: BabyLMModel, dst: BabyLMModel) -> None:
    dst.embed.load_state_dict(src.embed.state_dict())
    dst.pos.load_state_dict(src.pos.state_dict())
    dst.norm_f.load_state_dict(src.norm_f.state_dict())
    for s_block, d_block in zip(src.blocks, dst.blocks, strict=True):
        d_block.norm_rnn.load_state_dict(s_block.norm_rnn.state_dict())
        d_block.norm_mlp.load_state_dict(s_block.norm_mlp.state_dict())
        d_block.mlp.load_state_dict(s_block.mlp.state_dict())
        d_cell = d_block.rnn.layers[0]
        s_cell = s_block.rnn.layers[0]
        d_cell.W_x.load_state_dict(s_cell.W_x.state_dict())


def _fill_r(student: BabyLMModel, teacher: BabyLMModel, *, mode: str) -> None:
    for s_block, t_block in zip(student.blocks, teacher.blocks, strict=True):
        t_cell = t_block.rnn.layers[0]
        s_cell = s_block.rnn.layers[0]
        if mode == "diag":
            s_cell.R.data.copy_(_diag_of_head_r(t_cell.R_head).to(dtype=s_cell.R.dtype))
        elif mode == "zero":
            s_cell.R.data.zero_()
        else:
            raise ValueError(f"unknown R mode {mode!r}")


def _ppl(nll: float) -> float:
    return math.exp(nll) if nll < 20.0 else float("inf")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    spec = yaml.safe_load(cfg_path.read_text())
    setup_gpu_logging()
    device = select_device(str(spec["gpu"]))
    torch.cuda.set_device(device)
    wait_until_free(device, min_free_gib=4.0, poll_s=30.0)

    _train_tok, val_tok, tokenizer = prepare_packed(spec, ROOT)
    spec = dict(spec)
    spec["vocab_size"] = int(tokenizer.get_vocab_size())
    amp_dtype = _dtype(str(spec["dtype"]))
    n_report = int(spec["val_report_sequences"])
    batch = int(spec["batch"])
    ckpt_path = ROOT / spec["ckpt_dir"] / "dense.pt"
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    teacher = BabyLMModel(spec, cell_type="dense")
    teacher.load_state_dict(blob["model"])
    teacher.to(device=device, dtype=amp_dtype)
    teacher.eval()

    eval_kw = {
        "device": device,
        "batch": batch,
        "max_rows": n_report,
        "amp_dtype": amp_dtype,
    }
    log.info(
        "dropin_eval_start ckpt=%s n_val=%d batch=%d gpu=%s",
        ckpt_path,
        n_report,
        batch,
        torch.cuda.get_device_name(device),
    )
    nll_dense = _eval_nll(teacher, val_tok, **eval_kw)
    ppl_dense = _ppl(nll_dense)
    log.info("dropin_eval arm=dense nll=%.4f ppl=%.2f", nll_dense, ppl_dense)

    arms: dict[str, dict] = {
        "dense": {
            "nll": nll_dense,
            "ppl": ppl_dense,
            "params": count_params(teacher),
            "r": "head 48x48",
        }
    }
    for mode in ("diag", "zero"):
        student = BabyLMModel(spec, cell_type="diag_seq")
        _copy_stem(teacher, student)
        _fill_r(student, teacher, mode=mode)
        student.to(device=device, dtype=amp_dtype)
        student.eval()
        nll = _eval_nll(student, val_tok, **eval_kw)
        ppl = _ppl(nll)
        arms[mode] = {
            "nll": nll,
            "ppl": ppl,
            "params": count_params(student),
            "r": "diag(R_head)" if mode == "diag" else "R=0",
        }
        log.info(
            "dropin_eval arm=%s nll=%.4f ppl=%.2f delta_ppl=%+.2f",
            mode,
            nll,
            ppl,
            ppl - ppl_dense,
        )
        del student
        torch.cuda.empty_cache()

    result = {
        "task": "babylm_dropin_diag",
        "ckpt": str(ckpt_path),
        "n_val": n_report,
        "seq_len": int(spec["seq_len"]),
        "gpu": torch.cuda.get_device_name(device),
        "arms": arms,
        "delta_ppl_diag": arms["diag"]["ppl"] - ppl_dense,
        "delta_ppl_zero": arms["zero"]["ppl"] - ppl_dense,
        "trained_diag_seq_ppl": None,
    }
    trained = ROOT / spec["results_dir"] / "babylm_diag_seq.json"
    if trained.exists():
        result["trained_diag_seq_ppl"] = json.loads(trained.read_text()).get("val_ppl_final")

    out = ROOT / spec["results_dir"] / "babylm_dropin_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    import mlflow

    mlflow.set_experiment(str(spec["mlflow_experiment"]))
    with mlflow.start_run(run_name="babylm-dropin-diag"):
        mlflow.log_params(
            {
                "mode": "zero_shot_dropin_diag",
                "n_val": n_report,
                "git": git_commit(),
                "uv_lock": lock_hash(),
                "uv_export": uv_export_hash(),
                "gpu": result["gpu"],
            }
        )
        mlflow.log_metric("ppl_dense", ppl_dense)
        mlflow.log_metric("ppl_dropin_diag", arms["diag"]["ppl"])
        mlflow.log_metric("ppl_dropin_R0", arms["zero"]["ppl"])
        mlflow.log_metric("delta_ppl_diag", result["delta_ppl_diag"])
        mlflow.log_metric("nll_dense", nll_dense)
        mlflow.log_metric("nll_dropin_diag", arms["diag"]["nll"])
        mlflow.log_artifact(str(out))
        mlflow.log_text(
            "Zero-shot: copy dense BabyLM stem + W_x; R is diag(R_head) or 0. "
            "Same packed val as babylm.yaml val_report_sequences.\n",
            "why.txt",
        )
    log.info("dropin_eval_done wrote=%s delta_ppl_diag=%+.2f", out, result["delta_ppl_diag"])


if __name__ == "__main__":
    main()
