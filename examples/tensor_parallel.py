"""Tensor parallel along channelwise ``d_h``: local fused scan, one AllReduce.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      uv run torchrun --nproc_per_node=2 examples/tensor_parallel.py

``mix='diag'`` (ParaGRU here) keeps channels independent, so Newton+scan
needs no NCCL. ``W_x`` on each rank is column-parallel
(``d_in → 3·d_h/N``). ``RowParallelLinear`` maps ``d_h/N → d_out`` and
AllReduces once per layer.

``d_h=32`` so each of two ranks holds 16 channels (divides evenly).
``max_iters=3`` is App. A. ``lr=1e-3`` is one AdamW step to prove the
collective, not a train curve. fp32: this box pairs Ampere with Turing.

Local smoke: no MLflow. See ``docs/distributed.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.distributed as dist

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import NewtonConfig
from pararnn.distributed import last_newton_residuals, warmup_scan_kernels
from pararnn.tensor_parallel import local_hidden_size, tensor_parallel_diag_block, tp_rank

log = logging.getLogger("tensor_parallel")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    if "RANK" not in os.environ:
        raise SystemExit(
            "Launch with torchrun, e.g. "
            "uv run torchrun --nproc_per_node=2 examples/tensor_parallel.py"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} "
                "visible CUDA devices; set nproc_per_node to the visible count"
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")

    d_in, d_h, d_out = 32, 32, 32
    seq_len, batch = 32, 4
    d_local = local_hidden_size(d_h, world)
    config = NewtonConfig(max_iters=3, scan_backend="auto")
    torch.manual_seed(0)
    model = tensor_parallel_diag_block("gru", d_in, d_h, d_out, config=config, device=device)
    dummy = torch.randn(batch, seq_len, d_in, device=device)
    warmup_scan_kernels(model, dummy)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    gpu_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    cc = torch.cuda.get_device_capability(device) if use_cuda else None
    cc_s = None if cc is None else f"{cc[0]}.{cc[1]}"
    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(args.steps):
        torch.manual_seed(2000 + step)
        x = torch.randn(batch, seq_len, d_in, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        res = last_newton_residuals(model)
        residual = res[0] if res else float("nan")
        log.info(
            "tp_step rank=%s world=%s gpu=%s cc=%s d_h_local=%s loss=%s residual=%s "
            "seq_len=%s batch=%s newton_iters=3",
            tp_rank(),
            world,
            gpu_name,
            cc_s,
            d_local,
            last_loss,
            residual,
            seq_len,
            batch,
            extra={
                "rank": rank,
                "world": world,
                "gpu": gpu_name,
                "cc": cc_s,
                "d_h_local": d_local,
                "loss": last_loss,
                "newton_residual": residual,
                "seq_len": seq_len,
                "batch": batch,
                "newton_iters": 3,
            },
        )
    dt_ms = (time.perf_counter() - t0) * 1e3
    log.info(
        "tp_done rank=%s gpu=%s d_h_local=%s loss=%s dt_ms=%.1f",
        rank,
        gpu_name,
        d_local,
        last_loss,
        dt_ms,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
