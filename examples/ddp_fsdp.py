"""Data-parallel smoke: ``DDP`` or FSDP2 ``fully_shard`` around ``ParaRNN``.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py

    uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py --strategy fsdp

``ParaRNN`` is a normal ``nn.Module``. This script is a collective/wiring
check (one AdamW step), not a quality run. ``max_iters=3`` is App. A.
``d_h=32``, ``T=32``, ``batch=4`` fit both cards on this box with headroom.
``lr=1e-3`` only exists so the optimizer runs. fp32 is the default because
this machine pairs Ampere (CC 8.6) with Turing (CC 7.5); bf16 fused needs
CC ≥ 8.0 on every rank.

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
from torch import Tensor, nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import NewtonConfig, ParaGRU, ParaRNN
from pararnn.distributed import last_newton_residuals, warmup_scan_kernels

log = logging.getLogger("ddp_fsdp")


class _Toy(nn.Module):
    def __init__(self, d_h: int, config: NewtonConfig) -> None:
        super().__init__()
        self.rnn = ParaRNN(ParaGRU(d_h, d_h), config=config)
        self.head = nn.Linear(d_h, d_h)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.rnn(x))


def _dtype(name: str) -> torch.dtype:
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), default="ddp")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)

    if "RANK" not in os.environ:
        raise SystemExit(
            "Launch with torchrun, e.g. uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py"
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
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend)
    dtype = _dtype(args.dtype)
    d_h, seq_len, batch = 32, 32, 4
    # App. A: K=3. auto: fused Triton on CUDA for ParaGRU.
    config = NewtonConfig(max_iters=3, scan_backend="auto")
    torch.manual_seed(0)
    model = _Toy(d_h, config).to(device=device, dtype=dtype)
    dummy = torch.randn(batch, seq_len, d_h, device=device, dtype=dtype)
    warmup_scan_kernels(model, dummy)

    if args.strategy == "ddp":
        if use_cuda:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            model = nn.parallel.DistributedDataParallel(model)
    else:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh(device.type, (world,))
        fully_shard(model, mesh=mesh)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    gpu_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    cc = torch.cuda.get_device_capability(device) if use_cuda else None
    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(args.steps):
        torch.manual_seed(1000 + rank * 17 + step)
        x = torch.randn(batch, seq_len, d_h, device=device, dtype=dtype)
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        res = last_newton_residuals(model)
        residual = res[0] if res else float("nan")
        cc_s = None if cc is None else f"{cc[0]}.{cc[1]}"
        log.info(
            "ddp_fsdp_step rank=%s world=%s strategy=%s step=%s gpu=%s cc=%s "
            "dtype=%s loss=%s residual=%s seq_len=%s batch=%s d_h=%s newton_iters=3",
            rank,
            world,
            args.strategy,
            step,
            gpu_name,
            cc_s,
            dtype,
            last_loss,
            residual,
            seq_len,
            batch,
            d_h,
            extra={
                "rank": rank,
                "world": world,
                "strategy": args.strategy,
                "step": step,
                "device": str(device),
                "gpu": gpu_name,
                "cc": cc_s,
                "dtype": str(dtype),
                "loss": last_loss,
                "newton_residual": residual,
                "seq_len": seq_len,
                "batch": batch,
                "d_h": d_h,
                "newton_iters": 3,
            },
        )
    dt_ms = (time.perf_counter() - t0) * 1e3
    log.info(
        "ddp_fsdp_done rank=%s strategy=%s gpu=%s loss=%s dt_ms=%.1f",
        rank,
        args.strategy,
        gpu_name,
        last_loss,
        dt_ms,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
