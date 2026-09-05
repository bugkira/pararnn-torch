"""Data-parallel smoke: DDP or FSDP2 fully_shard around ParaRNN.

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ddp_fsdp.py

One AdamW step; wiring check. K=3 is App. A. fp32 so mixed CC
(Ampere + Turing) works; bf16 fused needs CC ≥ 8.0 on every rank.

Flip STRATEGY to \"fsdp\" for FSDP2.
"""

import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.distributed as dist
from torch import nn

from pararnn import NewtonConfig, ParaGRU, ParaRNN
from pararnn.distributed import last_newton_residuals, warmup_scan_kernels

D_H, SEQ_LEN, BATCH = 32, 32, 4
LR = 1e-3
STRATEGY = "ddp"  # or "fsdp"

if "RANK" not in os.environ:
    raise SystemExit(
        "Launch with torchrun, e.g. torchrun --nproc_per_node=2 ddp_fsdp.py"
    )

rank = int(os.environ["RANK"])
local_rank = int(os.environ.get("LOCAL_RANK", rank))
world = int(os.environ["WORLD_SIZE"])
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")

use_cuda = torch.cuda.is_available()
if use_cuda:
    if local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} visible devices"
        )
    torch.cuda.set_device(local_rank)
    backend, device = "nccl", torch.device(f"cuda:{local_rank}")
else:
    backend, device = "gloo", torch.device("cpu")

dist.init_process_group(backend)
torch.manual_seed(0)
model = nn.Sequential(
    ParaRNN(ParaGRU(D_H, D_H), config=NewtonConfig(max_iters=3, scan_backend="auto")),
    nn.Linear(D_H, D_H),
).to(device)
dummy = torch.randn(BATCH, SEQ_LEN, D_H, device=device)
warmup_scan_kernels(model, dummy)

if STRATEGY == "ddp":
    if use_cuda:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        model = nn.parallel.DistributedDataParallel(model)
elif STRATEGY == "fsdp":
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    fully_shard(model, mesh=init_device_mesh(device.type, (world,)))
else:
    raise SystemExit(f"unknown STRATEGY={STRATEGY!r}; use 'ddp' or 'fsdp'")

opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
gpu = torch.cuda.get_device_name(device) if use_cuda else "cpu"
t0 = time.perf_counter()

torch.manual_seed(1000 + rank * 17)
x = torch.randn(BATCH, SEQ_LEN, D_H, device=device)
opt.zero_grad(set_to_none=True)
loss = model(x).square().mean()
loss.backward()
opt.step()
res = last_newton_residuals(model)
residual = res[0] if res else float("nan")
print(
    f"rank={rank}/{world} {STRATEGY} gpu={gpu} "
    f"loss={loss.item():.4f} residual={residual:.3e} "
    f"dt_ms={(time.perf_counter() - t0) * 1e3:.1f}"
)
dist.destroy_process_group()
