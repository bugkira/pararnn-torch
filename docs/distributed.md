# Data parallel (DDP / FSDP)

`ParaRNN` is an `nn.Module`. After `init_process_group`, wrap it or a parent
that contains it:

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from pararnn.distributed import warmup_scan_kernels, last_newton_residuals

warmup_scan_kernels(model, dummy)  # compile Triton before NCCL watches the step
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# or:
# fully_shard(model, mesh=init_device_mesh("cuda", (world_size,)))
```

Smoke:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py

uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py --strategy fsdp
```

`nproc_per_node` is the number of **visible** devices. On this box nvidia-smi
0 is the RTX 3060 (CC 8.6) and 1 is the 2080 Ti (CC 7.5). Pin
`CUDA_DEVICE_ORDER=PCI_BUS_ID` before importing torch.

## What the wrapper sees

Newton's `Autograd.Function` uses `cell.parameters()`. DDP's reducer and
FSDP2's all-gather hook those parameters the same way they hook a Linear.
`find_unused_parameters=False` (the default) is enough: every cell weight
runs in the first Newton iteration.

`ParaRNN.last_stats` is a Python list of `NewtonStats` on the module. Each
rank keeps the residual of its own batch. Read it with
`last_newton_residuals(model)` (walks through DDP).

## Warmup

Triton compiles the fused scan on first use. That compile can sit longer
than the default NCCL heartbeat. `warmup_scan_kernels` runs one `.train()`
Newton forward under `no_grad` on each rank **before** the wrap (DDP) or
before the first step (FSDP). The example also sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`.

## Dtype on mixed cards

Fused bf16 needs compute capability ≥ 8.0 on the tensor's device. A 2080 Ti
rank in a 3060+2080 Ti process group uses fp32 or fp16. The smoke defaults
to fp32.

## Tests

- CPU gloo, two processes: DDP grads match the mean of the two rank losses
  (`tests/numerics/test_ddp.py`). This is the reducer + eq. 2.6 check.
- CUDA, two visible devices: after one SGD step the DDP parameters match
  across ranks. FSDP2 runs one backward. Skip when `device_count() < 2`
  (a shell with `CUDA_VISIBLE_DEVICES=1` hides the 3060).

Sequence-parallel scan over NCCL is a separate catalog row
(`pararnn.solvers.seq_parallel` is two CUDA streams on one device).
