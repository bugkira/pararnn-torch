# Data parallel (DDP / FSDP) and tensor / context parallel

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

Smoke (two visible GPUs):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py
# FSDP2: set STRATEGY = "fsdp" at the top of the script.
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

## Tensor parallel along $`d_h`$

Channelwise-diagonal cells (ParaGRU, ParaLSTM, `ParaSLSTM mix='diag'`) keep
features independent through Newton+scan. Rank $r$ owns $`d_h/N`$
channels. The cell's `W_x` is already column-parallel (replicated `x`,
sharded gate rows). The fused kernel runs on that slice with zero NCCL.
`RowParallelLinear` maps $`d_h/N \to d_{\mathrm{out}}`$ and AllReduces once
per layer. Backward of that AllReduce is identity (replicated loss on the
shared output).

```python
from pararnn.tensor_parallel import tensor_parallel_diag_block

# d_h=32, world=2 → each rank holds 16 channels
block = tensor_parallel_diag_block("gru", d_in=32, d_h=32, d_out=32, config=cfg)
y = block(x)  # (B, T, 32), one AllReduce
```

`d_h` must divide the tensor-parallel size. Factory kinds: `gru`, `lstm`,
`slstm` (`mix='diag'`).

CPU gloo: sharded Newton + row-parallel grads match a full `Linear(d_h, d_out)`
(`tests/numerics/test_tp_allreduce.py`). CUDA: both ranks hold the same
AllReduced `y`.

## Context parallel (time shard)

Sequence-parallel scan over NCCL splits **time** (`scan_diag_context_parallel`
in `pararnn.solvers.seq_parallel`). Each rank holds `T/N` steps, scans
locally, and AllGathers the tile monoid `(P_end, δ_end)` of shape `(B, d)`.
The exclusive prefix of that monoid is the carry. Rank 0's incoming carry
is 0; Rank 1 applying the AllGather carry is the numeric check.
`NewtonConfig(scan_backend="context_parallel")` shards the diag scan of a
**replicated** Newton trajectory (`H` stays full on every rank; scan work
is `T/N`). Eq. 2.6 reverse scan sends `(μ, J)` at the tile head Rank N-1 → 0.
Standalone `scan_diag_context_parallel` VJP uses eager local scans and
`torch.distributed.nn.functional.all_gather` of the monoid (Triton has no
scan autograd). Two CUDA streams on one device remain `scan_diag_two_ranks`
(virtual ranks).

```python
from pararnn.solvers.seq_parallel import scan_diag_context_parallel, time_shard_bounds

start, end = time_shard_bounds(T, rank, world)
delta_local = scan_diag_context_parallel(jac[:, start:end], residual[:, start:end])
```

CPU gloo: each rank's tile matches `scan_diag` of the full system, including
remainder `T=17`. Rank 1 `max_abs` is the carry path (`<1e-4`). Local-tile
`grad_jac` / `grad_residual` match a full eager scan. Newton forward and
eq. 2.6 grads match `scan_backend="eager"`. CUDA NCCL: concat of the two
eager tiles matches CPU `scan_diag` (`tests/numerics/test_context_parallel.py`).

Two-card torchrun demos for TP / CP / paged pool lived in
`examples/` and are archived on branch `archive/distributed-demos`.
API and numerics tests above stay on `main`.
