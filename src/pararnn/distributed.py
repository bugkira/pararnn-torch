"""Data-parallel wrapping for ``ParaRNN``.

``ParaRNN`` is an ``nn.Module``. After ``init_process_group``, wrap it (or a
parent that contains it) with ``DistributedDataParallel`` or FSDP2
``fully_shard``. The Newton ``Autograd.Function`` participates in the reducer
like any other op that uses ``cell.parameters()``.

Call ``warmup_scan_kernels`` on each rank before the first training step that
runs an NCCL collective. Triton compiles the fused scan on first use; that
can outlast the default NCCL watchdog.

Each rank keeps ``ParaRNN.last_stats`` as a Python list of ``NewtonStats``
for its own batch. Read residuals with ``last_newton_residuals``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from pararnn.layers.para_rnn import ParaRNN

log = logging.getLogger(__name__)


def unwrap_distributed(module: nn.Module) -> nn.Module:
    """Strip ``DistributedDataParallel``. FSDP2 ``fully_shard`` leaves the module."""
    while isinstance(module, DistributedDataParallel):
        module = module.module
    return module


def warmup_scan_kernels(module: nn.Module, x: Tensor) -> Tensor:
    """Run one Newton forward so Triton compiles before NCCL watches the step.

    Uses ``.train()`` so ``solver='auto'`` takes the parallel path. Restores
    the previous training flag. Gradients stay disabled.
    """
    root = unwrap_distributed(module)
    was_train = root.training
    root.train()
    if not torch.compiler.is_compiling():
        log.debug(
            "warmup_scan_kernels",
            extra={
                "device": str(x.device),
                "dtype": str(x.dtype),
                "batch": int(x.shape[0]),
                "seq_len": int(x.shape[1]),
            },
        )
    with torch.no_grad():
        y = root(x)
    root.train(was_train)
    return y


def last_newton_residuals(module: nn.Module) -> list[float]:
    """``max|F|`` from the last Newton forward on each ``ParaRNN`` in ``module``."""
    root = unwrap_distributed(module)
    out: list[float] = []
    for child in root.modules():
        if isinstance(child, ParaRNN):
            out.extend(float(st.max_residual) for st in child.last_stats)
    return out
