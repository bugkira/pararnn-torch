"""Pick the experiment GPU by name. Indices are not portable.

On this box ``nvidia-smi`` lists 3060 as GPU 0 and 2080 Ti as GPU 1.
PyTorch currently enumerates them the other way around (2080 Ti = cuda:0).
Always match on the device *name*. Never use cuda:0/cuda:1 from nvidia-smi.
Never fall back to the 3060 for benches — wait until the 2080 Ti is free.
"""

from __future__ import annotations

import logging
import os
import time

import torch

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_GPU_NAME = "2080 Ti"


def experiment_device(
    name_substring: str = DEFAULT_EXPERIMENT_GPU_NAME,
    *,
    allow_cpu: bool = False,
) -> torch.device:
    """Return the CUDA device whose ``get_device_name`` contains ``name_substring``."""
    override = os.environ.get("PARARNN_DEVICE")
    if override:
        device = torch.device(override)
        log.info("device_from_env", extra={"device": str(device)})
        return device

    if not torch.cuda.is_available():
        if allow_cpu:
            log.warning("cuda_unavailable_using_cpu")
            return torch.device("cpu")
        raise RuntimeError("CUDA is required unless allow_cpu=True")

    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    matches = [i for i, nm in enumerate(names) if name_substring.lower() in nm.lower()]
    if not matches:
        raise RuntimeError(
            f"No GPU matching {name_substring!r}. Visible devices: "
            + ", ".join(f"cuda:{i} ({nm})" for i, nm in enumerate(names))
        )
    index = matches[0]
    device = torch.device(f"cuda:{index}")
    log.info(
        "experiment_device",
        extra={"cuda_index": index, "gpu_name": names[index], "all_gpus": names},
    )
    return device


def wait_until_free(
    device: torch.device,
    *,
    min_free_gib: float = 8.0,
    poll_s: float = 30.0,
) -> None:
    """Block until ``device`` has ``min_free_gib`` free. Does not switch GPUs."""
    if device.type != "cuda":
        return
    while True:
        free_b, total_b = torch.cuda.mem_get_info(device)
        free_gib = free_b / (1024**3)
        total_gib = total_b / (1024**3)
        log.info(
            "gpu_mem device=%s free_gib=%.2f total_gib=%.2f",
            device,
            free_gib,
            total_gib,
        )
        if free_gib >= min_free_gib:
            return
        log.info(
            "waiting_for_gpu need=%.1f GiB free, have=%.2f (staying on this card)",
            min_free_gib,
            free_gib,
        )
        time.sleep(poll_s)
