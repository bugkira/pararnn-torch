"""This box only. Not shipped in pararnn-torch.

Benches pin the 2080 Ti by name and wait out whoever else is on the card.
Restrict the visible set with ``CUDA_VISIBLE_DEVICES`` (PyTorch/Linux default).
Not a notebook sandbox.
"""

from __future__ import annotations

import logging
import sys
import time

import torch

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_GPU_NAME = "2080 Ti"


def setup_logging(level: int = logging.INFO) -> None:
    """CLI root logger for benches. Does not run on ``import pararnn``."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def select_device(name_substring: str = DEFAULT_EXPERIMENT_GPU_NAME) -> torch.device:
    """CUDA device whose ``get_device_name`` contains ``name_substring``.

    Lab helper. Not a library API. ``CUDA_VISIBLE_DEVICES`` still wins on the
    visible set; this only picks among cards PyTorch can already see.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bench")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    matches = [
        i for i, nm in enumerate(names) if name_substring.lower() in nm.lower()
    ]
    if not matches:
        raise RuntimeError(
            f"No GPU matching {name_substring!r}. Visible devices: "
            + ", ".join(f"cuda:{i} ({nm})" for i, nm in enumerate(names))
            + ". Set CUDA_VISIBLE_DEVICES to restrict the set."
        )
    index = matches[0]
    device = torch.device(f"cuda:{index}")
    log.info(
        "select_gpu",
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
