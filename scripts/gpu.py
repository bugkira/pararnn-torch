"""This box only. Not shipped in pararnn-torch.

Benches pin the 2080 Ti by name and wait out whoever else is on the card.
Restrict the visible set with ``CUDA_VISIBLE_DEVICES``.

This box has two GPUs whose nvidia-smi indices disagree with CUDA's default
``FASTEST_FIRST`` order. Pin ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so index 0 is
the RTX 3060 (CC 8.6) and index 1 is the RTX 2080 Ti (CC 7.5). That must
happen before the CUDA runtime initializes (before ``import torch``).
"""

from __future__ import annotations

import logging
import os
import sys
import time

_CUDA_ORDER = "PCI_BUS_ID"
_torch_imported_before_pin = "torch" in sys.modules
if not _torch_imported_before_pin:
    os.environ["CUDA_DEVICE_ORDER"] = _CUDA_ORDER

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


def _warn_if_order_unpinned() -> None:
    if _torch_imported_before_pin and os.environ.get("CUDA_DEVICE_ORDER") != _CUDA_ORDER:
        log.warning(
            "torch imported before CUDA_DEVICE_ORDER=%s; indices may not "
            "match nvidia-smi. Set the env var in the shell before importing torch.",
            _CUDA_ORDER,
        )


def _card_info(index: int) -> tuple[str, str]:
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    return name, f"{major}.{minor}"


def select_device(name_substring: str = DEFAULT_EXPERIMENT_GPU_NAME) -> torch.device:
    """CUDA device whose ``get_device_name`` contains ``name_substring``.

    Lab helper. Not a library API. ``CUDA_VISIBLE_DEVICES`` still wins on the
    visible set; this only picks among cards PyTorch can already see.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bench")
    _warn_if_order_unpinned()
    n = torch.cuda.device_count()
    infos = [_card_info(i) for i in range(n)]
    names = [name for name, _cc in infos]
    matches = [i for i, nm in enumerate(names) if name_substring.lower() in nm.lower()]
    if not matches:
        raise RuntimeError(
            f"No GPU matching {name_substring!r}. Visible devices: "
            + ", ".join(f"cuda:{i} ({nm}, CC {cc})" for i, (nm, cc) in enumerate(infos))
            + ". Set CUDA_VISIBLE_DEVICES to restrict the set."
        )
    index = matches[0]
    name, cc = infos[index]
    device = torch.device(f"cuda:{index}")
    log.info(
        "select_gpu cuda:%s %s (CC %s) order=%s visible=%s",
        index,
        name,
        cc,
        os.environ.get("CUDA_DEVICE_ORDER"),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        extra={
            "cuda_index": index,
            "gpu_name": name,
            "compute_capability": cc,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "all_gpus": names,
        },
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


def main() -> None:
    """Print every visible CUDA device under PCI_BUS_ID order."""
    setup_logging()
    _warn_if_order_unpinned()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '<unset>')}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    n = torch.cuda.device_count()
    if n == 0:
        raise SystemExit("No CUDA devices visible")
    for i in range(n):
        name, cc = _card_info(i)
        print(f"cuda:{i}  {name}  CC {cc}")


if __name__ == "__main__":
    main()
