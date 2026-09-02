"""FlashRNN optional-dep glue for benches/examples. Not shipped."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)

_backend_memo: str | bool | None = False


def ensure_cuda_home(*, logger: logging.Logger | None = None) -> None:
    """FlashRNN import calls torch cpp_extension, which requires CUDA_HOME."""
    if os.environ.get("CUDA_HOME"):
        return
    nvidia = Path(torch.__file__).resolve().parent.parent / "nvidia"
    runtime = nvidia / "cuda_runtime"
    if (runtime / "include" / "cuda.h").is_file():
        os.environ["CUDA_HOME"] = str(runtime)
        (logger or log).info("flashrnn CUDA_HOME=%s (pip cuda_runtime)", runtime)


def flashrnn_backend(
    *,
    required: bool = False,
    logger: logging.Logger | None = None,
) -> str | None:
    """NX-AI FlashRNN. ``cuda_fused`` needs CC 8.0; 2080 Ti is 7.5."""
    global _backend_memo
    if _backend_memo is not False:
        return _backend_memo  # type: ignore[return-value]
    lg = logger or log
    ensure_cuda_home(logger=lg)
    try:
        import flashrnn  # noqa: F401
    except (ImportError, RuntimeError, OSError) as exc:
        if required:
            raise RuntimeError(
                "FlashRNN is required for this example. uv sync --extra flashrnn --group dev"
            ) from exc
        lg.warning("flashrnn skip: %s", exc)
        _backend_memo = None
        return None
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        _backend_memo = "cuda_fused"
    else:
        _backend_memo = "triton_fused"
        lg.warning(
            "flashrnn backend=triton_fused (CC %d.%d < 8.0; cuda_fused is Ampere+)",
            major,
            minor,
        )
    return _backend_memo  # type: ignore[return-value]


def flashrnn_heads(d_h: int) -> tuple[int, int]:
    """``(n_heads, d_head)`` with ``n_heads * d_head == d_h``."""
    if d_h % 32 == 0:
        return d_h // 32, 32
    if d_h % 16 == 0:
        return d_h // 16, 16
    raise ValueError(f"d_h={d_h} not divisible by 16 for FlashRNN heads")
