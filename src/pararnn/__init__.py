from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.hw import wait_until_free
from pararnn.layers import ParaRNN
from pararnn.solvers import (
    LIBRARY_NEWTON_ITERS,
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    sequential_apply,
)

if TYPE_CHECKING:
    import torch

    device: torch.device

__all__ = [
    "LIBRARY_NEWTON_ITERS",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "ParaGRU",
    "ParaLSTM",
    "ParaRNN",
    "ParaSLSTM",
    "device",
    "newton_apply",
    "sequential_apply",
    "wait_until_free",
]


def __getattr__(name: str) -> Any:
    if name == "device":
        from pararnn.hw import select_device

        value = select_device(allow_cpu=True)
        globals()["device"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.3.0"
