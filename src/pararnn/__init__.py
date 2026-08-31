from pararnn.cells import ParaGRU, ParaLSTM
from pararnn.device import experiment_device
from pararnn.solvers import (
    NewtonConfig,
    newton_apply,
    sequential_apply,
    sequential_apply_compiled,
)

__all__ = [
    "NewtonConfig",
    "ParaGRU",
    "ParaLSTM",
    "experiment_device",
    "newton_apply",
    "sequential_apply",
    "sequential_apply_compiled",
]
__version__ = "0.1.0"
