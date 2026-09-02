from importlib.metadata import PackageNotFoundError, version

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM, RNNCell
from pararnn.layers import ParaRNN
from pararnn.models import xLSTMBlock
from pararnn.solvers import (
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    sequential_apply,
)

__all__ = [
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "ParaGRU",
    "ParaLSTM",
    "ParaRNN",
    "ParaSLSTM",
    "RNNCell",
    "newton_apply",
    "sequential_apply",
    "xLSTMBlock",
]

try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.3.0"
