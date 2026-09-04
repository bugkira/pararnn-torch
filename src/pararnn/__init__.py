from importlib.metadata import PackageNotFoundError, version

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM, RNNCell
from pararnn.layers import ParaRNN
from pararnn.paged import PagedStatePool, paged_apply
from pararnn.solvers import (
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    sequential_apply,
)
from pararnn.speculative import LinearDraftResult, verify_linear_draft

__all__ = [
    "LinearDraftResult",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "PagedStatePool",
    "ParaGRU",
    "ParaLSTM",
    "ParaRNN",
    "ParaSLSTM",
    "RNNCell",
    "newton_apply",
    "paged_apply",
    "sequential_apply",
    "verify_linear_draft",
]

try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.6.0"
