"""Parallel Newton training and O(1) decode for nonlinear RNNs (Danieli et al.).

Public surface is ``__all__``: cells / ``ParaRNN``, ``newton_apply``,
paged pool + decode, ``ParaSLSTMForCausalLM``, speculative verify.
"""

from importlib.metadata import PackageNotFoundError, version

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM, RNNCell
from pararnn.kernels import can_decode_step, decode_step, decode_wx
from pararnn.layers import ParaRNN, ParaSLSTMBlock, SwiGLU
from pararnn.models import ParaSLSTMConfig, ParaSLSTMForCausalLM
from pararnn.paged import PagedStatePool, paged_apply
from pararnn.serve import BlockStackPool
from pararnn.solvers import (
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    sequential_apply,
)
from pararnn.speculative import LinearDraftResult, verify_linear_draft

__all__ = [
    "BlockStackPool",
    "LinearDraftResult",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "PagedStatePool",
    "ParaGRU",
    "ParaLSTM",
    "ParaRNN",
    "ParaSLSTM",
    "ParaSLSTMBlock",
    "ParaSLSTMConfig",
    "ParaSLSTMForCausalLM",
    "RNNCell",
    "SwiGLU",
    "can_decode_step",
    "decode_step",
    "decode_wx",
    "newton_apply",
    "paged_apply",
    "sequential_apply",
    "verify_linear_draft",
]

try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.9.0"
