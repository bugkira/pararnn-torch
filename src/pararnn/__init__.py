"""Parallel Newton training and O(1) decode for nonlinear RNNs (Danieli et al.).

Public surface is ``__all__``: cells / ``ParaRNN``, ``newton_apply``,
``verify_agreement``, paged pool + decode, ``ParaSLSTMForCausalLM``,
speculative verify.
"""

from importlib.metadata import PackageNotFoundError, version

from pararnn.cells import (
    ParaCfC,
    ParaGRU,
    ParaHopfield,
    ParaLSTM,
    ParaM2RNN,
    ParaNLRU,
    ParaRWKV7,
    ParaSLSTM,
    ParaTitans,
    RNNCell,
)
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
from pararnn.verify import (
    AgreementError,
    AgreementReport,
    default_agreement_atol,
    verify_agreement,
)

__all__ = [
    "AgreementError",
    "AgreementReport",
    "BlockStackPool",
    "LinearDraftResult",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "PagedStatePool",
    "ParaCfC",
    "ParaGRU",
    "ParaHopfield",
    "ParaLSTM",
    "ParaM2RNN",
    "ParaNLRU",
    "ParaRNN",
    "ParaRWKV7",
    "ParaSLSTM",
    "ParaSLSTMBlock",
    "ParaSLSTMConfig",
    "ParaSLSTMForCausalLM",
    "ParaTitans",
    "RNNCell",
    "SwiGLU",
    "can_decode_step",
    "decode_step",
    "decode_wx",
    "default_agreement_atol",
    "newton_apply",
    "paged_apply",
    "sequential_apply",
    "verify_agreement",
    "verify_linear_draft",
]

try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.17.2"
