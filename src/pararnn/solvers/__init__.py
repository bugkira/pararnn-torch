from pararnn.kernels.decode import can_decode_step, decode_step, decode_wx
from pararnn.solvers.newton import (
    LIBRARY_NEWTON_ITERS,
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    newton_apply,
    slstm_auto_picard,
    slstm_picard_next,
)
from pararnn.solvers.newton.k_star import (
    auto_newton_iters,
    cfc_auto_newton_iters,
    hopfield_auto_newton_iters,
    titans_auto_newton_iters,
)
from pararnn.solvers.sequential import sequential_apply
from pararnn.solvers.slstm_log import (
    SLSTMLogCoords,
    slstm_clamp_log_coords,
    slstm_decode_log,
    slstm_encode_log,
)
from pararnn.solvers.slstm_picard import (
    slstm_frozen_gate_scan,
    slstm_frozen_gate_scan_eager,
    slstm_picard_init,
    slstm_zero_hidden_init,
)

__all__ = [
    "LIBRARY_NEWTON_ITERS",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "SLSTMLogCoords",
    "auto_newton_iters",
    "can_decode_step",
    "cfc_auto_newton_iters",
    "decode_step",
    "decode_wx",
    "hopfield_auto_newton_iters",
    "newton_apply",
    "sequential_apply",
    "slstm_auto_picard",
    "slstm_clamp_log_coords",
    "slstm_decode_log",
    "slstm_encode_log",
    "slstm_frozen_gate_scan",
    "slstm_frozen_gate_scan_eager",
    "slstm_picard_init",
    "slstm_picard_next",
    "slstm_zero_hidden_init",
    "titans_auto_newton_iters",
]
