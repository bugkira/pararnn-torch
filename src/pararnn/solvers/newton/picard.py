"""sLSTM Picard initial guess and adaptive P-rung retry (para-slstm.md)."""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import torch
from torch import Tensor, nn

from pararnn.cells.para_slstm import ParaSLSTM
from pararnn.solvers.newton.config import NewtonConfig, NewtonStats
from pararnn.solvers.slstm_picard import slstm_picard_init, slstm_zero_hidden_init

log = logging.getLogger("pararnn.solvers.newton")

# Auto Picard rungs (slstm_auto_picard). Train adapt climbs picard_iters.
_PICARD_RUNGS = (1, 3, 5)


def slstm_auto_picard(seq_len: int) -> int:
    """Library Picard P for ParaSLSTM: 1 if T≤64, 3 if T≤2048, else 5.

    Measured at ``d_h=256``, ``x_scale=1``, K=3 (para-slstm.md). Explicit 0 is
    zero-hidden only. Fallback if ``residual_fail`` fires: raise ``picard_iters``.
    """
    t = int(seq_len)
    if t <= 64:
        return 1
    if t <= 2048:
        return 3
    return 5


def slstm_picard_next(picard_iters: int) -> int | None:
    """Next auto Picard rung after ``picard_iters``, or None at the cap (5).

    Ladder is ``{1, 3, 5}`` (slstm_auto_picard / para-slstm.md).
    """
    p = int(picard_iters)
    for rung in _PICARD_RUNGS:
        if p < rung:
            return rung
    return None


def _resolve_picard(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    if config.picard_iters is None:
        if isinstance(cell, ParaSLSTM):
            chosen = slstm_auto_picard(x.shape[1])
            if not torch.compiler.is_compiling():
                log.debug(
                    "slstm_picard_auto",
                    extra={"chosen": chosen, "seq_len": x.shape[1], "d_h": cell.d_h},
                )
            return replace(config, picard_iters=chosen)
        return replace(config, picard_iters=0)
    if config.picard_iters and not isinstance(cell, ParaSLSTM):
        raise TypeError(
            f"NewtonConfig(picard_iters=) is ParaSLSTM only (got {type(cell).__name__})"
        )
    return config


def _picard_retry_needed(res: float, config: NewtonConfig) -> bool:
    if not math.isfinite(res):
        return True
    cap = config.picard_retry_atol
    if cap is not None and res > cap:
        return True
    fail = config.residual_fail
    return fail is not None and res > fail


def _newton_forward_picard_adapt(
    cell: nn.Module,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    stats: NewtonStats | None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Re-run Alg. 1 at the next P rung when the guess was outside the basin."""
    from pararnn.solvers.newton import _fill_stats, _newton_solve

    cfg = config
    while True:
        st = stats if stats is not None else NewtonStats()
        quiet = replace(cfg, residual_fail=None)
        states = _newton_solve(cell, x, quiet, h0=h0, stats=st, cu_seqlens=cu_seqlens)
        res = st.max_residual
        nxt = slstm_picard_next(int(cfg.picard_iters or 0))
        if _picard_retry_needed(res, config) and nxt is not None:
            if not torch.compiler.is_compiling():
                log.info(
                    "picard_adapt",
                    extra={
                        "from_p": int(cfg.picard_iters or 0),
                        "to_p": nxt,
                        "max_residual": res,
                        "seq_len": int(x.shape[1]),
                        "batch": int(x.shape[0]),
                        "d_h": cell.d_h,
                        "scan_backend": cfg.scan_backend,
                    },
                )
            cfg = replace(cfg, picard_iters=nxt)
            continue
        cap = config.residual_fail
        if cap is not None and (not math.isfinite(res) or res > cap):
            _fill_stats(
                cell,
                x,
                states,
                h0,
                config,
                iters=st.iters,
                stats=st,
                residual_history=st.residual_history,
                known_residual=res,
                cu_seqlens=cu_seqlens,
            )
        return states


def _slstm_newton_guess(
    cell: ParaSLSTM,
    x: Tensor,
    config: NewtonConfig,
    *,
    h0: Tensor | None,
    wx: Tensor | None,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    """Zero-hidden, or extra frozen-gate Picard scans. Still O(log T)."""
    pre = wx if wx is not None else cell.W_x(x)
    if config.picard_iters:
        if not torch.compiler.is_compiling():
            log.debug(
                "slstm_picard_init",
                extra={
                    "n_picard": config.picard_iters,
                    "seq_len": x.shape[1],
                    "batch": x.shape[0],
                    "d_h": cell.d_h,
                },
            )
        return slstm_picard_init(
            cell, pre, h0=h0, n_picard=config.picard_iters, cu_seqlens=cu_seqlens
        )
    return slstm_zero_hidden_init(pre, eps=cell.eps, h0=h0, cu_seqlens=cu_seqlens)
