"""Critical Newton depth ``K*(T)`` schedules and auto resolution.

``K* = min{K : max|H_par − H_seq|_∞ < τ}``. Library auto schedules are
**measured envelopes** (lab: RTX 3060 / 2080 Ti), not App. A folklore.
Pin ``NewtonConfig.max_iters=int`` to override; leave ``None`` for auto.

Hypotheses recorded in benches (``scripts/bench_k_star.py``):
  H1  K* = O(1)     — Danieli App. A GRU/LSTM band
  H2  K* = Θ(log T) — Gonzalez / PL rate
  H3  K* = Θ(√T)
  H0  K* = Θ(T)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import replace

from torch import Tensor, nn

from pararnn.solvers.newton.config import LIBRARY_NEWTON_ITERS, NewtonConfig

log = logging.getLogger(__name__)

# Default agreement band for K* measurement / schedule margin (fp32).
DEFAULT_KSTAR_TAU = 1e-4


def lookup_iters_by_t(schedule: Mapping[int, int], seq_len: int) -> int:
    """Left-step schedule: largest key ``<= seq_len``, else smallest key."""
    if not schedule:
        raise ValueError("empty newton_iters_by_t schedule")
    keys = sorted(int(k) for k in schedule)
    chosen = int(schedule[keys[0]])
    for k in keys:
        if k <= seq_len:
            chosen = int(schedule[k])
        else:
            break
    return chosen


def _ceil_log2_ramp(seq_len: int, *, base: int, log_coef: float, k_max: int) -> int:
    """Envelope ``base + ceil(log_coef * log2(T))``, clipped to ``[0, k_max]``."""
    t = max(1, int(seq_len))
    k = int(base + math.ceil(log_coef * math.log2(t)))
    return max(0, min(int(k_max), k))


# --- Measured envelopes (scripts/bench_k_star.py, τ=1e-4, RTX 3060) -----------
# Short grid T∈{32…4096} (3 seeds) + long T∈{8k…131072} (2 seeds, B=1).
# Recipe ceilings = max_seed K* + 1 (nondecreasing). H1 O(1) through 131k
# for CfC / Hopfield / Titans; RWKV-7 is linear (K*=0).

# ParaCfC (diag Liquid): K*=2 flat through T=131072 → H1 O(1).
_CFC_BY_T: dict[int, int] = {
    1: 3,
}

# ParaHopfield (dense): short-T K*∈{1,2}; T≥64 → 2 through 131072.
_HOPFIELD_BY_T: dict[int, int] = {
    1: 2,
    64: 3,
}

# ParaTitans (diag L=1 memory): K*=2 flat through T=131072 → H1 O(1).
_TITANS_BY_T: dict[int, int] = {
    1: 3,
}


def cfc_auto_newton_iters(seq_len: int) -> int:
    """ParaCfC recipe ``K(T)`` (measured O(1); pin ``max_iters`` to override)."""
    return lookup_iters_by_t(_CFC_BY_T, seq_len)


def hopfield_auto_newton_iters(seq_len: int) -> int:
    """ParaHopfield recipe ``K(T)`` (dense; measured O(1) on lab grid)."""
    return lookup_iters_by_t(_HOPFIELD_BY_T, seq_len)


def titans_auto_newton_iters(seq_len: int) -> int:
    """ParaTitans recipe ``K(T)`` (diag; measured O(1) pending long-T rebench)."""
    return lookup_iters_by_t(_TITANS_BY_T, seq_len)


def auto_newton_iters(
    cell: nn.Module,
    seq_len: int,
    *,
    schedule: Mapping[int, int] | Callable[[int], int] | None = None,
) -> int:
    """Resolve Newton ``K`` for ``seq_len``.

    Order: explicit ``schedule`` → cell-type table → ``LIBRARY_NEWTON_ITERS``.
    Linear monoids (``jac_structure='rwkv7'``) return ``0``.
    """
    if schedule is not None:
        if callable(schedule):
            return int(schedule(int(seq_len)))
        return lookup_iters_by_t(schedule, int(seq_len))
    jac = getattr(cell, "jac_structure", None)
    if jac == "rwkv7":
        return 0
    name = type(cell).__name__
    if name == "ParaCfC":
        return cfc_auto_newton_iters(seq_len)
    if name == "ParaHopfield":
        return hopfield_auto_newton_iters(seq_len)
    if name == "ParaTitans":
        return titans_auto_newton_iters(seq_len)
    if name == "ParaM2RNN":
        # Factorized matrix cell: recipe floor from m2rnn-jacobian / k_scale.
        # Prefer pin; auto uses log2 ramp capped (campaign: bench_m2rnn_k_scale).
        return _ceil_log2_ramp(seq_len, base=2, log_coef=0.75, k_max=12)
    return int(LIBRARY_NEWTON_ITERS)


def resolve_max_iters(cell: nn.Module, x: Tensor, config: NewtonConfig) -> NewtonConfig:
    """Fill ``max_iters`` when ``None`` (auto ``K*(T)`` envelope).

    Explicit ``int`` is a manual pin and is left unchanged.
    """
    if config.max_iters is not None:
        return config
    seq_len = int(x.shape[1])
    chosen = auto_newton_iters(
        cell,
        seq_len,
        schedule=config.newton_iters_by_t,
    )
    if not torch_compiler_is_compiling():
        log.info(
            "newton_iters_auto cell=%s seq_len=%s max_iters=%s manual_schedule=%s",
            type(cell).__name__,
            seq_len,
            chosen,
            config.newton_iters_by_t is not None,
        )
    return replace(config, max_iters=int(chosen))


def torch_compiler_is_compiling() -> bool:
    import torch

    return bool(torch.compiler.is_compiling())
