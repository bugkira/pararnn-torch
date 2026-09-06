"""Deterministic-algorithm checks for eq. 2.6 packed VJP.

Packed ParaGRU / ParaLSTM / ParaSLSTM ``mix='diag'`` / ``ParaNLRU`` / ``ParaCfC``
reduce with tile ``tl.sum`` then PyTorch ``.sum`` (no Triton atomics). When the
host enables
``torch.use_deterministic_algorithms(True)``, the first packed ``cell_vjp``
re-runs once and emits a **single** warning if parameter grads drift.

cuBLAS GEMM (``∇x`` / ``W_x``) still needs ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
(or ``:16:8``) in the process environment; we warn once if the flag is on and
that variable is missing.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch import nn

log = logging.getLogger(__name__)

_CHECKED_VJP = False
_WARNED_VJP = False
_WARNED_CUBLAS = False


def reset_determinism_warnings() -> None:
    """Test hook: allow the one-shot warnings / check to fire again."""
    global _CHECKED_VJP, _WARNED_VJP, _WARNED_CUBLAS
    _CHECKED_VJP = False
    _WARNED_VJP = False
    _WARNED_CUBLAS = False


def deterministic_algorithms_enabled() -> bool:
    fn = getattr(torch, "are_deterministic_algorithms_enabled", None)
    if fn is None:
        return False
    return bool(fn())


def warn_cublas_workspace_once() -> None:
    """One-shot note when deterministic algos are on without cuBLAS workspace."""
    global _WARNED_CUBLAS
    if _WARNED_CUBLAS or not deterministic_algorithms_enabled():
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        return
    _WARNED_CUBLAS = True
    log.warning(
        "torch.use_deterministic_algorithms is on, but CUBLAS_WORKSPACE_CONFIG "
        "is unset. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) so cuBLAS "
        "GEMMs used by W_x / ∇x stay deterministic; otherwise PyTorch may error "
        "or fall back."
    )


def _param_grads_bitmatch(a: tuple[Tensor | None, ...], b: tuple[Tensor | None, ...]) -> bool:
    if len(a) != len(b):
        return False
    for g1, g2 in zip(a, b, strict=True):
        if g1 is None and g2 is None:
            continue
        if g1 is None or g2 is None:
            return False
        if int((g1 != g2).sum().item()) != 0:
            return False
    return True


def maybe_check_packed_vjp_once(
    cell: nn.Module,
    h_prev: Tensor,
    x: Tensor,
    mu: Tensor,
    *,
    packed: bool,
    vjp_fn,
) -> None:
    """On first packed VJP under deterministic algos, re-run and warn on drift.

    ``vjp_fn`` is ``cell_vjp`` (passed in to avoid a circular import). Only
    recurrent parameter grads are compared; ``∇x`` is a GEMM.
    """
    global _CHECKED_VJP, _WARNED_VJP
    if _CHECKED_VJP or not packed or not deterministic_algorithms_enabled():
        return
    _CHECKED_VJP = True
    warn_cublas_workspace_once()
    # Second call must see the same inputs; no mutation in packed VJP.
    _gx2, gp2 = vjp_fn(cell, h_prev, x, mu, packed=packed)
    _gx1, gp1 = vjp_fn(cell, h_prev, x, mu, packed=packed)
    del _gx1, _gx2
    if _param_grads_bitmatch(gp1, gp2):
        return
    if _WARNED_VJP:
        return
    _WARNED_VJP = True
    log.warning(
        "ParaRNN packed eq. 2.6 VJP parameter gradients differed across two "
        "identical calls while torch.use_deterministic_algorithms(True). "
        "Expected bit-stable tile-sum reductions (no atomics). Check dtype / "
        "device and file an issue with cell=%s shape h=%s x=%s.",
        type(cell).__name__,
        tuple(h_prev.shape),
        tuple(x.shape),
    )
