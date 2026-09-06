"""Finite-difference check of the eq. 2.6 adjoint.

``newton_apply`` wraps a custom ``torch.autograd.Function`` whose backward is
one reverse scan (Danieli et al. 2025 eq. 2.6). There is no other gradcheck
in the repo; these tests are the CPU CI signal that ∇_x, ∇_θ, and ∇_{h0}
match central differences in float64.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call

from pararnn.cells import ParaCfC, ParaGRU, ParaHopfield, ParaLSTM, ParaNLRU, ParaSLSTM, ParaTitans
from pararnn.solvers import NewtonConfig, newton_apply

# Tiny shapes: gradcheck is O(n_inputs) forwards. B=2, T=3, d_in=2, d_h=3
# is enough for a Blelloch pad-to-4 and a multi-slot state, and keeps this
# file well under a minute on CPU (measured 5.2 s for all 9 cases).
# Hopfield uses dense J; d_h=3 keeps jacrev / scan_dense cheap for CI.
_B, _T, _D_IN, _D_H = 2, 3, 2, 3

# eps=1e-6: PyTorch float64 gradcheck default (gradcheck.py). Central
# differences of a C^∞ map; Newton residual is 0 at these shapes after K=3
# (measured). Fallback: if FD noise dominates (Higham), raise eps to 1e-5
# before loosening atol.
_GRADCHECK_EPS = 1e-6
# atol=1e-5: PyTorch float64 default. IFT adjoint at a K=3 fixed point should
# match FD to ~1e-8 here; 1e-5 still fails an O(1) adjoint bug. Fallback:
# print the failing Jacobian entries; do not raise above 1e-4 (float32
# sequential-BPTT band in tests/numerics/test_parallel.py).
_GRADCHECK_ATOL = 1e-5
# rtol=1e-4: tighter than PyTorch's 1e-3 so tiny-magnitude entries cannot hide
# a wrong adjoint behind relative slack. Fallback: same as atol — inspect,
# do not loosen.
_GRADCHECK_RTOL = 1e-4

_KINDS = ("gru", "lstm", "slstm", "nlru", "cfc", "hopfield", "titans")


def _make_cell(kind: str) -> nn.Module:
    # max_recurrent_norm=None: App. C.1 clamp is piecewise-linear. At |a|=cap
    # the packed VJP uses ``<=`` while ``Tensor.clamp`` is 0 on the boundary.
    # Disable the clip so this file tests the smooth eq. 2.6 map.
    kwargs = {
        "input_size": _D_IN,
        "hidden_size": _D_H,
        "dtype": torch.float64,
        "max_recurrent_norm": None,
    }
    if kind == "gru":
        return ParaGRU(**kwargs)
    if kind == "lstm":
        return ParaLSTM(**kwargs)
    if kind == "nlru":
        return ParaNLRU(**kwargs)
    if kind == "cfc":
        return ParaCfC(**kwargs)
    if kind == "hopfield":
        # No App. C.1 clamp on this cell; dense Softmax map is C^∞.
        return ParaHopfield(input_size=_D_IN, hidden_size=_D_H, dtype=torch.float64)
    if kind == "titans":
        return ParaTitans(**kwargs)
    return ParaSLSTM(**kwargs, mix="diag")


def _x_randn(*, requires_grad: bool = False) -> Tensor:
    """Random ``(B,T,d_in)``; CfC last channel is positive Δt (above clamp floor)."""
    x = torch.randn(_B, _T, _D_IN, dtype=torch.float64)
    # Soft floor so clamp_min(_DT_EPS) is inactive and gradcheck stays C¹.
    x[..., -1] = 0.05 + torch.rand(_B, _T, dtype=torch.float64)
    if requires_grad:
        x.requires_grad_(True)
    return x


def _gradcheck_config(cell: nn.Module) -> NewtonConfig:
    # residual_atol defaults to 1e-5 and early-stops when max|F| drops below
    # that. K then depends on the residual — a step function of (x, θ, h0) —
    # so gradcheck fails even with a correct adjoint. Disable it and run all
    # K=3 (App. A). residual_fail is a post-hoc raise when it fires; None also
    # skips the D2H in _fill_stats. picard_adapt retries P from the residual
    # (discontinuous).
    kwargs: dict[str, object] = {
        "max_iters": 3,
        "scan_backend": "eager",
        "residual_atol": None,
        "residual_fail": None,
        "picard_adapt": False,
    }
    if isinstance(cell, ParaSLSTM):
        # T=3 is in the auto-P=1 bucket (slstm_auto_picard, T<=64). Pin P so
        # a residual retry cannot change the guess.
        kwargs["picard_iters"] = 1
    if isinstance(cell, ParaHopfield):
        # Dense Softmax couples channels; K=6 matches MixTanh / hopfield
        # numerics tests. Fallback: raise to 8 on residual before loosening FD.
        kwargs["max_iters"] = 6
        kwargs["jac_structure"] = "dense"
    return NewtonConfig(**kwargs)


def _h0_like(cell: nn.Module) -> Tensor:
    slots = getattr(cell, "state_slots", 1)
    if slots == 1:
        return torch.randn(_B, cell.d_h, dtype=torch.float64)
    return torch.randn(_B, slots, cell.d_h, dtype=torch.float64)


_TARGET_OFFSET = {"x": 0, "h0": 1, "theta": 2}


def _seed(kind: str, target: str) -> None:
    # sLSTM stabilizer is torch.maximum(z_f+m, z_i). At a tie the analytic
    # subgradient splits 0.5/0.5 (_maximum_subgrad_left); PyTorch maximum
    # matches that, but only off-tie is C^1. float64 randn hits an exact tie
    # with probability 0; these seeds are measured off-tie.
    torch.manual_seed(1000 + 10 * _KINDS.index(kind) + _TARGET_OFFSET[target])


class _NewtonModule(nn.Module):
    """Put cell parameters on a module so ``functional_call`` can perturb them."""

    def __init__(self, cell: nn.Module, config: NewtonConfig) -> None:
        super().__init__()
        self.cell = cell
        self._config = config

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        return newton_apply(self.cell, x, self._config, h0=h0)


def _gradcheck(fn, inputs: tuple[Tensor, ...]) -> None:
    assert torch.autograd.gradcheck(
        fn,
        inputs,
        eps=_GRADCHECK_EPS,
        atol=_GRADCHECK_ATOL,
        rtol=_GRADCHECK_RTOL,
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_gradcheck_input(kind: str) -> None:
    _seed(kind, "x")
    cell = _make_cell(kind)
    cfg = _gradcheck_config(cell)
    x = _x_randn(requires_grad=True)

    def fn(xx: Tensor) -> Tensor:
        return newton_apply(cell, xx, cfg)

    _gradcheck(fn, (x,))


@pytest.mark.parametrize("kind", _KINDS)
def test_gradcheck_h0(kind: str) -> None:
    _seed(kind, "h0")
    cell = _make_cell(kind)
    cfg = _gradcheck_config(cell)
    x = _x_randn()
    h0 = _h0_like(cell).requires_grad_(True)

    def fn(hh: Tensor) -> Tensor:
        return newton_apply(cell, x, cfg, h0=hh)

    _gradcheck(fn, (h0,))


@pytest.mark.parametrize("kind", _KINDS)
def test_gradcheck_parameters(kind: str) -> None:
    _seed(kind, "theta")
    cell = _make_cell(kind)
    cfg = _gradcheck_config(cell)
    x = _x_randn()
    wrap = _NewtonModule(cell, cfg)
    names = [n for n, _ in wrap.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_(True) for p in wrap.parameters())

    def fn(*ps: Tensor) -> Tensor:
        return functional_call(wrap, dict(zip(names, ps, strict=True)), (x,))

    _gradcheck(fn, params)
