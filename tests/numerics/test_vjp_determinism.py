"""Packed VJP: kernel vs eager formulas; CUDA param-grad determinism.

Reduction is tile ``tl.sum`` into fp32 ``(B, n_tiles, d_h)`` then PyTorch
``.sum`` for GRU, LSTM, and sLSTM. No atomics: two packed VJPs on the same
tensors must bit-match, including Turing (CC 7.5).

Under ``torch.use_deterministic_algorithms(True)`` (+ ``CUBLAS_WORKSPACE_CONFIG``),
``newton_apply`` backward must complete; the one-shot packed-VJP drift check
stays quiet when grads bit-match.
"""

from __future__ import annotations

import logging
import os

import pytest
import torch
from torch import Tensor, nn

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.determinism import reset_determinism_warnings
from pararnn.kernels.vjp_gru import gru_recurrence_vjp, gru_recurrence_vjp_eager
from pararnn.kernels.vjp_lstm import lstm_recurrence_vjp, lstm_recurrence_vjp_eager
from pararnn.kernels.vjp_slstm import slstm_recurrence_vjp, slstm_recurrence_vjp_eager
from pararnn.solvers import NewtonConfig, newton_apply
from pararnn.solvers.vjp import cell_vjp

_KINDS = ("gru", "lstm", "slstm")
_DTYPES = (torch.float32, torch.float16)


def _make_cell(kind: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    kw = {"d_in": 8, "d_h": 32, "device": device, "dtype": dtype}
    if kind == "gru":
        return ParaGRU(**kw)
    if kind == "lstm":
        return ParaLSTM(**kw)
    return ParaSLSTM(mix="diag", **kw)


def _inputs(kind: str, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    batch, time, d_in, d_h = 4, 17, 8, 32
    x = torch.randn(batch, time, d_in, device=device, dtype=dtype)
    if kind == "gru":
        h = torch.randn(batch, time, d_h, device=device, dtype=dtype)
        mu = torch.randn(batch, time, d_h, device=device, dtype=dtype)
    elif kind == "lstm":
        h = torch.randn(batch, time, 2, d_h, device=device, dtype=dtype)
        mu = torch.randn(batch, time, 2, d_h, device=device, dtype=dtype)
    else:
        c = torch.randn(batch, time, d_h, device=device, dtype=dtype) * 0.5
        n = torch.nn.functional.softplus(torch.randn(batch, time, d_h, device=device, dtype=dtype))
        m = torch.randn(batch, time, d_h, device=device, dtype=dtype) * 0.5
        hh = torch.randn(batch, time, d_h, device=device, dtype=dtype) * 0.5
        h = torch.stack((c, n, m, hh), dim=-2)
        mu = torch.randn(batch, time, 4, d_h, device=device, dtype=dtype)
    return h, x, mu


def _atol_rtol(kind: str, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 5e-3, 5e-3
    if kind == "slstm":
        return 1e-4, 1e-4
    return 2e-5, 2e-5


@pytest.mark.cuda
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_packed_vjp_param_grads_bitwise_stable(
    kind: str, dtype: torch.dtype, cuda_device: torch.device
) -> None:
    """Recurrent ``a_*`` / ``R`` and ``W_x`` grads bit-match across two VJPs.

    ``∇x`` is a GEMM (``g_wx @ W``); cuBLAS bit-stability is out of scope.
    """
    torch.manual_seed(7)
    cell = _make_cell(kind, cuda_device, dtype)
    h, x, mu = _inputs(kind, cuda_device, dtype)
    _gx1, gp1 = cell_vjp(cell, h, x, mu, packed=True)
    _gx2, gp2 = cell_vjp(cell, h, x, mu, packed=True)
    names = [n for n, _p in cell.named_parameters()]
    for name, a, b in zip(names, gp1, gp2, strict=True):
        if a is None and b is None:
            continue
        assert a is not None, name
        assert b is not None, name
        n_diff = int((a != b).sum().item())
        assert n_diff == 0, f"{kind} {dtype} {name}: {n_diff}/{a.numel()} differing"


@pytest.mark.cuda
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_vjp_matches_eager_formulas(
    kind: str, dtype: torch.dtype, cuda_device: torch.device
) -> None:
    torch.manual_seed(11)
    cell = _make_cell(kind, cuda_device, dtype)
    h, x, mu = _inputs(kind, cuda_device, dtype)
    atol, rtol = _atol_rtol(kind, dtype)
    if kind == "gru":
        a_z, a_r, a_n = cell.clipped_a()
        wx = cell.W_x(x)
        got = gru_recurrence_vjp(h, wx, a_z, a_r, a_n, mu)
        ref = gru_recurrence_vjp_eager(h, wx, a_z, a_r, a_n, mu)
    elif kind == "lstm":
        a_f, a_z, a_o, c_f, c_o = cell.clipped_recurrent()
        wx = cell.W_x(x)
        got = lstm_recurrence_vjp(h, wx, a_f, a_z, a_o, c_f, c_o, mu)
        ref = lstm_recurrence_vjp_eager(h, wx, a_f, a_z, a_o, c_f, c_o, mu)
    else:
        rec = cell.clipped_r()
        wx = cell.W_x(x)
        got = slstm_recurrence_vjp(h, wx, rec, mu, cell.eps)
        ref = slstm_recurrence_vjp_eager(h, wx, rec, mu, cell.eps)
    assert len(got) == len(ref)
    for g, e in zip(got, ref, strict=True):
        torch.testing.assert_close(g, e, atol=atol, rtol=rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", _KINDS)
def test_newton_under_use_deterministic_algorithms(
    kind: str, cuda_device: torch.device, caplog: pytest.LogCaptureFixture
) -> None:
    """Flag on + cuBLAS workspace: forward/backward finish; no drift warning."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reset_determinism_warnings()
    torch.manual_seed(13)
    cell = _make_cell(kind, cuda_device, torch.float32)
    batch, time, d_in = 2, 16, 8
    x = torch.randn(batch, time, d_in, device=cuda_device, dtype=torch.float32)
    cfg = NewtonConfig(max_iters=2, residual_atol=None)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with caplog.at_level(logging.WARNING, logger="pararnn.determinism"):
            h, _stats = newton_apply(cell, x, cfg)
            h.sum().backward()
        drift = [r for r in caplog.records if "differed across two" in r.getMessage()]
        assert not drift, "packed VJP should bit-match; no one-shot drift warning"
    finally:
        torch.use_deterministic_algorithms(prev)
        reset_determinism_warnings()


@pytest.mark.cuda
def test_cublas_workspace_warn_once(
    cuda_device: torch.device, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One-shot cuBLAS env note when deterministic algos are on and config unset."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    reset_determinism_warnings()
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        cell = _make_cell("gru", cuda_device, torch.float32)
        h, x, mu = _inputs("gru", cuda_device, torch.float32)
        with caplog.at_level(logging.WARNING, logger="pararnn.determinism"):
            cell_vjp(cell, h, x, mu, packed=True)
            cell_vjp(cell, h, x, mu, packed=True)
        msgs = [r.getMessage() for r in caplog.records if "CUBLAS_WORKSPACE_CONFIG" in r.getMessage()]
        assert len(msgs) == 1
    finally:
        torch.use_deterministic_algorithms(prev)
        reset_determinism_warnings()
