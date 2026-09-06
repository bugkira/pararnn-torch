"""Call-site ``torch.compile(newton_apply)``: compile-safe preset and fullgraph.

Compile is not baked into ``src/``. The compile-safe preset disables residual
host sync and sLSTM Picard ``while True``. With that preset, Dynamo traces
``newton_apply`` as a single graph (``fullgraph=True``) for eager and fused
scans (fused kernels are ``custom_op`` + ``register_fake``).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.solvers import NewtonConfig, newton_apply

# Dynamo eager: same semantics as inductor without a 10 s+ CPU compile
# (measured). CUDA tests use the default inductor backend.
_CPU_BACKEND = "eager"

# atol=1e-5 / rtol=1e-5: fp32 compile vs eager. Measured max |Δ| was 3e-8
# (CPU inductor) and 0 (CUDA fused) on these shapes. 1e-5 sits above fp32
# noise and below the 1e-4 sequential-agreement band (test_parallel.py).
# Fallback: if inductor reorders the W_x GEMM, raise to 1e-4 and log the
# observed maxdiff; that is not an adjoint bug.
_COMPILE_ATOL = 1e-5
_COMPILE_RTOL = 1e-5

_KINDS = ("gru", "lstm", "slstm")


def compile_safe_config(*, scan_backend: str = "eager") -> NewtonConfig:
    """Preset that avoids data-dependent compile hazards.

    ``residual_atol=None``: no ``float(residual.amax())`` in the Newton loop.
    ``residual_fail=None``: no host sync in ``_fill_stats`` after K.
    ``picard_adapt=False``: sLSTM residual retry is ``while True``.
    max_iters=3 is App. A. Fixed-K pure loop + top-level Autograd.Function
    + fused ``custom_op`` are ``fullgraph=True``-safe.
    """
    return NewtonConfig(
        max_iters=3,
        scan_backend=scan_backend,
        residual_atol=None,
        residual_fail=None,
        picard_adapt=False,
    )


def _make_cell(kind: str, device: torch.device | None = None) -> nn.Module:
    kwargs: dict[str, object] = {"input_size": 4, "hidden_size": 8}
    if device is not None:
        kwargs["device"] = device
    if kind == "gru":
        return ParaGRU(**kwargs)
    if kind == "lstm":
        return ParaLSTM(**kwargs)
    return ParaSLSTM(**kwargs, mix="diag")


def _fwd(cell: nn.Module, config: NewtonConfig):
    def fn(x: Tensor) -> Tensor:
        return newton_apply(cell, x, config)

    return fn


def _assert_close(got: Tensor, ref: Tensor) -> None:
    torch.testing.assert_close(got, ref, atol=_COMPILE_ATOL, rtol=_COMPILE_RTOL)


@pytest.mark.parametrize("kind", _KINDS)
@torch.no_grad()
def test_compile_safe_inference_matches_eager(kind: str) -> None:
    torch.manual_seed(0)
    cell = _make_cell(kind).eval()
    x = torch.randn(2, 8, 4)
    cfg = compile_safe_config()
    fn = _fwd(cell, cfg)
    compiled = torch.compile(fn, backend=_CPU_BACKEND)
    _assert_close(compiled(x), fn(x))


@pytest.mark.parametrize("kind", _KINDS)
def test_compile_safe_training_matches_eager(kind: str) -> None:
    torch.manual_seed(1)
    cell_e = _make_cell(kind)
    cell_c = _make_cell(kind)
    cell_c.load_state_dict(cell_e.state_dict())
    cfg = compile_safe_config()
    x_e = torch.randn(2, 8, 4, requires_grad=True)
    x_c = x_e.detach().clone().requires_grad_(True)
    y_e = newton_apply(cell_e, x_e, cfg)
    w = torch.randn_like(y_e)
    (y_e * w).sum().backward()
    compiled = torch.compile(_fwd(cell_c, cfg), backend=_CPU_BACKEND)
    y_c = compiled(x_c)
    (y_c * w).sum().backward()
    _assert_close(y_c, y_e.detach())
    assert x_e.grad is not None
    assert x_c.grad is not None
    _assert_close(x_c.grad, x_e.grad)
    for p_e, p_c in zip(cell_e.parameters(), cell_c.parameters(), strict=True):
        assert p_e.grad is not None
        assert p_c.grad is not None
        _assert_close(p_c.grad, p_e.grad)


@torch.no_grad()
def test_compile_safe_dynamo_explain_zero_graph_breaks() -> None:
    """Compile-safe eager Newton: ``torch._dynamo.explain`` reports 0 breaks."""
    torch.compiler.reset()
    torch.manual_seed(2)
    cell = ParaGRU(4, 8).eval()
    x = torch.randn(2, 8, 4)
    fn = _fwd(cell, compile_safe_config())
    explanation = torch._dynamo.explain(fn)(x)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count >= 1
    compiled = torch.compile(fn, backend=_CPU_BACKEND, fullgraph=True)
    _assert_close(compiled(x), fn(x))


@torch.no_grad()
def test_compile_safe_fullgraph_inference_matches_eager() -> None:
    torch.compiler.reset()
    torch.manual_seed(2)
    cell = ParaGRU(4, 8).eval()
    x = torch.randn(2, 8, 4)
    fn = _fwd(cell, compile_safe_config())
    compiled = torch.compile(fn, backend=_CPU_BACKEND, fullgraph=True)
    _assert_close(compiled(x), fn(x))


def test_compile_safe_fullgraph_training_matches_eager() -> None:
    torch.compiler.reset()
    torch.manual_seed(3)
    cell_e = ParaGRU(4, 8)
    cell_c = ParaGRU(4, 8)
    cell_c.load_state_dict(cell_e.state_dict())
    cfg = compile_safe_config()
    x_e = torch.randn(2, 8, 4, requires_grad=True)
    x_c = x_e.detach().clone().requires_grad_(True)
    y_e = newton_apply(cell_e, x_e, cfg)
    w = torch.randn_like(y_e)
    (y_e * w).sum().backward()
    compiled = torch.compile(_fwd(cell_c, cfg), backend=_CPU_BACKEND, fullgraph=True)
    y_c = compiled(x_c)
    (y_c * w).sum().backward()
    _assert_close(y_c, y_e.detach())
    assert x_e.grad is not None
    assert x_c.grad is not None
    _assert_close(x_c.grad, x_e.grad)
    for p_e, p_c in zip(cell_e.parameters(), cell_c.parameters(), strict=True):
        assert p_e.grad is not None
        assert p_c.grad is not None
        _assert_close(p_c.grad, p_e.grad)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["gru", "lstm", "slstm"])
@torch.no_grad()
def test_compile_safe_fused_inference_matches_eager(kind: str, cuda_device: torch.device) -> None:
    torch.manual_seed(4)
    cell = _make_cell(kind, device=cuda_device).eval()
    x = torch.randn(2, 16, 4, device=cuda_device)
    cfg = compile_safe_config(scan_backend="auto")
    fn = _fwd(cell, cfg)
    compiled = torch.compile(fn)
    _assert_close(compiled(x), fn(x))


@pytest.mark.cuda
def test_compile_safe_fused_training_matches_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(5)
    cell_e = ParaGRU(4, 8, device=cuda_device)
    cell_c = ParaGRU(4, 8, device=cuda_device)
    cell_c.load_state_dict(cell_e.state_dict())
    cfg = compile_safe_config(scan_backend="auto")
    x_e = torch.randn(2, 16, 4, device=cuda_device, requires_grad=True)
    x_c = x_e.detach().clone().requires_grad_(True)
    y_e = newton_apply(cell_e, x_e, cfg)
    y_e.sum().backward()
    compiled = torch.compile(_fwd(cell_c, cfg))
    y_c = compiled(x_c)
    y_c.sum().backward()
    _assert_close(y_c, y_e.detach())
    assert x_e.grad is not None
    assert x_c.grad is not None
    _assert_close(x_c.grad, x_e.grad)
    for p_e, p_c in zip(cell_e.parameters(), cell_c.parameters(), strict=True):
        assert p_e.grad is not None
        assert p_c.grad is not None
        _assert_close(p_c.grad, p_e.grad)


@pytest.mark.cuda
@torch.no_grad()
def test_compile_safe_fused_fullgraph_inference(cuda_device: torch.device) -> None:
    """Fused GRU is a ``custom_op``: ``fullgraph=True`` matches eager."""
    torch.compiler.reset()
    torch.manual_seed(6)
    cell = ParaGRU(4, 8, device=cuda_device).eval()
    x = torch.randn(2, 16, 4, device=cuda_device)
    fn = _fwd(cell, compile_safe_config(scan_backend="auto"))
    compiled = torch.compile(fn, fullgraph=True)
    _assert_close(compiled(x), fn(x))


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:mix='head' is block-diagonal ParaGRU:UserWarning")
@torch.no_grad()
def test_compile_safe_head_gru_fullgraph_inference(cuda_device: torch.device) -> None:
    """Head fused Newton is ``pararnn::newton_gru_head_fused``: fullgraph OK."""
    torch.compiler.reset()
    torch.manual_seed(8)
    cell = ParaGRU(16, 32, mix="head", n_heads=4, device=cuda_device).eval()
    x = torch.randn(2, 16, 16, device=cuda_device)
    fn = _fwd(cell, compile_safe_config(scan_backend="fused"))
    explanation = torch._dynamo.explain(fn)(x)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    compiled = torch.compile(fn, fullgraph=True)
    _assert_close(compiled(x), fn(x))


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:mix='head' is block-diagonal ParaGRU:UserWarning")
def test_compile_safe_head_gru_fullgraph_training(cuda_device: torch.device) -> None:
    torch.compiler.reset()
    torch.manual_seed(9)
    cell_e = ParaGRU(16, 32, mix="head", n_heads=4, device=cuda_device)
    cell_c = ParaGRU(16, 32, mix="head", n_heads=4, device=cuda_device)
    cell_c.load_state_dict(cell_e.state_dict())
    cfg = compile_safe_config(scan_backend="fused")
    x_e = torch.randn(2, 16, 16, device=cuda_device, requires_grad=True)
    x_c = x_e.detach().clone().requires_grad_(True)
    y_e = newton_apply(cell_e, x_e, cfg)
    w = torch.randn_like(y_e)
    (y_e * w).sum().backward()
    compiled = torch.compile(_fwd(cell_c, cfg), fullgraph=True)
    y_c = compiled(x_c)
    (y_c * w).sum().backward()
    _assert_close(y_c, y_e.detach())
    assert x_e.grad is not None and x_c.grad is not None
    _assert_close(x_c.grad, x_e.grad)
    for p_e, p_c in zip(cell_e.parameters(), cell_c.parameters(), strict=True):
        assert p_e.grad is not None and p_c.grad is not None
        _assert_close(p_c.grad, p_e.grad)

