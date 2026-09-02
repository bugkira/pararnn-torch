"""Call-site ``torch.compile(newton_apply)``: what is actually supported.

Compile is not baked into ``src/``. The compile-safe preset below disables the
D2H residual checks and the sLSTM Picard ``while True``. It still does **not**
trace as a single graph: ``fullgraph=True`` raises. With graph breaks allowed,
both inference (``no_grad``) and training (eq. 2.6 backward) match eager.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch._dynamo.exc import Unsupported
from torch._inductor.exc import InductorError

from pararnn.cells import ParaGRU, ParaLSTM, ParaSLSTM
from pararnn.solvers import NewtonConfig, newton_apply

# Dynamo eager: same graph breaks as inductor without a 10 s+ CPU compile
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
    """Preset that avoids the known data-dependent compile hazards.

    ``residual_atol=None``: default 1e-5 does ``float(residual.amax())``
    inside the Newton loop (D2H + graph break; also a discontinuity for
    grads). ``residual_fail=None``: default 1.0 still D2H in ``_fill_stats``
    after K. ``picard_adapt=False``: sLSTM residual retry is ``while True``.
    max_iters=3 is App. A. This is still not ``fullgraph=True``-safe — see
    ``test_compile_safe_fullgraph_*``.
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
def test_compile_safe_fullgraph_inference_breaks_on_logger() -> None:
    """Eager Newton loop: ``log.isEnabledFor`` is traced before ``is_compiling()``.

    ``newton.py:459`` is ``if log.isEnabledFor(DEBUG) and not is_compiling()``.
    Dynamo evaluates the logger call first (gb0291), then skips the K-loop
    frame. Reordering the conjunct would be a src/ fix; this test pins today.
    """
    torch.compiler.reset()
    torch.manual_seed(2)
    cell = ParaGRU(4, 8).eval()
    x = torch.randn(2, 8, 4)
    compiled = torch.compile(_fwd(cell, compile_safe_config()), fullgraph=True)
    with pytest.raises(Unsupported, match="isEnabledFor"):
        compiled(x)


def test_compile_safe_fullgraph_training_breaks_on_nested_function() -> None:
    """``_NewtonFixedPoint`` is a class defined inside ``newton_apply``.

    Dynamo cannot trace ``builtins.__build_class__`` (gb0007). The Function
    is nested to avoid a racy module-level closure; compile pays for that.
    """
    torch.compiler.reset()
    torch.manual_seed(3)
    cell = ParaGRU(4, 8)
    x = torch.randn(2, 8, 4, requires_grad=True)
    compiled = torch.compile(_fwd(cell, compile_safe_config()), fullgraph=True)
    with pytest.raises(Unsupported, match="__build_class__"):
        compiled(x)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["gru", "lstm"])
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
@torch.no_grad()
def test_compile_safe_fused_slstm_inductor_hits_associative_scan(
    cuda_device: torch.device,
) -> None:
    """Fused ParaSLSTM + default inductor dies inside ``tl.associative_scan``.

    Dynamo traces into ``_slstm_cell_local_scan_kernel``; inductor then
    recompiles that Triton kernel and ``associative_scan`` asserts
    (``scan_op.verify()``). GRU/LSTM fused compile does not hit this. A src/
    fix would mark the kernel as a custom op / ``allow_in_graph``. CPU
    ``backend='eager'`` sLSTM compile (above) is unaffected.
    """
    torch.compiler.reset()
    torch.manual_seed(4)
    cell = _make_cell("slstm", device=cuda_device).eval()
    x = torch.randn(2, 16, 4, device=cuda_device)
    compiled = torch.compile(_fwd(cell, compile_safe_config(scan_backend="auto")))
    with pytest.raises(InductorError, match="associative_scan"):
        compiled(x)


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
def test_compile_safe_fused_fullgraph_inference_breaks_on_logger(
    cuda_device: torch.device,
) -> None:
    """Fused path never reaches the K-loop logger; it hits unguarded ``log.debug``.

    ``_newton_fused`` (newton.py:608) logs without ``is_compiling()``. Other
    debug logs in this file are guarded; this one is not.
    """
    torch.compiler.reset()
    torch.manual_seed(6)
    cell = ParaGRU(4, 8, device=cuda_device).eval()
    x = torch.randn(2, 16, 4, device=cuda_device)
    compiled = torch.compile(
        _fwd(cell, compile_safe_config(scan_backend="auto")),
        fullgraph=True,
    )
    with pytest.raises(Unsupported, match="newton_fused"):
        compiled(x)
