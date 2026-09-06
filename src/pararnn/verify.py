"""Numerics contract: parallel Newton vs sequential oracle.

Users call :func:`verify_agreement` before filing a bug, or enable
``NewtonConfig(verify_first_step=True)`` on a :class:`~pararnn.layers.ParaRNN`
for a one-shot smoke at train start. The sequential unroll is the reference;
Newton+scan must land within ``atol`` / ``rtol``.

See ``docs/numerics-contract.md``: residual gate (``max|F|``) is a fuse;
agreement τ is the brand claim.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pararnn.layers.para_rnn import ParaRNN
from pararnn.solvers.newton import NewtonConfig, NewtonStats, newton_apply
from pararnn.solvers.sequential import sequential_apply

log = logging.getLogger(__name__)

# Match tests/numerics (fp32) and K*(T) campaign τ≈1e-4.
_DEFAULT_ATOL_FP32 = 1e-4
_DEFAULT_ATOL_LOWP = 1e-2


@dataclass(frozen=True)
class AgreementReport:
    """Result of :func:`verify_agreement`.

    Attributes
    ----------
    ok : bool
        ``True`` when every element satisfies the ``atol`` / ``rtol`` band
        (same rule as ``torch.allclose``).
    max_abs : float
        ``(parallel - sequential).abs().amax()``.
    max_rel : float
        Max ``|Δ| / max(|seq|, eps)`` over the trajectory.
    mean_abs : float
        Mean absolute error.
    atol, rtol : float
        Thresholds used for ``ok``.
    shape : tuple of int
        Trajectory shape.
    dtype, device : str
        Parallel tensor dtype / device.
    newton_iters : int or None
        Newton iterations when the cell path reported :class:`NewtonStats`.
    max_residual : float or None
        Final Newton residual when available.
    path : {'cell', 'para_rnn'}
        Which comparison path ran.
    """

    ok: bool
    max_abs: float
    max_rel: float
    mean_abs: float
    atol: float
    rtol: float
    shape: tuple[int, ...]
    dtype: str
    device: str
    newton_iters: int | None = None
    max_residual: float | None = None
    path: str = "cell"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the fields."""
        return asdict(self)


class AgreementError(RuntimeError):
    """Raised when ``verify_agreement(..., raise_on_fail=True)`` fails."""

    def __init__(self, message: str, report: AgreementReport) -> None:
        super().__init__(message)
        self.report = report


def default_agreement_atol(dtype: torch.dtype) -> float:
    """Default absolute tolerance for :func:`verify_agreement`.

    float32 / float64 → ``1e-4`` (numerics tests / K* τ). float16 / bfloat16 →
    ``1e-2``.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return _DEFAULT_ATOL_LOWP
    return _DEFAULT_ATOL_FP32


def verify_agreement(
    module: nn.Module,
    x: Tensor,
    *,
    config: NewtonConfig | None = None,
    h0: Tensor | None = None,
    atol: float | None = None,
    rtol: float = 0.0,
    raise_on_fail: bool = False,
    cu_seqlens: Tensor | None = None,
) -> AgreementReport:
    """Compare parallel Newton+scan to the sequential unroll on ``x``.

    Accepts an ``RNNCell``-like module (``step``), a :class:`ParaRNN`, or a
    :class:`~pararnn.layers.ParaSLSTMBlock` (uses the inner ``.rnn``). One
    call is enough to check whether a suspected bug is a real numerics
    mismatch before opening an issue.

    Parameters
    ----------
    module : nn.Module
        Cell, ``ParaRNN``, or block with ``.rnn: ParaRNN``.
    x : Tensor
        Batch-first input ``(B, T, d_in)`` (or packed shape for cells).
    config : NewtonConfig or None, default=None
        Newton settings for the cell path. ``ParaRNN`` uses ``module.config``.
    h0 : Tensor or None, default=None
        Optional initial state (cell path only).
    atol : float or None, default=None
        Absolute tolerance. Default from :func:`default_agreement_atol`.
    rtol : float, default=0.0
        Relative tolerance (``torch.allclose`` rule).
    raise_on_fail : bool, default=False
        If ``True`` and ``ok`` is false, raise :class:`AgreementError`.
    cu_seqlens : Tensor or None, default=None
        Packed offsets for the cell path.

    Returns
    -------
    AgreementReport
        Metrics and thresholds. Does not retain the trajectories.

    Raises
    ------
    TypeError
        When ``module`` is not a supported surface.
    AgreementError
        When ``raise_on_fail`` and the band is violated.
    ValueError
        Propagated from ``newton_apply`` / ``sequential_apply`` (e.g. ``T < 1``).

    Examples
    --------
    >>> from pararnn import ParaGRU, NewtonConfig, verify_agreement
    >>> cell = ParaGRU(8, 16)
    >>> x = torch.randn(2, 32, 8)
    >>> report = verify_agreement(cell, x, config=NewtonConfig(max_iters=3))
    >>> report.ok
    True
    """
    if x.dim() < 2:
        raise ValueError(f"x must be at least rank-2 (B, T, …), got shape {tuple(x.shape)}")

    atol_v = float(default_agreement_atol(x.dtype) if atol is None else atol)
    rtol_v = float(rtol)

    target = _resolve_target(module)
    with torch.no_grad():
        if isinstance(target, ParaRNN):
            parallel, sequential, stats = _compare_para_rnn(target, x)
            path = "para_rnn"
        else:
            parallel, sequential, stats = _compare_cell(
                target, x, config=config, h0=h0, cu_seqlens=cu_seqlens
            )
            path = "cell"

        diff = (parallel - sequential).abs()
        max_abs = float(diff.amax().item())
        mean_abs = float(diff.mean().item())
        denom = sequential.abs().clamp_min(torch.finfo(sequential.dtype).eps)
        max_rel = float((diff / denom).amax().item())
        ok = bool(torch.allclose(parallel, sequential, atol=atol_v, rtol=rtol_v))

    report = AgreementReport(
        ok=ok,
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        atol=atol_v,
        rtol=rtol_v,
        shape=tuple(int(s) for s in parallel.shape),
        dtype=str(parallel.dtype).replace("torch.", ""),
        device=str(parallel.device),
        newton_iters=None if stats is None else int(stats.iters),
        max_residual=None if stats is None else float(stats.max_residual),
        path=path,
    )
    log.info(
        "verify_agreement",
        extra={
            "ok": report.ok,
            "max_abs": report.max_abs,
            "atol": report.atol,
            "rtol": report.rtol,
            "shape": report.shape,
            "dtype": report.dtype,
            "path": report.path,
            "newton_iters": report.newton_iters,
            "max_residual": report.max_residual,
        },
    )
    if raise_on_fail and not report.ok:
        raise AgreementError(
            f"parallel vs sequential max_abs={report.max_abs:.3e} "
            f"exceeds atol={report.atol:.3e} (rtol={report.rtol:.3e}); "
            f"shape={report.shape} dtype={report.dtype} path={report.path}",
            report,
        )
    return report


def _resolve_target(module: nn.Module) -> nn.Module:
    if isinstance(module, ParaRNN):
        return module
    rnn = getattr(module, "rnn", None)
    if isinstance(rnn, ParaRNN):
        return rnn
    if callable(getattr(module, "step", None)):
        return module
    raise TypeError(
        "verify_agreement expects an RNN cell with step(), a ParaRNN, "
        f"or a block with .rnn: ParaRNN; got {type(module).__name__}"
    )


def _compare_cell(
    cell: nn.Module,
    x: Tensor,
    *,
    config: NewtonConfig | None,
    h0: Tensor | None,
    cu_seqlens: Tensor | None,
) -> tuple[Tensor, Tensor, NewtonStats]:
    cfg = config if config is not None else NewtonConfig()
    stats = NewtonStats()
    sequential = sequential_apply(cell, x, h0=h0, cu_seqlens=cu_seqlens)
    parallel = newton_apply(cell, x, cfg, h0=h0, stats=stats, cu_seqlens=cu_seqlens)
    return parallel, sequential, stats


def _compare_para_rnn(model: ParaRNN, x: Tensor) -> tuple[Tensor, Tensor, NewtonStats | None]:
    """Force newton vs sequential on the same ``ParaRNN`` weights."""
    prev_solver = model.solver
    was_training = model.training
    # eval() kills between-layer dropout so the two paths share the same graph.
    model.eval()
    try:
        model.solver = "sequential"
        sequential = model(x)
        if isinstance(sequential, tuple):
            sequential = sequential[0]
        model.solver = "newton"
        parallel = model(x)
        if isinstance(parallel, tuple):
            parallel = parallel[0]
    finally:
        model.solver = prev_solver
        model.train(was_training)
    stats: NewtonStats | None = None
    if model.last_stats:
        stats = model.last_stats[-1]
    return parallel, sequential, stats
