"""NewtonConfig, stats, and library constants (Danieli et al. 2025 App. A)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# App. A: K=3 reaches machine precision on these cells. Sequential agreement
# tests use 1e-4. Stop a wasted extra iter below that and above fp32 noise.
_DEFAULT_RESIDUAL_ATOL = 1e-5
# Library contract: K=3 (App. A). sLSTM basin: raise picard_iters (para-slstm.md).
# CfC / Hopfield / Titans / M²RNN: prefer max_iters=None → measured K*(T).
LIBRARY_NEWTON_ITERS = 3  # default Newton K (Danieli et al. App. A)
# After K steps, max|F| above this is divergence.
# Sequential agreement is 1e-4…2e-3; diverged sLSTM is 1e2…1e14 (para-slstm.md).
# 1.0 sits between. None disables (K-curves, P=0 timing benches).
_DEFAULT_RESIDUAL_FAIL = 1.0
# Warn when max|F| is past sequential-agreement but under residual_fail.
# Train can miss the P=1 basin on one batch after Adam (para-slstm.md).
_RESIDUAL_WARN = 1e-3


class NewtonDivergenceError(RuntimeError):
    """Raised when ``max |F|`` exceeds ``NewtonConfig.residual_fail`` after ``K`` steps.

    For ParaSLSTM, increase ``picard_iters`` (or leave ``None`` for auto)
    before increasing ``max_iters``.
    """


@dataclass
class NewtonStats:
    """Filled by ``newton_apply(..., stats=)``; also ``ParaRNN.last_stats``.

    Attributes
    ----------
    max_residual : float
        Final ``max |F(H)|`` (NaN if unused).
    iters : int
        Newton residual evals (``<= max_iters``); may be lower with ``fused_early_exit``.
    scan_backend : str
        Resolved backend (``fused`` / ``triton`` / ``eager`` / ``context_parallel``).
    picard_iters : int
        Picard depth used (ParaSLSTM).
    residual_history : tuple of float
        Per-iter residuals when recorded.
    """

    max_residual: float = float("nan")
    # Residual evaluations in the Newton loop (≤ max_iters), including the
    # eval that triggered early-stop. 0 if max_iters=0. Fused default: max_iters;
    # with ``fused_early_exit`` this can be ``< max_iters``.
    iters: int = 0
    scan_backend: str = ""
    picard_iters: int = 0
    residual_history: tuple[float, ...] = ()


@dataclass
class NewtonConfig:
    """Knobs for the parallel Newton+scan forward (Danieli et al. App. A).

    Default ``max_iters=3`` matches App. A for GRU/LSTM. ParaSLSTM usually
    also needs ``picard_iters`` (``None`` = auto from ``T``). For CfC /
    Hopfield / Titans / M²RNN, ``max_iters=None`` selects a measured
    ``K*(T)`` envelope (``auto_newton_iters``); pin an ``int`` to override.

    Attributes
    ----------
    max_iters : int or None
        Newton steps ``K``. Default ``LIBRARY_NEWTON_ITERS`` (= 3, App. A).
        ``None`` → cell ``K*(T)`` schedule (``newton_iters_by_t`` /
        ``auto_newton_iters``). Explicit ``int`` is a manual pin.
    omega : float
        Step damping (``1.0`` = undamped; ``(0, 1)`` damps).
    scan_backend : str
        ``"auto"`` | ``"eager"`` | ``"triton"`` | ``"fused"`` |
        ``"context_parallel"``.
    jacobian : str
        ``"auto"`` | ``"analytic"`` | ``"autograd"``.
    jac_structure : str or None
        ``"diag"`` | ``"block2"`` | ``"block4"`` | ``"head"`` | ``"dense"``,
        or ``None`` to infer.
    residual_atol : float or None
        Early-stop threshold on ``max |F|`` (default ``1e-5``; ``None`` = run all ``K``).
    residual_fail : float or None
        Raise :exc:`NewtonDivergenceError` above this (default ``1.0``; ``None`` disables).
    coords : str
        ``"native"`` or ``"log"`` (ParaSLSTM LSE cell).
    chunk_len : int or None
        Windowed solve length; ``None`` = full ``T``.
    picard_iters : int or None
        ParaSLSTM warm-start depth ``P``; ``None`` = auto ``{1,3,5}``; ``0`` = zero-hidden.
        On ``ParaM2RNN``, ``>=1`` selects frozen-``W`` warm-start
        (``m2rnn_frozen_w_scan``); ``0`` / ``None`` keep App. A init.
    picard_adapt : bool or None
        Auto-raise ``P`` on large residual when ``picard_iters`` was auto.
    picard_retry_atol : float or None
        Residual threshold for that retry (default ``1e-3``).
    newton_iters_by_t : mapping or None
        Manual ``K*(T)`` table when ``max_iters is None`` (left-step: largest
        key ``<= T``). Overrides the cell default envelope.
        Example: ``{64: 2, 1024: 3, 4096: 4}``.
    scan_tile : str
        Fused/Triton tile algebra: ``"assoc"`` (default) | ``"seq"`` | Thomas variants.
    fused_time_loop : bool
        Windowed fused walk with state carry (default ``False`` = global scan).
    fused_window_len : int or None
        Window size when ``fused_time_loop`` (``32``/``64``/``128``; default ``64``).
    recompute : bool
        Rematerialize ``H*`` in backward (VRAM trade); default stores ``H*`` (IFT).
    fused_early_exit : bool
        Host-sync early-stop inside fused ``K`` (experimental; needs ``residual_atol``).
    verify_first_step : bool
        On a ``ParaRNN``, run :func:`pararnn.verify_agreement` once on the first
        Newton forward (smoke: shapes / device / dialect vs sequential τ).
        Default ``False`` (opt-in; ~one sequential unroll cost).

    See Also
    --------
    newton_apply, NewtonStats, NewtonDivergenceError, auto_newton_iters
    """

    # App. A: K=3. None → measured K*(T) envelope (CfC/Hopfield/Titans/M²RNN).
    max_iters: int | None = LIBRARY_NEWTON_ITERS
    omega: float = 1.0  # 1 = vanilla Newton; <1 damps (Gonzalez et al. ELK)
    # auto: fused CUDA GRU/LSTM/sLSTM-diag; else Triton scan + step; else eager Blelloch.
    # context_parallel: diag scan shards T across the default process group (Newton
    # trajectory stays replicated; scan work is T/N). Requires init_process_group.
    scan_backend: str = "auto"
    # auto: analytic J if step_with_jacobian else Autograd. analytic: require it.
    jacobian: str = "auto"
    # None infers from state / cell.jac_structure. diag | block2 | block4 | head | dense.
    jac_structure: str | None = None
    # None: run all K. Default 1e-5 skips leftover K when max|F| is already small (App. A).
    residual_atol: float | None = _DEFAULT_RESIDUAL_ATOL
    # After last K, raise if max|F| exceeds this. Default 1.0. None: K-curves / benches.
    residual_fail: float | None = _DEFAULT_RESIDUAL_FAIL
    # native | log. log: ParaSLSTM LSE cell in (u, log n, m, h).
    coords: str = "native"
    # None = one Newton over T. int: sequential chunks; 64 from T=64 K=3 at d_h=256.
    # Backward windows the eq. 2.6 reverse scan the same way (carry μ via ∇_{h0}).
    chunk_len: int | None = None
    # None = auto P ∈ {1, 3, 5} from T for ParaSLSTM. Explicit 0 is zero-hidden.
    # ParaM2RNN: >=1 → frozen-W (W=0) linear scan warm-start before Newton.
    picard_iters: int | None = None
    # None: retry P when picard_iters was auto. True/False force. Better initial guess.
    picard_adapt: bool | None = None
    # Retry next P if max|F| exceeds this (1e-3 = sequential-agreement band).
    # None: only residual_fail.
    picard_retry_atol: float | None = _RESIDUAL_WARN
    # Manual K*(T) when max_iters is None (overrides cell envelope).
    newton_iters_by_t: Mapping[int, int] | None = None
    # assoc: tl.associative_scan. seq: serial prefix in the tile (ablation).
    # thomas / thomas4: C=4 sequential compose then PCR. thomas2: C=2.
    # C is an ablation (this repo's bench_slstm_scan_opt.py on 3060 / 2080 Ti),
    # not Apple App. C. Default assoc until that bench prefers Thomas.
    scan_tile: str = "assoc"
    # True: one fused launch walks T tiles with solved-state carry (windowed
    # Newton, same residual as chunk_len=window). False: global two-level
    # scan. vs global assoc is a different residual, not a lost m_t. Default
    # False: sequential agreement uses the global solve.
    fused_time_loop: bool = False
    # None: FUSED_WINDOW_DEFAULT (64) when fused_time_loop. 32 trips
    # newton_residual_high (~8e-2) at T=1024 d_h=256 P=3; 64 matched assoc
    # (~3e-5, vs sequential ~4e-7). 128 if 64 stays >1e-3.
    fused_window_len: int | None = None
    # False (Level 1): IFT saves H* on the Autograd Function; 0 extra FLOPs
    # (Danieli et al. eq. 2.6). True (Level 2): drop H* from save_for_backward
    # and rematerialize via a second Newton forward in backward — for ultra-long
    # train T (≳64k…128k) when VRAM of S* is the limiter; sketch +15–20% FLOPs
    # / −70–80% of the Function-held trajectory (this repo IDEAS). Prefer outer
    # ``torch.utils.checkpoint`` on short T; combine both only if measured.
    recompute: bool = False
    # False (default): fused Alg. 1 always runs ``max_iters`` (train / DDP /
    # compile-safe fixed K). True: after each fused Newton step, host-sync
    # ``max|F|`` and stop when ``< residual_atol``. Experimental — variable K
    # straggles multi-GPU and breaks CUDA-graph assumptions; use for inference
    # ablations / residual curves. Requires ``residual_atol``; rejects
    # ``fused_time_loop``. Eager already early-stops via ``residual_atol`` alone.
    fused_early_exit: bool = False
    # Opt-in numerics smoke on ParaRNN: first Newton forward calls
    # verify_agreement(..., raise_on_fail=True). Skipped under torch.compile.
    verify_first_step: bool = False


# Lengths compiled as Triton BLOCK_T in the fused walk kernel. Not Apple App. C.
FUSED_WINDOW_LENS = (32, 64, 128)
# chunk_len=64 at T=1024 d_h=256 P=3 ~2e-4 vs sequential (this repo).
FUSED_WINDOW_DEFAULT = 64


def compile_safe_config(*, scan_backend: str = "eager") -> NewtonConfig:
    """``NewtonConfig`` preset for ``torch.compile(..., fullgraph=True)``.

    Disables residual host sync and sLSTM Picard residual retry so Dynamo
    sees a fixed-``K`` pure loop. Fused backends stay Dynamo-opaque via
    ``custom_op`` + ``register_fake``. See ``docs/compile-amp.md``.

    Parameters
    ----------
    scan_backend : str, default='eager'
        Passed through to ``NewtonConfig`` (``eager`` / ``triton`` / ``fused`` /
        ``auto`` / ``context_parallel``).

    Returns
    -------
    NewtonConfig
        ``max_iters=3`` (App. A), ``residual_atol=None``, ``residual_fail=None``,
        ``picard_adapt=False``.
    """
    return NewtonConfig(
        max_iters=LIBRARY_NEWTON_ITERS,
        scan_backend=scan_backend,
        residual_atol=None,
        residual_fail=None,
        picard_adapt=False,
    )


def _validate_config(config: NewtonConfig) -> None:
    if config.scan_backend not in ("auto", "eager", "triton", "fused", "context_parallel"):
        raise ValueError(f"unknown scan backend {config.scan_backend!r}")
    if config.jacobian not in ("auto", "analytic", "autograd"):
        raise ValueError(f"unknown jacobian {config.jacobian!r}")
    if config.coords not in ("native", "log"):
        raise ValueError(f"unknown newton coords {config.coords!r}")
    if config.chunk_len is not None and int(config.chunk_len) < 1:
        raise ValueError(f"chunk_len must be >= 1, got {config.chunk_len!r}")
    if config.max_iters is not None and int(config.max_iters) < 0:
        raise ValueError(f"max_iters must be >= 0 or None, got {config.max_iters!r}")
    if config.picard_iters is not None and int(config.picard_iters) < 0:
        raise ValueError(f"picard_iters must be >= 0, got {config.picard_iters!r}")
    if config.picard_retry_atol is not None and float(config.picard_retry_atol) < 0:
        raise ValueError(
            f"picard_retry_atol must be >= 0 or None, got {config.picard_retry_atol!r}"
        )
    if config.residual_fail is not None and float(config.residual_fail) < 0:
        raise ValueError(f"residual_fail must be >= 0 or None, got {config.residual_fail!r}")
    if config.newton_iters_by_t is not None:
        if config.max_iters is not None:
            raise ValueError("newton_iters_by_t requires max_iters=None (auto K*(T))")
        if not config.newton_iters_by_t:
            raise ValueError("newton_iters_by_t must be a non-empty mapping")
        for k, v in config.newton_iters_by_t.items():
            if int(k) < 1 or int(v) < 0:
                raise ValueError(f"newton_iters_by_t bad entry T={k!r} → K={v!r}")
    if config.scan_tile not in ("assoc", "seq", "thomas", "thomas2", "thomas4"):
        raise ValueError(f"unknown scan_tile {config.scan_tile!r}")
    if config.fused_time_loop and config.chunk_len is not None:
        raise ValueError("fused_time_loop and chunk_len both set; pick one windowed path")
    if config.fused_time_loop and config.coords == "log":
        raise ValueError("fused_time_loop is native coords only")
    if config.fused_window_len is not None:
        w = int(config.fused_window_len)
        if w not in FUSED_WINDOW_LENS:
            raise ValueError(f"fused_window_len must be one of {FUSED_WINDOW_LENS}, got {w}")
        if not config.fused_time_loop:
            raise ValueError("fused_window_len requires fused_time_loop=True")
    if config.fused_early_exit:
        if config.residual_atol is None:
            raise ValueError("fused_early_exit requires residual_atol (got None)")
        if config.fused_time_loop:
            raise ValueError("fused_early_exit cannot combine with fused_time_loop")
