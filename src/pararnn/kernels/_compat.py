"""Triton pin, capability probes, and lazy fused preflight.

ParaRNN pins ``triton>=3.6,<3.7`` (Torch 2.11 cu128 dialect). Loads already
go through :mod:`pararnn.kernels.precision` (``load_acc`` / ``store_acc``) —
masked ``tl.load``, no ``tl.make_block_ptr``.

``require_fused_triton`` runs once per process: pin / probes, then a tiny
JIT smoke (masked load + ``associative_scan``) so a broken dialect fails with
a clear ``[ParaRNN]`` RuntimeError on the first fused launch.
"""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# Matches pyproject.toml Linux pin (exclusive upper bound).
_TRITON_MIN = (3, 6, 0)
_TRITON_MAX_EXCLUSIVE = (3, 7, 0)


def _parse_version(raw: str) -> tuple[int, int, int]:
    core = raw.split("+")[0].split("-")[0]
    parts = core.split(".")
    nums: list[int] = []
    for p in parts[:3]:
        digits = "".join(c for c in p if c.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _probe_triton() -> tuple[str, tuple[int, int, int], bool, bool, bool]:
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return "missing", (0, 0, 0), False, False, False

    ver_s = getattr(triton, "__version__", "0.0.0")
    ver = _parse_version(ver_s)
    has_scan = hasattr(tl, "associative_scan")
    has_static = hasattr(tl, "static_range")
    has_tanh = False
    try:
        from triton.language.extra.cuda.libdevice import tanh as _tanh  # noqa: F401

        has_tanh = True
    except Exception:
        has_tanh = False
    return ver_s, ver, has_scan, has_static, has_tanh


(
    TRITON_VERSION_STR,
    TRITON_VERSION,
    HAS_ASSOCIATIVE_SCAN,
    HAS_STATIC_RANGE,
    HAS_LIBDEVICE_TANH,
) = _probe_triton()

TRITON_IN_PIN = (
    TRITON_VERSION_STR != "missing" and _TRITON_MIN <= TRITON_VERSION < _TRITON_MAX_EXCLUSIVE
)


def _env_versions() -> str:
    try:
        import torch

        torch_v = torch.__version__
    except Exception:
        torch_v = "?"
    return f"torch=={torch_v}, triton=={TRITON_VERSION_STR}"


@lru_cache(maxsize=1)
def check_triton_pin(*, hard: bool = False) -> None:
    """Warn (or raise) when Triton is missing or outside the Linux 3.6.x pin.

    Soft by default so CPU-only imports stay quiet until a fused launch.
    """
    if TRITON_VERSION_STR == "missing":
        msg = (
            "[ParaRNN] triton is not installed; fused CUDA kernels need Linux + "
            "triton>=3.6,<3.7 (same index as torch). "
            "Use NewtonConfig(scan_backend='eager') on CPU, or see INSTALL.md."
        )
        if hard:
            raise RuntimeError(msg)
        log.debug(msg)
        return
    if TRITON_IN_PIN:
        return
    msg = (
        f"[ParaRNN] triton {TRITON_VERSION_STR} is outside the supported pin "
        f">={_TRITON_MIN[0]}.{_TRITON_MIN[1]},<{_TRITON_MAX_EXCLUSIVE[0]}."
        f"{_TRITON_MAX_EXCLUSIVE[1]} (Torch 2.11 cu128 dialect). "
        f"Detected {_env_versions()}. "
        "Reinstall Triton from the same wheel index as torch, or set "
        "NewtonConfig(scan_backend='eager'). See INSTALL.md / FAQs.md."
    )
    if hard:
        raise RuntimeError(msg)
    log.warning(msg)


@lru_cache(maxsize=1)
def _preflight_kernels():
    """Lazy-define JIT helpers once (import triton only when needed)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _preflight_add(a, b):
        return a + b

    @triton.jit
    def _preflight_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.associative_scan(x, 0, _preflight_add)
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    return _preflight_kernel


def _run_preflight_smoke() -> None:
    """Compile+run a tiny masked-load + associative_scan kernel on CUDA."""
    import torch

    kernel = _preflight_kernels()
    n = 8
    device = torch.device("cuda")
    x = torch.arange(n, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    kernel[(1,)](x, y, n, BLOCK=16)
    torch.cuda.synchronize(device)
    expect = torch.cumsum(x, dim=0)
    if not torch.allclose(y, expect, atol=1e-5, rtol=0.0):
        raise RuntimeError(f"preflight scan mismatch: got {y.tolist()} expect {expect.tolist()}")


@lru_cache(maxsize=1)
def check_triton_environment() -> None:
    """Pin + one-shot CUDA JIT smoke (lazy; not on import).

    Safe to call explicitly before training. Also invoked from
    :func:`require_fused_triton` on the first fused / Triton scan launch.
    """
    check_triton_pin(hard=True)
    if not HAS_ASSOCIATIVE_SCAN:
        raise RuntimeError(
            f"[ParaRNN] triton {TRITON_VERSION_STR} lacks tl.associative_scan; "
            f"need triton>=3.6,<3.7. Detected {_env_versions()}."
        )
    if not HAS_LIBDEVICE_TANH:
        raise RuntimeError(
            "[ParaRNN] triton.language.extra.cuda.libdevice.tanh unavailable; "
            f"fused GRU/LSTM/sLSTM need NVIDIA CUDA Triton. Detected {_env_versions()}."
        )

    import torch

    if not torch.cuda.is_available():
        return
    if torch.compiler.is_compiling():
        return

    try:
        _run_preflight_smoke()
    except Exception as exc:
        raise RuntimeError(
            "[ParaRNN] Triton/CUDA environment failed a fused preflight. "
            f"Detected {_env_versions()}. "
            "Install triton from the same index as torch (pin >=3.6,<3.7), "
            "or use NewtonConfig(scan_backend='eager'). See INSTALL.md / FAQs.md. "
            f"Original error: {exc}"
        ) from None

    log.info(
        "triton_preflight_ok",
        extra={"torch_triton": _env_versions(), "device": str(torch.cuda.current_device())},
    )


def require_fused_triton() -> None:
    """Hard gate before launching a fused / Triton scan kernel."""
    check_triton_environment()
