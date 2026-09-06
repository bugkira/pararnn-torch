"""Triton pin / capability probes / lazy fused preflight."""

from __future__ import annotations

import pytest
import torch

from pararnn.kernels._compat import (
    HAS_ASSOCIATIVE_SCAN,
    HAS_LIBDEVICE_TANH,
    TRITON_IN_PIN,
    TRITON_VERSION_STR,
    check_triton_environment,
    check_triton_pin,
    require_fused_triton,
)


def test_triton_pin_matches_pyproject_on_linux():
    pytest.importorskip("triton")
    assert TRITON_VERSION_STR != "missing"
    assert TRITON_IN_PIN, f"expected 3.6.x pin, got {TRITON_VERSION_STR}"
    assert HAS_ASSOCIATIVE_SCAN
    check_triton_pin(hard=True)
    assert HAS_LIBDEVICE_TANH  # NVIDIA CUDA Triton on this lab box


@pytest.mark.cuda
def test_triton_environment_preflight_smoke():
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for JIT smoke")
    check_triton_environment.cache_clear()
    check_triton_environment()
    # Second call is cached (no second JIT).
    require_fused_triton()
