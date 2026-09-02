"""Shared pytest fixtures for the ParaRNN test suite."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return torch.device("cuda")
