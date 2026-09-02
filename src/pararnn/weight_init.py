"""Initialization from Danieli et al. 2025 Appendix C.1 (language-model recipe)."""

from __future__ import annotations

import math

from torch import Tensor, nn


def xavier_gaussian_vec_(tensor: Tensor) -> None:
    """Xavier-normal as the diagonal of a square matrix of size ``numel``.

    Paper C.1: Xavier Gaussian on ``a_*`` / ``c_*`` because small values
    improved Newton stability.
    """
    n = max(tensor.numel(), 1)
    std = math.sqrt(2.0 / (n + n))
    nn.init.normal_(tensor, mean=0.0, std=std)


def kaiming_uniform_linear_(weight: Tensor) -> None:
    """Kaiming uniform on ``B_*`` (paper C.1, He et al. 2015)."""
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
