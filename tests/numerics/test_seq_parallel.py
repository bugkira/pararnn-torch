"""Two-rank diag scan matches Blelloch; CUDA streams are virtual ranks."""

from __future__ import annotations

import pytest
import torch

from pararnn.solvers.scan import scan_diag
from pararnn.solvers.seq_parallel import (
    scan_diag_two_ranks,
    sequential_prefix_two_ranks,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rand_system(batch: int, time: int, dim: int, seed: int):
    torch.manual_seed(seed)
    jac = torch.randn(batch, time, dim, device=device) * 0.3
    residual = torch.randn(batch, time, dim, device=device)
    return jac, residual


@pytest.mark.parametrize("time", [1, 2, 3, 8, 17, 64])
def test_two_rank_matches_scan_diag(time: int):
    jac, residual = _rand_system(4, time, 5, seed=20 + time)
    ref = scan_diag(jac, residual)
    got = scan_diag_two_ranks(jac, residual)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("time", [1, 2, 17, 64])
def test_sequential_prefix_matches_scan_diag(time: int):
    jac, residual = _rand_system(3, time, 4, seed=40 + time)
    ref = scan_diag(jac, residual)
    got = sequential_prefix_two_ranks(jac, residual)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_two_rank_rejects_rank_mismatch():
    jac = torch.randn(2, 8, 3, device=device)
    residual = torch.randn(2, 8, 4, device=device)
    with pytest.raises(ValueError, match="!="):
        scan_diag_two_ranks(jac, residual)


@pytest.mark.cuda
def test_two_rank_cuda_streams_match_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(7)
    jac = torch.randn(2, 32, 8, device=cuda_device) * 0.3
    residual = torch.randn(2, 32, 8, device=cuda_device)
    ref = scan_diag(jac, residual)
    s0 = torch.cuda.Stream()
    s1 = torch.cuda.Stream()
    got = scan_diag_two_ranks(jac, residual, streams=(s0, s1))
    torch.cuda.synchronize()
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
