"""CPU helpers for time sharding (no process group)."""

from __future__ import annotations

import pytest

from pararnn.solvers.seq_parallel import time_shard_bounds


def test_time_shard_even() -> None:
    assert time_shard_bounds(16, 0, 2) == (0, 8)
    assert time_shard_bounds(16, 1, 2) == (8, 16)


def test_time_shard_remainder_on_last() -> None:
    assert time_shard_bounds(17, 0, 2) == (0, 8)
    assert time_shard_bounds(17, 1, 2) == (8, 17)


def test_time_shard_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="outside"):
        time_shard_bounds(8, 2, 2)
