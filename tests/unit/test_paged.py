"""Paged slot pool: allocate/free, gather/scatter, leftover wipe."""

from __future__ import annotations

import pytest
import torch

from pararnn import NewtonConfig, PagedStatePool, ParaGRU, ParaRNN, paged_apply


def _gru(d: int = 6) -> ParaRNN:
    return ParaRNN(
        ParaGRU(d, d),
        config=NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None),
    )


def test_allocate_and_free_reuse() -> None:
    pool = PagedStatePool(_gru(), capacity=4)
    a = pool.allocate(3)
    assert pool.allocator.n_used == 3
    assert pool.allocator.n_free == 1
    pool.free(a[:2])
    assert pool.allocator.n_used == 1
    b = pool.allocate(2)
    assert pool.allocator.n_used == 3
    assert len(set(b.tolist()) & set(a[:2].tolist())) == 2


def test_oom_when_pool_exhausted() -> None:
    pool = PagedStatePool(_gru(), capacity=2)
    pool.allocate(2)
    with pytest.raises(RuntimeError, match="paged cache OOM"):
        pool.allocate(1)


def test_free_unknown_slot_raises() -> None:
    pool = PagedStatePool(_gru(), capacity=2)
    with pytest.raises(KeyError, match="not allocated"):
        pool.free([0])


def test_freed_slot_is_zero() -> None:
    torch.manual_seed(0)
    model = _gru(4)
    pool = PagedStatePool(model, capacity=3)
    ids = pool.allocate(1)
    x = torch.randn(1, 5, 4)
    paged_apply(pool, ids, x, solver="sequential")
    st = pool.gather(ids)
    assert st.abs().sum() > 0
    pool.free(ids)
    # slot is zero in the buffer; gather after re-allocate
    ids2 = pool.allocate(1)
    got = pool.gather(ids2)
    assert torch.equal(got, torch.zeros_like(got))


def test_offload_frees_gpu_slot() -> None:
    torch.manual_seed(0)
    model = _gru(4)
    pool = PagedStatePool(model, capacity=2, host_capacity=2)
    ids = pool.allocate(2)
    x = torch.randn(2, 3, 4)
    paged_apply(pool, ids, x, solver="sequential")
    host = pool.offload(ids[:1])
    assert pool.allocator.n_used == 1
    assert pool.host_allocator.n_used == 1
    reused = pool.allocate(1)
    assert pool.allocator.n_used == 2
    assert int(reused.item()) in set(ids.tolist())
    got = pool.gather(reused)
    assert torch.equal(got, torch.zeros_like(got))
    pool.free_host(host)


def test_offload_reload_roundtrip() -> None:
    torch.manual_seed(1)
    model = _gru(4)
    pool = PagedStatePool(model, capacity=2)
    ids = pool.allocate(1)
    x = torch.randn(1, 5, 4)
    paged_apply(pool, ids, x, solver="sequential")
    saved = pool.gather(ids).clone()
    host = pool.offload(ids)
    assert pool.allocator.n_used == 0
    back = pool.reload(host)
    assert pool.host_allocator.n_used == 0
    torch.testing.assert_close(pool.gather(back), saved)


def test_host_oom() -> None:
    pool = PagedStatePool(_gru(), capacity=2, host_capacity=1)
    ids = pool.allocate(2)
    pool.offload(ids[:1])
    with pytest.raises(RuntimeError, match="paged host OOM"):
        pool.offload(ids[1:])


def test_reload_gpu_oom_keeps_host() -> None:
    torch.manual_seed(2)
    model = _gru(4)
    pool = PagedStatePool(model, capacity=1, host_capacity=1)
    ids = pool.allocate(1)
    paged_apply(pool, ids, torch.randn(1, 3, 4), solver="sequential")
    host = pool.offload(ids)
    pool.allocate(1)
    with pytest.raises(RuntimeError, match="paged cache OOM"):
        pool.reload(host)
    assert pool.host_allocator.n_used == 1


def test_offload_unknown_slot_raises() -> None:
    pool = PagedStatePool(_gru(), capacity=2)
    with pytest.raises(KeyError, match="not allocated"):
        pool.offload([0])


def test_duplicate_offload_raises() -> None:
    pool = PagedStatePool(_gru(), capacity=2)
    ids = pool.allocate(1)
    with pytest.raises(ValueError, match="duplicate"):
        pool.offload(torch.cat((ids, ids)))
