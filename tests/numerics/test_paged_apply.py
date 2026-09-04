"""Paged pool vs dense sequential / Newton; mixed prefill + decode."""

from __future__ import annotations

import torch

from pararnn import (
    NewtonConfig,
    PagedStatePool,
    ParaGRU,
    ParaRNN,
    ParaSLSTM,
    paged_apply,
    sequential_apply,
)
from pararnn.solvers.newton import newton_apply

_ATOL = 1e-4
_RTOL = 1e-4


def test_gru_prefill_then_decode_matches_dense() -> None:
    torch.manual_seed(1)
    d, b, t = 8, 3, 6
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), config=cfg)
    x_pre = torch.randn(b, t, d)
    x_dec = torch.randn(b, 1, d)
    ref = sequential_apply(model.layers[0], torch.cat((x_pre, x_dec), dim=1))
    pool = PagedStatePool(model, capacity=8)
    ids = pool.allocate(b)
    y_pre = paged_apply(pool, ids, x_pre, solver="sequential")
    y_dec = paged_apply(pool, ids, x_dec, solver="sequential")
    torch.testing.assert_close(y_pre, ref[:, :t], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(y_dec[:, 0], ref[:, t], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(pool.gather(ids), ref[:, -1], atol=_ATOL, rtol=_RTOL)


def test_slstm_slots_roundtrip() -> None:
    torch.manual_seed(2)
    d, b, t = 8, 2, 5
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaSLSTM(d, d, mix="diag"), config=cfg)
    x = torch.randn(b, t, d)
    ref = sequential_apply(model.layers[0], x)
    pool = PagedStatePool(model, capacity=4)
    ids = pool.allocate(b)
    assert pool.buffers[0].shape == (4, 4, d)
    paged_apply(pool, ids, x, solver="sequential")
    torch.testing.assert_close(pool.gather(ids), ref[:, -1], atol=_ATOL, rtol=_RTOL)


def test_mixed_packed_prefill_and_decode() -> None:
    """One packed batch: lengths 4, 1, 3 — prefill / decode / prefill."""
    torch.manual_seed(3)
    d = 8
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), config=cfg)
    x0 = torch.randn(1, 4, d)
    x1 = torch.randn(1, 1, d)
    x2 = torch.randn(1, 3, d)
    h0_1 = sequential_apply(model.layers[0], torch.randn(1, 2, d))[:, -1]
    ref0 = sequential_apply(model.layers[0], x0)
    ref1 = sequential_apply(model.layers[0], x1, h0_1)
    ref2 = sequential_apply(model.layers[0], x2)

    pool = PagedStatePool(model, capacity=8)
    ids = pool.allocate(3)
    pool.scatter(ids[1:2], h0_1)
    packed = torch.cat((x0, x1, x2), dim=1)
    cs = torch.tensor([0, 4, 5, 8])
    y = paged_apply(pool, ids, packed, cu_seqlens=cs, solver="sequential")
    torch.testing.assert_close(y[:, :4], ref0, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(y[:, 4:5], ref1, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(y[:, 5:], ref2, atol=_ATOL, rtol=_RTOL)
    got = pool.gather(ids)
    torch.testing.assert_close(got[0], ref0[0, -1], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(got[1], ref1[0, -1], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(got[2], ref2[0, -1], atol=_ATOL, rtol=_RTOL)


def test_newton_paged_matches_sequential_pool() -> None:
    torch.manual_seed(4)
    d, b, t = 8, 2, 7
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), config=cfg)
    x = torch.randn(b, t, d)
    pool_s = PagedStatePool(model, capacity=4)
    pool_n = PagedStatePool(model, capacity=4)
    ids_s = pool_s.allocate(b)
    ids_n = pool_n.allocate(b)
    y_s = paged_apply(pool_s, ids_s, x, solver="sequential")
    y_n = paged_apply(pool_n, ids_n, x, solver="newton")
    torch.testing.assert_close(y_n, y_s, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(pool_n.gather(ids_n), pool_s.gather(ids_s), atol=2e-4, rtol=2e-4)
    dense = newton_apply(model.layers[0], x, cfg)
    torch.testing.assert_close(y_n, dense, atol=2e-4, rtol=2e-4)


def test_two_layer_pool() -> None:
    torch.manual_seed(5)
    d, b, t = 6, 2, 4
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d), num_layers=2, config=cfg)
    x = torch.randn(b, t, d)
    h0 = sequential_apply(model.layers[0], x)
    h1 = sequential_apply(model.layers[1], h0)
    pool = PagedStatePool(model, capacity=4)
    ids = pool.allocate(b)
    y = paged_apply(pool, ids, x, solver="sequential")
    torch.testing.assert_close(y, h1, atol=_ATOL, rtol=_RTOL)
    g0, g1 = pool.gather(ids)
    torch.testing.assert_close(g0, h0[:, -1], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(g1, h1[:, -1], atol=_ATOL, rtol=_RTOL)
