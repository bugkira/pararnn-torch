"""T=1 Triton decode_step vs eager cell.step on CUDA."""

from __future__ import annotations

import pytest
import torch

from pararnn import (
    ParaGRU,
    ParaLSTM,
    ParaRNN,
    ParaSLSTM,
    can_decode_step,
    decode_step,
    sequential_apply,
)
from pararnn.kernels.precision import is_fused_dtype_supported

_ATOL = 1e-5
_RTOL = 1e-5
_ATOL_F16 = 2e-3
_RTOL_F16 = 2e-3


def _force_eager_step(cell, state, x, *, wx=None):
    """Bypass decode_step so the reference stays PyTorch ``cell.step``."""
    if wx is None:
        return cell.step(state, x)
    return cell.step(state, x, wx=wx)


def _prefill_state(cell, batch: int, d_in: int, device: torch.device, *, t: int = 8):
    """Last state of a T>1 unroll (eager). Serving decode starts from a prefill carry."""
    x = torch.randn(batch, t, d_in, device=device, dtype=cell.W_x.weight.dtype)
    return sequential_apply(cell, x)[:, -1]


@pytest.mark.parametrize("kind", ["gru", "lstm", "slstm"])
@torch.no_grad()
def test_decode_step_matches_eager(cuda_device: torch.device, kind: str) -> None:
    torch.manual_seed(11)
    d_in, d_h, b = 16, 32, 4
    kw = {"d_in": d_in, "d_h": d_h, "device": cuda_device}
    if kind == "gru":
        cell = ParaGRU(**kw)
    elif kind == "lstm":
        cell = ParaLSTM(**kw)
    else:
        cell = ParaSLSTM(mix="diag", **kw)
    state = _prefill_state(cell, b, d_in, cuda_device)
    x = torch.randn(b, d_in, device=cuda_device)
    assert can_decode_step(cell, x)
    got = decode_step(cell, state, x)
    ref = _force_eager_step(cell, state, x)
    torch.testing.assert_close(got, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("kind", ["gru", "lstm", "slstm"])
@torch.no_grad()
def test_decode_step_precomputed_wx(cuda_device: torch.device, kind: str) -> None:
    torch.manual_seed(12)
    d_in, d_h, b = 8, 17, 3  # d_h not a multiple of BLOCK_D=128
    kw = {"d_in": d_in, "d_h": d_h, "device": cuda_device}
    if kind == "gru":
        cell = ParaGRU(**kw)
    elif kind == "lstm":
        cell = ParaLSTM(**kw)
    else:
        cell = ParaSLSTM(mix="diag", **kw)
    state = _prefill_state(cell, b, d_in, cuda_device)
    x = torch.randn(b, d_in, device=cuda_device)
    wx = cell.W_x(x)
    got = decode_step(cell, state, wx=wx)
    ref = _force_eager_step(cell, state, x, wx=wx)
    torch.testing.assert_close(got, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("kind", ["gru", "lstm", "slstm"])
@torch.no_grad()
def test_decode_from_prefill_d256(cuda_device: torch.device, kind: str) -> None:
    """Serving shape: B=8 d_h=256, same as examples/decode_step.py."""
    torch.manual_seed(18)
    d, b = 256, 8
    kw = {"d_in": d, "d_h": d, "device": cuda_device}
    if kind == "gru":
        cell = ParaGRU(**kw)
    elif kind == "lstm":
        cell = ParaLSTM(**kw)
    else:
        cell = ParaSLSTM(mix="diag", **kw)
    state = _prefill_state(cell, b, d, cuda_device)
    x = torch.randn(b, d, device=cuda_device)
    got = decode_step(cell, state, x)
    ref = _force_eager_step(cell, state, x)
    torch.testing.assert_close(got, ref, atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_sequential_apply_t1_uses_triton(cuda_device: torch.device) -> None:
    torch.manual_seed(13)
    cell = ParaGRU(d_in=8, d_h=16, device=cuda_device)
    x = torch.randn(5, 1, 8, device=cuda_device)
    h0 = torch.randn(5, 16, device=cuda_device)
    got = sequential_apply(cell, x, h0)
    ref = _force_eager_step(cell, h0, x[:, 0]).unsqueeze(1)
    torch.testing.assert_close(got, ref, atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_pararnn_eval_t1(cuda_device: torch.device) -> None:
    torch.manual_seed(14)
    cell = ParaSLSTM(d_in=8, d_h=12, mix="diag", device=cuda_device)
    model = ParaRNN(cell).to(cuda_device)
    model.eval()
    h0 = sequential_apply(cell, torch.randn(3, 4, 8, device=cuda_device))[:, -1]
    x = torch.randn(3, 1, 8, device=cuda_device)
    y = model(x, h0)
    ref = _force_eager_step(cell, h0, x[:, 0])[..., 3, :]
    torch.testing.assert_close(y[:, 0], ref, atol=_ATOL, rtol=_RTOL)


def test_decode_step_with_grad_stays_eager(cuda_device: torch.device) -> None:
    torch.manual_seed(15)
    cell = ParaGRU(d_in=6, d_h=8, device=cuda_device)
    h = torch.randn(2, 8, device=cuda_device, requires_grad=True)
    x = torch.randn(2, 6, device=cuda_device, requires_grad=True)
    y = decode_step(cell, h, x)
    assert y.requires_grad
    y.sum().backward()
    assert h.grad is not None
    assert x.grad is not None


@torch.no_grad()
def test_fp16_decode_step(cuda_device: torch.device) -> None:
    torch.manual_seed(16)
    cell = ParaGRU(d_in=8, d_h=16, device=cuda_device, dtype=torch.float16)
    h = torch.randn(4, 16, device=cuda_device, dtype=torch.float16)
    x = torch.randn(4, 8, device=cuda_device, dtype=torch.float16)
    got = decode_step(cell, h, x)
    ref = _force_eager_step(cell, h, x)
    torch.testing.assert_close(got, ref, atol=_ATOL_F16, rtol=_RTOL_F16)


@torch.no_grad()
def test_bf16_decode_step(cuda_device: torch.device) -> None:
    if not is_fused_dtype_supported(torch.bfloat16, cuda_device):
        pytest.skip("bf16 fused needs SM >= 8.0")
    torch.manual_seed(17)
    cell = ParaLSTM(d_in=8, d_h=16, device=cuda_device, dtype=torch.bfloat16)
    s = torch.randn(4, 2, 16, device=cuda_device, dtype=torch.bfloat16)
    x = torch.randn(4, 8, device=cuda_device, dtype=torch.bfloat16)
    got = decode_step(cell, s, x)
    ref = _force_eager_step(cell, s, x)
    torch.testing.assert_close(got, ref, atol=_ATOL_F16, rtol=_RTOL_F16)


@torch.no_grad()
def test_decode_step_out_same_storage(cuda_device: torch.device) -> None:
    torch.manual_seed(19)
    cell = ParaGRU(d_in=8, d_h=16, device=cuda_device)
    state = _prefill_state(cell, 4, 8, cuda_device)
    x = torch.randn(4, 8, device=cuda_device)
    out = torch.empty_like(state)
    got = decode_step(cell, state, x, out=out)
    assert got.data_ptr() == out.data_ptr()
    ref = _force_eager_step(cell, state, x)
    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_decode_block_table_matches_dense(cuda_device: torch.device) -> None:
    torch.manual_seed(20)
    d, b, cap = 16, 3, 8
    cell = ParaSLSTM(d_in=d, d_h=d, mix="diag", device=cuda_device)
    dense = _prefill_state(cell, b, d, cuda_device)
    pool = torch.zeros(cap, 4, d, device=cuda_device)
    # Non-contiguous slots: 1, 4, 6.
    ids = torch.tensor([1, 4, 6], device=cuda_device, dtype=torch.int32)
    pool[ids.long()] = dense
    x = torch.randn(b, d, device=cuda_device)
    decode_step(cell, pool, x, block_table=ids)
    ref = _force_eager_step(cell, dense, x)
    torch.testing.assert_close(pool.index_select(0, ids.long()), ref, atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_paged_apply_t1_block_table(cuda_device: torch.device) -> None:
    from pararnn import NewtonConfig, PagedStatePool, ParaRNN, paged_apply

    torch.manual_seed(21)
    d, b, t = 16, 3, 6
    cfg = NewtonConfig(max_iters=3, residual_atol=None, residual_fail=None)
    model = ParaRNN(ParaGRU(d, d, device=cuda_device), config=cfg).to(cuda_device)
    cell = model.layers[0]
    x_pre = torch.randn(b, t, d, device=cuda_device)
    x_dec = torch.randn(b, 1, d, device=cuda_device)
    ref = sequential_apply(cell, torch.cat((x_pre, x_dec), dim=1))
    pool = PagedStatePool(model, capacity=8)
    ids = pool.allocate(b)
    paged_apply(pool, ids, x_pre, solver="sequential")
    y_dec = paged_apply(pool, ids, x_dec, solver="sequential")
    torch.testing.assert_close(y_dec[:, 0], ref[:, t], atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(pool.gather(ids), ref[:, -1], atol=_ATOL, rtol=_RTOL)


@torch.no_grad()
def test_decode_cuda_graph_wx_then_step(cuda_device: torch.device) -> None:
    from pararnn import decode_wx

    torch.manual_seed(22)
    cell = ParaGRU(d_in=16, d_h=32, device=cuda_device)
    state = _prefill_state(cell, 4, 16, cuda_device)
    x = torch.randn(4, 16, device=cuda_device)
    s_work = state.clone()
    x_work = x.clone()
    wx_buf = torch.empty(4, 3 * 32, device=cuda_device)
    out_buf = torch.empty_like(s_work)
    decode_wx(cell, x_work, out=wx_buf)
    decode_step(cell, s_work, wx=wx_buf, out=out_buf)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode_wx(cell, x_work, out=wx_buf)
            decode_step(cell, s_work, wx=wx_buf, out=out_buf)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_wx(cell, x_work, out=wx_buf)
        decode_step(cell, s_work, wx=wx_buf, out=out_buf)
    x2 = torch.randn(4, 16, device=cuda_device)
    s2 = _prefill_state(cell, 4, 16, cuda_device)
    x_work.copy_(x2)
    s_work.copy_(s2)
    g.replay()
    ref = _force_eager_step(cell, s2, x2)
    torch.testing.assert_close(out_buf, ref, atol=_ATOL, rtol=_RTOL)


@pytest.mark.filterwarnings("ignore:mix='head' is Beck-style dense R:UserWarning")
@torch.no_grad()
def test_head_mix_stays_eager(cuda_device: torch.device) -> None:
    cell = ParaSLSTM(d_in=8, d_h=16, mix="head", n_heads=2, device=cuda_device)
    x = torch.randn(2, 8, device=cuda_device)
    assert can_decode_step(cell, x) is False
    state = torch.randn(2, 4, 16, device=cuda_device)
    torch.testing.assert_close(decode_step(cell, state, x), cell.step(state, x))
