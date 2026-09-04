"""T=1 decode-step Triton vs eager cell.step.

    uv run python examples/decode_step.py

``.eval()`` already unrolls ``cell.step``. This kernel is that step at T=1:
one SRAM trip for gates + mix. ``W_x`` stays a GEMM (``decode_wx`` into a
buffer). Prefill T=8 builds a carry; the bench is the next token, with
``out=`` and a CUDA graph of GEMM → step. B=8, d=256. Local smoke: no MLflow.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pararnn import ParaGRU, ParaLSTM, ParaSLSTM, decode_step, decode_wx, sequential_apply

log = logging.getLogger("decode_step")

_BATCH = 8
_DIM = 256
_PREFILL = 8  # T>1 eager unroll; decode starts from that carry
_WARMUP = 10
_RUNS = 50


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _minmax_ms(fn, device: torch.device) -> tuple[float, float]:
    for _ in range(_WARMUP):
        fn()
    _sync(device)
    samples: list[float] = []
    for _ in range(_RUNS):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), float(sum(samples) / len(samples))


def _bench_cell(
    name: str, cell, state: torch.Tensor, x: torch.Tensor, device: torch.device
) -> None:
    wx = cell.W_x(x)
    state = state.contiguous()
    x = x.contiguous()

    def _eager() -> None:
        cell.step(state, x, wx=wx)

    def _triton() -> None:
        decode_step(cell, state, wx=wx, out=out_buf)

    wx_buf = torch.empty(x.shape[0], cell.W_x.out_features, device=device, dtype=x.dtype)
    out_buf = torch.empty_like(state)

    def _graph_body() -> None:
        decode_wx(cell, x, out=wx_buf)
        decode_step(cell, state, wx=wx_buf, out=out_buf)

    with torch.no_grad():
        got = decode_step(cell, state, wx=wx, out=out_buf)
        ref = cell.step(state, x, wx=wx)
        err = float((got - ref).abs().amax())
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
        e_min, e_mean = _minmax_ms(_eager, device)
        t_min, t_mean = _minmax_ms(_triton, device)
        g_min = g_mean = float("nan")
        if device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _graph_body()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _graph_body()

            def _replay() -> None:
                graph.replay()

            g_min, g_mean = _minmax_ms(_replay, device)
    log.info(
        "%s B=%s d=%s max_abs=%.3e eager_min_ms=%.3f eager_mean_ms=%.3f "
        "triton_min_ms=%.3f triton_mean_ms=%.3f graph_min_ms=%.3f graph_mean_ms=%.3f",
        name,
        _BATCH,
        _DIM,
        err,
        e_min,
        e_mean,
        t_min,
        t_mean,
        g_min,
        g_mean,
    )


def _carry(cell, device: torch.device) -> torch.Tensor:
    x_pre = torch.randn(_BATCH, _PREFILL, _DIM, device=device)
    return sequential_apply(cell, x_pre)[:, -1]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    torch.manual_seed(0)
    log.info("gpu=%s B=%s d=%s", gpu, _BATCH, _DIM)

    gru = ParaGRU(_DIM, _DIM, device=device)
    x_dec = torch.randn(_BATCH, _DIM, device=device)
    _bench_cell("ParaGRU", gru, _carry(gru, device), x_dec, device)

    lstm = ParaLSTM(_DIM, _DIM, device=device)
    _bench_cell(
        "ParaLSTM",
        lstm,
        _carry(lstm, device),
        torch.randn(_BATCH, _DIM, device=device),
        device,
    )

    slstm = ParaSLSTM(_DIM, _DIM, mix="diag", device=device)
    _bench_cell(
        "ParaSLSTM",
        slstm,
        _carry(slstm, device),
        torch.randn(_BATCH, _DIM, device=device),
        device,
    )


if __name__ == "__main__":
    main()
