"""T=1 decode-step Triton vs eager cell.step.

Prefill T=8 builds a carry; the bench is the next token, with ``out=`` and
a CUDA graph of GEMM → step.

Usage:
    python decode_step.py
"""

import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from pararnn import ParaGRU, ParaLSTM, ParaSLSTM, decode_step, decode_wx, sequential_apply

BATCH, DIM, PREFILL = 8, 256, 8
WARMUP, RUNS = 10, 50

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"decode_step: device={device} B={BATCH} d={DIM}")


def _sync() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _minmax_ms(fn) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    _sync()
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples), sum(samples) / len(samples)


def _carry(cell) -> torch.Tensor:
    return sequential_apply(cell, torch.randn(BATCH, PREFILL, DIM, device=device))[:, -1]


def _bench(name: str, cell, state: torch.Tensor, x: torch.Tensor) -> None:
    wx = cell.W_x(x)
    state, x = state.contiguous(), x.contiguous()
    wx_buf = torch.empty(x.shape[0], cell.W_x.out_features, device=device, dtype=x.dtype)
    out_buf = torch.empty_like(state)

    with torch.no_grad():
        got = decode_step(cell, state, wx=wx, out=out_buf)
        ref = cell.step(state, x, wx=wx)
        err = float((got - ref).abs().amax())
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

        e_min, e_mean = _minmax_ms(lambda: cell.step(state, x, wx=wx))
        t_min, t_mean = _minmax_ms(lambda: decode_step(cell, state, wx=wx, out=out_buf))

        g_min = g_mean = float("nan")
        if device.type == "cuda":

            def _graph_body() -> None:
                decode_wx(cell, x, out=wx_buf)
                decode_step(cell, state, wx=wx_buf, out=out_buf)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _graph_body()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _graph_body()
            g_min, g_mean = _minmax_ms(graph.replay)

    print(
        f"{name} max_abs={err:.3e} | "
        f"eager {e_min:.3f}/{e_mean:.3f} ms | "
        f"triton {t_min:.3f}/{t_mean:.3f} ms | "
        f"graph {g_min:.3f}/{g_mean:.3f} ms"
    )


gru = ParaGRU(DIM, DIM, device=device)
_bench("ParaGRU", gru, _carry(gru), torch.randn(BATCH, DIM, device=device))

lstm = ParaLSTM(DIM, DIM, device=device)
_bench("ParaLSTM", lstm, _carry(lstm), torch.randn(BATCH, DIM, device=device))

slstm = ParaSLSTM(DIM, DIM, mix="diag", device=device)
_bench("ParaSLSTM", slstm, _carry(slstm), torch.randn(BATCH, DIM, device=device))
