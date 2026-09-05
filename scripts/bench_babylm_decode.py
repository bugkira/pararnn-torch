"""Greedy decode tok/s on the trained BabyLM 6×384 stack.

    uv run python scripts/bench_babylm_decode.py

Loads ``checkpoints/babylm/diag_fused.pt``. Prefill T=32, then greedy
argmax. Reports tokens/s for the **full** block (LN, sLSTM T=1, SwiGLU,
tied head), not the RNN kernel alone. Default 60 s per setting. Local
smoke: no MLflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import torch
import yaml
from torch import Tensor

from pararnn.kernels.decode import decode_step, decode_wx
from pararnn.layout import SLSTM_HIDDEN
from pararnn.solvers.sequential import sequential_apply
from scripts.babylm_model import BabyLMModel, count_params

from gpu import select_device
from gpu import setup_logging as setup_gpu_logging

log = logging.getLogger("bench_babylm_decode")
DEFAULT_CONFIG = _REPO / "configs" / "train" / "babylm.yaml"
DEFAULT_CKPT = _REPO / "checkpoints" / "babylm" / "diag_fused.pt"
_PREFILL = 32
_WARMUP = 32


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load(spec: dict, ckpt: Path, device: torch.device) -> BabyLMModel:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cell_type = str(blob.get("cell_type", "diag_fused"))
    model = BabyLMModel(spec, cell_type=cell_type)
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    return model


def _prefill(model: BabyLMModel, tokens: Tensor) -> list[Tensor]:
    """Run prompt through the stack; return last sLSTM state per layer."""
    t = int(tokens.shape[1])
    pos = torch.arange(t, device=tokens.device)
    x = model.embed(tokens) + model.pos(pos)
    states: list[Tensor] = []
    for block in model.blocks:
        cell = block.rnn.layers[0]
        xn = block.norm_rnn(x)
        traj = sequential_apply(cell, xn)
        states.append(traj[:, -1].contiguous())
        x = x + traj[:, :, SLSTM_HIDDEN, :]
        x = x + block.mlp(block.norm_mlp(x))
    return states


class DecodeLoop:
    """Static buffers for T=1 greedy. RNN writes ``states`` in place."""

    def __init__(self, model: BabyLMModel, states: list[Tensor], token: Tensor, pos: Tensor):
        self.model = model
        self.states = states
        self.token = token
        self.pos = pos
        d = int(model.embed.embedding_dim)
        b = int(token.shape[0])
        device = token.device
        dtype = model.embed.weight.dtype
        self.wx = [torch.empty(b, 4 * d, device=device, dtype=dtype) for _ in model.blocks]
        self.seq_cap = int(model.pos.num_embeddings) - 1

    def step(self) -> Tensor:
        x = self.model.embed(self.token) + self.model.pos(self.pos)
        for i, block in enumerate(self.model.blocks):
            cell = block.rnn.layers[0]
            xn = block.norm_rnn(x)
            decode_wx(cell, xn, out=self.wx[i])
            decode_step(cell, self.states[i], wx=self.wx[i], out=self.states[i])
            x = x + self.states[i][:, SLSTM_HIDDEN, :]
            x = x + block.mlp(block.norm_mlp(x))
        logits = self.model.lm_head(self.model.norm_f(x))
        self.token.copy_(logits.argmax(dim=-1))
        self.pos.add_(1).clamp_(max=self.seq_cap)
        return logits


def _run_for(
    loop: DecodeLoop, *, seconds: float, device: torch.device, label: str
) -> tuple[int, float]:
    _sync(device)
    t0 = time.perf_counter()
    n = 0
    while True:
        loop.step()
        n += 1
        if n % 64 == 0:
            _sync(device)
            if time.perf_counter() - t0 >= seconds:
                break
    _sync(device)
    elapsed = time.perf_counter() - t0
    log.info(
        "%s steps=%s elapsed_s=%.3f step_ms=%.3f tok_per_s=%.1f (B=%s → %.1f tok/s total)",
        label,
        n,
        elapsed,
        1e3 * elapsed / n,
        n / elapsed,
        int(loop.token.shape[0]),
        n * int(loop.token.shape[0]) / elapsed,
    )
    return n, elapsed


def _try_graph(loop: DecodeLoop, device: torch.device) -> torch.cuda.CUDAGraph | None:
    if device.type != "cuda":
        return None
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            loop.step()
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            loop.step()
    except RuntimeError:
        log.exception("cuda_graph_capture_failed")
        return None
    return graph


def _run_graph_for(
    graph: torch.cuda.CUDAGraph,
    loop: DecodeLoop,
    *,
    seconds: float,
    device: torch.device,
    label: str,
) -> tuple[int, float]:
    _sync(device)
    t0 = time.perf_counter()
    n = 0
    while True:
        graph.replay()
        n += 1
        if n % 64 == 0:
            _sync(device)
            if time.perf_counter() - t0 >= seconds:
                break
    _sync(device)
    elapsed = time.perf_counter() - t0
    b = int(loop.token.shape[0])
    log.info(
        "%s steps=%s elapsed_s=%.3f step_ms=%.3f tok_per_s=%.1f (B=%s → %.1f tok/s total)",
        label,
        n,
        elapsed,
        1e3 * elapsed / n,
        n / elapsed,
        b,
        n * b / elapsed,
    )
    return n, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--prefill", type=int, default=_PREFILL)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    setup_gpu_logging()
    device = select_device("2080 Ti")
    spec = yaml.safe_load(args.config.read_text())
    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {args.ckpt}")
    torch.manual_seed(int(spec["seed"]))
    model = _load(spec, args.ckpt, device)
    d = int(spec["d_model"])
    n_layers = int(spec["num_layers"])
    vocab = int(spec["vocab_size"])
    log.info(
        "babylm_decode gpu=%s params=%s d=%s layers=%s vocab=%s mlp=%s seconds=%.0f",
        torch.cuda.get_device_name(device),
        count_params(model),
        d,
        n_layers,
        vocab,
        spec["mlp_act"],
        args.seconds,
    )
    with torch.no_grad():
        for batch in args.batches:
            prompt = torch.randint(0, vocab, (batch, args.prefill), device=device)
            states = _prefill(model, prompt)
            token = prompt[:, -1].contiguous()
            pos = torch.full((batch,), args.prefill, device=device, dtype=torch.long)
            loop = DecodeLoop(model, states, token, pos)
            for _ in range(_WARMUP):
                loop.step()
            _run_for(
                loop,
                seconds=args.seconds,
                device=device,
                label=f"python_loop B={batch}",
            )
            graph = _try_graph(loop, device)
            if graph is None:
                log.info("skip_graph B=%s", batch)
                continue
            _run_graph_for(
                graph,
                loop,
                seconds=args.seconds,
                device=device,
                label=f"cuda_graph B={batch}",
            )


if __name__ == "__main__":
    main()
