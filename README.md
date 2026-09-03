# ParaRNN

PyTorch sequence module for **ParaRNN** (Danieli et al., ICLR 2026 Oral): parallel *training* of nonlinear RNNs by Newton + associative scan ([arXiv:2510.21450](https://arxiv.org/abs/2510.21450)). Decode is the usual sequential unroll.

Package **`pararnn-torch`**, import **`pararnn`**. Alpha; the fused path is Triton.

## Install

Python 3.10+.

**Users** — install from PyPI and bring your own PyTorch (CPU or CUDA):

```bash
pip install pararnn-torch
```

**Contributors** — clone and sync dev deps (torch from the cu128 index in `pyproject.toml`):

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
uv run pytest -q
```

Place the module on a device like any `nn.Module` (`.to(device)`, or `device=` / `dtype=` on the cell and `ParaRNN`).

Fused/Triton bf16 requires CUDA compute capability ≥ 8.0. Below that, `scan_backend="auto"` falls back.

`scripts/` is development tooling (benchmarks, profiling) and is omitted from the wheel.

## Quickstart

```python
import torch
from pararnn import NewtonConfig, ParaGRU, ParaRNN, ParaSLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaGRU(input_size=32, hidden_size=64, device=device)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3))
x = torch.randn(4, 128, 32, device=device)

model.train()  # solver='auto' → Newton + scan
y_train = model(x)

model.eval()  # sequential cell.step
y_eval = model(x)

slstm = ParaRNN(ParaSLSTM(64, 64, mix="diag"), device=device)
y = slstm(torch.randn(4, 128, 64, device=device))
```

`d_in` / `d_h` are aliases for `input_size` / `hidden_size`. `solver='newton'` / `'sequential'` force that path regardless of `train()` / `eval()`.

LSTM / sLSTM default output is the **hidden slot** `(B, T, hidden_size)`. Full state: `output_hidden=False`. Paper slot order is `(c, h)`; `return_hidden` last state is the last layer only (`(B, 2, hidden_size)` for LSTM). `hidden_layout="pytorch"` (ParaLSTM only) returns `(output, (h_n, c_n))` like `nn.LSTM`: `h_n` / `c_n` are `(num_layers, B, H)` regardless of `batch_first`, and `h0` slots are `(h, c)`. `dropout` is between layers, same as `nn.LSTM` (warns and is a no-op at `num_layers==1`). Unsupported: `bidirectional`, `proj_size`, packed sequences. `ParaRNN` stacks cells. sLSTM default is `mix='diag'` (fused 4×4 Newton). `mix='head'` is an unfused ablation vs Beck mixing; `mix='dense'` is a small-width Jacobian oracle. LayerNorm / residual / FFN are the caller's (or `examples/xlstm_hybrid.py` for an NX-AI `sLSTMBlock` around `ParaRNN(ParaSLSTM)`).

Low-level solvers:

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)  # (B, T, hidden_size)
h = sequential_apply(cell, x)
```

`NewtonConfig(scan_backend="auto")` picks fused Triton on CUDA for ParaGRU / ParaLSTM / ParaSLSTM `mix='diag'`, else a Triton scan + `step`, else eager Blelloch. Backward is paper eq. 2.6 (one reverse scan).

Examples: `uv run python examples/toy_copy.py`, `examples/dyck_language.py`, `examples/parity.py`. NX-AI block around this cell: `uv add xlstm` then `examples/xlstm_hybrid.py`.

## Method

Constraints \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) are Newton-solved with a parallel scan (Alg. 1, \(K=3\)). Index map (paper 1-based vs code 0-based): [`src/pararnn/layout.py`](src/pararnn/layout.py).

GRU/LSTM init follows App. A: \(h_l^{(0)} = f(0, x_l)\). sLSTM starts from the zero-hidden unroll (running \(m\)/\(n\)). Recurrent weights are clipped elementwise (App. C.1). Channels stay separate inside diagonal / 2×2 / 4×4 cells (eq. 3.3).

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). Official CUDA: [apple/ml-pararnn](https://github.com/apple/ml-pararnn).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

sLSTM layout: [`docs/xlstm.md`](docs/xlstm.md).

## License

`src/`, `tests/`, `configs/` — MIT, [LICENSE](LICENSE). Apple's CUDA at [`apple/ml-pararnn`](https://github.com/apple/ml-pararnn) uses a separate custom license.
