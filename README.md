# ParaRNN

PyTorch sequence module for **ParaRNN** (Danieli et al., ICLR 2026 Oral): parallel *training* of nonlinear RNNs by Newton + associative scan ([arXiv:2510.21450](https://arxiv.org/abs/2510.21450)). Decode is the usual sequential unroll.

Package **`pararnn-torch`**, import **`pararnn`**. Alpha: our Triton fused path, not Apple’s CUDA kernels ([`apple/ml-pararnn`](https://github.com/apple/ml-pararnn), custom license).

## Install

Python 3.10+.

**Users** — install from PyPI and bring your own PyTorch (CPU or CUDA):

```bash
pip install pararnn-torch
```

**Contributors** — clone, sync dev deps (torch from the cu128 index in `pyproject.toml`):

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
uv run pytest -q
```

Place the module on a device like any `nn.Module` (`.to(device)`, or `device=` / `dtype=` on the cell, `ParaRNN`, and `xLSTMBlock`). Timing tables: [`docs/bottlenecks.md`](docs/bottlenecks.md).

**Hardware.** Fused/Triton bf16 requires CUDA compute capability ≥ 8.0; on lower capability (e.g. Turing, CC 7.5) the fused bf16 path refuses and `scan_backend="auto"` falls back. CI runs CPU on Python 3.10–3.12 plus two self-hosted GPU jobs: Ampere (RTX 3060, CC 8.6) and Turing (RTX 2080 Ti, CC 7.5).

**Benchmarks.** CUDA defaults to `CUDA_DEVICE_ORDER=FASTEST_FIRST`, so `CUDA_VISIBLE_DEVICES=0` is not necessarily the card `nvidia-smi` labels 0. Pin `CUDA_DEVICE_ORDER=PCI_BUS_ID` when comparing numbers. On the development box, `nvidia-smi` index 0 is the RTX 3060 while PyTorch's default index 0 is the RTX 2080 Ti.

`scripts/` is development tooling (benchmarks, profiling); it is not installed with the wheel.

## Quickstart

```python
import torch
from pararnn import NewtonConfig, ParaGRU, ParaRNN, ParaSLSTM, xLSTMBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaGRU(input_size=32, hidden_size=64, device=device)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3))
x = torch.randn(4, 128, 32, device=device)

model.train()  # solver='auto' → Newton + scan
y_train = model(x)

model.eval()  # sequential cell.step
y_eval = model(x)

block = xLSTMBlock(64, solver="auto", device=device)  # LN → sLSTM → residual
y = block(torch.randn(4, 128, 64, device=device))
```

`d_in` / `d_h` are aliases for `input_size` / `hidden_size`. `solver='newton'` / `'sequential'` force that path regardless of `train()` / `eval()`.

LSTM / sLSTM default output is the **hidden slot** `(B, T, hidden_size)`. Full state: `output_hidden=False`. Paper slot order is `(c, h)`; `return_hidden` last state is the last layer only (`(B, 2, hidden_size)` for LSTM). `hidden_layout="pytorch"` (ParaLSTM only) returns `(output, (h_n, c_n))` like `nn.LSTM`: `h_n` / `c_n` are `(num_layers, B, H)` regardless of `batch_first`, and `h0` slots are `(h, c)`. `dropout` is between layers, same as `nn.LSTM` (warns and is a no-op at `num_layers==1`). Not supported: `bidirectional`, `proj_size`, packed sequences — reverse-direction Newton would double the Triton kernel surface; packed/ragged batches do not fit the rectangular scan. `ParaRNN` stacking has no residual or LayerNorm; `xLSTMBlock` is the pre-norm residual for sLSTM (`mix='diag'` fused, `mix='head'` on `step`). mLSTM is not here.

Low-level solvers:

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)  # (B, T, hidden_size)
h = sequential_apply(cell, x)
```

`NewtonConfig(scan_backend="auto")` picks fused Triton on CUDA for ParaGRU / ParaLSTM / ParaSLSTM `mix='diag'`, else a Triton scan + `step`, else eager Blelloch. Backward is paper eq. 2.6 (one reverse scan), not autograd through the Newton loop.

Examples: `uv run python examples/toy_copy.py`, `examples/dyck_language.py`, `examples/parity.py`.

## Method

Constraints \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) are Newton-solved with a parallel scan (Alg. 1, \(K=3\)). Index map (paper 1-based vs code 0-based): [`src/pararnn/layout.py`](src/pararnn/layout.py).

GRU/LSTM init follows App. A: \(h_l^{(0)} = f(0, x_l)\). sLSTM starts from the zero-hidden unroll (running \(m\)/\(n\)). Recurrent weights are clipped elementwise (App. C.1), not a spectral bound. Channels do not mix inside diagonal / 2×2 / 4×4 cells (eq. 3.3).

Not implemented: Hugging Face models, Mamba warm-start, IFT adjoint, pretrained checkpoints.

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). Official CUDA: [apple/ml-pararnn](https://github.com/apple/ml-pararnn).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

Bibliography: [`docs/literature.md`](docs/literature.md). Layout / sLSTM notes: [`docs/xlstm.md`](docs/xlstm.md).

## License

`src/`, `tests/`, `configs/` — MIT, [LICENSE](LICENSE). Apple's CUDA at [`apple/ml-pararnn`](https://github.com/apple/ml-pararnn) is a separate custom license; do not copy it into this tree.
