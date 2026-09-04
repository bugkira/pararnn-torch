# ParaRNN

[![PyPI](https://img.shields.io/pypi/v/pararnn-torch?color=blue)](https://pypi.org/project/pararnn-torch/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/pararnn-torch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/static/v1?label=Paper&message=2510.21450&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.21450)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302587.svg)](https://doi.org/10.5281/zenodo.22302587)

PyTorch sequence module for parallel *training* of nonlinear RNNs (GRU, LSTM, sLSTM). Decode is the usual sequential unroll.

Package **`pararnn-torch`**, import **`pararnn`**. **Alpha** — fused kernels are Triton on CUDA (compute capability ≥ 8.0).

## About

`ParaRNN` wraps a recurrent cell as an `nn.Module`. In `.train()` mode it solves the fixed-point constraints with Newton iterations and an associative scan (span \(O(\log T)\)). In `.eval()` mode it runs the standard sequential `step` unroll.

`ParaSLSTM` with `mix='diag'` is the main fused path for exponentially gated sLSTM and xLSTM-style stacks. `ParaGRU` and `ParaLSTM` follow the same Newton wrapper.

Implementation follows [Danieli et al., ICLR 2026](https://arxiv.org/abs/2510.21450). Apple's official CUDA kernels at [apple/ml-pararnn](https://github.com/apple/ml-pararnn) are reference-only under a separate license; this repo is MIT for our code.

## Install

| | Users | Contributors |
|---|---|---|
| Command | `pip install pararnn-torch` | `git clone … && uv sync --group dev` |
| PyTorch | bring your own (CPU or CUDA) | pinned in `pyproject.toml` (cu128 index) |
| Python | 3.10+ | 3.10+ |

**Users** — install from PyPI with your existing PyTorch:

```bash
pip install pararnn-torch
```

**Contributors** — clone and sync dev deps:

```bash
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
uv run pytest -q
```

**Hardware:** fused Triton bf16 needs CUDA compute capability ≥ 8.0 (Ampere and newer). Below that, `NewtonConfig(scan_backend="auto")` picks an eager fallback.

Place modules on a device like any `nn.Module` (`.to(device)`, or `device=` / `dtype=` on the cell and `ParaRNN`).

`scripts/` holds development benchmarks and profiling; it is omitted from the wheel.

## Quickstart

```python
import torch
from pararnn import NewtonConfig, ParaGRU, ParaRNN, ParaSLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = ParaGRU(input_size=32, hidden_size=64, device=device)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3))
x = torch.randn(4, 128, 32, device=device)

model.train()
y_train = model(x)  # Newton + associative scan
y_train.sum().backward()

model.eval()
y_eval = model(x)   # sequential cell.step
```

- `.train()` with `solver='auto'` selects the parallel Newton path.
- `.eval()` selects sequential `step`. Use `solver='newton'` or `solver='sequential'` to force either path.

### sLSTM

```python
slstm = ParaRNN(ParaSLSTM(64, 64, mix="diag"), device=device)
y = slstm(torch.randn(4, 128, 64, device=device))
```

## Examples

All scripts read YAML from `configs/train/`. Smoke runs log to MLflow when the `train` extra is installed (`uv sync --extra train`).

| Script | What it shows | Command | Extras |
|---|---|---|---|
| [`examples/toy_copy.py`](examples/toy_copy.py) | GRU smoke: CE + AdamW + MLflow | `uv run python examples/toy_copy.py --config configs/train/toy.yaml` | — |
| [`examples/dyck_language.py`](examples/dyck_language.py) | ParaSLSTM Newton grads, fail-loud on divergence | `uv run python examples/dyck_language.py --config configs/train/dyck.yaml` | — |
| [`examples/parity.py`](examples/parity.py) | Z₂ prefix tagging vs linear SSM | `uv run python examples/parity.py --config configs/train/parity_t16.yaml` | pins lab GPU in script |
| [`examples/xlstm_hybrid.py`](examples/xlstm_hybrid.py) | NX-AI `sLSTMBlock` around fused `ParaSLSTM` | `uv add xlstm && uv run python examples/xlstm_hybrid.py` | `xlstm` |
| [`examples/slstm_vs_flashrnn.py`](examples/slstm_vs_flashrnn.py) | ParaSLSTM Newton vs FlashRNN baseline | `uv sync --extra flashrnn --group dev && uv run python examples/slstm_vs_flashrnn.py --config configs/train/dyck_vs_flashrnn.yaml` | `flashrnn` |

## API overview

- **Cells:** `ParaGRU`, `ParaLSTM`, `ParaSLSTM` — recurrent maps \(f(h_{t-1}, x_t)\).
- **Sequence module:** `ParaRNN(cell, config=NewtonConfig(max_iters=3))` — stacks one or more cells.
- **Solver config:** `NewtonConfig(scan_backend="auto")` picks fused Triton on CUDA when available, else Triton scan + `step`, else eager Blelloch.
- **Low-level solvers** (bypass `ParaRNN`):

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)      # (B, T, hidden_size)
h = sequential_apply(cell, x)
```

**Details:** output shapes, `mix=`, LSTM layout, scan backends — [`docs/xlstm.md`](docs/xlstm.md#api-notes). Repo layout — [`docs/structure.md`](docs/structure.md).

## Method

Training imposes \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) and Newton-solves it with a parallel scan (Alg. 1, \(K=3\)). Paper 1-based indices vs code 0-based slots: [`src/pararnn/layout.py`](src/pararnn/layout.py).

GRU/LSTM warm-start follows App. A: \(h_l^{(0)} = f(0, x_l)\). sLSTM starts from the zero-hidden unroll (running \(m\) and \(n\)). Recurrent weights are clipped elementwise (App. C.1). Channels stay separate inside diagonal / 2×2 / 4×4 cells (eq. 3.3). Backward uses paper eq. 2.6 (one reverse scan).

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450). Official CUDA: [apple/ml-pararnn](https://github.com/apple/ml-pararnn).
- Sereda. *ParaSLSTM: Work-Efficient Parallel Training of Nonlinear sLSTM via Tropical Warm-Starts*. [doi:10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

## License

`src/`, `tests/`, `configs/` — MIT, [LICENSE](LICENSE). Apple's CUDA at [apple/ml-pararnn](https://github.com/apple/ml-pararnn) uses a separate custom license.
