# ParaRNN

[![PyPI](https://img.shields.io/pypi/v/pararnn-torch?color=blue)](https://pypi.org/project/pararnn-torch/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/pararnn-torch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302587.svg)](https://doi.org/10.5281/zenodo.22302587)
[![ParaRNN](https://img.shields.io/static/v1?label=ParaRNN&message=ICLR%202026&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.21450)

PyTorch sequence module for parallel *training* of nonlinear RNNs (GRU, LSTM, sLSTM). Decode is the sequential unroll; on CUDA, T=1 uses a Triton step kernel.

Package **`pararnn-torch`**, import **`pararnn`**. **Alpha** — fused kernels are Triton on CUDA (compute capability ≥ 8.0).

## About

`ParaRNN` wraps a recurrent cell as an `nn.Module`. In `.train()` mode it solves the fixed-point constraints with Newton iterations and an associative scan (span \(O(\log T)\)). In `.eval()` mode it runs the sequential `step` unroll. On CUDA, `.eval()` at `T=1` uses a Triton decode kernel (one SRAM trip for gates + mix; `W_x` stays a GEMM).

`ParaSLSTM` with `mix='diag'` is the main fused path for exponentially gated sLSTM and xLSTM-style stacks. `ParaGRU` and `ParaLSTM` follow the same Newton wrapper.

Implementation follows [Danieli et al., ICLR 2026](https://arxiv.org/abs/2510.21450).

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

Place modules on a device like any `nn.Module` (`.to(device)`, or `device=` / `dtype=` on the cell and `ParaRNN`). Data parallel: wrap that module with `DistributedDataParallel` or FSDP2 `fully_shard` ([`docs/distributed.md`](docs/distributed.md)).

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
y_eval = model(x)  # sequential cell.step
```

- `.train()` with `solver='auto'` selects the parallel Newton path.
- `.eval()` selects sequential `step`. At `T=1` on CUDA with gradients off, that step is `decode_step` (Triton). Pass `out=` in a decode loop; capture `decode_wx` then `decode_step` in a CUDA graph. Use `solver='newton'` or `solver='sequential'` to force either path.

### sLSTM

```python
slstm = ParaRNN(ParaSLSTM(64, 64, mix="diag"), device=device)
y = slstm(torch.randn(4, 128, 64, device=device))
```

## Examples

Standalone scripts: install `pararnn-torch`, copy a file, run it. Knobs live
in the script; metrics go to stdout.

| Script | What it shows | Command | Extras |
|---|---|---|---|
| [`examples/train_smoke.py`](examples/train_smoke.py) | GRU identity CE smoke: AdamW | `uv run python examples/train_smoke.py` | — |
| [`examples/ddp_fsdp.py`](examples/ddp_fsdp.py) | DDP / FSDP2 one-step wrap of `ParaRNN` | `uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py` | two visible GPUs for NCCL |
| [`examples/speculative_draft.py`](examples/speculative_draft.py) | Greedy linear-draft verify: one Newton scan vs sequential | `uv run python examples/speculative_draft.py` | — |
| [`examples/decode_step.py`](examples/decode_step.py) | T=1 Triton decode vs eager `cell.step` | `uv run python examples/decode_step.py` | — |
| [`examples/dyck_language.py`](examples/dyck_language.py) | ParaSLSTM Newton grads, fail-loud on divergence | `uv run python examples/dyck_language.py` | — |
| [`examples/parity.py`](examples/parity.py) | Z₂ prefix tagging vs linear SSM | `uv run python examples/parity.py` | CUDA; writes `parity_curves.json` |
| [`examples/xlstm_hybrid.py`](examples/xlstm_hybrid.py) | NX-AI `sLSTMBlock` around fused `ParaSLSTM` | `uv add xlstm && uv run python examples/xlstm_hybrid.py` | `xlstm` |

FlashRNN train comparison and other benches live under `scripts/` (e.g. `scripts/slstm_vs_flashrnn.py`, extra `flashrnn`). Two-card TP / CP / paged-pool demos are on branch [`archive/distributed-demos`](https://github.com/bugkira/pararnn-torch/tree/archive/distributed-demos); API notes stay in [`docs/distributed.md`](docs/distributed.md).

## API overview

- **Cells:** `ParaGRU`, `ParaLSTM`, `ParaSLSTM` — recurrent maps \(f(h_{t-1}, x_t)\).
- **Sequence module:** `ParaRNN(cell, config=NewtonConfig(max_iters=3))` — stacks one or more cells.
- **Solver config:** `NewtonConfig(scan_backend="auto")` picks fused Triton on CUDA when available, else Triton scan + `step`, else eager Blelloch.
- **Low-level solvers** (bypass `ParaRNN`):

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)  # (B, T, hidden_size)
h = sequential_apply(cell, x)
```

- **Speculative verify:** `verify_linear_draft` — one Newton scan of a K-token draft, first mismatch \(k^\star\), state truncated to \(h_{k^\star}\).
- **Paged state:** `PagedStatePool` / `paged_apply` — O(1) slot per request; sequential CUDA and fused Newton index the pool through `block_table`. `offload` / `reload` park a slot on pinned host RAM.
- **Decode step:** `decode_step` — T=1 Triton recurrent step (gates + mix). `out=` reuses a buffer; `block_table` is slot ids into a pool. `decode_wx` fills `W_x(x)` for CUDA graphs. `can_decode_step` reports whether the kernel will run.

**Details:** output shapes, `mix=`, LSTM layout, scan backends — [`docs/xlstm.md`](docs/xlstm.md#api-notes). Data / tensor parallel — [`docs/distributed.md`](docs/distributed.md). Repo layout — [`docs/structure.md`](docs/structure.md).

## Method

Training imposes \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) and Newton-solves it with a parallel scan (Alg. 1, \(K=3\)). Paper 1-based indices vs code 0-based slots: [`src/pararnn/layout.py`](src/pararnn/layout.py).

GRU/LSTM warm-start follows App. A: \(h_l^{(0)} = f(0, x_l)\). sLSTM starts from the zero-hidden unroll (running \(m\) and \(n\)). Recurrent weights are clipped elementwise (App. C.1). Channels stay separate inside diagonal / 2×2 / 4×4 cells (eq. 3.3). Backward uses paper eq. 2.6 (one reverse scan).

## Citation

If you use this library, please cite the ParaSLSTM preprint and the ParaRNN framework.

```bibtex
@misc{sereda2026paraslstm,
  author       = {Sereda, Daniil},
  title        = {{ParaSLSTM}: Work-Efficient Parallel Training of Nonlinear {sLSTM} via Tropical Warm-Starts},
  month        = sep,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22302587},
  url          = {https://doi.org/10.5281/zenodo.22302587}
}

@inproceedings{danieli2026pararnn,
  title        = {{ParaRNN}: Unlocking Parallel Training of Nonlinear {RNNs} for Large Language Models},
  author       = {Danieli, Federico and Rodr{\'i}guez, Pau and Sarabia, Miguel and Suau, Xavier and Zappella, Luca},
  booktitle    = {International Conference on Learning Representations},
  year         = {2026},
  note         = {Oral. arXiv:2510.21450},
  url          = {https://arxiv.org/abs/2510.21450}
}
```

When an arXiv identifier is assigned, the Zenodo badge and `@misc` entry above will point to that preprint; the Zenodo DOI keeps the deposit timestamp.

## References

- Danieli, Rodríguez, Sarabia, Suau, Zappella. *ParaRNN*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450).
- Sereda. *ParaSLSTM*. [doi:10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

## License

MIT, [LICENSE](LICENSE).
