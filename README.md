# ParaRNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)
[![CI](https://github.com/bugkira/pararnn-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/bugkira/pararnn-torch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/bugkira/pararnn-torch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302587.svg)](https://doi.org/10.5281/zenodo.22302587)
[![ParaRNN](https://img.shields.io/static/v1?label=ParaRNN&message=ICLR%202026&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.21450)
[![M²RNN](https://img.shields.io/static/v1?label=M%C2%B2RNN&message=arXiv%3A2603.14360&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2603.14360)

PyTorch sequence module for parallel *training* of nonlinear RNNs (GRU, LSTM, sLSTM, M²RNN). Decode is the sequential unroll; on CUDA, T=1 uses a Triton step kernel.

Package **`pararnn-torch`**, import **`pararnn`**. **Alpha** — fused kernels are Triton on CUDA (compute capability ≥ 8.0).

## About

`ParaRNN` wraps a recurrent cell as an `nn.Module`. In `.train()` mode it solves the fixed-point constraints with Newton iterations and an associative scan (span \(O(\log T)\)). In `.eval()` mode it runs the sequential `step` unroll. On CUDA, `.eval()` at `T=1` uses a Triton decode kernel (one SRAM trip for gates + mix; `W_x` stays a GEMM).

`ParaSLSTM` with `mix='diag'` is the main fused path for exponentially gated sLSTM and xLSTM-style stacks. `ParaGRU` and `ParaLSTM` follow the same Newton wrapper. For Dreamer-style block recurrence, use `ParaGRU(mix='head', n_heads=8)` (block-diagonal `A_*`, full `W_x`; CUDA factorized Newton; `scan_backend='eager'` for the dense-J oracle). Cho gates only — LayerNorm in Dreamer LN-GRU sits outside this cell.

`ParaM2RNN` is a research cell for the matrix-state recurrence in [Mishra et al., arXiv:2603.14360](https://arxiv.org/abs/2603.14360): state \(H\in\mathbb{R}^{K\times V}\) with dense value-axis mix \(HW\) inside \(\tanh\). Training uses a *factorized* Newton Jacobian (\(O(KV^{2})\) matvecs, no dense \((KV)^{2}\)); measured critical depth \(K^{*}(T)\) is consistent with \(\Theta(\log T)\). Upstream product kernels keep time sequential; this library supplies the parallel-train path on the same math.

Implementation of the Newton+scan core follows [Danieli et al., ICLR 2026](https://arxiv.org/abs/2510.21450).

## Install

| | Users / clone | Contributors |
|---|---|---|
| Command | see below | `git clone … && uv sync --group dev` |
| PyTorch | bring your own (CPU or CUDA) | pinned in `pyproject.toml` (cu128 index) |
| Python | 3.10+ | 3.10+ |

**From source** (current release path; PyPI Trusted Publishing is wired in
[`.github/workflows/release.yml`](.github/workflows/release.yml) for the first
`v*` tag once the GitHub `pypi` environment is linked):

```bash
pip install "pararnn-torch @ git+https://github.com/bugkira/pararnn-torch"
# or editable:
git clone https://github.com/bugkira/pararnn-torch
cd pararnn-torch
uv sync --group dev
uv run pytest -q -m "not cuda"
```

**Hardware:** fused Triton bf16 needs CUDA compute capability ≥ 8.0 (Ampere and newer). Below that, `NewtonConfig(scan_backend="auto")` picks an eager fallback.

Place modules on a device like any `nn.Module` (`.to(device)`, or `device=` / `dtype=` on the cell and `ParaRNN`). Data parallel: wrap that module with `DistributedDataParallel` or FSDP2 `fully_shard` ([`docs/distributed.md`](docs/distributed.md)).

`scripts/` holds development benchmarks and profiling; it is omitted from the wheel ([`scripts/README.md`](scripts/README.md)). Local literature PDFs: `bash scripts/fetch_papers.sh` → [`docs/sources/`](docs/sources/) (gitignored).

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

### Dreamer-style block GRU

```python
# Block-diagonal recurrence (CUDA factorized Newton). Cho gates; LN stays outside.
# n_heads sets block size: d_head = hidden // n_heads. Factorized path avoids
# dense d×d; Dreamer-like 512/8 → d_head=64. Prefer more heads if latency-bound.
rssm_h = ParaRNN(ParaGRU(512, 512, mix="head", n_heads=8), device=device)
y = rssm_h(torch.randn(4, 64, 512, device=device))
```

Head fused Newton medians (ms), float32, RTX 2080 Ti, `K=3`
(`scripts/bench_gru_head.py`). Paths: `d_head≤64` full SRAM, `≤128` streamed-`A`,
larger hybrid tiled. Long-T train VRAM: `NewtonConfig(recompute=True)`.

| setup | `d_head` | Newton | fwd+bwd |
|------:|---------:|-------:|-------:|
| `B=4`, `T=128` | 64 | 2.4 | — |
| `B=4`, `T=128` | 96 / 128 | 9–10 | — |
| `B=1`, `T=4096` | 64 | 51 | 83 |
| `B=1`, `T=4096` | 96 | 220 | 293 |

### M²RNN (research)

```python
from pararnn import ParaM2RNN, newton_apply, sequential_apply

m2 = ParaM2RNN(d_in=32, k_dim=16, v_dim=16, device=device)
x = 0.15 * torch.randn(2, 128, 32, device=device)
h_seq = sequential_apply(m2, x)
h_par = newton_apply(m2, x, NewtonConfig(max_iters=8, residual_atol=1e-5))
```

State shape is `(B, T, K, V)`. Prefer `residual_atol` early-stop over a fixed
over-provisioned \(K\) on large \(K{\times}V\). Critical-depth recipe:
`scripts/bench_m2rnn_k_scale.py`.

## Results

Interactive API tour: [`notebooks/paraslstm_demo.ipynb`](notebooks/paraslstm_demo.ipynb)
([Open in Colab](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb); private clones need a `GITHUB_TOKEN` secret — see [`notebooks/README.md`](notebooks/README.md)).
\(\mathbb{Z}_2\) training lives in [`examples/parity.py`](examples/parity.py).

Diag-sLSTM forward median latency (ms), \(B{=}8\), \(d_h{=}256\), float32,
RTX 2080 Ti, 10 seeds (`scripts/slstm_vs_flashrnn.py` / paper Tier-A timing).
Fused Newton is this library's Alg. 1 path; sequential is `torch.compile` of
the same cell's `step` unroll. FlashRNN is a sequential head-mix kernel on the
same GPU for context.

| \(T\) | fused Newton | sequential compiled | FlashRNN |
|------:|-------------:|--------------------:|---------:|
| 256 | 6.4 | 112 | 1.2 |
| 1024 | 15.3 | 429 | 3.1 |
| 2048 | 29.1 | 840 | 5.8 |
| 4096 | 69.1 | 1735 | 11.3 |

Z₂ prefix tagging (`examples/parity.py`): last-token accuracy **1.0** at train
length 16 and held-out length 32 for ParaSLSTM Newton; a matched-width linear
SSM arm lands near chance (~0.53–0.56) on the same protocol.

Reproduce:

```bash
uv run python examples/train_smoke.py
uv run python examples/parity.py
uv run python scripts/slstm_vs_flashrnn.py --config configs/bench/newton_slstm_flashrnn.yaml
uv run python scripts/train_babylm.py --config configs/train/babylm.yaml   # needs --extra lm
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

FlashRNN train comparison and other benches live under `scripts/` (see
[`scripts/README.md`](scripts/README.md)). Two-card TP / CP / paged-pool demos
are on branch [`archive/distributed-demos`](https://github.com/bugkira/pararnn-torch/tree/archive/distributed-demos);
API notes stay in [`docs/distributed.md`](docs/distributed.md).

## API overview

- **Cells:** `ParaGRU`, `ParaLSTM`, `ParaSLSTM`, `ParaM2RNN` — recurrent maps \(f(h_{t-1}, x_t)\). M²RNN state is `(B, T, K, V)`.
- **Sequence module:** `ParaRNN(cell, config=NewtonConfig(max_iters=3))` — stacks one or more cells (`ParaM2RNN` also works through `newton_apply` / `sequential_apply` directly).
- **Trunk block:** `ParaSLSTMBlock(d_model, mlp_ratio=4)` — RMSNorm + ParaSLSTM + SwiGLU residuals for LM stacks (`docs/xlstm.md`).
- **CausalLM / vLLM:** `ParaSLSTMForCausalLM` + `BlockStackPool` continuous batch +
  `vllm.general_plugins` registration (`docs/vllm.md`, `examples/continuous_batch.py`).
- **Solver config:** `NewtonConfig(scan_backend="auto")` picks fused Triton on CUDA when available, else Triton scan + `step`, else eager Blelloch. For `ParaM2RNN`, `picard_iters>=1` selects a frozen-\(W\) warm-start.
- **Low-level solvers** (bypass `ParaRNN`):

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)  # (B, T, hidden_size) or (B, T, K, V) for ParaM2RNN
h = sequential_apply(cell, x)
```

- **Speculative verify:** `verify_linear_draft` — one Newton scan of a K-token draft, first mismatch \(k^\star\), state truncated to \(h_{k^\star}\).
- **Paged state:** `PagedStatePool` / `paged_apply` — O(1) slot per request; sequential CUDA and fused Newton index the pool through `block_table`. `offload` / `reload` park a slot on pinned host RAM.
- **Decode step:** `decode_step` — T=1 Triton recurrent step (gates + mix). `out=` reuses a buffer; `block_table` is slot ids into a pool. `decode_wx` fills `W_x(x)` for CUDA graphs. `can_decode_step` reports whether the kernel will run.

**Details:** output shapes, `mix=`, LSTM layout, scan backends — [`docs/xlstm.md`](docs/xlstm.md#api-notes). Data / tensor parallel — [`docs/distributed.md`](docs/distributed.md). vLLM plugin — [`docs/vllm.md`](docs/vllm.md). Repo layout — [`docs/structure.md`](docs/structure.md).

## Compatibility

- **`torch.compile`:** with the compile-safe preset (fixed K, no residual host sync), `newton_apply` traces as a single graph (`fullgraph=True`) on eager and fused paths (including `ParaM2RNN`). Fused Alg. 1 kernels are `pararnn::newton_*_fused` custom ops with `register_fake` (`tests/numerics/test_compile.py`, `kernels/custom_ops.py`). Eq. 2.6 stays on the module-level `Autograd.Function` (fused ops do not carry `W_x`).
- **Precision / AMP:** put the module and `x` in fp16/bf16/fp32 explicitly. Under outer `torch.autocast`, Newton opts out and stays in the tensor dtype so the eq. 2.6 VJP keeps one dtype (`tests/numerics/test_autocast.py`).
- **DDP / FSDP / checkpoint:** wrap `ParaRNN` with DDP or FSDP2 (`docs/distributed.md`, `examples/ddp_fsdp.py`). Non-reentrant `torch.utils.checkpoint` and `state_dict` round-trip: `tests/numerics/test_checkpoint.py`. For ultra-long train \(T\), `NewtonConfig(recompute=True)` rematerializes \(H^\star\) in the eq. 2.6 backward (`tests/numerics/test_recompute.py`).
- **Deterministic algorithms:** packed eq. 2.6 VJP (diag GRU/LSTM/sLSTM and `ParaM2RNN`) reduces with tile `tl.sum` then `.sum` where applicable — no Triton atomics on the packed path; parameter grads bit-match across identical calls (`tests/numerics/test_vjp_determinism.py`). With `torch.use_deterministic_algorithms(True)`, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for cuBLAS GEMMs (`W_x` / `∇x`); the first packed `cell_vjp` under that flag re-checks param grads once and logs a single warning if they drift (`pararnn.determinism`).

## Method

Training imposes \(F(H)_t = h_t - f(h_{t-1}, x_t) = 0\) and Newton-solves it with a parallel scan (Alg. 1, \(K=3\)). Paper 1-based indices vs code 0-based slots: [`src/pararnn/layout.py`](src/pararnn/layout.py).

GRU/LSTM warm-start follows App. A: \(h_l^{(0)} = f(0, x_l)\). sLSTM starts from the zero-hidden unroll (running \(m\) and \(n\)). Recurrent weights are clipped elementwise (App. C.1). Channels stay separate inside diagonal / 2×2 / 4×4 cells (eq. 3.3). Backward uses paper eq. 2.6 (one reverse scan). For `ParaM2RNN`, the Jacobian is the factorized map \(J[\Delta]=f\Delta+(1-f)(1-Z^{\odot2})\odot(\Delta W)\).

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
- Mishra, Tan, Stoica, Gonzalez, Dao. *M²RNN*. [arXiv:2603.14360](https://arxiv.org/abs/2603.14360).
- Sereda. *ParaSLSTM*. [doi:10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

## License

MIT, [LICENSE](LICENSE).
