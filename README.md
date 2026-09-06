# ParaRNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)
[![CI](https://github.com/bugkira/pararnn-torch/actions/workflows/ci.yml/badge.svg)](https://github.com/bugkira/pararnn-torch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pararnn-torch.svg)](https://pypi.org/project/pararnn-torch/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/bugkira/pararnn-torch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22302587.svg)](https://doi.org/10.5281/zenodo.22302587)
[![ParaRNN](https://img.shields.io/static/v1?label=ParaRNN&message=ICLR%202026&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2510.21450)
[![M²RNN](https://img.shields.io/static/v1?label=M%C2%B2RNN&message=arXiv%3A2603.14360&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2603.14360)

**Train nonlinear RNNs in parallel over the sequence, decode one step at a
time.** Package **`pararnn-torch`**, import **`pararnn`**.

Ordinary GRU/LSTM/sLSTM (and cousins) usually walk token-by-token. That is
fine for generation, and slow for training long contexts. ParaRNN trains the
*same* cell with a few Newton updates + a parallel scan over time — so
training cost grows much more gently with sequence length `T`, while
`.eval()` still uses the familiar one-step recurrence.

Alpha. Fused kernels need **Linux + NVIDIA CUDA** (fp32/fp16; lab includes
RTX 2080 Ti). **bf16 fused** needs Ampere+ (CC ≥ 8.0). CPU / Mac / Windows
without CUDA use the eager fallback via `scan_backend="auto"`.

- [News](#news)
- [What you get](#what-you-get)
- [Glossary](#glossary)
- [Models](#models)
- [Install](#install)
- [Quickstart](#quickstart)
- [Feel it (60s)](#feel-it-60s)
- [Benchmarks](#benchmarks)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [API overview](#api-overview)
- [Compatibility](#compatibility)
- [Method](#method)
- [Citation](#citation)

## News

- **[2026-09]** Long-context campaign through **T=131072**: several cells need
  only **2 Newton iterations** to match sequential unroll; RWKV-7 needs **0**
  (exact parallel scan). `NewtonConfig(max_iters=None)` picks those schedules.
- **[2026-09]** Cell zoo: `ParaTitans`, `ParaRWKV7`, `ParaHopfield`, `ParaCfC`,
  `ParaNLRU` (v0.13–0.17).
- **[2026-09]** Product entry: `ParaSLSTMBlock`, `ParaSLSTMForCausalLM`
  (`labels` CE, safetensors), continuous batch + vLLM plugin hooks.
- **[2026-09]** Factorized Newton for Dreamer-style `ParaGRU(mix='head')`;
  matrix-state M²RNN with measured iteration growth.

## What you get

A library of **nonlinear recurrent cells** with one training/inference API —
in the same niche as FlashAttention / Flash Linear Attention, for recurrence
instead of attention.

- **Fast training on long `T`** — a handful of Newton iterations + parallel
  scan (paper Alg. 1); measured **100–1000×** vs walking the same cell
  token-by-token
- **Cheap decode** — `.eval()` uses sequential `step`; on CUDA with `T=1`,
  a fused `decode_step` (+ optional CUDA graphs)
- **One API, many cells** — GRU, LSTM, sLSTM, Liquid CfC, Hopfield, RWKV-7,
  Titans-style memory, …
- **Stack-ready** — residual trunk block, CausalLM, paged state, continuous
  batch, speculative verify, optional vLLM registration
- **Honest iteration budgets** — we measure how many Newton steps you need
  vs `T` (see [Glossary](#glossary)); `max_iters=None` uses those tables
- **Train hygiene** — packed backward, `torch.compile` presets, DDP / FSDP2

**Swap paths** (details in [`docs/adoption.md`](docs/adoption.md)):

- Attention trunk → `ParaSLSTMBlock`
- Dreamer RSSM → `ParaGRU(mix='head', n_heads=8)`
- Liquid / CfC → `ParaCfC` (put Δt in the last channel of `x`)

## Glossary

| Term | Meaning here |
|---|---|
| **T** | Sequence length (tokens / time steps). |
| **Cell** | One recurrent update `h_t = f(h_{t-1}, x_t)` (GRU, sLSTM, …). |
| **Sequential** | The usual for-loop over time — ground truth and the decode path. |
| **Newton + scan** | Parallel train path: refine a whole-sequence guess with a few Newton steps; each step uses an associative scan over `T`. |
| **K / `max_iters`** | Number of Newton iterations you budget per forward. |
| **K\*(T)** | Smallest `K` where parallel output still matches sequential within tolerance τ≈1e-4. Measured per cell vs `T`; often **constant** (2) through 131k tokens. |
| **Agreement τ** | Max abs gap Newton vs sequential. Brand claim. Checked by `verify_agreement`. |
| **Residual gate** | `max|F(H)|`. Explosion fuse (`residual_fail=1.0` → `NewtonDivergenceError`). Separate from τ. |
| **Fused** | Triton CUDA kernel for Newton+scan (Linux + NVIDIA; bf16 needs Ampere+). |
| **Jacobian class** | Structure of ∂f/∂h — `diag` / `head` / `dense` / matrix-state — picks which kernel/scan we use. |
| **`verify_agreement`** | One-batch Newton vs sequential check ([numerics contract](docs/numerics-contract.md)). |

More on iterations: [`FAQs.md`](FAQs.md#what-is-critical-newton-depth-k).

## Models

| Cell | Jacobian | Parallel path | Notes |
|---|---|---|---|
| [`ParaSLSTM`](docs/cells.md#paraslstm-xlstm-style) | diag / head | fused Newton | main xLSTM-style path (`mix='diag'`) |
| [`ParaGRU`](docs/cells.md#paragru--paralstm-dreamer-style-block-gru) / `ParaLSTM` | diag / head | fused / factorized | Dreamer: `mix='head', n_heads=8` |
| [`ParaM2RNN`](docs/cells.md#param2rnn-research) | factor K×V | factorized Newton | matrix state; iterations grow ~log T |
| [`ParaNLRU`](docs/cells.md#paranlru) | diag | fused | Griffin / RG-LRU-style nonlinear slot |
| [`ParaCfC`](docs/cells.md#paracfc) | diag | fused | Liquid CfC; Δt = last channel of `x` |
| [`ParaHopfield`](docs/cells.md#parahopfield) | dense | dense scan | Modern Hopfield; keep d_h ≤ 32 |
| [`ParaRWKV7`](docs/cells.md#pararwkv7) | linear monoid | exact `(G,U)` scan | RWKV-7 Goose; **no Newton** (K\*=0) |
| [`ParaTitans`](docs/cells.md#paratitans) | diag | fused | shallow L=1 surprise-GD memory |

Full snippets: [`docs/cells.md`](docs/cells.md). Core algorithm:
[Danieli et al., ICLR 2026](https://arxiv.org/abs/2510.21450). HF wrappers and
serve hooks are this repo’s extras.

## Install

| | Users | Contributors |
|---|---|---|
| Command | see below | `git clone … && uv sync --group dev` |
| PyTorch | bring your own (CPU or CUDA) | pinned in `pyproject.toml` (cu128) |
| Python | 3.10+ | 3.10+ |
| Fused Triton | Linux + NVIDIA (fp32/fp16; bf16 → CC ≥ 8.0) | same; else `scan_backend="auto"` → eager |

```bash
# when published on PyPI:
pip install pararnn-torch

# bleeding edge / until first PyPI release:
pip install "pararnn-torch @ git+https://github.com/bugkira/pararnn-torch"

# editable:
git clone https://github.com/bugkira/pararnn-torch && cd pararnn-torch
uv sync --group dev
uv run pytest -q -m "not cuda"
```

PyPI Trusted Publishing is wired in
[`.github/workflows/release.yml`](.github/workflows/release.yml). Hardware and
Triton troubleshooting: [`INSTALL.md`](INSTALL.md) · [`FAQs.md`](FAQs.md).

Place modules with `.to(device)` like any `nn.Module`. Data parallel:
[`docs/distributed.md`](docs/distributed.md). Lab benches:
[`scripts/`](scripts/README.md) (omitted from the wheel).

## Quickstart

### 1. Trunk block (drop-in residual mixer)

```python
import torch
from pararnn import NewtonConfig, ParaSLSTMBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block = ParaSLSTMBlock(64, mlp_ratio=4.0, config=NewtonConfig(max_iters=3)).to(device)
x = torch.randn(2, 128, 64, device=device)

block.train()
y = block(x)          # Newton + scan inside the recurrent branch
y.sum().backward()

block.eval()
y_eval = block(x)     # sequential step (CUDA T=1: decode_step)
```

### 2. CausalLM

```python
from pararnn import ParaSLSTMConfig, ParaSLSTMForCausalLM

cfg = ParaSLSTMConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2, mlp_ratio=2.0)
model = ParaSLSTMForCausalLM(cfg).to(device)
ids = torch.randint(0, 256, (2, 32), device=device)

logits, loss = model(ids, labels=ids)
loss.backward()
out = model.generate(ids[:1, :8], max_new_tokens=16)
model.save_pretrained("./ckpt")   # config.json + model.safetensors
```

Smoke: [`examples/causal_lm_smoke.py`](examples/causal_lm_smoke.py) · continuous
batch: [`examples/continuous_batch.py`](examples/continuous_batch.py).

### 3. Cell + `ParaRNN`

```python
from pararnn import NewtonConfig, ParaGRU, ParaRNN

cell = ParaGRU(32, 64, device=device)
model = ParaRNN(cell, config=NewtonConfig(max_iters=3))
y = model(torch.randn(4, 128, 32, device=device))  # .train() → Newton
```

`.train()` → parallel Newton · `.eval()` → sequential `step` · CUDA `T=1` →
`decode_step`. Force either path with `solver='newton'|'sequential'`.

## Feel it (60s)

Toy task: running XOR on a bit string (prefix parity). A stacked ParaSLSTM
trained with Newton reaches **last-token accuracy 1.0** at length 16 and still
**1.0** at held-out length 32 — the nonlinear recurrence carries the bit.

```bash
uv run python examples/parity.py
```

Expect a JSON summary with `"newton"` arm accuracy `1.0` at `T=16` and `T=32`.
Interactive tour: [`notebooks/paraslstm_demo.ipynb`](notebooks/paraslstm_demo.ipynb)
([Colab](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)).

## Benchmarks

Two comparisons, both on **the same cell**:

1. **Fused Newton** (this library’s parallel train path) vs **sequential**
   (`torch.compile` of the usual time loop).
2. At very long `T`, how many Newton iterations you actually need (**K\***)
   and what wall time that costs.

Lab GPUs (RTX 2080 Ti / 3060) — consumer cards; regenerate plots with
`uv run python scripts/plot_readme_assets.py`.

![ParaSLSTM fused Newton vs sequential](assets/slstm_fused_vs_sequential.png)

Diag-sLSTM forward median (ms), B=8, d_h=256, float32, RTX 2080 Ti
(`scripts/slstm_vs_flashrnn.py`):

| T | fused Newton | sequential compiled |
|------:|-------------:|--------------------:|
| 256 | 6.4 | 112 |
| 1024 | 15.3 | 429 |
| 2048 | 29.1 | 840 |
| 4096 | 69.1 | 1735 |

![Newton iterations and wall time at T=131072](assets/k_star_wall_131k.png)

At **T=131072**, agreement tolerance τ=1e-4, B=1, RTX 3060
(`scripts/bench_k_star.py`). Labels under each cell are the measured Newton
budget K\* (how many iterations match sequential). Blue = parallel path;
brown = sequential for-loop:

| Cell | Newton iters K\* through T=131072 | @131k parallel vs sequential |
|---|---|---|
| ParaCfC | 2 (flat in T) | ~11 ms vs ~52 s (~4600×) |
| ParaTitans | 2 (flat in T) | ~13 ms vs ~62 s (~5000×) |
| ParaHopfield | 2 (flat in T) | ~250 ms vs ~38 s (~150×) |
| ParaRWKV7 | 0 (exact scan) | ~237 ms vs ~81 s (~340×, slim 1×16) |

ParaNLRU smoke (3060, B=8, T=2048, d_h=256, K=3): fused
**~2.7 ms** vs sequential **~549 ms**.

```bash
uv run python scripts/bench_k_star.py --cell all --time
uv run python scripts/slstm_vs_flashrnn.py --config configs/bench/newton_slstm_flashrnn.yaml
```

## Usage

Layered entrypoints live above. Cell zoo snippets:
[`docs/cells.md`](docs/cells.md).

### sLSTM / xLSTM-style

```python
from pararnn import ParaRNN, ParaSLSTM

slstm = ParaRNN(ParaSLSTM(64, 64, mix="diag"), device=device)
y = slstm(torch.randn(4, 128, 64, device=device))
```

### Dreamer-style block GRU

```python
from pararnn import ParaGRU, ParaRNN

# Block-diagonal A_*; CUDA factorized Newton. LN stays outside the cell.
rssm_h = ParaRNN(ParaGRU(512, 512, mix="head", n_heads=8), device=device)
y = rssm_h(torch.randn(4, 64, 512, device=device))
```

Smoke: [`examples/rssm_recurrent.py`](examples/rssm_recurrent.py).

### Research cells

`ParaM2RNN`, `ParaNLRU`, `ParaCfC`, `ParaHopfield`, `ParaRWKV7`, `ParaTitans` —
see [`docs/cells.md`](docs/cells.md). Pin Newton depth with `max_iters=int`,
use `max_iters=None` for auto schedules from measured K\*(T), or pass
`newton_iters_by_t={…}`.

## Training

BabyLM / diag-sLSTM stack: [`scripts/train_babylm.py`](scripts/train_babylm.py)
with configs under [`configs/train/`](configs/train/). Optional extras:
`pip install "pararnn-torch[train,lm]"`. Log runs with MLflow (`mlflow` group).
Distributed wrap: [`docs/distributed.md`](docs/distributed.md) ·
[`examples/ddp_fsdp.py`](examples/ddp_fsdp.py).

## Evaluation

- **Numerics** — parallel vs sequential agreement (`pytest -m cuda`); residual
  history if Newton fails to converge
- **Expressivity** — running-XOR parity ([`examples/parity.py`](examples/parity.py))
- **Wall-clock** — [`scripts/bench_k_star.py`](scripts/bench_k_star.py),
  [`scripts/slstm_vs_flashrnn.py`](scripts/slstm_vs_flashrnn.py)
- **LM smoke** — BabyLM notes under [`results/`](results/)

## Examples

| Script | What | Command |
|---|---|---|
| [`causal_lm_smoke.py`](examples/causal_lm_smoke.py) | CausalLM CE + generate + safetensors | `uv run python examples/causal_lm_smoke.py` |
| [`continuous_batch.py`](examples/continuous_batch.py) | Packed prefill + T=1 via `BlockStackPool` | `uv run python examples/continuous_batch.py` |
| [`rssm_recurrent.py`](examples/rssm_recurrent.py) | Dreamer `ParaGRU(mix='head')` | `uv run python examples/rssm_recurrent.py` |
| [`parity.py`](examples/parity.py) | Running XOR vs linear SSM | `uv run python examples/parity.py` |
| [`decode_step.py`](examples/decode_step.py) | T=1 Triton vs eager | `uv run python examples/decode_step.py` |
| [`ddp_fsdp.py`](examples/ddp_fsdp.py) | DDP / FSDP2 wrap | `uv run torchrun --nproc_per_node=2 examples/ddp_fsdp.py` |
| [`speculative_draft.py`](examples/speculative_draft.py) | Linear-draft verify | `uv run python examples/speculative_draft.py` |
| [`xlstm_hybrid.py`](examples/xlstm_hybrid.py) | NX-AI `sLSTMBlock` + fused ParaSLSTM | `uv add xlstm && uv run python examples/xlstm_hybrid.py` |

More: [`scripts/README.md`](scripts/README.md). Distributed demos:
[`archive/distributed-demos`](https://github.com/bugkira/pararnn-torch/tree/archive/distributed-demos).

## API overview

- **Cells** — [`docs/cells.md`](docs/cells.md); wrap with `ParaRNN` or call
  `newton_apply` / `sequential_apply` (`ParaM2RNN`, `ParaRWKV7`).
- **Trunk** — `ParaSLSTMBlock(d_model, mlp_ratio=4)` ([`docs/adoption.md`](docs/adoption.md)).
- **CausalLM / serve** — `ParaSLSTMForCausalLM`, `BlockStackPool`,
  `vllm.general_plugins` ([`docs/vllm.md`](docs/vllm.md)).
- **Solver** — `NewtonConfig(scan_backend="auto", max_iters=None|int)`;
  `picard_iters` warms the first guess for sLSTM / M²RNN;
  `verify_first_step=True` for a one-shot agreement smoke on `ParaRNN`;
  `compile_safe_config()` for `torch.compile(..., fullgraph=True)`.
- **Numerics check** — `verify_agreement(module, x)` → `AgreementReport`;
  full contract: [`docs/numerics-contract.md`](docs/numerics-contract.md).
- **Speculative** — `verify_linear_draft`.
- **Paged** — `PagedStatePool` / `paged_apply`.
- **Decode** — `decode_step`, `decode_wx`, `can_decode_step`.

```python
from pararnn.solvers import newton_apply, sequential_apply

h = newton_apply(cell, x)
h = sequential_apply(cell, x)
```

**Docs:** adoption · cells · [`xlstm.md`](docs/xlstm.md) ·
[`distributed.md`](docs/distributed.md) · [`vllm.md`](docs/vllm.md) ·
[`numerics-contract.md`](docs/numerics-contract.md) ·
[`oom-cookbook.md`](docs/oom-cookbook.md) ·
[`compile-amp.md`](docs/compile-amp.md) ·
[`shapes-layout.md`](docs/shapes-layout.md) ·
[`structure.md`](docs/structure.md) · [`INSTALL.md`](INSTALL.md) ·
[`FAQs.md`](FAQs.md).

## Compatibility

- **`torch.compile` / AMP:** [`docs/compile-amp.md`](docs/compile-amp.md) —
  `compile_safe_config()` for `fullgraph=True`; Newton opts out of outer
  autocast (explicit `.to(dtype)` for half). Tests:
  `tests/numerics/test_{compile,autocast}.py`.
- **DDP / FSDP / checkpoint:** [`docs/distributed.md`](docs/distributed.md);
  ultra-long train VRAM → `NewtonConfig(recompute=True)` and the
  [OOM cookbook](docs/oom-cookbook.md) (Hopfield `d_h` cap, RWKV slim heads).
- **Shapes / packing:** [`docs/shapes-layout.md`](docs/shapes-layout.md) —
  contiguous copies on fused paths; `cu_seqlens` support matrix.
- **Determinism:** packed VJP uses tile `tl.sum` then `.sum` (no `tl.atomic*`);
  set `CUBLAS_WORKSPACE_CONFIG=:4096:8` under
  `torch.use_deterministic_algorithms(True)`.

## Method

We solve for the whole hidden trajectory at once:

$`F(H)_t = h_t - f(h_{t-1}, x_t) = 0`$

with Newton updates; each linear solve is an associative scan over time
(Danieli et al., Alg. 1). Layout: [`src/pararnn/layout.py`](src/pararnn/layout.py).

**K\*(T)** is the smallest Newton iteration count that still matches a
sequential unroll within τ≈1e-4. We measure it per cell on a `T` grid and
ship the envelopes in `pararnn.solvers.newton.k_star`;
`NewtonConfig(max_iters=None)` reads them. Campaign script:
[`scripts/bench_k_star.py`](scripts/bench_k_star.py).

GRU/LSTM warm-start follows paper App. A ($`h_l^{(0)}=f(0,x_l)`$); sLSTM uses
zero-hidden + Picard. Backward follows paper eq. 2.6. M²RNN uses a factorized
Jacobian map
$`J[\Delta]=f\Delta+(1-f)(1-Z^{\odot 2})\odot(\Delta W)`$.

## Citation

If you use this library, please cite the software and the ParaRNN framework.

```bibtex
@software{sereda2026pararnn,
  author       = {Sereda, Daniil},
  title        = {{pararnn-torch}: Hardware-efficient parallel training for nonlinear {RNNs}},
  year         = {2026},
  url          = {https://github.com/bugkira/pararnn-torch},
  version      = {0.17.3}
}

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

## References

- Danieli et al. *ParaRNN*. ICLR 2026 (Oral). [arXiv:2510.21450](https://arxiv.org/abs/2510.21450).
- Mishra et al. *M²RNN*. [arXiv:2603.14360](https://arxiv.org/abs/2603.14360).
- Sereda. *ParaSLSTM*. [doi:10.5281/zenodo.22302587](https://doi.org/10.5281/zenodo.22302587).
- Beck et al. *xLSTM*. [arXiv:2405.04517](https://arxiv.org/abs/2405.04517).
- Lim et al. *DEER*. ICLR 2024. [arXiv:2309.12252](https://arxiv.org/abs/2309.12252).
- Merrill et al. *The Illusion of State in State-Space Models*. [arXiv:2404.08819](https://arxiv.org/abs/2404.08819).

## License

MIT, [LICENSE](LICENSE).
