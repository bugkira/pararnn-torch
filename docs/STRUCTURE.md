# File structure

Target layout. Directories that exist today are marked; the rest is planned. Do not add Hugging Face / CUDA / hybrid modules until the corresponding milestone has working tests.

```
ParaRNN/
├── .cursor/rules/              # agent conventions (uv, MLflow, citations, Apple license)
├── docs/
│   ├── literature.md           # annotated bibliography
│   ├── apple-ml-pararnn.md     # notes on the official repo
│   ├── bottlenecks.md          # eager Newton+scan: measured bottlenecks, ranked fixes
│   ├── torchification.md       # plan: solver prototype → drop-in nn.Module
│   ├── STRUCTURE.md            # this file
│   └── papers/                 # PDFs via scripts/fetch_papers.sh (gitignored)
├── third_party/
│   ├── README.md               # how to clone Apple's repo
│   └── ml-pararnn/             # local clone, gitignored
├── src/pararnn/                # our package (import: pararnn)
│   ├── cells/                  # ParaGRU (diag), ParaLSTM (CIFG, 2x2 block-diag)
│   ├── layers/                 # ParaRNN nn.Module (train Newton / eval sequential)
│   ├── solvers/                # sequential; Newton; Blelloch; Autograd J; eq. 2.6 bwd
│   ├── kernels/                # Triton diag + 2×2 scan; fused Newton; packed VJP
│   ├── hybrid/                 # later: linear SSM predictor + 1-step Newton
│   ├── models/                 # later: HF PreTrainedModel
│   └── train/                  # MLflow entrypoints (toy copy smoke)
├── tests/
│   ├── unit/                   # shapes, configs, inits
│   └── numerics/               # sequential vs parallel agreement, residual vs K
├── configs/                    # YAML/TOML; no buried argparse defaults
├── scripts/
│   ├── bench_time.py           # App. B; --config newton_compile.yaml for Dynamo
│   ├── profile_hotpath.py      # CUPTI: GRU T=64/2048, LSTM T=512
│   └── fetch_papers.sh
├── pyproject.toml              # uv; package name pararnn-torch
└── README.md
```

## v0.2 (implemented)

`cells/` + `layers/` + `solvers/` + `tests/numerics/` + toy train:

1. Sequential unroll of diagonal ParaGRU and CIFG ParaLSTM (paper §3).
2. Newton + parallel reduction in vectorized PyTorch (paper Alg. 1, App. A init \(h_l^0=f(0,x_l)\), \(K=3\)). Nonzero `h0` is supported; fused + nonzero `h0` falls back to eager.
3. `ParaRNN` sequence module: `.train()` Newton, `.eval()` sequential. Custom cells: `d_h` + `step` (`cells/protocol.py`).
4. Test: max residual and max |H_par − H_seq| vs sequential, float32 then **fp16** (Turing; not bf16). Wrapper vs raw solvers; grads vs BPTT.
5. Generic cell: Autograd Jacobian (`jacobian.py`); packed eq. 2.6 VJP for ParaGRU/LSTM.
6. Toy copy smoke: `python -m pararnn.train.toy` (MLflow `toy-copy`).

Not in v0.2: Mamba predictor, IFT adjoint, HF LM, pretrained weights. Optional Triton **diag / 2×2 scan**, **fused Newton**, and **packed VJP** are in `kernels/` (eager remains default).

## Naming

- PyPI / project: `pararnn-torch` (Apple already occupies `pararnn` conceptually).
- Import: `pararnn`.
- Do not name modules after Apple's `GRUDiagMH` / `LSTMCIFGDiagMH` class names; use `ParaGRU` / `ParaLSTM` as in the paper.
