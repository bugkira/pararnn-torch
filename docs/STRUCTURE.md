# File structure

Target layout. Directories that exist today are marked; the rest is planned. Do not add Hugging Face / CUDA / hybrid modules until the corresponding milestone has working tests.

```
ParaRNN/
├── .cursor/rules/              # agent conventions (uv, MLflow, citations, Apple license)
├── docs/
│   ├── literature.md           # annotated bibliography
│   ├── apple-ml-pararnn.md     # notes on the official repo
│   ├── bottlenecks.md          # eager Newton+scan: measured bottlenecks, ranked fixes
│   ├── torchification.md       # archive: v0.2 solver → nn.Module (not the live backlog)
│   ├── para-slstm.md           # sLSTM Newton prototype (research branch)
│   ├── STRUCTURE.md            # this file
│   └── papers/                 # PDFs via scripts/fetch_papers.sh (gitignored)
├── third_party/
│   ├── README.md               # how to clone Apple's repo
│   └── ml-pararnn/             # local clone, gitignored
├── src/pararnn/                # our package (import: pararnn)
│   ├── cells/                  # ParaGRU (diag), ParaLSTM (CIFG), ParaSLSTM (stage 1)
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

## v0.3 (implemented)

v0.2 plus: `NewtonConfig(scan_backend="auto")` (fused if CUDA ParaGRU/LSTM fp16/32, else Triton scan, else eager); fused kernels prepend `h0`; `NewtonStats` + residual early-stop; `ParaRNN` list-of-cells, `return_hidden`, LSTM `output_hidden`.

Not in v0.3: Mamba predictor, IFT adjoint, HF LM, Para-sLSTM, pretrained weights.

## Naming

- PyPI / project: `pararnn-torch` (Apple already occupies `pararnn` conceptually).
- Import: `pararnn`.
- Do not name modules after Apple's `GRUDiagMH` / `LSTMCIFGDiagMH` class names; use `ParaGRU` / `ParaLSTM` as in the paper.
