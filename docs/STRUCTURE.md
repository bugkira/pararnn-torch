# File structure

One trunk (`main`). Cells of the same solver live in one package, not one git
branch per cell. Directories that exist today are marked; the rest is planned.
Do not add Hugging Face / CUDA / hybrid modules until the corresponding
milestone has working tests.

Three layers, mapped to **existing** dirs (not a rename of working imports):

1. **Ops** — `src/pararnn/kernels/`: Triton fused Newton + scans. Not a public
   `from pararnn import newton_fused_gru`; pick them with `NewtonConfig(scan_backend=)`.
2. **Modules** — `cells/` is \(f\) (`step`); `solvers/` is Alg. 1; `layers/ParaRNN`
   is the sequence `nn.Module` (Newton in `.train()`, sequential in `.eval()`).
   A cell is not a sequence layer: Autograd-generic `step(h, x)` has to live
   somewhere, and `ParaGRU.forward` vs `ParaGRU.step` would collide.
3. **Integrations** — `train/` (copy, Dyck-1), `scripts/` (timing). An xLSTM
   block with `backend="newton"|"flashrnn"` lands in `models/` when that switch
   actually runs, not as a stub.

```
ParaRNN/
├── .cursor/rules/              # agent conventions (uv, MLflow, citations, Apple license)
├── docs/
│   ├── literature.md           # annotated bibliography
│   ├── apple-ml-pararnn.md     # notes on the official repo
│   ├── bottlenecks.md          # eager Newton+scan: measured bottlenecks, ranked fixes
│   ├── torchification.md       # archive: v0.2 solver → nn.Module (not the live backlog)
│   ├── para-slstm.md           # sLSTM Newton (diag fused; head/dense eager)
│   ├── STRUCTURE.md            # this file
│   └── papers/                 # PDFs via scripts/fetch_papers.sh (gitignored)
├── third_party/
│   ├── README.md               # how to clone Apple's repo
│   └── ml-pararnn/             # local clone, gitignored
├── src/pararnn/                # our package (import: pararnn)
│   ├── cells/                  # ParaGRU (diag), ParaLSTM (CIFG), ParaSLSTM
│   ├── layers/                 # ParaRNN nn.Module (train Newton / eval sequential)
│   ├── solvers/                # sequential; Newton; Blelloch; Autograd J; eq. 2.6 bwd
│   ├── kernels/                # Triton diag + 2×2 + 4×4 scan; fused Newton; packed VJP
│   ├── hybrid/                 # later: linear SSM predictor + 1-step Newton
│   ├── models/                 # later: xLSTM adapter / HF — only with working tests
│   └── train/                  # MLflow entrypoints (toy copy + Dyck-1)
├── tests/
│   ├── unit/                   # shapes, configs, inits
│   └── numerics/               # sequential vs parallel agreement, residual vs K
├── configs/                    # YAML/TOML; no buried argparse defaults
├── scripts/
│   ├── bench_time.py           # App. B; newton_fused / newton_slstm / compile YAML
│   ├── profile_hotpath.py      # CUPTI: GRU T=64/2048, LSTM T=512
│   └── fetch_papers.sh
├── pyproject.toml              # uv; package name pararnn-torch
└── README.md
```

## v0.3 (implemented)

v0.2 plus: `NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats` + residual early-stop + **fail-loud** (`NewtonDivergenceError` if max|F|>1 after K); `ParaRNN` list-of-cells, `return_hidden`, LSTM `output_hidden`; **ParaSLSTM** `mix='diag'` (fused 4×4, library K=3, Picard P∈{1,3,5} from T). Toy copy + Dyck-1 smokes.

Not in v0.3: Mamba predictor, IFT adjoint, HF LM, sLSTM head-fused, pretrained weights.

## Naming

- PyPI / project: `pararnn-torch` (Apple already occupies `pararnn` conceptually).
- Import: `pararnn`.
- Do not name modules after Apple's `GRUDiagMH` / `LSTMCIFGDiagMH` class names; use `ParaGRU` / `ParaLSTM` as in the paper.
