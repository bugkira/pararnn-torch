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
   3. **Integrations** — `examples/` (copy, Dyck-1, Z2 parity), `scripts/` (timing + this-box GPU).
   `models/xLSTMBlock`: pre-norm + residual around ParaSLSTM, `solver="auto"|"newton"|"sequential"`.
   FlashRNN stays in `scripts/` and `examples/slstm_vs_flashrnn.py`. HF LM later.

```
ParaRNN/
├── .cursor/rules/              # agent conventions (uv, MLflow, citations, Apple license)
├── docs/
│   ├── literature.md           # annotated bibliography
│   ├── apple-ml-pararnn.md     # notes on the official repo
│   ├── bottlenecks.md          # eager Newton+scan: measured bottlenecks, ranked fixes
│   ├── accelerator-review.md   # GPU/Triton review checklist (memory, sync, numerics)
│   ├── para-slstm.md           # sLSTM Newton (diag fused; head/dense eager)
│   ├── xlstm.md                # if NX-AI xlstm took our sLSTM Newton (not our mLSTM)
│   ├── seq-parallel-report.md  # two-stream virtual ranks vs FlashRNN DDP claim
│   ├── tex/                    # seq-parallel-pararnn.tex (DDP vs time-span)
│   ├── structure.md            # this file
│   └── papers/                 # PDFs via scripts/fetch_papers.sh (gitignored)
├── third_party/
│   ├── README.md               # how to clone Apple's repo
│   └── ml-pararnn/             # local clone, gitignored
├── src/pararnn/                # our package (import: pararnn)
│   ├── cells/                  # ParaGRU (diag), ParaLSTM (CIFG), ParaSLSTM
│   ├── layers/                 # ParaRNN nn.Module (train Newton / eval sequential)
│   ├── solvers/                # sequential; Newton; Blelloch; Autograd J; eq. 2.6 bwd
│   ├── weight_init.py          # App. C.1 Xavier/Kaiming (not `init.py`)
│   ├── kernels/                # Triton scans + fused Newton
│   │   ├── precision.py        # fp16 DRAM / fp32 accum (not a preconditioner)
│   │   ├── fused_newton.py     # cell dispatcher (not a generic fused.py)
│   │   ├── scan_lstm_block.py  # 2×2 LSTM jac; API still scan_block2
│   │   ├── scan_slstm_block.py # 4×4 sLSTM jac; API still scan_block4
│   │   └── …                   # newton_*.py, scan_diag, VJP, Picard
│   ├── layout.py               # sLSTM slot order, LSTM ch swap, prepend_state
│   └── models/                 # xLSTMBlock (prenorm + residual; solver auto|newton|sequential)
├── examples/                   # toy_copy.py, dyck_language.py, parity.py, slstm_vs_flashrnn.py; not a package
├── tests/
│   ├── unit/                   # shapes, configs, inits
│   └── numerics/               # sequential vs parallel agreement, residual vs K
├── configs/                    # YAML/TOML; no buried argparse defaults
├── scripts/
│   ├── gpu.py                  # this-box GPU name + wait_until_free (not in the wheel)
│   ├── utils/                  # mlflow_helper.py for examples
│   ├── bench_time.py           # App. B; newton_fused / newton_slstm / compile YAML
│   ├── bench_slstm_tiled.py    # serial tile scan vs assoc; vs FlashRNN
│   ├── seq_parallel_ranks.py   # two CUDA streams as virtual scan ranks
│   ├── profile_hotpath.py      # CUPTI: GRU T=64/2048, LSTM T=512
│   └── fetch_papers.sh
├── pyproject.toml              # uv; package name pararnn-torch
└── README.md
```

## v0.3 (implemented)

v0.2 plus: `NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats` + residual early-stop + **fail-loud** (`NewtonDivergenceError` if max|F|>1 after K); `ParaRNN` list-of-cells, `return_hidden`, LSTM `output_hidden`; **ParaSLSTM** `mix='diag'` (fused 4×4) and **`mix='head'`** (eager `scan_dense`, K=4, train smoke vs FlashRNN); **xLSTMBlock** (pre-norm + residual, `solver="auto"|"newton"|"sequential"`). Examples: toy copy + Dyck-1 + Z2 parity (`examples/parity.py`). Sequence-parallel two-tile scan: `scan_diag_two_ranks` (one GPU, two streams).

Not in v0.3: Mamba predictor, IFT adjoint, HF LM, sLSTM head-fused, pretrained weights. We do not ship mLSTM (xLSTM already has it).

## Naming

- PyPI / project: `pararnn-torch` (Apple already occupies `pararnn` conceptually).
- Import: `pararnn`.
- Do not name modules after Apple's `GRUDiagMH` / `LSTMCIFGDiagMH` class names; use `ParaGRU` / `ParaLSTM` as in the paper.
- Docs: lowercase kebab-case (`structure.md`, `apple-ml-pararnn.md`).
- `weight_init.py` not `init.py` — that sat next to `__init__.py` and confused humans and autocompletion.
- `kernels/precision.py` is DRAM dtype / fp32 accumulators, not a linear-algebra preconditioner.
- `kernels/fused_newton.py` is the cell dispatcher. Scan kernels are named by cell (`scan_lstm_block`, `scan_slstm_block`), not by the Jacobian tile size in the filename. Function names stay `scan_block2` / `scan_block4` because that is the `(B,T,2,2,d)` / `(B,T,4,4,d)` API.
- `examples/` is scripts, not a package (no `__init__.py`). Dyck-1 lives in `dyck_language.py`. MLflow helpers live in `scripts/utils/mlflow_helper.py`. `scripts/` is CLI, not a package. GPU pin is `gpu.py`, not a sandbox.
