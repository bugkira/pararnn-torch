# Layout

One trunk (`main`). Three layers:

1. **Ops** — `src/pararnn/kernels/`: Triton fused Newton and scans, selected with `NewtonConfig(scan_backend=)`.
2. **Modules** — `cells/` is \(f\) (`step`); `solvers/` is Alg. 1; `layers/ParaRNN` is the sequence `nn.Module` (Newton in `.train()`, sequential in `.eval()`).
3. **Integrations** — `examples/`, `scripts/`, `models/xLSTMBlock` (pre-norm residual around ParaSLSTM).

```
ParaRNN/
├── docs/
│   ├── xlstm.md
│   └── structure.md
├── src/pararnn/
│   ├── cells/                  # ParaGRU, ParaLSTM, ParaSLSTM
│   ├── layers/                 # ParaRNN
│   ├── solvers/                # sequential, Newton, scan, VJP
│   ├── kernels/                # Triton scans + fused Newton
│   ├── layout.py
│   ├── weight_init.py          # App. C.1
│   └── models/                 # xLSTMBlock
├── examples/
├── tests/
│   ├── unit/
│   └── numerics/
├── configs/
├── scripts/
├── pyproject.toml              # package name pararnn-torch
└── README.md
```

## Current surface (0.4)

`NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats` and residual early-stop (`NewtonDivergenceError` if max|F|>1 after K). `ParaRNN` takes a cell or a list, `return_hidden`, LSTM `output_hidden`. **ParaSLSTM** `mix='diag'` (fused 4×4) and `mix='head'` (eager `scan_dense`). **xLSTMBlock**: pre-norm + residual, `solver="auto"|"newton"|"sequential"`. Examples: copy, Dyck-1, Z2 parity. Two-tile scan: `scan_diag_two_ranks`.

## Naming

- PyPI / project: `pararnn-torch`. Import: `pararnn`.
- Cells follow the paper: `ParaGRU`, `ParaLSTM`, `ParaSLSTM`.
- Docs: lowercase kebab-case.
- Scan APIs are `scan_block2` / `scan_block4` for the `(B,T,2,2,d)` / `(B,T,4,4,d)` layouts; files are named by cell (`scan_lstm_block`, `scan_slstm_block`).
