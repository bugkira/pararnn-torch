# Layout

One trunk (`main`). Three layers:

1. **Ops** — `src/pararnn/kernels/`: Triton fused Newton and scans, selected with `NewtonConfig(scan_backend=)`.
2. **Modules** — `cells/` is \(f\) (`step`); `solvers/` is Alg. 1; `layers/ParaRNN` is the sequence `nn.Module` (Newton in `.train()`, sequential in `.eval()`).
3. **Integrations** — `examples/`, `scripts/`. Residual / FFN stacking lives in examples (`parity.py`, `xlstm_hybrid.py`).

```
ParaRNN/
├── docs/
│   ├── xlstm.md
│   ├── distributed.md          # DDP / FSDP2 / TP / context-parallel scan
│   ├── structure.md
│   └── backward-scan-cap.md
├── src/pararnn/
│   ├── cells/                  # ParaGRU, ParaLSTM, ParaSLSTM
│   ├── layers/                 # ParaRNN
│   ├── solvers/                # sequential, Newton, scan, VJP
│   ├── kernels/                # Triton scans + fused Newton
│   ├── distributed.py          # warmup + unwrap for DDP/FSDP
│   ├── tensor_parallel.py      # Megatron TP along d_h
│   ├── speculative.py          # linear-draft verify (one Newton scan)
│   ├── layout.py
│   └── weight_init.py          # App. C.1
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

`NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats` and residual early-stop (`NewtonDivergenceError` if max|F|>1 after K). `ParaRNN` takes a cell or a list, `return_hidden`, LSTM `output_hidden`. **ParaSLSTM** `mix='diag'` is the fused 4×4 path; `mix='head'` is an unfused ablation (`scan_dense`); `mix='dense'` is a `d_h<=8` test oracle. Examples: copy, Dyck-1, Z2 parity, NX-AI `sLSTMBlock` hybrid, DDP/FSDP2, tensor-parallel diag block, context-parallel scan, linear-draft verify. Two-tile scan: `scan_diag_two_ranks` (streams). NCCL time split: `scan_diag_context_parallel`. `verify_linear_draft`: one Newton scan of a K-token chain, first mismatch \(k^\star\). Data / tensor / context parallel: `docs/distributed.md`.

## Naming

- PyPI / project: `pararnn-torch`. Import: `pararnn`.
- Cells follow the paper: `ParaGRU`, `ParaLSTM`, `ParaSLSTM`.
- Docs: lowercase kebab-case.
- Scan APIs are `scan_block2` / `scan_block4` for the `(B,T,2,2,d)` / `(B,T,4,4,d)` layouts; files are named by cell (`scan_lstm_block`, `scan_slstm_block`).
