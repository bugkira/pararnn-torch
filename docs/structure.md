# Layout

One trunk (`main`). Three layers:

1. **Ops** — `src/pararnn/kernels/`: Triton fused Newton and scans, selected with `NewtonConfig(scan_backend=)`.
2. **Modules** — `cells/` is \(f\) (`step`); `solvers/` is Alg. 1; `layers/ParaRNN` is the sequence `nn.Module` (Newton in `.train()`, sequential in `.eval()`).
3. **Integrations** — `examples/` (onboarding), `scripts/` (benches / training; see [`scripts/README.md`](../scripts/README.md)), `notebooks/` (Colab demo).

```
ParaRNN/
├── docs/
│   ├── xlstm.md
│   ├── distributed.md          # DDP / FSDP2 / TP / context-parallel scan
│   ├── vllm.md                 # vLLM general_plugins + CausalLM
│   ├── structure.md
│   └── backward-scan-cap.md
├── src/pararnn/
│   ├── cells/                  # ParaGRU, ParaLSTM, ParaSLSTM
│   ├── layers/                 # ParaRNN, ParaSLSTMBlock
│   ├── models/                 # ParaSLSTMConfig + CausalLM
│   ├── serve/                  # BlockStackPool continuous batch
│   ├── vllm_plugin/            # ModelRegistry entry point
│   ├── solvers/                # sequential, Newton, scan, VJP
│   ├── kernels/                # Triton scans + fused Newton + T=1 decode_step
│   ├── distributed.py          # warmup + unwrap for DDP/FSDP
│   ├── tensor_parallel.py      # Megatron TP along d_h
│   ├── speculative.py          # linear-draft verify (one Newton scan)
│   ├── paged.py                # O(1) state slot pool (continuous batching)
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

## Current surface (0.7)

`NewtonConfig(scan_backend="auto")`; fused kernels prepend `h0`; `NewtonStats` and residual early-stop (`NewtonDivergenceError` if max|F|>1 after K). Compile-safe preset: fixed-K Autograd path, fused Alg. 1 as `pararnn::newton_*_fused` custom ops. `ParaRNN` takes a cell or a list, `return_hidden`, LSTM `output_hidden`. **ParaSLSTM** `mix='diag'` is the fused 4×4 path; `mix='head'` is an unfused ablation (`scan_dense`); `mix='dense'` is a `d_h<=8` test oracle.

Examples on `main`: `train_smoke`, Dyck-1, Z₂ parity, NX-AI `sLSTMBlock` hybrid, DDP/FSDP2, linear-draft verify, T=1 `decode_step`. `PagedStatePool` / `block_table` and tensor / context-parallel APIs live in the library + numerics tests; two-card torchrun demos sit on branch [`archive/distributed-demos`](https://github.com/bugkira/pararnn-torch/tree/archive/distributed-demos). Architecture notes: [`docs/distributed.md`](distributed.md).

## Naming

- PyPI / project: `pararnn-torch`. Import: `pararnn`.
- Cells follow the paper: `ParaGRU`, `ParaLSTM`, `ParaSLSTM`.
- Docs: lowercase kebab-case.
- Scan APIs are `scan_block2` / `scan_block4` for the `(B,T,2,2,d)` / `(B,T,4,4,d)` layouts; files are named by cell (`scan_lstm_block`, `scan_slstm_block`).
