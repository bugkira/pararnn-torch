# sLSTM in this library

[Beck et al.](https://arxiv.org/abs/2405.04517) split xLSTM into **mLSTM** (matrix memory, associative scan) and **sLSTM** (exponential gates, stabilizer, mixing). This package implements the sLSTM recurrence as a ParaRNN cell.

## Cell

`ParaSLSTM` state is `(B, T, 4, d_h) = (c, n, m, h)` with hidden slot 3 (`pararnn.layout`).

Three mix modes; they are not interchangeable for training:

| `mix` | Role | Scan / kernel |
|---|---|---|
| `'diag'` (default) | Fused training cell. Channelwise `R` of shape `(4, d_h)`. Jacobian is 4×4 per feature. | Triton fused Newton + packed VJP (eq. 2.6). `O(d)` combine. |
| `'head'` | Ablation vs Beck block-diagonal `R` `(4, n_heads, d_head, d_head)`. Emits a warning. `n_heads` must divide `d_h`. | Unfused `scan_dense` of a `(4 d_head)×(4 d_head)` Jacobian per head. No fused kernel. |
| `'dense'` | Autograd oracle for tests. Full-width `R`. `hidden_size <= 8`. | Eager dense Jacobian. |

```python
from pararnn import NewtonConfig, ParaRNN, ParaSLSTM

cell = ParaSLSTM(64, 64)  # mix='diag'
model = ParaRNN(cell, config=NewtonConfig(max_iters=3), solver="auto")
# .train() → Newton; .eval() → sequential step
```

`ParaRNN` is the Newton/sequential wrapper for any cell (GRU, LSTM, sLSTM).

`mix='head'` stays so a paper ablation can run Beck-style mixing through the same Newton loop (see `configs/train/dyck_vs_flashrnn_head.yaml`). Composing dense Jacobians in the scan is cubic in `4 d_head`; the fused path is diagonal for that reason.

## Stacking

LayerNorm, residual, and FFN live outside the library cell. The Z₂ parity smoke (`examples/parity.py`) wraps `ParaRNN(ParaSLSTM)` in a local pre-norm residual. `examples/xlstm_hybrid.py` keeps an NX-AI `sLSTMBlock` (their LN, skip, FFN) and puts `ParaRNN(ParaSLSTM)` (`mix='diag'`) in the recurrent slot. Install: `uv add xlstm` (NX-AI package, Python 3.11+).

## API notes

### Aliases and solver mode

- `d_in` / `d_h` are aliases for `input_size` / `hidden_size` on cells.
- `solver='newton'` or `solver='sequential'` on `ParaRNN` forces that path regardless of `.train()` / `.eval()`.
- Default `solver='auto'`: Newton in train mode, sequential in eval mode.

### Outputs

LSTM and sLSTM default output is the **hidden slot** `(B, T, hidden_size)`.

- `output_hidden=False` returns the full internal state tensor.
- Paper slot order is `(c, h)`; see `pararnn.layout`.
- `return_hidden=True` adds the last-layer final state. LSTM shape: `(B, 2, hidden_size)` for `(c, h)`.

### ParaLSTM PyTorch layout

`hidden_layout="pytorch"` (ParaLSTM only) returns `(output, (h_n, c_n))` like `nn.LSTM`:

- `h_n` / `c_n` are `(num_layers, B, H)` regardless of `batch_first`.
- Initial states `h0` use slots `(h, c)` in that order.

### Stacking and dropout

- `ParaRNN` accepts one cell or a list of cells (multi-layer stack).
- `dropout` applies between layers, same convention as `nn.LSTM`. A warning is emitted when `num_layers==1` (no-op).

### Unsupported `nn.LSTM` options

`bidirectional`, `proj_size`, and packed sequences are outside the current API.

### Scan backend

`NewtonConfig(scan_backend="auto")` resolution:

1. Fused Triton on CUDA for `ParaGRU`, `ParaLSTM`, and `ParaSLSTM` with `mix='diag'`.
2. Triton associative scan + per-step `step` when fused kernels are unavailable.
3. Eager Blelloch scan as the CPU / fallback path.

See also [`structure.md`](structure.md) for kernel file layout.

### Backward

The adjoint follows paper eq. 2.6: one reverse associative scan of \(J^\top\), then a packed cell VJP for \(\nabla R\) and \(\nabla W_x\).
