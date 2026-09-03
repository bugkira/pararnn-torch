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

`ParaRNN` is the Newton/sequential wrapper for any cell (GRU, LSTM, sLSTM). It is not a second sLSTM.

`mix='head'` stays so a paper ablation can run the original mixing through the same Newton loop (see `configs/train/dyck_vs_flashrnn_head.yaml`). Composing dense Jacobians in the scan is cubic in `4 d_head`; that is why the fused path is diagonal.

## Stacking

LayerNorm, residual, and FFN are not part of the library cell. The Z2 parity smoke (`examples/parity.py`) wraps `ParaRNN(ParaSLSTM)` in a local pre-norm residual. `examples/xlstm_hybrid.py` keeps an NX-AI `sLSTMBlock` (their LN, skip, FFN) and puts `ParaRNN(ParaSLSTM)` (`mix='diag'`) in the recurrent slot. Install: `uv add xlstm` (NX-AI package, Python 3.11+).
