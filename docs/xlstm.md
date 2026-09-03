# sLSTM in this library

[Beck et al.](https://arxiv.org/abs/2405.04517) split xLSTM into **mLSTM** (matrix memory, associative scan) and **sLSTM** (exponential gates, stabilizer, mixing). This package implements the sLSTM recurrence for ParaRNN training.

## Cell

`ParaSLSTM` state is `(B, T, 4, d_h) = (c, n, m, h)` with hidden slot 3 (`pararnn.layout`).

`mix='diag'` — channelwise `R` of shape `(4, d_h)`. Jacobian is 4×4 per channel. On CUDA, `scan_backend='auto'` runs cell + analytic J + 4×4 scan in one Triton kernel. Backward is eq. 2.6 packed VJP.

`mix='head'` — `R` of shape `(4, n_heads, d_head, d_head)`, dense inside a head. Jacobian is dense per head; scan uses `scan_dense`. `n_heads` must divide `d_h`.

## Block

`xLSTMBlock` is LayerNorm + `ParaRNN(ParaSLSTM)` + residual. `solver` is `'auto'` | `'newton'` | `'sequential'`, same as `ParaRNN`.

```python
from pararnn import NewtonConfig, xLSTMBlock

block = xLSTMBlock(d_model, mix="diag", solver="newton", config=NewtonConfig())
# .train() → Newton; .eval() → sequential step
```
