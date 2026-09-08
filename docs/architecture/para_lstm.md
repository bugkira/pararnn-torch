# ParaLSTM

**Fidelity:** paper-faithful  
**Sources:** Danieli et al., arXiv:2510.21450, eq. 3.1b, 3.3; CIFG peephole (Greff et al. 2017).  
**Code:** `pararnn.cells.para_lstm.ParaLSTM`

## State

- Two slots, layout `(..., 2, d_h)`: index `0` = cell \(c\), index `1` = hidden \(h\).
- `hidden_slot = 1` for readout.

## Recurrence

Diagonal recurrent \(a_f, a_z, a_o\) and peepholes \(c_f, c_o\). Packed input
`Linear` \(d_{\mathrm{in}}\to 3 d_h\) → \((f_x, z_x, o_x)\).

\[
\begin{aligned}
f_t &= \sigma(a_f \odot h_{t-1} + c_f \odot c_{t-1} + f_x), \\
z_t &= \tanh(a_z \odot h_{t-1} + z_x), \\
c_t &= f_t \odot c_{t-1} + (1-f_t)\odot z_t, \\
o_t &= \sigma(a_o \odot h_{t-1} + c_o \odot c_t + o_x), \\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
\]

Coupled input-forget: input gate is \(1-f_t\).

## Jacobian class

`jac_structure='block2'` — per-channel \(2\times 2\) blocks over \((c,h)\) (eq. 3.2b with diagonal \(A,C\)).

## Parallel path

Newton + scan on the stacked state. App. A guess from \(f(0,x_t)\). Typical \(K{=}3\). Default `max_recurrent_norm=0.5` (App. C.1).

## Deviations

None relative to Danieli CIFG peephole diag-\(A,C\) presentation used in this library.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaLSTM
from pararnn.layout import LSTM_CELL, LSTM_HIDDEN

cell = ParaLSTM(32, 32)
state = torch.zeros(2, 2, 32)  # (batch, slots, d_h)
x = torch.randn(2, 32)
state = cell.step(state, x)
h = state[..., LSTM_HIDDEN, :]
```

## Agreement

Wrap with `ParaRNN` and call `verify_agreement` ([numerics contract](../numerics-contract.md)).
