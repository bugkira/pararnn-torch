# ParaM2RNN

**Fidelity:** research-variant / paper-core  
**Sources:** Mishra et al., *M²RNN*, arXiv:2603.14360 (matrix-state recurrence).  
**Code:** `pararnn.cells.para_m2rnn.ParaM2RNN`

## State

- Matrix state \(H_t \in \mathbb{R}^{K\times V}\) (`state_shape = (k_dim, v_dim)`).
- Flat width \(d_h = K\cdot V\) for protocol aliases.

## Recurrence (library core)

MVP projections: one `Linear` \(d_{\mathrm{in}}\to K+V+1\) yields \((k, v, f_{\mathrm{logit}})\), \(f=\sigma(f_{\mathrm{logit}})\). Right-multiply transition \(W\in\mathbb{R}^{V\times V}\) (default init: identity):

\[
H_t = f_t\, H_{t-1} + (1-f_t)\,\tanh(H_{t-1} W + k_t v_t^\top)
\]

with \(k_t\in\mathbb{R}^{K}\), \(v_t\in\mathbb{R}^{V}\), and \(f_t\) broadcast over the matrix (implementation: `m2rnn_gates`).

## Jacobian class

`jac_structure='m2rnn'` — factorized Newton (`newton_m2rnn_factorized`). Dense \((KV)^2\) exists as an oracle only.

## Parallel path

Factorized Newton on the matrix residual. Residual / \(K\) budgets are measured separately (growth with \(K,V\) can be steeper than diag cells).

## Deviations (read carefully)

The paper’s **full block** includes short causal conv + SiLU on \(q,k,v\) and a richer forget / readout path. In this library:

- The **Newton cell** is the matrix recurrence above with input-only \(k,v,f\) from a single linear map.
- Conv, \(H^\top q\) readout, gates, and \(W_o\) live in **outer** modules when used in BabyLM-style stacks (lab).
- Treating `ParaM2RNN` alone as a drop-in of the full Mishra block is incorrect; match the paper block diagram before claiming architecture parity.

This is the class of mistake we already hit once: implementing a **core** and presenting it as the **full architecture**.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaM2RNN

cell = ParaM2RNN(d_in=32, k_dim=8, v_dim=8)
H = torch.zeros(2, 8, 8)
x = torch.randn(2, 32)
H = cell.step(H, x)
```

## Agreement

Compare `sequential_apply` / looped `step` to `newton_apply` with a generous residual atol on small \(K,V\). Full paper-block parity requires the outer stack, documented in lab notes.
