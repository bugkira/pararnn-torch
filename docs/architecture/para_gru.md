# ParaGRU

**Fidelity:** paper-faithful (`mix='diag'`); `mix='head'` is the same Cho gates with per-head dense recurrent matrices (Dreamer-style).  
**Sources:** Danieli et al., *ParaRNN*, ICLR 2026 / arXiv:2510.21450, eq. 3.1a, 3.3; Cho et al. 2014 gates.  
**Code:** `pararnn.cells.para_gru.ParaGRU`

## State

- Single slot \(h_t \in \mathbb{R}^{d_h}\).
- `state_slots = 1`.

## Recurrence

Gates: update \(z\), reset \(r\), candidate \(n\). Activations: sigmoid / sigmoid / tanh.

**`mix='diag'`** (fused path). Diagonal recurrent vectors \(a_z, a_r, a_n \in \mathbb{R}^{d_h}\):

\[
\begin{aligned}
z_t &= \sigma(a_z \odot h_{t-1} + W_z x_t + b_z), \\
r_t &= \sigma(a_r \odot h_{t-1} + W_r x_t + b_r), \\
n_t &= \tanh(a_n \odot (r_t \odot h_{t-1}) + W_n x_t + b_n), \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot n_t.
\end{aligned}
\]

Input affines are packed as one `Linear` \(d_{\mathrm{in}}\to 3 d_h\).

**`mix='head'`.** Same gates; recurrent maps are dense per head
\(A_z, A_r, A_n \in \mathbb{R}^{n_{\mathrm{heads}}\times d_{\mathrm{head}}\times d_{\mathrm{head}}}\). LayerNorm used in some Dreamer LN-GRU stacks sits **outside** this cell.

## Jacobian class

- `mix='diag'` → `jac_structure='diag'`
- `mix='head'` → `jac_structure='head'`

## Parallel path

Newton + scan (Alg. 1). Default App. A guess \(h_t^{(0)} = f(0, x_t)\). Typical pin \(K{=}3\). App. C.1 clamp on recurrent entries: default `max_recurrent_norm=0.5` for diag; often `None` for head so dense \(A_*\) are not silently clipped.

## Deviations

- None for Cho-gate math vs Danieli eq. 3.1a / 3.3 on `mix='diag'`.
- `mix='head'` is the library’s block-diagonal recurrent option; Dreamer papers may wrap LN around the GRU — that LN is a separate module.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaGRU

cell = ParaGRU(32, 32, mix="diag")
h = torch.zeros(2, 32)
x = torch.randn(2, 32)
h = cell.step(h, x)
```

Pseudocode for one channel (diag): compute \(z,r,n\) as above, then
`h = (1-z)*h + z*n`.

## Agreement

```python
from pararnn import verify_agreement, ParaGRU, ParaRNN
import torch

m = ParaRNN(ParaGRU(32, 32, mix="diag"))
x = torch.randn(2, 64, 32)
print(verify_agreement(m, x).to_dict())
```

See [numerics contract](../numerics-contract.md). API snippets: [cell catalog](../cells.md).
