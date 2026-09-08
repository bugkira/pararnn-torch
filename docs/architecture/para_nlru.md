# ParaNLRU

**Fidelity:** research-variant relative to **linear** Griffin / RecurrentGemma RG-LRU  
**Sources:** De et al., *Griffin* (2024) RG-LRU (linear-in-\(h\) core); this cell adds nonlinearity while keeping a diagonal Newton Jacobian.  
**Code:** `pararnn.cells.para_nlru.ParaNLRU`

## State

- Single slot \(h_t \in \mathbb{R}^{d_h}\).

## Recurrence

Input-only gate \(a_t = a_t(x_t)\); diagonal recurrent mix \(u \in \mathbb{R}^{d_h}\):

\[
\begin{aligned}
a_t &= \sigma(W_a x_t + b_a), \\
h_t &= a_t \odot h_{t-1}
  + (1-a_t)\odot\tanh(W_c x_t + b_c + u \odot h_{t-1}).
\end{aligned}
\]

Packed `Linear` \(d_{\mathrm{in}}\to 2 d_h\) for \((a,c)\) affines.

## Jacobian class

`jac_structure='diag'` — fused diag Newton/scan class.

## Parallel path

Same Alg. 1 path as other diag cells. Typical \(K{=}3\). Default `max_recurrent_norm=0.5` on \(u\). Picard warm-start is unused for this cell in our stacks.

## Deviations

- Griffin RG-LRU is **linear** in \(h\) inside the recurrent core. ParaNLRU places \(\tanh\) and \(u\odot h_{t-1}\) inside the candidate so the cell is nonlinear while \(J\) stays channelwise diagonal.
- Naming “NLRU” marks that nonlinear slot; do not treat trajectories as bit-matching a Griffin checkpoint.

## Reproduce (sequential)

```python
import torch
from pararnn import ParaNLRU

cell = ParaNLRU(64, 64)
h = torch.zeros(2, 64)
for t in range(16):
    h = cell.step(h, torch.randn(2, 64))
```

## Agreement

`verify_agreement` on `ParaRNN(ParaNLRU(...))` ([numerics contract](../numerics-contract.md)).
