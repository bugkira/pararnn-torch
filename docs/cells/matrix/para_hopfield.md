# ParaHopfield

**Fidelity:** paper-core  
**Sources:** Ramsauer et al., *Hopfield Networks is All You Need* (2021) style one-step update  
**Upstream:** Modern Hopfield / soft-attention recurrence  
**Code:** `pararnn.cells.para_hopfield.ParaHopfield`

## Diff (upstream → Para-native)

| Component | Upstream | Para-native | Kind | Rationale |
|-----------|----------|-------------|------|-----------|
| Patterns | Stored / learned memories | \(K_t,V_t\) from \(x_t\) each step | para | Input-conditioned patterns |
| Update | Softmax attention hop | \(h'=V\,\mathrm{softmax}(\beta K h)\) | author | Core one-step math |
| Stack | Full Hopfield layer / energy | Cell = one Newton slot | para | Outer stack owns depth |
| Width | Often large | Cap warn \(d_h\le 32\) | numerics | Dense \(J\) is \(O(d_h^3)\) |
| \(\beta\) | Inverse temperature | Default \(1/\sqrt{d_h}\) | numerics | Attention scale |

## Strict Spec Contract

```yaml
Cell: ParaHopfield
Fidelity: paper-core
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, d_h]"; layout_train: "[B, T, d_h]"
Shapes:
  h,a: "[B, d_h]"
  K,V: "[B, d_h, d_h]"       # from reshape W_x(x)
  jac: "[B, T, d_h, d_h]"    # [..., out, in]
  W_x.out: "[B, 2*d_h*d_h]"
Broadcast:
  - "logits = beta * (K @ h); softmax over last dim d_h"
  - "h_new = V @ a"
Parameters:
  W_x: "Linear(d_in, 2*d_h*d_h, bias=True)"
  W_x_layout: "reshape (..., 2, d_h, d_h) -> K=[...,0], V=[...,1]"
  beta: "1/sqrt(d_h) default"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
Constants:
  d_h_dense_cap: 32
Jacobian:
  structure: dense
  storage: "(..., d_h, d_h)"
  formula: "J = V @ (diag(a)-a a^T) @ (beta * K); a=softmax(beta K h)"
Newton:
  residual: "F = f(h_prev,x) - h"
  linear_recurrence: "delta_t = J_t @ delta_{t-1} + F_t"
  solve: "associative scan_dense (bmm monoid); NOT linalg.solve on Td×Td"
  default_K_pin: null
  K_star: "hopfield_auto_newton_iters; short-T K* in {1,2}"
  fused_op: "scan_dense"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| \(h,a\) | `(B, d_h)` | |
| \(K,V\) | `(B, d_h, d_h)` | halves of `W_x(x).reshape(..., 2, d_h, d_h)` |
| \(J\) | `(B, T, d_h, d_h)` | `[..., out, in]` |
| \(\beta\) | scalar | default \(1/\sqrt{d_h}\) |

## Recurrence

\[
a_t = \mathrm{softmax}(\beta K_t h_{t-1}),\qquad
h_t = V_t a_t,
\]

with \(K_t,V_t\in\mathbb{R}^{d_h\times d_h}\) the two halves of \(\mathrm{reshape}(W_x x_t)\).

## Jacobian class

**Dispatch:** `jac_structure='dense'`.  
**Storage:** `(..., d_h, d_h)`.  
**Operator:**

\[
J = V\bigl(\mathrm{diag}(a)-a a^\top\bigr)\,(\beta K).
\]

Softmax couples channels. Patterns \(K,V\) are input-only.

## Parallel path

Residual \(F=f-H\). \(\delta_t = J_t\delta_{t-1}+F_t\) with dense \(J\).
**Solve:** associative `scan_dense` (`bmm` compose of \((J,r)\)). Prefer
`NewtonConfig(max_iters=None)`. Keep \(d_h\le 32\).
[Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

d_in, d_h = 16, 16
beta = 1.0 / math.sqrt(d_h)
W_x = nn.Linear(d_in, 2 * d_h * d_h)

def step(h, x):
    # h: (B, d_h); x: (B, d_in)
    kv = W_x(x).reshape(*x.shape[:-1], 2, d_h, d_h)  # (B, 2, d, d)
    K, V = kv[..., 0, :, :], kv[..., 1, :, :]          # each (B, d, d)
    logits = beta * torch.matmul(K, h.unsqueeze(-1)).squeeze(-1)  # (B, d)
    a = F.softmax(logits, dim=-1)
    return torch.matmul(V, a.unsqueeze(-1)).squeeze(-1)

h = torch.zeros(2, d_h)
h = step(h, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaHopfield, ParaRNN, verify_agreement

m = ParaRNN(ParaHopfield(16, 16))
x = torch.randn(2, 32, 16)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

([numerics contract](../../core/numerics_contract.md)).
