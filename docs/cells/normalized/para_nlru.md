# ParaNLRU

**Fidelity:** research-variant  
**Sources:** De et al., *Griffin* (2024) RG-LRU (linear-in-\(h\) core)  
**Upstream:** Griffin / RecurrentGemma RG-LRU  
**Code:** `pararnn.cells.para_nlru.ParaNLRU`

## Diff (upstream → Para-native)

| Component | Upstream (RG-LRU) | Para-native | Kind | Rationale |
|-----------|-------------------|-------------|------|-----------|
| Candidate | Linear in \(h\) | \(\tanh(c_x + u\odot h)\) | para | Nonlinear brick; diag \(J\) kept |
| Gate | Input-only \(a(x)\) | Same \(\sigma(a_x)\) | author | 1∶1 gate side |
| Pack | — | `(a_x, c_x)` | para | `ax,cx = wx.chunk(2)` |
| Name | RG-LRU | NLRU | para | Marks nonlinear slot |
| Clamp | — | `max_recurrent_norm=0.5` on \(u\) | numerics | App. C.1 |

## Strict Spec Contract

```yaml
Cell: ParaNLRU
Fidelity: research-variant
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, d_h]"; layout_train: "[B, T, d_h]"
Shapes:
  h,a,n,u: "[B, d_h]" / u: "[d_h]"
  jac: "[B, T, d_h]"
Broadcast:
  - "a, n, u ⊙ h all elementwise on d_h"
  - "a = σ(a_x) with a_x from x only"
Parameters:
  W_x: "Linear(d_in, 2*d_h, bias=True)"
  W_x_chunk: "[0]=a_x, [1]=c_x"
  u: "Parameter(d_h)"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    u: xavier_gaussian_vec
Constants:
  max_recurrent_norm: 0.5
Jacobian:
  structure: diag
  storage: "(..., d_h)"
  formula: "J = a + (1-a)*(1-n**2)*u; a=a(x) input-only"
Newton:
  residual: "F = f(h_prev,x) - h"
  linear_recurrence: "delta_t = j_t ⊙ delta_{t-1} + F_t"
  solve: "associative scan_diag / fused newton_nlru"
  default_K_pin: 3
  fused_op: "kernels.newton_nlru"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape |
|--------|-------|
| \(h,a,n\) | `(B, d_h)` |
| \(u\) | `(d_h,)` |
| \(j\) | `(B, T, d_h)` |

## Recurrence

\[
\begin{aligned}
a_t &= \sigma(W_a x_t + b_a), \\
h_t &= a_t \odot h_{t-1}
  + (1-a_t)\odot\tanh(W_c x_t + b_c + u \odot h_{t-1}).
\end{aligned}
\]

## Jacobian class

**Dispatch:** `jac_structure='diag'`.  
**Storage:** `(..., d_h)`.  
**Operator** (\(n=\tanh(c_x+u\odot h)\), \(n'=1-n^2\)):

\[
J = a + (1-a)\,n'\,u.
\]

Gate \(a=\sigma(a_x)\) is **input-only** (frozen w.r.t. \(h\)). Candidate tanh
derivative included.

## Parallel path

Residual \(F=f-H\). \(\delta_t=j_t\odot\delta_{t-1}+F_t\).
**Solve:** associative `scan_diag` / fused `kernels.newton_nlru`. App. A guess.
Typical \(K{=}3\). [Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn

d_in, d_h = 64, 64
W_x = nn.Linear(d_in, 2 * d_h)
u = torch.randn(d_h) * 0.1  # (d_h,)
u = u.clamp(-0.5, 0.5)

def step(h, x):
    # h: (B, d_h); x: (B, d_in)
    ax, cx = W_x(x).chunk(2, dim=-1)  # each (B, d_h)
    a = torch.sigmoid(ax)
    n = torch.tanh(cx + u * h)
    return a * h + (1.0 - a) * n

h = torch.zeros(2, d_h)
for _ in range(16):
    h = step(h, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaNLRU, ParaRNN, verify_agreement

m = ParaRNN(ParaNLRU(64, 64))
x = torch.randn(2, 64, 64)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 `atol=1e-4` ([numerics contract](../../core/numerics_contract.md)).
