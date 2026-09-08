# ParaTitans

**Fidelity:** research-variant  
**Sources:** Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663  
**Upstream:** Titans surprise-GD neural memory (deep MLP \(M\) in the paper)  
**Code:** `pararnn.cells.para_titans.ParaTitans`

## Diff (upstream → Para-native)

| Component | Upstream (Titans) | Para-native | Kind | Rationale |
|-----------|-------------------|-------------|------|-----------|
| Memory | Deep multi-layer MLP \(M\) | Channel vector \(h\) (diag associative map) | para | Diag-\(J\) Newton brick |
| Surprise | Grad of associative loss | \(g=(h\odot k-v)\odot k\) | author | Elementwise assoc. loss |
| Update | GD step + deeper polish | Forget + \(\theta\)-scaled \(g\) + tanh polish | para | Shallow \(L{=}1\) |
| Pack | — | `(α_pre, θ_pre, k, v, n_pre)` = `5 d_h` | para | Single `W_x` |
| \(\theta\) bias | Learned LR | `W_x.bias[θ]=-1` | numerics | softplus modest at init |
| Clamp | — | `max_recurrent_norm=0.5` on \(u,r\) | numerics | App. C.1 |

## Strict Spec Contract

```yaml
Cell: ParaTitans
Fidelity: research-variant
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, d_h]"; layout_train: "[B, T, d_h]"
Shapes:
  h,alpha,theta,k,v,n,g: "[B, d_h]"
  u,r: "[d_h]"
  jac: "[B, T, d_h]"
  W_x.out: "[B, 5*d_h]"
Broadcast:
  - "all surprise / polish terms elementwise on d_h"
  - "α,θ,k,v from x only; n depends on h via r"
Parameters:
  W_x: "Linear(d_in, 5*d_h, bias=True)"
  W_x_chunk: "[0]=α_pre, [1]=θ_pre, [2]=k, [3]=v, [4]=n_pre"
  u, r: "Parameter(d_h) each"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    W_x.bias[d_h:2*d_h]: -1.0
    u, r: xavier_gaussian_vec
Constants:
  max_recurrent_norm: 0.5
Jacobian:
  structure: diag
  storage: "(..., d_h)"
  formula: "J=(1-α)-θ⊙k²+u⊙(1-n²)⊙r; α,θ,k,v from x"
Newton:
  residual: "F = f(h_prev,x) - h"
  linear_recurrence: "delta_t = j_t ⊙ delta_{t-1} + F_t"
  solve: "associative scan_diag / fused newton_titans"
  default_K_pin: 3
  fused_op: "kernels.newton_titans"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape |
|--------|-------|
| \(h,\alpha,\theta,k,v,n,g\) | `(B, d_h)` |
| \(u,r\) | `(d_h,)` |
| \(j\) | `(B, T, d_h)` |

## Recurrence

\[
\begin{aligned}
\alpha_t &= \sigma(\alpha^{\mathrm{pre}}_t),\quad
\theta_t = \mathrm{softplus}(\theta^{\mathrm{pre}}_t), \\
g_t &= (h_{t-1}\odot k_t - v_t)\odot k_t, \\
h^{\mathrm{lin}}_t &= (1-\alpha_t)\odot h_{t-1} - \theta_t\odot g_t, \\
h_t &= h^{\mathrm{lin}}_t + u\odot\tanh(n^{\mathrm{pre}}_t + r\odot h_{t-1}).
\end{aligned}
\]

## Jacobian class

**Dispatch:** `jac_structure='diag'`.  
**Storage:** `(..., d_h)`.  
**Operator** (\(n=\tanh(n_{\mathrm{pre}}+r\odot h)\)):

\[
J = (1-\alpha) - \theta\odot k^{\odot 2} + u\odot(1-n^{\odot 2})\odot r.
\]

\(\alpha,\theta,k,v\) are **input-only**; surprise and tanh polish contribute the
\(h\)-dependent terms above.

## Parallel path

Residual \(F=f-H\). \(\delta_t=j_t\odot\delta_{t-1}+F_t\).
**Solve:** associative `scan_diag` / fused `kernels.newton_titans`. App. A guess.
Typical \(K{=}3\). [Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn
import torch.nn.functional as F

d_in, d_h = 64, 64
W_x = nn.Linear(d_in, 5 * d_h)
with torch.no_grad():
    W_x.bias[d_h : 2 * d_h].fill_(-1.0)
u = torch.randn(d_h) * 0.1  # (d_h,)
r = torch.randn(d_h) * 0.1
u, r = u.clamp(-0.5, 0.5), r.clamp(-0.5, 0.5)

def step(h, x):
    # h: (B, d_h); x: (B, d_in)
    a_pre, th_pre, k, v, n_pre = W_x(x).chunk(5, dim=-1)  # each (B, d_h)
    alpha = torch.sigmoid(a_pre)
    theta = F.softplus(th_pre)
    g = (h * k - v) * k
    h_lin = (1.0 - alpha) * h - theta * g
    n = torch.tanh(n_pre + r * h)
    return h_lin + u * n

h = torch.zeros(2, d_h)
h = step(h, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaTitans, ParaRNN, verify_agreement

m = ParaRNN(ParaTitans(64, 64))
x = torch.randn(2, 64, 64)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 `atol=1e-4` ([numerics contract](../../core/numerics_contract.md)).
