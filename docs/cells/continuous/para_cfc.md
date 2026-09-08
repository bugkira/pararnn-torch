# ParaCfC

**Fidelity:** research-variant  
**Sources:** Hasani et al., *Closed-form continuous-time neural networks*, Nat. Mach. Intell. 2022 / arXiv:2106.13898, eq. (10), Fig. 4  
**Upstream:** `ncps.torch.CfC` / `CfCCell` (Apache-2.0)  
**Code:** `pararnn.cells.para_cfc.ParaCfC`

## Diff (upstream → Para-native)

| Component | Upstream (Hasani / ncps default) | Para-native | Kind | Rationale |
|-----------|----------------------------------|-------------|------|-----------|
| Scope | Backbone + heads `ff1`/`ff2`/`time_*` (or paper \(f,g,h\)) | Recurrence brick only | para | Outer MLP is the caller’s stack |
| Update | Mix two attractors \(\mathrm{ff1},\mathrm{ff2}\) | Mix \(h_{t-1}\) with candidate \(n_t\) | para | Continuous-GRU form; keeps channelwise \(J\) |
| Time gate | \(\sigma(t_a\cdot\mathrm{ts}+t_b)\), \(t_{a,b}\) on \(\mathrm{concat}(x,h)\) | \(\sigma(-(\mathrm{softplus}(f_{\mathrm{pre}})\,\Delta t + W_b(x)))\) | para | \(\Delta t\) as data channel; \(W_b\) features-only → diag \(J\) |
| Rate | Softplus optional / paper \(\sigma(-f\,t)\) | `softplus` then \(\times\Delta t\) | para | Positive liquid rate |
| Pack | Separate linears / concat backbone | `W_x`: `(f_pre, c_x)`; `project_wx`: `(f_pre, c_x, Δt, b)` = `4 d_h` | para | Single fused GEMM for Newton |
| Init | ncps defaults | `W_b.bias=-3`, `W_b.weight=0`; Xavier \(u\); Kaiming `W_x` | numerics | \(a\approx 0.95\) as \(\Delta t\to 0\) |
| Clamp | — | `max_recurrent_norm=0.5` on \(u\)/\(v\) | numerics | App. C.1 |
| Weights | ncps state dict | Separate | para | Architecturally distinct — task oracle, separate trainings |

Optional `gate_mix='diag_h'`: rate uses \(f+v\odot h\); fused path stays on `input` only.

## Strict Spec Contract

```yaml
Cell: ParaCfC
Fidelity: research-variant
Inputs:
  x_step: "[B, d_in]"          # d_in = feature_size + 1
  x_train: "[B, T, d_in]"
  features: "x[..., :-1]"      # [B, feature_size]
  dt: "x[..., -1:].clamp_min(1e-4)"  # [B, 1]
State:
  layout_step: "[B, d_h]"; layout_train: "[B, T, d_h]"
Shapes:
  h,a,n,u,b,f_pre,c_x: "[B, d_h]"
  dt_exp: "[B, d_h]"           # dt.expand_as(f_pre) or broadcast [B,1]
  project_wx: "[B, 4*d_h]"
  jac: "[B, T, d_h]"
Broadcast:
  - "dt [B,1] broadcasts over d_h in softplus(f)*dt + b"
  - "gate_mix=input: a independent of h"
Parameters:
  W_x: "Linear(feature_size, 2*d_h, bias=True)"
  W_x_chunk: "[0]=f_pre, [1]=c_x"
  W_b: "Linear(feature_size, d_h, bias=True)"
  project_wx: "cat(W_x(feat), dt.expand(..., d_h), W_b(feat)) -> [..., 4*d_h]"
  project_wx_chunk: "[0]=f_pre, [1]=c_x, [2]=dt, [3]=b"
  u: "Parameter(d_h)"
  v: "Parameter(d_h) if gate_mix=diag_h else None"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    W_b.weight: zeros
    W_b.bias: -3.0
    u: xavier_gaussian_vec
    v: xavier_gaussian_vec
Constants:
  dt_clamp_min: 1.0e-4
  max_recurrent_norm: 0.5
  gate_mix: input
Jacobian:
  structure: diag
  storage: "(..., d_h)"
  formula: "gate_mix=input: J=a+(1-a)*(1-n**2)*u; diag_h adds da/dh*(h-n)"
Newton:
  residual: "F = f(h_prev,x) - h"
  linear_recurrence: "delta_t = j_t ⊙ delta_{t-1} + F_t"
  solve: "associative scan_diag / pararnn::newton_cfc_fused"
  default_K_pin: 3
  K_star: "H1 O(1), recipe {1: 3}"
  fused_op: "pararnn::newton_cfc_fused"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
  residual_fuse: { warn: 1.0e-3, fail: 1.0 }
Weight_bridge: distinct
Lab_tests:
  - tests/numerics/test_cfc.py
  - tests/numerics/test_cfc_fidelity.py
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| \(x\) | `(B, feature_size+1)` | last channel \(\Delta t\) |
| \(\Delta t\) | `(B, 1)` → broadcast `(B, d_h)` | `clamp_min(1e-4)` |
| \(h,a,n,b,u\) | `(B, d_h)` / \(u\): `(d_h,)` | |
| `project_wx` | `(B, 4*d_h)` | `(f_pre, c_x, Δt, b)` |

## Recurrence

\[
\begin{aligned}
b_t &= W_b(x_t), \\
a_t &= \sigma\bigl(-(\mathrm{softplus}(f_{\mathrm{pre}}(x_t))\,\Delta t_t + b_t)\bigr), \\
n_t &= \tanh\bigl(c_x(x_t) + u \odot h_{t-1}\bigr), \\
h_t &= a_t \odot h_{t-1} + (1-a_t)\odot n_t.
\end{aligned}
\]

`gate_mix='diag_h'`: replace \(f_{\mathrm{pre}}\) inside softplus with \(f_{\mathrm{pre}}+v\odot h_{t-1}\).

## Jacobian class

**Dispatch:** `jac_structure='diag'`.  
**Storage:** `(..., d_h)`.

**`gate_mix='input'`** (\(a=a(x)\) only; fused path):

\[
J_t = a_t + (1-a_t)\odot(1-n_t^{\odot 2})\odot u.
\]

**`gate_mix='diag_h'`** (eager analytic; fused refuses this path):

\[
\frac{\partial a}{\partial h} = a(1-a)\,(-\Delta t)\,\sigma(z)\,v,\qquad
J \mathrel{+}= \frac{\partial a}{\partial h}\odot(h-n),
\]

with \(z=f_{\mathrm{pre}}+v\odot h\).

## Parallel path

Residual \(F=f-H\). \(\delta_t=j_t\odot\delta_{t-1}+F_t\).
**Solve:** associative diag scan / `pararnn::newton_cfc_fused` (`gate_mix='input'`).
App. A guess \(h_t^{(0)}=f(0,x_t)\).  
`NewtonConfig(max_iters=None)` → `cfc_auto_newton_iters` (**H1**, recipe `{1: 3}`).
[Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn
import torch.nn.functional as F

feat_size, d_h = 8, 16
W_x = nn.Linear(feat_size, 2 * d_h)
W_b = nn.Linear(feat_size, d_h)
nn.init.zeros_(W_b.weight)
nn.init.constant_(W_b.bias, -3.0)
u = torch.randn(d_h) * 0.1  # (d_h,)
u = u.clamp(-0.5, 0.5)

def step(h, x):
    # h: (B, d_h); x: (B, feat_size+1)
    feat, dt = x[..., :-1], x[..., -1:].clamp_min(1e-4)  # (B,F), (B,1)
    f_pre, c_x = W_x(feat).chunk(2, dim=-1)              # (B, d_h)
    b = W_b(feat)                                        # (B, d_h)
    a = torch.sigmoid(-(F.softplus(f_pre) * dt + b))     # dt broadcasts
    n = torch.tanh(c_x + u * h)
    return a * h + (1.0 - a) * n

h = torch.zeros(2, d_h)
feat = torch.randn(2, feat_size)
dt = 0.05 + torch.rand(2, 1)
h = step(h, torch.cat((feat, dt), dim=-1))
```

## Agreement check

```python
import torch
from pararnn import ParaCfC, ParaRNN, verify_agreement

m = ParaRNN(ParaCfC(9, 16))
feat = torch.randn(2, 64, 8)
dt = 0.05 + torch.rand(2, 64, 1)
x = torch.cat((feat, dt), dim=-1)  # (B, T, feature_size+1)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 within `atol=1e-4` at recommended \(K\) ([numerics contract](../../core/numerics_contract.md)).  
Task oracle vs ncps (separate weights): lab `train_physionet_ncps_baseline.py`.
