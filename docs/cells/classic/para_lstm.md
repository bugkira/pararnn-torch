# ParaLSTM

**Fidelity:** paper-faithful  
**Sources:** Danieli et al., arXiv:2510.21450, eq. 3.1b, 3.3; CIFG peephole (Greff et al. 2017)  
**Upstream:** ParaRNN paper CIFG diag-\(A,C\) (related: `torch.nn.LSTM` four-gate)  
**Code:** `pararnn.cells.para_lstm.ParaLSTM`

## Diff (upstream → Para-native)

| Component | Upstream | Para-native | Kind | Rationale |
|-----------|----------|-------------|------|-----------|
| Gates | Often 4-way `(i,f,g,o)` (`nn.LSTM`) | CIFG 3-way `(f,z,o)` | author | Paper CIFG; input gate \(=1-f\) |
| Recurrent | Full matrices | Diag \(a_f,a_z,a_o\) + peepholes \(c_f,c_o\) | para | Channelwise / block2 Newton |
| State | \(c,h\) | Slots `LSTM_CELL=0`, `LSTM_HIDDEN=1` | para | `pararnn.layout` |
| Bias | Sometimes forget `+1` (Jozefowicz) | `W_x.bias=0` | numerics | App. C.1 LM recipe |
| Pack | Vendor-specific | `W_x` → `(f_x, z_x, o_x)` | para | Fused GEMM |
| Clamp | — | `max_recurrent_norm=0.5` | numerics | App. C.1 |

## Strict Spec Contract

```yaml
Cell: ParaLSTM
Fidelity: paper-faithful
Inputs:
  x_step: "[B, d_in]"; x_train: "[B, T, d_in]"
State:
  layout_step: "[B, 2, d_h]"; layout_train: "[B, T, 2, d_h]"
  slots: { LSTM_CELL: 0, LSTM_HIDDEN: 1 }
Shapes:
  c,h,f,z,o: "[B, d_h]"
  a_*, c_*: "[d_h]"
  jac: "[B, T, 2, 2, d_h]"   # [..., out, in, d]; out/in in {0:c, 1:h}
  residual: "[B, T, 2, d_h]"
Broadcast:
  - "all gate products elementwise on d_h; 2×2 couples (c,h) per channel"
Parameters:
  W_x: "Linear(d_in, 3*d_h, bias=True)"
  W_x_chunk: "[0]=f_x, [1]=z_x, [2]=o_x"
  recurrent: "a_f, a_z, a_o, c_f, c_o  # Parameter(d_h) each"
  init:
    W_x.weight: kaiming_uniform
    W_x.bias: zeros
    a_*, c_*: xavier_gaussian_vec
Constants:
  max_recurrent_norm: 0.5
Jacobian:
  structure: block2
  storage: "(..., 2, 2, d_h)  # jac[..., out, in, :]; 0=c, 1=h"
  formula: "J_cc=f+(c-z)*f'*c_f; J_ch=(c-z)*f'*a_f+(1-f)*z'*a_z; J_hc=(h_act*o'*c_o+o*h_act')*J_cc; J_hh=h_act*o'*(a_o+c_o*J_ch)+o*h_act'*J_ch"
Newton:
  residual: "F = f(state_prev,x) - state"
  linear_recurrence: "delta_t = J_t delta_{t-1} + F_t  # per-channel 2×2"
  solve: "associative scan_block2 (2×2 mul/matvec compose); NOT torch.linalg.solve"
  default_K_pin: 3
  fused_op: "kernels.newton_lstm"
Agreement:
  FP32: { atol: 1.0e-4, rtol: 0.0 }
  BF16: { atol: 1.0e-2, rtol: 0.0 }
Weight_bridge: distinct
```

## Shapes & broadcasting

| Symbol | Shape | Notes |
|--------|-------|-------|
| state | `(B, 2, d_h)` | slot 0 = \(c\), 1 = \(h\) |
| \(f,z,o,c,h\) | `(B, d_h)` | channelwise |
| \(a_*,c_*\) | `(d_h,)` | |
| \(J\) | `(B, T, 2, 2, d_h)` | `jac[..., out, in, :]` |

## Recurrence

\[
\begin{aligned}
f_t &= \sigma(a_f \odot h_{t-1} + c_f \odot c_{t-1} + f_x), \\
z_t &= \tanh(a_z \odot h_{t-1} + z_x), \\
c_t &= f_t \odot c_{t-1} + (1-f_t)\odot z_t, \\
o_t &= \sigma(a_o \odot h_{t-1} + c_o \odot c_t + o_x), \\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
\]

## Jacobian class

**Dispatch:** `jac_structure='block2'` (inferred from 2-slot state).  
**Storage:** `(..., 2, 2, d_h)` with `jac[..., out, in, :]`, slots \(0{=}c\), \(1{=}h\).  
**Operator** (primes \(f'=f(1-f)\), \(z'=1-z^2\), \(o'=o(1-o)\), \(h_{\mathrm{act}}'=1-\tanh(c)^2\)):

\[
\begin{aligned}
J_{cc} &= f + (c-z)\,f'\,c_f, \\
J_{ch} &= (c-z)\,f'\,a_f + (1-f)\,z'\,a_z, \\
J_{hc} &= \bigl(h_{\mathrm{act}}\,o'\,c_o + o\,h_{\mathrm{act}}'\bigr)\,J_{cc}, \\
J_{hh} &= h_{\mathrm{act}}\,o'\,(a_o + c_o\,J_{ch}) + o\,h_{\mathrm{act}}'\,J_{ch}.
\end{aligned}
\]

Full forget / candidate / output + peephole derivatives (eq. 3.2b).

## Parallel path

Residual \(F=f-H\). \(\delta_t = J_t\delta_{t-1}+F_t\) with per-channel \(2\times2\).
**Solve:** associative `scan_block2` (compose with closed-form \(2\times2\) mul /
matvec). The scan **does not** invert each \(2\times2\) in isolation as the
whole answer; it scans the time monoid. App. A guess. Typical \(K{=}3\).
`kernels.newton_lstm`. [Newton + scan](../../core/newton_scan.md).

## Reproduce (sequential)

```python
import torch
from torch import nn

d_in, d_h = 32, 32
W_x = nn.Linear(d_in, 3 * d_h)
a_f, a_z, a_o = (torch.randn(d_h) * 0.1 for _ in range(3))
c_f, c_o = (torch.randn(d_h) * 0.1 for _ in range(2))
cap = 0.5
a_f, a_z, a_o, c_f, c_o = (t.clamp(-cap, cap) for t in (a_f, a_z, a_o, c_f, c_o))

def step(state, x):
    # state: (B, 2, d_h); x: (B, d_in)
    c, h = state[..., 0, :], state[..., 1, :]  # each (B, d_h)
    fx, zx, ox = W_x(x).chunk(3, dim=-1)
    f = torch.sigmoid(a_f * h + c_f * c + fx)
    z = torch.tanh(a_z * h + zx)
    c_new = f * c + (1.0 - f) * z
    o = torch.sigmoid(a_o * h + c_o * c_new + ox)
    h_new = o * torch.tanh(c_new)
    return torch.stack((c_new, h_new), dim=-2)

state = torch.zeros(2, 2, d_h)
state = step(state, torch.randn(2, d_in))
```

## Agreement check

```python
import torch
from pararnn import ParaLSTM, ParaRNN, verify_agreement

m = ParaRNN(ParaLSTM(32, 32))
x = torch.randn(2, 64, 32)
res = verify_agreement(m, x, atol=1e-4)
assert res.ok, f"Agreement failed: max_abs={res.max_abs}"
```

FP32 within `atol=1e-4` ([numerics contract](../../core/numerics_contract.md)).
