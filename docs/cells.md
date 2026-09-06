# Cell catalog

Snippets for each cell in the Models table. README keeps block → CausalLM →
`ParaRNN`; this page is the zoo.

Shared pattern: wrap with `ParaRNN` or call `newton_apply` / `sequential_apply`.
`.train()` → parallel Newton · `.eval()` → sequential `step` · CUDA `T=1` →
`decode_step`. Force either path with `solver='newton'|'sequential'`.

```python
import torch
from pararnn import NewtonConfig, ParaRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## ParaSLSTM (xLSTM-style)

Main fused path: `mix='diag'`.

```python
from pararnn import ParaSLSTM

slstm = ParaRNN(ParaSLSTM(64, 64, mix="diag"), device=device)
y = slstm(torch.randn(4, 128, 64, device=device))
```

Stacking / API notes: [`xlstm.md`](xlstm.md).

## ParaGRU / ParaLSTM (Dreamer-style block GRU)

Block-diagonal A_*; CUDA factorized Newton. LayerNorm stays outside the cell.

```python
from pararnn import ParaGRU

rssm_h = ParaRNN(ParaGRU(512, 512, mix="head", n_heads=8), device=device)
y = rssm_h(torch.randn(4, 64, 512, device=device))
```

Smoke: [`examples/rssm_recurrent.py`](../examples/rssm_recurrent.py). Head fused
medians (`scripts/bench_gru_head.py`, 2080 Ti, K=3):

| setup | `d_head` | Newton | fwd+bwd |
|------:|---------:|-------:|-------:|
| `B=4`, `T=128` | 64 | 2.4 | — |
| `B=1`, `T=4096` | 64 | 51 | 83 |
| `B=1`, `T=4096` | 96 | 220 | 293 |

## ParaM2RNN (research)

```python
from pararnn import ParaM2RNN, newton_apply

m2 = ParaM2RNN(d_in=32, k_dim=16, v_dim=16, device=device)
x = 0.15 * torch.randn(2, 128, 32, device=device)
h_par = newton_apply(m2, x, NewtonConfig(max_iters=8, residual_atol=1e-5))
```

State `(B, T, K, V)`. Critical depth: `scripts/bench_m2rnn_k_scale.py`.

## ParaNLRU

Griffin / RG-LRU-style nonlinear slot.

```python
from pararnn import ParaNLRU

nlru = ParaRNN(ParaNLRU(256, 256), device=device)
y = nlru(torch.randn(4, 128, 256, device=device))
```

## ParaCfC

Liquid CfC; Δt is the last channel of `x`.

```python
from pararnn import ParaCfC

cfc = ParaRNN(ParaCfC(257, 256), device=device)  # 256 features + Δt
feat = torch.randn(4, 128, 256, device=device)
dt = 0.05 + torch.rand(4, 128, 1, device=device)
y = cfc(torch.cat((feat, dt), dim=-1))
```

## ParaHopfield

Modern Hopfield; keep d_h ≤ 32 for the dense Jacobian path.
VRAM / long-T: [`oom-cookbook.md`](oom-cookbook.md).

```python
from pararnn import ParaHopfield

hop = ParaRNN(
    ParaHopfield(64, 8),
    config=NewtonConfig(max_iters=None, jac_structure="dense"),  # K*(T) auto
    device=device,
)
y = hop(torch.randn(4, 128, 64, device=device))
```

Pin depth with `max_iters=int` or `newton_iters_by_t={64: 2, 1024: 3, …}`.

## ParaRWKV7

RWKV-7 Goose; linear monoid, K*=0.

```python
from pararnn import ParaRWKV7, newton_apply

cell = ParaRWKV7(d_in=64, n_heads=4, d_head=16, device=device)
x = torch.randn(4, 128, 64, device=device)
s = newton_apply(cell, x)       # redirects to linear (G,U) scan
y = cell.scan_apply(x)          # (B, T, n_heads*d_head) readout
```

On 12 GiB cards, wall-clock at T ≳ 64k prefers slim
`n_heads=1, d_head=16` (state is `(B,T,H,D,D)`). See
[`oom-cookbook.md`](oom-cookbook.md).

## ParaTitans

Shallow L=1 surprise-GD memory. Deep multi-layer MLP memory is parked.

```python
from pararnn import ParaTitans

titans = ParaRNN(ParaTitans(256, 256), device=device)
y = titans(torch.randn(4, 128, 256, device=device))
```
