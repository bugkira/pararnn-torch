# Audit matrix

One-page map from **upstream** to **Para-native** for every shipped cell.
Full Diff + YAML live on each cell page; this table is the index.

Families follow **state / Jacobian** (nav): Classic gated · Normalized &
resonant · Matrix & associative · Continuous & liquid.
Shared agreement τ: [numerics contract](../core/numerics_contract.md)  
(`FP32 atol=1e-4`, lowp `1e-2`). Spec template: [spec template](spec_template.md).  
Diff **Kind**: `author` | `para` | `numerics` — see [fidelity](fidelity.md).

| Cell | Family | Fidelity | Upstream | Biggest `para` / `numerics` | Pack order (chunk) | Spec |
|------|--------|----------|----------|-----------------------------|--------------------|------|
| ParaGRU | classic | paper-faithful (diag) | Danieli / Cho GRU | `para`: diag \(a_*\); `numerics`: clamp 0.5 | `(z, r, n)` | [para_gru](classic/para_gru.md) |
| ParaLSTM | classic | paper-faithful | Danieli CIFG peephole | `para`: CIFG 3-way + diag \(A,C\); `numerics`: bias 0 | `(f, z, o)` | [para_lstm](classic/para_lstm.md) |
| ParaNLRU | normalized | research-variant | Griffin RG-LRU | `para`: nonlinear candidate \(u\odot h\) | `(a, c)` | [para_nlru](normalized/para_nlru.md) |
| ParaSLSTM | normalized | paper-faithful (diag) | xLSTM sLSTM lineage | `para`: channelwise \(R\); `numerics`: `eps=1e-6`, tie 0.5 | `(i, f, z, o)` | [para_slstm](normalized/para_slstm.md) |
| ParaHopfield | matrix | paper-core | Ramsauer Modern Hopfield | `para`: patterns from \(x\); `numerics`: \(d_h\le32\) | flat \(K\|V\) (`2 d_h²`) | [para_hopfield](matrix/para_hopfield.md) |
| ParaM2RNN | matrix | research-variant / core | Mishra M²RNN | `para`: matrix core only; `numerics`: \(W{=}I\) init | `k \| v \| f_logit` | [para_m2rnn](matrix/para_m2rnn.md) |
| ParaRWKV7 | matrix | paper-core | Peng RWKV-7 Goose | `para`: cell = monoid transition; Newton → scan | `(w,a,κ,v,k,r)` | [para_rwkv7](matrix/para_rwkv7.md) |
| ParaTitans | matrix | research-variant | Behrouz Titans | `para`: shallow \(L{=}1\) diag memory | surprise + polish pack | [para_titans](matrix/para_titans.md) |
| ParaCfC | continuous | research-variant | Hasani / ncps CfC | `para`: \(h\)↔candidate; \(W_b(x)\); `4 d_h` pack | `(f_pre, c_x, Δt, b)` | [para_cfc](continuous/para_cfc.md) |

## How to audit

1. Open the cell’s **Diff** (check **Kind**) + **Strict Spec Contract**.
2. Reimplement `step` from the YAML + Reproduce snippet alone.
3. `verify_agreement` at the YAML τ.
4. For task-level upstream comparison, train separate stacks (weight bridge =
   distinct) — never silent `load_state_dict` across architectures.
