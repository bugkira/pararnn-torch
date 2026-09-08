# Cell catalog

The zoo of Newton cells, grouped by **state layout and Jacobian class**
(how Newton+scan pays). Specs lead with **Diff** (Kind: `author` | `para` |
`numerics`), **Shapes & broadcasting**, a **Strict Spec Contract** (YAML), and
a named Newton **solve** primitive. Reproduce snippets reimplement `step` from
pack order. API call-sites: [api.md](api.md).

## By family

### Classic gated

Vector / CIFG slots; \(\sigma\) / \(\tanh\) gates. Jacobian `diag` or `block2`.

| Cell | Spec | Fidelity |
|------|------|----------|
| [ParaGRU](classic/para_gru.md) | yes | paper-faithful (`mix='diag'`) |
| [ParaLSTM](classic/para_lstm.md) | yes | paper-faithful (CIFG) |

### Normalized & resonant

Vector state with exp / normalizer / nonlinear LRU poles. Jacobian `block4` or
`diag`.

| Cell | Spec | Fidelity |
|------|------|----------|
| [ParaNLRU](normalized/para_nlru.md) | yes | research-variant (vs Griffin RG-LRU) |
| [ParaSLSTM](normalized/para_slstm.md) | yes | paper-faithful (`mix='diag'`) |

### Matrix & associative memory

Matrix / dense-associative / fast-weight family. Jacobian `m2rnn`, `dense`,
`rwkv7`, or (Titans) shallow associative slot with `diag` vector state.

| Cell | Spec | Fidelity |
|------|------|----------|
| [ParaHopfield](matrix/para_hopfield.md) | yes | paper-core |
| [ParaM2RNN](matrix/para_m2rnn.md) | yes | research-variant / paper-core |
| [ParaRWKV7](matrix/para_rwkv7.md) | yes | paper-core |
| [ParaTitans](matrix/para_titans.md) | yes | research-variant |

### Continuous & liquid (ODE)

Explicit \(\Delta t\) in the input; liquid rate. Jacobian `diag`.

| Cell | Spec | Fidelity |
|------|------|----------|
| [ParaCfC](continuous/para_cfc.md) | yes | research-variant (vs Hasani / ncps) |

## Alphabetical

[ParaCfC](continuous/para_cfc.md) ·
[ParaGRU](classic/para_gru.md) ·
[ParaHopfield](matrix/para_hopfield.md) ·
[ParaLSTM](classic/para_lstm.md) ·
[ParaM2RNN](matrix/para_m2rnn.md) ·
[ParaNLRU](normalized/para_nlru.md) ·
[ParaRWKV7](matrix/para_rwkv7.md) ·
[ParaSLSTM](normalized/para_slstm.md) ·
[ParaTitans](matrix/para_titans.md)

## Fidelity & audit

- [Fidelity labels](fidelity.md)
- [Audit matrix](audit_matrix.md) — one-page upstream → Para-native map
- [Spec template](spec_template.md)
- Shared τ: [Numerics contract](../core/numerics_contract.md)
- Jacobian taxonomy: [Jacobian classes](../core/jacobian_classes.md)
