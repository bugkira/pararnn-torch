# Architecture fidelity

Each cell page is a **spec**, not an apology. Order on every page:

1. **Diff** — upstream → Para-native; every row has **Kind**
   (`author` | `para` | `numerics`).
2. **Strict Spec Contract** — YAML: shapes, `chunk` indices, init, ε, τ,
   Jacobian operator.
3. Recurrence / Jacobian / parallel path / reproduce / agreement check.

**DoD:** Diff Kind + YAML + pack-order Reproduce + **Shapes & broadcasting** +
named Newton **solve primitive** are enough to reimplement `step` and the
linear time-scan without reading `src/` or guessing Sylvester/CG.
Template: [spec template](spec_template.md).
Newton algebra: [Newton + scan](../core/newton_scan.md).

Shared product τ: [numerics contract](../core/numerics_contract.md).  
Zoo index: [audit matrix](audit_matrix.md).  
Jacobian taxonomy: [Jacobian classes](../core/jacobian_classes.md).

## Labels

| Label | Meaning |
|-------|---------|
| **paper-faithful** | Para-native recurrence matches the cited equations (notation may differ). Diff rows are mostly `Kind=author`. |
| **paper-core** | Core recurrence matches; wrappers (conv, readout, full stack) sit outside the Newton cell — Diff says so (`Kind=para`). |
| **research-variant** | Intentional Para-native design; Diff **Rationale** + `Kind=para` is the contract. |
| **unverified** | Shipped API; Diff / YAML audit still pending. |

## Diff Kind

| Kind | Meaning |
|------|---------|
| `author` | Matches cited upstream equations / library cell |
| `para` | Intentional Para-native redesign (diag \(R\), CIFG pack, matrix core only, …) |
| `numerics` | App. C.1 / solver / stability classics (clamp, ε, App. A guess, \(K\) pin, dense cap, Picard) |

## Zoo status

| Cell | Family | Spec | Fidelity | Primary source |
|------|--------|------|----------|----------------|
| [`ParaGRU`](classic/para_gru.md) | classic | yes | paper-faithful (`mix='diag'`); head mix documented | Danieli et al. 2025 eq. 3.1a, 3.3 |
| [`ParaLSTM`](classic/para_lstm.md) | classic | yes | paper-faithful | Danieli et al. 2025 eq. 3.1b, 3.3; Greff CIFG |
| [`ParaNLRU`](normalized/para_nlru.md) | normalized | yes | research-variant | Griffin / RG-LRU + nonlinear slot |
| [`ParaSLSTM`](normalized/para_slstm.md) | normalized | yes | paper-faithful (`mix='diag'`); head / dense documented | Danieli et al. / xLSTM sLSTM lineage |
| [`ParaHopfield`](matrix/para_hopfield.md) | matrix | yes | paper-core | Ramsauer et al. Modern Hopfield |
| [`ParaM2RNN`](matrix/para_m2rnn.md) | matrix | yes | research-variant / paper-core | Mishra et al. arXiv:2603.14360 |
| [`ParaRWKV7`](matrix/para_rwkv7.md) | matrix | yes | paper-core | Peng et al. RWKV-7 Goose |
| [`ParaTitans`](matrix/para_titans.md) | matrix | yes | research-variant | Behrouz et al. Titans (shallow \(L{=}1\)) |
| [`ParaCfC`](continuous/para_cfc.md) | continuous | yes | research-variant | Hasani et al. / ncps CfCCell |

## How to challenge a cell

1. Diff + Strict Spec Contract (YAML) only — ignore the paper while coding the port.
2. Implement sequential `step` from the YAML pack order, init, and Reproduce snippet.
3. `verify_agreement` at YAML τ (fp32 `atol=1e-4`).
4. Issue with shapes, dtype, `report.to_dict()` if agreement fails at recommended \(K\).
