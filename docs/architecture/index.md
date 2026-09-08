# Architecture fidelity

Each cell has an **architecture spec**: equations, state layout, Jacobian class,
Newton defaults, and an explicit **fidelity** label. Specs are the contract
observers use to reimplement the recurrence by hand and compare against our
sequential oracle within the [numerics contract](../numerics-contract.md).

## Labels

| Label | Meaning |
|-------|---------|
| **paper-faithful** | Recurrence matches cited paper equations (up to documented notation). Parallel path is Alg. 1 on that \(f\). |
| **paper-core** | Core recurrence matches; block / conv / readout wrappers may sit outside the Newton cell. |
| **research-variant** | Intentional design choice that diverges from a named paper; deviations listed on the spec. |
| **unverified** | Shipped API; fidelity against a specific paper equation set is not yet audited. |

## Zoo status

| Cell | Spec | Fidelity | Primary source |
|------|------|----------|----------------|
| [`ParaGRU`](para_gru.md) | yes | paper-faithful (`mix='diag'`); paper-faithful / Dreamer-style for `mix='head'` gates | Danieli et al. 2025 eq. 3.1a, 3.3 |
| [`ParaLSTM`](para_lstm.md) | yes | paper-faithful | Danieli et al. 2025 eq. 3.1b, 3.3; Greff CIFG |
| [`ParaSLSTM`](para_slstm.md) | yes | paper-faithful (`mix='diag'`); Beck-style head mix documented | Danieli et al. / xLSTM sLSTM lineage |
| [`ParaNLRU`](para_nlru.md) | yes | research-variant vs linear Griffin RG-LRU | Griffin / RG-LRU + nonlinear slot (ours) |
| [`ParaM2RNN`](para_m2rnn.md) | yes | research-variant / paper-core | Mishra et al. arXiv:2603.14360 |
| [`ParaCfC`](para_cfc.md) | yes | research-variant | Hasani et al. Nat. Mach. Intell. 2022 / arXiv:2106.13898 eq. (10); ncps CfCCell |
| ParaHopfield, ParaRWKV7, ParaTitans | pending | unverified | see [cell catalog](../cells.md) |

## How to challenge a cell

1. Read the spec’s **Recurrence** and **Deviations**.
2. Implement sequential `step` from the equations alone.
3. Run `verify_agreement` (or compare trajectories) against `pararnn` at the stated τ.
4. Open an issue with shapes, dtype, and `report.to_dict()` if agreement fails at recommended \(K\).

Template for new specs: [TEMPLATE.md](TEMPLATE.md).
