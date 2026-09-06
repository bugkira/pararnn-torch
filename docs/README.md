# Docs

API: numpydoc docstrings on root ``__all__`` (``help(pararnn.NewtonConfig)``).
Triton ``@jit`` kernels stay module-level one-liners.

| File | What |
|---|---|
| [`../INSTALL.md`](../INSTALL.md) | User / contributor install, hardware, Triton |
| [`../FAQs.md`](../FAQs.md) | Turing vs Ampere bf16, K*, residual vs τ, `verify_agreement` |
| [`numerics-contract.md`](numerics-contract.md) | Brand invariant: residual gate ≠ agreement τ; four echelons |
| [`cells.md`](cells.md) | Cell zoo snippets (Models table links here) |
| [`adoption.md`](adoption.md) | Attention / RSSM / Liquid swap paths |
| [`xlstm.md`](xlstm.md) | `ParaSLSTM` cell, stacking, API notes |
| [`distributed.md`](distributed.md) | DDP / FSDP2; tensor + context parallel |
| [`structure.md`](structure.md) | Layout of `src/`, tests, configs |
| [`backward-scan-cap.md`](backward-scan-cap.md) | Long-T scan: chunked adjoint; tile scan |
| [`vllm.md`](vllm.md) | vLLM plugin + continuous batch / CausalLM |
| [`sources/`](sources/) | Local PDF cache + positioning snapshots |
| [`sources/positioning/user-friction/`](sources/positioning/user-friction/) | Peer GitHub issue/PR friction study (lab) |
| [`../scripts/README.md`](../scripts/README.md) | Bench / train / diagnostic script index |
| [`../notebooks/README.md`](../notebooks/README.md) | Colab / Jupyter ParaSLSTM demo |

Drafts of *our* notes/preprints live under `internal/papers/` (gitignored):
`ParaSLSTM/`, `M2RNN/`. Positioning audit:
[`sources/positioning/AUDIT.md`](sources/positioning/AUDIT.md).
