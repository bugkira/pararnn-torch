# Docs

API: numpydoc docstrings on root ``__all__`` (``help(pararnn.NewtonConfig)``).
Triton ``@jit`` kernels stay module-level one-liners.

| File | What |
|---|---|
| [`../INSTALL.md`](../INSTALL.md) | User / contributor install, hardware, Triton |
| [`../FAQs.md`](../FAQs.md) | Turing vs Ampere bf16, K*, residual vs τ, OOM cheat sheet |
| [`numerics-contract.md`](numerics-contract.md) | Brand invariant: residual gate and agreement τ; four echelons |
| [`oom-cookbook.md`](oom-cookbook.md) | Long-T / OOM: recompute, Hopfield d_h, RWKV slim heads |
| [`compile-amp.md`](compile-amp.md) | `torch.compile` fullgraph preset + autocast / AMP policy |
| [`shapes-layout.md`](shapes-layout.md) | Contiguity, `cu_seqlens` fused/eager/raise matrix |
| [`cells.md`](cells.md) | Cell zoo snippets (Models table links here) |
| [`adoption.md`](adoption.md) | Attention / RSSM / Liquid swap paths |
| [`xlstm.md`](xlstm.md) | `ParaSLSTM` cell, stacking, API notes |
| [`distributed.md`](distributed.md) | DDP / FSDP2; tensor + context parallel |
| [`structure.md`](structure.md) | Layout of `src/`, tests, configs |
| [`backward-scan-cap.md`](backward-scan-cap.md) | Long-T scan: chunked adjoint; tile scan |
| [`vllm.md`](vllm.md) | vLLM plugin + continuous batch / CausalLM |
| [`sources/`](sources/) | Local PDF cache + positioning snapshots (lab) |
| [`../scripts/README.md`](../scripts/README.md) | Bench / train / diagnostic script index |
| [`../notebooks/README.md`](../notebooks/README.md) | Colab / Jupyter ParaSLSTM demo |

Drafts of *our* notes/preprints live under `internal/papers/` (gitignored):
`ParaSLSTM/`, `M2RNN/`. Positioning audit:
[`sources/positioning/AUDIT.md`](sources/positioning/AUDIT.md).
Parked fused packing for head-mix: `docs/internal/fused-packing-todo.md`
(gitignored).
