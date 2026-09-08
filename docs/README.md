# Docs (repo index)

Published site: [bugkira.github.io/pararnn-torch](https://bugkira.github.io/pararnn-torch/).

Local preview:

```bash
uv sync --group docs
uv run mkdocs serve
```

API: numpydoc docstrings on root ``__all__`` (``help(pararnn.NewtonConfig)``).

| File | What |
|---|---|
| [`architecture/`](architecture/index.md) | Fidelity labels + per-cell equation specs |
| [`../INSTALL.md`](../INSTALL.md) | User / contributor install, hardware, Triton |
| [`../FAQs.md`](../FAQs.md) | Turing vs Ampere bf16, K*, residual vs τ, OOM cheat sheet |
| [`numerics-contract.md`](numerics-contract.md) | Agreement τ and residual gate; four echelons |
| [`oom-cookbook.md`](oom-cookbook.md) | Long-T / OOM: recompute, Hopfield d_h, RWKV slim heads |
| [`compile-amp.md`](compile-amp.md) | `torch.compile` fullgraph preset + autocast / AMP policy |
| [`shapes-layout.md`](shapes-layout.md) | Contiguity, `cu_seqlens` fused/eager/raise matrix |
| [`inference.md`](inference.md) | Decode carry, `decode_step` / CUDA Graph, `generate()` |
| [`adoption.md`](adoption.md) | Attention / RSSM / Liquid swap paths |
| [`xlstm.md`](xlstm.md) | `ParaSLSTM` cell, stacking, API notes |
| [`distributed.md`](distributed.md) | DDP / FSDP2; tensor + context parallel |
| [`structure.md`](structure.md) | Layout of `src/`, configs, examples |
| [`backward-scan-cap.md`](backward-scan-cap.md) | Long-T scan: chunked adjoint; tile scan |
| [`vllm.md`](vllm.md) | vLLM plugin: `MambaBase` / MAMBA1 boundaries + continuous batch |
| [`cells.md`](cells.md) | Call-site snippets for the zoo |
| [`../notebooks/README.md`](../notebooks/README.md) | Colab / Jupyter ParaSLSTM demo |

Benches, BabyLM verification, HF cards, and paper drafts live in the sibling
`pararnn-lab` tree on the maintainer machine.
