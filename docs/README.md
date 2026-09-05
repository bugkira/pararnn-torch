# Docs

API: numpydoc docstrings on root ``__all__`` (``help(pararnn.NewtonConfig)``).
Triton ``@jit`` kernels stay module-level one-liners.

| File | What |
|---|---|
| [`xlstm.md`](xlstm.md) | `ParaSLSTM` cell, stacking examples, [API notes](xlstm.md#api-notes) (outputs, `mix=`, scan backends) |
| [`distributed.md`](distributed.md) | DDP / FSDP2; tensor parallel along \(d_h\); context-parallel scan over NCCL |
| [`structure.md`](structure.md) | Layout of `src/`, tests, configs |
| [`backward-scan-cap.md`](backward-scan-cap.md) | Long-T scan: chunked adjoint; hierarchical / eager-aggregate tile scan |
| [`vllm.md`](vllm.md) | vLLM plugin + `BlockStackPool` continuous batch / CausalLM |
| [`../scripts/README.md`](../scripts/README.md) | Bench / train / diagnostic script index |
| [`../notebooks/README.md`](../notebooks/README.md) | Colab / Jupyter ParaSLSTM demo |
