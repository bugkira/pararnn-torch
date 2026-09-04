# Docs

| File | What |
|---|---|
| [`xlstm.md`](xlstm.md) | `ParaSLSTM` cell, stacking examples, [API notes](xlstm.md#api-notes) (outputs, `mix=`, scan backends) |
| [`distributed.md`](distributed.md) | DDP / FSDP2 wrap of `ParaRNN`; Triton warmup before NCCL |
| [`structure.md`](structure.md) | Layout of `src/`, tests, configs |
| [`backward-scan-cap.md`](backward-scan-cap.md) | Long-T scan: chunked adjoint; hierarchical / eager-aggregate tile scan |
