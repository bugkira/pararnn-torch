# Long-T block scan

## Status

**Diagonal (ParaGRU).** Superchunk scan in ``scan_diag_triton`` / fused GRU. Tests: ``test_triton_scan_diag_three_level_matches_eager``, ``test_fused_paragru_three_level_matches_eager``.

**2×2 / 4×4 (ParaLSTM, Diag-sLSTM).** When tile count exceeds the SRAM pad, ``incl_block_aggregates`` scans the reductions with eager Blelloch (same monoid, no T cap). Wired through ``run_block_scan_triton``, fused LSTM, and fused sLSTM. Picard already falls back to eager ``slstm_frozen_gate_scan`` past T=16384.

Leaf pads: LSTM ``64×64=4096``, unfused sLSTM scan ``64×32=2048``, fused sLSTM ``512×32=16384``.

**Head GRU (``ParaGRU(mix='head')``).** No hard T pad: fused / streamed-``A`` / tiled paths walk time sequentially in one program per ``(batch, head)``. Checked through **T=4096** (Newton, reverse, VJP, fwd+bwd) on CUDA; dense ``scan_backend='eager'`` oracle builds ``(B,T,d_h,d_h)`` and OOMs first — compare factorized paths instead. Test: ``test_paragru_head_long_t_fused_smoke``.

Median latency (ms), float32, RTX 2080 Ti, ``K=3``, ``scripts/bench_gru_head.py`` / ad-hoc long-T:

| setup | ``d_head`` | Newton | reverse | VJP | fwd+bwd |
|---|---:|---:|---:|---:|---:|
| ``B=4``, ``T=128`` | 64 | 2.4 | — | — | — |
| ``B=4``, ``T=128`` | 96 / 128 | 9–10 | — | — | — |
| ``B=4``, ``T=128`` | 192 | 30 | — | — | — |
| ``B=1``, ``T=4096`` | 64 | 51 | 63 | 4 | 83 |
| ``B=1``, ``T=4096`` | 96 | 220 | 86 | 14 | 293 |

Packed ``cu_seqlens``: explicit ``scan_backend='fused'`` raises ``TypeError``;
``auto`` remaps to ``eager`` with a ``UserWarning``. Use ``eager`` for ragged packs
or pad to a rectangular batch.

## Remaining

VRAM of the stored trajectory still bounds train T. Opt-in Level 2:
``NewtonConfig(recompute=True)`` rematerializes H* in backward (eq. 2.6
unchanged; extra Newton forward FLOPs) for diag and head GRU alike
(``tests/numerics/test_recompute.py``). Outer ``torch.utils.checkpoint`` and
sequence-parallel remain complementary. Windowed fused sLSTM
(``fused_time_loop``) keeps its own tile pad. Host-loop superchunks on the
diagonal path are polish at T≫10⁶.

User-facing long-T / OOM cheat sheet: [`oom-cookbook.md`](oom-cookbook.md).
