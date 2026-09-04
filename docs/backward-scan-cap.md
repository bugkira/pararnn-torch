# Long-T block scan

## Status

**Diagonal (ParaGRU).** Superchunk scan in ``scan_diag_triton`` / fused GRU. Tests: ``test_triton_scan_diag_three_level_matches_eager``, ``test_fused_paragru_three_level_matches_eager``.

**2×2 / 4×4 (ParaLSTM, Diag-sLSTM).** When tile count exceeds the SRAM pad, ``incl_block_aggregates`` scans the reductions with eager Blelloch (same monoid, no T cap). Wired through ``run_block_scan_triton``, fused LSTM, and fused sLSTM. Picard already falls back to eager ``slstm_frozen_gate_scan`` past T=16384.

Leaf pads: LSTM ``64×64=4096``, unfused sLSTM scan ``64×32=2048``, fused sLSTM ``512×32=16384``.

## Remaining

VRAM of the stored trajectory still bounds train T (sequence-parallel / checkpointing). Windowed fused sLSTM (``fused_time_loop``) keeps its own tile pad. Host-loop superchunks on the diagonal path are polish at T≫10⁶.
