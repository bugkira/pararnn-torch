# BabyLM-10M mixing ablation

RTX 3060, **float32**, T=512, 6×384, 16k BPE, BabyLM 2026 Strict-Small.
One epoch = 603 steps, B=16, grad accum=2.

| Model Arm | Mixing R | Solver | Params | Val PPL | tok/s | Peak VRAM |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| `sLSTM-Dense` | Dense \(d_{\mathrm{head}}\times d_{\mathrm{head}}\) | Sequential | 21.0M | 399.70 | 977 | 5.69 GB |
| `Diag-sLSTM-Seq` | Vector \(d_h\) | Sequential | 20.5M | 380.96 | 1168 | 4.85 GB |
| `ParaSLSTM` (Ours) | Vector \(d_h\) | Fused Newton-Scan | 20.5M | 387.55 | 15401 | 4.00 GB |
