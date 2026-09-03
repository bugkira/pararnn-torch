# Literature

PDFs: `bash scripts/fetch_papers.sh` → `docs/papers/`. This is the working set for cells, solvers, hybrid PC, and later evals. Hyperparameters in configs must cite a paper or a measured ablation in this repo.

## Core algorithm

| Paper | ArXiv | Why it matters |
|---|---|---|
| Danieli et al., **ParaRNN**, ICLR 2026 Oral | [2510.21450](https://arxiv.org/abs/2510.21450) | Primary spec. Nonlinear RNN as \(F(H)=0\); Newton + parallel reduction; diagonal ParaGRU / block-diag CIFG ParaLSTM; \(K=3\); init \(h_l^0=f(0,x_l)\); LM up to 7B vs Transformer/Mamba2 on SlimPajama. |
| Danieli et al., **DeepPCR**, NeurIPS 2023 | [2309.16318](https://arxiv.org/abs/2309.16318) | Same Newton+PCR idea on ResNets / diffusion. Historical parent; not the cell definitions. |
| Lim et al., **DEER**, ICLR 2024 | [2309.12252](https://arxiv.org/abs/2309.12252) | Newton as parallel fixed-point for RNN/NeuralODE **without** changing the cell. Dense J is \(O(d_h^3)\). Notes quadratic convergence when the guess is close — this is the theoretical hook for a better predictor than \(f(0,x_l)\). |
| Gonzalez et al., **quasi-DEER / ELK** | [2407.19115](https://arxiv.org/abs/2407.19115) | Diagonal quasi-Newton (cheaper, less stable) + Levenberg–Marquardt via Kalman (ELK). Apple instead **changes the cell** so J is exactly (block-)diagonal. If our Newton blows up on a new cell, ELK/damping (`omega_sor<1`) is the documented fallback, not “add iterations”. Analytic trust-region \(\lambda\) (Nocedal–Wright 4.3) they skip as intractable. |
| Eisenstat & Walker, inexact Newton forcing terms | [SIAM J. Sci. Comput. 1996](https://doi.org/10.1137/0917003) | Adapt **linear** solve tolerance from residual ratios. **Not** our Picard P (we already exact-scan \(J\delta=-F\)). |
| Bai et al., Jacobian regularization for DEQs | [ICML 2021](https://arxiv.org/abs/2106.14342) | Inner NFE **grows during training** (same as P=1 after Adam). Dual fix: penalize \(\|J\|_F\). We clip \(R\) already; P-retry is the first train-diag fix. |

## Linear SSMs (predictor, baselines, expressivity)

| Paper | ArXiv | Why it matters |
|---|---|---|
| Gu & Dao, **Mamba** | [2312.00752](https://arxiv.org/abs/2312.00752) | Selective linear SSM + hardware scan. Candidate **predictor** for hybrid PC (our idea, not Apple's). Inits (S4D-Real, \(\Delta\in[10^{-3},10^{-1}]\)) are the cited defaults if we add a Mamba warm-start. |
| Dao & Gu, **Mamba-2** | [2405.21060](https://arxiv.org/abs/2405.21060) | SSD / faster scan. ParaRNN paper's LM baseline. Prefer Mamba-2 as the linear baseline and as a possible predictor. |
| Feng et al., **minGRU / minLSTM** | [2410.01201](https://arxiv.org/abs/2410.01201) | Parallel **linear** recurrences (no \(h_{t-1}\) nonlinearity in the state). Contrast class: not ParaRNN. |
| Beck et al., **xLSTM** | [2405.04517](https://arxiv.org/abs/2405.04517) | **mLSTM** is the matrix memory that scans like linear attention. **sLSTM** is the nonlinear cell (exp gates, stabilizer, mixing) and stays sequential in FlashRNN. We offer Newton for that sLSTM half; mLSTM stays theirs. See [`xlstm.md`](xlstm.md). Apple's `a_init_fn="xlstm"` is init, not this cell. |
| Merrill et al., **Illusion of State** | [2404.08819](https://arxiv.org/abs/2404.08819) | Linear SSMs stay in \(\mathsf{TC}^0\); they do not get RNN-style state tracking (parity, permutation composition). Justification for **keeping** nonlinear recurrence, not replacing it with Mamba. |
| Sarrof et al., SSM formal languages | [2405.17394](https://arxiv.org/abs/2405.17394) | Finer map of which regular languages SSMs can/cannot do (parity vs flip-flop). Eval ideas for later syntactic benchmarks. |

No paper found that already does **Mamba scan as Newton warm-start for a nonlinear RNN**. Closest: DEER (warm start helps) + ParaRNN footnote that a linear SSM is Newton with \(K=1\). Hybrid PC is our claim; implement only after vanilla \(K=3\) matches sequential.

## Implicit differentiation (later IFT adjoint)

| Paper | ArXiv | Why it matters |
|---|---|---|
| Bai et al., **DEQ** | [1909.01377](https://arxiv.org/abs/1909.01377) | IFT: backprop through the fixed point without storing Newton iterates. Memory \(O(1)\) in solver depth. Stability issues later in Bai 2021 (Jacobian reg). |
| Kolter et al., Deep Implicit Layers | tutorial | Clean adjoint derivation. Use when we implement IFT; Apple does **not** — they reverse-scan Jacobians once (ParaRNN eq. 2.6). |

## Training-scale knobs (when we train LMs)

From ParaRNN §5.2 and App. C (cite these, do not invent):

- Dataset: SlimPajama minus Books3.
- Optimizer: AdamW, detached weight decay, cosine LR, 10% warmup, decay to **0** ([Bergsma et al. 2025](https://arxiv.org/abs/2502.15938)).
- Tokens / batch: ~1× Chinchilla (Hoffmann et al. 2022); seq len **2048**.
- AMP: weights bf16, grads/reductions fp32; FSDP; PyTorch 2.6 in the paper.
- Architecture: DCLM Transformer backbone, attention swapped for the RNN cell, plus Mamba causal conv + gated residual.
- Synthetic tasks (parity, k-hop): AdamW \(\beta=(0.9,0.999)\), cosine LR \(5\times10^{-4}\), wd \(10^{-6}\), batch 16, \(L=100\), clip \(\|a\|,\|c\|\le 0.90\) except parity.

Table 3 in the PDF has per-scale LR / wd / width / depth. Copy from the paper into `configs/` when we train, not from memory.

## Use cases in the README (eval later, not v0)

| Topic | Pointer | Honest take |
|---|---|---|
| Virtual analog / guitar amps | Wright-style LSTM snapshots; [2403.08559](https://arxiv.org/abs/2403.08559); NAM | Sequential LSTM-32 is already the production baseline (real-time, ESR). ParaRNN helps **training** long audio, not a free accuracy win. |
| Nested / automata / code | Merrill 2024; Dyck literature | Nonlinear RNNs are the theoretically motivated tool; Transformers win needle-in-haystack. README star table must not claim 100k retrieval. This repo: Z2 tagging smoke in `examples/parity.py` (last-token 1.0 vs S4D-Real SSM chance; not A5). |
| Robotics / Hodgkin–Huxley | Neural ODE + DEER | Same Newton-on-a-path idea; different cells. Out of v0. |

## Corrections to the draft README

1. Apple's Newton guess is \(f(0,x_l)\), not \(H=0\).
2. Apple already ships ParaGRU/ParaLSTM + CUDA PCR. Our gap is HF, CPU/PyTorch prototype, logging, hybrid PC, IFT.
3. `pip` / MIT wrapping of Apple code is wrong; we use **uv** and reimplement.
4. 665× is vs **naive sequential**, not vs Mamba. Fused ParaGRU is ~2.6× vs Mamba at \(L=2^9\) (paper §5.1).
5. **Multi-GPU:** data-parallel SGD works for sequential FlashRNN and for Newton. Sequence-parallel *time* tiles need an associative scan carry (this repo: `scan_diag_two_ranks`, two streams on one GPU — not a cluster result). Informal “FlashRNN cannot use many cards” is DDP-false.
