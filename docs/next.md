# Where next (diag-only)

Living backlog after we dropped `mix='head'`/`dense` as a product path.
`mix='diag'` is the feature: 4×4 fused Newton, same modeling cut as ParaGRU
in Danieli et al. (channels mix in later \(W_x\), not inside \(R h\)).
FlashRNN head-mix is a different cell; do not spend time matching it.

## Do not

- Fused `mix='head'`, 1×32 FlashRNN ablations, NCCL-on-one-GPU theater.
- Empty Hugging Face `PreTrainedModel`, SlimPajama / 125M on the 2080 Ti.
- IFT adjoint until eq. 2.6 is the measured bottleneck.
- Racing FlashRNN on **forward ms**. Fused diag sLSTM is ~4× slower than
  `triton_fused` at \(T=2048\) on this card; that is a sequential kernel, not
  a mix bug.

## Order of work

1. **Train diag does not lie** (this file, §Train Picard). **Done on the
   Dyck seed:** auto-P retry, 2026-09-01 replay of
   `dyck_vs_flashrnn.yaml`. Step 5: `picard_adapt`, residual \(7.6\times10^{-6}\)
   (was 0.557). Packed VJP for sLSTM diag is **in** (same split as GRU/LSTM:
   elementwise + one `W_x` GEMM; Triton on CUDA). Head/dense stay Autograd.
2. **Hybrid predictor.** **Measured 2026-09-02** (`scripts/bench_hybrid_pk.py`).
   Picard is the guess. At T=2048, P=5 K=1 is **24.2 → 20.1 ms** fwd vs
   library P=3 K=3 (same snap). P=3 K=1 is faster but misses (err 0.15).
   Parked: do not flip library K=3 / auto-P yet (one seed, init-scale;
   Dyck T=64 train not faster; GRU/LSTM stay App. A). Untrained S4D-Real
   as \(H^{(0)}\) diverges. Trained Mamba-2 is a different cell.
3. **Quality of diag.** **Measured 2026-09-02** (`examples/parity.py`).
   Stacked `xLSTMBlock` `mix='diag'` on Z2 tagging (Merrill §5, running XOR;
   not A5). T=32 / 300 steps / App. C \(5\times10^{-4}\) is copy-only
   (eval_tok ≈ 0.52 = t=0). T=16 / 2000 steps / lr \(10^{-3}\) (Shallue
   3-point leftover): Newton last-token **1.0** at T=16 by step 399 and at
   T=32 (unseen length) by 449; sequential **same CE**; S4D-Real selective
   SSM last-token **chance** (0.56 / 0.53). Residual snaps (\(\sim10^{-6}\));
   Picard adapt still fires on some post-solution batches. Not mamba-ssm.
   Not L=100. Deeper Dyck / A5 later.
4. **xLSTM could take our sLSTM Newton** — not a second mLSTM.
   They already scan matrix memory. FlashRNN leaves sLSTM sequential.
   Takeable: `ParaSLSTM` / `xLSTMBlock` `mix='diag'` (fused + **packed
   VJP**, in). Head mix is their FlashRNN cell, not that path. Gaps: no
   NX-AI weight map, no Figure 3 wrapper. [`docs/xlstm.md`](xlstm.md).

## Train Picard: we are not the first

The outer loop is AdamW. The thing that needs an on-the-fly schedule is the
**inner root finder** \(F(H)=0\), not the learning rate. Closest prior art:

| Source | What they adapt | Maps to us? |
|---|---|---|
| Eisenstat & Walker 1996, *Choosing the forcing terms in an inexact Newton method* (SIAM J. Sci. Comput.) | Inner **linear** tolerance \(\eta_k\) from residual ratios, so you do not over-solve \(J\delta=-F\) when far from the root | **No for P.** Our scan already solves the Newton linear system (near) exactly. EW would matter if we truncated the scan. Do not copy \(\eta_k=\gamma(\|F_k\|/\|F_{k-1}\|)^\alpha\) onto Picard depth. |
| Dembo–Eisenstat–Steihaug 1982, inexact Newton | Same forcing-term idea | Same. |
| Gonzalez et al. 2024 (quasi-DEER / ELK), arXiv:2407.19115 | **Damping** \(\lambda\) / trust region, not more Newton K. Analytic \(\lambda\) is Nocedal–Wright Alg. 4.3 (factorize \(\partial r/\partial s\)); they call that intractable and **sweep log-spaced \(\lambda\)**. Heuristic: reset unstable \(s_{t>i}\) to a finite value (Prop. 1: after \(i\) Newton steps, \(t\le i\) is exact). That heuristic **slows** wall time, which is why they built ELK. | \(\omega<1\) is already `NewtonConfig.omega`. We measured \(\omega=0.5\) **kills** the K=4 snap on sLSTM (`para-slstm.md`). Do not copy ELK \(\lambda\) as the default. The reset-prefix heuristic is our `chunk_len` (linear span) in disguise. |
| Bai et al. 2019, DEQ | Broyden **until** \(\|f(z)-z\|<\varepsilon\); **NFE grows over training** (fixed point gets harder). Cap iterations. | Same phenomenon as P=1 after Adam. Their knob is solver **tol / max iter**, not an outer-step formula. |
| Bai et al. 2021, Jacobian regularization (ICML) | Dual: penalize \(\|J_f\|_F\) so the **model** stays easy to solve (unregularized DEQ takes \(>3\times\) NFE by the end of train). | We already clip \(R\) to 0.5 (App. C.1). A \(\|J\|_F\) loss is a later ablation, not the first train-diag fix. |
| Pal et al. 2021, *Opening the Blackbox* (Neural ODE solver heuristics) | Regularize **internal** local-error / NFE so the learned dynamics stay cheap | Dual again: make \(f\) nicer. Optional after P-retry works. |
| Pal et al. 2023, locally regularized NDEs | Adaptive solvers already spend steps where the dynamics are stiff; regularize there | Not a P schedule. |
| torchdeq | `tol` + `max_iter`; stop on abs/rel residual | We already have `residual_atol` (early-stop K) and `residual_fail`. Missing piece was **retry with a better guess**. |
| ARDN, arXiv:2501.03487 | Per-component residual weights inside Newton line search | Overkill; our \(F\) is one trajectory residual. |

**Analytic shortcut we will not take.** Newton–Kantorovich: if
\(\|J^{-1}F\|\cdot\mathrm{Lip}(J)<1/2\), the quadratic basin is guaranteed.
Estimating \(\mathrm{Lip}(J)\) along a trained sLSTM is as expensive as
another residual eval and is not in the fused kernel. Practical proxy:
if after library K=3 we still have \(\max|F|\) above sequential-agreement
scale, the **guess** was outside the basin → more Picard P (still
\(O(\log T)\)), not more K (library contract, App. A).

Picard P is a frozen-gate warm start, not an inexact linear solve. Ladder
stays the measured triple \(\{1,3,5\}\) from `slstm_auto_picard` (init-scale
\(d_h=256\), `para-slstm.md`). No interpolation P=2,4: those cutovers failed
unseeded.

## Policy (implemented)

`NewtonConfig.picard_adapt`:

- `None` (default): **on** for ParaSLSTM when `picard_iters` was left `None`
  (auto from \(T\)). **Off** if the caller set P explicitly (benches,
  `picard_iters=0` divergence timings, head-smoke P=3).
- `True` / `False`: force.

Retry after a Newton attempt if \(\max|F|\) is non-finite or
\(>\) `picard_retry_atol` (default \(10^{-3}\), same as
`newton_residual_high` / sequential-agreement band). Dyck P=1 miss was 0.557
— under `residual_fail=1.0`, so fail-loud alone was not enough.

Rungs: current P \(\mapsto\) next in \(\{1,3,5\}\). Log `picard_adapt`.
If P=5 still misses `residual_fail`, raise `NewtonDivergenceError` as now.

Not Eisenstat–Walker. Not a change to AdamW. Not extra K.

## Later on this track

- Optional Bai-style \(\|J\|_F\) term if adapt fires too often after the
  first epoch.
- Trained Mamba-2 as a **cell**, not as an untrained Newton guess.
- A5 / deeper Dyck if we want Merrill-scale; the Z2 T=16 smoke is not that.
