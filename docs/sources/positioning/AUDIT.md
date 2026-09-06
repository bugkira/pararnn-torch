# Positioning audit: ParaRNN as “FLA for nonlinear RNNs”

Date: 2026-09-06. Peers: FlashAttention, Mamba, flash-linear-attention (FLA),
vLLM, Karpathy nanoGPT (+ llm.c notes). Snapshots:
[`NOTES.md`](NOTES.md).

**Positioning claim:** hardware-efficient building blocks for *nonlinear*
recurrence — the FLA / FlashAttention role for cells whose Jacobian is
structured enough for Newton + associative scan.

---

## 1. Peer file structure + README posture

| Repo | Top-level posture | README first screen | GitHub product |
|---|---|---|---|
| **FLA** | `fla/`, `benchmarks/`, `evals/`, `images/`, `INSTALL.md`, `FAQs.md` | Capability → TOC → News → Models → Install | Topics, Discussions, Releases, Wiki used; Discord + HF badges |
| **FlashAttention** | `flash_attn/`, `assets/` plots, `csrc/`, `usage.md` | Papers + banner → versioned kernels → install → plots | Discussions, heavy Releases; empty topics |
| **Mamba** | `mamba_ssm/`, `assets/` diagrams, `evals/` | Paper figures → install table → op→block→LM | Discussions, Releases |
| **vLLM** | Short README → docs.vllm.ai | Logo + slogan + feature bullets | Homepage, community links, media kit |
| **nanoGPT** | Flat scripts + `assets/` | Hero image + 3-min Shakespeare → sample text | Almost no org chrome; personality is the product |
| **pararnn-torch** | `src/pararnn`, `examples/`, `scripts/`, `docs/`, CI/release | Capability → News → Models → Install → Quickstart → Benches | Topics filled; **private** (2026-09-06); Releases yes; Discussions/Wiki off; homepage=Zenodo |

Lab-only weight in this checkout (`docs/internal/`, `mlruns/`, `outputs/`,
`third_party/`) stays local / gitignored. Public surface should match the
remote: `src/`, `examples/`, `notebooks/`, `.github/`, `CITATION.cff`,
`SECURITY.md`.

### FLA API ladder → ParaRNN analogue

| Level | FLA | ParaRNN |
|---|---|---|
| Op / kernel | `fla.ops.*` | Newton+PCR / fused cell kernels |
| Layer / block | `fla.layers.*` | `Para*Block` / `ParaRNN(cell)` |
| Model | HF `*ForCausalLM` | `ParaSLSTMForCausalLM` |
| Train / eval / bench | `flame`, lm-eval, `benchmarks/` | `scripts/train_*`, `scripts/bench_*`, MLflow |
| Hybrid | attn layer list in config | mixed stacks / predictor-corrector |

---

## 2. Comparison vs current ParaRNN

### Already strong

- FLA-shaped README spine (capability, TOC, News, Models, Install, Quickstart
  ladder, benches with machine context, Citation last).
- Real depth: cell catalog, Triton fused path, CausalLM, continuous batch,
  vLLM plugin hooks, CHANGELOG, Trusted Publishing, Colab, Zenodo DOI.
- Hygiene peers often skip: `SECURITY.md`, bug issue template, PR template,
  `uv` + typed package.

### Critical gaps

1. **Repo private** — unauthenticated URL 404; discovery / stars / Colab
   recruitment blocked.
2. **No `assets/` proof figures** — badges + tables only.
3. **GitHub About undersells** — “sequence module…” vs hardware-efficient
   nonlinear RNN brick; homepage = Zenodo DOI.
4. **Community surface empty** — no Discussions, Discord, feature template, HF hub.
5. **Install friction** — git URL only; PyPI workflow exists, not led in README.
6. **README density** — every cell in Usage; no Karpathy 60s magic path;
   no Training / Evaluation sections.
7. **Bench path naming** — `scripts/` vs peers’ `benchmarks/` (document clearly).
8. **Public docs boundary** — thin `docs/README.md`; no root `INSTALL.md` /
   `FAQs.md`.

---

## 3. Visual / interactivity / GitHub features

| Feature | Peers | ParaRNN (audit) | Target |
|---|---|---|---|
| Public visibility | All | Private | Public on explicit launch OK |
| About description | Capability one-liner | Narrow | Match `pyproject` description |
| Homepage | Product / self | Zenodo DOI | GitHub repo URL; DOI in Citation |
| Topics | Mixed | 10 good tags | Add `associative-scan`, `hardware-efficient`, `recurrent-neural-networks` |
| Discussions | FLA/FA/Mamba | Off | Enable on go-public |
| Releases | Labs | Through v0.17.x | Keep + CHANGELOG notes |
| README plots | FA/Mamba/nanoGPT | None → `assets/` | Speedup + K*(T) figures |
| Logo | FLA/vLLM | None | Optional wordmark later |
| Colab | Rare | Yes | Keep |
| Feature issue template | Common | Bug only | Add |
| PyPI version badge | Common | Missing | After first publish |
| Sample output in README | nanoGPT | Missing | Parity / tiny path |

---

## 4. Backlog P0–P3

### P0 — Launch blockers

- [ ] Make repo **public** (explicit user OK — gated).
- [x] Ready About description + homepage + topics (`docs/sources/positioning/GH_ABOUT.md`); soft metadata applied 2026-09-06.
- [ ] First **PyPI** publish via Trusted Publishing (explicit user OK — gated).
- [x] `assets/` speedup + K*(T) figures; embed in README Benchmarks.
- [x] Install polish (PyPI-first when published; git as bleeding-edge; hardware row).

### P1 — README product polish

- [x] 60-second “feel it” path (parity) after Quickstart.
- [x] Collapse Usage; cell catalog in `docs/cells.md`.
- [x] Training + Evaluation stubs.
- [x] Adoption strip (Attention / RSSM / Liquid).
- [x] Bench footnotes (lab GPU honesty).
- [x] Library `@software` citation identity.

### P2 — GitHub / community chrome

- [x] Feature request issue template (+ config.yml contacts).
- [ ] Enable Discussions on go-public.
- [x] Short `INSTALL.md` + `FAQs.md`.
- [ ] Discord / HF badge when community + hub exist.
- Keep benches under `scripts/` with README pointer (no `benchmarks/` symlink).

### P3 — Ecosystem

- HF hub demo checkpoint.
- Short GIF of fused vs sequential wall-clock.
- Public roadmap Discussion.
- Org / sibling train repo only after the brick is known.
- Never copy Apple `third_party/ml-pararnn` into MIT tree.

### What not to copy

- FLA News emoji flood; Mamba papers-first layout; vLLM-only short README
  without a docs site; nanoGPT flat-untested layout.

---

## Ready `gh` commands (do not run until user OK)

See [`GH_ABOUT.md`](GH_ABOUT.md). Visibility + PyPI remain gated.
