# Positioning references (lab)

Fetched 2026-09-06 for README / product-entry rewrite.
nanoGPT snapshot added the same day. Full gap analysis:
[`AUDIT.md`](AUDIT.md).

| File | Upstream |
|---|---|
| [`flash-attention.README.md`](flash-attention.README.md) | https://github.com/Dao-AILab/flash-attention |
| [`mamba.README.md`](mamba.README.md) | https://github.com/state-spaces/mamba |
| [`flash-linear-attention.README.md`](flash-linear-attention.README.md) | https://github.com/fla-org/flash-linear-attention |
| [`vllm.README.md`](vllm.README.md) | https://github.com/vllm-project/vllm |
| [`nanogpt.README.md`](nanogpt.README.md) | https://github.com/karpathy/nanoGPT |

PDFs under `docs/sources/` stay gitignored; these markdown snapshots are lab notes
(also under `docs/sources/`, track only if you want them in-tree — currently
`*.pdf` ignore only).

## Pattern (what the top of the page sells)

1. **One job in one sentence** — IO-aware attention; selective SSM; serving
   throughput; hardware-efficient sequence mixers. Paper is a badge, the
   product is the capability.
2. **Levels of interface** — op → block → model → train/serve. Readers land on
   the level they need in <30 seconds.
3. **Living catalog** — News / Models table (FLA); feature bullets (vLLM);
   versioned kernels (FA-1/2/3). The repo *accumulates* architectures.
4. **Install that matches reality** — core vs CUDA opt-in (Mamba); hopper
   beta (FA-3); `uv pip install vllm`.
5. **Proof** — speedup figures, adoption links, HF models, lm-eval, generation
   benches. Numbers before taxonomy dump.
6. **Citation last** — papers at the bottom; the first screen is the brick.
7. **Karpathy add-on** — one 3-minute magical path with expected sample output
   (nanoGPT Shakespeare); personality + teeth, packaging optional.

## Primary template

**FLA** for catalog + News + layered Usage. **FlashAttention / Mamba / nanoGPT**
for `assets/` proof figures. **vLLM** for brand chrome when a docs site exists.
**nanoGPT** for the “feel it” onboarding strip.

## ParaRNN status (2026-09-06)

`README.md` leads with capability, News, Models table, Install, Quickstart
(block → LM → cell), Benchmarks, then Usage — FLA / vLLM / Mamba shape.
Remaining gaps (private repo, no `assets/` plots, git-only install, dense
Usage): see [`AUDIT.md`](AUDIT.md) P0–P3.
