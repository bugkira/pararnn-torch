# GitHub About — ready commands (gated)

Do **not** flip visibility or publish to PyPI until the maintainer explicitly
asks. Soft metadata (`description`, `homepage`, `topics`) is safe anytime.

## Recommended About fields

| Field | Value |
|---|---|
| Description | `Train nonlinear RNNs in parallel (Newton+scan); decode one step at a time.` |
| Homepage | **empty** (peers leave it blank; self-link is useless; arXiv only if you want a paper home) |
| Topics | see set below |

### Topics (honest + searchable)

GitHub topic browse is weak SEO, but empty/`hardware-efficient`-tier tags
help nobody. Mix **high-traffic umbrellas** with **domain identity**.

| Keep / add | Why (~repo count order of magnitude) |
|---|---|
| `pytorch`, `deep-learning`, `machine-learning` | huge browse buckets |
| `cuda`, `gpu`, `triton` | systems people hunting kernels |
| `llm`, `neural-networks` | CausalLM / stack readers |
| `rnn`, `lstm`, `gru`, `recurrent-neural-networks` | domain |
| `sequence-modeling`, `xlstm`, `slstm` | catalog peers / Beck readers |
| `parallel-scan`, `associative-scan` | tiny but exact (our differentiator) |

**Drop:** `hardware-efficient` (~2 repos), `newton` (ambiguous; physics /
optimization noise).

**Skip as bait:** `transformers`, `mamba`, `flash-attention` — wrong product.

Cap is 20 topics; stay under it.

## Safe metadata (can run now)

```bash
gh repo edit bugkira/pararnn-torch \
  --description "Train nonlinear RNNs in parallel (Newton+scan); decode one step at a time." \
  --homepage "" \
  --remove-topic hardware-efficient \
  --remove-topic newton \
  --add-topic deep-learning \
  --add-topic machine-learning \
  --add-topic cuda \
  --add-topic llm \
  --add-topic neural-networks \
  --add-topic gpu
```

## Gated — needs explicit OK

```bash
# 1) Public launch
gh repo edit bugkira/pararnn-torch --visibility public
# then enable Discussions in Settings → General → Features

# 2) PyPI via Trusted Publishing (tag a release; see .github/workflows/release.yml)
git tag v0.17.3   # or next version
git push origin v0.17.3
```

After PyPI is live, optional homepage → `https://pypi.org/project/pararnn-torch/`
(or leave empty). README badge already points at PyPI.

## Status

- Soft About applied: punchier description, **cleared** self-homepage,
  topics rebalanced (2026-09-07).
- Visibility remains **private**; PyPI publish **not** run — waiting on
  explicit maintainer OK.
