# GitHub About — ready commands (gated)

Soft metadata (`description`, `homepage`, `topics`) is safe anytime.
Visibility / PyPI cuts need an explicit maintainer OK (now granted for
public launch 2026-09-07).

## Recommended About fields

| Field | Value |
|---|---|
| Description | `Train nonlinear RNNs in parallel (Newton+scan); decode one step at a time.` |
| Homepage | Concept DOI `https://doi.org/10.5281/zenodo.22302586` (or PyPI after first upload) |
| Topics | see set below |
| Social preview | Upload [`.github/social-preview.png`](../../.github/social-preview.png) in Settings → General → Social preview (no REST API) |

### Topics (honest + searchable)

| Keep / add | Why |
|---|---|
| `pytorch`, `deep-learning`, `machine-learning` | huge browse buckets |
| `cuda`, `gpu`, `triton` | systems people hunting kernels |
| `llm`, `neural-networks` | CausalLM / stack readers |
| `rnn`, `lstm`, `gru`, `recurrent-neural-networks` | domain |
| `sequence-modeling`, `xlstm`, `slstm` | catalog peers / Beck readers |
| `parallel-scan`, `associative-scan` | tiny but exact (our differentiator) |

**Drop:** `hardware-efficient`, `newton` (ambiguous).

**Skip as bait:** `transformers`, `mamba`, `flash-attention`.

## Public launch (2026-09-07)

```bash
gh repo edit bugkira/pararnn-torch \
  --visibility public --accept-visibility-change-consequences \
  --enable-discussions \
  --homepage "https://doi.org/10.5281/zenodo.22302586"
```

Social preview (UI once):

1. Open https://github.com/bugkira/pararnn-torch/settings
2. General → Social preview → Upload image
3. Choose `.github/social-preview.png` (1280×640 speedup card)

## PyPI Trusted Publishing

Register at https://pypi.org/manage/account/publishing/ :

| Field | Value |
|---|---|
| Owner | `bugkira` |
| Repository | `pararnn-torch` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Then tag matching `pyproject.toml`:

```bash
bash scripts/check_wheel.sh
git tag -a v0.17.4 -m "pararnn-torch 0.17.4 — public launch"
git push origin v0.17.4
```

`.github/workflows/release.yml` runs tests → `uv build` → `uv publish`
(OIDC) → GitHub Release with wheels.

## Priority / timestamp

ParaSLSTM preprint **v2** is the dated public claim:
https://doi.org/10.5281/zenodo.22558086 (published 2026-09-06 UTC via Zenodo
API). Concept DOI (always latest): https://doi.org/10.5281/zenodo.22302586 .
This repo stays **unsigned** for git (personal vs work x509 key); Zenodo is
the archival timestamp.

## Status

- Repo **public** (2026-09-07).
- Zenodo v2 published (`10.5281/zenodo.22558086`).
- Social preview image checked in; UI upload still needed (no GitHub API).
- PyPI: wait for Trusted Publishing link, then `v0.17.4` tag.
