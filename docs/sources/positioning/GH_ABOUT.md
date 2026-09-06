# GitHub About — ready commands (gated)

Do **not** flip visibility or publish to PyPI until the maintainer explicitly
asks. Soft metadata (`description`, `homepage`, `topics`) is safe anytime.

## Recommended About fields

| Field | Value |
|---|---|
| Description | Hardware-efficient parallel training and O(1) decode for nonlinear RNNs (Newton+scan) |
| Homepage | `https://github.com/bugkira/pararnn-torch` |
| Topics (add) | `associative-scan`, `hardware-efficient`, `recurrent-neural-networks` |

Existing topics to keep: `gru`, `lstm`, `newton`, `parallel-scan`, `pytorch`,
`rnn`, `sequence-modeling`, `slstm`, `triton`, `xlstm`.

Zenodo DOI stays on the README badge and Citation section — remove it from
GitHub homepage so the product home is the repo.

## Safe metadata (can run now)

```bash
gh repo edit bugkira/pararnn-torch \
  --description "Hardware-efficient parallel training and O(1) decode for nonlinear RNNs (Newton+scan)" \
  --homepage "https://github.com/bugkira/pararnn-torch" \
  --add-topic associative-scan \
  --add-topic hardware-efficient \
  --add-topic recurrent-neural-networks
```

## Gated — needs explicit OK

```bash
# 1) Public launch
gh repo edit bugkira/pararnn-torch --visibility public
# then enable Discussions in Settings → General → Features

# 2) PyPI via Trusted Publishing (tag a release; see .github/workflows/release.yml)
git tag v0.17.1   # or next version
git push origin v0.17.1
```

After PyPI is live, the README PyPI badge resolves; Install already leads with
`pip install pararnn-torch`.

## Status (2026-09-06)

- Soft About metadata applied via `gh repo edit` (description, homepage, topics).
- Visibility remains **private**; PyPI publish **not** run — waiting on
  explicit maintainer OK.
