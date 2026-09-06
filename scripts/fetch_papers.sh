#!/usr/bin/env bash
# Download the working bibliography into docs/sources/. Run from repo root:
#   bash scripts/fetch_papers.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/docs/sources"
mkdir -p "$DEST"

fetch() {
  local id="$1"
  local name="$2"
  local url="https://arxiv.org/pdf/${id}"
  local out="$DEST/${name}.pdf"
  if [[ -f "$out" ]]; then
    echo "skip  $name"
    return
  fi
  echo "get   $name  ($id)"
  curl -fsSL -A "ParaRNN-torch literature fetch" -o "$out" "$url"
}

fetch 2510.21450 pararnn-danieli-2025
fetch 2603.14360 m2rnn-mishra-2026
fetch 2309.16318 deeppcr-danieli-2023
fetch 2309.12252 deer-lim-2024
fetch 2407.19115 quasi-deer-elk-gonzalez-2024
fetch 2312.00752 mamba-gu-dao-2023
fetch 2405.21060 mamba2-dao-gu-2024
fetch 2404.08819 illusion-of-state-merrill-2024
fetch 2405.17394 ssm-expressive-capacity-2024
fetch 1909.01377 deq-bai-2019
fetch 2410.01201 mingru-feng-2024
fetch 2405.04517 xlstm-beck-2024
fetch 2403.08559 neural-amp-wright-2024
fetch 2502.15938 lr-to-zero-bergsma-2025

echo "done -> $DEST"
