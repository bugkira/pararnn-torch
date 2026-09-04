"""Prepare BabyLM 10M tokens + 16k BPE cache.

    uv run --extra lm python scripts/prepare_babylm.py --config configs/train/babylm.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.babylm_data import prepare_packed
from scripts.utils.mlflow_helper import ROOT, setup_logging

log = logging.getLogger("prepare_babylm")
DEFAULT_CONFIG = ROOT / "configs" / "train" / "babylm.yaml"


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    train, val, tokenizer = prepare_packed(spec, ROOT)
    log.info(
        "prepare_done train_rows=%d val_rows=%d vocab=%d",
        int(train.shape[0]),
        int(val.shape[0]),
        tokenizer.get_vocab_size(),
    )


if __name__ == "__main__":
    main()
