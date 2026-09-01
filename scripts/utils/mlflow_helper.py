"""Git / lock hashes for example MLflow runs. Not part of the library."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"


def lock_hash() -> str:
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return "none"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def uv_export_hash() -> str:
    try:
        out = subprocess.check_output(
            ["uv", "export", "--frozen"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "none"
    return hashlib.sha256(out.encode()).hexdigest()[:16]
