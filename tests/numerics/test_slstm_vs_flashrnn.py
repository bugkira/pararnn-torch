"""Config gate for the Dyck-1 Newton vs FlashRNN train example."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from examples.slstm_vs_flashrnn import _flashrnn_heads, _validate_spec

ROOT = Path(__file__).resolve().parents[2]


def test_dyck_vs_flashrnn_config_validates():
    spec = yaml.safe_load((ROOT / "configs" / "train" / "dyck_vs_flashrnn.yaml").read_text())
    _validate_spec(spec)
    n_heads, d_head = _flashrnn_heads(int(spec["d_h"]))
    assert n_heads * d_head == int(spec["d_h"])
    assert d_head == 32


def test_dyck_vs_flashrnn_rejects_odd_length():
    spec = yaml.safe_load((ROOT / "configs" / "train" / "dyck_vs_flashrnn.yaml").read_text())
    spec["seq_len"] = 63
    with pytest.raises(ValueError, match="even"):
        _validate_spec(spec)


def test_dyck_vs_flashrnn_head_config_validates():
    spec = yaml.safe_load(
        (ROOT / "configs" / "train" / "dyck_vs_flashrnn_head.yaml").read_text()
    )
    _validate_spec(spec)
    assert spec["mix"] == "head"
    assert int(spec["newton_iters"]) == 4
    assert int(spec["picard_iters"]) >= 3
    assert int(spec["d_h"]) % int(spec["n_heads"]) == 0
    n_heads, d_head = _flashrnn_heads(int(spec["d_h"]))
    assert (n_heads, d_head) == (1, 32)


def test_dyck_vs_flashrnn_head_rejects_p1():
    spec = yaml.safe_load(
        (ROOT / "configs" / "train" / "dyck_vs_flashrnn_head.yaml").read_text()
    )
    spec["picard_iters"] = 1
    with pytest.raises(ValueError, match="picard_iters"):
        _validate_spec(spec)


def test_dyck_vs_flashrnn_head_rejects_k3():
    spec = yaml.safe_load(
        (ROOT / "configs" / "train" / "dyck_vs_flashrnn_head.yaml").read_text()
    )
    spec["newton_iters"] = 3
    with pytest.raises(ValueError, match="K=4"):
        _validate_spec(spec)
