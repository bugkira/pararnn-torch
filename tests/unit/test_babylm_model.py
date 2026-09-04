"""CPU shape / param smoke for the BabyLM stack (no HF download)."""

from __future__ import annotations

import torch

from scripts.babylm_model import BabyLMModel, count_params


def _tiny_spec(**overrides) -> dict:
    spec = {
        "d_model": 32,
        "n_heads": 4,
        "vocab_size": 64,
        "seq_len": 8,
        "num_layers": 2,
        "mlp_mult": 4,
        "mlp_act": "swiglu",
        "newton_iters": 3,
        "picard_iters": 3,
        "picard_adapt": False,
        "max_recurrent_norm": 0.5,
    }
    spec.update(overrides)
    return spec


def test_diag_seq_forward_cpu():
    model = BabyLMModel(_tiny_spec(), cell_type="diag_seq")
    tokens = torch.randint(0, 64, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()
    assert count_params(model) > 0


def test_dense_head_forward_cpu():
    model = BabyLMModel(_tiny_spec(), cell_type="dense")
    tokens = torch.randint(0, 64, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 64)


def test_weight_tying():
    model = BabyLMModel(_tiny_spec(), cell_type="diag_seq")
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
