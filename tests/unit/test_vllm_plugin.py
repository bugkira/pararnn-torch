"""Standalone CausalLM + vLLM plugin discovery (vllm optional)."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import torch

from pararnn.models import ParaSLSTMConfig, ParaSLSTMForCausalLM
from pararnn.solvers.newton import NewtonConfig


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg = ParaSLSTMConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        newton_iters=2,
        scan_backend="eager",
    )
    cfg.save_pretrained(tmp_path)
    loaded = ParaSLSTMConfig.from_pretrained(tmp_path)
    assert loaded.vocab_size == 128
    assert loaded.architectures == ["ParaSLSTMForCausalLM"]
    assert loaded.intermediate_size == 64


def test_causal_lm_forward_and_save(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = ParaSLSTMConfig(
        vocab_size=64,
        hidden_size=24,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        newton_iters=2,
        scan_backend="eager",
        max_recurrent_norm=0.5,
    )
    model = ParaSLSTMForCausalLM(cfg)
    # Force eager Newton in blocks (already via config).
    for block in model.blocks:
        block.rnn.config = NewtonConfig(max_iters=2, scan_backend="eager", residual_fail=None)
    ids = torch.randint(0, 64, (2, 8))
    model.train()
    logits = model(ids)
    assert logits.shape == (2, 8, 64)
    logits.sum().backward()
    assert any(p.grad is not None for p in model.parameters())

    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "pytorch_model.bin").is_file()
    restored = ParaSLSTMForCausalLM.from_pretrained(tmp_path)
    restored.eval()
    with torch.no_grad():
        a = model.eval()(ids)
        b = restored(ids)
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_causal_lm_labels_loss_backward() -> None:
    torch.manual_seed(2)
    cfg = ParaSLSTMConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg)
    for block in model.blocks:
        block.rnn.config = NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None)
    ids = torch.randint(0, 32, (2, 6))
    labels = ids.clone()
    labels[:, 0] = -100
    model.train()
    logits, loss = model(ids, labels=labels)
    assert logits.shape == (2, 6, 32)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_causal_lm_from_pretrained_bin_fallback(tmp_path: Path) -> None:
    torch.manual_seed(3)
    cfg = ParaSLSTMConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg)
    for block in model.blocks:
        block.rnn.config = NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None)
    model.save_pretrained(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    restored = ParaSLSTMForCausalLM.from_pretrained(tmp_path)
    ids = torch.randint(0, 40, (1, 4))
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), restored(ids), atol=1e-5, rtol=1e-5)


def test_causal_lm_generate_grows() -> None:
    torch.manual_seed(1)
    cfg = ParaSLSTMConfig(
        vocab_size=48,
        hidden_size=16,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg).eval()
    for block in model.blocks:
        block.rnn.config = NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None)
    prompt = torch.randint(0, 48, (1, 4))
    out = model.generate(prompt, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 7)


def test_vllm_plugin_entry_point_declared() -> None:
    eps = entry_points(group="vllm.general_plugins")
    names = {ep.name for ep in eps}
    assert "pararnn_paraslstm" in names
    ep = next(ep for ep in eps if ep.name == "pararnn_paraslstm")
    assert "pararnn.vllm_plugin" in ep.value


def test_vllm_register_callable() -> None:
    from pararnn.vllm_plugin import register

    register()  # no-op when vllm is absent; registers when present
    try:
        import vllm  # noqa: F401
    except ImportError:
        return
    register()
    from vllm import ModelRegistry

    assert "ParaSLSTMForCausalLM" in ModelRegistry.get_supported_archs()
