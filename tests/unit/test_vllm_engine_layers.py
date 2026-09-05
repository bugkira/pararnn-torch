"""Engine-path mixer with synthetic Mamba metadata (no vLLM install required)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from pararnn.layout import SLSTM_SLOTS
from pararnn.models.config import ParaSLSTMConfig
from pararnn.solvers.newton import NewtonConfig
from pararnn.vllm_plugin.layers import ParaSLSTMDecoderLayer, ParaSLSTMRecurrentLayer
from pararnn.vllm_plugin.modeling import VLLMParaSLSTMForCausalLM


def _meta(
    *,
    n_decode: int = 0,
    n_prefill: int = 0,
    idx_d=None,
    idx_p=None,
    qsl_p=None,
):
    return SimpleNamespace(
        num_decode_tokens=n_decode,
        num_prefill_tokens=n_prefill,
        num_decodes=n_decode,
        num_prefills=0 if qsl_p is None else int(qsl_p.numel()) - 1,
        state_indices_tensor_d=idx_d,
        state_indices_tensor_p=idx_p,
        query_start_loc_p=qsl_p,
        has_initial_states_p=None,
    )


def test_recurrent_layer_decode_updates_cache() -> None:
    torch.manual_seed(0)
    d, c = 8, 4
    layer = ParaSLSTMRecurrentLayer(
        d, newton=NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None)
    )
    pool = torch.zeros(c, SLSTM_SLOTS, d)
    layer.kv_cache = (torch.zeros(c, 1), pool)
    meta = _meta(n_decode=2, idx_d=torch.tensor([1, 3]))
    h = torch.randn(2, d)

    # Patch metadata lookup
    layer._attn_metadata = lambda: meta  # type: ignore[method-assign]
    y = layer(h)
    assert y.shape == (2, d)
    assert not torch.allclose(pool[1], torch.zeros_like(pool[1]))
    assert not torch.allclose(pool[3], torch.zeros_like(pool[3]))
    assert torch.allclose(pool[0], torch.zeros_like(pool[0]))


def test_recurrent_layer_prefill_packed() -> None:
    torch.manual_seed(1)
    d, c = 8, 3
    layer = ParaSLSTMRecurrentLayer(
        d, newton=NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None)
    )
    pool = torch.zeros(c, SLSTM_SLOTS, d)
    layer.kv_cache = (torch.zeros(c, 1), pool)
    # two requests lengths 3 and 2
    qsl = torch.tensor([0, 3, 5], dtype=torch.int32)
    meta = _meta(
        n_prefill=5,
        idx_p=torch.tensor([0, 2]),
        qsl_p=qsl,
    )
    layer._attn_metadata = lambda: meta  # type: ignore[method-assign]
    y = layer(torch.randn(5, d))
    assert y.shape == (5, d)
    assert not torch.allclose(pool[0], torch.zeros_like(pool[0]))
    assert not torch.allclose(pool[2], torch.zeros_like(pool[2]))


def test_decoder_layer_shape() -> None:
    layer = ParaSLSTMDecoderLayer(
        12,
        mlp_ratio=2.0,
        newton=NewtonConfig(max_iters=1, scan_backend="eager", residual_fail=None),
        prefix="model.layers.0",
    )
    y = layer(torch.randn(4, 12))
    assert y.shape == (4, 12)


def test_vllm_model_engine_bind_and_decode() -> None:
    cfg = ParaSLSTMConfig(
        vocab_size=20,
        hidden_size=12,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.float32),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=None,
    )
    model = VLLMParaSLSTMForCausalLM(vllm_config=vllm_config)
    for layer in model.layers:
        layer.mixer.newton = NewtonConfig(
            max_iters=1, scan_backend="eager", residual_fail=None
        )
    c = 4
    states = [torch.zeros(c, SLSTM_SLOTS, 12) for _ in model.layers]
    model.bind_kv_caches(states, mark_used=[0, 1])
    meta = _meta(n_decode=2, idx_d=torch.tensor([0, 1]))
    for layer in model.layers:
        layer.mixer._attn_metadata = lambda m=meta: m  # type: ignore[method-assign]

    ids = torch.randint(0, 20, (2,))
    hidden = model.forward(ids)
    assert hidden.shape == (2, 12)
    logits = model.compute_logits(hidden)
    assert logits.shape == (2, 20)
    assert not torch.allclose(states[0][0], torch.zeros_like(states[0][0]))


def test_get_mamba_helpers() -> None:
    cfg = ParaSLSTMConfig(hidden_size=16, vocab_size=10, num_hidden_layers=1)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.float32)
    )
    shapes = VLLMParaSLSTMForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    assert shapes == ((1,), (SLSTM_SLOTS, 16))
    dtypes = VLLMParaSLSTMForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
    assert len(dtypes) == 2
