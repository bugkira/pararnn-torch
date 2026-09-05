"""Continuous-batch ``BlockStackPool`` for ``ParaSLSTMBlock`` stacks."""

from __future__ import annotations

import torch

from pararnn import NewtonConfig, ParaSLSTMBlock
from pararnn.layout import SLSTM_SLOTS
from pararnn.models import ParaSLSTMConfig, ParaSLSTMForCausalLM
from pararnn.serve import BlockStackPool, state_shapes_for_vllm


def _blocks(n: int, d: int) -> list[ParaSLSTMBlock]:
    cfg = NewtonConfig(max_iters=2, scan_backend="eager", residual_fail=None)
    return [
        ParaSLSTMBlock(d, mlp_ratio=2.0, config=cfg, solver="sequential") for _ in range(n)
    ]


def test_block_stack_prefill_decode_matches_dense() -> None:
    torch.manual_seed(0)
    d, b, t = 16, 2, 5
    blocks = _blocks(2, d)
    stack = torch.nn.Sequential(*blocks)
    x_pre = torch.randn(b, t, d)
    x_dec = torch.randn(b, 1, d)
    with torch.no_grad():
        ref = stack(torch.cat((x_pre, x_dec), dim=1))

    pool = BlockStackPool(blocks, capacity=4)
    ids = pool.allocate(b)
    with torch.no_grad():
        y_pre = pool.forward_hidden(x_pre, ids, solver="sequential")
        y_dec = pool.forward_hidden(x_dec, ids, solver="sequential")
    torch.testing.assert_close(y_pre, ref[:, :t], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y_dec[:, 0], ref[:, t], atol=1e-4, rtol=1e-4)


def test_block_stack_packed_mixed() -> None:
    torch.manual_seed(1)
    d = 12
    blocks = _blocks(1, d)
    pool = BlockStackPool(blocks, capacity=4)
    ids = pool.allocate(3)
    # lengths 3, 1, 2
    x0, x1, x2 = torch.randn(1, 3, d), torch.randn(1, 1, d), torch.randn(1, 2, d)
    packed = torch.cat((x0, x1, x2), dim=1)
    cs = torch.tensor([0, 3, 4, 6])
    with torch.no_grad():
        y = pool.forward_hidden(packed, ids, cu_seqlens=cs, solver="sequential")
        r0 = blocks[0](x0)
        r1 = blocks[0](x1)
        # After first forward on slot 1, re-run decode alone for ref is harder —
        # check shapes and that slots stay allocated.
    assert y.shape == packed.shape
    assert pool.allocator.n_used == 3
    torch.testing.assert_close(y[:, :3], r0, atol=1e-4, rtol=1e-4)
    # Fresh slot 1: length-1 equals dense block on x1
    torch.testing.assert_close(y[:, 3:4], r1, atol=1e-4, rtol=1e-4)


def test_causal_lm_forward_continuous() -> None:
    torch.manual_seed(2)
    cfg = ParaSLSTMConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg).eval()
    for block in model.blocks:
        block.rnn.config = NewtonConfig(
            max_iters=1, scan_backend="eager", residual_fail=None
        )
    pool = model.attach_pool(4)
    ids = pool.allocate(2)
    prompt = torch.randint(0, 32, (2, 4))
    logits = model.forward_continuous(prompt, ids, solver="sequential")
    assert logits.shape == (2, 4, 32)
    # Decode one more token per request
    nxt = torch.randint(0, 32, (2, 1))
    logits2 = model.forward_continuous(nxt, ids, solver="sequential")
    assert logits2.shape == (2, 1, 32)
    pool.free(ids)
    assert pool.allocator.n_used == 0


def test_causal_lm_generate_uses_pool() -> None:
    torch.manual_seed(3)
    cfg = ParaSLSTMConfig(
        vocab_size=24,
        hidden_size=12,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    model = ParaSLSTMForCausalLM(cfg).eval()
    for block in model.blocks:
        block.rnn.config = NewtonConfig(
            max_iters=1, scan_backend="eager", residual_fail=None
        )
    model.attach_pool(2)
    out = model.generate(torch.randint(0, 24, (1, 3)), max_new_tokens=2)
    assert out.shape == (1, 5)
    assert model.pool is not None
    assert model.pool.allocator.n_used == 0


def test_bind_external_and_vllm_shapes() -> None:
    assert state_shapes_for_vllm(32) == ((1,), (SLSTM_SLOTS, 32))
    blocks = _blocks(2, 8)
    pool = BlockStackPool(blocks, capacity=3)
    ext = [torch.zeros(3, SLSTM_SLOTS, 8) for _ in range(2)]
    pool.bind_external(ext, mark_used=[0, 2])
    assert pool.allocator.n_used == 2
    assert pool.pools[0].buffers[0].data_ptr() == ext[0].data_ptr()


def test_vllm_model_continuous_kwargs() -> None:
    from types import SimpleNamespace

    from pararnn.layout import SLSTM_SLOTS
    from pararnn.vllm_plugin.modeling import VLLMParaSLSTMForCausalLM

    cfg = ParaSLSTMConfig(
        vocab_size=20,
        hidden_size=12,
        num_hidden_layers=1,
        mlp_ratio=2.0,
        newton_iters=1,
        scan_backend="eager",
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=cfg, dtype=torch.float32),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=None,
    )
    wrapper = VLLMParaSLSTMForCausalLM(vllm_config=vllm_config)
    for block in wrapper.library.blocks:
        block.rnn.config = NewtonConfig(
            max_iters=1, scan_backend="eager", residual_fail=None
        )
    wrapper.library.eval()
    assert wrapper.library.pool is not None
    ids = wrapper.library.pool.allocate(2)
    logits = wrapper.forward(
        torch.randint(0, 20, (2, 3)),
        slot_ids=ids,
        solver="sequential",
    )
    assert logits.shape == (2, 3, 20)
    flat = torch.randint(0, 20, (2,))
    logits_d = wrapper.forward(flat, slot_ids=ids, solver="sequential")
    assert logits_d.shape[-1] == 20
    shapes = VLLMParaSLSTMForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    assert shapes == ((1,), (SLSTM_SLOTS, 12))
