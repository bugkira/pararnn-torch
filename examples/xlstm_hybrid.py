"""NX-AI sLSTMBlock around ParaRNN(ParaSLSTM mix='diag').

Their block keeps pre-LN, residual skip, and the gated FFN.
The recurrent slot (``block.xlstm``) is ``ParaRNN(ParaSLSTM)``.
Newton K=3 is App. A / Danieli et al. §2.1.

Usage:
    uv add xlstm   # Python 3.11+
    python xlstm_hybrid.py
"""

import torch
from torch import Tensor, nn

from pararnn import NewtonConfig, ParaRNN, ParaSLSTM

_INSTALL = "uv add xlstm"


def require_xlstm():
    try:
        from xlstm.blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
        from xlstm.blocks.slstm.layer import sLSTMLayerConfig
        from xlstm.components.feedforward import FeedForwardConfig
    except ImportError as exc:
        raise RuntimeError(f"NX-AI xlstm is not installed. Install with: {_INSTALL}") from exc
    return sLSTMBlock, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig


class ParaSLSTMAsXlstmSlot(nn.Module):
    """sLSTMLayer stand-in: (B, T, d) → (B, T, d). Swallows NX-AI kwargs."""

    def __init__(self, d_model: int, *, config: NewtonConfig, solver: str = "auto") -> None:
        super().__init__()
        self.rnn = ParaRNN(
            ParaSLSTM(d_model, d_model, mix="diag"),
            config=config,
            output_hidden=True,
            solver=solver,
        )

    def forward(self, x: Tensor, **_kwargs) -> Tensor:
        return self.rnn(x)

    def step(self, x: Tensor, **_kwargs):
        raise RuntimeError(
            "token-wise NX-AI .step is unwired; call .eval() on the block "
            "for a sequential ParaSLSTM unroll of the full sequence"
        )


def build_hybrid_slstm_block(
    d_model: int = 64,
    n_heads: int = 4,
    *,
    config: NewtonConfig | None = None,
    solver: str = "auto",
) -> nn.Module:
    """NX-AI sLSTMBlock with ParaRNN(ParaSLSTM mix='diag') in .xlstm."""
    sLSTMBlock, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig = require_xlstm()
    if d_model % n_heads != 0:
        raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")
    newton = config or NewtonConfig(max_iters=3)
    cfg = sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            embedding_dim=d_model,
            num_heads=n_heads,
            conv1d_kernel_size=0,
            backend="vanilla",
            dtype="float32",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    )
    cfg.slstm.embedding_dim = d_model
    if cfg.feedforward is not None:
        cfg.feedforward.embedding_dim = d_model
    cfg._num_blocks = 1
    cfg._block_idx = 0
    cfg.__post_init__()
    block = sLSTMBlock(cfg)
    block.xlstm = ParaSLSTMAsXlstmSlot(d_model, config=newton, solver=solver)
    return block


if __name__ == "__main__":
    D_MODEL, N_HEADS, BATCH, SEQ_LEN = 64, 4, 2, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = build_hybrid_slstm_block(D_MODEL, N_HEADS).to(device)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)

    block.train()
    y_train = block(x)
    y_train.sum().backward()
    stats = block.xlstm.rnn.last_stats
    res = stats[0].max_residual if stats else float("nan")
    block.zero_grad(set_to_none=True)

    block.eval()
    y_eval = block(x)
    err = (y_train - y_eval).abs().max().item()
    assert torch.isfinite(y_train).all(), "hybrid forward produced non-finite values"
    print(
        f"hybrid OK device={device} shape={tuple(y_train.shape)} "
        f"residual={res:.3e} train_vs_eval_maxabs={err:.3e}"
    )
