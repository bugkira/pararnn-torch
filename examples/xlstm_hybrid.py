"""NX-AI sLSTMBlock around ParaRNN(ParaSLSTM mix='diag').

    uv add xlstm   # NX-AI package; Python 3.11+ (mlstm-kernels).
    uv run python examples/xlstm_hybrid.py

Their block keeps pre-LN, residual skip, and the gated FFN.
The recurrent slot (``block.xlstm``) is the fused cell: ``ParaRNN(ParaSLSTM)``.
Newton ``K=3`` is ParaRNN App. A / Danieli et al. §2.1.

Channel mix in the recurrence is diagonal (fused 4×4).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from torch import Tensor, nn

from pararnn import NewtonConfig, ParaRNN, ParaSLSTM

log = logging.getLogger("xlstm_hybrid")

_INSTALL = "uv add xlstm"


def require_xlstm():
    """Import NX-AI block configs. Raise with the install command if missing."""
    try:
        from xlstm.blocks.slstm.block import sLSTMBlock, sLSTMBlockConfig
        from xlstm.blocks.slstm.layer import sLSTMLayerConfig
        from xlstm.components.feedforward import FeedForwardConfig
    except ImportError as exc:
        raise RuntimeError(f"NX-AI xlstm is not installed. Install with: {_INSTALL}") from exc
    return sLSTMBlock, sLSTMBlockConfig, sLSTMLayerConfig, FeedForwardConfig


class ParaSLSTMAsXlstmSlot(nn.Module):
    """``sLSTMLayer`` stand-in: same ``(B, T, d) → (B, T, d)`` as their layer.

    Swallows extra kwargs that NX-AI ``sLSTMBlock.forward`` forwards.
    """

    def __init__(
        self,
        d_model: int,
        *,
        config: NewtonConfig,
        solver: str = "auto",
    ) -> None:
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
            "token-wise NX-AI .step is not wired; call .eval() on the block "
            "for a sequential ParaSLSTM unroll of the full sequence"
        )


def build_hybrid_slstm_block(
    d_model: int = 64,
    n_heads: int = 4,
    *,
    config: NewtonConfig | None = None,
    solver: str = "auto",
) -> nn.Module:
    """NX-AI ``sLSTMBlock`` with ``ParaRNN(ParaSLSTM mix='diag')`` in ``.xlstm``.

    ``d_model=64``, ``n_heads=4`` are NX-AI's default width / head count
    (must divide width). Our cell is channelwise. ``K=3`` is the paper default
    for Newton.
    """
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
    log.info(
        "hybrid_slstm_block d_model=%d nx_num_heads=%d mix=diag newton_iters=%d solver=%s",
        d_model,
        n_heads,
        newton.max_iters,
        solver,
    )
    return block


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, n_heads, batch, seq_len = 64, 4, 2, 16
    block = build_hybrid_slstm_block(d_model, n_heads).to(device)
    x = torch.randn(batch, seq_len, d_model, device=device)
    block.train()
    y_train = block(x)
    y_train.sum().backward()
    stats = block.xlstm.rnn.last_stats
    res = stats[0].max_residual if stats else float("nan")
    block.zero_grad(set_to_none=True)
    block.eval()
    y_eval = block(x)
    err = (y_train - y_eval).abs().max().item()
    log.info(
        "hybrid_smoke device=%s shape=%s residual=%.3e train_vs_eval_maxabs=%.3e",
        device,
        tuple(y_train.shape),
        res,
        err,
    )
    if not torch.isfinite(y_train).all():
        raise RuntimeError("hybrid forward produced non-finite values")


if __name__ == "__main__":
    main()
