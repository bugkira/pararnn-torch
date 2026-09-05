"""HuggingFace-shaped config for ``ParaSLSTMForCausalLM``."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParaSLSTMConfig:
    """Config for the library CausalLM / vLLM architecture name.

    ``architectures`` matches what vLLM ``ModelRegistry`` looks up after the
    ``vllm.general_plugins`` entry point registers ``ParaSLSTMForCausalLM``.
    """

    model_type: str = "paraslstm"
    architectures: list[str] = field(default_factory=lambda: ["ParaSLSTMForCausalLM"])
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 6
    mlp_ratio: float = 4.0
    # App. C.1 clip on R (Danieli et al. / this repo BabyLM default).
    max_recurrent_norm: float | None = 0.5
    mix: str = "diag"
    n_heads: int | None = None
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    # Newton K=3 (App. A). Serve/eval uses sequential ``decode_step``.
    newton_iters: int = 3
    scan_backend: str = "auto"
    picard_iters: int | None = None
    # Dummy Mamba-shaped fields so vLLM ``IsAttentionFree`` state calculators
    # can allocate a cache slot; ParaSLSTM does not use conv/SSM buffers.
    intermediate_size: int | None = None
    state_size: int = 16
    conv_kernel: int = 4
    time_step_rank: int = 1
    use_conv_bias: bool = False
    use_bias: bool = False
    hidden_act: str = "silu"
    layer_norm_epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.intermediate_size is None:
            self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        if self.layer_norm_epsilon is None:
            self.layer_norm_epsilon = self.rms_norm_eps
        if self.mix == "head" and self.n_heads is None:
            raise ValueError("mix='head' requires n_heads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParaSLSTMConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ParaSLSTMConfig:
        data = json.loads((Path(directory) / "config.json").read_text())
        return cls.from_dict(data)
