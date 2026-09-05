"""Library CausalLM configs and modules (HF-shaped JSON, no transformers dep)."""

from pararnn.models.causal_lm import ParaSLSTMForCausalLM
from pararnn.models.config import ParaSLSTMConfig

__all__ = ["ParaSLSTMConfig", "ParaSLSTMForCausalLM"]
