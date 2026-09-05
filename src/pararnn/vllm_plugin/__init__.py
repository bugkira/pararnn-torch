"""vLLM ``general_plugins`` registration for ``ParaSLSTMForCausalLM``.

Install with the optional extra::

    uv sync --extra vllm

vLLM discovers ``[project.entry-points."vllm.general_plugins"]`` and calls
``register()`` in every process. The architecture string in ``config.json``
must be ``ParaSLSTMForCausalLM``.

Registration plus ``MambaBase`` layers in ``modeling`` / ``layers`` is the
engine path: worker pages + ``Mamba1AttentionMetadata`` → ``decode_step``.
See ``docs/vllm.md``.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_REGISTERED = False

_ARCH = "ParaSLSTMForCausalLM"
_LAZY = "pararnn.vllm_plugin.modeling:VLLMParaSLSTMForCausalLM"


def register() -> None:
    """Idempotent ``ModelRegistry.register_model`` for the plugin entry point."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from vllm import ModelRegistry
    except ImportError:
        log.debug("vllm not installed; skip ModelRegistry registration")
        return

    supported = ModelRegistry.get_supported_archs()
    if _ARCH not in supported:
        # Lazy string avoids importing CUDA-touching code in the parent before fork.
        ModelRegistry.register_model(_ARCH, _LAZY)
        log.info("registered %s -> %s", _ARCH, _LAZY)
    _REGISTERED = True
