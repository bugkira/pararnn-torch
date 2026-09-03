"""The hybrid example imports without NX-AI xlstm; build fails loudly."""

from __future__ import annotations

import pytest


def test_xlstm_hybrid_module_imports():
    from examples import xlstm_hybrid

    assert xlstm_hybrid.ParaSLSTMAsXlstmSlot is not None


def test_require_xlstm_hints_install(monkeypatch):
    import builtins

    import examples.xlstm_hybrid as hybrid

    real_import = builtins.__import__

    def _block_xlstm(name, *args, **kwargs):
        if name == "xlstm" or name.startswith("xlstm."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_xlstm)
    with pytest.raises(RuntimeError, match="uv add xlstm"):
        hybrid.require_xlstm()
