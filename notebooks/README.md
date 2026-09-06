# Notebooks

## ParaSLSTM demo

[`paraslstm_demo.ipynb`](paraslstm_demo.ipynb) — short Colab Run All (API poke):

1. Install / device
2. `ParaRNN(ParaSLSTM)` train + eval
3. Sequential ↔ Newton trust table
4. Forward latency table
5. T=1 `decode_step`
6. Links to `examples/parity.py` / `scripts/` + citation

Training curves and BabyLM runs live under `examples/` and `scripts/`.

### Local

```bash
uv sync --group dev
uv run python -m ipykernel install --prefix=.venv --name=pararnn
uv run jupyter lab notebooks/paraslstm_demo.ipynb
```

Skip the install cell; the editable `pararnn` package from `uv sync` is enough.

Regenerate the `.ipynb`:

```bash
uv run python notebooks/_build_demo.py
```

### Google Colab (private repo)

1. Runtime → Change runtime type → **GPU**
2. Runtime → Secrets → add `GITHUB_TOKEN` with read access to `bugkira/pararnn-torch`
3. Open:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)

Expected wall clock: about 30–90 seconds on a T4 / 2080 Ti-class GPU (no in-notebook training loop).
