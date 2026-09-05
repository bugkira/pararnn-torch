# Notebooks

## ParaSLSTM demo

[`paraslstm_demo.ipynb`](paraslstm_demo.ipynb) — Colab-ready Run All:

1. Install / device check (`float32`, `scan_backend="auto"`)
2. Drop-in `ParaRNN(ParaSLSTM)` train + eval
3. Sequential ↔ Newton max-abs error table
4. Latency vs \(T\) (same cell)
5. \(\mathbb{Z}_2\) prefix tagging: ParaSLSTM Newton vs S4D-Real SSM
6. T=1 `decode_step` agreement
7. Citation

### Local

```bash
uv sync --group dev
uv run python -m ipykernel install --prefix=.venv --name=pararnn
uv run jupyter lab notebooks/paraslstm_demo.ipynb
```

Skip the install cell; the editable `pararnn` package from `uv sync` is enough.

To regenerate the `.ipynb` from the builder script:

```bash
uv run python notebooks/_build_demo.py
```

### Google Colab (private repo)

1. Runtime → Change runtime type → **GPU**
2. Runtime → Secrets → add `GITHUB_TOKEN` with read access to `bugkira/pararnn-torch`
3. Open:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bugkira/pararnn-torch/blob/main/notebooks/paraslstm_demo.ipynb)

While the repository is private, Colab needs the token secret (or a manual upload of the notebook + `pip install` from a local wheel). After the repo is public or on PyPI, the install cell can drop the token.

Expected wall clock: about 2–4 minutes on a T4 / 2080 Ti-class GPU (parity is 2×2000 AdamW steps).
