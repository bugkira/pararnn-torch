#!/usr/bin/env bash
# Build sdist+wheel, twine-check, install into a clean venv, smoke outside the repo.
# Usage (from repo root):
#   bash scripts/check_wheel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VER="$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml | head -n1)"
test -n "$VER"
WHL="dist/pararnn_torch-${VER}-py3-none-any.whl"
SDIST="dist/pararnn_torch-${VER}.tar.gz"
ENV="${TMPDIR:-/tmp}/pararnn_wheel_smoke_${VER}"

rm -f dist/pararnn_torch-*.whl dist/pararnn_torch-*.tar.gz
uv build
test -f "$WHL" && test -f "$SDIST"

uvx twine check "$WHL" "$SDIST"

# Junk / lab paths must stay out of the wheel and sdist.
# (Avoid `unzip | grep -q` under `pipefail`: SIGPIPE → exit 141.)
WLIST="$(mktemp)"
SLIST="$(mktemp)"
trap 'rm -f "$WLIST" "$SLIST"' EXIT
unzip -l "$WHL" >"$WLIST"
tar -tzf "$SDIST" >"$SLIST"
if grep -qiE 'outputs/|mlruns/|docs/internal|third_party/|\.cursor/|user-friction' "$WLIST"; then
  echo "ERROR: suspicious path inside wheel" >&2
  exit 1
fi
if grep -qiE 'outputs/|mlruns/|docs/internal|third_party/|\.cursor/|user-friction' "$SLIST"; then
  echo "ERROR: suspicious path inside sdist" >&2
  exit 1
fi
grep -q 'pararnn/kernels/' "$WLIST" || {
  echo "ERROR: kernels/ missing from wheel" >&2
  exit 1
}
unzip -p "$WHL" "pararnn_torch-${VER}.dist-info/entry_points.txt" | grep -q 'vllm.general_plugins' || {
  echo "ERROR: vllm entry point missing" >&2
  exit 1
}

rm -rf "$ENV"
uv venv "$ENV"
# Resolve heavy deps from the Torch cu128 index (same as project), then the wheel.
uv pip install --python "$ENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  "torch>=2.11.0" "triton>=3.6.0,<3.7" "numpy>=2.2.6" "safetensors>=0.8.0"
uv pip install --python "$ENV/bin/python" --no-deps "$WHL"

# Critical: leave the repo so import does not hit src/.
cd /tmp
"$ENV/bin/python" - <<'PY'
import pararnn
assert "/site-packages/" in pararnn.__file__.replace("\\", "/"), pararnn.__file__
print("version:", pararnn.__version__)
print("file:", pararnn.__file__)

import torch
from pararnn import NewtonConfig, ParaSLSTM, newton_apply, sequential_apply

cell = ParaSLSTM(32, 32, mix="diag")
cfg = NewtonConfig(max_iters=2, scan_backend="eager")
x = torch.randn(2, 8, 32)
y = newton_apply(cell, x, cfg)
z = sequential_apply(cell, x)
err = float((y - z).detach().abs().max())
print(f"eager newton↔seq max|diff|={err:.3e}")
assert err < 1e-4

if torch.cuda.is_available():
    from pararnn import can_decode_step, decode_step, decode_wx

    device = torch.device("cuda")
    cell = ParaSLSTM(32, 32, mix="diag").to(device).eval()
    x = torch.randn(1, 8, 32, device=device)
    with torch.no_grad():
        carry = sequential_apply(cell, x)[:, -1].contiguous()
        assert can_decode_step(cell, carry)
        wx = torch.empty(1, cell.W_x.out_features, device=device, dtype=x.dtype)
        out = torch.empty_like(carry)
        decode_wx(cell, torch.randn(1, 32, device=device), out=wx)
        decode_step(cell, carry, wx=wx, out=out)
    print("cuda decode_step ok", tuple(out.shape))
print("wheel smoke passed")
PY

echo "OK  ${WHL}"
echo "OK  ${SDIST}"
echo "venv was ${ENV}"
