#!/usr/bin/env bash
# One-time pod setup. Run from the repo root after cloning.
#
#   bash scripts/setup_pod.sh
#
# Clones timm's training script and patches it to import our BatchNorm
# model, so `--model vit_tiny_patch16_224_bn` resolves. Also installs
# wandb for live training charts.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMM_SRC="${TIMM_SRC:-/workspace/timm-src}"

if [ ! -d "$TIMM_SRC" ]; then
    echo "cloning timm source to $TIMM_SRC"
    git clone -q https://github.com/huggingface/pytorch-image-models.git "$TIMM_SRC"
fi

# timm's train.py knows nothing about our model. Importing the package
# runs @register_model, which puts it in timm's registry.
if ! head -1 "$TIMM_SRC/train.py" | grep -q vitbn; then
    sed -i "1i import sys; sys.path.insert(0, '$REPO/src'); import vitbn" "$TIMM_SRC/train.py"
    echo "patched $TIMM_SRC/train.py to import vitbn"
else
    echo "$TIMM_SRC/train.py already patched"
fi

pip install -q -U timm wandb

echo
echo "--- verifying the model ---"
PYTHONPATH="$REPO/src" python - <<'PY'
import torch, torch.nn as nn, timm, vitbn

m = timm.create_model('vit_tiny_patch16_224_bn')
bn = sum(isinstance(x, vitbn.BatchNorm) for x in m.modules())
ln = sum(isinstance(x, nn.LayerNorm) for x in m.modules())
params = sum(p.numel() for p in m.parameters())
out = tuple(m(torch.randn(2, 3, 224, 224)).shape)

print(f"BatchNorms: {bn}")
print(f"LayerNorms: {ln}")
print(f"params:     {params:,}")
print(f"output:     {out}")

assert bn == 25, f"expected 25 BatchNorms, got {bn}"
assert ln == 0, f"expected 0 LayerNorms, got {ln}"
assert out == (2, 1000)
print("\nGATE PASSED: model is all-BatchNorm, DeiT-Ti sized")
PY
