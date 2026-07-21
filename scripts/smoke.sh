#!/usr/bin/env bash
# 15-minute smoke test. Proves the pipeline learns before committing a day.
#
#   bash scripts/smoke.sh
#
# Expected: train loss starts near 6.9 (= ln 1000, pure chance) and falls.
# That is the entire test -- accuracy will be terrible and that is fine.
#
#   loss flat at 6.9    -> not learning. Check --model resolved and lr > 0.
#   loss goes NaN       -> BatchNorm diverged. Add --warmup-epochs 3.
#
# Ctrl-C once the loss is clearly coming down; you do not need a full epoch.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/workspace/imagenet-wds}"
TIMM_SRC="${TIMM_SRC:-/workspace/timm-src}"
WANDB="${WANDB:-1}"

# Live charts at wandb.ai, saved permanently. Run `wandb login` once first.
# Set WANDB=0 to run without it.
EXTRA=()
[ "$WANDB" = "1" ] && EXTRA+=(--log-wandb)

PYTHONPATH="$REPO/src" python "$TIMM_SRC/train.py" \
    --data-dir "$DATA" \
    --dataset wds/imagenet1k \
    --train-split train --val-split validation \
    --model vit_tiny_patch16_224_bn \
    --batch-size 256 --workers 12 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --epochs 2 --warmup-epochs 0 \
    --smoothing 0.1 --clip-grad 1.0 --amp \
    --log-interval 20 \
    --output "$REPO/output" --experiment smoke \
    "${EXTRA[@]}"
