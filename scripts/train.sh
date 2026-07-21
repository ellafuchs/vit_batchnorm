#!/usr/bin/env bash
# ViT-Tiny + BatchNorm, DeiT recipe, ImageNet-1k. The real run.
#
#   bash scripts/train.sh                     # 300 epochs, 224px, ~14h
#   EPOCHS=100 RES=160 bash scripts/train.sh  # faster, lower ceiling
#
# Baseline to approach: DeiT-Ti at 72.2% top-1 -- same data, same recipe,
# same schedule, differing only in LayerNorm vs BatchNorm.
#
# BatchNorm-specific choices, all deliberate:
#   --batch-size 512   statistics come from the batch; small batches are noisy
#   --clip-grad 1.0    mandatory; BatchNorm divergence appears as a loss spike
#   --warmup-epochs 5  raise to 10 if the loss spikes in the first epochs
#   NO gradient accumulation -- it computes statistics per micro-batch while
#   the optimizer sees the accumulated batch, silently changing semantics.
#   If memory forces a smaller batch, lower it and let LR rescale below.
#
# Watch the first two epochs for a spike, then it is safe to walk away.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/workspace/imagenet-wds}"
TIMM_SRC="${TIMM_SRC:-/workspace/timm-src}"

EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-512}"
RES="${RES:-224}"
WANDB="${WANDB:-1}"

LR=$(python3 -c "print(1e-3 * ${BATCH} / 512)")
EXP="bn_e${EPOCHS}_r${RES}_b${BATCH}"

EXTRA=()
[ "$WANDB" = "1" ] && EXTRA+=(--log-wandb --experiment "$EXP")

echo "run:    $EXP"
echo "lr:     $LR  (scaled from batch $BATCH)"
echo "data:   $DATA"
echo "target: DeiT-Ti 72.2% top-1"
echo

PYTHONPATH="$REPO/src" python "$TIMM_SRC/train.py" \
    --data-dir "$DATA" \
    --dataset wds/imagenet1k \
    --train-split train --val-split validation \
    --model vit_tiny_patch16_224_bn \
    --epochs "$EPOCHS" --batch-size "$BATCH" --workers 12 \
    --opt adamw --lr "$LR" --weight-decay 0.05 \
    --sched cosine --warmup-epochs 5 --min-lr 1e-5 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 \
    --smoothing 0.1 --reprob 0.25 \
    --clip-grad 1.0 --amp --channels-last \
    --model-ema --model-ema-decay 0.9998 \
    --input-size 3 "$RES" "$RES" \
    --output "$REPO/output" \
    --checkpoint-hist 3 --log-interval 100 \
    "${EXTRA[@]}"
