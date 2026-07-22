#!/usr/bin/env bash
# Continue training from an existing checkpoint, adding more epochs.
#
#   CKPT=output/<run>/last.pth.tar EPOCHS=200 bash scripts/resume.sh
#
# Reuses the weights you already trained -- nothing starts from scratch.
# Starts a fresh cosine schedule from the current weights, so the learning
# rate rises again and the model keeps improving rather than sitting at the
# near-zero LR the previous run ended on.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/workspace/imagenet-wds}"
TIMM_SRC="${TIMM_SRC:-/workspace/timm-src}"

CKPT="${CKPT:?set CKPT to the checkpoint to resume from, e.g. output/<run>/last.pth.tar}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-512}"
RES="${RES:-224}"
LR=$(python3 -c "print(1e-3 * ${BATCH} / 512)")

echo "resuming from: $CKPT"
echo "adding:        $EPOCHS epochs, fresh cosine schedule"
echo

# --initial-checkpoint loads weights only (fresh optimizer + schedule),
# which is what a warm restart needs -- as opposed to --resume, which would
# also restore the old near-zero LR and make further training a no-op.
PYTHONPATH="$REPO/src" python "$TIMM_SRC/train.py" \
    --data-dir "$DATA" \
    --dataset wds/imagenet1k \
    --train-split train --val-split validation \
    --model vit_tiny_patch16_224_bn \
    --initial-checkpoint "$CKPT" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --workers 12 \
    --opt adamw --lr "$LR" --weight-decay 0.05 \
    --sched cosine --warmup-epochs 3 --min-lr 1e-5 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 \
    --smoothing 0.1 --reprob 0.25 \
    --clip-grad 1.0 --amp --channels-last \
    --model-ema --model-ema-decay 0.9998 \
    --input-size 3 "$RES" "$RES" \
    --output "$REPO/output" \
    --checkpoint-hist 3 --log-interval 100
