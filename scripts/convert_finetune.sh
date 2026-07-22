#!/usr/bin/env bash
# Experiment 2, Stage C: fine-tune the converted BatchNorm model, distilling
# from the pretrained LayerNorm teacher.
#
#   bash scripts/convert_finetune.sh
#
# Uses timm's built-in knowledge distillation: the frozen LayerNorm DeiT-Ti
# is the teacher, the converted BatchNorm model is the student. The student
# learns to match the teacher's outputs rather than raw labels -- a much
# denser signal, and exactly the right target since we want to reproduce the
# teacher, not beat it.
#
# Far cheaper than Experiment 1: the features are already learned, so this is
# repairing the normalization change, not learning vision. Expect a good
# number in a fraction of the from-scratch epochs.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-/workspace/imagenet-wds}"
TIMM_SRC="${TIMM_SRC:-/workspace/timm-src}"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-512}"
CKPT="${CKPT:-$REPO/output/converted_init.pth}"
# Lower LR than from-scratch: nudging a nearly-right model, not building one.
LR=$(python3 -c "print(2e-4 * ${BATCH} / 512)")

if [ ! -f "$CKPT" ]; then
    echo "ERROR: converted weights not found at $CKPT"
    echo "Run first:  PYTHONPATH=$REPO/src python scripts/convert.py"
    exit 1
fi

echo "Experiment 2 Stage C: fine-tune converted BatchNorm model"
echo "start weights: $CKPT (DeiT-Ti with LayerNorm swapped for BatchNorm)"
echo "epochs:        $EPOCHS   lr: $LR"
echo

# --initial-checkpoint loads the converted weights (weights only, fresh
# optimizer and schedule). Without it this would train from random init and
# not be a conversion experiment at all.
PYTHONPATH="$REPO/src" python "$TIMM_SRC/train.py" \
    --data-dir "$DATA" \
    --dataset wds/imagenet1k \
    --train-split train --val-split validation \
    --model vit_tiny_patch16_224_bn \
    --initial-checkpoint "$CKPT" \
    --epochs "$EPOCHS" --batch-size "$BATCH" --workers 12 \
    --opt adamw --lr "$LR" --weight-decay 0.05 \
    --sched cosine --warmup-epochs 2 --min-lr 1e-6 \
    --smoothing 0.1 \
    --clip-grad 1.0 --amp --channels-last \
    --model-ema --model-ema-decay 0.9998 \
    --input-size 3 224 224 \
    --output "$REPO/output" --experiment convert_finetune \
    --checkpoint-hist 3 --log-interval 100
