# Training ViT-Tiny with BatchNorm from Scratch

**Date:** 2026-07-21
**Status:** Approved
**Supersedes:** `2026-07-21-vit-ln-to-bn-conversion-design.md` (parked; still valid as a fallback)

## Goal

Train ViT-Tiny/16 from random initialization on ImageNet-1k using **BatchNorm**
instead of LayerNorm, and compile it for the Hailo-10H.

## Why

LayerNorm is the only operator the Hailo Dataflow Compiler rejects for this
model — the rest of the ViT already compiles. LayerNorm computes its statistics
at runtime, so there is nothing to fold away. BatchNorm at inference is a fixed
per-channel affine transform that folds into the preceding layer, which is a
standard compiler pass.

Training from scratch rather than converting a pretrained model: a model trained
with BatchNorm from the start never develops a dependence on per-token
rescaling, so there is nothing to repair.

## Model

ViT-Tiny/16: patch 16, width 192, depth 12, 3 heads, 224×224 input, 197 tokens.
25 normalization layers (12 blocks × 2, plus the final norm).

**Normalization:** standard BatchNorm — one mean and one variance per channel,
pooled over images and tokens. For activations `(B, N, D)` this is
`nn.BatchNorm1d(D)` applied to the tensor reshaped to `(B·N, D)`. Conventional,
adds no parameters, folds exactly into the following `Linear`.

## Data

ImageNet-1k. Full train split (1.28M images) for training, full val (50k) for
evaluation. Top-1 accuracy is the metric.

## Recipe

DeiT-style:

- AdamW, lr `1e-3 × batch/512`, weight decay 0.05
- Cosine schedule, 5-epoch linear warmup
- Label smoothing 0.1, mixup 0.8, cutmix 1.0
- RandAugment `rand-m9-mstd0.5-inc1`, random erasing 0.25
- Gradient clipping at 1.0
- Mixed precision, channels-last
- EMA of weights, decay 0.9998

### BatchNorm-specific constraints

1. **Batch size ≥ 256, 512 preferred.** BatchNorm statistics are computed per
   batch; small batches make them noisy.
2. **No gradient accumulation.** It computes statistics per micro-batch while the
   optimizer sees the accumulated batch, silently changing BatchNorm's semantics.
   If memory forces a smaller batch, lower the batch and rescale the LR instead.
3. **Extend warmup to 10 epochs if training destabilizes.** BatchNorm divergence
   shows up as a loss spike in the first few epochs.
4. **Gradient clipping is mandatory**, not optional.

## Compute

Rented single 4090. The run is **data-bound** — JPEG decoding dominates, not
matrix multiplication — so the data pipeline is built first, not last.

Staged schedule: a 100-epoch run at 160px gives a real signal in about a day and
decides whether a full 300-epoch 224px run is warranted. Always report which
schedule produced a number.

## LayerNorm baseline: DeiT-Ti, already trained

Only one model is trained. The LayerNorm reference is the published **DeiT-Ti**
result: **72.2% top-1**, `deit_tiny_patch16_224` in `timm`.

DeiT-Ti is the correct comparison because it was trained on ImageNet-1k alone,
from scratch, for 300 epochs, with the recipe this spec adopts. Data, schedule
and augmentation all match; the only intended difference is LayerNorm versus
BatchNorm.

Do **not** compare against `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (~75.5%).
That checkpoint was pretrained on ImageNet-21k before fine-tuning on 1k — roughly
ten times the data — so measuring against it would attribute a data advantage to
LayerNorm.

## Gates

1. **Smoke run** — 2 epochs on 5% of the data must show falling loss and top-1
   above chance, before days of GPU time are committed.
2. **Fold equivalence** — the folded model must match the unfolded model to
   `atol=1e-4`.
3. **ONNX parity** — the exported graph must match PyTorch to `atol=1e-4` and
   contain no normalization node.

## Deliverables

- Top-1 on ImageNet-1k val, with the schedule stated
- Training curves
- A compiled `.hef` for Hailo-10H, plus on-device throughput
- The BatchNorm recipe that worked, including anything that diverged

## Acceptable outcomes

Within 1–2 points of a comparable LayerNorm ViT-Tiny is a success — deployable,
with a quantified cost. A larger gap is still a result if the training curves
show where it opened. Divergence that resists the mitigations above is a finding,
and the parked conversion approach becomes the fallback.

## References

- Dosovitskiy et al., 2020. *An Image is Worth 16x16 Words.*
- Touvron et al., 2021. *Training data-efficient image transformers (DeiT).*
- Shen et al., 2020. *PowerNorm: Rethinking Batch Normalization in Transformers.*
