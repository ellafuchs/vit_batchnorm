# Converting Pretrained LayerNorm ViTs to Foldable BatchNorm

**Date:** 2026-07-21
**Status:** Design approved, pending implementation plan

## Claim

A pretrained pre-LN Vision Transformer can be converted post-hoc into an
all-BatchNorm ViT that folds entirely into adjacent linear layers, recovering
most of its ImageNet-1k top-1 accuracy at a small fraction of pretraining cost.

## Motivation

LayerNorm computes its statistics at inference time, so it can never be folded
away. BatchNorm at inference is a fixed affine transform, so it folds exactly
into an adjacent `Linear`:

```
BN(x) = (γ/σ)·x + (β − γμ/σ)
W' = W · diag(γ/σ)
b' = b + W·(β − γμ/σ)
```

In a pre-norm ViT every norm is immediately followed by a `Linear` — the
attention norm feeds `qkv`, the MLP norm feeds `fc1`, and the final norm feeds
`head`. A successful conversion therefore does not merely match accuracy; it
**removes every normalization layer from the inference graph**. Norms are
memory-bound operations that break kernel fusion, and they occupy a larger
fraction of runtime at small model widths. The payoff is measured as wall-clock
latency, not FLOPs.

## The target artifact

The deliverable is a model with **zero normalization operations in the inference
graph**:

```
before:   x = x + Attn( BN(x) )        BN → qkv:  Linear(D → 3D)
          x = x + MLP(  BN(x) )        BN → fc1:  Linear(D → 4D)

after:    x = x + Attn( x )            qkv' absorbs the BN
          x = x + MLP(  x )            fc1' absorbs the BN
```

```python
s  = gamma / sqrt(running_var + eps)
W' = W * s                                    # scale columns
b' = b + W @ (beta - running_mean * s)
```

All 25 norms are absorbed into `qkv`, `fc1`, and `head`. Parameter count is
marginally *lower* (25×2×D affine parameters removed), outputs are identical to
floating-point tolerance, and 25 memory-bound operations — each a barrier to
kernel fusion — leave the graph.

Two consequences constrain the pipeline:

- **Folding is irreversible and must come last.** Once BN is absorbed it can no
  longer be trained. Fixed order: convert → reconstruct → distill → fold →
  benchmark.
- **Accuracy is reported from the folded model**, with fold equivalence asserted
  against the unfolded student. If the two disagree, the result is void.

## Why this is hard

### Calibration alone cannot work, in principle

```
LN(x)_t  = (x_t − μ(x_t)) / σ(x_t) · γ + β        μ, σ computed per token
BN(x)_t  = (x_t − μ_run)  / σ_run   · γ + β        μ, σ fixed constants
```

LayerNorm's normalizer is computed from each token at runtime, making LN a
*nonlinear* function of its input. BatchNorm at eval time divides by a single
constant vector shared across all tokens, making it *affine* — which is exactly
why it folds, and exactly why it cannot imitate LayerNorm. No choice of
`μ_run, σ_run` makes an affine map equal a nonlinear one; calibration merely
selects the best constants within the wrong function class.

This is why arms A and B are **controls that are expected to fail**, not
candidate methods. Their purpose is to quantify the gap so that recovery in arm
C can be attributed to the right cause.

The actual bet is therefore not *"can BatchNorm mimic LayerNorm?"* — it cannot —
but **"can the surrounding weights be re-fit so the network no longer requires
per-token normalization?"** That is why arms C and D modify weights while A and B
do not. The network is not asked to reproduce LN's function; it is asked to
become a network that does not need it. This is plausible because much of what LN
provides is scale control that a well-conditioned network may not strictly
require, but it is genuinely uncertain, and it is the research question.

### Why the change of axis damages a trained model

LayerNorm normalizes each token over the channel dimension. The BatchNorm
replacement normalizes each channel over batch and tokens jointly. That change
of axis is the crux: trained ViT residual streams carry outlier channels and
tokens with extreme norms (the `[CLS]` token and attention-sink tokens differ
from ordinary patch tokens by an order of magnitude). LayerNorm rescales each
token independently and the network learned to depend on that; BatchNorm
averages across tokens and erases it. This is the documented reason BatchNorm
has historically underperformed in transformers (Shen et al., 2020, *PowerNorm*).

Characterizing that damage is half the contribution.

## Scope

**Task:** image classification. Top-1 on ImageNet-1k val, stock `timm`
classification head. The research question concerns normalization inside the
encoder blocks and is task-independent, so classification is chosen because the
pretrained weights already do it and it yields one universally comparable number.

**Models:** all `timm` pretrained, published top-1 as the reference line.

| Model | Width | Blocks | Norms | Reference top-1 |
|---|---|---|---|---|
| ViT-Tiny/16 | 192 | 12 | 25 | ~75.5% |
| ViT-S/16 | 384 | 12 | 25 | ~81.4% |
| ViT-B/16 | 768 | 12 | 25 | ~81.1% |

ViT-Tiny is primary; S/16 and B/16 form the scaling arm.

**Data:**
- Eval: full ImageNet-1k val, 50k images.
- Calibration/conversion: subset of ImageNet train, ~10k images (~10/class) for
  reconstruction, ~100k for global distillation. **Label-free** — distillation
  targets the teacher's outputs, so images suffice.

**Compute:** rented GPU for both conversion and evaluation. ImageNet is streamed
from `timm/imagenet-1k-wds` on HuggingFace or cached to a persistent volume; the
full 150 GB training corpus is never required.

**Out of scope:** training BN ViTs from scratch; architecture search; other
normalization schemes (RMSNorm, PowerNorm, DyT) except as related work;
segmentation or detection heads.

## Architecture

ViT-Tiny, `d=192`, 12 blocks:

```
input        (B, 3, 224, 224)
patch embed  Conv2d(3, 192, k=16, s=16) → (B, 192, 14, 14) → (B, 196, 192)
+ CLS + pos                              → (B, 197, 192)

per block ×12:
    x = x + Attn( LN1(x) )      LN1 → qkv:  Linear(192 → 576)
    x = x + MLP(  LN2(x) )      LN2 → fc1:  Linear(192 → 768)

final        LN_f(x)[:, 0]      LN_f → head: Linear(192 → 1000)
```

25 norms total (12×2 + 1), all foldable.

**The swap:** tensors are `(B, N, D)`. The replacement is `BatchNorm1d(D)` applied
to the tensor reshaped to `(B·N, D)`. Reshaping is preferred over transposing to
`(B, D, N)` for simplicity; the statistics are identical.

## Components

Five independent, separately testable modules:

| Module | Responsibility | Depends on |
|---|---|---|
| `norm_swap` | Replace LN modules with BN in a `timm` ViT, in-place, selectable by layer index. Copies LN's learned `γ`/`β` into the BN's `weight`/`bias` — same shape `(D,)`, same role, a far better initialization than the default 1/0. Pure surgery, no training. | `timm` |
| `calibrate` | Forward-only pass populating BN running statistics. No loss, no gradients, no optimizer — BatchNorm updates `running_mean`/`running_var` as a side effect of the forward pass in `train()` mode. Measurement, not learning. | `norm_swap` |
| `reconstruct` | Block-wise: cache teacher activations for block *k*, optimize student block *k* to match. Loops over blocks. | `norm_swap` |
| `distill` | Short global fine-tune, student against frozen teacher. | `reconstruct` |
| `evaluate` | Top-1 on val, fold correctness verification, latency benchmark. | — |

### Data flow

```
timm pretrained ViT ──┬── frozen teacher ──→ activations, logits
                      │
                      └── norm_swap ──→ BN student
                                         ├─ arm A: eval directly
                                         ├─ arm B: calibrate → eval
                                         ├─ arm C: reconstruct → eval
                                         └─ arm D: distill → eval → fold → latency
```

### Non-negotiable correctness gates

1. **Teacher reproduction.** Before any conversion work, confirm the unmodified
   `timm` ViT-Tiny scores ~75.5% top-1 on val. Failure means the data pipeline is
   wrong and nothing downstream is trustworthy. Use
   `timm.data.resolve_data_config` for the eval transform (bicubic, `crop_pct`
   0.9) rather than hand-rolling it — mismatched preprocessing is the usual cause
   of failed reproduction.
2. **Fold equivalence.** After folding BN into the following `Linear`, assert
   outputs match the unfolded model to within 1e-4. If this fails, every latency
   number is meaningless. This is the single most important test in the project.

## Experimental protocol

### Experiment 0 — per-layer damage (run first)

Convert exactly one LN to BN, calibrate it, evaluate. Repeat for all 25. **No
training** — calibration is forward passes only. Roughly 25 eval passes on
ViT-Tiny, under an hour; run the sweep on a fixed 10k subset of val for speed and
confirm interesting layers on the full set.

Calibration-only is the correct diagnostic here *precisely because* it is
guaranteed not to repair anything. It isolates the damage caused by the change of
function class, with no confound from retraining. It measures harm; it does not
attempt a fix.

Identifies which norms are fragile before any method is built. Prediction: `LN_f`
and the early blocks hurt most. Produces a diagnostic figure and shapes all
downstream choices.

### The four arms

Each arm is a rung on a ladder, not a competing alternative. Each contributes a
point to the primary figure.

| Arm | Method | Expected top-1 |
|---|---|---|
| A | Naive swap, no adaptation | near chance |
| B | + BN statistic recalibration | poor but non-trivial |
| C | + block-wise reconstruction | the interesting result |
| D | + global distillation fine-tune | best |

**Arm A.** Replace all 25 norms, no adaptation, evaluate.

**Arm B.** Reset running statistics, set `momentum=None` for a cumulative
average, run ~150 forward-only batches of 64 in `train()` mode, evaluate in
`eval()` mode. No gradients.

**Arm C.** For each block *k* in order: cache the **student's** propagated input
and the **teacher's** output, then optimize the full student block to minimize
MSE against the teacher output. Student-input/teacher-output is deliberate — it
lets each block correct drift accumulated upstream, which teacher forcing would
conceal. Adam, lr 1e-4, ~1000 steps, batch 64, gradient clip 1.0. The whole block
is trainable, not only BN's affine parameters: it is cheap and strictly more
expressive.

This follows the block-reconstruction approach used in post-training
quantization (AdaRound, BRECQ). It transfers because the problem has the same
shape: a frozen reference network, a locally perturbed operator, and a small
calibration set. It is also more stable than global fine-tuning, which would ask
the optimizer to repair a fully broken network in one shot.

**Arm D.** Student against frozen teacher: KL divergence on logits at
temperature 2, plus feature MSE on block outputs. AdamW, lr 1e-5, cosine
schedule, low weight decay, 1–3 epochs over ~100k images. Gradient clipping is
mandatory; BatchNorm instability manifests as loss spikes.

Then fold and benchmark.

### Metrics

Logged for every arm: top-1 accuracy, norms remaining in the inference graph,
GPU latency (warmed up, CUDA-synchronized, on a single named GPU), CPU latency,
parameter count.

CPU latency is included because removing normalization matters most in exactly
the deployment settings where a ViT-Tiny would be used.

### Ablations

Cheap on ViT-Tiny:
- Calibration set size: 1k / 10k / 50k
- BN affine parameters only vs. full block trainable
- Retaining `LN_f` as LayerNorm rather than converting it
- Reconstruction step count

### Scaling arm

Run the winning recipe unchanged on ViT-S/16 and ViT-B/16. Report recovered top-1
against model size. Whether conversion becomes easier or harder with scale is
open — smaller models have less capacity slack to absorb the change in
normalization axis, but larger models have more structure invested in
per-token scaling. Either outcome is a genuine finding, and the experiment is
nearly free given the other arms.

## Deliverables

**Primary figure:** accuracy recovered vs. latency saved, points A→D, one line
per model scale.

**Secondary figures:** per-layer damage (Experiment 0); recovered top-1 vs. model
size (scaling arm).

**Artifact:** the conversion recipe, as reproducible code.

## Acceptable outcomes

A result of the form "24 of 25 norms folded, `LN_f` retained, 1.5 points of top-1
surrendered, *X*% latency reduction" is a success. Full conversion at zero
accuracy cost is not required for the project to be worth reporting; a
well-characterized trade-off curve is the contribution.

A negative result — conversion is not recoverable at any point on the ladder —
is publishable provided Experiment 0 and the ablations explain why.

## Future work

The recipe operates entirely on encoder blocks and is head-agnostic, so it
transfers unchanged to ViT backbones used for segmentation or detection. Not
built here.

## References

- Dosovitskiy et al., 2020. *An Image is Worth 16x16 Words.*
- Shen et al., 2020. *PowerNorm: Rethinking Batch Normalization in Transformers.*
- Nagel et al., 2020. *AdaRound: Up or Down? Adaptive Rounding for Post-Training Quantization.*
- Li et al., 2021. *BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction.*
