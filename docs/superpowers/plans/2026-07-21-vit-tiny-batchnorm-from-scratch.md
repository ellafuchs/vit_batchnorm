# ViT-Tiny BatchNorm From-Scratch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train ViT-Tiny/16 with BatchNorm from random initialization on ImageNet-1k, reach accuracy near DeiT-Ti's 72.2%, and compile it for Hailo-10H.

**Architecture:** A `BatchNorm` module (stock `nn.BatchNorm1d` plus a reshape) is registered as a `timm` model variant, then trained with `timm`'s own training script using DeiT hyperparameters. Delegating the recipe to `timm` avoids reimplementing mixup, cutmix, RandAugment, EMA and cosine scheduling — the largest source of silent bugs in a from-scratch ViT run.

**Tech Stack:** Python 3.11, PyTorch 2.x, `timm`, `pytest`, ONNX. Single rented 4090.

## Global Constraints

- Python 3.11 (the local `/opt/homebrew/bin/python3` is 3.14 and has no PyTorch wheels)
- Model: ViT-Tiny/16 — patch 16, width 192, depth 12, 3 heads, 224×224, 197 tokens, 25 norms
- Normalization: standard BatchNorm — `nn.BatchNorm1d(D)` on the tensor reshaped to `(B·N, D)`, one statistic pair per channel
- Baseline: published DeiT-Ti, **72.2%** top-1 (`deit_tiny_patch16_224`). Never compare against `vit_tiny_patch16_224.augreg_in21k_ft_in1k` — it saw ImageNet-21k
- **Batch size ≥ 256, 512 preferred** — BatchNorm statistics come from the batch
- **No gradient accumulation** — it changes BatchNorm's semantics silently
- **Gradient clipping at 1.0, always**
- Data: full ImageNet-1k, `ImageFolder` layout, train + val
- ONNX export: batch 1, static shapes, opset 17
- Deployment target: Hailo-10H

---

## Why each gate exists

The full run costs days. Every gate before it is there so you never spend a day discovering something a minute would have caught.

| Gate | Task | Predicted | If it fails |
|---|---|---|---|
| DeiT-Ti reproduction | 1 | 72.2% ± 0.3 | Data pipeline is wrong. Every number afterwards is wrong the same way. Stop. |
| 25 norms, all BatchNorm | 2 | exactly 25, zero LayerNorm | The model isn't the one you think you're training. Stop. |
| Smoke run | 3 | loss falls, top-1 above chance | Recipe or pipeline is broken. Do not launch the full run. |
| Fold equivalence | 5 | max diff < 1e-4 | The exported model is a different function. Stop. |

---

## File Structure

```
vit_batchnorm/
├── pyproject.toml
├── src/vitbn/
│   ├── __init__.py
│   ├── norm.py            BatchNorm module
│   ├── models.py          registered timm variant, fold pairs
│   ├── fold.py            BatchNorm -> Linear folding
│   └── export.py          ONNX export + gates
├── scripts/
│   ├── smoke.sh           2-epoch pipeline check
│   ├── train.sh           the full run
│   └── export_hailo.py
├── tests/
│   ├── test_norm.py
│   ├── test_models.py
│   └── test_fold.py
└── results/
```

Small on purpose. The training loop, augmentation and schedule all come from `timm`; the only original code is the norm layer, the model registration, and the export path.

---

## Task 1: Environment, data, and the DeiT-Ti reproduction gate

**Why first:** the data pipeline is both the correctness foundation and the performance bottleneck. Reproducing DeiT-Ti's published 72.2% proves ImageNet is laid out correctly and the eval path is sound — before you spend days training against it.

**Files:** Create `pyproject.toml`, `src/vitbn/__init__.py`

- [ ] **Step 1: Create the project and install**

```toml
# pyproject.toml
[project]
name = "vitbn"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.2", "torchvision>=0.17", "timm>=1.0.9",
    "onnx>=1.16", "onnxruntime>=1.18", "matplotlib>=3.8",
]
[project.optional-dependencies]
dev = ["pytest>=8.0"]
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
[tool.pytest.ini_options]
testpaths = ["tests"]
```

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
git clone https://github.com/huggingface/pytorch-image-models.git timm-src
```

`timm-src` gives you `train.py` and `validate.py`. Expected: install succeeds, `import timm` works.

- [ ] **Step 2: Download ImageNet into ImageFolder layout**

Needs a HuggingFace account with ImageNet-1k terms accepted — **do this first, approval is not always instant.**

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset \
    --local-dir /data/imagenet_raw
```

Extract to `$IMAGENET/train/<wnid>/*.JPEG` and `$IMAGENET/val/<wnid>/*.JPEG`. If val comes out flat, regroup it with the standard `valprep.sh`.

Verify:
```bash
ls $IMAGENET/train | wc -l    # expect 1000
ls $IMAGENET/val   | wc -l    # expect 1000
find $IMAGENET/val -name "*.JPEG" | wc -l   # expect 50000
```

- [ ] **Step 3: Run the reproduction gate**

```bash
.venv/bin/python timm-src/validate.py $IMAGENET \
    --model deit_tiny_patch16_224 --pretrained --batch-size 256
```

Expected: **top-1 ≈ 72.2%**, a few minutes on a 4090.

If it is far off, the data layout or label mapping is wrong. Every number you produce afterwards would be wrong in the same direction without looking wrong. Do not proceed.

Record the exact number — it is your baseline, in preference to the published figure.

- [ ] **Step 4: Measure data-loading throughput**

```bash
.venv/bin/python -c "
import time, torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
ds = ImageFolder('$IMAGENET/train', transform=T.Compose([
    T.RandomResizedCrop(224), T.ToTensor()]))
dl = DataLoader(ds, batch_size=512, num_workers=16, shuffle=True, pin_memory=True)
t = time.time(); n = 0
for i, (x, y) in enumerate(dl):
    n += x.shape[0]
    if i == 20: break
print(f'{n / (time.time() - t):.0f} img/s')
"
```

Expected: 1,500–4,000 img/s depending on CPU. **This number sets your training time.** At 2,000 img/s a 300-epoch run is 384M/2000 ≈ 53 hours; at 4,000 it is 27.

If below ~1,500 img/s, fix it before training rather than after — raise `--workers`, or pre-resize the dataset to shorter-side 160px, which typically doubles throughput.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/vitbn/__init__.py
git commit -m "feat: project setup; DeiT-Ti reproduction gate passes"
```

---

## Task 2: The BatchNorm ViT model

**Why:** this is the only architectural change in the project. It has to be exactly right, and it has to be verifiable without training anything.

**Files:** Create `src/vitbn/norm.py`, `src/vitbn/models.py`; Test `tests/test_norm.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `class BatchNorm(nn.Module)` with attribute `.bn: nn.BatchNorm1d`; timm model `vit_tiny_patch16_224_bn`; `fold_pairs(model) -> list[tuple[str, str]]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_norm.py
import torch
from vitbn.norm import BatchNorm


def test_preserves_shape():
    assert BatchNorm(192)(torch.randn(4, 197, 192)).shape == (4, 197, 192)


def test_statistics_are_per_channel_pooled_over_images_and_tokens():
    """Standard BatchNorm: one mean/var per channel, pooling everything else."""
    m = BatchNorm(8)
    m.train()
    y = m(torch.randn(16, 20, 8) * 5 + 3)
    flat = y.reshape(-1, 8)
    assert torch.allclose(flat.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(flat.std(0, unbiased=False), torch.ones(8), atol=1e-4)


def test_has_one_stat_pair_per_channel():
    m = BatchNorm(192)
    assert m.bn.running_mean.shape == (192,)
    assert m.bn.running_var.shape == (192,)


def test_eval_mode_is_batch_independent():
    """The property Hailo needs: at inference, output does not depend on what
    else is in the batch. Batch size 1 must equal batch size 8."""
    m = BatchNorm(16)
    m.train()
    m(torch.randn(32, 197, 16))      # populate running stats
    m.eval()
    x = torch.randn(8, 197, 16)
    alone = m(x[:1])
    together = m(x)[:1]
    assert torch.allclose(alone, together, atol=1e-6)
```

```python
# tests/test_models.py
import torch
import torch.nn as nn
from vitbn.models import fold_pairs, make_bn_vit_tiny
from vitbn.norm import BatchNorm


def test_model_runs_and_has_right_output_shape():
    m = make_bn_vit_tiny()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 1000)


def test_exactly_25_batchnorms_and_no_layernorm():
    m = make_bn_vit_tiny()
    assert sum(isinstance(x, BatchNorm) for x in m.modules()) == 25
    assert not any(isinstance(x, nn.LayerNorm) for x in m.modules())


def test_parameter_count_matches_vit_tiny():
    """BatchNorm has the same affine parameter count as LayerNorm, so the
    model must not have grown."""
    import timm
    ref = timm.create_model("deit_tiny_patch16_224", pretrained=False)
    got = make_bn_vit_tiny()
    a = sum(p.numel() for p in ref.parameters())
    b = sum(p.numel() for p in got.parameters())
    assert abs(a - b) < 1000, f"{a} vs {b}"


def test_every_norm_is_followed_by_a_linear_with_bias():
    """Required for folding, which is required for Hailo."""
    m = make_bn_vit_tiny()
    for norm_path, lin_path in fold_pairs(m):
        assert isinstance(m.get_submodule(norm_path), BatchNorm)
        lin = m.get_submodule(lin_path)
        assert isinstance(lin, nn.Linear) and lin.bias is not None


def test_fc_norm_is_identity():
    """The final norm folds into head; anything between them would break it."""
    assert isinstance(make_bn_vit_tiny().fc_norm, nn.Identity)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.norm'`

- [ ] **Step 3: Implement `norm.py`**

```python
# src/vitbn/norm.py
"""Standard BatchNorm for ViT activations.

One mean and one variance per channel, pooled over images and tokens --
the conventional formulation. At inference these are stored constants, so
the layer is a fixed affine map that folds into the following Linear.
LayerNorm cannot do this, which is why Hailo rejects it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BatchNorm(nn.Module):
    """Drop-in replacement for `nn.LayerNorm(D)` in a ViT.

    ViT activations are (B, N, D) with channels last; `nn.BatchNorm1d`
    expects channels at dim 1. Reshaping to (B*N, D) puts them there and
    pools statistics over images and tokens jointly. The normalization
    itself is stock PyTorch BatchNorm.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return self.bn(x.reshape(b * n, d)).reshape(b, n, d)

    def extra_repr(self) -> str:
        return f"num_features={self.bn.num_features}"
```

- [ ] **Step 4: Implement `models.py`**

```python
# src/vitbn/models.py
"""ViT-Tiny with BatchNorm, registered as a timm model so timm's own
training script can drive it."""
from __future__ import annotations

import torch.nn as nn
from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer

from .norm import BatchNorm


def make_bn_vit_tiny(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-Tiny/16, identical to DeiT-Ti except norm_layer."""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3,
        num_classes=num_classes, norm_layer=BatchNorm, **kwargs,
    )


@register_model
def vit_tiny_patch16_224_bn(pretrained: bool = False, **kwargs):
    """Registered so `timm-src/train.py --model vit_tiny_patch16_224_bn` works.

    pretrained is unsupported: no BatchNorm ViT checkpoint exists, which is
    the point of the project.
    """
    if pretrained:
        raise ValueError("no pretrained weights exist for this variant")
    return make_bn_vit_tiny(**kwargs)


def fold_pairs(model) -> list[tuple[str, str]]:
    """(norm_path, linear_path) -- each norm's fold target.

    In a pre-norm ViT every norm feeds exactly one Linear.
    """
    pairs = [(f"blocks.{i}.norm1", f"blocks.{i}.attn.qkv")
             for i in range(len(model.blocks))]
    pairs += [(f"blocks.{i}.norm2", f"blocks.{i}.mlp.fc1")
              for i in range(len(model.blocks))]
    pairs.append(("norm", "head"))
    return sorted(pairs)


def set_submodule(model: nn.Module, path: str, new: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, name, new)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS, 9 tests.

`test_exactly_25_batchnorms_and_no_layernorm` is the gate — if it fails, `timm` is applying `norm_layer` somewhere other than you expect and you would train a different model than you think.

- [ ] **Step 6: Commit**

```bash
git add src/vitbn/norm.py src/vitbn/models.py tests/
git commit -m "feat: BatchNorm ViT-Tiny registered as a timm model"
```

---

## Task 3: Smoke run

**Why:** the full run costs days. This costs fifteen minutes and catches almost everything that could waste them — broken augmentation, a model that won't converge, a learning rate that diverges immediately, `timm` not finding your registered model.

**Files:** Create `scripts/smoke.sh`

- [ ] **Step 1: Write the smoke script**

```bash
#!/usr/bin/env bash
# scripts/smoke.sh -- 2 epochs on a small subset. Proves the pipeline works.
#
# Expected: train loss falls from ~6.9 (ln 1000 = chance) and top-1 rises
# above 0.1%. Absolute numbers will be terrible -- that is fine, the point
# is that learning is happening at all.
#
# If loss stays flat at 6.9, the model is not learning: check LR and that
# --model resolved to the BatchNorm variant.
# If loss goes to NaN, BatchNorm has diverged: raise --warmup-epochs.
set -euo pipefail
: "${IMAGENET:?set IMAGENET to your dataset root}"

PYTHONPATH=src .venv/bin/python timm-src/train.py "$IMAGENET" \
    --model vit_tiny_patch16_224_bn \
    --epochs 2 --batch-size 256 --workers 8 \
    --opt adamw --lr 5e-4 --weight-decay 0.05 \
    --sched cosine --warmup-epochs 0 \
    --smoothing 0.1 --clip-grad 1.0 --amp \
    --train-crop-mode rrc --input-size 3 160 160 \
    --output results/smoke --experiment smoke \
    --log-interval 20
```

`PYTHONPATH=src` is what makes `timm` see the registered model — the `@register_model` decorator only runs if `vitbn.models` is imported.

- [ ] **Step 2: Ensure the registration is imported**

```python
# src/vitbn/__init__.py
"""Importing this package registers the BatchNorm ViT with timm."""
from . import models  # noqa: F401
from .norm import BatchNorm  # noqa: F401
```

`timm-src/train.py` must import it. Simplest reliable route — add to the top of `timm-src/train.py`:

```python
import vitbn  # registers vit_tiny_patch16_224_bn
```

- [ ] **Step 3: Run the smoke test**

Run: `chmod +x scripts/smoke.sh && IMAGENET=$IMAGENET ./scripts/smoke.sh`
Expected: two epochs, ~15 minutes. Loss falls from ~6.9. Top-1 above 0.1%.

**Do not launch the full run until this passes.** Interpretation:
- *Loss flat at 6.9* — not learning. Check the model resolved correctly and the LR is non-zero.
- *Loss NaN in the first hundred steps* — BatchNorm divergence. Add `--warmup-epochs 3` and retry.
- *Crash on unknown model* — the registration was not imported; check `PYTHONPATH` and the `import vitbn` line.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke.sh src/vitbn/__init__.py
git commit -m "feat: smoke run; pipeline verified end to end"
```

---

## Task 4: The full training run

**Why:** this is the experiment. Everything before it exists to make this one launch trustworthy.

**Files:** Create `scripts/train.sh`

- [ ] **Step 1: Write the training script**

```bash
#!/usr/bin/env bash
# scripts/train.sh -- ViT-Tiny + BatchNorm, DeiT recipe, ImageNet-1k.
#
# Baseline to beat/approach: DeiT-Ti at 72.2% top-1, same data, same recipe,
# same schedule -- differing only in LayerNorm vs BatchNorm.
#
# BatchNorm-specific choices, all deliberate:
#   --batch-size 512   statistics come from the batch; small batches are noisy
#   --clip-grad 1.0    mandatory; BatchNorm divergence appears as a loss spike
#   --warmup-epochs 5  raise to 10 if the loss spikes early
#   NO gradient accumulation -- it computes statistics per micro-batch while
#   the optimizer sees the accumulated batch, silently changing semantics.
#   If memory forces a smaller batch, lower it and rescale lr = 1e-3*batch/512.
set -euo pipefail
: "${IMAGENET:?set IMAGENET to your dataset root}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-512}"
RES="${RES:-224}"
LR=$(python3 -c "print(1e-3 * ${BATCH} / 512)")

PYTHONPATH=src .venv/bin/python timm-src/train.py "$IMAGENET" \
    --model vit_tiny_patch16_224_bn \
    --epochs "$EPOCHS" --batch-size "$BATCH" --workers 16 \
    --opt adamw --lr "$LR" --weight-decay 0.05 \
    --sched cosine --warmup-epochs 5 --min-lr 1e-5 \
    --aa rand-m9-mstd0.5-inc1 --mixup 0.8 --cutmix 1.0 \
    --smoothing 0.1 --reprob 0.25 \
    --clip-grad 1.0 --amp --channels-last \
    --model-ema --model-ema-decay 0.9998 \
    --input-size 3 "$RES" "$RES" \
    --output results --experiment "bn_e${EPOCHS}_r${RES}" \
    --checkpoint-hist 3 --log-interval 100
```

- [ ] **Step 2: Launch the staged run first**

A 100-epoch run at 160px gives a real signal in about a day and tells you whether the full schedule is worth it.

Run: `EPOCHS=100 RES=160 ./scripts/train.sh`
Expected: ~1 day. Top-1 in the mid-60s. `timm` writes `summary.csv` per epoch, so progress is checkable at any point.

Watch the first two epochs. A NaN or a loss spike means BatchNorm instability — kill it, set `--warmup-epochs 10`, relaunch. Better to lose an hour than a day.

- [ ] **Step 3: Launch the full run if the staged run looks healthy**

Run: `EPOCHS=300 RES=224 ./scripts/train.sh`
Expected: 2–3 days. Target top-1 near **72.2%**.

- [ ] **Step 4: Record the result**

```bash
tail -3 results/bn_e300_r224/summary.csv
git add results/bn_e300_r224/summary.csv
git commit -m "results: ViT-Tiny BatchNorm 300ep, top-1 <N>% (DeiT-Ti 72.2%)"
```

---

## Task 5: Fold, export, and compile for Hailo-10H

**Why:** the trained model is only useful if it reaches the device. Two gates run before the compiler, because a DFC rejection is slow and its diagnostics are far less specific.

**Files:** Create `src/vitbn/fold.py`, `src/vitbn/export.py`, `scripts/export_hailo.py`; Test `tests/test_fold.py`

- [ ] **Step 1: Write the failing fold tests**

```python
# tests/test_fold.py
import copy
import pytest
import torch
import torch.nn as nn

from vitbn.fold import assert_fold_equivalent, fold_bn_into_linear, fold_model
from vitbn.models import make_bn_vit_tiny
from vitbn.norm import BatchNorm


def test_fold_is_exact():
    torch.manual_seed(0)
    bn, lin = nn.BatchNorm1d(16), nn.Linear(16, 7)
    with torch.no_grad():
        bn.running_mean.normal_(1.0, 2.0)
        bn.running_var.uniform_(0.5, 3.0)
        bn.weight.normal_(1.0, 0.5)
        bn.bias.normal_()
    bn.eval()
    x = torch.randn(32, 16)
    expected = lin(bn(x))
    folded = copy.deepcopy(lin)
    fold_bn_into_linear(bn, folded)
    assert torch.allclose(expected, folded(x), atol=1e-5)


def test_fold_scales_input_channels_not_output_channels():
    """The transposition bug: W is (out, in) and BatchNorm precedes the
    Linear, so the scale applies along `in`. Getting this backwards yields a
    wrong model that still runs and still produces plausible logits."""
    bn = nn.BatchNorm1d(4)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        bn.bias.zero_(); bn.running_mean.zero_()
        bn.running_var.fill_(1.0 - bn.eps)
    bn.eval()
    lin = nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4)); lin.bias.zero_()
    fold_bn_into_linear(bn, lin)
    assert torch.allclose(lin.weight.diag(),
                          torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-4)


def test_fold_rejects_train_mode():
    bn = nn.BatchNorm1d(8); bn.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        fold_bn_into_linear(bn, nn.Linear(8, 4))


def test_folded_vit_matches_unfolded():
    torch.manual_seed(0)
    m = make_bn_vit_tiny()
    m.train(); m(torch.randn(8, 3, 224, 224)); m.eval()   # populate stats
    assert assert_fold_equivalent(m, torch.randn(2, 3, 224, 224)) < 1e-4


def test_fold_removes_every_norm():
    m = make_bn_vit_tiny()
    m.train(); m(torch.randn(8, 3, 224, 224)); m.eval()
    fold_model(m)
    assert not any(isinstance(x, (BatchNorm, nn.BatchNorm1d, nn.LayerNorm))
                   for x in m.modules())
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_fold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.fold'`

- [ ] **Step 3: Implement `fold.py`**

```python
# src/vitbn/fold.py
"""Fold BatchNorm into the Linear that consumes it.

At inference BatchNorm is a per-channel affine map, so composing it with a
following Linear yields another Linear. Exact algebra, not an approximation.

    BN(x) = s*x + shift,  s = gamma/sqrt(var+eps), shift = beta - mean*s
    Linear(BN(x)) = (W*s) x + (b + W @ shift)
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .models import fold_pairs, set_submodule
from .norm import BatchNorm


@torch.no_grad()
def fold_bn_into_linear(bn: nn.BatchNorm1d, lin: nn.Linear) -> None:
    if bn.training:
        raise RuntimeError(
            "BatchNorm is in train mode; folding would capture batch "
            "statistics rather than running statistics. Call .eval() first."
        )
    if lin.bias is None:
        raise RuntimeError("fold target has no bias; cannot absorb the shift")

    # fp64 once, so the equivalence check is limited by the model's precision.
    w = lin.weight.data.double()
    b = lin.bias.data.double()
    s = bn.weight.data.double() / torch.sqrt(bn.running_var.data.double() + bn.eps)
    shift = bn.bias.data.double() - bn.running_mean.data.double() * s

    b = b + w @ shift              # uses the ORIGINAL W, before scaling
    w = w * s.unsqueeze(0)         # W is (out, in); scale INPUT channels

    lin.weight.data.copy_(w.to(lin.weight.dtype))
    lin.bias.data.copy_(b.to(lin.bias.dtype))


@torch.no_grad()
def fold_model(model: nn.Module) -> nn.Module:
    """Fold all 25 norms, replacing each with Identity. Irreversible."""
    model.eval()
    for norm_path, lin_path in fold_pairs(model):
        norm = model.get_submodule(norm_path)
        if isinstance(norm, nn.Identity):
            continue
        fold_bn_into_linear(norm.bn, model.get_submodule(lin_path))
        set_submodule(model, norm_path, nn.Identity())
    return model


@torch.no_grad()
def assert_fold_equivalent(model: nn.Module, x: torch.Tensor,
                           atol: float = 1e-4) -> float:
    """Fold a copy and confirm it computes the same function.

    A failure means the exported model is not the model you evaluated. Do not
    relax the tolerance to make this pass.
    """
    model.eval()
    reference = model(x)
    got = fold_model(copy.deepcopy(model))(x)
    max_diff = (reference - got).abs().max().item()
    if max_diff > atol:
        raise RuntimeError(
            f"GATE FAILED: folded model differs by {max_diff:.3e} > {atol}. "
            "Check fold axis, bias computed pre-scaling, eval mode, eps."
        )
    return max_diff
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_fold.py -v`
Expected: PASS, 5 tests. `test_folded_vit_matches_unfolded` should report ~1e-6.

- [ ] **Step 5: Implement `export.py`**

```python
# src/vitbn/export.py
"""ONNX export and pre-compilation gates for Hailo-10H."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch

NORM_OPS = {"LayerNormalization", "BatchNormalization", "InstanceNormalization",
            "GroupNormalization", "SimplifiedLayerNormalization"}


def export_onnx(model, path, input_size=(3, 224, 224), opset: int = 17) -> Path:
    """Static shapes, batch 1 -- Hailo rejects dynamic axes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, torch.randn(1, *input_size), str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def assert_no_norm_nodes(path, raise_on_found: bool = True) -> list[str]:
    """Normalization nodes surviving export. Empty is what Hailo needs."""
    graph = onnx.load(str(path)).graph
    found = [f"{n.op_type}:{n.name}" for n in graph.node if n.op_type in NORM_OPS]
    if found and raise_on_found:
        raise RuntimeError(f"GATE FAILED: {len(found)} norm node(s): {found[:5]}")
    return found


def assert_onnx_matches_torch(model, path, x, atol: float = 1e-4) -> float:
    """Guards against export-time graph rewrites changing behaviour."""
    import onnxruntime as ort
    model.eval()
    with torch.no_grad():
        expected = model(x).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    if max_diff > atol:
        raise RuntimeError(f"GATE FAILED: ONNX differs by {max_diff:.3e} > {atol}")
    return max_diff
```

- [ ] **Step 6: Write the export script**

```python
# scripts/export_hailo.py
"""Export the trained BatchNorm ViT for Hailo-10H.

Two graphs:
  vit_bn.onnx         BatchNorm intact -- the DFC folds it. Deployment artifact.
  vit_bn_folded.onnx  folded in PyTorch, no norm nodes. Verification artifact.

Folding is exact algebra in both, so the two must agree on device.
"""
import argparse
from pathlib import Path

import torch

from vitbn.export import assert_no_norm_nodes, assert_onnx_matches_torch, export_onnx
from vitbn.fold import fold_model
from vitbn.models import make_bn_vit_tiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/hailo")
    args = ap.parse_args()

    model = make_bn_vit_tiny()
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd.get("state_dict_ema") or sd.get("state_dict") or sd)
    model.eval()

    out = Path(args.out)
    unfolded = export_onnx(model, out / "vit_bn.onnx")
    print(f"exported {unfolded} (BatchNorm intact, DFC folds it)")

    x = torch.randn(1, 3, 224, 224)
    print(f"  ONNX parity: {assert_onnx_matches_torch(model, unfolded, x):.3e}")

    folded_model = fold_model(model)
    folded = export_onnx(folded_model, out / "vit_bn_folded.onnx")
    assert_no_norm_nodes(folded)
    print(f"exported {folded}")
    print("  GATE PASSED: no normalization nodes in graph")
    print(f"  ONNX parity: {assert_onnx_matches_torch(folded_model, folded, x):.3e}")

    print(f"""
On the machine with the Hailo Dataflow Compiler:

  hailo parser onnx {unfolded} --hw-arch hailo10h
  hailo optimize  vit_bn.har --hw-arch hailo10h
  hailo compiler  vit_bn_optimized.har --hw-arch hailo10h

If the parser rejects an operator, record its exact name -- the premise is
that LayerNorm was the only blocker, and a different rejection disproves it.
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Export and compile**

Run: `PYTHONPATH=src .venv/bin/python scripts/export_hailo.py --checkpoint results/bn_e300_r224/model_best.pth.tar`
Expected: both graphs written, all gates pass.

Then run the three `hailo` commands. Expected: a `.hef`.

- [ ] **Step 8: Commit**

```bash
git add src/vitbn/fold.py src/vitbn/export.py scripts/export_hailo.py tests/test_fold.py
git commit -m "feat: fold, ONNX export with gates, Hailo-10H compilation"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| ViT-Tiny/16, 25 norms | 2 |
| Standard BatchNorm, per-channel | 2 (`test_statistics_are_per_channel...`) |
| DeiT-Ti 72.2% baseline | 1 (reproduction gate) |
| Batch ≥ 256 | 4 (default 512) |
| No gradient accumulation | 4 (documented in script header) |
| Gradient clipping mandatory | 3, 4 (`--clip-grad 1.0`) |
| Warmup extension on instability | 3, 4 (documented) |
| DeiT recipe | 4 |
| Data-bound pipeline measured first | 1 (step 4) |
| Staged 100ep/160px schedule | 4 (step 2) |
| Smoke run gate | 3 |
| Fold equivalence gate | 5 |
| ONNX parity + no-norm-node gates | 5 |
| Hailo-10H compilation | 5 |

**Not covered, deliberately:** the optional matched LayerNorm run (the spec resolves this in favour of published DeiT-Ti) and on-device throughput measurement, which needs hardware in hand.

**Type consistency:** `BatchNorm.bn` is the attribute name in every consumer. `fold_pairs`/`set_submodule` are defined in `models.py` and imported identically by `fold.py`. `make_bn_vit_tiny()` signature matches its use in tests, the registration, and the export script.

**Placeholder scan:** none.
