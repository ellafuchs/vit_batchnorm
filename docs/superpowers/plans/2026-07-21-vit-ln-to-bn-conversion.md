# LN→BN ViT Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a pretrained `timm` Vision Transformer's 25 LayerNorms into BatchNorms that fold exactly into the following linear layers, producing an inference graph with zero normalization operations and no accuracy loss.

**Architecture:** A frozen pretrained ViT acts as teacher. Norm surgery replaces LayerNorm modules with a `BatchNorm` wrapper. Recovery proceeds up a four-rung ladder — naive swap, statistic calibration, block-wise reconstruction against teacher activations, global distillation — after which BatchNorm is folded into `qkv`/`fc1`/`head` and the folded model is benchmarked. Every stage is gated by a numeric check with a predicted value.

**Tech Stack:** Python 3.11, PyTorch 2.x, `timm`, `pytest`, `matplotlib`. CUDA GPU (RTX 4090 sufficient).

## Global Constraints

- Python 3.11 (not 3.14 — PyTorch wheels lag; the local `/opt/homebrew/bin/python3` is 3.14 and will not work)
- All checkpoints from the `augreg_in21k_ft_in1k` family so the scaling arm varies only model size
- Eval transform obtained from `timm.data.resolve_data_config` — never hand-rolled
- Primary model `vit_tiny_patch16_224.augreg_in21k_ft_in1k`, width 192, 12 blocks, 25 norms
- Norm→Linear fold pairs are fixed: `blocks.{i}.norm1`→`blocks.{i}.attn.qkv`, `blocks.{i}.norm2`→`blocks.{i}.mlp.fc1`, `norm`→`head`
- Fold equivalence tolerance: `atol=1e-4`, computed in fp32
- All accuracy numbers are top-1 on ImageNet-1k val
- Random seeds fixed at 0 for every data subset selection
- **Deployment target: Hailo-10H.** LayerNorm is the sole blocking operator; the rest of the ViT already compiles. Folding happens in PyTorch before ONNX export, so the exported graph contains no normalization node of any kind.
- ONNX export: fixed batch size 1, fully static shapes, opset 17
- Quantization is out of scope for this phase

---

## Why each gate exists

This plan is built so that **every task ends with a number you can compare against a prediction.** When reality disagrees with the prediction, the bug is in the task you just wrote — not somewhere in the previous three weeks of work. The two gates that matter most:

| Gate | Task | Predicted | If it fails |
|---|---|---|---|
| Teacher reproduction | 1 | 75.5% ± 0.3 | Data pipeline is wrong. Every downstream number is meaningless. Stop. |
| Fold equivalence | 4 | max abs diff < 1e-4 | The folded model is a different function from the evaluated one. All latency claims void. Stop. |

---

## File Structure

```
vit_batchnorm/
├── pyproject.toml                    deps, pytest config
├── src/vitbn/
│   ├── __init__.py
│   ├── models.py                     checkpoint loading, norm path enumeration
│   ├── data.py                       val + calibration loaders, timm transforms
│   ├── evaluate.py                   top-1 accuracy, latency benchmark
│   ├── norm_swap.py                  BatchNorm, LN→BN surgery
│   ├── calibrate.py                  BN running-statistic calibration
│   ├── fold.py                       BN→Linear folding, equivalence check
│   ├── reconstruct.py                block-wise reconstruction (arm C)
│   └── distill.py                    global distillation (arm D)
├── experiments/
│   ├── exp0_per_layer.py             per-layer damage sweep
│   ├── run_arms.py                   arms A–D end to end
│   └── make_figures.py               result plots
├── tests/
│   ├── test_models.py
│   ├── test_norm_swap.py
│   ├── test_calibrate.py
│   ├── test_fold.py
│   └── test_data.py
└── results/                          JSON results, figures
```

Responsibility split follows the five components in the spec, plus `models.py` (checkpoint and path enumeration, used by everything) and `data.py` (loaders, used by everything that touches images). Tests that need no ImageNet — `test_norm_swap`, `test_fold`, `test_calibrate` — run on random tensors and stay fast.

---

## Task 1: Data pipeline and the teacher-reproduction gate

**Why this is first:** every number in the project is measured by this code. If the eval harness is subtly wrong — wrong interpolation, wrong crop ratio, wrong normalization constants — every subsequent result is wrong in the same direction and you will not notice. Reproducing the published 75.5% is the only proof the harness is correct.

**Files:**
- Create: `pyproject.toml`
- Create: `src/vitbn/__init__.py`
- Create: `src/vitbn/models.py`
- Create: `src/vitbn/data.py`
- Create: `src/vitbn/evaluate.py`
- Create: `experiments/reproduce_teacher.py`
- Test: `tests/test_models.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `load_teacher(name: str, device) -> nn.Module`; `norm_paths(model) -> list[str]`; `build_val_loader(root, model, batch_size, num_workers) -> DataLoader`; `build_calib_loader(root, model, batch_size, num_workers, num_images, seed) -> DataLoader`; `top1(model, loader, device, max_batches=None) -> float`

- [ ] **Step 1: Create the project skeleton and dependencies**

```toml
# pyproject.toml
[project]
name = "vitbn"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "timm>=1.0.9",
    "matplotlib>=3.8",
    "numpy>=1.26",
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
```

Expected: install succeeds, `torch` and `timm` import.

If it fails on Python version: the system `python3` is 3.14, which has no PyTorch wheels. Install 3.11 (`brew install python@3.11`) or use the rented GPU box, which will already have a working CUDA PyTorch.

- [ ] **Step 2: Write the failing test for norm path enumeration**

```python
# tests/test_models.py
import pytest
import torch
from vitbn.models import load_teacher, norm_paths


@pytest.fixture(scope="module")
def tiny():
    return load_teacher("vit_tiny_patch16_224.augreg_in21k_ft_in1k", torch.device("cpu"))


def test_norm_paths_are_25_in_forward_order(tiny):
    paths = norm_paths(tiny)
    assert len(paths) == 25
    assert paths[0] == "blocks.0.norm1"
    assert paths[1] == "blocks.0.norm2"
    assert paths[-2] == "blocks.11.norm2"
    assert paths[-1] == "norm"


def test_every_norm_path_resolves_to_a_layernorm(tiny):
    for p in norm_paths(tiny):
        mod = tiny.get_submodule(p)
        assert isinstance(mod, torch.nn.LayerNorm), f"{p} is {type(mod)}"
        assert mod.normalized_shape == (192,)


def test_fc_norm_is_identity(tiny):
    # The final `norm` folds into `head`. Anything non-trivial between them
    # would break that fold. Token selection commutes with a per-channel
    # affine; a real fc_norm would not.
    assert isinstance(tiny.fc_norm, torch.nn.Identity)


def test_fold_targets_are_linear_with_bias(tiny):
    from vitbn.models import fold_pairs
    for norm_path, lin_path in fold_pairs(tiny):
        lin = tiny.get_submodule(lin_path)
        assert isinstance(lin, torch.nn.Linear), f"{lin_path} is {type(lin)}"
        assert lin.bias is not None, f"{lin_path} has no bias; fold needs one"


def test_teacher_is_frozen(tiny):
    assert not any(p.requires_grad for p in tiny.parameters())
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.models'`

- [ ] **Step 4: Implement `models.py`**

```python
# src/vitbn/models.py
"""Checkpoint loading and the fixed map of norm layers to their fold targets."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

DEFAULT_MODEL = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"

SCALING_ARM = [
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
]


def load_teacher(name: str = DEFAULT_MODEL, device=None) -> nn.Module:
    """Load a pretrained ViT, frozen and in eval mode.

    The teacher is never trained. Freezing here means an accidental
    optimizer step on teacher parameters raises instead of silently
    corrupting the reference model.
    """
    model = timm.create_model(name, pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model.to(device)
    return model


def load_student(name: str = DEFAULT_MODEL, device=None) -> nn.Module:
    """Load a trainable copy of the same checkpoint, to be converted."""
    model = timm.create_model(name, pretrained=True)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def norm_paths(model: nn.Module) -> list[str]:
    """The 25 norm module paths, in forward order.

    Order matters: block-wise reconstruction walks these front to back, and
    Experiment 0 indexes damage results by position in this list.
    """
    paths: list[str] = []
    for i in range(len(model.blocks)):
        paths.append(f"blocks.{i}.norm1")
        paths.append(f"blocks.{i}.norm2")
    paths.append("norm")
    return paths


def fold_pairs(model: nn.Module) -> list[tuple[str, str]]:
    """(norm_path, linear_path) pairs — each norm's fold target.

    In a pre-norm ViT every norm feeds exactly one Linear:
      norm1 -> attn.qkv, norm2 -> mlp.fc1, norm -> head.
    """
    pairs: list[tuple[str, str]] = []
    for i in range(len(model.blocks)):
        pairs.append((f"blocks.{i}.norm1", f"blocks.{i}.attn.qkv"))
        pairs.append((f"blocks.{i}.norm2", f"blocks.{i}.mlp.fc1"))
    pairs.append(("norm", "head"))
    return pairs


def set_submodule(model: nn.Module, path: str, new: nn.Module) -> None:
    """Replace the module at a dotted path."""
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, name, new)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_models.py -v`
Expected: PASS, 5 tests. First run downloads the checkpoint (~22 MB).

If `test_fc_norm_is_identity` fails, this `timm` version puts a real norm between `norm` and `head`, and the `norm`→`head` fold is invalid. Stop and reconsider — do not work around it.

- [ ] **Step 6: Write the data loader test**

```python
# tests/test_data.py
import torch
from vitbn.data import build_transform


def test_transform_matches_timm_config():
    from vitbn.models import load_teacher
    m = load_teacher(device=torch.device("cpu"))
    tf, cfg = build_transform(m)
    assert cfg["input_size"] == (3, 224, 224)
    assert cfg["interpolation"] == "bicubic"
    assert abs(cfg["crop_pct"] - 0.9) < 0.05

    from PIL import Image
    img = Image.new("RGB", (500, 375), (128, 64, 32))
    out = tf(img)
    assert out.shape == (3, 224, 224)
    # ImageNet-normalized data is roughly zero-mean, definitely not in [0,1]
    assert out.min() < 0.0
```

- [ ] **Step 7: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.data'`

- [ ] **Step 8: Implement `data.py`**

```python
# src/vitbn/data.py
"""ImageNet loaders. The eval transform comes from timm, never hand-rolled --
mismatched preprocessing is the usual cause of failed reproduction."""
from __future__ import annotations

from pathlib import Path

import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


def build_transform(model):
    """The exact eval transform this checkpoint was validated with."""
    cfg = resolve_data_config({}, model=model)
    return create_transform(**cfg, is_training=False), cfg


def build_val_loader(root, model, batch_size=256, num_workers=8):
    """Full ImageNet-1k val, 50k images, in ImageFolder layout."""
    tf, _ = build_transform(model)
    ds = ImageFolder(str(Path(root) / "val"), transform=tf)
    assert len(ds) == 50_000, f"expected 50000 val images, found {len(ds)}"
    assert len(ds.classes) == 1000, f"expected 1000 classes, found {len(ds.classes)}"
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def build_val_subset_loader(root, model, num_images=10_000, batch_size=256,
                            num_workers=8, seed=0):
    """A fixed random subset of val, for fast sweeps.

    Fixed seed so every Experiment 0 measurement sees identical images and
    differences between layers are attributable to the layer, not the sample.
    """
    tf, _ = build_transform(model)
    ds = ImageFolder(str(Path(root) / "val"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:num_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def build_calib_loader(root, model, num_images=10_000, batch_size=64,
                       num_workers=8, seed=0):
    """A fixed random subset of ImageNet train.

    Labels are loaded but unused -- calibration and distillation both target
    the teacher's own outputs, so only the images matter.
    """
    tf, _ = build_transform(model)
    ds = ImageFolder(str(Path(root) / "train"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:num_images].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
```

- [ ] **Step 9: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_data.py -v`
Expected: PASS, 1 test.

- [ ] **Step 10: Implement `evaluate.py` (top-1 only; latency comes in Task 8)**

```python
# src/vitbn/evaluate.py
"""Accuracy measurement. Latency benchmarking is added in Task 8."""
from __future__ import annotations

import torch


@torch.no_grad()
def top1(model, loader, device, max_batches=None, amp=True) -> float:
    """Top-1 accuracy as a percentage."""
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total
```

- [ ] **Step 11: Download ImageNet val and write the reproduction script**

Val is ~6.7 GB. On the rented box:

```bash
# Requires a HuggingFace account with ImageNet-1k terms accepted.
pip install "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download ILSVRC/imagenet-1k \
    --repo-type dataset --include "data/val_images.tar.gz" \
    --local-dir /data/imagenet_raw
```

Extract into `ImageFolder` layout — `$IMAGENET_ROOT/val/<wnid>/*.JPEG`. If the archive is flat, use the standard `valprep.sh` regrouping script. The assertions in `build_val_loader` (50000 images, 1000 classes) will catch a wrong layout immediately.

```python
# experiments/reproduce_teacher.py
"""GATE 1. Establishes that the eval harness is correct.

Predicted: 75.5% +/- 0.3 for vit_tiny_patch16_224.augreg_in21k_ft_in1k.

If the number is far off, the harness is broken and every downstream result
would be wrong in the same direction without appearing wrong. Common causes,
in order of likelihood:
  - val not in ImageFolder layout, so labels are misassigned
  - transform hand-rolled instead of taken from timm
  - wrong checkpoint tag (in1k vs in21k_ft_in1k differ by several points)
Do not proceed past this script until it passes.
"""
import argparse

import torch

from vitbn.data import build_val_loader
from vitbn.evaluate import top1
from vitbn.models import DEFAULT_MODEL, load_teacher

EXPECTED = {
    "vit_tiny_patch16_224.augreg_in21k_ft_in1k": 75.5,
    "vit_small_patch16_224.augreg_in21k_ft_in1k": 81.4,
    "vit_base_patch16_224.augreg_in21k_ft_in1k": 84.5,
}
TOLERANCE = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_teacher(args.model, device)
    loader = build_val_loader(args.data_root, model, args.batch_size)

    acc = top1(model, loader, device)
    expected = EXPECTED[args.model]
    delta = acc - expected

    print(f"model     {args.model}")
    print(f"measured  {acc:.2f}%")
    print(f"expected  {expected:.2f}%")
    print(f"delta     {delta:+.2f}")

    if abs(delta) > TOLERANCE:
        raise SystemExit(
            f"GATE FAILED: off by {delta:+.2f} (tolerance {TOLERANCE}). "
            "The eval harness is wrong. Do not proceed."
        )
    print("GATE PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12: Run the gate**

Run: `.venv/bin/python experiments/reproduce_teacher.py --data-root $IMAGENET_ROOT`
Expected: `measured 75.5%` ± 0.3, then `GATE PASSED`. Takes a few minutes on a 4090.

**This is the single most important checkpoint in the plan.** Record the exact measured number — it is the reference line for every later comparison, in preference to the published 75.5%.

- [ ] **Step 13: Commit**

```bash
git add pyproject.toml src/vitbn/ tests/ experiments/reproduce_teacher.py
git commit -m "feat: data pipeline, eval harness, teacher reproduction gate"
```

---

## Task 2: Norm surgery (`norm_swap`)

**Why:** this is the actual intervention the project studies. It must be pure surgery — no training, no statistics — so that later stages are the only thing that can change behaviour. The `γ`/`β` copy matters: LayerNorm's learned affine parameters have the same shape and the same role as BatchNorm's, so copying them is a strictly better starting point than the default 1/0.

**Files:**
- Create: `src/vitbn/norm_swap.py`
- Test: `tests/test_norm_swap.py`

**Interfaces:**
- Consumes: `norm_paths`, `set_submodule` from `vitbn.models`
- Produces: `class BatchNorm(nn.Module)` with attribute `.bn: nn.BatchNorm1d`; `swap_norms(model, paths) -> nn.Module`; `swap_all_norms(model) -> nn.Module`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_norm_swap.py
import torch
import torch.nn as nn

from vitbn.models import load_student, norm_paths
from vitbn.norm_swap import BatchNorm, swap_all_norms, swap_norms


def test_token_batchnorm_preserves_shape():
    m = BatchNorm(192)
    x = torch.randn(4, 197, 192)
    assert m(x).shape == (4, 197, 192)


def test_token_batchnorm_normalizes_over_batch_and_tokens():
    """The defining difference from LayerNorm: statistics are per channel,
    pooled over batch and tokens -- not per token, pooled over channels."""
    m = BatchNorm(8)
    m.train()
    x = torch.randn(16, 20, 8) * 5 + 3
    y = m(x)
    # Each channel is standardized across the (batch, token) axes.
    per_channel = y.reshape(-1, 8)
    assert torch.allclose(per_channel.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(per_channel.std(0, unbiased=False), torch.ones(8), atol=1e-4)
    # Individual tokens are NOT standardized -- this is what LayerNorm would do.
    per_token = y[0, 0]
    assert per_token.mean().abs() > 1e-3 or per_token.std().sub(1).abs() > 1e-3


def test_swap_replaces_only_requested_paths():
    model = load_student(device=torch.device("cpu"))
    swap_norms(model, ["blocks.0.norm1"])
    assert isinstance(model.get_submodule("blocks.0.norm1"), BatchNorm)
    assert isinstance(model.get_submodule("blocks.0.norm2"), nn.LayerNorm)
    assert isinstance(model.get_submodule("norm"), nn.LayerNorm)


def test_swap_copies_layernorm_affine_parameters():
    model = load_student(device=torch.device("cpu"))
    ln = model.get_submodule("blocks.3.norm2")
    gamma, beta = ln.weight.detach().clone(), ln.bias.detach().clone()
    swap_norms(model, ["blocks.3.norm2"])
    bn = model.get_submodule("blocks.3.norm2").bn
    assert torch.equal(bn.weight.detach(), gamma)
    assert torch.equal(bn.bias.detach(), beta)


def test_swap_all_replaces_25_norms():
    model = load_student(device=torch.device("cpu"))
    swap_all_norms(model)
    n = sum(isinstance(m, BatchNorm) for m in model.modules())
    assert n == 25
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_swapped_model_still_runs_and_is_damaged():
    """Uncalibrated BN has running_mean=0, running_var=1, so in eval mode it
    barely normalizes at all. The model must still execute, and its output
    must differ from the teacher -- if it does not, the swap is a no-op and
    something is wrong."""
    teacher = load_student(device=torch.device("cpu")).eval()
    student = load_student(device=torch.device("cpu"))
    swap_all_norms(student)
    student.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        a, b = teacher(x), student(x)
    assert b.shape == (2, 1000)
    assert not torch.allclose(a, b, atol=1e-2), "swap had no effect"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_norm_swap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.norm_swap'`

- [ ] **Step 3: Implement `norm_swap.py`**

```python
# src/vitbn/norm_swap.py
"""LayerNorm -> BatchNorm surgery. Pure structural change: no training, no
statistics gathered, no forward passes."""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import norm_paths, set_submodule


class BatchNorm(nn.Module):
    """BatchNorm over channels, treating tokens as extra batch elements.

    ViT activations are (B, N, D) with channels last. `nn.BatchNorm1d`
    expects channels at dim 1, so reshaping to (B*N, D) puts D where
    BatchNorm1d looks for it, and pools statistics over batch and tokens
    jointly -- exactly the intended semantics.

    Drop-in for the `nn.LayerNorm(D)` it replaces: same input and output
    shape, same affine parameter shapes.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float | None = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return self.bn(x.reshape(b * n, d)).reshape(b, n, d)

    def extra_repr(self) -> str:
        return f"num_features={self.bn.num_features}"


def swap_norms(model: nn.Module, paths: list[str]) -> nn.Module:
    """Replace the LayerNorms at `paths` with BatchNorm, in place.

    LayerNorm's learned gamma/beta are copied into the BatchNorm. They have
    identical shape (D,) and identical meaning -- a per-channel affine applied
    after normalization -- so copying preserves whatever scaling the trained
    model relies on. The default 1/0 initialization would discard it.
    """
    for path in paths:
        ln = model.get_submodule(path)
        if not isinstance(ln, nn.LayerNorm):
            raise TypeError(f"{path} is {type(ln).__name__}, expected LayerNorm")
        (dim,) = ln.normalized_shape
        bn = BatchNorm(dim, eps=ln.eps)
        with torch.no_grad():
            bn.bn.weight.copy_(ln.weight)
            bn.bn.bias.copy_(ln.bias)
        bn.to(ln.weight.device, ln.weight.dtype)
        set_submodule(model, path, bn)
    return model


def swap_all_norms(model: nn.Module) -> nn.Module:
    """Replace all 25 norms."""
    return swap_norms(model, norm_paths(model))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_norm_swap.py -v`
Expected: PASS, 6 tests.

If `test_token_batchnorm_normalizes_over_batch_and_tokens` fails, the reshape is wrong and BatchNorm is pooling over the wrong axis. This would invalidate the entire project — fix it before moving on.

- [ ] **Step 5: Commit**

```bash
git add src/vitbn/norm_swap.py tests/test_norm_swap.py
git commit -m "feat: LayerNorm to BatchNorm surgery with affine parameter transfer"
```

---

## Task 3: Statistic calibration (`calibrate`)

**Why:** a fresh `BatchNorm1d` has `running_mean=0`, `running_var=1`, so in eval mode it does essentially nothing — which is wrong, because real activations are neither zero-mean nor unit-variance. Calibration **measures** the true per-channel statistics with forward passes. There is no loss, no `.backward()`, no optimizer. It is measurement, not learning, and it cannot repair the damage from the change of function class — that is precisely why it is the right diagnostic in Experiment 0.

**Files:**
- Create: `src/vitbn/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `BatchNorm` from `vitbn.norm_swap`
- Produces: `calibrate(model, loader, device, num_batches=150) -> nn.Module`; `reset_bn_stats(model) -> None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_calibrate.py
import torch
import torch.nn as nn

from vitbn.calibrate import calibrate, reset_bn_stats
from vitbn.norm_swap import BatchNorm


def _fake_loader(n_batches=4, b=8, n=197, d=192, scale=5.0, shift=2.0):
    for _ in range(n_batches):
        yield torch.randn(b, n, d) * scale + shift, torch.zeros(b, dtype=torch.long)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = BatchNorm(192)

    def forward(self, x):
        return self.norm(x)


def test_calibration_measures_the_true_statistics():
    m = Wrapper()
    calibrate(m, list(_fake_loader()), torch.device("cpu"), num_batches=4)
    bn = m.norm.bn
    # Data was N(2, 5^2): running stats must recover that, not stay at 0/1.
    assert torch.allclose(bn.running_mean, torch.full((192,), 2.0), atol=0.3)
    assert torch.allclose(bn.running_var, torch.full((192,), 25.0), rtol=0.15)


def test_calibration_updates_no_parameters():
    """Calibration is measurement. gamma/beta must be untouched."""
    m = Wrapper()
    g0 = m.norm.bn.weight.detach().clone()
    b0 = m.norm.bn.bias.detach().clone()
    calibrate(m, list(_fake_loader()), torch.device("cpu"), num_batches=4)
    assert torch.equal(m.norm.bn.weight.detach(), g0)
    assert torch.equal(m.norm.bn.bias.detach(), b0)


def test_calibration_leaves_model_in_eval_mode():
    m = Wrapper()
    calibrate(m, list(_fake_loader()), torch.device("cpu"), num_batches=4)
    assert not m.training
    assert not m.norm.bn.training


def test_reset_restores_default_statistics():
    m = Wrapper()
    calibrate(m, list(_fake_loader()), torch.device("cpu"), num_batches=4)
    reset_bn_stats(m)
    assert torch.allclose(m.norm.bn.running_mean, torch.zeros(192))
    assert torch.allclose(m.norm.bn.running_var, torch.ones(192))
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.calibrate'`

- [ ] **Step 3: Implement `calibrate.py`**

```python
# src/vitbn/calibrate.py
"""BatchNorm running-statistic calibration: forward passes only."""
from __future__ import annotations

import torch
import torch.nn as nn


def reset_bn_stats(model: nn.Module) -> None:
    """Clear running statistics and switch to cumulative averaging.

    momentum=None makes BatchNorm accumulate an exact running mean over all
    batches seen, rather than an exponential moving average. For a one-shot
    measurement pass that is what we want -- every calibration image
    contributes equally and the result does not depend on batch ordering.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None


@torch.no_grad()
def calibrate(model: nn.Module, loader, device, num_batches: int = 150) -> nn.Module:
    """Populate BatchNorm running statistics from data.

    No loss, no gradients, no optimizer. BatchNorm updates running_mean and
    running_var as a side effect of a forward pass in train mode; this
    function does nothing else.

    Only the BatchNorm modules are put in train mode. Putting the whole model
    in train mode would also enable dropout and stochastic depth, injecting
    noise into the statistics we are trying to measure.
    """
    reset_bn_stats(model)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()

    seen = 0
    for x, _ in loader:
        if seen >= num_batches:
            break
        model(x.to(device, non_blocking=True))
        seen += 1

    if seen == 0:
        raise RuntimeError("calibration loader yielded no batches")

    model.eval()
    return model
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_calibrate.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vitbn/calibrate.py tests/test_calibrate.py
git commit -m "feat: BatchNorm statistic calibration (forward-only)"
```

---

## Task 4: Folding and the equivalence gate (`fold`)

**Why:** the fold is the payoff — it removes all 25 normalization ops from the inference graph. It is exact algebra, so it must be *exactly* right, and a subtly wrong fold is more dangerous than an obviously wrong one because it still produces plausible accuracy while being a different model. The equivalence test is what makes every latency number in this project honest.

The arithmetic, for BatchNorm feeding `Linear(W, b)`:

```
s     = gamma / sqrt(running_var + eps)
shift = beta - running_mean * s
W'    = W * s          scaling INPUT channels, i.e. W.shape == (out, in), broadcast over `in`
b'    = b + W @ shift  computed with the ORIGINAL W, before scaling
```

**Files:**
- Create: `src/vitbn/fold.py`
- Test: `tests/test_fold.py`

**Interfaces:**
- Consumes: `fold_pairs`, `set_submodule` from `vitbn.models`; `BatchNorm` from `vitbn.norm_swap`
- Produces: `fold_bn_into_linear(bn, lin) -> None`; `fold_model(model) -> nn.Module`; `assert_fold_equivalent(model, x, atol=1e-4) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fold.py
import copy

import pytest
import torch
import torch.nn as nn

from vitbn.calibrate import calibrate
from vitbn.fold import assert_fold_equivalent, fold_bn_into_linear, fold_model
from vitbn.models import load_student
from vitbn.norm_swap import BatchNorm, swap_all_norms


def test_fold_is_exact_for_a_single_bn_linear_pair():
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(16)
    lin = nn.Linear(16, 7)
    # Give the BN non-trivial statistics and affine parameters, so the test
    # would catch a fold that silently ignores any of the four terms.
    with torch.no_grad():
        bn.running_mean.normal_(mean=1.0, std=2.0)
        bn.running_var.uniform_(0.5, 3.0)
        bn.weight.normal_(mean=1.0, std=0.5)
        bn.bias.normal_()
    bn.eval()

    x = torch.randn(32, 16)
    expected = lin(bn(x))

    folded = copy.deepcopy(lin)
    fold_bn_into_linear(bn, folded)
    got = folded(x)

    assert torch.allclose(expected, got, atol=1e-5), \
        f"max diff {(expected - got).abs().max().item():.3e}"


def test_fold_scales_input_channels_not_output_channels():
    """The classic transposition bug. W is (out, in) and BatchNorm precedes
    the Linear, so the scale applies along `in`. Scaling `out` instead gives
    a wrong model that still runs and still produces plausible logits."""
    bn = nn.BatchNorm1d(4)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0 - bn.eps)
    bn.eval()
    lin = nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(4))
        lin.bias.zero_()

    fold_bn_into_linear(bn, lin)
    # Column j must be scaled by s[j]. If output channels were scaled
    # instead, the matrix would be scaled by row and this fails.
    assert torch.allclose(lin.weight.diag(), torch.tensor([1.0, 2.0, 3.0, 4.0]),
                          atol=1e-4)


def test_fold_raises_when_bn_is_in_train_mode():
    """In train mode BatchNorm uses batch statistics, so the folded constants
    would not match what the model actually computes."""
    bn = nn.BatchNorm1d(8)
    bn.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        fold_bn_into_linear(bn, nn.Linear(8, 4))


def test_fold_removes_every_norm_from_the_graph():
    model = load_student(device=torch.device("cpu"))
    swap_all_norms(model)
    loader = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
              for _ in range(3)]
    calibrate(model, loader, torch.device("cpu"), num_batches=3)
    fold_model(model)
    assert not any(isinstance(m, (BatchNorm, nn.BatchNorm1d, nn.LayerNorm))
                   for m in model.modules())


def test_folded_vit_matches_unfolded_vit():
    """GATE 2. The most important test in the project."""
    torch.manual_seed(0)
    model = load_student(device=torch.device("cpu"))
    swap_all_norms(model)
    loader = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
              for _ in range(3)]
    calibrate(model, loader, torch.device("cpu"), num_batches=3)

    x = torch.randn(2, 3, 224, 224)
    max_diff = assert_fold_equivalent(model, x, atol=1e-4)
    assert max_diff < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_fold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.fold'`

- [ ] **Step 3: Implement `fold.py`**

```python
# src/vitbn/fold.py
"""Fold BatchNorm into the Linear that consumes it.

At inference BatchNorm is a per-channel affine map, so composing it with a
following Linear yields another Linear. This is exact algebra, not an
approximation -- the folded model computes the identical function up to
floating-point rounding.

LayerNorm cannot be folded because its normalizer depends on the input.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .models import fold_pairs, set_submodule
from .norm_swap import BatchNorm


@torch.no_grad()
def fold_bn_into_linear(bn: nn.BatchNorm1d, lin: nn.Linear) -> None:
    """Absorb `bn` into `lin`, in place. `bn` must precede `lin`.

        BN(x) = s * x + shift,  s = gamma / sqrt(var + eps)
                                shift = beta - mean * s
        Linear(BN(x)) = W(s*x + shift) + b
                      = (W * s) x + (b + W @ shift)
    """
    if bn.training:
        raise RuntimeError(
            "BatchNorm is in train mode; folding would capture batch "
            "statistics rather than running statistics. Call model.eval() first."
        )
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BatchNorm has no running statistics; calibrate first.")
    if lin.bias is None:
        raise RuntimeError("fold target Linear has no bias; cannot absorb the shift")

    # fp64 for the fold arithmetic: this is done once, and it keeps the
    # equivalence check limited by the model's own precision rather than ours.
    w = lin.weight.data.double()
    b = lin.bias.data.double()

    s = bn.weight.data.double() / torch.sqrt(bn.running_var.data.double() + bn.eps)
    shift = bn.bias.data.double() - bn.running_mean.data.double() * s

    # Bias first -- it needs the ORIGINAL W, before scaling.
    b = b + w @ shift
    # W is (out, in); BatchNorm precedes the Linear, so scale INPUT channels.
    w = w * s.unsqueeze(0)

    lin.weight.data.copy_(w.to(lin.weight.dtype))
    lin.bias.data.copy_(b.to(lin.bias.dtype))


@torch.no_grad()
def fold_model(model: nn.Module) -> nn.Module:
    """Fold all 25 BatchNorms into their targets, replacing each with
    Identity. Irreversible: the model can no longer be trained afterwards."""
    model.eval()
    for norm_path, lin_path in fold_pairs(model):
        norm = model.get_submodule(norm_path)
        if isinstance(norm, nn.Identity):
            continue
        if not isinstance(norm, BatchNorm):
            raise TypeError(
                f"{norm_path} is {type(norm).__name__}; only BatchNorm "
                "folds. Unconverted LayerNorms must be left in place."
            )
        fold_bn_into_linear(norm.bn, model.get_submodule(lin_path))
        set_submodule(model, norm_path, nn.Identity())
    return model


@torch.no_grad()
def assert_fold_equivalent(model: nn.Module, x: torch.Tensor,
                           atol: float = 1e-4) -> float:
    """GATE 2. Fold a copy and confirm it computes the same function.

    Returns the max absolute logit difference. Raises if it exceeds `atol`.

    A failure means the folded model is not the model whose accuracy was
    measured, which makes every latency number meaningless. Do not relax the
    tolerance to make this pass.
    """
    model.eval()
    reference = model(x)
    folded = fold_model(copy.deepcopy(model))
    got = folded(x)
    max_diff = (reference - got).abs().max().item()
    if max_diff > atol:
        raise RuntimeError(
            f"GATE FAILED: folded model differs by {max_diff:.3e} > {atol}. "
            "Check: fold axis (input vs output channels), bias computed with "
            "the pre-scaling weight, eval mode, eps consistency."
        )
    return max_diff
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_fold.py -v`
Expected: PASS, 5 tests. `test_folded_vit_matches_unfolded_vit` should report a max diff around 1e-6.

If `test_fold_scales_input_channels_not_output_channels` fails, the broadcast is transposed. This is the bug that produces a wrong-but-plausible model — fix it, do not work around it.

- [ ] **Step 5: Commit**

```bash
git add src/vitbn/fold.py tests/test_fold.py
git commit -m "feat: BatchNorm folding with exact-equivalence gate"
```

---

## Task 5: Experiment 0 — per-layer damage sweep

**Why:** this is the cheap early-warning system for the whole project. Converting one norm at a time, with calibration only and no training, isolates the damage caused purely by the change of function class. It tells you within about an hour which of the 25 layers a trained ViT genuinely depends on — before you invest in any recovery method. If most layers turn out cheap, full lossless conversion is plausibly in reach. If three are catastrophic, you know that now rather than in week three.

**Files:**
- Create: `experiments/exp0_per_layer.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3
- Produces: `results/exp0_<model>.json` — `{"baseline": float, "per_layer": {path: acc}}`

- [ ] **Step 1: Write the sweep script**

```python
# experiments/exp0_per_layer.py
"""Experiment 0: per-layer conversion damage.

For each of the 25 norms independently: convert only that one to BatchNorm,
calibrate it (forward passes only, no training), and measure top-1.

Calibration-only is the correct probe precisely because it cannot repair
anything. LayerNorm is nonlinear and BatchNorm at inference is affine, so no
choice of running statistics reproduces LayerNorm's function. This measures
harm; it does not attempt a fix.

Predicted: `norm` (the final one, feeding the head) and the early blocks hurt
most. Many middle-block norms should be nearly free.

If EVERY layer collapses to near chance (0.1%), suspect a bug in the swap
rather than a real result -- cross-check against tests/test_norm_swap.py.
If NO layer moves at all, the swap is not being applied.
"""
import argparse
import json
from pathlib import Path

import torch

from vitbn.calibrate import calibrate
from vitbn.data import build_calib_loader, build_val_subset_loader
from vitbn.evaluate import top1
from vitbn.models import DEFAULT_MODEL, load_student, load_teacher, norm_paths
from vitbn.norm_swap import swap_norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--val-images", type=int, default=10_000)
    ap.add_argument("--calib-batches", type=int, default=150)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.model, device)

    val = build_val_subset_loader(args.data_root, teacher, args.val_images)
    calib = build_calib_loader(args.data_root, teacher, num_images=10_000)

    baseline = top1(teacher, val, device)
    print(f"baseline (unmodified, {args.val_images} val images): {baseline:.2f}%\n")

    paths = norm_paths(teacher)
    results = {}
    for i, path in enumerate(paths):
        student = load_student(args.model, device)
        swap_norms(student, [path])
        calibrate(student, calib, device, num_batches=args.calib_batches)
        acc = top1(student, val, device)
        results[path] = acc
        print(f"[{i + 1:2d}/25] {path:20s} {acc:6.2f}%  "
              f"(drop {baseline - acc:+6.2f})")
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"exp0_{args.model.split('.')[0]}.json"
    path.write_text(json.dumps(
        {"model": args.model, "val_images": args.val_images,
         "baseline": baseline, "per_layer": results}, indent=2))

    drops = sorted(((baseline - a, p) for p, a in results.items()), reverse=True)
    print(f"\nwrote {path}")
    print("\nmost damaging:")
    for d, p in drops[:5]:
        print(f"  {p:20s} -{d:.2f}")
    print("\nleast damaging:")
    for d, p in drops[-5:]:
        print(f"  {p:20s} -{d:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on two layers before committing an hour of GPU time**

Run:
```bash
.venv/bin/python -c "
import sys; sys.argv = ['x', '--data-root', '$IMAGENET_ROOT', '--val-images', '512', '--calib-batches', '5']
exec(open('experiments/exp0_per_layer.py').read().replace('norm_paths(teacher)', 'norm_paths(teacher)[:2]'))
"
```
Expected: a baseline near 75% and two per-layer numbers, in under two minutes.

- [ ] **Step 3: Run the full sweep**

Run: `.venv/bin/python experiments/exp0_per_layer.py --data-root $IMAGENET_ROOT`
Expected: 25 lines, roughly 30–60 minutes on a 4090, and `results/exp0_vit_tiny_patch16_224.json`.

**Read the result before continuing.** It determines whether full lossless conversion is realistic, and it gives the damage ranking that the fallback strategy in the spec would use. Three outcomes:
- *Most layers cheap (< 1 point), a few expensive* — the expected case. Proceed to Task 6; expect reconstruction to close the remaining gap.
- *Nearly all layers cheap* — full lossless conversion may be within reach with calibration plus light reconstruction.
- *Nearly all layers catastrophic* — full conversion is unlikely to work. Discuss before investing in Tasks 6–7; the coverage-relaxation fallback becomes the plan.

- [ ] **Step 4: Commit**

```bash
git add experiments/exp0_per_layer.py results/
git commit -m "feat: Experiment 0 per-layer damage sweep, plus results"
```

---

## Task 6: Block-wise reconstruction (arm C)

**Why:** this is the actual method. It does not ask BatchNorm to imitate LayerNorm — that is impossible — but re-fits each block's weights so the block reproduces the teacher's output *without* per-token normalization. Working one block at a time avoids asking the optimizer to repair a fully broken network in a single shot, and it is the approach post-training quantization uses for the structurally identical problem (BRECQ, AdaRound).

Two deliberate choices, both worth understanding:

1. **Student input, teacher output.** Block *k* is fed the *student's* propagated activations but trained against the *teacher's* output. This lets each block correct drift accumulated upstream. Teacher-forcing both sides would hide that drift and the errors would compound.
2. **BatchNorm stays in eval mode during reconstruction.** Running statistics are frozen at their calibrated values and only `γ`/`β` and the block's other weights train. This makes the training objective exactly the function that will be deployed and folded — no train/eval mismatch.

**Files:**
- Create: `src/vitbn/reconstruct.py`
- Test: `tests/test_reconstruct.py`

**Interfaces:**
- Consumes: `BatchNorm` from `vitbn.norm_swap`
- Produces: `capture_block_io(model, x, k) -> tuple[Tensor, Tensor]`; `reconstruct_block(student, teacher, k, loader, device, steps, lr) -> float`; `reconstruct_all(student, teacher, loader, device, steps, lr) -> dict[int, float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reconstruct.py
import torch

from vitbn.calibrate import calibrate
from vitbn.models import load_student, load_teacher
from vitbn.norm_swap import swap_all_norms
from vitbn.reconstruct import capture_block_io, reconstruct_block


def _loader(n=4, b=2):
    return [(torch.randn(b, 3, 224, 224), torch.zeros(b, dtype=torch.long))
            for _ in range(n)]


def test_capture_returns_matching_input_and_output_shapes():
    m = load_teacher(device=torch.device("cpu"))
    x = torch.randn(2, 3, 224, 224)
    bin_, bout = capture_block_io(m, x, 3)
    assert bin_.shape == (2, 197, 192)
    assert bout.shape == (2, 197, 192)
    assert not torch.allclose(bin_, bout)


def test_capture_output_equals_block_applied_to_input():
    m = load_teacher(device=torch.device("cpu"))
    x = torch.randn(2, 3, 224, 224)
    bin_, bout = capture_block_io(m, x, 5)
    with torch.no_grad():
        assert torch.allclose(m.blocks[5](bin_), bout, atol=1e-5)


def test_reconstruction_reduces_the_block_error():
    torch.manual_seed(0)
    device = torch.device("cpu")
    teacher = load_teacher(device=device)
    student = load_student(device=device)
    swap_all_norms(student)
    loader = _loader(n=6)
    calibrate(student, loader, device, num_batches=6)

    x = loader[0][0]
    with torch.no_grad():
        _, t_out = capture_block_io(teacher, x, 0)
        s_in, s_out = capture_block_io(student, x, 0)
        before = torch.nn.functional.mse_loss(s_out, t_out).item()

    reconstruct_block(student, teacher, 0, loader, device, steps=30, lr=1e-4)

    with torch.no_grad():
        s_in2, s_out2 = capture_block_io(student, x, 0)
        after = torch.nn.functional.mse_loss(s_out2, t_out).item()

    assert after < before, f"error grew: {before:.4e} -> {after:.4e}"


def test_reconstruction_does_not_touch_the_teacher():
    device = torch.device("cpu")
    teacher = load_teacher(device=device)
    student = load_student(device=device)
    swap_all_norms(student)
    loader = _loader(n=4)
    calibrate(student, loader, device, num_batches=4)

    before = teacher.blocks[0].attn.qkv.weight.detach().clone()
    reconstruct_block(student, teacher, 0, loader, device, steps=10, lr=1e-4)
    assert torch.equal(teacher.blocks[0].attn.qkv.weight.detach(), before)


def test_reconstruction_leaves_running_stats_frozen():
    """BN stays in eval mode so the training objective matches deployment."""
    device = torch.device("cpu")
    teacher = load_teacher(device=device)
    student = load_student(device=device)
    swap_all_norms(student)
    loader = _loader(n=4)
    calibrate(student, loader, device, num_batches=4)

    bn = student.blocks[0].norm1.bn
    before = bn.running_mean.detach().clone()
    reconstruct_block(student, teacher, 0, loader, device, steps=10, lr=1e-4)
    assert torch.equal(bn.running_mean.detach(), before)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_reconstruct.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.reconstruct'`

- [ ] **Step 3: Implement `reconstruct.py`**

```python
# src/vitbn/reconstruct.py
"""Block-wise reconstruction (arm C).

Each converted block is re-fitted to reproduce the teacher block's output.
The network is not asked to reproduce LayerNorm's function -- it is asked to
become a network that does not need it.
"""
from __future__ import annotations

from itertools import cycle, islice

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm_swap import BatchNorm


def capture_block_io(model: nn.Module, x: torch.Tensor, k: int):
    """Return (input, output) of `model.blocks[k]` for input images `x`.

    A forward hook is the reliable way to do this -- reimplementing the stem
    (patch embed, CLS token, position embedding, pre-norm dropout) would risk
    diverging from timm's actual forward pass.
    """
    captured = {}

    def hook(_module, args, output):
        captured["in"] = args[0]
        captured["out"] = output

    handle = model.blocks[k].register_forward_hook(hook)
    try:
        model(x)
    finally:
        handle.remove()
    return captured["in"], captured["out"]


def _freeze_bn_stats(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode: running statistics frozen at their
    calibrated values, only the affine parameters train. The objective then
    matches exactly what gets folded and deployed."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


def reconstruct_block(student, teacher, k: int, loader, device,
                      steps: int = 1000, lr: float = 1e-4,
                      grad_clip: float = 1.0, log_every: int = 200) -> float:
    """Fit student block k to the teacher's block-k output. Returns final loss.

    The student block is fed the STUDENT's propagated activations, not the
    teacher's, so it can correct drift accumulated in earlier converted
    blocks. Its target is the TEACHER's output.
    """
    student.eval()
    _freeze_bn_stats(student)

    block = student.blocks[k]
    for p in student.parameters():
        p.requires_grad_(False)
    for p in block.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam([p for p in block.parameters() if p.requires_grad], lr=lr)

    loss_val = float("nan")
    for step, (x, _) in enumerate(islice(cycle(loader), steps)):
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            _, target = capture_block_io(teacher, x, k)
            block_in, _ = capture_block_io(student, x, k)

        pred = block(block_in)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), grad_clip)
        opt.step()

        loss_val = loss.item()
        if log_every and step % log_every == 0:
            print(f"    block {k:2d} step {step:5d}  mse {loss_val:.5e}")

    for p in student.parameters():
        p.requires_grad_(False)
    return loss_val


def reconstruct_all(student, teacher, loader, device, steps: int = 1000,
                    lr: float = 1e-4) -> dict[int, float]:
    """Reconstruct every block front to back.

    Order matters: block k is fed the student's activations, which depend on
    blocks 0..k-1 already being reconstructed.
    """
    losses = {}
    for k in range(len(student.blocks)):
        print(f"  reconstructing block {k}/{len(student.blocks) - 1}")
        losses[k] = reconstruct_block(student, teacher, k, loader, device,
                                      steps=steps, lr=lr)
    return losses
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_reconstruct.py -v`
Expected: PASS, 5 tests. Slow on CPU (a minute or two) — these do real optimization.

If `test_reconstruction_reduces_the_block_error` fails, the optimizer is not affecting the loss. Check that block parameters have `requires_grad=True` and that the loss is computed from `block(block_in)` rather than from a cached no-grad tensor.

- [ ] **Step 5: Commit**

```bash
git add src/vitbn/reconstruct.py tests/test_reconstruct.py
git commit -m "feat: block-wise reconstruction against teacher activations"
```

---

## Task 7: Global distillation (arm D)

**Why:** block-wise reconstruction optimizes each block against a local target, so small errors can still accumulate across twelve blocks in a way no single block sees. Global distillation fine-tunes the whole student against the teacher's logits and intermediate features, correcting exactly that accumulated drift. It comes last because it is the most expensive rung and benefits from starting at arm C's solution rather than from a broken network.

**Files:**
- Create: `src/vitbn/distill.py`
- Test: `tests/test_distill.py`

**Interfaces:**
- Consumes: `capture_block_io` from `vitbn.reconstruct`
- Produces: `distill(student, teacher, loader, device, epochs, lr, temperature, feature_weight) -> list[float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distill.py
import torch

from vitbn.calibrate import calibrate
from vitbn.distill import distill
from vitbn.models import load_student, load_teacher
from vitbn.norm_swap import swap_all_norms


def _loader(n=6, b=2):
    return [(torch.randn(b, 3, 224, 224), torch.zeros(b, dtype=torch.long))
            for _ in range(n)]


def test_distillation_reduces_logit_disagreement():
    torch.manual_seed(0)
    device = torch.device("cpu")
    teacher = load_teacher(device=device)
    student = load_student(device=device)
    swap_all_norms(student)
    loader = _loader()
    calibrate(student, loader, device, num_batches=6)

    x = loader[0][0]
    with torch.no_grad():
        before = (teacher(x) - student(x)).pow(2).mean().item()

    distill(student, teacher, loader, device, epochs=3, lr=1e-4,
            amp=False, log_every=0)

    with torch.no_grad():
        after = (teacher(x) - student(x)).pow(2).mean().item()

    assert after < before, f"disagreement grew: {before:.4e} -> {after:.4e}"


def test_distillation_uses_no_labels():
    """Labels in the loader are deliberately wrong. If distillation improves
    teacher agreement anyway, it is genuinely label-free."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    teacher = load_teacher(device=device)
    student = load_student(device=device)
    swap_all_norms(student)
    loader = [(torch.randn(2, 3, 224, 224), torch.full((2,), 999))
              for _ in range(6)]
    calibrate(student, loader, device, num_batches=6)

    x = loader[0][0]
    with torch.no_grad():
        before = (teacher(x) - student(x)).pow(2).mean().item()
    distill(student, teacher, loader, device, epochs=3, lr=1e-4,
            amp=False, log_every=0)
    with torch.no_grad():
        after = (teacher(x) - student(x)).pow(2).mean().item()
    assert after < before
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_distill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.distill'`

- [ ] **Step 3: Implement `distill.py`**

```python
# src/vitbn/distill.py
"""Global distillation (arm D): correct drift that block-wise reconstruction
cannot see, by matching the teacher end to end.

Label-free by construction -- the target is the teacher's own output, so any
images will do and ImageNet annotations are never read.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _collect_block_outputs(model: nn.Module, x: torch.Tensor):
    """Run the model, returning (logits, [output of each block])."""
    outs = []
    handles = [blk.register_forward_hook(lambda _m, _a, o: outs.append(o))
               for blk in model.blocks]
    try:
        logits = model(x)
    finally:
        for h in handles:
            h.remove()
    return logits, outs


def distill(student, teacher, loader, device, epochs: int = 1, lr: float = 1e-5,
            temperature: float = 2.0, feature_weight: float = 1.0,
            grad_clip: float = 1.0, amp: bool = True,
            log_every: int = 50) -> list[float]:
    """Fine-tune the student against the frozen teacher. Returns per-step loss.

    Loss = KL(student || teacher) on temperature-softened logits, scaled by
    T^2 to keep gradient magnitudes comparable across temperatures, plus MSE
    on every block output.

    Gradient clipping is mandatory: BatchNorm instability in transformers
    shows up as loss spikes, and an unclipped spike destroys the solution
    that reconstruction produced.
    """
    student.train()
    # Keep BatchNorm in eval mode so running statistics stay at their
    # calibrated values and the objective matches what will be folded.
    for m in student.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
    teacher.eval()

    for p in student.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, epochs * len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    losses = []
    step = 0
    for _ in range(epochs):
        for x, _ in loader:                      # labels deliberately ignored
            x = x.to(device, non_blocking=True)

            with torch.no_grad():
                t_logits, t_feats = _collect_block_outputs(teacher, x)

            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                s_logits, s_feats = _collect_block_outputs(student, x)
                kl = F.kl_div(
                    F.log_softmax(s_logits / temperature, dim=-1),
                    F.log_softmax(t_logits / temperature, dim=-1),
                    reduction="batchmean", log_target=True,
                ) * (temperature ** 2)
                feat = sum(F.mse_loss(s, t) for s, t in zip(s_feats, t_feats))
                loss = kl + feature_weight * feat

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            losses.append(loss.item())
            if log_every and step % log_every == 0:
                print(f"    step {step:6d}  loss {loss.item():.5e} "
                      f"(kl {kl.item():.3e} feat {feat.item():.3e})")
            step += 1

    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return losses
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_distill.py -v`
Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vitbn/distill.py tests/test_distill.py
git commit -m "feat: global distillation against frozen teacher"
```

---

## Task 8: Latency benchmark and the arms runner

**Why:** this produces the headline result. Latency must be measured honestly — against a `torch.compile`'d LayerNorm baseline as well as an eager one — because modern fused LayerNorm kernels are fast, and a win over an unoptimized baseline would overstate the payoff.

**Files:**
- Modify: `src/vitbn/evaluate.py` (append `benchmark_latency`)
- Create: `experiments/run_arms.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `benchmark_latency(model, device, batch_size, input_size, warmup, iters, compile_model) -> dict`; `results/arms_<model>.json`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import torch

from vitbn.evaluate import benchmark_latency
from vitbn.models import load_teacher


def test_benchmark_returns_positive_timings():
    m = load_teacher(device=torch.device("cpu"))
    r = benchmark_latency(m, torch.device("cpu"), batch_size=1,
                          warmup=1, iters=3, compile_model=False)
    assert r["ms_per_batch"] > 0
    assert r["img_per_s"] > 0
    assert r["iters"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL — `ImportError: cannot import name 'benchmark_latency'`

- [ ] **Step 3: Append `benchmark_latency` to `evaluate.py`**

```python
# append to src/vitbn/evaluate.py
import time


@torch.no_grad()
def benchmark_latency(model, device, batch_size: int = 64,
                      input_size: tuple = (3, 224, 224), warmup: int = 20,
                      iters: int = 100, compile_model: bool = False,
                      amp: bool = True) -> dict:
    """Wall-clock inference latency.

    Warmup matters: the first iterations include CUDA kernel autotuning and,
    if compiling, graph capture. Timing them would make every model look
    slower in proportion to how little it was run.

    CUDA synchronization matters: kernel launches are asynchronous, so
    without it you would be timing the launch, not the work.
    """
    model.eval()
    if compile_model:
        model = torch.compile(model)

    x = torch.randn(batch_size, *input_size, device=device)

    for _ in range(warmup):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = 1000.0 * elapsed / iters
    return {"ms_per_batch": ms, "img_per_s": batch_size * iters / elapsed,
            "batch_size": batch_size, "iters": iters, "compiled": compile_model}


def count_norms(model) -> int:
    """Normalization ops remaining in the inference graph."""
    from .norm_swap import BatchNorm
    return sum(isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm1d,
                              BatchNorm))
               for m in model.modules())
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: PASS, 1 test.

- [ ] **Step 5: Write the arms runner**

```python
# experiments/run_arms.py
"""Arms A-D end to end, then fold and benchmark.

Each arm builds on the previous one. Predicted top-1 for ViT-Tiny:
  A naive swap          near chance (~0.1-5%)
  B + calibration       poor but above chance
  C + reconstruction    the interesting result
  D + distillation      best

Arms A and B are controls that are EXPECTED to fail: LayerNorm is nonlinear
and BatchNorm at inference is affine, so no running statistics reproduce it.
Their purpose is to quantify the gap so that recovery in arm C is
attributable to the right cause. If arm B matches the teacher, something is
wrong -- most likely the swap is not being applied.
"""
import argparse
import json
from pathlib import Path

import torch

from vitbn.calibrate import calibrate
from vitbn.data import build_calib_loader, build_val_loader
from vitbn.distill import distill
from vitbn.evaluate import benchmark_latency, count_norms, top1
from vitbn.fold import assert_fold_equivalent, fold_model
from vitbn.models import DEFAULT_MODEL, load_student, load_teacher
from vitbn.norm_swap import swap_all_norms
from vitbn.reconstruct import reconstruct_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calib-images", type=int, default=10_000)
    ap.add_argument("--distill-images", type=int, default=100_000)
    ap.add_argument("--recon-steps", type=int, default=1000)
    ap.add_argument("--distill-epochs", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(args.model, device)
    val = build_val_loader(args.data_root, teacher)
    calib = build_calib_loader(args.data_root, teacher, num_images=args.calib_images)

    res = {"model": args.model}

    res["teacher"] = top1(teacher, val, device)
    print(f"teacher              {res['teacher']:.2f}%")

    student = load_student(args.model, device)
    swap_all_norms(student)
    res["arm_a"] = top1(student, val, device)
    print(f"A naive swap         {res['arm_a']:.2f}%")

    calibrate(student, calib, device, num_batches=150)
    res["arm_b"] = top1(student, val, device)
    print(f"B calibrated         {res['arm_b']:.2f}%")

    res["recon_losses"] = reconstruct_all(student, teacher, calib, device,
                                          steps=args.recon_steps)
    res["arm_c"] = top1(student, val, device)
    print(f"C reconstructed      {res['arm_c']:.2f}%")

    big = build_calib_loader(args.data_root, teacher,
                             num_images=args.distill_images, batch_size=128)
    distill(student, teacher, big, device, epochs=args.distill_epochs)
    res["arm_d"] = top1(student, val, device)
    print(f"D distilled          {res['arm_d']:.2f}%")

    # GATE 2 on the real trained model, before any latency claim.
    x = torch.randn(4, 3, 224, 224, device=device)
    res["fold_max_diff"] = assert_fold_equivalent(student, x, atol=1e-4)
    print(f"fold max diff        {res['fold_max_diff']:.3e}  GATE PASSED")

    folded = fold_model(student)
    res["arm_d_folded"] = top1(folded, val, device)
    res["norms_before"] = count_norms(teacher)
    res["norms_after"] = count_norms(folded)
    print(f"D folded             {res['arm_d_folded']:.2f}%")
    print(f"norms {res['norms_before']} -> {res['norms_after']}")

    res["latency"] = {
        "teacher_eager": benchmark_latency(teacher, device, compile_model=False),
        "teacher_compiled": benchmark_latency(teacher, device, compile_model=True),
        "folded_eager": benchmark_latency(folded, device, compile_model=False),
        "folded_compiled": benchmark_latency(folded, device, compile_model=True),
    }
    for k, v in res["latency"].items():
        print(f"  {k:20s} {v['ms_per_batch']:7.3f} ms  {v['img_per_s']:9.1f} img/s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"arms_{args.model.split('.')[0]}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the runner with tiny settings**

Run:
```bash
.venv/bin/python experiments/run_arms.py --data-root $IMAGENET_ROOT \
    --calib-images 512 --distill-images 512 --recon-steps 20 --distill-epochs 1 \
    --out results/smoke
```
Expected: completes in a few minutes, all four arms print, `GATE PASSED` appears. Accuracy will be poor — the point is that the pipeline runs end to end and the fold gate holds.

- [ ] **Step 7: Run the real thing**

Run: `.venv/bin/python experiments/run_arms.py --data-root $IMAGENET_ROOT`
Expected: a few hours on a 4090. Arm A near chance, monotone improvement through D, `norms 25 -> 0`, and `arm_d_folded` within 1e-4-induced noise of `arm_d`.

If arm C does not improve substantially over arm B, reconstruction is not working — check the per-block losses in `recon_losses` are decreasing before investing in longer runs.

- [ ] **Step 8: Commit**

```bash
git add src/vitbn/evaluate.py experiments/run_arms.py tests/test_evaluate.py results/
git commit -m "feat: latency benchmark and arms A-D runner"
```

---

## Task 9: Figures and the scaling arm

**Why:** the deliverable is the trade-off curve, not the raw JSON. The scaling arm answers the obvious "does this hold at scale?" objection and is nearly free given everything else is built.

**Files:**
- Create: `experiments/make_figures.py`

- [ ] **Step 1: Write the figure script**

```python
# experiments/make_figures.py
"""Figures: per-layer damage (Experiment 0), the arm ladder, and scaling."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_per_layer(path: Path, out: Path):
    d = json.loads(path.read_text())
    base, per = d["baseline"], d["per_layer"]
    names = list(per)
    drops = [base - per[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(names)), drops,
           color=["#c44" if x > 1.0 else "#48a" for x in drops])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("top-1 drop (points)")
    ax.set_title(f"Per-layer conversion damage — {d['model']}\n"
                 f"baseline {base:.2f}%, one norm converted at a time, "
                 "calibration only")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(out / "exp0_per_layer.png", dpi=150)
    print(f"wrote {out / 'exp0_per_layer.png'}")


def fig_arms(paths: list[Path], out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in paths:
        d = json.loads(p.read_text())
        arms = ["arm_a", "arm_b", "arm_c", "arm_d"]
        accs = [d[a] for a in arms]
        ax.plot(range(4), accs, "o-", label=d["model"].split(".")[0])
        ax.axhline(d["teacher"], ls="--", lw=0.8, alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["A naive", "B calib", "C recon", "D distill"])
    ax.set_ylabel("top-1 (%)")
    ax.set_title("Recovery ladder (dashed = teacher)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "arms_ladder.png", dpi=150)
    print(f"wrote {out / 'arms_ladder.png'}")


def fig_tradeoff(paths: list[Path], out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in paths:
        d = json.loads(p.read_text())
        lat = d["latency"]
        speedup = (lat["teacher_compiled"]["ms_per_batch"]
                   / lat["folded_compiled"]["ms_per_batch"])
        lost = d["teacher"] - d["arm_d_folded"]
        ax.scatter(speedup, lost, s=80)
        ax.annotate(d["model"].split(".")[0], (speedup, lost),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("speedup vs torch.compile'd LayerNorm baseline")
    ax.set_ylabel("top-1 lost (points)")
    ax.set_title("Accuracy cost vs latency gain, all 25 norms folded")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    print(f"wrote {out / 'tradeoff.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    r = Path(args.results)

    for p in sorted(r.glob("exp0_*.json")):
        fig_per_layer(p, r)
    arms = sorted(r.glob("arms_*.json"))
    if arms:
        fig_arms(arms, r)
        fig_tradeoff(arms, r)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate figures from the ViT-Tiny results**

Run: `.venv/bin/python experiments/make_figures.py`
Expected: `results/exp0_per_layer.png`, `results/arms_ladder.png`, `results/tradeoff.png`.

- [ ] **Step 3: Run the scaling arm**

Run:
```bash
for m in vit_small_patch16_224.augreg_in21k_ft_in1k \
         vit_base_patch16_224.augreg_in21k_ft_in1k; do
    .venv/bin/python experiments/reproduce_teacher.py --data-root $IMAGENET_ROOT --model $m
    .venv/bin/python experiments/run_arms.py --data-root $IMAGENET_ROOT --model $m
done
```
Expected: `GATE PASSED` for each before its arms run; two more `arms_*.json`.

Note the teacher gate runs first for each model. A checkpoint that does not reproduce its published number must not contribute a data point.

- [ ] **Step 4: Regenerate figures with all three models**

Run: `.venv/bin/python experiments/make_figures.py`
Expected: ladder and trade-off plots now show three curves.

- [ ] **Step 5: Commit**

```bash
git add experiments/make_figures.py results/
git commit -m "feat: result figures and scaling arm across Tiny/Small/Base"
```

---

## Task 10: ONNX export and Hailo-10H compilation

**Why:** this is the actual deliverable. Every preceding task exists to produce a graph the Hailo Dataflow Compiler will accept. LayerNorm is the sole blocking operator, so the decisive check is whether the exported graph still contains one.

Hailo supports BatchNorm and folds it itself, so **two graphs are exported**: `vit_bn.onnx` with BatchNorm intact (the deployment artifact, folded by the DFC) and `vit_bn_folded.onnx` folded in PyTorch (the verification artifact, and the one benchmarked on GPU). Folding is exact algebra in both implementations, so the two must agree on device; a disagreement is a real bug worth finding.

Two gates run before the compiler is ever invoked, because a DFC failure is slow and its diagnostics are far less specific than an assertion here.

**Files:**
- Create: `src/vitbn/export.py`
- Create: `experiments/export_hailo.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `fold_model` from `vitbn.fold`
- Produces: `export_onnx(model, path, input_size, opset) -> Path`; `assert_no_norm_nodes(path) -> list[str]`; `assert_onnx_matches_torch(model, path, x, atol) -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_export.py
import torch

from vitbn.calibrate import calibrate
from vitbn.export import assert_no_norm_nodes, assert_onnx_matches_torch, export_onnx
from vitbn.fold import fold_model
from vitbn.models import load_student, load_teacher
from vitbn.norm_swap import swap_all_norms


def _converted_folded_model():
    torch.manual_seed(0)
    device = torch.device("cpu")
    m = load_student(device=device)
    swap_all_norms(m)
    loader = [(torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
              for _ in range(3)]
    calibrate(m, loader, device, num_batches=3)
    return fold_model(m)


def test_unconverted_model_exports_with_layernorm(tmp_path):
    """Confirms the premise: the stock model DOES contain the blocking op."""
    p = export_onnx(load_teacher(device=torch.device("cpu")),
                    tmp_path / "stock.onnx")
    found = assert_no_norm_nodes(p, raise_on_found=False)
    assert found, "stock ViT should contain LayerNorm nodes"


def test_folded_model_exports_with_no_norm_nodes(tmp_path):
    """The decisive check. LayerNorm is the sole Hailo blocker, and folding
    must remove BatchNorm too -- neither may survive export."""
    p = export_onnx(_converted_folded_model(), tmp_path / "folded.onnx")
    assert assert_no_norm_nodes(p) == []


def test_onnx_output_matches_pytorch(tmp_path):
    m = _converted_folded_model()
    p = export_onnx(m, tmp_path / "folded.onnx")
    x = torch.randn(1, 3, 224, 224)
    assert assert_onnx_matches_torch(m, p, x, atol=1e-4) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vitbn.export'`

- [ ] **Step 3: Add the export dependencies**

Add to `pyproject.toml` dependencies: `"onnx>=1.16"`, `"onnxruntime>=1.18"`, then `.venv/bin/pip install -e ".[dev]"`.

- [ ] **Step 4: Implement `export.py`**

```python
# src/vitbn/export.py
"""ONNX export and pre-compilation gates for Hailo-10H.

Folding happens in PyTorch, so the exported graph contains no normalization
node of any kind. LayerNorm is the sole operator the Hailo Dataflow Compiler
rejects for this model; BatchNorm never reaches the compiler because folding
removes it first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch

# Normalization-shaped ONNX ops. LayerNormalization appears at opset >= 17;
# older exporters emit the ReduceMean/Sub/Pow/Sqrt/Div decomposition instead,
# so both spellings are checked.
NORM_OPS = {
    "LayerNormalization", "BatchNormalization",
    "InstanceNormalization", "GroupNormalization",
    "SimplifiedLayerNormalization", "RMSNormalization",
}


def export_onnx(model, path, input_size=(3, 224, 224), opset: int = 17,
                batch_size: int = 1) -> Path:
    """Export to ONNX with fully static shapes.

    Hailo requires a static graph: fixed batch and fixed resolution. Dynamic
    axes would be rejected at parse time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(batch_size, *input_size)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
        dynamo=False,          # legacy exporter: static shapes, stable op set
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def assert_no_norm_nodes(path, raise_on_found: bool = True) -> list[str]:
    """GATE 3. Return the normalization nodes present in the graph.

    An empty list is what Hailo needs. A non-empty list means folding did not
    remove everything, and the DFC will reject the model -- catching it here
    is far faster and far more specific than a compiler error.
    """
    graph = onnx.load(str(path)).graph
    found = [f"{n.op_type}:{n.name}" for n in graph.node if n.op_type in NORM_OPS]
    if found and raise_on_found:
        raise RuntimeError(
            f"GATE FAILED: {len(found)} normalization node(s) survived export: "
            f"{found[:5]}. Hailo will reject this graph."
        )
    return found


def assert_onnx_matches_torch(model, path, x: torch.Tensor,
                              atol: float = 1e-4) -> float:
    """GATE 4. Confirm export preserved the function.

    Guards against export-time graph rewrites silently changing behaviour.
    Without this, an accuracy regression on device is ambiguous between the
    conversion and the export.
    """
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        expected = model(x).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x.numpy()})[0]

    max_diff = float(np.abs(expected - got).max())
    if max_diff > atol:
        raise RuntimeError(
            f"GATE FAILED: ONNX differs from PyTorch by {max_diff:.3e} > {atol}"
        )
    return max_diff
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: PASS, 3 tests. `test_unconverted_model_exports_with_layernorm` confirms the premise from the other direction — the stock model really does contain the blocking op.

- [ ] **Step 6: Write the export script**

```python
# experiments/export_hailo.py
"""Export the converted, folded model for Hailo-10H.

Runs both pre-compilation gates, then prints the DFC invocation. The gates
run here rather than inside the compiler because a DFC rejection is slow and
its diagnostics are far less specific than these assertions.
"""
import argparse
from pathlib import Path

import torch

from vitbn.calibrate import calibrate
from vitbn.data import build_calib_loader
from vitbn.evaluate import count_norms, top1
from vitbn.export import assert_no_norm_nodes, assert_onnx_matches_torch, export_onnx
from vitbn.fold import fold_model
from vitbn.models import DEFAULT_MODEL, load_student
from vitbn.norm_swap import swap_all_norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", help="converted student .pt from run_arms")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/hailo")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student = load_student(args.model, device)
    swap_all_norms(student)
    if args.checkpoint:
        student.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        calib = build_calib_loader(args.data_root, student, num_images=10_000)
        calibrate(student, calib, device, num_batches=150)

    print(f"norms before fold: {count_norms(student)}")
    folded = fold_model(student).cpu()
    print(f"norms after fold:  {count_norms(folded)}")

    out = Path(args.out)
    path = export_onnx(folded, out / "vit_bn_folded.onnx")
    print(f"exported {path}")

    assert_no_norm_nodes(path)
    print("GATE 3 PASSED: no normalization nodes in graph")

    diff = assert_onnx_matches_torch(folded, path, torch.randn(1, 3, 224, 224))
    print(f"GATE 4 PASSED: ONNX matches PyTorch, max diff {diff:.3e}")

    print(f"""
Next, on the machine with the Hailo Dataflow Compiler:

  hailo parser onnx {path} --hw-arch hailo10h
  hailo optimize  vit_bn_folded.har --hw-arch hailo10h
  hailo compiler  vit_bn_folded_optimized.har --hw-arch hailo10h

If the parser still rejects an operator, capture the exact op name -- it is
something other than LayerNorm and the plan needs revisiting.
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the export**

Run: `.venv/bin/python experiments/export_hailo.py --data-root $IMAGENET_ROOT`
Expected: `norms after fold: 0`, both gates pass, ONNX written.

- [ ] **Step 8: Compile with the DFC and record the outcome**

Run the three `hailo` commands printed above.
Expected: a `.hef`. If the parser rejects an operator, record its exact name — the premise was that LayerNorm is the only blocker, and a different rejection means that premise needs revisiting before further work.

- [ ] **Step 9: Commit**

```bash
git add src/vitbn/export.py experiments/export_hailo.py tests/test_export.py pyproject.toml
git commit -m "feat: ONNX export with no-norm-node and ONNX-parity gates for Hailo-10H"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Teacher reproduction gate | 1 |
| Fold equivalence gate | 4 (unit), 8 (on the trained model) |
| `timm` transform via `resolve_data_config` | 1 |
| Label-free calibration/distillation data | 1 (loaders), 7 (test asserts it) |
| `norm_swap` with γ/β transfer | 2 |
| `calibrate`, forward-only | 3 |
| `reconstruct`, student-input/teacher-output | 6 |
| `distill`, KL + feature MSE | 7 |
| Fold math, all 25 norms | 4 |
| Experiment 0 per-layer damage | 5 |
| Arms A–D | 8 |
| Latency incl. `torch.compile` baseline | 8 |
| Norms-remaining metric | 8 (`count_norms`) |
| Scaling arm | 9 |
| Primary + secondary figures | 9 |

**Not covered, deliberately:** the ablations (calibration set size, BN-affine-only, retaining `LN_f`, reconstruction step count) and the coverage-relaxation fallback. Both are contingent on Experiment 0's outcome and would be planned separately once its result is known. The `--calib-images` and `--recon-steps` flags already parameterize two of the four ablations.

**Type consistency:** `norm_paths`/`fold_pairs`/`set_submodule` (models.py) used identically in norm_swap.py and fold.py. `BatchNorm.bn` is the attribute name in every consumer. `capture_block_io` returns `(input, output)` in that order in both reconstruct.py and its tests. `top1(model, loader, device)` signature is consistent across all three experiment scripts.

**Placeholder scan:** none.
