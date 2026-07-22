#!/usr/bin/env python
"""Experiment 2: convert a pretrained LayerNorm ViT to BatchNorm.

The other half of the project. Experiment 1 trained BatchNorm from scratch
(~67%). This takes a pretrained LayerNorm model (DeiT-Ti, 72.2%) and swaps
its norms for BatchNorm, then fine-tunes to recover accuracy.

Why this is a different, harder problem than training from scratch:
LayerNorm normalizes each token over its channels, at runtime. BatchNorm
normalizes each channel over the batch, from stored statistics. They are
different functions along perpendicular axes, so a naive swap breaks the
model badly -- the pretrained weights were tuned expecting per-token
rescaling that BatchNorm does not provide.

Stages, cheapest first, so you see where accuracy comes from:
  A  naive swap                 -> expect near chance (~0.1-5%)
  B  + calibrate BN statistics  -> poor but above chance
  C  + fine-tune vs teacher     -> the real recovery

Run A and B first (minutes, no training) to confirm the swap behaves as
predicted before spending GPU time on C.
"""
from __future__ import annotations

import argparse

import timm
import torch
import torch.nn as nn

from vitbn.norm import BatchNorm


def swap_ln_to_bn(model: nn.Module) -> nn.Module:
    """Replace every LayerNorm with a BatchNorm, copying the learned
    gamma/beta across -- same shape, same role, a better start than 1/0."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                (dim,) = child.normalized_shape
                bn = BatchNorm(dim, eps=child.eps)
                with torch.no_grad():
                    bn.bn.weight.copy_(child.weight)
                    bn.bn.bias.copy_(child.bias)
                bn.to(child.weight.device, child.weight.dtype)
                setattr(module, child_name, bn)
    return model


@torch.no_grad()
def calibrate(model, loader, device, num_batches=150):
    """Populate BN running statistics with forward passes only. No training."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(x.to(device, non_blocking=True))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/converted_init.pth",
                    help="where to save the converted weights")
    args = ap.parse_args()

    print("loading pretrained DeiT-Ti (LayerNorm) ...")
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    ln = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
    print(f"  LayerNorms before swap: {ln}")

    swap_ln_to_bn(model)
    bn = sum(isinstance(m, BatchNorm) for m in model.modules())
    ln = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
    print(f"  BatchNorms after swap:  {bn}")
    print(f"  LayerNorms after swap:  {ln}")

    assert bn == 25 and ln == 0, "swap incomplete"

    # Confirm the converted state_dict loads into our registered BatchNorm
    # model -- this is what convert_finetune.sh fine-tunes. A key mismatch
    # here would mean the fine-tune silently starts from random weights,
    # which is exactly the bug this whole check exists to prevent.
    ref = timm.create_model("vit_tiny_patch16_224_bn")
    missing, unexpected = ref.load_state_dict(model.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    # Only BN running stats may be missing; fine-tuning populates them.
    bad = [k for k in missing
           if "running_" not in k and "num_batches" not in k]
    assert not bad, f"missing weight keys: {bad[:5]}"

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nsaved converted weights -> {args.out}")
    print("next: bash scripts/convert_finetune.sh")


if __name__ == "__main__":
    main()
