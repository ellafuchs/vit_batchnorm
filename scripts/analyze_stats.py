#!/usr/bin/env python
"""Measure activation and BatchNorm statistics of a trained model.

Tests, on the real trained model, the claim that from-scratch BatchNorm
gives milder activation outliers than LayerNorm does -- the thing that
governs how well it will quantize to int8.

Reports, per block:
  - the largest activation magnitude in the residual stream (the outlier)
  - the ratio of that max to the typical (99th-percentile) value
  - the spread of the learned BatchNorm running statistics

A high max/typical ratio means a few values dominate the range, which is
exactly what makes int8 hard. Run it on both the from-scratch model and,
later, the converted-from-LayerNorm one to compare.
"""
from __future__ import annotations

import argparse

import timm
import torch
import torch.nn as nn

from vitbn.norm import BatchNorm


def load(checkpoint: str, device):
    model = timm.create_model("vit_tiny_patch16_224_bn")
    sd = torch.load(checkpoint, map_location="cpu")
    sd = sd.get("state_dict_ema") or sd.get("state_dict") or sd
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def block_activation_stats(model, x):
    """Max and 99th-percentile magnitude of each block's output."""
    stats = []
    outs = []
    handles = [b.register_forward_hook(lambda _m, _a, o: outs.append(o))
               for b in model.blocks]
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    for i, o in enumerate(outs):
        a = o.abs().flatten()
        mx = a.max().item()
        p99 = torch.quantile(a[:200000] if a.numel() > 200000 else a, 0.99).item()
        stats.append((i, mx, p99, mx / max(p99, 1e-9)))
    return stats


def bn_stat_spread(model):
    """How wide the learned running statistics are, across all BN layers."""
    means, stds = [], []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            means.append(m.running_mean.abs().max().item())
            stds.append(m.running_var.sqrt().max().item())
    return means, stds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.checkpoint}")
    model = load(args.checkpoint, device)

    # Random input is enough to reveal structural outliers -- the sink
    # dimensions are baked into the weights, not the specific image.
    x = torch.randn(8, 3, 224, 224, device=device)
    stats = block_activation_stats(model, x)

    print("\nper-block activation magnitude (the outlier story):")
    print(f"  {'block':>5}  {'max':>9}  {'p99':>8}  {'max/p99':>8}")
    for i, mx, p99, ratio in stats:
        flag = "  <-- strong outlier" if ratio > 20 else ""
        print(f"  {i:>5}  {mx:9.2f}  {p99:8.2f}  {ratio:8.1f}{flag}")

    worst = max(s[3] for s in stats)
    print(f"\nworst max/p99 ratio: {worst:.1f}")
    print("  (higher = harder to quantize with a single int8 scale;")
    print("   per-channel quantization handles it, per-tensor struggles)")

    means, stds = bn_stat_spread(model)
    print(f"\nBatchNorm running stats across {len(means)} layers:")
    print(f"  |running_mean| max: {max(means):.2f}")
    print(f"  running_std    max: {max(stds):.2f}")


if __name__ == "__main__":
    main()
