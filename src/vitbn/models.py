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
    """Registered so `train.py --model vit_tiny_patch16_224_bn` resolves.

    pretrained is unsupported: no BatchNorm ViT checkpoint exists, which is
    the point of the project.
    """
    if pretrained:
        raise ValueError("no pretrained weights exist for this variant")
    # timm's factory injects bookkeeping kwargs that VisionTransformer
    # does not accept.
    for k in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(k, None)
    return make_bn_vit_tiny(**kwargs)


def norm_paths(model) -> list[str]:
    """The 25 norm module paths, in forward order."""
    paths: list[str] = []
    for i in range(len(model.blocks)):
        paths.append(f"blocks.{i}.norm1")
        paths.append(f"blocks.{i}.norm2")
    paths.append("norm")
    return paths


def fold_pairs(model) -> list[tuple[str, str]]:
    """(norm_path, linear_path) -- each norm's fold target.

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
