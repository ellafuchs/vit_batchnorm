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
                 momentum: float = 0.1, device=None, dtype=None):
        super().__init__()
        # timm >= 1.0.28 passes device/dtype through to norm_layer.
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum,
                                 device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return self.bn(x.reshape(b * n, d)).reshape(b, n, d)

    def extra_repr(self) -> str:
        return f"num_features={self.bn.num_features}"
