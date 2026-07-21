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
    """Absorb `bn` into `lin`, in place. `bn` must precede `lin`."""
    if bn.training:
        raise RuntimeError(
            "BatchNorm is in train mode; folding would capture batch "
            "statistics rather than running statistics. Call .eval() first."
        )
    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BatchNorm has no running statistics")
    if lin.bias is None:
        raise RuntimeError("fold target has no bias; cannot absorb the shift")

    # fp64 for the fold arithmetic: done once, so the equivalence check is
    # limited by the model's own precision rather than ours.
    w = lin.weight.data.double()
    b = lin.bias.data.double()

    s = bn.weight.data.double() / torch.sqrt(bn.running_var.data.double() + bn.eps)
    shift = bn.bias.data.double() - bn.running_mean.data.double() * s

    b = b + w @ shift          # uses the ORIGINAL W, before scaling
    w = w * s.unsqueeze(0)     # W is (out, in); scale INPUT channels

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
        if not isinstance(norm, BatchNorm):
            raise TypeError(
                f"{norm_path} is {type(norm).__name__}; only BatchNorm folds"
            )
        fold_bn_into_linear(norm.bn, model.get_submodule(lin_path))
        set_submodule(model, norm_path, nn.Identity())
    return model


@torch.no_grad()
def assert_fold_equivalent(model: nn.Module, x: torch.Tensor,
                           atol: float = 1e-4) -> float:
    """Fold a copy and confirm it computes the same function.

    Returns the max absolute logit difference. Raises if it exceeds `atol`.

    A failure means the exported model is not the model you evaluated, which
    makes every device measurement meaningless. Do not relax the tolerance to
    make this pass.
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
