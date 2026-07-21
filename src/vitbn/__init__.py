"""Importing this package registers the BatchNorm ViT with timm."""
from . import models  # noqa: F401  (side effect: @register_model runs)
from .norm import BatchNorm  # noqa: F401
