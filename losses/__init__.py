"""
Loss Functions for ASG-Net (Adaptive Spectral-Spatial Gated Fusion Network).

Exports:
    - CombinedLoss: Multi-task composite loss combining Cross-Entropy,
      Lovász-Softmax, and Boundary-Aware Binary Cross-Entropy.
    - LovaszSoftmax: Direct IoU surrogate optimization loss based on the
      Lovász extension of submodular set functions.
"""

from .combined_loss import CombinedLoss
from .lovasz_loss import LovaszSoftmax

__all__ = [
    "CombinedLoss",
    "LovaszSoftmax",
]
