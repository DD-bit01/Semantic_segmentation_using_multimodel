"""
ASG-Net: Data Pipeline Package
==============================
Provides dataset loaders, synchronized multimodal transformations, and preprocessing
utilities for the ISPRS Potsdam RGB + Multispectral benchmark.
"""

from .potsdam_dataset import PotsdamDataset, get_dataloader
from .transforms import (
    MultimodalTransform,
    generate_edge_map,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "PotsdamDataset",
    "get_dataloader",
    "get_train_transforms",
    "get_val_transforms",
    "MultimodalTransform",
    "generate_edge_map",
]
