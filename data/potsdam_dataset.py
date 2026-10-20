"""
ASG-Net: ISPRS Potsdam Dataset and DataLoader
============================================
PyTorch Dataset and DataLoader implementation for the ISPRS Potsdam benchmark.
Supports synchronized multimodal loading of RGB, Near-Infrared (NIR),
semantic segmentation masks, and dynamically computed boundary edge maps.

Directory Structure Expected:
-----------------------------
{root_dir}/
    rgb/
        train/
            top_potsdam_2_10_y0_x0.png
            ...
        val/
        test/
    nir/
        train/
            top_potsdam_2_10_y0_x0.png
            ...
        val/
        test/
    labels/
        train/
            top_potsdam_2_10_y0_x0.png
            ...
        val/
        test/
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import yaml

from .transforms import (
    MultimodalTransform,
    ToTensor,
    generate_edge_map,
    get_train_transforms,
    get_val_transforms,
)


class PotsdamDataset(Dataset):
    """
    PyTorch Dataset for ISPRS Potsdam Multimodal Semantic Segmentation.

    Loads aligned triplets of:
      1. RGB imagery (3 channels, 5cm GSD)
      2. Near-Infrared (NIR) band (1 channel, 5cm GSD)
      3. Semantic class mask (6 urban classes, indices 0-5)
      4. Ground-truth semantic boundary edge map (derived via morphological gradient)

    Args:
        root_dir: Root directory where preprocessed patches are stored.
        split: Dataset split ('train', 'val', or 'test').
        transform: Optional synchronized multimodal transform pipeline.
        patch_size: Expected height and width of patches in pixels (default: 512).
        stride: Extraction stride used when patches were prepared (default: 256).
        generate_edges: Whether to compute boundary edge maps on the fly (default: True).
    """

    NUM_CLASSES: int = 6
    CLASS_NAMES: Sequence[str] = (
        "Impervious Surface",  # Class 0: White
        "Building",            # Class 1: Blue
        "Low Vegetation",      # Class 2: Cyan
        "Tree",                # Class 3: Green
        "Car",                 # Class 4: Yellow
        "Clutter",             # Class 5: Red
    )

    CLASS_COLORS: Sequence[Tuple[int, int, int]] = (
        (255, 255, 255),  # 0: Impervious Surface
        (0, 0, 255),      # 1: Building
        (0, 255, 255),    # 2: Low Vegetation
        (0, 255, 0),      # 3: Tree
        (255, 255, 0),    # 4: Car
        (255, 0, 0),      # 5: Clutter
    )

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        transform: Optional[Union[MultimodalTransform, Callable]] = None,
        patch_size: int = 512,
        stride: int = 256,
        generate_edges: bool = True,
        **kwargs: Any
    ):
        super().__init__()
        # Allow root or root_dir interchangeably
        actual_root = root_dir if root_dir is not None else kwargs.get("root", "data/potsdam")
        # Allow transform or transforms interchangeably
        self.transform = transform if transform is not None else kwargs.get("transforms", None)

        self.split = split.lower().strip()
        self.patch_size = patch_size
        self.stride = stride
        self.generate_edges = generate_edges

        self.root_dir = Path(actual_root)
        # Automatic fallback to data/potsdam_mini if specified root split directory does not exist
        if not (self.root_dir / "rgb" / self.split).is_dir():
            alt_mini = Path("data/potsdam_mini")
            if (alt_mini / "rgb" / self.split).is_dir():
                self.root_dir = alt_mini
            elif (Path("/content/data/potsdam_mini") / "rgb" / self.split).is_dir():
                self.root_dir = Path("/content/data/potsdam_mini")

        valid_splits = {"train", "val", "test"}
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split '{self.split}'. Expected one of {valid_splits}")

        self.rgb_dir = self.root_dir / "rgb" / self.split
        self.nir_dir = self.root_dir / "nir" / self.split
        self.label_dir = self.root_dir / "labels" / self.split

        self.filenames: List[str] = self._discover_samples()

    def _discover_samples(self) -> List[str]:
        """
        Discovers and validates all patch filenames present across rgb, nir, and label folders.
        """
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(
                f"Potsdam RGB directory not found: {self.rgb_dir}\n"
                f"Please run 'python data/download_potsdam.py --raw_dir <RAW_DIR> "
                f"--output_dir {self.root_dir}' to prepare the dataset patches."
            )

        # Collect image files in the rgb split directory
        valid_extensions = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
        rgb_files = [
            f.name
            for f in os.scandir(self.rgb_dir)
            if f.is_file() and Path(f.name).suffix.lower() in valid_extensions
        ]

        if not rgb_files:
            raise FileNotFoundError(
                f"No patch files found in {self.rgb_dir}. "
                "Ensure patches have been preprocessed with download_potsdam.py."
            )

        # Sort for deterministic indexing
        rgb_files.sort()

        # Check corresponding NIR and label files
        missing_nir = []
        missing_labels = []
        for fname in rgb_files:
            if not (self.nir_dir / fname).is_file():
                missing_nir.append(fname)
            if not (self.label_dir / fname).is_file():
                missing_labels.append(fname)

        if missing_nir:
            raise FileNotFoundError(
                f"Missing {len(missing_nir)} corresponding NIR files in {self.nir_dir}. "
                f"First missing: {missing_nir[0]}"
            )
        if missing_labels:
            raise FileNotFoundError(
                f"Missing {len(missing_labels)} corresponding label files in {self.label_dir}. "
                f"First missing: {missing_labels[0]}"
            )

        return rgb_files

    def __len__(self) -> int:
        """Returns the total number of patches in this dataset split."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single patch triplet, applies synchronized multimodal transformations,
        and dynamically computes the boundary edge map.

        Returns:
            A dictionary containing:
              - 'rgb': FloatTensor of shape (3, H, W)
              - 'nir': FloatTensor of shape (1, H, W)
              - 'mask': LongTensor of shape (H, W) with class indices [0, 5]
              - 'edge': FloatTensor of shape (H, W) with binary boundary values (0.0 or 1.0)
              - 'filename': str base filename of the patch
        """
        fname = self.filenames[idx]
        rgb_path = self.rgb_dir / fname
        nir_path = self.nir_dir / fname
        label_path = self.label_dir / fname

        # 1. Load RGB image (H, W, 3) in [0, 255]
        if HAS_CV2 and cv2 is not None:
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb_arr = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        else:
            rgb_arr = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        # 2. Load NIR image (H, W, 1) or (H, W) in [0, 255]
        if HAS_CV2 and cv2 is not None:
            nir_raw = cv2.imread(str(nir_path), cv2.IMREAD_UNCHANGED)
            if nir_raw is not None:
                nir_arr = nir_raw
            else:
                nir_arr = np.array(Image.open(nir_path), dtype=np.uint8)
        else:
            nir_arr = np.array(Image.open(nir_path), dtype=np.uint8)

        if nir_arr.ndim == 2:
            nir_arr = nir_arr[:, :, np.newaxis]
        elif nir_arr.ndim == 3 and nir_arr.shape[-1] > 1:
            nir_arr = nir_arr[:, :, :1]

        # 3. Load label mask (H, W) with discrete class indices 0..5
        if HAS_CV2 and cv2 is not None:
            label_raw = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if label_raw is not None:
                mask_arr = label_raw
            else:
                mask_arr = np.array(Image.open(label_path), dtype=np.uint8)
        else:
            mask_arr = np.array(Image.open(label_path), dtype=np.uint8)

        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

        mask_arr = mask_arr.astype(np.int64)

        # 4. Apply synchronized multimodal transforms
        if self.transform is not None:
            sample = {"rgb": rgb_arr, "nir": nir_arr, "mask": mask_arr}
            transformed = self.transform(sample)
            rgb_out = transformed["rgb"]
            nir_out = transformed["nir"]
            mask_out = transformed["mask"]
        else:
            # If no transform specified, convert directly to tensors
            to_tensor = ToTensor()
            rgb_out, nir_out, mask_out = to_tensor.apply(rgb_arr, nir_arr, mask_arr)

        # Ensure mask is a 2D LongTensor
        if isinstance(mask_out, torch.Tensor):
            if mask_out.ndim == 3 and mask_out.shape[0] == 1:
                mask_out = mask_out.squeeze(0)
            mask_out = mask_out.long()
        else:
            mask_out = torch.from_numpy(mask_out).long()

        # 5. Generate edge map from the transformed mask
        if self.generate_edges:
            edge_out = generate_edge_map(mask_out)
            if not isinstance(edge_out, torch.Tensor):
                edge_out = torch.from_numpy(edge_out).float()
            else:
                edge_out = edge_out.float()
        else:
            edge_out = torch.zeros_like(mask_out, dtype=torch.float32)

        return {
            "rgb": rgb_out,
            "nir": nir_out,
            "mask": mask_out,
            "edge": edge_out,
            "filename": fname,
        }

    @classmethod
    def create_synthetic_dataset(
        cls,
        root_dir: Union[str, Path],
        num_samples_per_split: int = 4,
        patch_size: int = 512,
    ) -> None:
        """
        Creates a lightweight synthetic dataset with the proper folder hierarchy
        and random patches for rapid pipeline testing and verification without downloading
        large raw files.
        """
        root = Path(root_dir)
        splits = ["train", "val", "test"]

        for split in splits:
            rgb_dir = root / "rgb" / split
            nir_dir = root / "nir" / split
            label_dir = root / "labels" / split

            rgb_dir.mkdir(parents=True, exist_ok=True)
            nir_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for i in range(num_samples_per_split):
                fname = f"synthetic_{split}_{i:04d}.png"

                # Synthetic RGB (H, W, 3)
                rgb = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(rgb_dir / fname)

                # Synthetic NIR (H, W)
                nir = np.random.randint(0, 256, (patch_size, patch_size), dtype=np.uint8)
                Image.fromarray(nir).save(nir_dir / fname)

                # Synthetic Labels (H, W) in {0, 1, 2, 3, 4, 5}
                mask = np.random.randint(0, 6, (patch_size, patch_size), dtype=np.uint8)
                Image.fromarray(mask).save(label_dir / fname)


def _load_config_dict(config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Resolves config from dictionary or path."""
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return config


def get_dataloader(
    config: Union[Dict[str, Any], str, Path],
    split: str = "train",
    shuffle: Optional[bool] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """
    Factory function to instantiate a PyTorch DataLoader for the Potsdam dataset
    configured according to the project config.yaml.

    Args:
        config: Configuration dictionary or path to configs/config.yaml.
        split: Split to load ('train', 'val', or 'test').
        shuffle: Whether to shuffle data. If None, defaults to True for train, False for val/test.
        batch_size: Override batch size from config.
        num_workers: Override worker count from config.

    Returns:
        torch.utils.data.DataLoader configured for the requested split.
    """
    cfg = _load_config_dict(config)

    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})

    root_dir = dataset_cfg.get("root", "data/potsdam")
    patch_size = dataset_cfg.get("patch_size", 512)
    stride = dataset_cfg.get("stride", 256)

    is_train = split.lower().strip() == "train"

    # Set batch size
    if batch_size is None:
        batch_size = training_cfg.get("batch_size", 4) if is_train else 1

    # Set worker count
    if num_workers is None:
        num_workers = training_cfg.get("num_workers", 4)

    # Set shuffle flag
    if shuffle is None:
        shuffle = is_train

    # Choose transform pipeline
    if is_train:
        transform = get_train_transforms(cfg)
    else:
        transform = get_val_transforms(cfg)

    # Instantiate dataset
    dataset = PotsdamDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        patch_size=patch_size,
        stride=stride,
        generate_edges=True,
    )

    # CUDA pin_memory setting
    pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )
