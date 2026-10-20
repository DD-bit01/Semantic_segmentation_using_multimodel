"""
ASG-Net: Synchronized Multimodal Augmentation Pipeline
=====================================================
Provides synchronized spatial and spectral transformations for multimodal
semantic segmentation (RGB + Near-Infrared + Semantic Mask + Edge Map).

Key Architecture Decision:
-------------------------
In multimodal remote sensing, spatial transformations (flips, rotations, scaling,
and cropping) MUST be applied identically across all modalities (RGB, NIR, Mask)
to preserve pixel-level spatial alignment. Conversely, photometric transforms
(e.g., ColorJitter) MUST only be applied to RGB imagery to prevent corrupting
the radiometric integrity of the physical NIR reflectance band.

All transforms operate on NumPy arrays (H, W, C) / (H, W) or dictionary samples,
and output PyTorch tensors formatted for SegFormer / MiT encoders.
"""

import math
import random
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
import torchvision.transforms.functional as TF
import yaml
from PIL import Image


def generate_edge_map(
    mask: Union[np.ndarray, torch.Tensor],
    edge_width: int = 3,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a binary semantic boundary/edge map from a multiclass label mask.

    Boundaries are detected where adjacent pixels have different semantic class labels.
    Uses morphological gradient (dilation - erosion) with a configurable kernel size.
    Boundary pixels are assigned 1.0 (edge) and interior pixels 0.0 (non-edge).

    Args:
        mask: 2D integer mask of shape (H, W), either NumPy array or PyTorch Tensor.
        edge_width: Width in pixels of the structuring element (kernel size). Default: 3.

    Returns:
        Binary edge map of shape (H, W) with float values 0.0 and 1.0, matching the input type.
    """
    is_tensor = isinstance(mask, torch.Tensor)
    if is_tensor:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = mask

    # Ensure mask is 2D integer
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np.squeeze(-1)

    mask_uint8 = mask_np.astype(np.uint8)

    # A morphological gradient (dilation - erosion) flags any pixel where
    # the local window contains at least two distinct class indices.
    if HAS_CV2 and cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
        grad = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)
        edge_np = (grad > 0).astype(np.float32)
    else:
        # Shift-based boundary detection in pure NumPy (zero external dependencies)
        pad = max(1, edge_width // 2)
        h, w = mask_uint8.shape[:2]
        padded = np.pad(mask_uint8, pad, mode='edge')
        edge_np = np.zeros((h, w), dtype=np.float32)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w]
                edge_np = np.maximum(edge_np, (mask_uint8 != shifted).astype(np.float32))

    if is_tensor:
        return torch.from_numpy(edge_np).to(dtype=torch.float32, device=mask.device)
    return edge_np


class MultimodalTransform:
    """
    Abstract base class for all synchronized multimodal transformations.

    Supports both calling conventions:
      1. sample dict: `transform({'rgb': rgb, 'nir': nir, 'mask': mask})`
      2. unpacked args: `rgb, nir, mask = transform(rgb, nir, mask)`
    """

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        """
        Subclasses must implement this method to transform the modalities.
        """
        raise NotImplementedError

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], Tuple[Any, ...]]:
        """
        Dispatches call based on whether a dictionary or unpacked arguments are supplied.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            sample = args[0]
            rgb = sample["rgb"]
            nir = sample["nir"]
            mask = sample.get("mask", None)

            out_rgb, out_nir, out_mask = self.apply(rgb, nir, mask)

            result = dict(sample)
            result["rgb"] = out_rgb
            result["nir"] = out_nir
            if out_mask is not None:
                result["mask"] = out_mask
            return result

        elif len(args) >= 2:
            rgb = args[0]
            nir = args[1]
            mask = args[2] if len(args) > 2 else kwargs.get("mask", None)
            return self.apply(rgb, nir, mask)

        elif "rgb" in kwargs and "nir" in kwargs:
            rgb = kwargs["rgb"]
            nir = kwargs["nir"]
            mask = kwargs.get("mask", None)
            return self.apply(rgb, nir, mask)

        else:
            raise ValueError(
                "MultimodalTransform expects either a dict with 'rgb'/'nir' keys "
                "or positional arguments (rgb, nir, [mask])."
            )


class Compose(MultimodalTransform):
    """
    Composes several multimodal transforms sequentially.
    """

    def __init__(self, transforms: Sequence[Union[MultimodalTransform, Callable]]):
        self.transforms = list(transforms)

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        for t in self.transforms:
            if isinstance(t, MultimodalTransform):
                rgb, nir, mask = t.apply(rgb, nir, mask)
            else:
                out = t({"rgb": rgb, "nir": nir, "mask": mask})
                rgb = out["rgb"]
                nir = out["nir"]
                mask = out.get("mask", mask)
        return rgb, nir, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "(\n"
        for t in self.transforms:
            format_string += f"    {t}\n"
        format_string += ")"
        return format_string


class RandomHorizontalFlip(MultimodalTransform):
    """
    Randomly flips RGB, NIR, and Mask horizontally with probability `p`.
    Spatial alignment across modalities is strictly preserved.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.p:
            rgb = np.fliplr(rgb).copy()
            nir = np.fliplr(nir).copy()
            if mask is not None:
                mask = np.fliplr(mask).copy()
        return rgb, nir, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip(MultimodalTransform):
    """
    Randomly flips RGB, NIR, and Mask vertically with probability `p`.
    Spatial alignment across modalities is strictly preserved.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.p:
            rgb = np.flipud(rgb).copy()
            nir = np.flipud(nir).copy()
            if mask is not None:
                mask = np.flipud(mask).copy()
        return rgb, nir, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotation90(MultimodalTransform):
    """
    Randomly rotates RGB, NIR, and Mask by orthogonal angles (0, 90, 180, 270 degrees).
    Preserves exact pixel boundaries without introducing interpolation artifacts.
    """

    def __init__(self, angles: Sequence[int] = (0, 90, 180, 270)):
        self.angles = list(angles)

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        angle = random.choice(self.angles)
        k = (angle // 90) % 4
        if k > 0:
            rgb = np.rot90(rgb, k=k, axes=(0, 1)).copy()
            nir = np.rot90(nir, k=k, axes=(0, 1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
        return rgb, nir, mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"


class RandomScale(MultimodalTransform):
    """
    Randomly scales RGB, NIR, and Mask by a factor chosen from `scale_range`.
    Bilinear interpolation is applied to continuous imagery (RGB, NIR).
    Nearest-neighbor interpolation is strictly applied to categorical class masks.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 2.0)):
        if scale_range[0] <= 0 or scale_range[1] < scale_range[0]:
            raise ValueError(f"Invalid scale_range: {scale_range}")
        self.scale_range = scale_range

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        if math.isclose(scale, 1.0, rel_tol=1e-3):
            return rgb, nir, mask

        h, w = rgb.shape[:2]
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        if HAS_CV2 and cv2 is not None:
            # Bilinear for RGB
            rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Bilinear for NIR (handle 2D or 3D)
            nir_ndim = nir.ndim
            if nir_ndim == 3 and nir.shape[-1] == 1:
                nir_scaled = cv2.resize(nir[:, :, 0], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                nir_scaled = nir_scaled[:, :, np.newaxis]
            else:
                nir_scaled = cv2.resize(nir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Nearest-neighbor for categorical segmentation mask
            mask_scaled = None
            if mask is not None:
                mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            # Fallback to Pillow
            rgb_scaled = np.array(Image.fromarray(rgb).resize((new_w, new_h), resample=Image.BILINEAR))
            nir_ndim = nir.ndim
            if nir_ndim == 3 and nir.shape[-1] == 1:
                nir_scaled = np.array(Image.fromarray(nir[:, :, 0]).resize((new_w, new_h), resample=Image.BILINEAR))[:, :, np.newaxis]
            else:
                nir_scaled = np.array(Image.fromarray(nir).resize((new_w, new_h), resample=Image.BILINEAR))
            mask_scaled = None
            if mask is not None:
                mask_scaled = np.array(Image.fromarray(mask).resize((new_w, new_h), resample=Image.NEAREST))

        return rgb_scaled, nir_scaled, mask_scaled

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"


class RandomCrop(MultimodalTransform):
    """
    Synchronously crops a patch of size `(size, size)` from RGB, NIR, and Mask.
    If the image dimensions are smaller than `size` (e.g. after scaling down),
    the imagery is safely padded before cropping:
      - RGB and NIR are padded with constant reflection/zeros
      - Mask is padded with `pad_mask_value` (default: 5, clutter/background)
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 512,
        pad_if_needed: bool = True,
        pad_mask_value: int = 5,
    ):
        if isinstance(size, int):
            self.target_h = size
            self.target_w = size
        else:
            self.target_h, self.target_w = size
        self.pad_if_needed = pad_if_needed
        self.pad_mask_value = pad_mask_value

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w = rgb.shape[:2]

        # Pad if imagery is smaller than target crop window
        if self.pad_if_needed and (h < self.target_h or w < self.target_w):
            pad_h = max(0, self.target_h - h)
            pad_w = max(0, self.target_w - w)

            # Pad RGB
            pad_mode = "reflect" if (h > 1 and w > 1) else "constant"
            pad_kwargs = {} if pad_mode == "reflect" else {"constant_values": 0}
            rgb = np.pad(
                rgb,
                ((0, pad_h), (0, pad_w), (0, 0)) if rgb.ndim == 3 else ((0, pad_h), (0, pad_w)),
                mode=pad_mode,
                **pad_kwargs,
            )

            # Pad NIR
            if nir.ndim == 3:
                nir = np.pad(
                    nir,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode=pad_mode,
                    **pad_kwargs,
                )
            else:
                nir = np.pad(
                    nir,
                    ((0, pad_h), (0, pad_w)),
                    mode=pad_mode,
                    **pad_kwargs,
                )

            # Pad Mask
            if mask is not None:
                mask = np.pad(
                    mask,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=self.pad_mask_value,
                )

            h, w = rgb.shape[:2]

        # Compute valid random crop origin
        if h > self.target_h:
            y0 = random.randint(0, h - self.target_h)
        else:
            y0 = 0

        if w > self.target_w:
            x0 = random.randint(0, w - self.target_w)
        else:
            x0 = 0

        y1 = y0 + self.target_h
        x1 = x0 + self.target_w

        rgb_crop = rgb[y0:y1, x0:x1].copy()
        nir_crop = nir[y0:y1, x0:x1].copy()
        mask_crop = mask[y0:y1, x0:x1].copy() if mask is not None else None

        return rgb_crop, nir_crop, mask_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size=({self.target_h}, {self.target_w}))"


class ColorJitter(MultimodalTransform):
    """
    Applies random brightness, contrast, saturation, and hue jitter ONLY to RGB.
    NIR and Mask remain completely untouched to preserve spectral fidelity.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def apply(
        self,
        rgb: np.ndarray,
        nir: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Ensure RGB is uint8 in range [0, 255] for PIL / torchvision transform
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb

        pil_img = Image.fromarray(rgb_uint8)

        # Apply random factors
        if self.brightness > 0:
            b_factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            pil_img = TF.adjust_brightness(pil_img, b_factor)

        if self.contrast > 0:
            c_factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            pil_img = TF.adjust_contrast(pil_img, c_factor)

        if self.saturation > 0:
            s_factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
            pil_img = TF.adjust_saturation(pil_img, s_factor)

        if self.hue > 0:
            h_factor = random.uniform(-self.hue, self.hue)
            pil_img = TF.adjust_hue(pil_img, h_factor)

        jittered_rgb = np.array(pil_img)

        # Match incoming dtype
        if rgb.dtype != np.uint8:
            jittered_rgb = (jittered_rgb / 255.0).astype(rgb.dtype)

        return jittered_rgb, nir, mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"
        )


class Normalize(MultimodalTransform):
    """
    Normalizes RGB and NIR channels using modality-specific mean and standard deviation.

    RGB default stats: ImageNet statistics [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    NIR default stats: Potsdam NIR approximate statistics [0.5], [0.25]
    """

    def __init__(
        self,
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        nir_mean: Sequence[float] = (0.5,),
        nir_std: Sequence[float] = (0.25,),
    ):
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.nir_mean = np.array(nir_mean, dtype=np.float32)
        self.nir_std = np.array(nir_std, dtype=np.float32)

    def apply(
        self,
        rgb: Union[np.ndarray, torch.Tensor],
        nir: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[Any, Any, Optional[Any]]:
        # Normalize RGB
        if isinstance(rgb, torch.Tensor):
            mean = torch.tensor(self.rgb_mean, device=rgb.device, dtype=rgb.dtype).view(-1, 1, 1)
            std = torch.tensor(self.rgb_std, device=rgb.device, dtype=rgb.dtype).view(-1, 1, 1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            norm_rgb = (rgb - mean) / std
        else:
            rgb_f = rgb.astype(np.float32)
            if rgb_f.max() > 1.0:
                rgb_f = rgb_f / 255.0
            norm_rgb = (rgb_f - self.rgb_mean) / self.rgb_std

        # Normalize NIR
        if isinstance(nir, torch.Tensor):
            mean_n = torch.tensor(self.nir_mean, device=nir.device, dtype=nir.dtype).view(-1, 1, 1)
            std_n = torch.tensor(self.nir_std, device=nir.device, dtype=nir.dtype).view(-1, 1, 1)
            if nir.max() > 1.0:
                nir = nir / 255.0
            norm_nir = (nir - mean_n) / std_n
        else:
            nir_f = nir.astype(np.float32)
            if nir_f.max() > 1.0:
                nir_f = nir_f / 255.0
            norm_nir = (nir_f - self.nir_mean) / self.nir_std

        return norm_rgb, norm_nir, mask

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(rgb_mean={self.rgb_mean.tolist()}, "
            f"rgb_std={self.rgb_std.tolist()}, nir_mean={self.nir_mean.tolist()}, "
            f"nir_std={self.nir_std.tolist()})"
        )


class ToTensor(MultimodalTransform):
    """
    Converts NumPy arrays to PyTorch Tensors:
      - RGB: (H, W, 3) -> (3, H, W) FloatTensor in [0, 1] (if not already normalized)
      - NIR: (H, W, 1) or (H, W) -> (1, H, W) FloatTensor in [0, 1]
      - Mask: (H, W) -> (H, W) LongTensor with discrete class indices
    """

    def apply(
        self,
        rgb: Union[np.ndarray, torch.Tensor],
        nir: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Convert RGB
        if isinstance(rgb, torch.Tensor):
            tensor_rgb = rgb.float()
        else:
            # (H, W, C) -> (C, H, W)
            if rgb.ndim == 3:
                tensor_rgb = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
            else:
                tensor_rgb = torch.from_numpy(rgb).unsqueeze(0).float()

        # Convert NIR
        if isinstance(nir, torch.Tensor):
            tensor_nir = nir.float()
            if tensor_nir.ndim == 2:
                tensor_nir = tensor_nir.unsqueeze(0)
        else:
            if nir.ndim == 3:
                tensor_nir = torch.from_numpy(nir.transpose(2, 0, 1)).float()
            else:
                tensor_nir = torch.from_numpy(nir).unsqueeze(0).float()

        # Convert Mask
        tensor_mask = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                tensor_mask = mask.long()
            else:
                tensor_mask = torch.from_numpy(mask).long()

        return tensor_rgb, tensor_nir, tensor_mask

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _load_config(config: Optional[Union[Dict[str, Any], str, Path]] = None) -> Dict[str, Any]:
    """Helper to load config from dictionary, file path, or default project path."""
    if config is None:
        default_path = Path("configs/config.yaml")
        if default_path.is_file():
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}
    if isinstance(config, (str, Path)):
        with open(config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return config


def get_train_transforms(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> Compose:
    """
    Builds the composed training augmentation pipeline configured in config.yaml.

    Pipeline Order:
      1. RandomScale (optional)
      2. RandomCrop (ensures fixed patch size, e.g. 512x512)
      3. RandomHorizontalFlip (p=0.5)
      4. RandomVerticalFlip (p=0.5)
      5. RandomRotation90
      6. ColorJitter (RGB only)
      7. Normalize (RGB ImageNet stats, NIR stats)
      8. ToTensor ((C, H, W) tensors)

    Args:
        config: Loaded config dictionary, path to config.yaml, or None for defaults.

    Returns:
        Compose transform object containing all synchronized stages.
    """
    cfg = _load_config(config)
    aug_cfg = cfg.get("augmentation", {})
    dataset_cfg = cfg.get("dataset", {})
    patch_size = dataset_cfg.get("patch_size", 512)

    transforms_list: List[MultimodalTransform] = []

    # 1. Random Scale
    if aug_cfg.get("random_scale", True):
        scale_range = tuple(aug_cfg.get("scale_range", [0.5, 2.0]))
        transforms_list.append(RandomScale(scale_range=scale_range))

    # 2. Random Crop
    transforms_list.append(RandomCrop(size=patch_size, pad_if_needed=True, pad_mask_value=5))

    # 3. Random Horizontal Flip
    if aug_cfg.get("random_horizontal_flip", True):
        transforms_list.append(RandomHorizontalFlip(p=0.5))

    # 4. Random Vertical Flip
    if aug_cfg.get("random_vertical_flip", True):
        transforms_list.append(RandomVerticalFlip(p=0.5))

    # 5. Random 90-degree Rotation
    if aug_cfg.get("random_rotation", True):
        degrees = aug_cfg.get("rotation_degrees", [0, 90, 180, 270])
        transforms_list.append(RandomRotation90(angles=degrees))

    # 6. Color Jitter (RGB only)
    cj_cfg = aug_cfg.get("color_jitter", {})
    if cj_cfg.get("enabled", True):
        transforms_list.append(
            ColorJitter(
                brightness=cj_cfg.get("brightness", 0.3),
                contrast=cj_cfg.get("contrast", 0.3),
                saturation=cj_cfg.get("saturation", 0.3),
                hue=cj_cfg.get("hue", 0.1),
            )
        )

    # 7. Normalize
    norm_cfg = aug_cfg.get("normalize", {})
    rgb_mean = norm_cfg.get("rgb_mean", [0.485, 0.456, 0.406])
    rgb_std = norm_cfg.get("rgb_std", [0.229, 0.224, 0.225])
    nir_mean = norm_cfg.get("nir_mean", [0.5])
    nir_std = norm_cfg.get("nir_std", [0.25])
    transforms_list.append(
        Normalize(
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            nir_mean=nir_mean,
            nir_std=nir_std,
        )
    )

    # 8. ToTensor
    transforms_list.append(ToTensor())

    return Compose(transforms_list)


def get_val_transforms(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
) -> Compose:
    """
    Builds the evaluation/validation transformation pipeline (Normalize + ToTensor only).

    Args:
        config: Loaded config dictionary, path to config.yaml, or None for defaults.

    Returns:
        Compose transform object for validation and testing.
    """
    cfg = _load_config(config)
    aug_cfg = cfg.get("augmentation", {})

    norm_cfg = aug_cfg.get("normalize", {})
    rgb_mean = norm_cfg.get("rgb_mean", [0.485, 0.456, 0.406])
    rgb_std = norm_cfg.get("rgb_std", [0.229, 0.224, 0.225])
    nir_mean = norm_cfg.get("nir_mean", [0.5])
    nir_std = norm_cfg.get("nir_std", [0.25])

    return Compose(
        [
            Normalize(
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                nir_mean=nir_mean,
                nir_std=nir_std,
            ),
            ToTensor(),
        ]
    )
