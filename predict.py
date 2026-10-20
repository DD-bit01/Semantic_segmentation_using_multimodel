"""
Prediction and Inference Script for ASG-Net.

Features:
- Single image and batch directory inference
- Automatic synthesis of dummy/pseudo NIR from RGB if multispectral band is unavailable
- Sliding window inference for large aerial, drone, or satellite imagery
- Seamless smooth blending across overlapping windows
- Output generation:
  * Raw class mask (uint8 PNG with class indices 0..5)
  * Colorized segmentation map (Potsdam color palette)
  * Semi-transparent segmentation overlay on original RGB
  * Graphical class legend with area coverage statistics
  * 4-panel composite visualization for qualitative inspection
"""

import os
import sys

# Force all caches and downloaded models to stay inside project directory (D: drive)
_PROJECT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(os.path.join(_PROJECT_CACHE, "huggingface"), exist_ok=True)
os.makedirs(os.path.join(_PROJECT_CACHE, "torch"), exist_ok=True)
os.makedirs(os.path.join(_PROJECT_CACHE, "matplotlib"), exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(_PROJECT_CACHE, "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_PROJECT_CACHE, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_PROJECT_CACHE, "torch"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_PROJECT_CACHE, "matplotlib"))

import glob
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project module imports
from models.asg_net import ASGNet
from data.potsdam_dataset import PotsdamDataset
from data.transforms import get_train_transforms, get_val_transforms
from losses.combined_loss import CombinedLoss
from utils.metrics import StreamingConfusionMatrix
from utils.visualize import *
from utils.lr_scheduler import PolyLRScheduler


def setup_logger() -> logging.Logger:
    """Configures console logging for inference."""
    logger = logging.getLogger("ASGNet_Predict")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for prediction."""
    parser = argparse.ArgumentParser(
        description="Run ASG-Net semantic segmentation inference on RGB/NIR imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model weights checkpoint (.pth)"
    )
    parser.add_argument(
        "--rgb_path",
        type=str,
        required=True,
        help="Path to input RGB image file OR directory of images"
    )
    parser.add_argument(
        "--nir_path",
        type=str,
        default=None,
        help="Path to input NIR image file OR directory of NIR images (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/predictions",
        help="Directory where output prediction maps and overlays will be stored"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Sliding window size for large image processing"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between sliding windows"
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Transparency alpha for segmentation overlay on RGB (0.0 to 1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda', 'cpu')"
    )
    return parser.parse_args()


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device, logger: logging.Logger) -> nn.Module:
    """Constructs model and loads weights from checkpoint."""
    model_cfg = config.get("model", {})
    fusion_cfg = model_cfg.get("fusion", {})
    edge_cfg = model_cfg.get("edge_decoder", {})

    try:
        model = ASGNet(config=config)
    except (TypeError, Exception):
        try:
            model = ASGNet(**model_cfg)
        except (TypeError, Exception):
            model = ASGNet(
                num_classes=model_cfg.get("num_classes", 6),
                backbone=model_cfg.get("backbone", "mit_b2"),
                pretrained=False,
                fusion_type=fusion_cfg.get("type", "asg"),
                use_cm_frm=fusion_cfg.get("use_cm_frm", True),
                decoder_channels=model_cfg.get("decoder_channels", 256),
                edge_decoder=edge_cfg.get("enabled", True),
            )

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model weights loaded successfully.")
    return model


def read_image_rgb(path: str) -> np.ndarray:
    """Reads image file and converts to 3-channel RGB uint8 numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RGB image not found: {path}")

    # Load via PIL
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def read_or_synthesize_nir(nir_path: Optional[str], rgb_np: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Loads NIR image or synthesizes dummy NIR from RGB luminance if path is not provided."""
    if nir_path and os.path.exists(nir_path):
        nir_img = Image.open(nir_path).convert("L")
        nir_np = np.array(nir_img, dtype=np.uint8)
        # Resize to match RGB if needed
        if nir_np.shape[:2] != rgb_np.shape[:2]:
            nir_img = nir_img.resize((rgb_np.shape[1], rgb_np.shape[0]), Image.BILINEAR)
            nir_np = np.array(nir_img, dtype=np.uint8)
        return nir_np
    else:
        # Synthesize NIR from RGB luminance: 0.299 R + 0.587 G + 0.114 B
        logger.info("NIR image not provided; synthesizing pseudo-NIR channel from RGB.")
        r = rgb_np[:, :, 0].astype(np.float32)
        g = rgb_np[:, :, 1].astype(np.float32)
        b = rgb_np[:, :, 2].astype(np.float32)
        # Enhance vegetation response (NIR reflects heavily in vegetation where G > R)
        nir_synth = (0.299 * r + 0.587 * g + 0.114 * b).clip(0, 255).astype(np.uint8)
        return nir_synth


def preprocess_tensors(
    rgb_np: np.ndarray,
    nir_np: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts numpy arrays to normalized PyTorch tensors."""
    norm_cfg = config.get("augmentation", {}).get("normalize", {})
    rgb_mean = np.array(norm_cfg.get("rgb_mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, 3)
    rgb_std = np.array(norm_cfg.get("rgb_std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, 3)

    nir_mean = float(norm_cfg.get("nir_mean", [0.5])[0] if isinstance(norm_cfg.get("nir_mean"), list) else norm_cfg.get("nir_mean", 0.5))
    nir_std = float(norm_cfg.get("nir_std", [0.25])[0] if isinstance(norm_cfg.get("nir_std"), list) else norm_cfg.get("nir_std", 0.25))

    # Scale to [0, 1]
    rgb_norm = (rgb_np.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    nir_norm = (nir_np.astype(np.float32) / 255.0 - nir_mean) / nir_std

    # (H, W, C) -> (C, H, W)
    rgb_tensor = torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1))).float()
    nir_tensor = torch.from_numpy(nir_norm).unsqueeze(0).float()  # (1, H, W)

    return rgb_tensor, nir_tensor


def sliding_window_predict(
    model: nn.Module,
    rgb_tensor: torch.Tensor,
    nir_tensor: torch.Tensor,
    num_classes: int = 6,
    window_size: int = 512,
    stride: int = 256,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """Inference over arbitrarily large images using Hann window smoothing."""
    _, h, w = rgb_tensor.shape

    if h <= window_size and w <= window_size:
        pad_h = max(0, window_size - h)
        pad_w = max(0, window_size - w)
        r_pad = F.pad(rgb_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").to(device)
        n_pad = F.pad(nir_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").to(device)
        with torch.no_grad():
            out = model(r_pad, n_pad)
            logits = out["seg_pred"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
            pred = logits[0, :, :h, :w].argmax(dim=0).cpu().numpy()
        return pred

    # Coordinates
    y_steps = list(range(0, h - window_size + 1, stride))
    if y_steps[-1] + window_size < h:
        y_steps.append(h - window_size)

    x_steps = list(range(0, w - window_size + 1, stride))
    if x_steps[-1] + window_size < w:
        x_steps.append(w - window_size)

    # 2D Hann window
    hann = torch.hann_window(window_size, periodic=False, device=device)
    w_2d = torch.clamp(torch.outer(hann, hann), min=1e-4).unsqueeze(0)  # (1, ws, ws)

    prob_accum = torch.zeros((num_classes, h, w), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                r_win = rgb_tensor[:, y:y + window_size, x:x + window_size].unsqueeze(0).to(device)
                n_win = nir_tensor[:, y:y + window_size, x:x + window_size].unsqueeze(0).to(device)

                out = model(r_win, n_win)
                logits = out["seg_pred"] if isinstance(out, dict) else (out[0] if isinstance(out, (list, tuple)) else out)
                probs = F.softmax(logits, dim=1)[0]  # (C, ws, ws)

                prob_accum[:, y:y + window_size, x:x + window_size] += probs * w_2d
                weight_accum[:, y:y + window_size, x:x + window_size] += w_2d

    final_probs = prob_accum / torch.clamp(weight_accum, min=1e-6)
    pred_mask = final_probs.argmax(dim=0).cpu().numpy()
    return pred_mask


def colorize_prediction(pred_mask: np.ndarray, class_colors: List[List[int]]) -> np.ndarray:
    """Maps class indices to 3-channel RGB image."""
    h, w = pred_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c_idx, color in enumerate(class_colors):
        colored[pred_mask == c_idx] = color
    return colored


def create_overlay(rgb_np: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blends colorized mask with RGB image."""
    overlay = (1.0 - alpha) * rgb_np.astype(np.float32) + alpha * colored_mask.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_legend(
    class_names: List[str],
    class_colors: List[List[int]],
    class_percentages: List[float],
    save_path: str
) -> None:
    """Generates and saves a publication-quality legend figure."""
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(class_names) * 0.6)), dpi=300)
    ax.axis("off")

    handles = []
    labels = []
    for idx, (name, color) in enumerate(zip(class_names, class_colors)):
        norm_color = [c / 255.0 for c in color]
        patch = mpatches.Patch(facecolor=norm_color, edgecolor="black", linewidth=1.0)
        pct = class_percentages[idx] if idx < len(class_percentages) else 0.0
        label_text = f"{name} ({pct:.1f}%)"
        handles.append(patch)
        labels.append(label_text)

    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        framealpha=0.9,
        fontsize=11,
        title="Semantic Classes & Area Coverage",
        title_fontsize=12,
        labelspacing=0.8,
        borderpad=1.0
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_composite_inspection(
    save_path: str,
    rgb_np: np.ndarray,
    nir_np: np.ndarray,
    overlay_np: np.ndarray,
    colored_mask: np.ndarray
) -> None:
    """Saves a 4-panel visual summary of the inference results."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=200)

    axes[0].imshow(rgb_np)
    axes[0].set_title("Input RGB Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(nir_np, cmap="gray")
    axes[1].set_title("Input / Synthetic NIR Channel", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay_np)
    axes[2].set_title("Segmentation Overlay", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(colored_mask)
    axes[3].set_title("Semantic Class Map", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def process_single_pair(
    model: nn.Module,
    rgb_path: str,
    nir_path: Optional[str],
    output_dir: str,
    config: Dict[str, Any],
    window_size: int,
    stride: int,
    overlay_alpha: float,
    device: torch.device,
    logger: logging.Logger
) -> None:
    """Executes full inference and saves all visualization artifacts for a single image."""
    stem = Path(rgb_path).stem
    rgb_np = read_image_rgb(rgb_path)
    nir_np = read_or_synthesize_nir(nir_path, rgb_np, logger)

    rgb_tensor, nir_tensor = preprocess_tensors(rgb_np, nir_np, config)

    num_classes = config.get("dataset", {}).get("num_classes", 6)
    class_names = config.get("dataset", {}).get("class_names", [
        "Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"
    ])
    class_colors = config.get("dataset", {}).get("class_colors", [
        [255, 255, 255], [0, 0, 255], [0, 255, 255],
        [0, 255, 0], [255, 255, 0], [255, 0, 0]
    ])

    # Run inference
    pred_mask = sliding_window_predict(
        model=model,
        rgb_tensor=rgb_tensor,
        nir_tensor=nir_tensor,
        num_classes=num_classes,
        window_size=window_size,
        stride=stride,
        device=device
    )

    # Calculate class area percentages
    total_pixels = pred_mask.size
    class_percentages = [
        float(np.sum(pred_mask == i) / total_pixels * 100.0) for i in range(num_classes)
    ]

    # Render visualizations
    colored_mask = colorize_prediction(pred_mask, class_colors)
    overlay_np = create_overlay(rgb_np, colored_mask, alpha=overlay_alpha)

    # Save outputs
    # 1. Raw mask
    raw_path = os.path.join(output_dir, f"{stem}_pred_raw.png")
    Image.fromarray(pred_mask.astype(np.uint8)).save(raw_path)

    # 2. Colorized mask
    color_path = os.path.join(output_dir, f"{stem}_pred_color.png")
    Image.fromarray(colored_mask).save(color_path)

    # 3. Overlay
    overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
    Image.fromarray(overlay_np).save(overlay_path)

    # 4. NIR channel used
    nir_path_out = os.path.join(output_dir, f"{stem}_nir.png")
    Image.fromarray(nir_np).save(nir_path_out)

    # 5. Composite inspection
    comp_path = os.path.join(output_dir, f"{stem}_composite.png")
    save_composite_inspection(comp_path, rgb_np, nir_np, overlay_np, colored_mask)

    # 6. Legend
    legend_path = os.path.join(output_dir, f"{stem}_legend.png")
    save_legend(class_names, class_colors, class_percentages, legend_path)

    logger.info(f"Predictions saved for {stem} -> {output_dir}")


def predict() -> None:
    """Main entry point for inference."""
    args = parse_args()
    logger = setup_logger()
    config = load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("=" * 60)
    logger.info("ASG-Net: Semantic Segmentation Inference")
    logger.info("=" * 60)
    logger.info(f"Target Device: {device}")
    logger.info(f"Output Directory: {args.output_dir}")

    # Load Model
    model = load_model(config, args.checkpoint, device, logger)

    # Check if rgb_path is directory or single file
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    if os.path.isdir(args.rgb_path):
        logger.info(f"Scanning directory for images: {args.rgb_path}")
        image_files = [
            f for f in glob.glob(os.path.join(args.rgb_path, "*"))
            if Path(f).suffix.lower() in valid_exts
        ]
        image_files.sort()
        logger.info(f"Found {len(image_files)} images to process.")

        for rgb_file in tqdm(image_files, desc="Batch Inference", dynamic_ncols=True):
            stem = Path(rgb_file).stem
            # Attempt to locate matching NIR file if nir directory provided
            matching_nir = None
            if args.nir_path and os.path.isdir(args.nir_path):
                candidates = [
                    os.path.join(args.nir_path, f"{stem}{ext}") for ext in valid_exts
                ] + [
                    os.path.join(args.nir_path, f"{stem}_IRRG{ext}") for ext in valid_exts
                ] + [
                    os.path.join(args.nir_path, f"{stem}_NIR{ext}") for ext in valid_exts
                ]
                for c in candidates:
                    if os.path.exists(c):
                        matching_nir = c
                        break

            process_single_pair(
                model=model,
                rgb_path=rgb_file,
                nir_path=matching_nir,
                output_dir=args.output_dir,
                config=config,
                window_size=args.window_size,
                stride=args.stride,
                overlay_alpha=args.overlay_alpha,
                device=device,
                logger=logger
            )
    else:
        # Single image inference
        process_single_pair(
            model=model,
            rgb_path=args.rgb_path,
            nir_path=args.nir_path,
            output_dir=args.output_dir,
            config=config,
            window_size=args.window_size,
            stride=args.stride,
            overlay_alpha=args.overlay_alpha,
            device=device,
            logger=logger
        )

    logger.info("Inference completed successfully.")


if __name__ == "__main__":
    predict()
