"""
Evaluation Script for ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network
for RGB-Multispectral Semantic Segmentation.

Features:
- Checkpoint loading with architecture reconstruction
- Sliding window inference for large aerial tiles with smooth window blending
- Test-time augmentation (TTA) with multi-axis flips
- Comprehensive quantitative evaluation: mIoU, OA, Macro F1, Per-class IoU/F1
- Formatted summary table reporting
- Generation and saving of:
  * Colored prediction segmentation maps
  * Side-by-side qualitative comparisons (RGB | GT | Pred | Error)
  * Normalized confusion matrix heatmap
  * Per-class IoU bar chart
  * JSON metrics file
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

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project module imports
from models.asg_net import ASGNet
from data.potsdam_dataset import PotsdamDataset
from data.transforms import get_val_transforms
from losses.combined_loss import CombinedLoss
from utils.metrics import StreamingConfusionMatrix
from utils.visualize import *
from utils.lr_scheduler import PolyLRScheduler


def setup_logger(output_dir: str) -> logging.Logger:
    """Configures logger for evaluation run."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("ASGNet_Eval")
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

    fh = logging.FileHandler(os.path.join(output_dir, "evaluate.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration file from YAML path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ASG-Net on ISPRS Potsdam benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Whether to save colored prediction maps and comparison figures"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results and figures"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Window size for sliding window inference"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride for sliding window inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for patch-based dataloader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda', 'cpu')"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation (multi-scale / multi-flip)"
    )
    parser.add_argument(
        "--exclude_building",
        action="store_true",
        help="Exclude Building class from evaluation metrics (mIoU and OA)"
    )
    return parser.parse_args()


def build_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device, logger: logging.Logger) -> nn.Module:
    """Instantiates ASGNet and loads weights from checkpoint."""
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

    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove unexpected 'module.' prefix if trained with DistributedDataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully into evaluation mode.")
    return model


def create_2d_window_weight(window_size: int, device: torch.device) -> torch.Tensor:
    """Creates a 2D Hann smoothing window to blend overlapping inference patches."""
    hanning = torch.hann_window(window_size, periodic=False, device=device)
    window_2d = torch.outer(hanning, hanning)
    # Clip lower values to prevent zero division
    window_2d = torch.clamp(window_2d, min=1e-4)
    return window_2d.unsqueeze(0)  # Shape: (1, window_size, window_size)


def predict_window_with_tta(
    model: nn.Module,
    rgb_win: torch.Tensor,
    nir_win: torch.Tensor,
    use_tta: bool = False
) -> torch.Tensor:
    """Runs model forward pass with optional flip test-time augmentation.
    
    Args:
        model: ASGNet model
        rgb_win: (B, 3, H, W)
        nir_win: (B, 1, H, W)
        use_tta: whether to average predictions across flips
        
    Returns:
        Predicted class probability tensor: (B, C, H, W)
    """
    def _forward(r: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        out = model(r, n)
        if isinstance(out, dict):
            logits = out["seg_pred"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.shape[-2:] != r.shape[-2:]:
            logits = F.interpolate(logits, size=r.shape[-2:], mode="bilinear", align_corners=False)
        return F.softmax(logits, dim=1)

    prob = _forward(rgb_win, nir_win)

    if use_tta:
        # Horizontal flip
        r_h = torch.flip(rgb_win, dims=[-1])
        n_h = torch.flip(nir_win, dims=[-1])
        prob_h = torch.flip(_forward(r_h, n_h), dims=[-1])

        # Vertical flip
        r_v = torch.flip(rgb_win, dims=[-2])
        n_v = torch.flip(nir_win, dims=[-2])
        prob_v = torch.flip(_forward(r_v, n_v), dims=[-2])

        # Horizontal + Vertical flip
        r_hv = torch.flip(rgb_win, dims=[-2, -1])
        n_hv = torch.flip(nir_win, dims=[-2, -1])
        prob_hv = torch.flip(_forward(r_hv, n_hv), dims=[-2, -1])

        prob = (prob + prob_h + prob_v + prob_hv) / 4.0

    return prob


def sliding_window_inference(
    model: nn.Module,
    rgb: torch.Tensor,
    nir: torch.Tensor,
    num_classes: int = 6,
    window_size: int = 512,
    stride: int = 256,
    use_tta: bool = False,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Performs sliding window inference with Hann window blending for large images.
    
    Args:
        model: Evaluated segmentation model
        rgb: (3, H, W)
        nir: (1, H, W)
        num_classes: Number of semantic classes
        window_size: Spatial size of square window
        stride: Stride step between windows
        use_tta: Test-time augmentation
        device: Computation device
        
    Returns:
        prob_map: Class probability tensor of shape (num_classes, H, W)
    """
    _, h, w = rgb.shape

    # Handle smaller inputs with direct padding
    if h <= window_size and w <= window_size:
        pad_h = max(0, window_size - h)
        pad_w = max(0, window_size - w)
        rgb_padded = F.pad(rgb.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")
        nir_padded = F.pad(nir.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")
        with torch.no_grad():
            prob = predict_window_with_tta(model, rgb_padded.to(device), nir_padded.to(device), use_tta=use_tta)
        return prob[0, :, :h, :w]

    # Calculate steps
    y_steps = list(range(0, h - window_size + 1, stride))
    if y_steps[-1] + window_size < h:
        y_steps.append(h - window_size)

    x_steps = list(range(0, w - window_size + 1, stride))
    if x_steps[-1] + window_size < w:
        x_steps.append(w - window_size)

    blend_weight = create_2d_window_weight(window_size, device=device)  # (1, ws, ws)

    prob_accum = torch.zeros((num_classes, h, w), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                rgb_crop = rgb[:, y:y + window_size, x:x + window_size].unsqueeze(0).to(device)
                nir_crop = nir[:, y:y + window_size, x:x + window_size].unsqueeze(0).to(device)

                prob_crop = predict_window_with_tta(model, rgb_crop, nir_crop, use_tta=use_tta)[0]  # (C, ws, ws)

                weighted_prob = prob_crop * blend_weight
                prob_accum[:, y:y + window_size, x:x + window_size] += weighted_prob
                weight_accum[:, y:y + window_size, x:x + window_size] += blend_weight

    prob_map = prob_accum / torch.clamp(weight_accum, min=1e-6)
    return prob_map


def colorize_mask(mask: np.ndarray, class_colors: List[List[int]]) -> np.ndarray:
    """Converts 2D class index mask to 3-channel RGB image.
    
    Args:
        mask: 2D array of class indices (H, W)
        class_colors: List of [R, G, B] per class
        
    Returns:
        RGB image of shape (H, W, 3) as uint8
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(class_colors):
        colored[mask == idx] = color
    return colored


def save_qualitative_results(
    save_path: str,
    rgb_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_colors: List[List[int]],
    class_names: List[str],
    norm_cfg: Dict[str, Any]
) -> None:
    """Saves a 4-panel comparison: RGB, Ground Truth, Prediction, and Error Map."""
    # Unnormalize RGB
    mean = np.array(norm_cfg.get("rgb_mean", [0.485, 0.456, 0.406])).reshape(3, 1, 1)
    std = np.array(norm_cfg.get("rgb_std", [0.229, 0.224, 0.225])).reshape(3, 1, 1)
    rgb_np = rgb_tensor.cpu().numpy()
    rgb_disp = (rgb_np * std + mean).clip(0, 1)
    rgb_disp = np.transpose(rgb_disp, (1, 2, 0))

    gt_colored = colorize_mask(gt_mask, class_colors)
    pred_colored = colorize_mask(pred_mask, class_colors)

    # Error map: green for correct, red for incorrect
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    correct = (gt_mask == pred_mask)
    error_map[correct] = [40, 180, 40]    # Green
    error_map[~correct] = [220, 40, 40]   # Red

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=200)
    axes[0].imshow(rgb_disp)
    axes[0].set_title("Input RGB", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(gt_colored)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(pred_colored)
    axes[2].set_title("ASG-Net Prediction", fontsize=12)
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("Classification Error Map", fontsize=12)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(
    cm_matrix: np.ndarray,
    class_names: List[str],
    save_path: str
) -> None:
    """Generates and saves a normalized confusion matrix heatmap."""
    try:
        # Check if utils.visualize has a specific plot_confusion_matrix function
        if "plot_confusion_matrix" in globals() and callable(globals()["plot_confusion_matrix"]):
            try:
                plot_confusion_matrix(cm_matrix, class_names, save_path=save_path)
                return
            except Exception:
                pass
    except Exception:
        pass

    # Matplotlib / Seaborn publication-quality fallback
    row_sums = cm_matrix.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm_matrix, row_sums, out=np.zeros_like(cm_matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(norm_cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Accuracy", rotation=270, labelpad=15, fontsize=11)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Text annotations inside matrix cells
    thresh = norm_cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = norm_cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)
    ax.set_title("Normalized Confusion Matrix - ASG-Net", fontsize=14, pad=15)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_per_class_iou_chart(
    per_class_iou: np.ndarray,
    class_names: List[str],
    save_path: str
) -> None:
    """Generates and saves a bar chart of per-class IoU scores."""
    try:
        if "plot_per_class_iou" in globals() and callable(globals()["plot_per_class_iou"]):
            try:
                plot_per_class_iou(per_class_iou, class_names, save_path=save_path)
                return
            except Exception:
                pass
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    scores = per_class_iou * 100
    bars = ax.bar(class_names, scores, color="#2b5c8f", edgecolor="#1a3b5c", alpha=0.85, width=0.55)

    # Mean IoU line
    miou_score = float(np.mean(scores))
    ax.axhline(miou_score, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean IoU: {miou_score:.2f}%")

    # Add numeric labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Intersection over Union (IoU %)", fontsize=12)
    ax.set_title("Per-Class Segmentation Performance (IoU)", fontsize=14, pad=12)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=20, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def format_results_table(
    class_names: List[str],
    per_class_iou: Any,
    per_class_f1: Any,
    miou: float,
    oa: float,
    macro_f1: float,
    miou_no_building: Optional[float] = None,
    oa_no_building: Optional[float] = None,
    exclude_building: bool = False
) -> str:
    """Constructs a clean ASCII / Markdown summary table of evaluation metrics."""
    lines = []
    lines.append("=" * 64)
    lines.append(f"{'ASG-NET QUANTITATIVE EVALUATION RESULTS':^64}")
    lines.append("=" * 64)
    lines.append(f"| {'ID':<3} | {'Class Name':<22} | {'IoU (%)':>12} | {'F1 (%)':>12} |")
    lines.append("|" + "-" * 5 + "|" + "-" * 24 + "|" + "-" * 14 + "|" + "-" * 14 + "|")

    for idx, name in enumerate(class_names):
        is_bldg = (name.lower() == "building")
        display_name = f"{name} *" if (is_bldg and exclude_building) else name
        iou_val = per_class_iou.get(name, per_class_iou.get(idx, 0.0)) * 100
        f1_val = per_class_f1.get(name, per_class_f1.get(idx, 0.0)) * 100
        lines.append(f"| {idx:<3} | {display_name:<22} | {iou_val:>11.2f}% | {f1_val:>11.2f}% |")

    lines.append("=" * 64)
    if exclude_building and miou_no_building is not None:
        lines.append(f"  * [Building excluded from evaluation metrics]")
        lines.append(f"  Mean IoU (Excl. Building):    {miou_no_building * 100:6.2f}%")
        if oa_no_building is not None:
            lines.append(f"  Overall Accuracy (Excl. Bldg):{oa_no_building * 100:6.2f}%")
        lines.append(f"  Mean IoU (All Classes):       {miou * 100:6.2f}%")
        lines.append(f"  Overall Accuracy (All):       {oa * 100:6.2f}%")
    else:
        lines.append(f"  Mean IoU (mIoU):       {miou * 100:6.2f}%")
        lines.append(f"  Overall Accuracy (OA): {oa * 100:6.2f}%")
        lines.append(f"  Macro F1-Score:        {macro_f1 * 100:6.2f}%")
    lines.append("=" * 64)
    return "\n".join(lines)


def evaluate() -> None:
    """Main evaluation pipeline."""
    args = parse_args()
    config = load_config(args.config)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    pred_dir = os.path.join(output_dir, "predictions")
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("=" * 60)
    logger.info("ASG-Net: Model Evaluation on Potsdam Benchmark")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Target Split:  {args.split}")
    logger.info(f"Checkpoint:    {args.checkpoint}")
    logger.info(f"Sliding Window Size: {args.window_size} | Stride: {args.stride}")
    logger.info(f"Test-Time Augmentation (TTA): {args.tta}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Evaluation Device: {device}")

    # Build model
    model = build_model(config, args.checkpoint, device, logger)

    # Build Dataset
    logger.info(f"Loading {args.split} dataset...")
    val_transforms = get_val_transforms(config)
    ds_root = config.get("dataset", {}).get("root", "data/potsdam")
    patch_size = args.window_size
    stride = args.stride

    try:
        dataset = PotsdamDataset(
            root=ds_root,
            split=args.split,
            patch_size=patch_size,
            stride=stride,
            transforms=val_transforms
        )
    except TypeError:
        dataset = PotsdamDataset(root=ds_root, split=args.split, transforms=val_transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("training", {}).get("num_workers", 4),
        pin_memory=(device.type == "cuda")
    )

    num_classes = config.get("dataset", {}).get("num_classes", 6)
    class_names = config.get("dataset", {}).get("class_names", [
        "Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"
    ])
    class_colors = config.get("dataset", {}).get("class_colors", [
        [255, 255, 255], [0, 0, 255], [0, 255, 255],
        [0, 255, 0], [255, 255, 0], [255, 0, 0]
    ])
    norm_cfg = config.get("augmentation", {}).get("normalize", {})

    cm = StreamingConfusionMatrix(num_classes=num_classes, class_names=class_names)
    saved_count = 0
    max_to_save = 25  # Maximum number of detailed qualitative figures to save

    logger.info(f"Running inference across {len(dataset)} samples...")

    pbar = tqdm(dataloader, desc=f"Evaluating ({args.split})", dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                rgb = batch["rgb"]
                nir = batch["nir"]
                mask = batch["mask"]
            else:
                rgb = batch[0]
                nir = batch[1]
                mask = batch[2]

            batch_size = rgb.size(0)

            for b in range(batch_size):
                sample_rgb = rgb[b]   # (3, H, W)
                sample_nir = nir[b]   # (1, H, W)
                sample_gt = mask[b]   # (H, W) or (1, H, W)

                if sample_gt.dim() == 3:
                    sample_gt = sample_gt.squeeze(0)

                # Check if image is larger than window_size
                _, h, w = sample_rgb.shape
                if h > args.window_size or w > args.window_size:
                    # Run full sliding window
                    prob_map = sliding_window_inference(
                        model=model,
                        rgb=sample_rgb,
                        nir=sample_nir,
                        num_classes=num_classes,
                        window_size=args.window_size,
                        stride=args.stride,
                        use_tta=args.tta,
                        device=device
                    )
                    pred_mask = prob_map.argmax(dim=0).cpu().numpy()
                else:
                    # Direct window forward pass
                    rgb_in = sample_rgb.unsqueeze(0).to(device)
                    nir_in = sample_nir.unsqueeze(0).to(device)
                    prob_win = predict_window_with_tta(model, rgb_in, nir_in, use_tta=args.tta)
                    pred_mask = prob_win[0].argmax(dim=0).cpu().numpy()

                gt_np = sample_gt.cpu().numpy()

                # Update confusion matrix
                cm.update(pred_mask, gt_np)

                # Save predictions if requested
                if args.save_predictions and saved_count < max_to_save:
                    sample_name = f"sample_{batch_idx:04d}_{b:02d}"

                    # 1. Save colored prediction map
                    pred_color = colorize_mask(pred_mask, class_colors)
                    Image.fromarray(pred_color).save(os.path.join(pred_dir, f"{sample_name}_pred.png"))

                    # 2. Save 4-panel comparison
                    save_qualitative_results(
                        save_path=os.path.join(pred_dir, f"{sample_name}_comparison.png"),
                        rgb_tensor=sample_rgb,
                        gt_mask=gt_np,
                        pred_mask=pred_mask,
                        class_colors=class_colors,
                        class_names=class_names,
                        norm_cfg=norm_cfg
                    )
                    saved_count += 1

    # Compute final metrics
    miou = cm.compute_miou()
    oa = cm.compute_overall_accuracy()
    f1 = cm.compute_f1()
    per_class_iou = cm.compute_per_class_iou()

    # Compute metrics excluding Building
    miou_no_bldg = cm.compute_miou(exclude_classes=["Building"])
    oa_no_bldg = cm.compute_overall_accuracy(exclude_classes=["Building"])
    f1_no_bldg = cm.compute_f1(exclude_classes=["Building"])

    # Compute per-class F1 if available on cm, else compute from confusion matrix
    if hasattr(cm, "compute_per_class_f1"):
        per_class_f1 = cm.compute_per_class_f1()
    else:
        # Calculate per-class F1: 2 * TP / (2*TP + FP + FN)
        mat = cm.matrix if hasattr(cm, "matrix") else cm.confusion_matrix
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        per_class_f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)

    # Print Table
    table_str = format_results_table(
        class_names=class_names,
        per_class_iou=per_class_iou,
        per_class_f1=per_class_f1,
        miou=miou,
        oa=oa,
        macro_f1=f1,
        miou_no_building=miou_no_bldg,
        oa_no_building=oa_no_bldg,
        exclude_building=args.exclude_building
    )
    print("\n" + table_str)
    logger.info("\n" + table_str)

    # Save metrics to JSON
    metrics_dict = {
        "mIoU": float(miou_no_bldg if args.exclude_building else miou),
        "Overall_Accuracy": float(oa_no_bldg if args.exclude_building else oa),
        "Macro_F1": float(f1_no_bldg if args.exclude_building else f1),
        "mIoU_all_classes": float(miou),
        "Overall_Accuracy_all_classes": float(oa),
        "mIoU_excluding_building": float(miou_no_bldg),
        "Overall_Accuracy_excluding_building": float(oa_no_bldg),
        "Macro_F1_excluding_building": float(f1_no_bldg),
        "exclude_building": args.exclude_building,
        "per_class_iou": {name: float(per_class_iou.get(name, per_class_iou.get(i, 0.0))) for i, name in enumerate(class_names)},
        "per_class_f1": {name: float(per_class_f1.get(name, per_class_f1.get(i, 0.0))) for i, name in enumerate(class_names)},
        "split": args.split,
        "checkpoint": args.checkpoint,
        "tta": args.tta
    }
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Evaluation metrics saved to JSON: {json_path}")

    # Generate Confusion Matrix Figure
    cm_matrix = cm.matrix if hasattr(cm, "matrix") else cm.confusion_matrix
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(cm_matrix, class_names, cm_plot_path)
    logger.info(f"Confusion matrix plot saved: {cm_plot_path}")

    # Generate Per-Class IoU Bar Chart
    chart_plot_path = os.path.join(output_dir, "per_class_iou.png")
    save_per_class_iou_chart(per_class_iou, class_names, chart_plot_path)
    logger.info(f"Per-class IoU bar chart saved: {chart_plot_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
