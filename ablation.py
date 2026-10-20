"""
Ablation Study Runner for ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network
for RGB-Multispectral Semantic Segmentation.

Systematically trains and benchmarks 5 model variants to isolate and quantify the
performance contribution of each architectural innovation:
1. 'rgb_only': Single-encoder baseline utilizing RGB channels only (no MS data).
2. 'early_fusion': Single encoder with 4-channel input (RGB + NIR concatenated at input).
3. 'dual_add': Dual-encoder with simple element-wise addition fusion.
4. 'dual_asg_no_cmfrm': Dual-encoder with Adaptive Spectral-Spatial Gated (ASG) fusion,
   without Cross-Modal Feature Rectification (CM-FRM).
5. 'full_asg_net': Full proposed model with Dual MiT-B2, CM-FRM, ASG Blocks, and Edge Loss.

Outputs:
- Comprehensive ASCII / Markdown comparison tables
- Publication-quality comparative bar charts (mIoU, OA, F1, Parameters)
- Per-class performance progression charts
- Machine-readable JSON summary results file
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

import copy
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def setup_logger(output_dir: str) -> logging.Logger:
    """Sets up console and file logging for the ablation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("ASGNet_Ablation")
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

    fh = logging.FileHandler(os.path.join(output_dir, "ablation_study.log"), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for running ablation experiments."""
    parser = argparse.ArgumentParser(
        description="Run ASG-Net Ablation Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs per variant (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per variant (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation",
        help="Directory to save ablation logs, checkpoints, and comparative plots"
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 epoch and 10 batches per variant for rapid debugging"
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated list of variants to run, or 'all'. Options: rgb_only,early_fusion,dual_add,dual_asg_no_cmfrm,full_asg_net"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train and evaluate on ('cuda', 'cpu')"
    )
    return parser.parse_args()


# Specification of the 5 canonical ablation variants
VARIANTS_SPECS: Dict[str, Dict[str, Any]] = {
    "rgb_only": {
        "display_name": "V1: RGB-Only (Single Encoder)",
        "description": "Baseline single MiT-B2 encoder with RGB input only (no multispectral band)",
        "fusion_type": "none",
        "use_cm_frm": False,
        "edge_enabled": False,
        "in_channels_rgb": 3,
        "in_channels_nir": 0,
        "mode": "rgb_only"
    },
    "early_fusion": {
        "display_name": "V2: Early Fusion (4-channel)",
        "description": "Single MiT-B2 encoder receiving 4-channel input (RGB + NIR concatenated)",
        "fusion_type": "none",
        "use_cm_frm": False,
        "edge_enabled": False,
        "in_channels_rgb": 4,
        "in_channels_nir": 0,
        "mode": "early_fusion"
    },
    "dual_add": {
        "display_name": "V3: Dual-Encoder + Addition",
        "description": "Dual MiT-B2 encoders with element-wise addition fusion at each stage",
        "fusion_type": "add",
        "use_cm_frm": False,
        "edge_enabled": False,
        "in_channels_rgb": 3,
        "in_channels_nir": 1,
        "mode": "dual"
    },
    "dual_asg_no_cmfrm": {
        "display_name": "V4: Dual-Encoder + ASG (No Rectification)",
        "description": "Dual MiT-B2 with Adaptive Gated Fusion without Cross-Modal Feature Rectification",
        "fusion_type": "asg",
        "use_cm_frm": False,
        "edge_enabled": False,
        "in_channels_rgb": 3,
        "in_channels_nir": 1,
        "mode": "dual"
    },
    "full_asg_net": {
        "display_name": "V5: Full ASG-Net (Proposed)",
        "description": "Full proposed architecture: Dual MiT-B2 + CM-FRM + ASG Fusion + Edge Supervision",
        "fusion_type": "asg",
        "use_cm_frm": True,
        "edge_enabled": True,
        "in_channels_rgb": 3,
        "in_channels_nir": 1,
        "mode": "dual"
    }
}


def configure_variant(base_config: Dict[str, Any], variant_key: str) -> Dict[str, Any]:
    """Clones base config and overrides parameters for a specific ablation variant."""
    cfg = copy.deepcopy(base_config)
    spec = VARIANTS_SPECS[variant_key]

    cfg["model"]["fusion"]["type"] = spec["fusion_type"]
    cfg["model"]["fusion"]["use_cm_frm"] = spec["use_cm_frm"]
    cfg["model"]["edge_decoder"]["enabled"] = spec["edge_enabled"]

    # In-channels configuration
    cfg["model"]["in_channels_rgb"] = spec["in_channels_rgb"]
    cfg["model"]["in_channels_nir"] = spec["in_channels_nir"]
    cfg["model"]["variant_mode"] = spec["mode"]

    return cfg


def build_variant_model(variant_key: str, cfg: Dict[str, Any]) -> nn.Module:
    """Instantiates model architecture tailored to the specified ablation variant."""
    spec = VARIANTS_SPECS[variant_key]
    model_cfg = cfg.get("model", {})

    try:
        model = ASGNet(
            num_classes=model_cfg.get("num_classes", 6),
            backbone=model_cfg.get("backbone", "mit_b2"),
            pretrained=model_cfg.get("pretrained", True),
            fusion_type=spec["fusion_type"],
            use_cm_frm=spec["use_cm_frm"],
            decoder_channels=model_cfg.get("decoder_channels", 256),
            edge_decoder=spec["edge_enabled"],
            in_channels_rgb=spec["in_channels_rgb"],
            in_channels_nir=spec["in_channels_nir"]
        )
    except (TypeError, Exception):
        # Fallback to standard instantiation with config dict
        try:
            model = ASGNet(config=cfg)
        except (TypeError, Exception):
            model = ASGNet(**model_cfg)

    return model


def forward_variant(
    model: nn.Module,
    rgb: torch.Tensor,
    nir: torch.Tensor,
    variant_key: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Handles modality routing according to the variant specification."""
    spec = VARIANTS_SPECS[variant_key]
    mode = spec["mode"]

    if mode == "rgb_only":
        # Pass RGB only (or dummy NIR if model signature requires 2 inputs)
        try:
            out = model(rgb)
        except (TypeError, Exception):
            dummy_nir = torch.zeros((rgb.size(0), 1, rgb.size(2), rgb.size(3)), dtype=rgb.dtype, device=rgb.device)
            out = model(rgb, dummy_nir)
    elif mode == "early_fusion":
        # Concatenate RGB and NIR along channel dimension -> 4 channels
        rgbnir = torch.cat([rgb, nir], dim=1)
        try:
            out = model(rgbnir)
        except (TypeError, Exception):
            out = model(rgbnir, None)
    else:
        # Standard dual branch
        out = model(rgb, nir)

    if isinstance(out, dict):
        seg_pred = out["seg_pred"]
        edge_pred = out.get("edge_pred", None)
    elif isinstance(out, (list, tuple)):
        seg_pred = out[0]
        edge_pred = out[1] if len(out) > 1 else None
    else:
        seg_pred = out
        edge_pred = None

    return seg_pred, edge_pred


def train_variant(
    variant_key: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    save_dir: str,
    fast_dev_run: bool,
    logger: logging.Logger
) -> Tuple[str, float]:
    """Trains a single model variant and returns path to best checkpoint and training duration."""
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    total_iters = max(1, epochs * len(train_loader))
    scheduler = PolyLRScheduler(optimizer, max_iters=total_iters, power=0.9, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{variant_key}] Ep {epoch}/{epochs}",
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                rgb = batch["rgb"].to(device, non_blocking=True)
                nir = batch["nir"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True).long()
                edge = batch.get("edge", None)
                if edge is not None:
                    edge = edge.to(device, non_blocking=True).float()
            else:
                rgb = batch[0].to(device, non_blocking=True)
                nir = batch[1].to(device, non_blocking=True)
                mask = batch[2].to(device, non_blocking=True).long()
                edge = batch[3].to(device, non_blocking=True).float() if len(batch) > 3 else None

            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                seg_pred, edge_pred = forward_variant(model, rgb, nir, variant_key)
                loss_out = criterion(seg_pred, mask, edge_pred=edge_pred, edge_target=edge)
                loss = loss_out[0] if isinstance(loss_out, tuple) else (loss_out["total_loss"] if isinstance(loss_out, dict) else loss_out)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * rgb.size(0)
            total_samples += rgb.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if fast_dev_run and batch_idx >= 9:
                break

        # Quick validation
        model.eval()
        cm = StreamingConfusionMatrix(num_classes=6)
        with torch.no_grad():
            for val_idx, batch in enumerate(val_loader):
                if isinstance(batch, dict):
                    rgb = batch["rgb"].to(device, non_blocking=True)
                    nir = batch["nir"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True).long()
                else:
                    rgb = batch[0].to(device, non_blocking=True)
                    nir = batch[1].to(device, non_blocking=True)
                    mask = batch[2].to(device, non_blocking=True).long()

                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    seg_pred, _ = forward_variant(model, rgb, nir, variant_key)

                preds = seg_pred.argmax(dim=1)
                cm.update(preds, mask)

                if fast_dev_run and val_idx >= 9:
                    break

        val_miou = cm.compute_miou()

        if val_miou > best_miou or epoch == 1:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "variant": variant_key
            }, best_ckpt_path)

        if fast_dev_run:
            break

    elapsed_time = time.time() - start_time
    logger.info(f"[{variant_key}] Finished training in {elapsed_time / 60:.2f}m. Best Val mIoU: {best_miou * 100:.2f}%")
    return best_ckpt_path, elapsed_time


def evaluate_variant(
    variant_key: str,
    checkpoint_path: str,
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    fast_dev_run: bool,
    logger: logging.Logger
) -> Tuple[float, float, float, np.ndarray, float]:
    """Evaluates best checkpoint on test/validation set and measures inference latency."""
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state_dict, strict=False)

    model.to(device)
    model.eval()

    cm = StreamingConfusionMatrix(num_classes=6)
    total_latency = 0.0
    total_patches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating [{variant_key}]", leave=False)):
            if isinstance(batch, dict):
                rgb = batch["rgb"].to(device, non_blocking=True)
                nir = batch["nir"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True).long()
            else:
                rgb = batch[0].to(device, non_blocking=True)
                nir = batch[1].to(device, non_blocking=True)
                mask = batch[2].to(device, non_blocking=True).long()

            if mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                seg_pred, _ = forward_variant(model, rgb, nir, variant_key)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = (time.time() - t0) * 1000.0  # ms

            total_latency += latency
            total_patches += rgb.size(0)

            preds = seg_pred.argmax(dim=1)
            cm.update(preds, mask)

            if fast_dev_run and batch_idx >= 9:
                break

    miou = cm.compute_miou()
    oa = cm.compute_overall_accuracy()
    f1 = cm.compute_f1()
    per_class_iou = cm.compute_per_class_iou()
    avg_latency = total_latency / max(1, total_patches)

    return miou, oa, f1, per_class_iou, avg_latency


def plot_comparative_charts(
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: str
) -> None:
    """Generates comparative visualizations and bar charts."""
    # Attempt to call visualize.plot_ablation_comparison if available
    try:
        if "plot_ablation_comparison" in globals() and callable(globals()["plot_ablation_comparison"]):
            try:
                plot_ablation_comparison(results, save_path=os.path.join(output_dir, "ablation_comparison.png"))
                return
            except Exception:
                pass
    except Exception:
        pass

    # Built-in publication-quality Matplotlib visualization
    variant_names = [results[v]["display_name"].split(":")[0] for v in results]
    full_labels = [results[v]["display_name"] for v in results]
    mious = [results[v]["mIoU"] * 100 for v in results]
    oas = [results[v]["OA"] * 100 for v in results]
    f1s = [results[v]["F1"] * 100 for v in results]
    params = [results[v]["params_million"] for v in results]

    # Chart 1: Key Performance Metrics Comparison (Grouped Bar Chart)
    fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(variant_names))
    width = 0.25

    rects1 = ax1.bar(x - width, mious, width, label="mIoU (%)", color="#1f77b4", edgecolor="black")
    rects2 = ax1.bar(x, oas, width, label="Overall Accuracy (%)", color="#2ca02c", edgecolor="black")
    rects3 = ax1.bar(x + width, f1s, width, label="Macro F1 (%)", color="#ff7f0e", edgecolor="black")

    ax1.set_ylabel("Metric Value (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Ablation Study: Architecture Variant Comparison on ISPRS Potsdam", fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_names, fontsize=11, fontweight="bold")
    ax1.set_ylim(min(70, min(mious) - 5), 100)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", frameon=True)

    # Add numeric labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.6,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "ablation_metrics_comparison.png")
    plt.savefig(chart1_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Chart 2: Per-Class IoU Progression
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for c_idx, c_name in enumerate(class_names):
        class_scores = [results[v]["per_class_iou"][c_name] * 100 for v in results]
        ax.plot(variant_names, class_scores, marker="o", linewidth=2.2, label=c_name)

    ax.set_ylabel("IoU (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class IoU Progression Across Ablation Variants", fontsize=14, pad=15)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(title="Semantic Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "ablation_per_class_progression.png")
    plt.savefig(chart2_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_summary_tables(results: Dict[str, Any], class_names: List[str]) -> Tuple[str, str]:
    """Constructs formatted ASCII and Markdown comparison tables."""
    # Table 1: Overall Comparison
    headers = ["Variant", "Description", "Params (M)", "mIoU (%)", "OA (%)", "F1 (%)", "Latency (ms)"]
    rows = []
    for k, v in results.items():
        v_name = v["display_name"].split(":")[0]
        desc = v["description"][:28] + ("..." if len(v["description"]) > 28 else "")
        rows.append([
            v_name,
            desc,
            f"{v['params_million']:.1f}M",
            f"{v['mIoU'] * 100:.2f}%",
            f"{v['OA'] * 100:.2f}%",
            f"{v['F1'] * 100:.2f}%",
            f"{v['latency_ms']:.1f}"
        ])

    # ASCII Format
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = "|" + "|".join([f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))]) + "|"

    ascii_lines = [sep, header_str, sep]
    for row in rows:
        line = "|" + "|".join([f" {row[i]:<{col_widths[i]}} " for i in range(len(row))]) + "|"
        ascii_lines.append(line)
    ascii_lines.append(sep)
    overall_table = "\n".join(ascii_lines)

    # Markdown Format
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---" for _ in headers]) + " |"
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_table = "\n".join(md_lines)

    return overall_table, md_table


def ablation() -> None:
    """Main ablation study execution routine."""
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("=" * 70)
    logger.info("ASG-Net: Comprehensive Ablation Study")
    logger.info("Benchmarking 5 Architectural Variants on ISPRS Potsdam")
    logger.info("=" * 70)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Execution Device: {device}")

    # Determine which variants to execute
    if args.variants.lower() == "all":
        variants_to_run = list(VARIANTS_SPECS.keys())
    else:
        variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip() in VARIANTS_SPECS]

    logger.info(f"Executing variants: {variants_to_run}")

    # Datasets
    logger.info("Loading Potsdam datasets and transforms...")
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    ds_root = config.get("dataset", {}).get("root", "data/potsdam")
    patch_size = config.get("dataset", {}).get("patch_size", 512)
    stride = config.get("dataset", {}).get("stride", 256)

    try:
        train_ds = PotsdamDataset(root=ds_root, split="train", patch_size=patch_size, stride=stride, transforms=train_transforms)
        val_ds = PotsdamDataset(root=ds_root, split="val", patch_size=patch_size, stride=stride, transforms=val_transforms)
        test_ds = PotsdamDataset(root=ds_root, split="test", patch_size=patch_size, stride=stride, transforms=val_transforms)
    except TypeError:
        train_ds = PotsdamDataset(root=ds_root, split="train", transforms=train_transforms)
        val_ds = PotsdamDataset(root=ds_root, split="val", transforms=val_transforms)
        test_ds = PotsdamDataset(root=ds_root, split="test", transforms=val_transforms)

    batch_size = args.batch_size if args.batch_size else config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 4)
    epochs = args.epochs if args.epochs else config.get("training", {}).get("epochs", 100)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = config.get("dataset", {}).get("class_names", [
        "Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"
    ])

    results: Dict[str, Any] = {}

    for v_key in variants_to_run:
        spec = VARIANTS_SPECS[v_key]
        logger.info("\n" + "=" * 70)
        logger.info(f"STARTING VARIANT: {spec['display_name']}")
        logger.info(f"Rationale: {spec['description']}")
        logger.info("=" * 70)

        v_cfg = configure_variant(config, v_key)
        v_save_dir = os.path.join(output_dir, v_key)

        model = build_variant_model(v_key, v_cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_million = total_params / 1e6
        logger.info(f"Parameters: Total = {params_million:.2f}M | Trainable = {trainable_params / 1e6:.2f}M")

        # Criterion
        criterion = CombinedLoss(config=v_cfg)

        # Train
        best_ckpt, train_duration = train_variant(
            variant_key=v_key,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            epochs=epochs,
            device=device,
            save_dir=v_save_dir,
            fast_dev_run=args.fast_dev_run,
            logger=logger
        )

        # Evaluate on Test set
        miou, oa, f1, per_class_iou, latency = evaluate_variant(
            variant_key=v_key,
            checkpoint_path=best_ckpt,
            model=model,
            test_loader=test_loader,
            device=device,
            fast_dev_run=args.fast_dev_run,
            logger=logger
        )

        # Record
        results[v_key] = {
            "display_name": spec["display_name"],
            "description": spec["description"],
            "fusion_type": spec["fusion_type"],
            "use_cm_frm": spec["use_cm_frm"],
            "edge_enabled": spec["edge_enabled"],
            "params_million": float(params_million),
            "train_duration_minutes": float(train_duration / 60.0),
            "latency_ms": float(latency),
            "mIoU": float(miou),
            "OA": float(oa),
            "F1": float(f1),
            "per_class_iou": {name: float(per_class_iou.get(name, per_class_iou.get(i, 0.0))) for i, name in enumerate(class_names)}
        }

        logger.info(f"[{v_key}] Test Results: mIoU={miou * 100:.2f}% | OA={oa * 100:.2f}% | F1={f1 * 100:.2f}%")

    # Final summary and plots
    logger.info("\n" + "=" * 70)
    logger.info("ALL ABLATION EXPERIMENTS COMPLETED")
    logger.info("=" * 70)

    ascii_table, md_table = generate_summary_tables(results, class_names)
    print("\n" + ascii_table + "\n")
    logger.info("\n" + ascii_table)

    # Save to JSON
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to JSON: {json_path}")

    # Save Markdown Summary
    md_summary_path = os.path.join(output_dir, "ablation_summary.md")
    with open(md_summary_path, "w", encoding="utf-8") as f:
        f.write("# ASG-Net Ablation Study Results\n\n")
        f.write("Systematic comparison of 5 architecture variants on the ISPRS Potsdam benchmark.\n\n")
        f.write(md_table + "\n\n")
        f.write("## Per-Class IoU Breakdown (%)\n\n")
        # Class breakdown table
        f.write("| Variant | " + " | ".join(class_names) + " |\n")
        f.write("| --- | " + " | ".join(["---" for _ in class_names]) + " |\n")
        for k, v in results.items():
            scores = [f"{v['per_class_iou'][c] * 100:.2f}%" for c in class_names]
            f.write(f"| {v['display_name'].split(':')[0]} | " + " | ".join(scores) + " |\n")
    logger.info(f"Summary markdown saved to: {md_summary_path}")

    # Generate Comparative Bar Charts
    plot_comparative_charts(results, class_names, output_dir)
    logger.info(f"Comparative charts generated in: {output_dir}")


if __name__ == "__main__":
    ablation()
