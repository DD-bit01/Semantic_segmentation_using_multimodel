"""
Training Script for ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network
for RGB-Multispectral Semantic Segmentation.

This script implements full training with:
- Mixed precision training (torch.cuda.amp)
- PolyLR scheduling and AdamW optimization
- Combined loss (Cross-Entropy + Lovász-Softmax + Boundary/Edge Loss)
- Streaming confusion matrix evaluation (mIoU, OA, F1, per-class IoU)
- TensorBoard logging of scalar metrics and sample visual predictions
- Model checkpointing (best model, periodic checkpoints, resume support)
- Early stopping mechanism
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

import time
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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


def setup_logger(log_dir: str) -> logging.Logger:
    """Configures console and file logging.
    
    Args:
        log_dir: Directory where the log file will be saved.
        
    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ASGNet_Train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(log_dir, f"train_{int(time.time())}.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across standard library, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration dictionary from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_args() -> argparse.Namespace:
    """Parses command line arguments, supporting overrides for configuration file options."""
    parser = argparse.ArgumentParser(
        description="Train ASG-Net for RGB-Multispectral Semantic Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth to resume training from"
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 epoch with 10 batches for smoke testing / debugging"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda', 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Dataloader worker processes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def build_model(config: Dict[str, Any]) -> ASGNet:
    """Instantiates the ASGNet model from configuration.
    
    Args:
        config: Full configuration dictionary.
        
    Returns:
        Instantiated ASGNet model.
    """
    model_cfg = config.get("model", {})
    # Instantiate with config or explicit parameters
    try:
        return ASGNet.from_config(config)
    except Exception:
        pass

    try:
        return ASGNet(config=config)
    except (TypeError, Exception):
        pass

    try:
        return ASGNet(**model_cfg)
    except (TypeError, Exception):
        pass

    fusion_cfg = model_cfg.get("fusion", {})
    edge_cfg = model_cfg.get("edge_decoder", {})
    return ASGNet(
        num_classes=model_cfg.get("num_classes", 6),
        backbone=model_cfg.get("backbone", "mit_b2"),
        pretrained=model_cfg.get("pretrained", True),
        fusion_type=fusion_cfg.get("type", "asg"),
        use_cm_frm=fusion_cfg.get("use_cm_frm", True),
        decoder_channels=model_cfg.get("decoder_channels", 256),
        edge_decoder=edge_cfg.get("enabled", True),
    )


def build_loss(config: Dict[str, Any], device: torch.device) -> CombinedLoss:
    """Constructs the CombinedLoss function using weights and parameters from config.
    
    Args:
        config: Full configuration dictionary.
        device: Device on which class weights will be allocated.
        
    Returns:
        Instantiated CombinedLoss module.
    """
    loss_cfg = config.get("training", {}).get("loss", {})
    dataset_cfg = config.get("dataset", {})

    ce_weight = loss_cfg.get("ce_weight", 1.0)
    lovasz_weight = loss_cfg.get("lovasz_weight", 0.75)
    edge_weight = loss_cfg.get("edge_weight", 0.5)

    raw_weights = dataset_cfg.get("class_weights", None)
    if raw_weights is not None:
        class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    try:
        return CombinedLoss(
            config=config,
            class_weights=class_weights
        )
    except (TypeError, Exception):
        pass

    return CombinedLoss(
        num_classes=dataset_cfg.get("num_classes", 6),
        ce_weight=ce_weight,
        lovasz_weight=lovasz_weight,
        edge_weight=edge_weight,
        class_weights=class_weights
    )


def extract_batch_data(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Extracts rgb, nir, mask, and edge tensors from dataloader batch.
    
    Args:
        batch: Dictionary or tuple from DataLoader.
        device: Target execution device.
        
    Returns:
        Tuple of (rgb, nir, mask, edge). Edge may be None if unavailable.
    """
    if isinstance(batch, dict):
        rgb = batch["rgb"].to(device, non_blocking=True)
        nir = batch["nir"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True).long()
        edge = batch.get("edge", None)
        if edge is not None:
            edge = edge.to(device, non_blocking=True).float()
    elif isinstance(batch, (list, tuple)):
        rgb = batch[0].to(device, non_blocking=True)
        nir = batch[1].to(device, non_blocking=True)
        mask = batch[2].to(device, non_blocking=True).long()
        edge = batch[3].to(device, non_blocking=True).float() if len(batch) > 3 else None
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    # Ensure mask is 2D spatial: (B, H, W)
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)

    return rgb, nir, mask, edge


def forward_and_compute_loss(
    model: nn.Module,
    criterion: nn.Module,
    rgb: torch.Tensor,
    nir: torch.Tensor,
    mask: torch.Tensor,
    edge: Optional[torch.Tensor],
    use_amp: bool
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Performs forward pass with mixed precision and computes total and component losses.
    
    Returns:
        total_loss: Loss tensor for backward pass.
        loss_components: Dict of scalar losses for tracking.
        seg_pred: Predicted class logits.
    """
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(rgb, nir)
        if isinstance(outputs, dict):
            seg_pred = outputs["seg_pred"]
            edge_pred = outputs.get("edge_pred", None)
        elif isinstance(outputs, (tuple, list)):
            seg_pred = outputs[0]
            edge_pred = outputs[1] if len(outputs) > 1 else None
        else:
            seg_pred = outputs
            edge_pred = None

        # Compute loss
        loss_out = criterion(seg_pred, mask, edge_pred=edge_pred, edge_target=edge)

        if isinstance(loss_out, tuple):
            total_loss, loss_dict = loss_out
        elif isinstance(loss_out, dict):
            loss_dict = loss_out
            total_loss = loss_dict.get("total_loss", loss_dict.get("total", list(loss_dict.values())[0]))
        else:
            total_loss = loss_out
            loss_dict = getattr(criterion, "last_loss_dict", {"total_loss": total_loss.item()})

        # Ensure loss_dict contains float values
        scalar_loss_dict = {}
        if isinstance(loss_dict, dict):
            for k, v in loss_dict.items():
                scalar_loss_dict[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        else:
            scalar_loss_dict["total_loss"] = total_loss.item()

    return total_loss, scalar_loss_dict, seg_pred


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    gradient_clip: float,
    use_amp: bool,
    fast_dev_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, Dict[str, float], float]:
    """Executes one epoch of training over the training DataLoader.
    
    Returns:
        avg_total_loss: Average total loss across all batches.
        avg_loss_components: Dict of average component losses.
        train_miou: Approximate training mIoU.
    """
    model.train()
    running_losses: Dict[str, float] = {}
    total_samples = 0

    pbar = tqdm(
        train_loader,
        desc=f"Train Epoch [{epoch}/{total_epochs}]",
        leave=False,
        dynamic_ncols=True
    )

    cm = StreamingConfusionMatrix(num_classes=6)

    for batch_idx, batch in enumerate(pbar):
        rgb, nir, mask, edge = extract_batch_data(batch, device)
        batch_size = rgb.size(0)

        optimizer.zero_grad(set_to_none=True)

        total_loss, loss_dict, seg_pred = forward_and_compute_loss(
            model=model,
            criterion=criterion,
            rgb=rgb,
            nir=nir,
            mask=mask,
            edge=edge,
            use_amp=use_amp
        )

        # Backward with amp scaler
        scaler.scale(total_loss).backward()

        # Gradient clipping
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # Step scheduler per iteration
        scheduler.step()

        # Update metrics and running loss
        with torch.no_grad():
            if seg_pred.shape[-2:] != mask.shape[-2:]:
                seg_pred_eval = F.interpolate(seg_pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            else:
                seg_pred_eval = seg_pred
            preds = seg_pred_eval.argmax(dim=1)
            cm.update(preds, mask)

        total_samples += batch_size
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v * batch_size

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update progress bar
        ce_val = loss_dict.get("ce_loss", 0.0)
        lov_val = loss_dict.get("lovasz_loss", 0.0)
        edge_val = loss_dict.get("edge_loss", 0.0)
        pbar.set_postfix({
            "Loss": f"{total_loss.item():.4f}",
            "CE": f"{ce_val:.3f}",
            "Lov": f"{lov_val:.3f}",
            "Edge": f"{edge_val:.3f}",
            "LR": f"{current_lr:.2e}"
        })

        if fast_dev_run and batch_idx >= 9:
            if logger:
                logger.info("Fast dev run: stopping training epoch after 10 batches.")
            break

    avg_total_loss = running_losses.get("total_loss", 0.0) / max(1, total_samples)
    avg_loss_components = {
        k: v / max(1, total_samples) for k, v in running_losses.items()
    }
    train_miou = cm.compute_miou()

    return avg_total_loss, avg_loss_components, train_miou


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 6,
    use_amp: bool = True,
    fast_dev_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float, float, float, np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Evaluates the model on validation set.
    
    Returns:
        val_loss: Average validation loss.
        miou: Mean Intersection-over-Union.
        oa: Overall Accuracy.
        f1: Macro F1-score.
        per_class_iou: Numpy array of per-class IoU values.
        sample_viz: Tuple of (sample_rgb, sample_gt, sample_pred) for visualization.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    cm = StreamingConfusionMatrix(num_classes=num_classes)

    sample_viz = None

    pbar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            rgb, nir, mask, edge = extract_batch_data(batch, device)
            batch_size = rgb.size(0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(rgb, nir)
                if isinstance(outputs, dict):
                    seg_pred = outputs["seg_pred"]
                    edge_pred = outputs.get("edge_pred", None)
                elif isinstance(outputs, (tuple, list)):
                    seg_pred = outputs[0]
                    edge_pred = outputs[1] if len(outputs) > 1 else None
                else:
                    seg_pred = outputs
                    edge_pred = None

                # Compute loss
                loss_out = criterion(seg_pred, mask, edge_pred=edge_pred, edge_target=edge)
                if isinstance(loss_out, tuple):
                    val_loss, _ = loss_out
                elif isinstance(loss_out, dict):
                    val_loss = loss_out.get("total_loss", loss_out.get("total", list(loss_out.values())[0]))
                else:
                    val_loss = loss_out

            if seg_pred.shape[-2:] != mask.shape[-2:]:
                seg_pred_eval = F.interpolate(seg_pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
            else:
                seg_pred_eval = seg_pred
            preds = seg_pred_eval.argmax(dim=1)
            cm.update(preds, mask)

            running_loss += val_loss.item() * batch_size
            total_samples += batch_size

            # Save first sample for visual logging
            if sample_viz is None and rgb.size(0) > 0:
                sample_viz = (rgb[0].detach().cpu(), mask[0].detach().cpu(), preds[0].detach().cpu())

            if fast_dev_run and batch_idx >= 9:
                if logger:
                    logger.info("Fast dev run: stopping validation after 10 batches.")
                break

    avg_val_loss = running_loss / max(1, total_samples)
    miou = cm.compute_miou()
    oa = cm.compute_overall_accuracy()
    f1 = cm.compute_f1()
    per_class_iou = cm.compute_per_class_iou()

    return avg_val_loss, miou, oa, f1, per_class_iou, sample_viz


def log_sample_visualizations(
    writer: SummaryWriter,
    sample_viz: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epoch: int,
    config: Dict[str, Any]
) -> None:
    """Renders and logs a side-by-side visualization to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter.
        sample_viz: Tuple of (rgb, mask, pred).
        epoch: Current training epoch.
        config: Configuration dictionary with class colors and normalization.
    """
    try:
        rgb_tensor, gt_mask, pred_mask = sample_viz
        # Unnormalize RGB for display
        norm_cfg = config.get("augmentation", {}).get("normalize", {})
        mean = torch.tensor(norm_cfg.get("rgb_mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
        std = torch.tensor(norm_cfg.get("rgb_std", [0.229, 0.224, 0.225])).view(3, 1, 1)
        rgb_disp = (rgb_tensor * std + mean).clamp(0, 1).numpy()
        rgb_disp = np.transpose(rgb_disp, (1, 2, 0))  # (H, W, 3)

        class_colors = config.get("dataset", {}).get("class_colors", [
            [255, 255, 255], [0, 0, 255], [0, 255, 255],
            [0, 255, 0], [255, 255, 0], [255, 0, 0]
        ])

        # Colorize masks
        gt_colored = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        pred_colored = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)

        for c_idx, color in enumerate(class_colors):
            gt_colored[gt_mask == c_idx] = color
            pred_colored[pred_mask == c_idx] = color

        gt_colored_norm = gt_colored.astype(np.float32) / 255.0
        pred_colored_norm = pred_colored.astype(np.float32) / 255.0

        # Stack horizontally: RGB | Ground Truth | Prediction
        stacked = np.concatenate([rgb_disp, gt_colored_norm, pred_colored_norm], axis=1)
        stacked_tensor = torch.from_numpy(np.transpose(stacked, (2, 0, 1))).float()

        writer.add_image("Visualizations/RGB_GT_Pred", stacked_tensor, epoch)
    except Exception as e:
        # Visual logging failure should never crash training
        pass


def train() -> None:
    """Main training routine for ASG-Net."""
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.output_dir is not None:
        config["paths"]["checkpoint_dir"] = args.output_dir
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    # Directories
    checkpoint_dir = config["paths"].get("checkpoint_dir", "checkpoints")
    log_dir = config["paths"].get("log_dir", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Logger
    logger = setup_logger(log_dir)
    logger.info("=" * 60)
    logger.info("ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network")
    logger.info("Initializing Training Pipeline...")
    logger.info("=" * 60)
    logger.info(f"Loaded configuration from: {args.config}")

    # Seed
    seed_everything(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute device: {device} (CUDA Available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Transforms & Datasets
    logger.info("Building data transforms and datasets...")
    try:
        train_transforms = get_train_transforms(config)
        val_transforms = get_val_transforms(config)
    except TypeError:
        # Fallback if transforms take explicit arguments
        patch_size = config.get("dataset", {}).get("patch_size", 512)
        train_transforms = get_train_transforms(patch_size)
        val_transforms = get_val_transforms(patch_size)

    ds_root = getattr(args, "data_dir", None) or config.get("dataset", {}).get("root", "data/potsdam")
    if not os.path.exists(os.path.join(ds_root, "rgb", "train")):
        if os.path.exists("data/potsdam_mini/rgb/train"):
            ds_root = "data/potsdam_mini"
            logger.info(f"Using demo dataset at: {ds_root}")
        elif os.path.exists("/content/data/potsdam_mini/rgb/train"):
            ds_root = "/content/data/potsdam_mini"
            logger.info(f"Using demo dataset at: {ds_root}")

    patch_size = config.get("dataset", {}).get("patch_size", 512)
    stride = config.get("dataset", {}).get("stride", 256)

    train_dataset = PotsdamDataset(
        root_dir=ds_root,
        split="train",
        patch_size=patch_size,
        stride=stride,
        transform=train_transforms
    )
    val_dataset = PotsdamDataset(
        root_dir=ds_root,
        split="val",
        patch_size=patch_size,
        stride=stride,
        transform=val_transforms
    )

    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["training"].get("num_workers", 2)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_dataset) > batch_size
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(f"Dataset Loaded: Train patches = {len(train_dataset)}, Val patches = {len(val_dataset)}")
    logger.info(f"DataLoader Batches: Train = {len(train_loader)}, Val = {len(val_loader)}")

    # Model
    logger.info("Instantiating ASG-Net model...")
    model = build_model(config).to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total = {total_params / 1e6:.2f}M | Trainable = {trainable_params / 1e6:.2f}M")

    # Loss, Optimizer, Scheduler
    criterion = build_loss(config, device).to(device)
    opt_cfg = config["training"].get("optimizer", {})
    lr = float(opt_cfg.get("lr", 6.0e-5))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )

    epochs = 1 if args.fast_dev_run else config["training"].get("epochs", 100)
    total_iters = max(1, epochs * len(train_loader))

    sched_cfg = config["training"].get("scheduler", {})
    power = float(sched_cfg.get("power", 0.9))
    min_lr = float(sched_cfg.get("min_lr", 1.0e-6))

    scheduler = PolyLRScheduler(
        optimizer=optimizer,
        max_epochs=total_iters,
        power=power,
        min_lr=min_lr
    )

    # Mixed precision scaler
    use_amp = bool(config["training"].get("amp", True)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"Mixed Precision (AMP) Enabled: {use_amp}")

    # TensorBoard
    tb_enabled = bool(config["logging"].get("tensorboard", True))
    writer = SummaryWriter(log_dir=log_dir) if tb_enabled else None
    if writer:
        logger.info(f"TensorBoard logging active in: {log_dir}")

    # Resume checkpoint if specified
    start_epoch = 1
    best_miou = 0.0
    history: Dict[str, list] = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": [],
        "val_oa": [],
        "val_f1": [],
        "learning_rate": []
    }

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception:
                    logger.warning("Could not restore scheduler state; reinitializing.")
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_miou = checkpoint.get("best_miou", 0.0)
            history = checkpoint.get("history", history)
            logger.info(f"Checkpoint restored: Epoch {start_epoch - 1}, Best mIoU: {best_miou * 100:.2f}%")
        else:
            logger.error(f"Checkpoint path not found: {args.resume}")
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    # Early stopping parameters
    es_cfg = config["training"].get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 15))
    epochs_without_improvement = 0

    gradient_clip = float(config["training"].get("gradient_clip", 1.0))
    edge_decay_epoch = int(config["training"].get("loss", {}).get("edge_decay_epoch", 50))
    save_freq = int(config["logging"].get("save_freq", 5))
    num_classes = config["dataset"].get("num_classes", 6)
    class_names = config["dataset"].get("class_names", [
        "Impervious Surface", "Building", "Low Vegetation", "Tree", "Car", "Clutter"
    ])

    logger.info("=" * 60)
    logger.info(f"Starting Training: Epochs {start_epoch} to {epochs}")
    logger.info("=" * 60)

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Boundary loss decay after edge_decay_epoch
        if epoch == edge_decay_epoch:
            if hasattr(criterion, "edge_weight"):
                criterion.edge_weight *= 0.5
                logger.info(f"Epoch {epoch}: Boundary loss weight decayed to {criterion.edge_weight}")

        # Training Phase
        train_loss, train_losses, train_miou = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            gradient_clip=gradient_clip,
            use_amp=use_amp,
            fast_dev_run=args.fast_dev_run,
            logger=logger
        )

        # Validation Phase
        val_loss, val_miou, val_oa, val_f1, per_class_iou, sample_viz = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            fast_dev_run=args.fast_dev_run,
            logger=logger
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # Record metrics
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_oa"].append(val_oa)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)

        # Formatted Progress Print
        progress_str = (
            f"Epoch [{epoch:3d}/{epochs:3d}] | "
            f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
            f"Train mIoU: {train_miou * 100:5.2f}% | "
            f"Val mIoU: {val_miou * 100:5.2f}% | "
            f"Val OA: {val_oa * 100:5.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        logger.info(progress_str)

        # TensorBoard Logging
        if writer:
            writer.add_scalar("Loss/Train_Total", train_loss, epoch)
            writer.add_scalar("Loss/Val_Total", val_loss, epoch)
            for k, v in train_losses.items():
                writer.add_scalar(f"Loss/Train_{k}", v, epoch)
            writer.add_scalar("Metrics/Train_mIoU", train_miou, epoch)
            writer.add_scalar("Metrics/Val_mIoU", val_miou, epoch)
            writer.add_scalar("Metrics/Val_OA", val_oa, epoch)
            writer.add_scalar("Metrics/Val_F1", val_f1, epoch)
            writer.add_scalar("Params/Learning_Rate", current_lr, epoch)

            for c_idx, c_name in enumerate(class_names):
                val = per_class_iou.get(c_name, 0.0)
                if not np.isnan(val):
                    writer.add_scalar(f"PerClass_IoU/{c_name}", val, epoch)

            if sample_viz is not None and epoch % 5 == 0:
                log_sample_visualizations(writer, sample_viz, epoch, config)

        # Checkpointing
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
            epochs_without_improvement = 0
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "scheduler_state_dict": getattr(scheduler, "state_dict", lambda: None)(),
                "best_miou": best_miou,
                "val_oa": val_oa,
                "val_f1": val_f1,
                "config": config,
                "history": history
            }, best_path)
            logger.info(f"★ New Best Model Saved to {best_path} (Val mIoU: {best_miou * 100:.2f}%)")
        else:
            epochs_without_improvement += 1

        # Latest Checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_miou": best_miou,
            "config": config,
            "history": history
        }, latest_path)

        # Periodic checkpoint
        if epoch % save_freq == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "config": config
            }, periodic_path)
            logger.info(f"Saved periodic checkpoint: {periodic_path}")

        # Early Stopping Check
        if es_enabled and epochs_without_improvement >= patience:
            logger.warning(
                f"Early stopping triggered: Val mIoU has not improved for {patience} epochs. "
                f"Best Val mIoU: {best_miou * 100:.2f}%"
            )
            break

        if args.fast_dev_run:
            logger.info("Fast dev run completed successfully.")
            break

    total_training_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training Completed in {total_training_time / 60:.2f} minutes.")
    logger.info(f"Best Validation mIoU: {best_miou * 100:.2f}%")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "best_miou": best_miou,
        "config": config,
        "history": history
    }, final_path)
    logger.info(f"Final model saved to: {final_path}")

    # Save training history
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Training history saved to: {history_path}")

    if writer:
        writer.close()


if __name__ == "__main__":
    train()
