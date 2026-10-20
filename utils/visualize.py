"""
Publication-Quality Visualization Utilities for ASG-Net.

Provides high-resolution (300 DPI) visualization routines for:
1. colorize_mask: Convert class index masks to RGB images using custom palettes.
2. plot_prediction: Side-by-side visualization of RGB input, Ground Truth, and Model Prediction.
3. plot_training_curves: Dual-panel training/validation loss and mIoU progression curves.
4. plot_confusion_matrix: Row-normalized confusion matrix heatmap via Seaborn.
5. plot_per_class_iou: Horizontal bar chart highlighting class-wise IoU and mean benchmark.
6. plot_ablation_comparison: Grouped bar chart comparing architectural ablation variants.
"""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
# Use non-interactive backend for headless environments, servers, and notebooks
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _setup_style() -> None:
    """Configures clean, publication-ready styling parameters for matplotlib."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
        }
    )


def colorize_mask(
    mask: np.ndarray | torch.Tensor,
    class_colors: Sequence[Sequence[int]] | np.ndarray,
    ignore_index: int = 255,
    ignore_color: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """
    Converts a 2D integer class index mask into an RGB color image.

    Args:
        mask: 2D integer array or tensor of shape [H, W] containing class indices.
        class_colors: Sequence of RGB color triples [R, G, B] in [0, 255] for each class index.
        ignore_index: Integer index designating void/unlabeled pixels (default: 255).
        ignore_color: RGB color triple for ignored pixels (default: (0, 0, 0) - black).

    Returns:
        RGB image as a NumPy array of shape [H, W, 3] with dtype uint8.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if mask.ndim != 2:
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        else:
            raise ValueError(f"Expected 2D mask [H, W], got shape {mask.shape}")

    mask_int = mask.astype(np.int64)
    h, w = mask_int.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    # Fill ignore color by default
    colored[:, :] = ignore_color

    # Map each class index to its corresponding RGB color
    for idx, color in enumerate(class_colors):
        class_mask = mask_int == idx
        colored[class_mask] = color

    return colored


def _denormalize_rgb(
    rgb: np.ndarray | torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalizes an RGB image from ImageNet standard distribution to uint8 [0, 255].

    Args:
        rgb: Image array or tensor. Shapes supported: [3, H, W], [1, 3, H, W], or [H, W, 3].
        mean: Sequence of per-channel means used during normalization.
        std: Sequence of per-channel standard deviations used during normalization.

    Returns:
        Denormalized image array of shape [H, W, 3] with dtype uint8.
    """
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()

    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb.squeeze(0)

    # Convert CHW -> HWC if necessary
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
        rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.dtype == np.uint8:
        return rgb

    rgb_float = rgb.astype(np.float32)

    # Check if image appears normalized (contains negative values or values close to 0-1)
    if rgb_float.min() < 0.0 or rgb_float.max() <= 1.0:
        if rgb_float.min() < -0.1:  # Normalized with mean/std
            mean_arr = np.array(mean, dtype=np.float32)
            std_arr = np.array(std, dtype=np.float32)
            rgb_float = (rgb_float * std_arr) + mean_arr
        rgb_float = np.clip(rgb_float * 255.0, 0, 255)

    return rgb_float.astype(np.uint8)


def plot_prediction(
    rgb: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    pred_mask: np.ndarray | torch.Tensor,
    class_colors: Sequence[Sequence[int]],
    class_names: Sequence[str],
    save_path: str | Path | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Generates a publication-quality side-by-side comparison figure:
    [ Input RGB ]  |  [ Ground Truth Mask ]  |  [ ASG-Net Prediction ]
    along with a shared class legend.

    Args:
        rgb: Input RGB image tensor [3, H, W] or array [H, W, 3].
        gt_mask: Ground truth 2D integer mask [H, W].
        pred_mask: Model prediction 2D integer mask [H, W] (e.g. from argmax).
        class_colors: List of RGB color triplets for each class.
        class_names: List of class category names matching class_colors.
        save_path: Optional file path to save the 300 DPI figure.
        title: Optional overarching figure super-title.

    Returns:
        The generated matplotlib Figure object.
    """
    _setup_style()

    rgb_img = _denormalize_rgb(rgb)
    gt_colored = colorize_mask(gt_mask, class_colors)
    pred_colored = colorize_mask(pred_mask, class_colors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    panels = [
        (rgb_img, "Input RGB Image"),
        (gt_colored, "Ground Truth Annotation"),
        (pred_colored, "ASG-Net Prediction"),
    ]

    for ax, (img, panel_title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(panel_title, pad=10, fontweight="bold", color="#222222")
        ax.axis("off")

    # Build legend handles
    legend_patches = [
        mpatches.Patch(
            facecolor=np.array(color) / 255.0,
            edgecolor="#555555",
            label=name,
        )
        for color, name in zip(class_colors, class_names)
    ]

    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=min(len(class_names), 6),
        bbox_to_anchor=(0.5, 0.02),
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#D0D0D0",
        fontsize=10,
    )

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    return fig


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_mious: Sequence[float],
    val_mious: Sequence[float],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plots training and validation progression curves for Loss and mIoU.

    Creates two subplots:
    - Left: Training vs. Validation Loss (with minimum val loss annotated)
    - Right: Training vs. Validation mIoU (%) (with maximum val mIoU annotated)

    Args:
        train_losses: Sequence of training loss values per epoch.
        val_losses: Sequence of validation loss values per epoch.
        train_mious: Sequence of training mIoU values (0.0-1.0 or 0-100%) per epoch.
        val_mious: Sequence of validation mIoU values per epoch.
        save_path: Optional path to save the 300 DPI plot.

    Returns:
        The generated matplotlib Figure object.
    """
    _setup_style()

    epochs = list(range(1, len(train_losses) + 1))

    # Convert mIoU to percentage if provided in [0, 1] range
    train_mious_pct = [
        val * 100.0 if (len(train_mious) > 0 and max(train_mious) <= 1.0) else val
        for val in train_mious
    ]
    val_mious_pct = [
        val * 100.0 if (len(val_mious) > 0 and max(val_mious) <= 1.0) else val
        for val in val_mious
    ]

    fig, (ax_loss, ax_miou) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Loss Progression ---
    ax_loss.plot(
        epochs,
        train_losses,
        label="Train Loss",
        color="#1F77B4",
        linewidth=2.0,
    )
    ax_loss.plot(
        epochs,
        val_losses,
        label="Val Loss",
        color="#FF7F0E",
        linewidth=2.0,
        linestyle="--",
    )

    if len(val_losses) > 0:
        min_loss_idx = int(np.argmin(val_losses))
        min_loss_val = val_losses[min_loss_idx]
        ax_loss.scatter(
            [epochs[min_loss_idx]],
            [min_loss_val],
            color="#D62728",
            s=80,
            zorder=5,
            edgecolors="black",
        )
        ax_loss.annotate(
            f"Best: {min_loss_val:.4f}\n(Epoch {epochs[min_loss_idx]})",
            xy=(epochs[min_loss_idx], min_loss_val),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF9C4", ec="#FBC02D", lw=0.8),
        )

    ax_loss.set_title("Training and Validation Loss", pad=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#D0D0D0")

    # --- Subplot 2: mIoU Progression ---
    ax_miou.plot(
        epochs,
        train_mious_pct,
        label="Train mIoU",
        color="#2CA02C",
        linewidth=2.0,
    )
    ax_miou.plot(
        epochs,
        val_mious_pct,
        label="Val mIoU",
        color="#9467BD",
        linewidth=2.0,
        linestyle="--",
    )

    if len(val_mious_pct) > 0:
        max_miou_idx = int(np.argmax(val_mious_pct))
        max_miou_val = val_mious_pct[max_miou_idx]
        ax_miou.scatter(
            [epochs[max_miou_idx]],
            [max_miou_val],
            color="#D62728",
            s=80,
            zorder=5,
            edgecolors="black",
        )
        ax_miou.annotate(
            f"Best: {max_miou_val:.2f}%\n(Epoch {epochs[max_miou_idx]})",
            xy=(epochs[max_miou_idx], max_miou_val),
            xytext=(-45, -35),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
            fontsize=9,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50", lw=0.8),
        )

    ax_miou.set_title("Mean IoU Progression", pad=12, fontweight="bold")
    ax_miou.set_xlabel("Epoch")
    ax_miou.set_ylabel("mIoU (%)")
    ax_miou.grid(True, linestyle="--", alpha=0.6)
    ax_miou.legend(frameon=True, facecolor="#FAFAFA", edgecolor="#D0D0D0")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: str | Path | None = None,
    normalize: bool = True,
    title: str = "Normalized Confusion Matrix",
) -> plt.Figure:
    """
    Renders a row-normalized confusion matrix heatmap using Seaborn.

    Rows represent Ground Truth categories and columns represent Predicted categories.

    Args:
        cm: Confusion matrix array of shape [num_classes, num_classes].
        class_names: Sequence of category labels for axes.
        save_path: Optional path to save the 300 DPI figure.
        normalize: If True, normalizes each row by its ground truth sum (shows percentages 0-100%).
        title: Figure title.

    Returns:
        The generated matplotlib Figure object.
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=(8, 7))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(
            cm.astype(np.float64),
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        ) * 100.0
        data_to_plot = norm_cm
        fmt = ".1f"
        cbar_label = "Recall (%)"
    else:
        data_to_plot = cm
        fmt = "d"
        cbar_label = "Pixel Count"

    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="#E0E0E0",
    )

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel("Predicted Class", labelpad=10, fontweight="semibold")
    ax.set_ylabel("True Class", labelpad=10, fontweight="semibold")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    return fig


def plot_per_class_iou(
    class_ious: Mapping[str, float] | Sequence[float],
    class_names: Sequence[str] | None = None,
    save_path: str | Path | None = None,
    title: str = "Per-Class Intersection-over-Union (IoU)",
) -> plt.Figure:
    """
    Generates a horizontal bar chart displaying per-class IoU scores.

    Highlights individual class performance and indicates the overall Mean IoU
    with a distinct vertical reference benchmark line.

    Args:
        class_ious: Either a dict mapping class name to IoU float value, or a sequence
                    of IoU float values corresponding to class_names.
        class_names: Sequence of class names. Required if class_ious is a sequence.
        save_path: Optional path to save the 300 DPI plot.
        title: Plot title.

    Returns:
        The generated matplotlib Figure object.
    """
    _setup_style()

    if isinstance(class_ious, Mapping):
        names = list(class_ious.keys())
        values = list(class_ious.values())
    else:
        if class_names is None:
            raise ValueError("class_names must be provided when class_ious is a sequence")
        names = list(class_names)
        values = list(class_ious)

    # Convert values to percentages if in [0, 1] range
    is_fraction = len(values) > 0 and max([v for v in values if not np.isnan(v)] or [0]) <= 1.0
    pct_values = [v * 100.0 if is_fraction else v for v in values]

    valid_vals = [v for v in pct_values if not np.isnan(v)]
    mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(names) * 0.75)))

    # Color palette: soft teal gradient
    bar_colors = ["#2B7A78" for _ in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, pct_values, color=bar_colors, edgecolor="#17252A", height=0.6)

    # Annotate numeric values at bar tips
    for bar, val in zip(bars, pct_values):
        width = bar.get_width()
        text_str = f" {val:.2f}%" if not np.isnan(val) else " N/A"
        ax.text(
            width + 0.8,
            bar.get_y() + bar.get_height() / 2,
            text_str,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="semibold",
            color="#222222",
        )

    # Vertical benchmark line for mIoU
    ax.axvline(
        mean_val,
        color="#D32F2F",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean IoU: {mean_val:.2f}%",
        zorder=4,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontweight="semibold")
    ax.invert_yaxis()  # Top-down order
    ax.set_xlim(0, 105)
    ax.set_xlabel("IoU (%)", labelpad=8, fontweight="semibold")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.legend(loc="lower right", frameon=True, facecolor="#FAFAFA", edgecolor="#D0D0D0")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    return fig


def plot_ablation_comparison(
    results_dict: Mapping[str, Mapping[str, float]],
    save_path: str | Path | None = None,
    title: str = "Ablation Study: Architecture Variant Performance",
) -> plt.Figure:
    """
    Renders a grouped bar chart comparing multiple architectural variants
    across evaluation metrics (e.g., mIoU, Overall Accuracy, F1-Score).

    Args:
        results_dict: Nested mapping of the structure:
            {
                "V1: RGB-Only": {"mIoU": 82.1, "OA": 88.2, "F1": 84.5},
                "V2: Early Fusion": {"mIoU": 84.3, "OA": 89.4, "F1": 86.1},
                ...
            }
        save_path: Optional path to save the 300 DPI plot.
        title: Plot title.

    Returns:
        The generated matplotlib Figure object.
    """
    _setup_style()

    variants = list(results_dict.keys())
    if len(variants) == 0:
        raise ValueError("results_dict must contain at least one variant entry")

    # Discover metric names across variants
    metric_names = list(results_dict[variants[0]].keys())
    num_metrics = len(metric_names)

    x = np.arange(len(variants))
    bar_width = 0.8 / max(1, num_metrics)

    # Distinct, harmonious academic palette
    palette = ["#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD", "#17BECF"]

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 2.2), 6))

    all_vals = []
    for i, metric in enumerate(metric_names):
        offset = (i - (num_metrics - 1) / 2) * bar_width
        vals = [results_dict[v].get(metric, 0.0) for v in variants]
        # Normalize to percentage if values are fractions <= 1.0
        vals_pct = [v * 100.0 if (v <= 1.0 and v > 0.0) else v for v in vals]
        all_vals.extend(vals_pct)

        color = palette[i % len(palette)]
        rects = ax.bar(
            x + offset,
            vals_pct,
            bar_width,
            label=metric,
            color=color,
            edgecolor="#333333",
            linewidth=0.8,
        )

        # Annotate values on top of bars
        for rect, val in zip(rects, vals_pct):
            height = rect.get_height()
            ax.annotate(
                f"{val:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )

    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontweight="semibold", rotation=15, ha="right")
    ax.set_ylabel("Score (%)", labelpad=8, fontweight="semibold")

    # Set smart y-axis limit to emphasize differences clearly
    min_val = min(all_vals) if len(all_vals) > 0 else 0
    max_val = max(all_vals) if len(all_vals) > 0 else 100
    y_lower = max(0, int(min_val - 10))
    y_upper = min(100, int(max_val + 7))
    ax.set_ylim(y_lower, y_upper)

    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#FAFAFA",
        edgecolor="#D0D0D0",
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    return fig
