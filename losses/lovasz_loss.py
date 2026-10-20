"""
Lovász-Softmax Loss for Semantic Segmentation in PyTorch.

Based on the CVPR 2018 paper:
"The Lovász-Softmax loss: A tractable surrogate for the optimization of the
intersection-over-union measure in neural networks"
by Maxim Berman, Amal Rannen Triki, Matthew B. Blaschko.
Paper URL: https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_Lovasz-Softmax_Loss_CVPR_2018_paper.pdf

The Lovász extension allows direct optimization of the submodular Jaccard loss (1 - IoU)
in deep neural networks, yielding consistent improvements in mIoU over standard
Cross-Entropy loss alone.
"""

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovász extension w.r.t. sorted errors.

    See Algorithm 1 in Berman et al. (CVPR 2018).

    Args:
        gt_sorted: 1D tensor of ground truth labels sorted in descending order
            of prediction errors.

    Returns:
        1D tensor representing the gradient of the Lovász extension.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        # Subtract adjacent prefix values: Delta(S_k) - Delta(S_{k-1})
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ignore: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens predictions and labels for binary case, removing ignored pixels.

    Args:
        scores: Predicted scores or probabilities (arbitrary shape).
        labels: Ground truth binary labels (same shape as scores).
        ignore: Optional label value to ignore.

    Returns:
        Tuple of (flattened_scores, flattened_labels).
    """
    scores = scores.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore is not None:
        valid = labels != ignore
        scores = scores[valid]
        labels = labels[valid]
    return scores, labels


def flatten_probas(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens multi-class probability tensor and label tensor, filtering ignored pixels.

    Args:
        probas: [B, C, H, W] or [B, C, L] tensor of class probabilities.
        labels: [B, H, W] or [B, L] tensor of integer class labels.
        ignore: Optional label index to ignore (e.g., 255).

    Returns:
        Tuple of (probas_flat, labels_flat) where probas_flat is [P, C]
        and labels_flat is [P].
    """
    if probas.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C) -> (-1, C)
        B, C, H, W = probas.shape
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.contiguous().view(-1)
    elif probas.dim() == 3:
        # (B, C, L) -> (B, L, C) -> (-1, C)
        B, C, L = probas.shape
        probas = probas.permute(0, 2, 1).contiguous().view(-1, C)
        labels = labels.contiguous().view(-1)
    elif probas.dim() == 2:
        # (P, C)
        labels = labels.contiguous().view(-1)
    else:
        raise ValueError(
            f"Expected probas to have 2, 3, or 4 dimensions, got {probas.dim()} (shape {probas.shape})"
        )

    if ignore is not None:
        valid = labels != ignore
        probas = probas[valid]
        labels = labels[valid]

    return probas, labels


def lovasz_softmax_flat(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | Sequence[int] = "present",
) -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss on flattened probability and label tensors.

    Args:
        probas: [P, C] tensor of class probabilities at valid pixels.
        labels: [P] tensor of integer ground truth labels in [0, C-1].
        classes: 'all' to evaluate across all classes 0..C-1,
                 'present' to evaluate only over classes present in labels,
                 or an explicit sequence of class indices to consider.

    Returns:
        Scalar tensor containing the computed Lovász-Softmax loss.
    """
    if probas.numel() == 0 or labels.numel() == 0:
        # Return zero loss maintaining autograd graph
        return probas.sum() * 0.0

    C = probas.size(1)
    losses: list[torch.Tensor] = []

    if classes in ("all", "present"):
        class_list = list(range(C))
    elif isinstance(classes, (list, tuple, range)):
        class_list = list(classes)
    else:
        raise ValueError(
            f"Invalid classes argument: {classes}. Expected 'all', 'present', or sequence of class indices."
        )

    for c in class_list:
        target_c = (labels == c).float()
        if classes == "present" and target_c.sum() == 0:
            continue

        # Error vector: e_i = |y_i - m_i|
        if C == 1:
            errors = (target_c - probas.squeeze(1)).abs()
        else:
            errors = (target_c - probas[:, c]).abs()

        # Sort errors in descending order
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = target_c[perm.data]

        # Compute gradient of the Lovász extension
        grad = lovasz_grad(gt_sorted)

        # Dot product produces the surrogate loss for class c
        loss_c = torch.dot(errors_sorted, grad.detach())
        losses.append(loss_c)

    if len(losses) == 0:
        return probas.sum() * 0.0

    return torch.stack(losses).mean()


def lovasz_softmax(
    probas: torch.Tensor,
    labels: torch.Tensor,
    classes: str | Sequence[int] = "present",
    per_image: bool = False,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    """
    Multi-class Lovász-Softmax loss for batch segmentation predictions.

    Args:
        probas: [B, C, H, W] tensor of class probabilities (e.g. from softmax).
        labels: [B, H, W] tensor of integer ground truth class labels.
        classes: 'all' to average over all C classes, 'present' for classes in labels,
                 or an explicit sequence of class indices.
        per_image: If True, loss is computed independently per image and averaged.
                   If False, loss is computed over the entire flattened batch.
        ignore_index: Integer ground truth label value to exclude from loss calculation
                      (default: 255).

    Returns:
        Scalar loss tensor.
    """
    if per_image and probas.dim() == 4 and probas.size(0) > 1:
        losses: list[torch.Tensor] = []
        for p, l in zip(probas, labels):
            p_flat, l_flat = flatten_probas(
                p.unsqueeze(0), l.unsqueeze(0), ignore=ignore_index
            )
            losses.append(lovasz_softmax_flat(p_flat, l_flat, classes=classes))
        return torch.stack(losses).mean()
    else:
        p_flat, l_flat = flatten_probas(probas, labels, ignore=ignore_index)
        return lovasz_softmax_flat(p_flat, l_flat, classes=classes)


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax Loss Module for Semantic Segmentation.

    Directly optimizes the Jaccard (IoU) metric using the Lovász extension of
    submodular functions. When paired with standard Cross-Entropy loss, it accelerates
    convergence toward higher mIoU benchmarks by penalizing global misclassifications
    proportionately to their overlap impact.

    Args:
        classes: 'present' (default) computes loss only on classes present in the batch,
                 'all' computes loss over all num_classes,
                 or an explicit sequence of class indices.
        per_image: If True, evaluates the loss per image and averages over the batch.
                   Default is False (evaluates across the entire batch).
        ignore_index: Label value indicating void / unannotated pixels (default: 255).
    """

    def __init__(
        self,
        classes: str | Sequence[int] = "present",
        per_image: bool = False,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes Lovász-Softmax loss from unnormalized class logits.

        Args:
            logits: [B, C, H, W] unnormalized class prediction scores.
            labels: [B, H, W] integer ground truth class annotations.

        Returns:
            Scalar Lovász-Softmax loss tensor.
        """
        # Apply softmax across channel dimension to produce probabilities
        probas = F.softmax(logits, dim=1)
        return lovasz_softmax(
            probas=probas,
            labels=labels,
            classes=self.classes,
            per_image=self.per_image,
            ignore_index=self.ignore_index,
        )
