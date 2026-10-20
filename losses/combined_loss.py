"""
Combined Loss Function for ASG-Net.

Combines three complementary objectives:
1. Class-Weighted Cross-Entropy Loss: Handles pixel-wise semantic classification and
   mitigates severe class imbalances (e.g., Car and Clutter vs. Impervious Surfaces).
2. Lovász-Softmax Loss: Directly optimizes the Jaccard index (mIoU) metric as a continuous
   surrogate for intersection-over-union.
3. Boundary-Aware Auxiliary Edge Loss: Weighted Binary Cross-Entropy with positive weight
   addressing extreme boundary sparsity (~5% boundary pixels). Includes an epoch-dependent
   decay schedule (decaying by 0.1x after edge_decay_epoch) to emphasize boundary localization
   early in training while refining interior semantic regions in later epochs.
"""

from typing import Any, Sequence, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz_loss import LovaszSoftmax


class CombinedLoss(nn.Module):
    """
    Multi-objective loss function combining Cross-Entropy, Lovász-Softmax,
    and boundary-aware auxiliary supervision for ASG-Net.

    Total Loss Formulation:
        L_total = lambda_ce * L_ce + lambda_lovasz * L_lovasz + lambda_edge * L_edge
    where lambda_edge decays by 0.1x when current_epoch >= edge_decay_epoch.

    Args:
        num_classes: Total number of semantic categories (e.g. 6 for ISPRS Potsdam).
        class_weights: Optional weighting factors for Cross-Entropy loss to balance
            underrepresented classes. Can be a list of floats, tuple, or torch.Tensor.
        ce_weight: Scaling coefficient lambda_ce for Cross-Entropy loss (default: 1.0).
        lovasz_weight: Scaling coefficient lambda_lovasz for Lovász-Softmax loss (default: 0.75).
        edge_weight: Base scaling coefficient lambda_edge for auxiliary edge loss (default: 0.5).
        ignore_index: Target label value designating void or unlabeled pixels (default: 255).
        edge_pos_weight: Positive class weight in Binary Cross-Entropy for edge pixels
            to counter extreme class sparsity (~5% boundary vs 95% background, default: 15.0).
    """

    def __init__(
        self,
        num_classes: int = 6,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        ce_weight: float = 1.0,
        lovasz_weight: float = 0.75,
        edge_weight: float = 0.5,
        ignore_index: int = 255,
        edge_pos_weight: float = 15.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.edge_weight = float(edge_weight)
        self.ignore_index = int(ignore_index)

        # Handle class weights for Cross-Entropy
        if class_weights is not None:
            weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights_tensor.numel() != num_classes:
                raise ValueError(
                    f"class_weights length ({weights_tensor.numel()}) must match num_classes ({num_classes})"
                )
            self.register_buffer("class_weights", weights_tensor)
            self.ce_loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=self.ignore_index,
            )
        else:
            self.register_buffer("class_weights", None)
            self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # Lovász-Softmax loss for direct mIoU surrogate optimization
        self.lovasz_loss_fn = LovaszSoftmax(
            classes="present",
            per_image=False,
            ignore_index=self.ignore_index,
        )

        # Register positive weight tensor for BCE edge loss as buffer for automatic device migration
        self.register_buffer(
            "edge_pos_weight",
            torch.tensor([edge_pos_weight], dtype=torch.float32),
        )

    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        edge_pred: Optional[torch.Tensor] = None,
        edge_target: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
        edge_decay_epoch: int = 50,
        **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """
        Computes multi-objective combined loss for the current forward step.

        Args:
            seg_pred: [B, C, H_pred, W_pred] semantic segmentation logits from the decoder.
            edge_pred: Optional [B, 1, H_edge, W_edge] or [B, H_edge, W_edge] edge prediction logits
                       from the auxiliary edge decoder. May be None during inference or if disabled.
            seg_target: [B, H_target, W_target] integer ground truth semantic mask.
            edge_target: Optional [B, 1, H_target, W_target] or [B, H_target, W_target] binary ground
                         truth edge map. If None, edge loss is set to 0.
            current_epoch: Current training epoch index (0-indexed). Used for edge weight decay.
            edge_decay_epoch: Epoch threshold after which lambda_edge is multiplied by 0.1
                              (default: 50).

        Returns:
            Dict containing:
                - 'total': Weighted sum of all active loss components.
                - 'ce': Unweighted Cross-Entropy loss tensor.
                - 'lovasz': Unweighted Lovász-Softmax loss tensor.
                - 'edge': Unweighted Edge BCE loss tensor.
        """
        target_size = seg_target.shape[-2:]

        # Upsample segmentation logits to target resolution if dimensions differ (e.g. 1/4 res decoder output)
        if seg_pred.shape[-2:] != target_size:
            seg_pred = F.interpolate(
                seg_pred,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        # Ensure ground truth is long dtype for CrossEntropyLoss
        seg_target = seg_target.long()

        # 1. Semantic Cross-Entropy Loss
        ce_loss = self.ce_loss_fn(seg_pred, seg_target)

        # 2. Semantic Lovász-Softmax Loss
        lovasz_loss = self.lovasz_loss_fn(seg_pred, seg_target)

        # 3. Boundary-Aware Edge Loss with Epoch-based Decay
        if current_epoch >= edge_decay_epoch:
            current_edge_weight = self.edge_weight * 0.1
        else:
            current_edge_weight = self.edge_weight

        if (
            edge_pred is not None
            and edge_target is not None
            and current_edge_weight > 0.0
        ):
            # Upsample edge logits to match edge target dimensions if needed
            edge_target_size = edge_target.shape[-2:]
            if edge_pred.shape[-2:] != edge_target_size:
                edge_pred = F.interpolate(
                    edge_pred,
                    size=edge_target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            # Squeeze channel dimension if present: (B, 1, H, W) -> (B, H, W)
            if edge_pred.dim() == 4 and edge_pred.size(1) == 1:
                edge_pred = edge_pred.squeeze(1)
            if edge_target.dim() == 4 and edge_target.size(1) == 1:
                edge_target = edge_target.squeeze(1)

            edge_target = edge_target.float()

            # Ignore boundary pixels corresponding to void/ignored semantic regions
            valid_edge_mask = (seg_target != self.ignore_index) & (
                edge_target != self.ignore_index
            )

            if valid_edge_mask.sum() > 0:
                edge_loss = F.binary_cross_entropy_with_logits(
                    edge_pred[valid_edge_mask],
                    edge_target[valid_edge_mask],
                    pos_weight=self.edge_pos_weight.to(edge_pred.device),
                )
            else:
                edge_loss = seg_pred.sum() * 0.0
        else:
            edge_loss = seg_pred.sum() * 0.0

        # Weighted Total Loss
        total_loss = (
            self.ce_weight * ce_loss
            + self.lovasz_weight * lovasz_loss
            + current_edge_weight * edge_loss
        )

        return {
            "total": total_loss,
            "total_loss": total_loss,
            "ce": ce_loss,
            "ce_loss": ce_loss,
            "lovasz": lovasz_loss,
            "lovasz_loss": lovasz_loss,
            "edge": edge_loss,
            "edge_loss": edge_loss,
        }
