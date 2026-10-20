"""
Auxiliary Edge / Boundary Decoder Head.

In remote sensing semantic segmentation (such as urban aerial imagery in ISPRS Potsdam),
boundaries between adjacent object classes (e.g. Building vs. Impervious Surfaces,
or Trees vs. Low Vegetation) suffer from blurring and confusion.

The auxiliary Edge Decoder supervises the model with boundary-aware gradients
extracted from high-resolution early encoder stages (Stage 1 at 1/4 res and Stage 2 at 1/8 res).
It outputs a binary boundary probability map at 1/4 resolution.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeDecoder(nn.Module):
    """
    Auxiliary Boundary Decoder Head for Edge Supervision.

    Args:
        stage1_channels: Channel dimension of Stage 1 features (default: 64).
        stage2_channels: Channel dimension of Stage 2 features (default: 128).
        edge_channels: Intermediate channel dimension for boundary processing (default: 64).
    """

    def __init__(
        self,
        stage1_channels: int = 64,
        stage2_channels: int = 128,
        edge_channels: int = 64,
    ) -> None:
        super().__init__()
        self.stage1_channels = stage1_channels
        self.stage2_channels = stage2_channels
        self.edge_channels = edge_channels

        # Stage 1 feature reduction: (B, stage1_channels, H/4, W/4) -> (B, edge_channels, H/4, W/4)
        self.conv_stage1 = nn.Sequential(
            nn.Conv2d(
                stage1_channels,
                edge_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 2 feature reduction: (B, stage2_channels, H/8, W/8) -> (B, edge_channels, H/8, W/8)
        self.conv_stage2 = nn.Sequential(
            nn.Conv2d(
                stage2_channels,
                edge_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True),
        )

        # Edge prediction head combining both stages: 2 * edge_channels -> 1 channel
        self.edge_head = nn.Sequential(
            nn.Conv2d(
                2 * edge_channels,
                edge_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stage1_feat: torch.Tensor,
        stage2_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for boundary probability estimation.

        Args:
            stage1_feat: High-resolution Stage 1 feature map (B, stage1_channels, H/4, W/4).
            stage2_feat: Stage 2 feature map (B, stage2_channels, H/8, W/8).

        Returns:
            Boundary probability map of shape (B, 1, H/4, W/4) with values in [0, 1].
        """
        # Feature reduction on Stage 1
        x1 = self.conv_stage1(stage1_feat)  # (B, edge_channels, H/4, W/4)

        # Feature reduction on Stage 2
        x2 = self.conv_stage2(stage2_feat)  # (B, edge_channels, H/8, W/8)

        # Upsample Stage 2 to match Stage 1 spatial resolution (1/4 resolution)
        target_size = (x1.shape[2], x1.shape[3])
        x2_up = F.interpolate(
            x2,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate along channel dimension: (B, 2 * edge_channels, H/4, W/4)
        x_cat = torch.cat([x1, x2_up], dim=1)

        # Predict boundary probability map
        edge_prob = self.edge_head(x_cat)  # (B, 1, H/4, W/4)

        return edge_prob
