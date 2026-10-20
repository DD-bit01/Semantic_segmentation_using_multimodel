"""
Cross-Modal Feature Rectification Module (CM-FRM).

Inspired by CMX (Cross-Modal Fusion for RGB-X Semantic Segmentation), this module
rectifies and calibrates feature representations across complementary modalities
(RGB and multispectral/NIR) prior to fused integration.

Modality discrepancy and sensor noise often lead to misaligned or sub-optimal feature
distributions. CM-FRM applies:
1. Channel Rectification: Global average pooling of the complementary modality
   generates channel-wise recalibration vectors, highlighting informative spectral channels.
2. Spatial Rectification: Depthwise separable convolutions extract cross-modal spatial
   attention maps, highlighting salient structural regions.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CMFRM(nn.Module):
    """
    Cross-Modal Feature Rectification Module (CM-FRM).

    Args:
        channels: Number of channels in the feature maps (C).
        reduction_ratio: Reduction ratio for the channel excitation MLP bottleneck.
        residual: Whether to add the original features via identity residual connection.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.residual = residual

        mid_channels = max(channels // reduction_ratio, 8)

        # Global Average Pooling for Channel Attention
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Channel Rectification: MS features guide RGB channels
        self.channel_ms2rgb = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

        # Channel Rectification: RGB features guide MS channels
        self.channel_rgb2ms = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Rectification: MS features guide RGB spatial attention
        self.spatial_ms2rgb = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Spatial Rectification: RGB features guide MS spatial attention
        self.spatial_rgb2ms = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_ms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Cross-Modal Feature Rectification.

        Args:
            f_rgb: RGB feature map of shape (B, C, H, W).
            f_ms: Multispectral / NIR feature map of shape (B, C, H, W).

        Returns:
            Tuple of rectified feature maps (rect_rgb, rect_ms) of shape (B, C, H, W).
        """
        if f_rgb.shape != f_ms.shape:
            raise ValueError(
                f"Feature map shapes must match. Got f_rgb: {f_rgb.shape}, f_ms: {f_ms.shape}"
            )

        b, c, h, w = f_rgb.shape

        # -------------------------------------------------------------
        # 1. Cross-Modal Channel Rectification
        # -------------------------------------------------------------
        # MS statistics calibrate RGB channels
        v_ms = self.gap(f_ms).flatten(1)  # (B, C)
        w_ch_rgb = self.channel_ms2rgb(v_ms).view(b, c, 1, 1)  # (B, C, 1, 1)

        # RGB statistics calibrate MS channels
        v_rgb = self.gap(f_rgb).flatten(1)  # (B, C)
        w_ch_ms = self.channel_rgb2ms(v_rgb).view(b, c, 1, 1)  # (B, C, 1, 1)

        # -------------------------------------------------------------
        # 2. Cross-Modal Spatial Rectification
        # -------------------------------------------------------------
        # MS spatial features guide RGB spatial attention
        w_sp_rgb = self.spatial_ms2rgb(f_ms)  # (B, 1, H, W)

        # RGB spatial features guide MS spatial attention
        w_sp_ms = self.spatial_rgb2ms(f_rgb)  # (B, 1, H, W)

        # -------------------------------------------------------------
        # 3. Combine Modulations with Residual Connection
        # -------------------------------------------------------------
        # Modulate features with joint channel and spatial attention weights
        delta_rgb = f_rgb * w_ch_rgb * w_sp_rgb
        delta_ms = f_ms * w_ch_ms * w_sp_ms

        if self.residual:
            rect_rgb = f_rgb + delta_rgb
            rect_ms = f_ms + delta_ms
        else:
            rect_rgb = delta_rgb
            rect_ms = delta_ms

        return rect_rgb, rect_ms
