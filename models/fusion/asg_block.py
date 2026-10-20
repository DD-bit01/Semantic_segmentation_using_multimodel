"""
Adaptive Spectral-Spatial Gated Fusion Block (ASG-Block).

This is the core contribution of ASG-Net for fusing RGB and multispectral (NIR)
features in remote sensing semantic segmentation. Unlike static fusion strategies
(such as simple summation or concatenation), ASG-Block adaptively computes
dynamic, content-dependent spectral channel weights and spatial confidence gates (alpha).

Processing pipeline:
    1. Cross-Modal Feature Rectification (CM-FRM): Aligns feature distributions across modalities.
    2. Spectral Channel Squeeze-and-Excitation: Dynamically recalibrates concatenated [RGB; MS] channels.
    3. Spatial Gating: Predicts per-pixel gate alpha in [0, 1] representing RGB vs. NIR trust.
    4. Convex Combination: Fused = alpha * F_rgb + (1 - alpha) * F_ms.
    5. Residual Connection: Adds 1x1 projection of the original unmodulated input pair.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from .cm_frm import CMFRM


class ASGBlock(nn.Module):
    """
    Adaptive Spectral-Spatial Gated Fusion Block.

    Args:
        channels: Number of channels in input feature maps (C).
        reduction_ratio: Bottleneck reduction ratio for channel attention (default: 16).
        use_cm_frm: Whether to apply Cross-Modal Feature Rectification before gating (default: True).
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        use_cm_frm: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.use_cm_frm = use_cm_frm

        # -------------------------------------------------------------
        # Step 1: Optional Cross-Modal Feature Rectification Module
        # -------------------------------------------------------------
        if self.use_cm_frm:
            self.cm_frm = CMFRM(channels=channels, reduction_ratio=reduction_ratio)
        else:
            self.cm_frm = None

        # -------------------------------------------------------------
        # Step 2: Channel Attention (Squeeze-and-Excitation on 2*C)
        # -------------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max((2 * channels) // reduction_ratio, 8)
        self.channel_se = nn.Sequential(
            nn.Linear(2 * channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 2 * channels, bias=False),
            nn.Sigmoid(),
        )

        # -------------------------------------------------------------
        # Step 3: Spatial Gating (1x1 Conv over 2*C channels -> 1 channel)
        # -------------------------------------------------------------
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # -------------------------------------------------------------
        # Step 5: Residual Connection (1x1 Conv on original concatenated input)
        # -------------------------------------------------------------
        self.residual_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Cache last spatial gate alpha for interpretability / visualization
        self.last_gate: Optional[torch.Tensor] = None

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_ms: torch.Tensor,
        return_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of ASG-Block.

        Args:
            f_rgb: RGB feature map of shape (B, C, H, W).
            f_ms: Multispectral / NIR feature map of shape (B, C, H, W).
            return_gate: If True, returns (output, alpha) tuple.

        Returns:
            Fused feature map of shape (B, C, H, W), or (fused, alpha) if return_gate=True.
        """
        if f_rgb.shape != f_ms.shape:
            raise ValueError(
                f"Input tensor shapes must match: f_rgb={f_rgb.shape}, f_ms={f_ms.shape}"
            )

        b, c, h, w = f_rgb.shape
        f_rgb_orig = f_rgb
        f_ms_orig = f_ms

        # -------------------------------------------------------------
        # Step 1: Feature Rectification
        # -------------------------------------------------------------
        if self.cm_frm is not None:
            f_rgb, f_ms = self.cm_frm(f_rgb, f_ms)

        # -------------------------------------------------------------
        # Step 2: Spectral Channel Attention on [f_rgb; f_ms]
        # -------------------------------------------------------------
        cat_ch = torch.cat([f_rgb, f_ms], dim=1)  # (B, 2C, H, W)
        pooled = self.gap(cat_ch).flatten(1)     # (B, 2C)
        ch_weights = self.channel_se(pooled).view(b, 2 * c, 1, 1)  # (B, 2C, 1, 1)

        # Split recalibration weights back to C channels each
        w_rgb, w_ms = torch.split(ch_weights, self.channels, dim=1)
        f_rgb_att = f_rgb * w_rgb
        f_ms_att = f_ms * w_ms

        # -------------------------------------------------------------
        # Step 3: Spatial Gating
        # -------------------------------------------------------------
        cat_spatial = torch.cat([f_rgb_att, f_ms_att], dim=1)  # (B, 2C, H, W)
        alpha = self.spatial_gate(cat_spatial)                 # (B, 1, H, W)
        self.last_gate = alpha.detach()

        # -------------------------------------------------------------
        # Step 4: Fused Convex Combination
        # -------------------------------------------------------------
        # alpha weights RGB trust; (1 - alpha) weights NIR trust
        fused = alpha * f_rgb_att + (1.0 - alpha) * f_ms_att

        # -------------------------------------------------------------
        # Step 5: Residual Connection
        # -------------------------------------------------------------
        cat_orig = torch.cat([f_rgb_orig, f_ms_orig], dim=1)
        residual = self.residual_conv(cat_orig)
        output = fused + residual

        if return_gate:
            return output, alpha
        return output
