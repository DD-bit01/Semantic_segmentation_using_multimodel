"""
SegFormer All-MLP Decoder Head.

Follows the lightweight All-MLP decoder design from SegFormer (Xie et al., NeurIPS 2021).
Unlike complex decoders with heavy dilated convolutions, SegFormer's All-MLP decoder:
1. Projects multi-scale hierarchical feature maps to a unified channel dimension (e.g. 256)
   using individual linear projections.
2. Upsamples all stages to 1/4 resolution via bilinear interpolation.
3. Concatenates the unified features across all stages.
4. Fuses them through an MLP with dropout to predict final class logits at 1/4 resolution.
"""

from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerHead(nn.Module):
    """
    SegFormer All-MLP Decoder Head.

    Args:
        encoder_channels: List of channel dimensions from each encoder stage
                          (default: [64, 128, 320, 512] for MiT-B2).
        decoder_channels: Unified projection dimension for all stages (default: 256).
        num_classes: Number of semantic segmentation target classes (default: 6 for Potsdam).
        dropout_ratio: Dropout probability before the classification head (default: 0.1).
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 128, 320, 512),
        decoder_channels: int = 256,
        num_classes: int = 6,
        dropout_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio

        # Linear projection layers for each encoder scale
        self.linear_projections = nn.ModuleList(
            [nn.Linear(c, decoder_channels) for c in self.encoder_channels]
        )

        # Fusion MLP to combine multi-level concatenated representations
        total_channels = len(self.encoder_channels) * decoder_channels
        self.mlp = nn.Sequential(
            nn.Linear(total_channels, decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(decoder_channels, num_classes),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of SegFormer All-MLP Decoder.

        Args:
            features: List of 4 feature maps from multi-scale stages:
                      - Stage 0: (B, C0, H/4,  W/4)
                      - Stage 1: (B, C1, H/8,  W/8)
                      - Stage 2: (B, C2, H/16, W/16)
                      - Stage 3: (B, C3, H/32, W/32)

        Returns:
            Logits tensor of shape (B, num_classes, H/4, W/4).
        """
        if len(features) != len(self.encoder_channels):
            raise ValueError(
                f"Expected {len(self.encoder_channels)} stage feature maps, but got {len(features)}."
            )

        # Stage 0 resolution defines the 1/4 target resolution
        target_h, target_w = features[0].shape[2], features[0].shape[3]
        batch_size = features[0].shape[0]

        projected_stages: List[torch.Tensor] = []

        for idx, feat in enumerate(features):
            b, c, h, w = feat.shape

            # Treat spatial dimensions as sequence tokens: (B, C, H, W) -> (B, H*W, C)
            feat_flat = feat.flatten(2).transpose(1, 2)

            # Apply stage-specific linear projection: (B, H*W, C) -> (B, H*W, D)
            proj_flat = self.linear_projections[idx](feat_flat)

            # Reshape back to spatial map: (B, D, H, W)
            proj = proj_flat.transpose(1, 2).reshape(b, self.decoder_channels, h, w)

            # Upsample to 1/4 resolution (Stage 0 size)
            if (h, w) != (target_h, target_w):
                proj = F.interpolate(
                    proj,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

            projected_stages.append(proj)

        # Concatenate along channel dimension: (B, num_stages * D, H/4, W/4)
        concat_features = torch.cat(projected_stages, dim=1)

        # Flatten spatial dimensions for MLP processing: (B, H/4*W/4, num_stages * D)
        concat_flat = concat_features.flatten(2).transpose(1, 2)

        # Apply MLP: (B, H/4*W/4, num_classes)
        logits_flat = self.mlp(concat_flat)

        # Reshape to 4D logits tensor: (B, num_classes, H/4, W/4)
        logits = logits_flat.transpose(1, 2).reshape(
            batch_size, self.num_classes, target_h, target_w
        )

        return logits
