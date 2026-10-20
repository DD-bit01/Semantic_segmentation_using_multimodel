"""
Mix-Transformer (MiT-B2) Hierarchical Encoder Wrapper.

This module wraps the SegformerModel architecture from the Hugging Face
transformers library to serve as the hierarchical multi-scale backbone for
ASG-Net. It extracts features at 4 distinct spatial stages:
    - Stage 1: 1/4 resolution,  64 channels
    - Stage 2: 1/8 resolution,  128 channels
    - Stage 3: 1/16 resolution, 320 channels
    - Stage 4: 1/32 resolution, 512 channels

Supports both 3-channel (RGB) and 1-channel (NIR / multispectral) inputs by
adapting the initial patch embedding convolution weights.
"""

from typing import List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

logger = logging.getLogger(__name__)


class MiTEncoder(nn.Module):
    """
    Mix-Transformer (MiT-B2) encoder backbone for hierarchical feature extraction.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB, 1 for NIR).
        pretrained: Whether to initialize with pretrained weights.
        pretrained_model_name: Hugging Face hub identifier or local path for pretrained SegFormer.
        drop_path_rate: Stochastic depth dropout rate.
    """

    STAGE_CHANNELS: List[int] = [64, 128, 320, 512]
    STAGE_REDUCTIONS: List[int] = [4, 8, 16, 32]

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.pretrained_model_name = pretrained_model_name
        self.drop_path_rate = drop_path_rate

        # Initialize Segformer backbone
        self.model = self._init_backbone()

        # Adapt first patch embedding projection if in_channels != 3
        if self.in_channels != 3:
            self._adapt_input_channels()

    def _get_b2_fallback_config(self) -> SegformerConfig:
        """Construct standard MiT-B2 configuration for offline or scratch initialization."""
        return SegformerConfig(
            num_channels=3,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
            num_encoder_blocks=4,
            mlp_ratios=[4, 4, 4, 4],
            num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            drop_path_rate=self.drop_path_rate,
        )

    def _init_backbone(self) -> SegformerModel:
        """
        Instantiate SegformerModel from pretrained checkpoint with offline fallback.
        """
        if self.pretrained:
            try:
                logger.info(f"Loading pretrained weights: {self.pretrained_model_name}")
                model = SegformerModel.from_pretrained(self.pretrained_model_name)
                return model
            except Exception as err:
                logger.warning(
                    f"Could not download/load pretrained weights '{self.pretrained_model_name}': {err}. "
                    "Falling back to randomly initialized Segformer-B2 configuration."
                )

        # Fallback / scratch initialization
        config = self._get_b2_fallback_config()
        return SegformerModel(config)

    def _adapt_input_channels(self) -> None:
        """
        Adapt the first stage patch embedding layer to accommodate input channels != 3.
        For 1-channel NIR input, the original 3-channel weights are averaged across the
        channel dimension to preserve activation scale while retaining learned spatial filters.
        Supports both modern (model.stages[0]) and older (model.encoder.patch_embeddings[0]) transformers API.
        """
        patch_embed = None
        if hasattr(self.model, "stages") and len(self.model.stages) > 0 and hasattr(self.model.stages[0], "patch_embeddings"):
            patch_embed = self.model.stages[0].patch_embeddings
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "patch_embeddings"):
            patch_embed = self.model.encoder.patch_embeddings[0]
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "stages") and len(self.model.encoder.stages) > 0:
            patch_embed = self.model.encoder.stages[0].patch_embeddings
        else:
            for module in self.model.modules():
                if hasattr(module, "proj") and isinstance(getattr(module, "proj"), nn.Conv2d):
                    patch_embed = module
                    break

        if patch_embed is None or not hasattr(patch_embed, "proj"):
            logger.warning("Could not find patch_embeddings layer for direct kernel adaptation. Will replicate channels at runtime.")
            self._need_repeat_input = True
            return

        old_proj: nn.Conv2d = patch_embed.proj

        new_proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None),
        )

        with torch.no_grad():
            if self.in_channels == 1:
                # Average across 3 RGB channels: (out_ch, 3, k_h, k_w) -> (out_ch, 1, k_h, k_w)
                new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            else:
                # Arbitrary channel adaptation: replicate averaged weight
                avg_weight = old_proj.weight.mean(dim=1, keepdim=True)
                new_proj.weight.copy_(avg_weight.repeat(1, self.in_channels, 1, 1))

            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        patch_embed.proj = new_proj
        self._need_repeat_input = False
        logger.info(
            f"Adapted patch_embeddings.proj: in_channels {old_proj.in_channels} -> {self.in_channels}"
        )

    @staticmethod
    def _format_stage_feature(
        feat: torch.Tensor,
        stage_idx: int,
        input_h: int,
        input_w: int,
        expected_channels: int,
    ) -> torch.Tensor:
        """
        Ensure feature map is formatted as a 4D Tensor (B, C, H, W).
        HuggingFace Segformer may return hidden states formatted as (B, H*W, C)
        or (B, C, H, W) depending on library version.
        """
        if feat.ndim == 3:
            # Shape is (Batch, SequenceLength, Channels) -> (Batch, Channels, Height, Width)
            b, seq_len, c = feat.shape
            reduction = 2 ** (stage_idx + 2)  # 4, 8, 16, 32
            h = input_h // reduction
            w = input_w // reduction
            if h * w != seq_len:
                # Estimate square or aspect ratio if integer division rounded
                h = int(math.isqrt(seq_len))
                w = seq_len // h
            feat = feat.transpose(1, 2).reshape(b, c, h, w)
        elif feat.ndim == 4:
            # Shape is already (B, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected hidden state shape: {feat.shape}")

        return feat

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale feature maps from input image tensor.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            List of 4 feature maps:
                - Stage 0: (B, 64,  H/4,  W/4)
                - Stage 1: (B, 128, H/8,  W/8)
                - Stage 2: (B, 320, H/16, W/16)
                - Stage 3: (B, 512, H/32, W/32)
        """
        b, c, h, w = x.shape

        # Channel shape adaptation check
        if getattr(self, "_need_repeat_input", False) and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c != self.in_channels:
            if self.in_channels == 3 and c == 1:
                # Repeat 1-channel input across 3 channels
                x = x.repeat(1, 3, 1, 1)
            elif self.in_channels == 1 and c == 3:
                # Average RGB channels down to 1 channel
                x = x.mean(dim=1, keepdim=True)
            else:
                raise ValueError(
                    f"Expected input with {self.in_channels} channels, but received tensor with {c} channels."
                )

        # Forward pass through Segformer backbone
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )

        raw_hidden_states = outputs.hidden_states
        if raw_hidden_states is None:
            raise RuntimeError("SegformerModel did not return hidden_states. Ensure output_hidden_states=True.")

        # Segformer has 4 stages; format each stage to (B, C, H, W)
        formatted_features: List[torch.Tensor] = []
        for idx in range(4):
            stage_feat = raw_hidden_states[idx]
            feat_nchw = self._format_stage_feature(
                feat=stage_feat,
                stage_idx=idx,
                input_h=h,
                input_w=w,
                expected_channels=self.STAGE_CHANNELS[idx],
            )
            formatted_features.append(feat_nchw)

        return formatted_features
