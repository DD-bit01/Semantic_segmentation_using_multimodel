"""
ASG-Net: Adaptive Spectral-Spatial Gated Fusion Network for RGB-Multispectral Semantic Segmentation.

Main architecture integrating:
1. Dual MiT-B2 hierarchical encoders (RGB 3-channel + NIR 1-channel).
2. Adaptive Spectral-Spatial Gated Fusion (ASG-Block) with CM-FRM at each encoder stage.
3. SegFormer All-MLP Decoder for multi-scale semantic segmentation prediction.
4. Auxiliary Boundary/Edge Decoder for boundary-aware multi-task supervision.
5. Flexible ablation support: 'asg' (full), 'add' (summation), 'concat' (convolutional), 'none' (RGB-only).
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import MiTEncoder
from .fusion import ASGBlock
from .decoders import SegFormerHead, EdgeDecoder

logger = logging.getLogger(__name__)


class ASGNet(nn.Module):
    """
    Adaptive Spectral-Spatial Gated Fusion Network (ASG-Net).

    Args:
        num_classes: Number of semantic classes (default: 6 for Potsdam).
        encoder_channels: Output channels for the 4 encoder stages (default: [64, 128, 320, 512]).
        decoder_channels: Channel dimension for unified MLP decoder (default: 256).
        pretrained: Whether to initialize encoders from pretrained SegFormer weights.
        pretrained_model: Hugging Face model identifier for weights.
        fusion_type: Fusion strategy ('asg', 'add', 'concat', 'none').
        use_cm_frm: Whether to enable Cross-Modal Feature Rectification in ASG blocks.
        gate_reduction: Bottleneck channel reduction ratio for gating in ASG blocks.
        use_edge_decoder: Whether to enable the auxiliary boundary/edge decoder.
        edge_channels: Intermediate channels for the edge decoder.
        drop_path_rate: Stochastic depth rate for encoder transformer blocks.
    """

    SUPPORTED_FUSION_TYPES: Sequence[str] = ("asg", "add", "concat", "none")

    def __init__(
        self,
        config: Optional[Union[dict, str, Path, int]] = None,
        num_classes: int = 6,
        encoder_channels: Sequence[int] = (64, 128, 320, 512),
        decoder_channels: int = 256,
        pretrained: bool = True,
        pretrained_model: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        fusion_type: str = "asg",
        use_cm_frm: bool = True,
        gate_reduction: int = 16,
        use_edge_decoder: bool = True,
        edge_channels: int = 64,
        drop_path_rate: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__()
        # If config is passed as dict or file path, extract model parameters
        if isinstance(config, (dict, str, Path)):
            if isinstance(config, (str, Path)):
                with open(config, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
            else:
                cfg = config
            model_cfg = cfg.get("model", {})
            fusion_cfg = model_cfg.get("fusion", {})
            edge_cfg = model_cfg.get("edge_decoder", {})

            num_classes = model_cfg.get("num_classes", 6)
            encoder_channels = model_cfg.get("encoder_channels", [64, 128, 320, 512])
            decoder_channels = model_cfg.get("decoder_channels", 256)
            pretrained = model_cfg.get("pretrained", True)
            pretrained_model = model_cfg.get(
                "pretrained_model", "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            fusion_type = fusion_cfg.get("type", "asg")
            use_cm_frm = fusion_cfg.get("use_cm_frm", True)
            gate_reduction = fusion_cfg.get("gate_reduction", 16)
            use_edge_decoder = edge_cfg.get("enabled", True)
            edge_channels = edge_cfg.get("channels", 64)
            drop_path_rate = model_cfg.get("drop_path_rate", 0.1)
        elif isinstance(config, int):
            num_classes = config

        self.num_classes = int(num_classes)
        self.encoder_channels = [int(c) for c in encoder_channels]
        self.decoder_channels = int(decoder_channels)
        self.pretrained = bool(pretrained)
        self.pretrained_model = str(pretrained_model)
        self.fusion_type = str(fusion_type).lower()
        self.use_cm_frm = bool(use_cm_frm)
        self.gate_reduction = int(gate_reduction)
        self.use_edge_decoder = bool(use_edge_decoder)
        self.edge_channels = int(edge_channels)
        self.drop_path_rate = float(drop_path_rate)

        if self.fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type '{self.fusion_type}'. "
                f"Must be one of {self.SUPPORTED_FUSION_TYPES}."
            )

        # -------------------------------------------------------------
        # 1. Dual MiT-B2 Encoders
        # -------------------------------------------------------------
        # RGB Encoder (3 input channels)
        self.encoder_rgb = MiTEncoder(
            in_channels=3,
            pretrained=pretrained,
            pretrained_model_name=pretrained_model,
            drop_path_rate=drop_path_rate,
        )

        # Multispectral / NIR Encoder (1 input channel)
        if self.fusion_type != "none":
            self.encoder_ms: Optional[MiTEncoder] = MiTEncoder(
                in_channels=1,
                pretrained=pretrained,
                pretrained_model_name=pretrained_model,
                drop_path_rate=drop_path_rate,
            )
        else:
            self.encoder_ms = None

        # -------------------------------------------------------------
        # 2. Stage-Wise Fusion Modules
        # -------------------------------------------------------------
        if self.fusion_type == "asg":
            # Proposed Adaptive Spectral-Spatial Gated Fusion Block
            self.fusion_blocks = nn.ModuleList(
                [
                    ASGBlock(
                        channels=ch,
                        reduction_ratio=gate_reduction,
                        use_cm_frm=use_cm_frm,
                    )
                    for ch in self.encoder_channels
                ]
            )
        elif self.fusion_type == "concat":
            # Concat + 1x1 Conv baseline
            self.concat_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(ch),
                        nn.ReLU(inplace=True),
                    )
                    for ch in self.encoder_channels
                ]
            )
        else:
            # 'add' or 'none' require no dedicated trainable fusion layers
            self.fusion_blocks = None
            self.concat_convs = None

        # -------------------------------------------------------------
        # 3. SegFormer All-MLP Semantic Decoder
        # -------------------------------------------------------------
        self.decoder = SegFormerHead(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

        # -------------------------------------------------------------
        # 4. Auxiliary Boundary/Edge Decoder
        # -------------------------------------------------------------
        if self.use_edge_decoder:
            self.edge_decoder: Optional[EdgeDecoder] = EdgeDecoder(
                stage1_channels=self.encoder_channels[0],
                stage2_channels=self.encoder_channels[1],
                edge_channels=edge_channels,
            )
        else:
            self.edge_decoder = None

    @classmethod
    def from_config(cls, config: Union[str, Path, dict]) -> "ASGNet":
        """
        Instantiate ASGNet directly from a YAML config file or dictionary.

        Args:
            config: Path to config.yaml or loaded dictionary.

        Returns:
            Configured ASGNet instance.
        """
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise TypeError(f"Expected path or dict for config, got {type(config)}")

        model_cfg = cfg.get("model", {})
        fusion_cfg = model_cfg.get("fusion", {})
        edge_cfg = model_cfg.get("edge_decoder", {})

        return cls(
            num_classes=model_cfg.get("num_classes", 6),
            encoder_channels=model_cfg.get("encoder_channels", [64, 128, 320, 512]),
            decoder_channels=model_cfg.get("decoder_channels", 256),
            pretrained=model_cfg.get("pretrained", True),
            pretrained_model=model_cfg.get(
                "pretrained_model", "nvidia/segformer-b2-finetuned-ade-512-512"
            ),
            fusion_type=fusion_cfg.get("type", "asg"),
            use_cm_frm=fusion_cfg.get("use_cm_frm", True),
            gate_reduction=fusion_cfg.get("gate_reduction", 16),
            use_edge_decoder=edge_cfg.get("enabled", True),
            edge_channels=edge_cfg.get("channels", 64),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        nir: Optional[torch.Tensor] = None,
        return_edge: Optional[bool] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass of ASG-Net.

        Args:
            rgb: RGB image tensor of shape (B, 3, H, W).
            nir: NIR / multispectral image tensor of shape (B, 1, H, W).
                 Required unless fusion_type is 'none'.
            return_edge: Whether to compute and return edge predictions.
                         Defaults to True during training if edge_decoder is enabled.

        Returns:
            Dictionary containing:
                - 'seg_pred': Semantic segmentation logits at 1/4 resolution (B, num_classes, H/4, W/4).
                - 'edge_pred': Edge probability map at 1/4 resolution (B, 1, H/4, W/4) or None.
        """
        # 1. Extract hierarchical features from RGB encoder
        rgb_features: List[torch.Tensor] = self.encoder_rgb(rgb)

        # 2. Extract hierarchical features from NIR encoder if applicable
        if self.fusion_type != "none":
            if nir is None:
                # Fallback: estimate 1-channel NIR via RGB luminance if missing
                nir = rgb.mean(dim=1, keepdim=True)
            if self.encoder_ms is None:
                raise RuntimeError("Multispectral encoder not initialized.")
            ms_features: List[torch.Tensor] = self.encoder_ms(nir)

        # 3. Fuse multi-scale features according to fusion_type
        if self.fusion_type == "asg":
            if self.fusion_blocks is None:
                raise RuntimeError("ASG fusion blocks not initialized.")
            fused_features: List[torch.Tensor] = [
                self.fusion_blocks[i](rgb_features[i], ms_features[i])
                for i in range(4)
            ]
        elif self.fusion_type == "concat":
            if self.concat_convs is None:
                raise RuntimeError("Concat convolution layers not initialized.")
            fused_features = [
                self.concat_convs[i](
                    torch.cat([rgb_features[i], ms_features[i]], dim=1)
                )
                for i in range(4)
            ]
        elif self.fusion_type == "add":
            fused_features = [
                rgb_features[i] + ms_features[i] for i in range(4)
            ]
        elif self.fusion_type == "none":
            # RGB-only baseline
            fused_features = rgb_features
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # 4. Predict semantic class logits via SegFormer All-MLP Decoder
        seg_pred = self.decoder(fused_features)  # (B, num_classes, H/4, W/4)

        # 5. Predict boundary map via auxiliary Edge Decoder
        compute_edge = (
            return_edge
            if return_edge is not None
            else (self.training and self.use_edge_decoder)
        )

        edge_pred: Optional[torch.Tensor] = None
        if compute_edge and self.edge_decoder is not None:
            # Stage 0 (64ch) and Stage 1 (128ch) fused representations
            edge_pred = self.edge_decoder(fused_features[0], fused_features[1])

        return {
            "seg_pred": seg_pred,
            "edge_pred": edge_pred,
        }

    def predict_full_res(
        self,
        rgb: torch.Tensor,
        nir: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference helper predicting class logits upsampled to full input resolution.

        Args:
            rgb: RGB image tensor of shape (B, 3, H, W).
            nir: NIR image tensor of shape (B, 1, H, W).

        Returns:
            Class logits of shape (B, num_classes, H, W).
        """
        outputs = self.forward(rgb=rgb, nir=nir, return_edge=False)
        seg_pred = outputs["seg_pred"]
        if seg_pred is None:
            raise RuntimeError("Decoder returned None for seg_pred.")

        target_size = (rgb.shape[2], rgb.shape[3])
        return F.interpolate(
            seg_pred,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def get_last_gates(self) -> List[Optional[torch.Tensor]]:
        """
        Retrieve cached spatial gating maps (alpha) from all ASG stages.
        Useful for visualization, qualitative analysis, and explainability.

        Returns:
            List of 4 tensors of shape (B, 1, H_stage, W_stage) or None if not 'asg' mode.
        """
        if self.fusion_type != "asg" or self.fusion_blocks is None:
            return [None, None, None, None]

        return [block.last_gate for block in self.fusion_blocks]
