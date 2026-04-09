"""
multitask_model.py
==================
Multi-task model with shared encoder and dual decoder heads:
  - Head 1: Glass surface segmentation (binary, logits output)
  - Head 2: Dirt level estimation (regression, sigmoid output)

Architecture
------------
  Shared Encoder (ResNet / EfficientNet / MiT)
       ↓
  Feature Pyramid Network (FPN) — multi-scale features
       ↓                ↓
  Seg Decoder     Reg Decoder
       ↓                ↓
  Seg Head (1-ch)  Reg Head (1-ch, sigmoid)

This design allows shared low-level features while learning
task-specific decoders independently.

Reference: "Attention U-Net: Learning Where to Look for the Pancreas"
           + "Rethinking ImageNet Pre-training" for encoder design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

try:
    import segmentation_models_pytorch as smp
    import timm
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch and timm are required. "
        "Install with: pip install segmentation-models-pytorch timm"
    ) from e


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Single up-sampling block: bilinear upsample + double conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Task-specific decoder
# ---------------------------------------------------------------------------

class TaskDecoder(nn.Module):
    """U-Net style decoder for a single task."""

    def __init__(
        self,
        encoder_channels: List[int],     # [enc_stage0, enc_stage1, ...]
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        # encoder_channels is reversed (deep → shallow)
        enc_ch = list(reversed(encoder_channels))
        blocks = []
        in_ch = enc_ch[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(
        self, features: List[torch.Tensor]
    ) -> torch.Tensor:
        # features: [bottleneck, skip_4, skip_3, skip_2, skip_1, skip_0]
        x = features[0]
        skips = features[1:]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Multi-task Model
# ---------------------------------------------------------------------------

class MultitaskFacadeModel(nn.Module):
    """
    Shared-encoder multi-task model for glass segmentation + dirt estimation.

    Returns
    -------
    On forward(): tuple (seg_logits, dirt_heatmap)
      seg_logits   : (B, 1, H, W) — raw logits for segmentation
      dirt_heatmap : (B, 1, H, W) — sigmoid-activated dirt map ∈ [0, 1]
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        seg_decoder_channels: tuple = (256, 128, 64, 32, 16),
        reg_decoder_channels: tuple = (256, 128, 64, 32, 16),
        seg_dropout: float = 0.2,
        reg_dropout: float = 0.3,
    ):
        super().__init__()

        # Build a dummy SMP UNet to extract the encoder and check channels
        _ref = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.encoder = _ref.encoder
        encoder_channels = list(self.encoder.out_channels)  # e.g. [3, 64, 256, 512, 1024, 2048]

        # Shared encoder is done; delete the ref model
        del _ref

        # Two independent decoders
        self.seg_decoder = TaskDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=seg_decoder_channels,
            dropout=seg_dropout,
        )
        self.reg_decoder = TaskDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=reg_decoder_channels,
            dropout=reg_dropout,
        )

        # Heads
        seg_last_ch = seg_decoder_channels[-1]
        reg_last_ch = reg_decoder_channels[-1]

        self.seg_head = nn.Sequential(
            nn.Conv2d(seg_last_ch, seg_last_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(seg_last_ch // 2, 1, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(reg_last_ch, reg_last_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reg_last_ch // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Uncertainty log-variances (used if MultiTaskLoss with "uncertainty")
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        features = self.encoder(x)
        # features is [input, stage1, stage2, stage3, stage4, stage5]
        features_reversed = list(reversed(features))

        # Decode for each task
        seg_feat = self.seg_decoder(features_reversed)
        reg_feat = self.reg_decoder(features_reversed)

        # Upsample to input resolution if needed
        h, w = x.shape[-2:]
        seg_feat = F.interpolate(seg_feat, size=(h, w), mode="bilinear", align_corners=False)
        reg_feat = F.interpolate(reg_feat, size=(h, w), mode="bilinear", align_corners=False)

        seg_logits = self.seg_head(seg_feat)
        dirt_heatmap = self.reg_head(reg_feat)

        return seg_logits, dirt_heatmap

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultitaskFacadeModel":
        model_cfg = config["model"]
        return cls(
            encoder_name=model_cfg.get("encoder", "resnet50"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            in_channels=model_cfg.get("in_channels", 3),
            seg_decoder_channels=tuple(model_cfg.get("seg_decoder_channels", [256, 128, 64, 32, 16])),
            reg_decoder_channels=tuple(model_cfg.get("reg_decoder_channels", [256, 128, 64, 32, 16])),
            seg_dropout=model_cfg.get("seg_dropout", 0.2),
            reg_dropout=model_cfg.get("reg_dropout", 0.3),
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> List[Dict]:
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr},
            {"params": self.seg_decoder.parameters(), "lr": decoder_lr},
            {"params": self.reg_decoder.parameters(), "lr": decoder_lr},
            {"params": self.seg_head.parameters(), "lr": decoder_lr},
            {"params": self.reg_head.parameters(), "lr": decoder_lr},
            {"params": [self.log_var_seg, self.log_var_reg], "lr": decoder_lr},
        ]
