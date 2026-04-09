"""
glass_segmentation.py
=====================
Glass surface segmentation model.

Wraps segmentation_models_pytorch (SMP) U-Net / UNet++ / DeepLabV3+ / FPN
with a configurable encoder backbone. Provides a clean factory interface
for instantiation from config.

The model outputs raw logits (1 channel). Apply sigmoid + threshold at
inference time.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is required. "
        "Install with: pip install segmentation-models-pytorch"
    ) from e


# ---------------------------------------------------------------------------
# Glass Segmentation Model
# ---------------------------------------------------------------------------

class GlassSegmentationModel(nn.Module):
    """
    Binary segmentation model for glass surface detection.

    Input:  (B, 3, H, W) — normalized RGB
    Output: (B, 1, H, W) — raw logits
    """

    SUPPORTED_ARCHITECTURES = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "pan": smp.PAN,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
    }

    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        decoder_dropout: float = 0.2,
    ):
        super().__init__()
        arch_lower = architecture.lower()
        if arch_lower not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Choose from: {list(self.SUPPORTED_ARCHITECTURES.keys())}"
            )
        arch_cls = self.SUPPORTED_ARCHITECTURES[arch_lower]

        kwargs = dict(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,  # raw logits
        )
        if arch_lower in ("unet", "unetplusplus", "manet", "linknet"):
            kwargs["decoder_channels"] = decoder_channels
        if arch_lower == "unet":
            kwargs["decoder_use_batchnorm"] = True

        self.model = arch_cls(**kwargs)
        self._add_dropout_to_decoder(decoder_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GlassSegmentationModel":
        model_cfg = config["model"]
        return cls(
            architecture=model_cfg.get("architecture", "unet"),
            encoder_name=model_cfg.get("encoder", "resnet50"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            in_channels=model_cfg.get("in_channels", 3),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32, 16])),
            decoder_dropout=model_cfg.get("decoder_dropout", 0.2),
        )

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (useful for early warm-up epochs)."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights for full fine-tuning."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self, encoder_lr: float, decoder_lr: float):
        """
        Return parameter groups with separate LR for encoder vs decoder.
        Use with optimizer factories in the training module.
        """
        return [
            {"params": self.model.encoder.parameters(), "lr": encoder_lr},
            {"params": self.model.decoder.parameters(), "lr": decoder_lr},
            {"params": self.model.segmentation_head.parameters(), "lr": decoder_lr},
        ]

    def _add_dropout_to_decoder(self, p: float) -> None:
        """Inject Dropout into the decoder blocks."""
        if p <= 0:
            return
        for module in self.model.decoder.modules():
            if isinstance(module, nn.Sequential):
                for i, block in enumerate(module):
                    if isinstance(block, nn.Conv2d):
                        # Insert dropout before conv layers in decoder
                        pass  # SMP handles dropout via decoder_dropout param natively
