"""
dirt_estimation.py
==================
Dirt level estimation model — dense regression for pixel-level dirt heatmap.

Architecture: U-Net with sigmoid output head (output ∈ [0,1]).
The model can optionally receive a glass mask as an additional input channel
to focus on glass regions only.

Input:  (B, C, H, W) — C=3 (RGB) or C=4 (RGB + glass mask)
Output: (B, 1, H, W) — dirt intensity map ∈ [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is required. "
        "Install with: pip install segmentation-models-pytorch"
    ) from e


class DirtEstimationModel(nn.Module):
    """
    Dense regression model for dirt level estimation on glass surfaces.

    Outputs a spatial heatmap matching input resolution where each pixel
    value ∈ [0, 1] represents the local dirt intensity.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        decoder_dropout: float = 0.3,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation="sigmoid",       # bounded output [0, 1]
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        glass_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image : (B, 3, H, W) normalized RGB
        glass_mask : (B, 1, H, W) optional glass mask to concatenate

        Returns
        -------
        (B, 1, H, W) dirt heatmap in [0, 1]
        """
        if glass_mask is not None and self.model.encoder.in_channels == 4:
            x = torch.cat([image, glass_mask], dim=1)
        else:
            x = image
        return self.model(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DirtEstimationModel":
        model_cfg = config["model"]
        use_mask_channel = config.get("dataset", {}).get("use_masked_input", False)
        in_channels = 4 if use_mask_channel else 3
        return cls(
            encoder_name=model_cfg.get("encoder", "efficientnet-b4"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            in_channels=in_channels,
            decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32, 16])),
            decoder_dropout=model_cfg.get("decoder_dropout", 0.3),
        )

    def freeze_encoder(self) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self, encoder_lr: float, decoder_lr: float):
        return [
            {"params": self.model.encoder.parameters(), "lr": encoder_lr},
            {"params": self.model.decoder.parameters(), "lr": decoder_lr},
            {"params": self.model.segmentation_head.parameters(), "lr": decoder_lr},
        ]
