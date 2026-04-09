"""
losses.py
=========
Custom loss functions for glass segmentation and dirt estimation.

Includes:
  - DiceLoss:          soft Dice for binary segmentation
  - FocalLoss:         focal loss for class imbalance
  - CombinedSegLoss:   BCE + Dice + Focal (weighted sum)
  - SSIMLoss:          structural similarity for regression
  - CombinedRegLoss:   MSE + MAE + SSIM (weighted sum)
  - MultiTaskLoss:     uncertainty-weighted multi-task loss (Kendall 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Segmentation losses
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    L_dice = 1 - (2 * sum(p*t) + smooth) / (sum(p) + sum(t) + smooth)
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.  FL = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            prob = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            prob = pred

        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedSegLoss(nn.Module):
    """
    Weighted combination of BCE + Dice + Focal for segmentation.
    """

    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.4,
        focal_weight: float = 0.2,
        pos_weight: Optional[float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss(from_logits=True)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, from_logits=True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        l_bce = self.bce(pred, target)
        l_dice = self.dice(pred, target)
        l_focal = self.focal(pred, target)
        total = (
            self.bce_weight * l_bce
            + self.dice_weight * l_dice
            + self.focal_weight * l_focal
        )
        return {
            "loss": total,
            "bce": l_bce.detach(),
            "dice": l_dice.detach(),
            "focal": l_focal.detach(),
        }


# ---------------------------------------------------------------------------
# Regression losses
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss: L = 1 - SSIM(pred, target)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self._window = self._gaussian_window(window_size, sigma)

    def _gaussian_window(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        window = self._window.to(pred.device, dtype=pred.dtype)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=1)
        mu2 = F.conv2d(target, window, padding=pad, groups=1)
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=1) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if mask is not None:
            # Only compute SSIM over glass pixels
            ssim_map = ssim_map * (mask > 0.5).float()
            denom = (mask > 0.5).float().sum().clamp(min=1.0)
            return 1.0 - ssim_map.sum() / denom
        return 1.0 - ssim_map.mean()


class CombinedRegLoss(nn.Module):
    """
    Weighted MSE + MAE + SSIM regression loss for dirt estimation.
    Optionally computes loss only over glass pixels.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        mae_weight: float = 0.3,
        ssim_weight: float = 0.2,
        mask_loss: bool = True,
        ssim_window_size: int = 11,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.mask_loss = mask_loss
        self.ssim = SSIMLoss(window_size=ssim_window_size)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        glass_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.mask_loss and glass_mask is not None:
            mask_bin = (glass_mask > 0.5).float()
            n = mask_bin.sum().clamp(min=1.0)
            diff = pred - target
            l_mse = (mask_bin * diff ** 2).sum() / n
            l_mae = (mask_bin * diff.abs()).sum() / n
            l_ssim = self.ssim(pred, target, mask=glass_mask)
        else:
            l_mse = F.mse_loss(pred, target)
            l_mae = F.l1_loss(pred, target)
            l_ssim = self.ssim(pred, target)

        total = (
            self.mse_weight * l_mse
            + self.mae_weight * l_mae
            + self.ssim_weight * l_ssim
        )
        return {
            "loss": total,
            "mse": l_mse.detach(),
            "mae": l_mae.detach(),
            "ssim": l_ssim.detach(),
        }


# ---------------------------------------------------------------------------
# Multi-task loss (Kendall et al. 2018 — Uncertainty weighting)
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Learnable uncertainty-weighted multi-task loss.

    L = sum_i { (1 / (2 * sigma_i^2)) * L_i + log(sigma_i) }

    For segmentation: sigma_seg  (learned log variance)
    For regression:   sigma_reg  (learned log variance)

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses
                for Scene Geometry and Semantics", Kendall et al. 2018.
    """

    def __init__(
        self,
        seg_loss: nn.Module,
        reg_loss: nn.Module,
        task_balancing: str = "uncertainty",
        seg_task_weight: float = 1.0,
        reg_task_weight: float = 1.0,
    ):
        super().__init__()
        self.seg_loss = seg_loss
        self.reg_loss = reg_loss
        self.task_balancing = task_balancing

        # Learnable log sigma^2 (uncertainty) for each task
        if task_balancing == "uncertainty":
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_reg = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("seg_weight", torch.tensor(seg_task_weight))
            self.register_buffer("reg_weight", torch.tensor(reg_task_weight))

    def forward(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        reg_pred: torch.Tensor,
        reg_target: torch.Tensor,
        glass_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        seg_losses = self.seg_loss(seg_pred, seg_target)
        reg_losses = self.reg_loss(reg_pred, reg_target, glass_mask)

        l_seg = seg_losses["loss"]
        l_reg = reg_losses["loss"]

        if self.task_balancing == "uncertainty":
            precision_seg = torch.exp(-self.log_var_seg)
            precision_reg = torch.exp(-self.log_var_reg)
            total = (
                precision_seg * l_seg + self.log_var_seg
                + precision_reg * l_reg + self.log_var_reg
            )
        else:
            total = self.seg_weight * l_seg + self.reg_weight * l_reg

        return {
            "loss": total,
            "seg_loss": l_seg.detach(),
            "reg_loss": l_reg.detach(),
            **{f"seg_{k}": v for k, v in seg_losses.items() if k != "loss"},
            **{f"reg_{k}": v for k, v in reg_losses.items() if k != "loss"},
        }
