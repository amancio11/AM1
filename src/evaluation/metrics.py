"""
metrics.py
==========
Evaluation metrics for glass segmentation and dirt estimation.

Segmentation metrics:
  - IoU (Intersection over Union / Jaccard Index)
  - Dice coefficient (F1 score for binary segmentation)
  - Pixel-level Precision, Recall, F1

Regression metrics:
  - MAE  (Mean Absolute Error)
  - MSE  (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)

All functions operate on CPU tensors.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute mean IoU over a batch.

    Parameters
    ----------
    pred   : (B, 1, H, W) — logits or probabilities
    target : (B, 1, H, W) — binary ground-truth [0,1]

    Returns
    -------
    scalar tensor
    """
    if pred.requires_grad:
        pred = pred.detach()
    prob = torch.sigmoid(pred) if pred.min() < 0 else pred
    binary = (prob > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (binary * target_bin).sum(dim=(1, 2, 3))
    union = binary.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Dice coefficient (same as F1 for binary segmentation)."""
    if pred.requires_grad:
        pred = pred.detach()
    prob = torch.sigmoid(pred) if pred.min() < 0 else pred
    binary = (prob > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (binary * target_bin).sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (
        binary.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) + smooth
    )
    return dice.mean()


def precision_recall_f1(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Precision, recall, and F1 for binary segmentation."""
    if pred.requires_grad:
        pred = pred.detach()
    prob = torch.sigmoid(pred) if pred.min() < 0 else pred
    binary = (prob > threshold).float().view(-1)
    target_bin = (target > threshold).float().view(-1)

    tp = (binary * target_bin).sum()
    fp = (binary * (1 - target_bin)).sum()
    fn = ((1 - binary) * target_bin).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics at once.

    Parameters
    ----------
    pred   : (B, 1, H, W) — logits or probabilities (or already thresholded)
    target : (B, 1, H, W) — binary ground-truth

    Returns
    -------
    dict of metric name → float
    """
    iou = iou_score(pred, target, threshold).item()
    dice = dice_score(pred, target, threshold).item()
    prf = precision_recall_f1(pred, target, threshold)
    return {
        "iou": iou,
        "dice": dice,
        **prf,
    }


# ---------------------------------------------------------------------------
# Regression / dirt estimation metrics
# ---------------------------------------------------------------------------

def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE, optionally computed only over glass pixels."""
    if pred.requires_grad:
        pred = pred.detach()
    diff_sq = (pred - target) ** 2
    if mask is not None:
        m = (mask > 0.5).float()
        n = m.sum().clamp(min=1.0)
        return (diff_sq * m).sum() / n
    return diff_sq.mean()


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MAE, optionally computed only over glass pixels."""
    if pred.requires_grad:
        pred = pred.detach()
    diff_abs = (pred - target).abs()
    if mask is not None:
        m = (mask > 0.5).float()
        n = m.sum().clamp(min=1.0)
        return (diff_abs * m).sum() / n
    return diff_abs.mean()


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    if pred.requires_grad:
        pred = pred.detach()
    mse_val = F.mse_loss(pred, target).item()
    if mse_val == 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse_val))


def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Compute SSIM between pred and target images.
    Both tensors should be (B, 1, H, W) in [0, 1].
    """
    if pred.requires_grad:
        pred = pred.detach()

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    window = _gaussian_window(window_size, sigma).to(pred.device, dtype=pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=1)
    mu2 = F.conv2d(target, window, padding=pad, groups=1)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu12 = mu1 * mu2

    sig1_sq = F.conv2d(pred * pred, window, padding=pad, groups=1) - mu1_sq
    sig2_sq = F.conv2d(target * target, window, padding=pad, groups=1) - mu2_sq
    sig12 = F.conv2d(pred * target, window, padding=pad, groups=1) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    )
    return ssim_map.mean().item()


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g.unsqueeze(1) * g.unsqueeze(0)
    return w.unsqueeze(0).unsqueeze(0)


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all regression metrics at once.

    Parameters
    ----------
    pred   : (B, 1, H, W) dirt heatmap in [0, 1]
    target : (B, 1, H, W) ground-truth dirt map in [0, 1]
    mask   : optional (B, 1, H, W) glass mask

    Returns
    -------
    dict
    """
    if pred.requires_grad:
        pred = pred.detach()

    mse_val = masked_mse(pred, target, mask).item()
    mae_val = masked_mae(pred, target, mask).item()
    rmse_val = math.sqrt(mse_val)
    psnr_val = psnr(pred, target)
    ssim_val = ssim_metric(pred.cpu(), target.cpu())

    return {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": rmse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
    }
