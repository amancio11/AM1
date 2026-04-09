"""
cleanliness_score.py
====================
Cleanliness score computation from model outputs.

The cleanliness score is defined as:

    cleanliness = 1 - mean(dirt_map[glass_mask > threshold])

Where:
  - dirt_map is the model's predicted dirt heatmap (∈ [0,1])
  - glass_mask is the binary glass segmentation mask
  - The mean is taken only over pixels identified as glass

A score of 1.0 = perfectly clean glass
A score of 0.0 = completely dirty glass

Additional utilities:
  - Per-window cleanliness score (region-based)
  - Score percentile bucketing (A/B/C/D grade)
  - Spatial cleanliness map generation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Score data structures
# ---------------------------------------------------------------------------

@dataclass
class CleanlinessResult:
    """Full cleanliness analysis result for a single image."""
    overall_score: float            # 0.0 (dirty) → 1.0 (clean)
    grade: str                      # A / B / C / D / F
    glass_coverage: float           # fraction of image that is glass
    glass_pixel_count: int
    total_pixel_count: int
    mean_dirt_over_glass: float     # mean dirt intensity on glass area
    max_dirt_over_glass: float      # peak dirt intensity
    spatial_dirt_map: np.ndarray    # (H, W) float32 — dirt map masked to glass
    per_region_scores: List[Dict]   # per detected window region scores


GRADE_THRESHOLDS = {
    "A": 0.85,  # score >= 0.85
    "B": 0.70,  # score >= 0.70
    "C": 0.50,  # score >= 0.50
    "D": 0.30,  # score >= 0.30
    "F": 0.0,   # score < 0.30
}


# ---------------------------------------------------------------------------
# Core score computation
# ---------------------------------------------------------------------------

def compute_cleanliness_score(
    dirt_map: torch.Tensor,
    glass_mask: torch.Tensor,
    glass_threshold: float = 0.5,
) -> float:
    """
    Compute the overall cleanliness score.

    Parameters
    ----------
    dirt_map     : (1, H, W) or (H, W) — predicted dirt heatmap in [0, 1]
    glass_mask   : (1, H, W) or (H, W) — predicted glass mask in [0, 1]
    glass_threshold : float — threshold for binarizing the glass mask

    Returns
    -------
    float ∈ [0, 1] — cleanliness score (1 = clean)
    """
    if dirt_map.requires_grad:
        dirt_map = dirt_map.detach()
    if glass_mask.requires_grad:
        glass_mask = glass_mask.detach()

    dirt_map = dirt_map.squeeze().float()
    glass_bin = (glass_mask.squeeze() > glass_threshold).float()

    glass_pixels = (glass_bin > 0.5)
    if not glass_pixels.any():
        return 1.0  # No glass detected → undefined, return clean

    mean_dirt = dirt_map[glass_pixels].mean().item()
    return float(1.0 - mean_dirt)


def compute_full_analysis(
    dirt_map: torch.Tensor,
    glass_mask: torch.Tensor,
    glass_threshold: float = 0.5,
    detect_regions: bool = True,
) -> CleanlinessResult:
    """
    Perform full cleanliness analysis for a single image prediction.

    Parameters
    ----------
    dirt_map     : (1, H, W) or (H, W) predicted dirt heatmap [0,1]
    glass_mask   : (1, H, W) or (H, W) predicted glass mask [0,1]
    glass_threshold : binarization threshold
    detect_regions  : whether to analyse individual glass regions

    Returns
    -------
    CleanlinessResult
    """
    if dirt_map.requires_grad:
        dirt_map = dirt_map.detach()
    if glass_mask.requires_grad:
        glass_mask = glass_mask.detach()

    dirt_np = dirt_map.squeeze().cpu().numpy().astype(np.float32)
    mask_np = glass_mask.squeeze().cpu().numpy().astype(np.float32)
    glass_bin = (mask_np > glass_threshold).astype(np.float32)

    h, w = dirt_np.shape
    total_pixels = h * w
    glass_pixels = int(glass_bin.sum())
    glass_coverage = glass_pixels / total_pixels

    if glass_pixels == 0:
        spatial_dirt = np.zeros_like(dirt_np)
        return CleanlinessResult(
            overall_score=1.0,
            grade="A",
            glass_coverage=0.0,
            glass_pixel_count=0,
            total_pixel_count=total_pixels,
            mean_dirt_over_glass=0.0,
            max_dirt_over_glass=0.0,
            spatial_dirt_map=spatial_dirt,
            per_region_scores=[],
        )

    spatial_dirt = dirt_np * glass_bin
    glass_dirty_vals = dirt_np[glass_bin > 0.5]

    mean_dirt = float(glass_dirty_vals.mean())
    max_dirt = float(glass_dirty_vals.max())
    overall_score = float(1.0 - mean_dirt)
    grade = _score_to_grade(overall_score)

    per_region_scores = []
    if detect_regions:
        per_region_scores = _compute_region_scores(dirt_np, glass_bin)

    return CleanlinessResult(
        overall_score=overall_score,
        grade=grade,
        glass_coverage=glass_coverage,
        glass_pixel_count=glass_pixels,
        total_pixel_count=total_pixels,
        mean_dirt_over_glass=mean_dirt,
        max_dirt_over_glass=max_dirt,
        spatial_dirt_map=spatial_dirt,
        per_region_scores=per_region_scores,
    )


def compute_batch_scores(
    dirt_maps: torch.Tensor,
    glass_masks: torch.Tensor,
    glass_threshold: float = 0.5,
) -> List[float]:
    """
    Compute cleanliness scores for a batch of predictions.

    Parameters
    ----------
    dirt_maps   : (B, 1, H, W)
    glass_masks : (B, 1, H, W)

    Returns
    -------
    List of B scores
    """
    scores = []
    b = dirt_maps.shape[0]
    for i in range(b):
        score = compute_cleanliness_score(
            dirt_maps[i], glass_masks[i], glass_threshold
        )
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# Region-level analysis
# ---------------------------------------------------------------------------

def _compute_region_scores(
    dirt_np: np.ndarray,
    glass_bin: np.ndarray,
) -> List[Dict]:
    """
    Detect individual glass regions (connected components) and compute
    per-region cleanliness scores using OpenCV.
    """
    try:
        import cv2
    except ImportError:
        return []

    mask_uint8 = (glass_bin * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_uint8, connectivity=8)

    region_scores = []
    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        pixel_count = int(region_mask.sum())
        if pixel_count < 50:  # skip tiny fragments
            continue

        dirt_vals = dirt_np[region_mask]
        mean_dirt = float(dirt_vals.mean())
        cleanliness = float(1.0 - mean_dirt)

        ys, xs = np.where(region_mask)
        bbox = {
            "x1": int(xs.min()),
            "y1": int(ys.min()),
            "x2": int(xs.max()),
            "y2": int(ys.max()),
        }
        region_scores.append({
            "region_id": label_id,
            "bbox": bbox,
            "pixel_count": pixel_count,
            "cleanliness_score": cleanliness,
            "grade": _score_to_grade(cleanliness),
            "mean_dirt": mean_dirt,
            "max_dirt": float(dirt_vals.max()),
        })

    return region_scores


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _score_to_grade(score: float) -> str:
    for grade, threshold in GRADE_THRESHOLDS.items():
        if score >= threshold:
            return grade
    return "F"
