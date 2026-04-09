"""
visualizer.py
=============
Visualization utilities for model outputs.

Produces:
  - Segmentation mask overlay on image
  - Dirt heatmap overlay on image
  - Side-by-side comparison panels
  - Cleanliness grade annotation
  - Per-window region score overlay

All functions return numpy uint8 images (BGR for OpenCV, RGB for display).
"""

import cv2
import numpy as np
import torch
import colorsys
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GRADE_COLORS_BGR = {
    "A": (0, 200, 0),       # green
    "B": (0, 180, 100),     # teal
    "C": (0, 180, 255),     # orange
    "D": (0, 80, 255),      # deep orange
    "F": (0, 0, 220),       # red
}

DIRT_COLORMAP = cv2.COLORMAP_JET   # yellow=dirty, blue=clean


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denormalize_image(
    image_tensor: torch.Tensor,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> np.ndarray:
    """
    Convert a normalized (C, H, W) tensor back to uint8 (H, W, 3) RGB.
    """
    img = image_tensor.cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)  # CHW → HWC
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def tensor_to_numpy_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) tensor to float32 numpy (H, W)."""
    arr = mask_tensor.cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay functions
# ---------------------------------------------------------------------------

def overlay_glass_mask(
    image_rgb: np.ndarray,
    glass_mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlay a binary glass mask on an RGB image.

    Parameters
    ----------
    image_rgb  : (H, W, 3) uint8 RGB
    glass_mask : (H, W) float32 [0,1]
    alpha      : overlay transparency
    color      : RGB tuple for glass highlight

    Returns
    -------
    (H, W, 3) uint8 RGB
    """
    overlay = image_rgb.copy()
    mask_bin = (glass_mask > 0.5).astype(np.uint8)
    color_layer = np.zeros_like(image_rgb)
    color_layer[:] = color
    overlay = np.where(
        mask_bin[:, :, np.newaxis] > 0,
        (1 - alpha) * image_rgb + alpha * color_layer,
        image_rgb,
    ).astype(np.uint8)
    # Draw contour
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color[::-1], 1)  # BGR for cv2
    return overlay


def overlay_dirt_heatmap(
    image_rgb: np.ndarray,
    dirt_map: np.ndarray,
    glass_mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    colormap: int = DIRT_COLORMAP,
) -> np.ndarray:
    """
    Overlay a dirt intensity heatmap on an RGB image.

    The heatmap is only applied to glass pixels when glass_mask is provided.
    """
    # Convert dirt_map [0,1] → colormap BGR
    dirt_uint8 = (np.clip(dirt_map, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(dirt_uint8, colormap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    if glass_mask is not None:
        glass_bin = (glass_mask > 0.5).astype(np.float32)[:, :, np.newaxis]
        overlay = np.where(
            glass_bin > 0,
            (1 - alpha) * image_rgb + alpha * heat_rgb,
            image_rgb,
        ).astype(np.uint8)
    else:
        # Apply everywhere where there's non-zero dirt
        dirt_mask = (dirt_map > 0.05).astype(np.float32)[:, :, np.newaxis]
        overlay = np.where(
            dirt_mask > 0,
            (1 - alpha) * image_rgb + alpha * heat_rgb,
            image_rgb,
        ).astype(np.uint8)

    return overlay


def draw_cleanliness_score(
    image_rgb: np.ndarray,
    score: float,
    grade: str,
    position: Tuple[int, int] = (20, 40),
    font_scale: float = 1.2,
    thickness: int = 2,
) -> np.ndarray:
    """Annotate an image with the cleanliness score and grade."""
    vis = image_rgb.copy()
    color_bgr = GRADE_COLORS_BGR.get(grade, (200, 200, 200))
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    text = f"Score: {score:.2f} | Grade: {grade}"
    # Background rectangle
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(vis, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(
        vis, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color_rgb[::-1],  # BGR for OpenCV
        thickness, cv2.LINE_AA,
    )
    return vis


def draw_region_scores(
    image_rgb: np.ndarray,
    region_scores: List[Dict],
) -> np.ndarray:
    """Draw bounding boxes and per-region scores on the image."""
    vis = image_rgb.copy()
    for region in region_scores:
        bbox = region["bbox"]
        score = region["cleanliness_score"]
        grade = region["grade"]
        color_bgr = GRADE_COLORS_BGR.get(grade, (200, 200, 200))
        color_rgb_cv = color_bgr  # OpenCV uses BGR

        cv2.rectangle(vis, (bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"]), color_bgr, 2)
        label = f"{score:.2f} ({grade})"
        lx = bbox["x1"] + 4
        ly = max(bbox["y1"] + 18, 20)
        cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Comparison panels
# ---------------------------------------------------------------------------

def make_result_panel(
    image_rgb: np.ndarray,
    glass_mask: np.ndarray,
    dirt_map: np.ndarray,
    score: float,
    grade: str,
    pred_glass_overlay: Optional[np.ndarray] = None,
    pred_dirt_overlay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a side-by-side visualization panel:
    [Original | Glass Overlay | Dirt Heatmap | Score Annotation]
    """
    h, w = image_rgb.shape[:2]

    glass_vis = overlay_glass_mask(image_rgb, glass_mask)
    dirt_vis = overlay_dirt_heatmap(image_rgb, dirt_map, glass_mask=glass_mask)
    scored_vis = draw_cleanliness_score(image_rgb.copy(), score, grade)

    panel = np.concatenate([image_rgb, glass_vis, dirt_vis, scored_vis], axis=1)
    return panel


def save_visualization(
    output_path: str,
    image_rgb: np.ndarray,
    glass_mask: np.ndarray,
    dirt_map: np.ndarray,
    score: float,
    grade: str,
    region_scores: Optional[List[Dict]] = None,
    dpi: int = 100,
) -> None:
    """Save a full result panel as a PNG file."""
    panel = make_result_panel(image_rgb, glass_mask, dirt_map, score, grade)
    if region_scores:
        panel_with_regions = draw_region_scores(panel, region_scores)
    else:
        panel_with_regions = panel

    # Save using OpenCV (convert RGB → BGR)
    cv2.imwrite(output_path, cv2.cvtColor(panel_with_regions, cv2.COLOR_RGB2BGR))


def make_heatmap_colorbar(
    height: int = 256,
    width: int = 30,
) -> np.ndarray:
    """Generate a vertical colorbar for the dirt heatmap."""
    bar = np.linspace(1, 0, height).astype(np.float32)  # top=dirty, bottom=clean
    bar = (bar * 255).astype(np.uint8)
    bar_2d = np.tile(bar[:, np.newaxis], (1, width))
    bar_colored = cv2.applyColorMap(bar_2d, DIRT_COLORMAP)
    cv2.putText(bar_colored, "Dirty", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(bar_colored, "Clean", (2, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return bar_colored
