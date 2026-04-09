"""src.evaluation — metrics, cleanliness scoring, visualization"""
from evaluation.metrics import (
    iou_score,
    dice_score,
    segmentation_metrics,
    regression_metrics,
    masked_mae,
    masked_mse,
)
from evaluation.cleanliness_score import (
    compute_cleanliness_score,
    compute_full_analysis,
    compute_batch_scores,
    CleanlinessResult,
)
from evaluation.visualizer import (
    overlay_glass_mask,
    overlay_dirt_heatmap,
    draw_cleanliness_score,
    draw_region_scores,
    make_result_panel,
    save_visualization,
    denormalize_image,
)

__all__ = [
    "iou_score",
    "dice_score",
    "segmentation_metrics",
    "regression_metrics",
    "masked_mae",
    "masked_mse",
    "compute_cleanliness_score",
    "compute_full_analysis",
    "compute_batch_scores",
    "CleanlinessResult",
    "overlay_glass_mask",
    "overlay_dirt_heatmap",
    "draw_cleanliness_score",
    "draw_region_scores",
    "make_result_panel",
    "save_visualization",
    "denormalize_image",
]
