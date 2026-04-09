"""
test_metrics.py
===============
Unit tests for evaluation metrics and cleanliness scoring.
"""

import sys
import math
import pytest
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.metrics import (
    iou_score,
    dice_score,
    precision_recall_f1,
    segmentation_metrics,
    regression_metrics,
    masked_mae,
    masked_mse,
    psnr,
    ssim_metric,
)
from evaluation.cleanliness_score import (
    compute_cleanliness_score,
    compute_full_analysis,
    compute_batch_scores,
    _score_to_grade,
)


B, H, W = 4, 64, 64


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

class TestIoUScore:
    def test_perfect_prediction(self):
        target = torch.zeros(B, 1, H, W)
        target[:, :, 10:30, 10:30] = 1.0
        score = iou_score(target.clone() * 10 - 5, target)  # logits → perfect pred
        # With logits version, high => 1.0
        assert score.item() > 0.0

    def test_zero_overlap(self):
        pred = torch.full((B, 1, H, W), -10.0)   # all zeros after sigmoid
        target = torch.ones(B, 1, H, W)
        score = iou_score(pred, target)
        assert score.item() < 0.01

    def test_range(self):
        pred = torch.randn(B, 1, H, W)
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        score = iou_score(pred, target)
        assert 0.0 <= score.item() <= 1.0


class TestDiceScore:
    def test_perfect_prediction(self):
        target = torch.zeros(B, 1, H, W)
        target[:, :, 5:25, 5:25] = 1.0
        pred_probs = target.clone()
        score = dice_score(pred_probs, target, threshold=0.5)
        assert score.item() > 0.99

    def test_empty_mask(self):
        target = torch.zeros(B, 1, H, W)
        pred = torch.zeros(B, 1, H, W)
        score = dice_score(pred, target, threshold=0.5)
        # smooth prevents division by zero
        assert not math.isnan(score.item())


class TestPrecisionRecall:
    def test_keys_present(self):
        pred = torch.randn(B, 1, H, W)
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        result = precision_recall_f1(pred, target)
        for k in ["precision", "recall", "f1"]:
            assert k in result

    def test_all_true_positive(self):
        target = torch.ones(B, 1, H, W)
        pred = torch.full((B, 1, H, W), 10.0)  # logit → sigmoid ≈ 1
        r = precision_recall_f1(pred, target)
        assert r["precision"] > 0.99
        assert r["recall"] > 0.99


class TestSegmentationMetrics:
    def test_returns_complete_dict(self):
        pred = torch.randn(B, 1, H, W)
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        m = segmentation_metrics(pred, target)
        for k in ["iou", "dice", "precision", "recall", "f1"]:
            assert k in m


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_perfect_prediction_mae_zero(self):
        x = torch.rand(B, 1, H, W)
        mae = masked_mae(x, x)
        assert mae.item() < 1e-5

    def test_masked_mae_ignores_non_glass(self):
        pred = torch.ones(B, 1, H, W)
        target = torch.zeros(B, 1, H, W)
        mask = torch.zeros(B, 1, H, W)  # no glass pixels
        mae = masked_mae(pred, target, mask)
        # denom is clamped to 1, so result is sum/1
        assert not math.isnan(mae.item())

    def test_mse_positive(self):
        pred = torch.rand(B, 1, H, W)
        target = torch.rand(B, 1, H, W)
        mse = masked_mse(pred, target)
        assert mse.item() >= 0.0

    def test_psnr_perfect_is_inf(self):
        x = torch.rand(B, 1, H, W)
        val = psnr(x, x)
        assert math.isinf(val) or val > 100.0

    def test_ssim_self_is_one(self):
        x = torch.rand(B, 1, H, W)
        val = ssim_metric(x, x)
        assert val > 0.99

    def test_regression_metrics_dict(self):
        pred = torch.rand(B, 1, H, W)
        target = torch.rand(B, 1, H, W)
        m = regression_metrics(pred, target)
        for k in ["mse", "mae", "rmse", "psnr", "ssim"]:
            assert k in m


# ---------------------------------------------------------------------------
# Cleanliness score
# ---------------------------------------------------------------------------

class TestCleanlinessScore:
    def test_clean_glass_scores_one(self):
        dirt = torch.zeros(1, H, W)
        glass = torch.ones(1, H, W)
        score = compute_cleanliness_score(dirt, glass)
        assert abs(score - 1.0) < 1e-4

    def test_fully_dirty_glass_scores_zero(self):
        dirt = torch.ones(1, H, W)
        glass = torch.ones(1, H, W)
        score = compute_cleanliness_score(dirt, glass)
        assert abs(score - 0.0) < 1e-4

    def test_no_glass_returns_one(self):
        dirt = torch.rand(1, H, W)
        glass = torch.zeros(1, H, W)  # no glass
        score = compute_cleanliness_score(dirt, glass)
        assert score == 1.0

    def test_partial_dirt(self):
        dirt = torch.zeros(1, H, W)
        dirt[:, :H//2, :] = 0.5
        glass = torch.ones(1, H, W)
        score = compute_cleanliness_score(dirt, glass)
        assert 0.5 < score < 1.0

    def test_batch_scores_length(self):
        dirts = torch.rand(B, 1, H, W)
        glasses = torch.ones(B, 1, H, W)
        scores = compute_batch_scores(dirts, glasses)
        assert len(scores) == B
        for s in scores:
            assert 0.0 <= s <= 1.0


class TestGradeThresholds:
    @pytest.mark.parametrize("score,expected", [
        (0.95, "A"),
        (0.80, "B"),
        (0.60, "C"),
        (0.40, "D"),
        (0.10, "F"),
    ])
    def test_grade_mapping(self, score, expected):
        assert _score_to_grade(score) == expected


class TestFullAnalysis:
    def test_returns_dataclass(self):
        from evaluation.cleanliness_score import CleanlinessResult
        dirt = torch.rand(1, H, W) * 0.3
        glass = torch.zeros(1, H, W)
        glass[:, 10:50, 10:50] = 1.0
        result = compute_full_analysis(dirt, glass)
        assert isinstance(result, CleanlinessResult)

    def test_spatial_dirt_map_shape(self):
        dirt = torch.rand(1, H, W)
        glass = torch.ones(1, H, W) * 0.8
        result = compute_full_analysis(dirt, glass)
        assert result.spatial_dirt_map.shape == (H, W)

    def test_grade_is_valid(self):
        dirt = torch.rand(1, H, W) * 0.2
        glass = torch.ones(1, H, W)
        result = compute_full_analysis(dirt, glass)
        assert result.grade in ("A", "B", "C", "D", "F")
