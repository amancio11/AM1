"""
test_models.py
==============
Unit tests for model architectures and loss functions.
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.losses import (
    DiceLoss,
    FocalLoss,
    CombinedSegLoss,
    SSIMLoss,
    CombinedRegLoss,
    MultiTaskLoss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 2
H, W = 64, 64


@pytest.fixture
def random_logits():
    return torch.randn(BATCH, 1, H, W)


@pytest.fixture
def random_prob():
    return torch.sigmoid(torch.randn(BATCH, 1, H, W))


@pytest.fixture
def binary_mask():
    mask = torch.zeros(BATCH, 1, H, W)
    mask[:, :, 10:50, 10:50] = 1.0
    return mask


@pytest.fixture
def dirt_pred():
    return torch.sigmoid(torch.randn(BATCH, 1, H, W))


@pytest.fixture
def dirt_target():
    return torch.rand(BATCH, 1, H, W)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestDiceLoss:
    def test_perfect_prediction_approaches_zero(self):
        loss_fn = DiceLoss(from_logits=False)
        # perfect prediction = target (map to logits space)
        target = torch.ones(BATCH, 1, H, W)
        pred = torch.ones(BATCH, 1, H, W)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_worst_prediction_approaches_one(self):
        loss_fn = DiceLoss(from_logits=False)
        target = torch.ones(BATCH, 1, H, W)
        pred = torch.zeros(BATCH, 1, H, W)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.9

    def test_from_logits(self, random_logits, binary_mask):
        loss_fn = DiceLoss(from_logits=True)
        loss = loss_fn(random_logits, binary_mask)
        assert 0.0 <= loss.item() <= 1.0

    def test_output_is_scalar(self, random_logits, binary_mask):
        loss_fn = DiceLoss()
        loss = loss_fn(random_logits, binary_mask)
        assert loss.ndim == 0


class TestFocalLoss:
    def test_output_scalar(self, random_logits, binary_mask):
        loss_fn = FocalLoss()
        loss = loss_fn(random_logits, binary_mask)
        assert loss.ndim == 0

    def test_positive_loss(self, random_logits, binary_mask):
        loss_fn = FocalLoss()
        loss = loss_fn(random_logits, binary_mask)
        assert loss.item() >= 0.0


class TestCombinedSegLoss:
    def test_returns_dict(self, random_logits, binary_mask):
        loss_fn = CombinedSegLoss()
        result = loss_fn(random_logits, binary_mask)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "bce" in result
        assert "dice" in result
        assert "focal" in result

    def test_total_loss_positive(self, random_logits, binary_mask):
        loss_fn = CombinedSegLoss()
        result = loss_fn(random_logits, binary_mask)
        assert result["loss"].item() > 0.0

    def test_gradients_flow(self, binary_mask):
        loss_fn = CombinedSegLoss()
        logits = torch.randn(BATCH, 1, H, W, requires_grad=True)
        result = loss_fn(logits, binary_mask)
        result["loss"].backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestSSIMLoss:
    def test_perfect_prediction_near_zero(self):
        loss_fn = SSIMLoss()
        x = torch.rand(BATCH, 1, H, W)
        loss = loss_fn(x, x)
        assert loss.item() < 0.01

    def test_different_images_positive_loss(self):
        loss_fn = SSIMLoss()
        x = torch.rand(BATCH, 1, H, W)
        y = torch.rand(BATCH, 1, H, W)
        loss = loss_fn(x, y)
        assert loss.item() > 0.0

    def test_masked_ssim(self, binary_mask):
        loss_fn = SSIMLoss()
        x = torch.rand(BATCH, 1, H, W)
        y = torch.rand(BATCH, 1, H, W)
        loss = loss_fn(x, y, mask=binary_mask)
        assert not torch.isnan(loss)


class TestCombinedRegLoss:
    def test_returns_dict(self, dirt_pred, dirt_target):
        loss_fn = CombinedRegLoss()
        result = loss_fn(dirt_pred, dirt_target)
        for key in ["loss", "mse", "mae", "ssim"]:
            assert key in result

    def test_masked_loss(self, dirt_pred, dirt_target, binary_mask):
        loss_fn = CombinedRegLoss(mask_loss=True)
        result = loss_fn(dirt_pred, dirt_target, glass_mask=binary_mask)
        assert result["loss"].item() > 0.0

    def test_gradients_flow(self, dirt_target, binary_mask):
        loss_fn = CombinedRegLoss()
        pred = torch.sigmoid(torch.randn(BATCH, 1, H, W, requires_grad=True))
        result = loss_fn(pred, dirt_target, binary_mask)
        result["loss"].backward()
        assert pred.grad is not None


class TestMultiTaskLoss:
    def test_uncertainty_balancing(self, random_logits, binary_mask, dirt_pred, dirt_target):
        seg_loss = CombinedSegLoss()
        reg_loss = CombinedRegLoss()
        mt_loss = MultiTaskLoss(seg_loss, reg_loss, task_balancing="uncertainty")
        result = mt_loss(random_logits, binary_mask, dirt_pred, dirt_target, binary_mask)
        assert "loss" in result
        assert result["loss"].item() > 0.0

    def test_fixed_balancing(self, random_logits, binary_mask, dirt_pred, dirt_target):
        seg_loss = CombinedSegLoss()
        reg_loss = CombinedRegLoss()
        mt_loss = MultiTaskLoss(seg_loss, reg_loss, task_balancing="fixed")
        result = mt_loss(random_logits, binary_mask, dirt_pred, dirt_target)
        assert result["loss"].item() > 0.0


# ---------------------------------------------------------------------------
# Model architecture tests — lazy import to avoid requiring GPU
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("segmentation_models_pytorch", reason="smp not installed"),
    reason="segmentation_models_pytorch not available",
)
class TestGlassSegmentationModel:
    def test_forward_shape(self):
        from models.glass_segmentation import GlassSegmentationModel
        model = GlassSegmentationModel(
            architecture="unet",
            encoder_name="resnet18",  # lightweight for testing
            encoder_weights=None,
        )
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_freeze_unfreeze_encoder(self):
        from models.glass_segmentation import GlassSegmentationModel
        model = GlassSegmentationModel(
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
        )
        model.freeze_encoder()
        for p in model.model.encoder.parameters():
            assert not p.requires_grad
        model.unfreeze_encoder()
        for p in model.model.encoder.parameters():
            assert p.requires_grad
