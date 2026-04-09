"""
test_inference.py
=================
Unit tests for the inference pipeline (predictor, image inference).
"""

import sys
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.predictor import Predictor


# ---------------------------------------------------------------------------
# Mock models for fast testing (no real weights)
# ---------------------------------------------------------------------------

class _IdentitySegModel(nn.Module):
    """Returns logits that produce a fixed mask pattern."""
    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros(b, 1, h, w)
        out[:, :, h//4:3*h//4, w//4:3*w//4] = 5.0  # central glass region
        return out


class _ConstantDirtModel(nn.Module):
    """Returns a constant 0.3 dirt map everywhere."""
    def forward(self, x, glass_mask=None):
        b, c, h, w = x.shape
        return torch.full((b, 1, h, w), 0.3)


class _MultitaskModel(nn.Module):
    """Returns fixed seg + dirt predictions."""
    def forward(self, x):
        b, c, h, w = x.shape
        seg = torch.zeros(b, 1, h, w)
        seg[:, :, h//4:3*h//4, w//4:3*w//4] = 5.0
        dirt = torch.full((b, 1, h, w), 0.2)
        return seg, dirt


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictorInit:
    def test_two_stage_mode(self):
        p = Predictor(
            glass_model=_IdentitySegModel(),
            dirt_model=_ConstantDirtModel(),
            device=torch.device("cpu"),
        )
        assert p._mode == "two_stage"

    def test_glass_only_mode(self):
        p = Predictor(
            glass_model=_IdentitySegModel(),
            device=torch.device("cpu"),
        )
        assert p._mode == "glass_only"

    def test_multitask_mode(self):
        p = Predictor(
            glass_model=None,
            multitask_model=_MultitaskModel(),
            device=torch.device("cpu"),
        )
        assert p._mode == "multitask"

    def test_no_models_raises(self):
        with pytest.raises(ValueError):
            Predictor(glass_model=None, device=torch.device("cpu"))


class TestPredictorPreprocess:
    def test_preprocess_shape(self):
        p = Predictor(glass_model=_IdentitySegModel(), device=torch.device("cpu"))
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = p._preprocess(img)
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.dtype == torch.float32

    def test_preprocess_normalized(self):
        """Verify the image is normalized (not in [0,255] range)."""
        p = Predictor(glass_model=_IdentitySegModel(), device=torch.device("cpu"))
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        tensor = p._preprocess(img)
        # After normalization with ImageNet stats, values should not be in raw [0,255]
        assert tensor.abs().max().item() < 10.0


class TestPredictImage:
    def test_returns_expected_keys(self, tmp_path):
        import cv2
        # Create a dummy test image
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, img)

        p = Predictor(
            glass_model=_IdentitySegModel(),
            dirt_model=_ConstantDirtModel(),
            device=torch.device("cpu"),
            image_size=(64, 64),
        )
        result = p.predict_image(img_path)
        for key in ["glass_mask", "glass_binary", "dirt_map", "score", "grade", "analysis"]:
            assert key in result

    def test_score_range(self, tmp_path):
        import cv2
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test2.png")
        cv2.imwrite(img_path, img)

        p = Predictor(
            glass_model=_IdentitySegModel(),
            dirt_model=_ConstantDirtModel(),
            device=torch.device("cpu"),
            image_size=(64, 64),
        )
        result = p.predict_image(img_path)
        assert 0.0 <= result["score"] <= 1.0

    def test_multitask_predict(self, tmp_path):
        import cv2
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test_mt.png")
        cv2.imwrite(img_path, img)

        p = Predictor(
            glass_model=None,
            multitask_model=_MultitaskModel(),
            device=torch.device("cpu"),
            image_size=(64, 64),
        )
        result = p.predict_image(img_path)
        assert "score" in result
        assert result["grade"] in ("A", "B", "C", "D", "F")

    def test_file_not_found(self):
        p = Predictor(glass_model=_IdentitySegModel(), device=torch.device("cpu"))
        with pytest.raises(FileNotFoundError):
            p.predict_image("nonexistent_path/image.png")


class TestPredictBatch:
    def test_batch_output_shape(self):
        p = Predictor(
            glass_model=_IdentitySegModel(),
            dirt_model=_ConstantDirtModel(),
            device=torch.device("cpu"),
            image_size=(64, 64),
        )
        batch = torch.randn(3, 3, 64, 64)
        glass_probs, dirt_maps = p.predict_batch(batch)
        assert glass_probs.shape == (3, 1, 64, 64)
        assert dirt_maps.shape == (3, 1, 64, 64)

    def test_glass_probs_in_range(self):
        p = Predictor(
            glass_model=_IdentitySegModel(),
            device=torch.device("cpu"),
            image_size=(64, 64),
        )
        batch = torch.randn(2, 3, 64, 64)
        glass_probs, _ = p.predict_batch(batch)
        assert glass_probs.min().item() >= 0.0
        assert glass_probs.max().item() <= 1.0
