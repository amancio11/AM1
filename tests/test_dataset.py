"""
test_dataset.py
===============
Unit tests for dataset classes and augmentations.
"""

import os
import sys
import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import (
    GlassSegmentationDataset,
    DirtEstimationDataset,
    MultitaskDataset,
    _discover_scene_ids,
    _split_ids,
)
from data.augmentations import get_train_transforms, get_val_transforms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path):
    """Create a minimal synthetic dataset directory structure."""
    (tmp_path / "images").mkdir()
    (tmp_path / "glass_masks").mkdir()
    (tmp_path / "dirt_maps").mkdir()

    import cv2
    for i in range(10):
        name = f"scene_{i:06d}.png"
        # RGB image
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "images" / name), img)
        # Binary glass mask
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        cv2.imwrite(str(tmp_path / "glass_masks" / name), mask)
        # Dirt map
        dirt = np.random.randint(0, 156, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "dirt_maps" / name), dirt)

    return tmp_path


@pytest.fixture
def minimal_config(synthetic_dataset_dir: Path) -> dict:
    return {
        "dataset": {
            "image_dir": str(synthetic_dataset_dir / "images"),
            "mask_dir": str(synthetic_dataset_dir / "glass_masks"),
            "glass_mask_dir": str(synthetic_dataset_dir / "glass_masks"),
            "dirt_map_dir": str(synthetic_dataset_dir / "dirt_maps"),
            "image_size": [64, 64],
            "train_split": 0.7,
            "val_split": 0.2,
            "test_split": 0.1,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "use_masked_input": False,
        },
        "training": {"batch_size": 2},
        "augmentations": {
            "train": {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
            "val": {
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSceneDiscovery:
    def test_discover_all_scenes(self, synthetic_dataset_dir):
        ids = _discover_scene_ids(
            str(synthetic_dataset_dir / "images"),
            str(synthetic_dataset_dir / "glass_masks"),
            str(synthetic_dataset_dir / "dirt_maps"),
        )
        assert len(ids) == 10

    def test_discover_intersection(self, synthetic_dataset_dir):
        # Remove one dirt map → should reduce count
        list(Path(str(synthetic_dataset_dir / "dirt_maps")).glob("*.png"))[0].unlink()
        ids = _discover_scene_ids(
            str(synthetic_dataset_dir / "images"),
            str(synthetic_dataset_dir / "glass_masks"),
            str(synthetic_dataset_dir / "dirt_maps"),
        )
        assert len(ids) == 9


class TestSplitIds:
    def test_split_proportions(self):
        ids = [str(i) for i in range(100)]
        train, val, test = _split_ids(ids, 0.8, 0.15, 0.05, seed=42)
        assert len(train) == 80
        assert len(val) == 15
        assert len(test) <= 5  # may be 4 or 5 due to int truncation
        assert len(set(train) & set(val)) == 0

    def test_reproducibility(self):
        ids = [str(i) for i in range(50)]
        tr1, _, _ = _split_ids(ids, 0.8, 0.1, 0.1, seed=42)
        tr2, _, _ = _split_ids(ids, 0.8, 0.1, 0.1, seed=42)
        assert tr1 == tr2


class TestGlassSegmentationDataset:
    def test_length(self, minimal_config):
        ds = GlassSegmentationDataset.from_config(minimal_config, split="train")
        assert len(ds) > 0

    def test_getitem_shapes(self, minimal_config):
        ds = GlassSegmentationDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (3, 64, 64)
        assert sample["mask"].shape == (1, 64, 64)

    def test_image_dtype(self, minimal_config):
        ds = GlassSegmentationDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32

    def test_mask_range(self, minimal_config):
        ds = GlassSegmentationDataset.from_config(
            minimal_config, split="val",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        assert sample["mask"].min() >= 0.0
        assert sample["mask"].max() <= 1.0


class TestDirtEstimationDataset:
    def test_has_three_outputs(self, minimal_config):
        ds = DirtEstimationDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        assert "image" in sample
        assert "glass_mask" in sample
        assert "dirt_map" in sample

    def test_dirt_map_range(self, minimal_config):
        ds = DirtEstimationDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        assert sample["dirt_map"].min() >= -0.01
        assert sample["dirt_map"].max() <= 1.01


class TestMultitaskDataset:
    def test_all_keys_present(self, minimal_config):
        ds = MultitaskDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        for key in ["image", "glass_mask", "dirt_map", "scene_id"]:
            assert key in sample, f"Missing key: {key}"

    def test_mask_shape_consistency(self, minimal_config):
        ds = MultitaskDataset.from_config(
            minimal_config, split="train",
            transform=get_val_transforms(minimal_config)
        )
        sample = ds[0]
        h = sample["image"].shape[1]
        w = sample["image"].shape[2]
        assert sample["glass_mask"].shape == (1, h, w)
        assert sample["dirt_map"].shape == (1, h, w)
