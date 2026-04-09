"""
tests/test_domain_adaptation.py
================================
Unit tests for the src/domain_adaptation module.

All tests are self-contained: they create temporary directories with
synthetic images so no real drone dataset is required.

Run with:
    pytest tests/test_domain_adaptation.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rgb_dir(tmp_path: Path) -> Path:
    """Write 10 fake 256×256 RGB images."""
    d = tmp_path / "images"
    d.mkdir()
    for i in range(10):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i:04d}.png"), img)
    return d


@pytest.fixture
def glass_mask_dir(tmp_path: Path) -> Path:
    """Write matching binary glass masks."""
    d = tmp_path / "glass_masks"
    d.mkdir()
    for i in range(10):
        mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        cv2.imwrite(str(d / f"img_{i:04d}.png"), mask)
    return d


@pytest.fixture
def pseudo_label_dir(tmp_path: Path, rgb_dir: Path) -> Path:
    """Write fake pseudo-label files for all images."""
    d = tmp_path / "pseudo_labels"
    d.mkdir()
    for img_path in sorted(rgb_dir.glob("*.png")):
        stem = img_path.stem
        for suffix in ["_glass.png", "_dirt.png", "_glass_conf.png", "_dirt_conf.png"]:
            arr = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(d / f"{stem}{suffix}"), arr)
        meta = {"mean_conf": 0.9, "mean_entropy": 0.1}
        with open(d / f"{stem}_meta.json", "w") as f:
            json.dump(meta, f)
    return d


@pytest.fixture
def da_config(tmp_path: Path, rgb_dir: Path, pseudo_label_dir: Path) -> Dict:
    return {
        "domain_adaptation": {
            "paths": {
                "real_images_dir": str(rgb_dir),
                "pseudo_label_dir": str(pseudo_label_dir),
                "cyclegan_checkpoint_dir": str(tmp_path / "ckpt_gan"),
                "finetune_checkpoint_dir": str(tmp_path / "ckpt_ft"),
            },
            "real_dataset": {
                "image_size": [256, 256],
                "has_glass_labels": False,
                "has_dirt_labels": False,
                "train_split": 0.8,
                "extensions": [".png"],
            },
            "domain_augmentations": {},
            "style_transfer": {
                "generator": {"ngf": 16, "n_res_blocks": 2},
                "discriminator": {"ndf": 16, "n_layers": 2},
                "lambda_cycle": 10.0,
                "lambda_identity": 5.0,
                "lr": 2e-4,
                "n_epochs": 2,
                "pool_size": 4,
            },
            "pseudo_labeling": {
                "glass_confidence_threshold": 0.5,
                "max_mean_entropy": 0.9,
                "filter_strategy": "combined",
                "max_pseudo_images": 5,
            },
            "finetuning": {
                "n_epochs": 2,
                "lr": 1e-4,
                "encoder_lr_multiplier": 0.1,
                "freeze_encoder_epochs": 1,
                "mix_synthetic": False,
                "batch_size": 2,
                "amp": False,
            },
        }
    }


# ===========================================================================
# RealFacadeDataset
# ===========================================================================

class TestRealFacadeDataset:
    def test_unlabeled_len(self, da_config: Dict):
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        ds = RealFacadeDataset.from_config(da_config, split="train")
        assert len(ds) > 0

    def test_returns_image_tensor(self, da_config: Dict):
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        ds = RealFacadeDataset.from_config(da_config, split="train")
        sample = ds[0]
        assert "image" in sample
        img = sample["image"]
        assert isinstance(img, torch.Tensor)
        assert img.ndim == 3
        assert img.shape[0] == 3

    def test_no_dummy_glass_mask(self, da_config: Dict):
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        ds = RealFacadeDataset.from_config(da_config, split="train")
        sample = ds[0]
        # Without label dirs, glass_mask should be None
        assert sample.get("glass_mask") is None

    def test_split_reproducible(self, da_config: Dict):
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        ds1 = RealFacadeDataset.from_config(da_config, split="train")
        ds2 = RealFacadeDataset.from_config(da_config, split="train")
        assert len(ds1) == len(ds2)
        assert [p.name for p in ds1.image_paths] == [p.name for p in ds2.image_paths]

    def test_with_glass_labels(self, tmp_path: Path, rgb_dir: Path, glass_mask_dir: Path):
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        config = {
            "domain_adaptation": {
                "paths": {
                    "real_images_dir": str(rgb_dir),
                    "real_glass_masks_dir": str(glass_mask_dir),
                },
                "real_dataset": {
                    "image_size": [256, 256],
                    "has_glass_labels": True,
                    "has_dirt_labels": False,
                    "train_split": 0.8,
                    "extensions": [".png"],
                },
            }
        }
        ds = RealFacadeDataset.from_config(config, split="train")
        assert ds.has_glass_labels
        sample = ds[0]
        assert sample["glass_mask"] is not None
        assert isinstance(sample["glass_mask"], torch.Tensor)


# ===========================================================================
# PseudoLabeledDataset
# ===========================================================================

class TestPseudoLabeledDataset:
    def test_len_and_shapes(self, rgb_dir: Path, pseudo_label_dir: Path):
        from src.domain_adaptation.real_dataset import PseudoLabeledDataset
        paths = list(sorted(rgb_dir.glob("*.png")))
        ds = PseudoLabeledDataset(
            image_paths=paths,
            pseudo_label_dir=str(pseudo_label_dir),
            image_size=(256, 256),
        )
        assert len(ds) == len(paths)
        sample = ds[0]
        assert sample["glass_mask"].shape == (1, 256, 256)
        assert sample["dirt_map"].shape == (1, 256, 256)
        assert sample["glass_confidence"].shape == (1, 256, 256)
        assert sample["is_pseudo_label"] is True

    def test_sample_weight_is_float(self, rgb_dir: Path, pseudo_label_dir: Path):
        from src.domain_adaptation.real_dataset import PseudoLabeledDataset
        paths = list(sorted(rgb_dir.glob("*.png")))
        ds = PseudoLabeledDataset(
            image_paths=paths,
            pseudo_label_dir=str(pseudo_label_dir),
            image_size=(256, 256),
        )
        sample = ds[0]
        assert isinstance(sample["sample_weight"], float)
        assert 0.0 <= sample["sample_weight"] <= 1.0


# ===========================================================================
# MixedDataset
# ===========================================================================

class TestMixedDataset:
    def _make_fake_ds(self, n: int, label: str):
        """Minimal dataset that emits {'image': tensor, 'label': label}."""
        class FakeDS(torch.utils.data.Dataset):
            def __len__(self):
                return n
            def __getitem__(self, i):
                return {"image": torch.randn(3, 256, 256), "test_label": label}
        return FakeDS()

    def test_len(self):
        from src.domain_adaptation.real_dataset import MixedDataset
        syn_ds = self._make_fake_ds(100, "syn")
        real_ds = self._make_fake_ds(20, "real")
        mixed = MixedDataset(syn_ds, real_ds, synthetic_ratio=0.3)
        assert len(mixed) == 120

    def test_domain_tag(self):
        from src.domain_adaptation.real_dataset import MixedDataset
        syn_ds = self._make_fake_ds(5, "syn")
        real_ds = self._make_fake_ds(5, "real")
        mixed = MixedDataset(syn_ds, real_ds)
        assert mixed[0]["domain"] == "synthetic"
        assert mixed[5]["domain"] == "real"

    def test_weighted_sampler(self):
        from src.domain_adaptation.real_dataset import MixedDataset
        syn_ds = self._make_fake_ds(100, "syn")
        real_ds = self._make_fake_ds(20, "real")
        mixed = MixedDataset(syn_ds, real_ds, synthetic_ratio=0.3)
        sampler = mixed.build_weighted_sampler()
        assert hasattr(sampler, "__iter__")


# ===========================================================================
# Domain Augmentations
# ===========================================================================

class TestDomainAugmentations:
    def test_transform_output_shape(self):
        from src.domain_adaptation.domain_augmentations import (
            build_domain_randomization_transform,
        )
        tfm = build_domain_randomization_transform({}, image_size=(256, 256))
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = tfm(image=img)
        assert out["image"].shape == (256, 256, 3)

    def test_val_transform_deterministic(self):
        from src.domain_adaptation.domain_augmentations import build_real_world_val_transform
        tfm = build_real_world_val_transform(image_size=(128, 128))
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out1 = tfm(image=img.copy())["image"]
        out2 = tfm(image=img.copy())["image"]
        assert np.allclose(out1, out2)

    def test_synthetic_dr_transform(self):
        from src.domain_adaptation.domain_augmentations import (
            build_synthetic_domain_randomization,
        )
        tfm = build_synthetic_domain_randomization(image_size=(256, 256))
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        out = tfm(image=img)
        assert out["image"].shape == (256, 256, 3)


# ===========================================================================
# Pseudo Labeling utilities
# ===========================================================================

class TestBinaryEntropy:
    def test_max_at_0_5(self):
        from src.domain_adaptation.pseudo_labeling import _binary_entropy
        p = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
        H = _binary_entropy(p)
        assert H[0, 0] == pytest.approx(1.0, abs=0.01)
        assert H[1, 0] < 0.01

    def test_no_nan(self):
        from src.domain_adaptation.pseudo_labeling import _binary_entropy
        p = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        H = _binary_entropy(p)
        assert not np.any(np.isnan(H))


class TestPseudoLabelFilter:
    def test_confidence_strategy_passes(self):
        from src.domain_adaptation.pseudo_labeling import _passes_filter
        assert _passes_filter(0.9, 0.5, "confidence", 0.85, 0.4) is True
        assert _passes_filter(0.7, 0.1, "confidence", 0.85, 0.4) is False

    def test_entropy_strategy(self):
        from src.domain_adaptation.pseudo_labeling import _passes_filter
        assert _passes_filter(0.5, 0.3, "entropy", 0.85, 0.4) is True
        assert _passes_filter(0.5, 0.5, "entropy", 0.85, 0.4) is False

    def test_combined_requires_both(self):
        from src.domain_adaptation.pseudo_labeling import _passes_filter
        assert _passes_filter(0.9, 0.2, "combined", 0.85, 0.4) is True
        assert _passes_filter(0.9, 0.5, "combined", 0.85, 0.4) is False
        assert _passes_filter(0.7, 0.2, "combined", 0.85, 0.4) is False


class TestPseudoLabelerGenerate:
    def test_generate_saves_files(self, da_config: Dict, rgb_dir: Path, tmp_path: Path):
        from src.domain_adaptation.pseudo_labeling import PseudoLabeler
        from src.domain_adaptation.real_dataset import _discover_images

        pseudo_out = tmp_path / "new_pseudo"
        pseudo_out.mkdir()

        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict_image.return_value = {
            "glass_mask": np.full((256, 256), 0.95, dtype=np.float32),
            "dirt_map": np.full((256, 256), 0.1, dtype=np.float32),
        }

        image_paths = list(sorted(rgb_dir.glob("*.png")))
        labeler = PseudoLabeler(
            predictor=mock_predictor,
            image_paths=image_paths,
            output_dir=str(pseudo_out),
            glass_confidence_threshold=0.5,
            max_mean_entropy=0.9,
            filter_strategy="combined",
            max_pseudo_images=5,
        )
        manifest = labeler.generate()

        assert manifest["stats"]["num_accepted"] > 0
        assert (pseudo_out / "manifest.json").exists()
        # Check first accepted image has all 4 output files
        first_accepted = Path(manifest["accepted"][0])
        stem = first_accepted.stem
        for suffix in ["_glass.png", "_dirt.png", "_glass_conf.png", "_dirt_conf.png"]:
            assert (pseudo_out / f"{stem}{suffix}").exists()


# ===========================================================================
# Style Transfer
# ===========================================================================

class TestResNetGenerator:
    def test_forward_shape(self):
        from src.domain_adaptation.style_transfer import ResNetGenerator
        G = ResNetGenerator(ngf=16, n_res_blocks=2)
        x = torch.randn(1, 3, 256, 256)
        out = G(x)
        assert out.shape == x.shape

    def test_output_range_tanh(self):
        from src.domain_adaptation.style_transfer import ResNetGenerator
        G = ResNetGenerator(ngf=16, n_res_blocks=2)
        x = torch.randn(1, 3, 64, 64)
        out = G(x)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5


class TestPatchGANDiscriminator:
    def test_output_is_4d(self):
        from src.domain_adaptation.style_transfer import PatchGANDiscriminator
        D = PatchGANDiscriminator(ndf=16, n_layers=2)
        x = torch.randn(1, 3, 256, 256)
        out = D(x)
        assert out.ndim == 4
        # Spatial dims should be smaller than input
        assert out.shape[-1] < 256

    def test_batch_size_preserved(self):
        from src.domain_adaptation.style_transfer import PatchGANDiscriminator
        D = PatchGANDiscriminator(ndf=16, n_layers=2)
        x = torch.randn(3, 3, 128, 128)
        out = D(x)
        assert out.shape[0] == 3


class TestImagePool:
    def test_pool_fills_and_queries(self):
        from src.domain_adaptation.style_transfer import ImagePool
        pool = ImagePool(pool_size=5)
        for _ in range(10):
            imgs = torch.randn(2, 3, 64, 64)
            out = pool.query(imgs)
            assert out.shape == imgs.shape

    def test_zero_pool_passthrough(self):
        from src.domain_adaptation.style_transfer import ImagePool
        pool = ImagePool(pool_size=0)
        imgs = torch.randn(2, 3, 64, 64)
        out = pool.query(imgs)
        assert torch.allclose(out, imgs)


class TestLSGANLoss:
    def test_real_loss_near_zero_for_ones(self):
        from src.domain_adaptation.style_transfer import LSGANLoss
        criterion = LSGANLoss()
        pred = torch.ones(4, 1, 8, 8)
        loss = criterion(pred, is_real=True)
        assert float(loss.item()) < 0.01

    def test_fake_loss_near_zero_for_zeros(self):
        from src.domain_adaptation.style_transfer import LSGANLoss
        criterion = LSGANLoss()
        pred = torch.zeros(4, 1, 8, 8)
        loss = criterion(pred, is_real=False)
        assert float(loss.item()) < 0.01


# ===========================================================================
# load_accepted_paths
# ===========================================================================

class TestLoadAcceptedPaths:
    def test_roundtrip(self, tmp_path: Path):
        from src.domain_adaptation.pseudo_labeling import load_accepted_paths
        paths = ["/data/img_0001.png", "/data/img_0002.png"]
        manifest = {"accepted": paths, "rejected": [], "stats": {}}
        mf = tmp_path / "manifest.json"
        with open(mf, "w") as f:
            json.dump(manifest, f)
        result = load_accepted_paths(str(mf))
        assert [str(p) for p in result] == paths
