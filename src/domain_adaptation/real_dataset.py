"""
real_dataset.py
===============
Dataset classes for real-world drone facade images.

Supports three label scenarios:
  1. Fully unlabeled  — only RGB images (for pseudo-labeling / style transfer)
  2. Weakly labeled   — images with user-drawn approximate glass masks
  3. Pseudo-labeled   — images with auto-generated labels from a pretrained model

The `RealFacadeDataset` is intentionally flexible: it reads whatever label
files are available and signals their presence via `has_glass_labels` and
`has_dirt_labels` flags so downstream code can adapt.

`PseudoLabeledDataset` is a subclass that merges real images with
their corresponding pseudo-labels + confidence maps so the fine-tuner
can weight uncertain samples during loss computation.

`MixedDataset` combines real (or pseudo-labeled) samples with synthetic
samples at a configurable ratio, enabling semi-supervised fine-tuning.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_rgb(path: str, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def _load_gray(path: str, size: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32) / 255.0


def _discover_images(directory: str, extensions: Tuple[str, ...]) -> List[Path]:
    """Recursively find all images in a directory, sorted."""
    p = Path(directory)
    found = [f for ext in extensions for f in sorted(p.rglob(f"*{ext}"))]
    return sorted(set(found), key=lambda x: x.stem)


def _to_tensor_image(img_np: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float32 tensor (NOT normalized)."""
    return torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0


def _to_tensor_mask(mask_np: np.ndarray) -> torch.Tensor:
    """(H,W) float32 → (1,H,W) float32 tensor."""
    if mask_np.ndim == 3:
        mask_np = mask_np.squeeze(-1)
    return torch.from_numpy(mask_np).unsqueeze(0).float()


# ---------------------------------------------------------------------------
# Real Facade Dataset
# ---------------------------------------------------------------------------

class RealFacadeDataset(Dataset):
    """
    Dataset for real-world building facade images from drone capture.

    Label availability is fully optional — the dataset works with just images.

    Parameters
    ----------
    image_paths : list of Path
        Sorted list of image file paths.
    glass_mask_dir : str or None
        Directory containing glass segmentation masks (same stems as images).
    dirt_map_dir : str or None
        Directory containing dirt map labels (same stems as images).
    image_size : tuple (H, W)
    transform : albumentations Compose or None
        Applied to image and all available masks jointly.
    normalize : bool
        Whether to apply ImageNet normalization before tensorizing.
        Set False when using albumentations Normalize inside transform.
    """

    def __init__(
        self,
        image_paths: List[Path],
        glass_mask_dir: Optional[str] = None,
        dirt_map_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        self.image_paths = image_paths
        self.glass_mask_dir = Path(glass_mask_dir) if glass_mask_dir else None
        self.dirt_map_dir = Path(dirt_map_dir) if dirt_map_dir else None
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        self.has_glass_labels = self._check_labels_exist(self.glass_mask_dir)
        self.has_dirt_labels = self._check_labels_exist(self.dirt_map_dir)

        # Pre-validate label files exist for labeled images
        if self.has_glass_labels or self.has_dirt_labels:
            self._validate_label_files()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path = self.image_paths[idx]
        image = _load_rgb(str(img_path), self.image_size)

        sample: Dict[str, any] = {"image_path": str(img_path)}
        masks = []
        mask_keys = []

        if self.has_glass_labels:
            glass_path = self.glass_mask_dir / f"{img_path.stem}.png"
            glass_mask = _load_gray(str(glass_path), self.image_size)
            masks.append(glass_mask)
            mask_keys.append("glass_mask")

        if self.has_dirt_labels:
            dirt_path = self.dirt_map_dir / f"{img_path.stem}.png"
            dirt_map = _load_gray(str(dirt_path), self.image_size)
            masks.append(dirt_map)
            mask_keys.append("dirt_map")

        # Apply joint augmentation
        if self.transform is not None:
            if masks:
                augmented = self.transform(image=image, masks=masks)
                image = augmented["image"]
                masks = augmented["masks"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Tensorize
        if isinstance(image, np.ndarray):
            if self.normalize:
                image = (image.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = _to_tensor_image(image)
        sample["image"] = image

        for i, key in enumerate(mask_keys):
            m = masks[i] if masks else None
            if m is not None:
                if isinstance(m, np.ndarray):
                    sample[key] = _to_tensor_mask(m)
                elif isinstance(m, torch.Tensor):
                    sample[key] = m.unsqueeze(0) if m.ndim == 2 else m

        # Placeholder zero tensors for missing labels
        h, w = self.image_size
        if "glass_mask" not in sample:
            sample["glass_mask"] = None       # caller must check
        if "dirt_map" not in sample:
            sample["dirt_map"] = None

        sample["has_glass_label"] = self.has_glass_labels
        sample["has_dirt_label"] = self.has_dirt_labels
        return sample

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Dict,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> "RealFacadeDataset":
        da_cfg = config.get("domain_adaptation", config)
        real_cfg = da_cfg.get("real_dataset", {})
        paths_cfg = da_cfg.get("paths", {})

        exts = tuple(real_cfg.get("extensions", [".png", ".jpg", ".jpeg"]))
        all_images = _discover_images(paths_cfg["real_images_dir"], exts)

        if not all_images:
            raise FileNotFoundError(
                f"No images found in: {paths_cfg['real_images_dir']}"
            )

        # Reproducible split
        rng = random.Random(42)
        images = list(all_images)
        rng.shuffle(images)
        n = len(images)
        train_frac = real_cfg.get("train_split", 0.8)
        n_train = int(n * train_frac)

        if split == "train":
            selected = images[:n_train]
        else:
            selected = images[n_train:]

        glass_dir = paths_cfg.get("real_glass_masks_dir") if real_cfg.get("has_glass_labels") else None
        dirt_dir = paths_cfg.get("real_dirt_maps_dir") if real_cfg.get("has_dirt_labels") else None

        return cls(
            image_paths=selected,
            glass_mask_dir=glass_dir,
            dirt_map_dir=dirt_dir,
            image_size=tuple(real_cfg.get("image_size", [512, 512])),
            transform=transform,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_labels_exist(self, label_dir: Optional[Path]) -> bool:
        if label_dir is None:
            return False
        if not label_dir.exists():
            return False
        return any(label_dir.glob("*.png"))

    def _validate_label_files(self) -> None:
        """Warn about images with no matching label file (non-fatal)."""
        missing = []
        for img_path in self.image_paths:
            if self.has_glass_labels:
                lp = self.glass_mask_dir / f"{img_path.stem}.png"
                if not lp.exists():
                    missing.append(str(lp))
            if self.has_dirt_labels:
                lp = self.dirt_map_dir / f"{img_path.stem}.png"
                if not lp.exists():
                    missing.append(str(lp))
        if missing:
            import warnings
            warnings.warn(f"Missing {len(missing)} label files. First: {missing[0]}")


# ---------------------------------------------------------------------------
# Pseudo-Labeled Dataset
# ---------------------------------------------------------------------------

class PseudoLabeledDataset(Dataset):
    """
    Dataset that pairs real images with auto-generated pseudo-labels.

    Each pseudo-label comes with a per-pixel confidence weight map.
    High-confidence pixels contribute more to the loss during fine-tuning.
    Uncertain pixels (confidence < threshold) are optionally masked out.

    Parameters
    ----------
    image_paths : list of Path
    pseudo_label_dir : str
        Directory produced by PseudoLabeler. Expects files:
          {stem}_glass.png         — binary glass mask [0,255]
          {stem}_dirt.png          — dirt heatmap [0,255]
          {stem}_glass_conf.png    — glass confidence [0,255]
          {stem}_dirt_conf.png     — dirt confidence [0,255]
    confidence_threshold : float
        Pixels below this confidence are zeroed in the weight map.
    image_size : tuple (H, W)
    transform : callable or None
    """

    def __init__(
        self,
        image_paths: List[Path],
        pseudo_label_dir: str,
        glass_confidence_threshold: float = 0.85,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.pseudo_dir = Path(pseudo_label_dir)
        self.glass_thresh = glass_confidence_threshold
        self.image_size = image_size
        self.transform = transform

        # Filter to only images that have pseudo-labels
        self.valid_paths = [
            p for p in image_paths
            if (self.pseudo_dir / f"{p.stem}_glass.png").exists()
        ]
        if not self.valid_paths:
            raise ValueError(
                f"No pseudo-labels found in {pseudo_label_dir}. "
                "Run PseudoLabeler first."
            )

    def __len__(self) -> int:
        return len(self.valid_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path = self.valid_paths[idx]
        stem = img_path.stem

        image = _load_rgb(str(img_path), self.image_size)

        def _pl(fname: str) -> np.ndarray:
            return _load_gray(str(self.pseudo_dir / fname), self.image_size)

        glass_mask = _pl(f"{stem}_glass.png")
        dirt_map = _pl(f"{stem}_dirt.png")
        glass_conf = _pl(f"{stem}_glass_conf.png")
        dirt_conf = _pl(f"{stem}_dirt_conf.png")

        if self.transform is not None:
            aug = self.transform(
                image=image,
                masks=[glass_mask, dirt_map, glass_conf, dirt_conf],
            )
            image = aug["image"]
            glass_mask, dirt_map, glass_conf, dirt_conf = aug["masks"]

        if isinstance(image, np.ndarray):
            image = (image.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        def _t(arr):
            if isinstance(arr, np.ndarray):
                return _to_tensor_mask(arr)
            return arr.unsqueeze(0) if arr.ndim == 2 else arr

        glass_mask_t = _t(glass_mask)
        dirt_map_t = _t(dirt_map)
        glass_conf_t = _t(glass_conf)
        dirt_conf_t = _t(dirt_conf)

        # Build sample-level confidence weight (mean glass confidence)
        sample_weight = float(glass_conf_t.mean().item())

        return {
            "image": image,
            "glass_mask": glass_mask_t,
            "dirt_map": dirt_map_t,
            "glass_confidence": glass_conf_t,
            "dirt_confidence": dirt_conf_t,
            "sample_weight": sample_weight,
            "image_path": str(img_path),
            "is_pseudo_label": True,
        }


# ---------------------------------------------------------------------------
# Mixed Dataset (synthetic + real / pseudo-labeled)
# ---------------------------------------------------------------------------

class MixedDataset(Dataset):
    """
    Combines a synthetic dataset and a real/pseudo-labeled dataset at a
    configurable ratio using weighted sampling.

    The sampler ensures that each batch contains approximately
    `synthetic_ratio` fraction of synthetic samples.

    Parameters
    ----------
    synthetic_dataset : Dataset
    real_dataset : Dataset
    synthetic_ratio : float ∈ [0, 1]
        Fraction of samples from the synthetic domain.
    """

    def __init__(
        self,
        synthetic_dataset: Dataset,
        real_dataset: Dataset,
        synthetic_ratio: float = 0.3,
    ):
        self.synthetic_ds = synthetic_dataset
        self.real_ds = real_dataset
        self.synthetic_ratio = synthetic_ratio

        n_syn = len(synthetic_dataset)
        n_real = len(real_dataset)
        self._total = n_syn + n_real

        # Assign domain labels for collation
        self._syn_len = n_syn
        self._real_len = n_real

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Dict:
        if idx < self._syn_len:
            sample = self.synthetic_ds[idx]
            sample["domain"] = "synthetic"
        else:
            sample = self.real_ds[idx - self._syn_len]
            sample["domain"] = "real"
        return sample

    def build_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler so each batch has ~synthetic_ratio
        fraction of synthetic samples.
        """
        syn_weight = self.synthetic_ratio / max(self._syn_len, 1)
        real_weight = (1.0 - self.synthetic_ratio) / max(self._real_len, 1)

        weights = (
            [syn_weight] * self._syn_len
            + [real_weight] * self._real_len
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=self._total,
            replacement=True,
        )
