"""
dataset.py
==========
PyTorch Dataset classes for the synthetic glass cleanliness dataset.

Three dataset variants:
  - GlassSegmentationDataset: image + glass mask (binary segmentation)
  - DirtEstimationDataset:    image + glass mask + dirt map (regression)
  - MultitaskDataset:         image + glass mask + dirt map (both tasks)

The datasets are designed to work with the output of the Blender
render pipeline and support flexible split strategies (train/val/test).
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _load_image_cv2(path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load an RGB image and resize to (H, W)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return img  # uint8 (H, W, 3)


def _load_mask_cv2(path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load a greyscale mask and resize to (H, W). Returns float32 [0,1]."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask.astype(np.float32) / 255.0)


def _load_heatmap_cv2(path: str, size: Tuple[int, int]) -> np.ndarray:
    """Load a dirt map and resize. Returns float32 [0,1]."""
    heatmap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if heatmap is None:
        raise FileNotFoundError(f"Cannot load heatmap: {path}")
    heatmap = cv2.resize(heatmap, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return (heatmap.astype(np.float32) / 255.0)


def _discover_scene_ids(
    image_dir: str,
    glass_mask_dir: Optional[str],
    dirt_map_dir: Optional[str],
) -> List[str]:
    """Return sorted list of scene base names present in all given directories."""
    image_dir = Path(image_dir)
    image_stems = {p.stem for p in image_dir.glob("*.png")}

    if glass_mask_dir:
        mask_stems = {p.stem for p in Path(glass_mask_dir).glob("*.png")}
        image_stems &= mask_stems

    if dirt_map_dir:
        dirt_stems = {p.stem for p in Path(dirt_map_dir).glob("*.png")}
        image_stems &= dirt_stems

    return sorted(image_stems)


def _split_ids(
    ids: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Reproducibly split a list of IDs into train/val/test."""
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = ids[:n_train]
    val = ids[n_train: n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class _BaseGlassDataset(Dataset):
    """Common base for all dataset variants."""

    def __init__(
        self,
        scene_ids: List[str],
        image_dir: str,
        image_size: Tuple[int, int],
        transform: Optional[Callable] = None,
    ):
        self.scene_ids = scene_ids
        self.image_dir = Path(image_dir)
        self.image_size = image_size  # (H, W)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _get_image_path(self, scene_id: str) -> str:
        return str(self.image_dir / f"{scene_id}.png")


# ---------------------------------------------------------------------------
# Glass Segmentation Dataset
# ---------------------------------------------------------------------------

class GlassSegmentationDataset(_BaseGlassDataset):
    """
    Dataset for glass surface segmentation.

    Returns
    -------
    dict with keys:
        "image"      : FloatTensor (3, H, W)  — normalized RGB
        "mask"       : FloatTensor (1, H, W)  — binary glass mask in [0, 1]
        "scene_id"   : str
    """

    def __init__(
        self,
        scene_ids: List[str],
        image_dir: str,
        mask_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
    ):
        super().__init__(scene_ids, image_dir, image_size, transform)
        self.mask_dir = Path(mask_dir)

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> "GlassSegmentationDataset":
        ds_cfg = config["dataset"]
        all_ids = _discover_scene_ids(
            ds_cfg["image_dir"],
            ds_cfg.get("mask_dir"),
            None,
        )
        train_ids, val_ids, test_ids = _split_ids(
            all_ids,
            ds_cfg["train_split"],
            ds_cfg["val_split"],
            ds_cfg["test_split"],
        )
        ids_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        return cls(
            scene_ids=ids_map[split],
            image_dir=ds_cfg["image_dir"],
            mask_dir=ds_cfg["mask_dir"],
            image_size=tuple(ds_cfg["image_size"]),
            transform=transform,
        )

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        scene_id = self.scene_ids[idx]
        image = _load_image_cv2(self._get_image_path(scene_id), self.image_size)
        mask = _load_mask_cv2(
            str(self.mask_dir / f"{scene_id}.png"), self.image_size
        )

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors if not already (albumentations ToTensorV2 handles this)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {"image": image, "mask": mask, "scene_id": scene_id}


# ---------------------------------------------------------------------------
# Dirt Estimation Dataset
# ---------------------------------------------------------------------------

class DirtEstimationDataset(_BaseGlassDataset):
    """
    Dataset for dirt level regression.

    Returns
    -------
    dict with keys:
        "image"      : FloatTensor (3, H, W)
        "glass_mask" : FloatTensor (1, H, W)  — used to compute masked loss
        "dirt_map"   : FloatTensor (1, H, W)  — continuous dirt heatmap [0,1]
        "scene_id"   : str
    """

    def __init__(
        self,
        scene_ids: List[str],
        image_dir: str,
        glass_mask_dir: str,
        dirt_map_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        use_masked_input: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__(scene_ids, image_dir, image_size, transform)
        self.glass_mask_dir = Path(glass_mask_dir)
        self.dirt_map_dir = Path(dirt_map_dir)
        self.use_masked_input = use_masked_input

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> "DirtEstimationDataset":
        ds_cfg = config["dataset"]
        all_ids = _discover_scene_ids(
            ds_cfg["image_dir"],
            ds_cfg.get("glass_mask_dir"),
            ds_cfg.get("dirt_map_dir"),
        )
        train_ids, val_ids, test_ids = _split_ids(
            all_ids,
            ds_cfg["train_split"],
            ds_cfg["val_split"],
            ds_cfg["test_split"],
        )
        ids_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        return cls(
            scene_ids=ids_map[split],
            image_dir=ds_cfg["image_dir"],
            glass_mask_dir=ds_cfg["glass_mask_dir"],
            dirt_map_dir=ds_cfg["dirt_map_dir"],
            image_size=tuple(ds_cfg["image_size"]),
            use_masked_input=ds_cfg.get("use_masked_input", False),
            transform=transform,
        )

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        scene_id = self.scene_ids[idx]
        image = _load_image_cv2(self._get_image_path(scene_id), self.image_size)
        glass_mask = _load_mask_cv2(
            str(self.glass_mask_dir / f"{scene_id}.png"), self.image_size
        )
        dirt_map = _load_heatmap_cv2(
            str(self.dirt_map_dir / f"{scene_id}.png"), self.image_size
        )

        if self.transform:
            augmented = self.transform(
                image=image, masks=[glass_mask, dirt_map]
            )
            image = augmented["image"]
            glass_mask, dirt_map = augmented["masks"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(glass_mask, np.ndarray):
            glass_mask = torch.from_numpy(glass_mask).unsqueeze(0).float()
        if isinstance(dirt_map, np.ndarray):
            dirt_map = torch.from_numpy(dirt_map).unsqueeze(0).float()

        if self.use_masked_input:
            # Zero out non-glass pixels in the input image
            image = image * (glass_mask > 0.5).float()

        return {
            "image": image,
            "glass_mask": glass_mask,
            "dirt_map": dirt_map,
            "scene_id": scene_id,
        }


# ---------------------------------------------------------------------------
# Multi-task Dataset
# ---------------------------------------------------------------------------

class MultitaskDataset(_BaseGlassDataset):
    """
    Combined dataset for multi-task training (segmentation + regression).

    Returns
    -------
    dict with keys:
        "image"      : FloatTensor (3, H, W)
        "glass_mask" : FloatTensor (1, H, W)
        "dirt_map"   : FloatTensor (1, H, W)
        "scene_id"   : str
    """

    def __init__(
        self,
        scene_ids: List[str],
        image_dir: str,
        glass_mask_dir: str,
        dirt_map_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Callable] = None,
    ):
        super().__init__(scene_ids, image_dir, image_size, transform)
        self.glass_mask_dir = Path(glass_mask_dir)
        self.dirt_map_dir = Path(dirt_map_dir)

    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> "MultitaskDataset":
        ds_cfg = config["dataset"]
        all_ids = _discover_scene_ids(
            ds_cfg["image_dir"],
            ds_cfg.get("glass_mask_dir"),
            ds_cfg.get("dirt_map_dir"),
        )
        train_ids, val_ids, test_ids = _split_ids(
            all_ids,
            ds_cfg["train_split"],
            ds_cfg["val_split"],
            ds_cfg["test_split"],
        )
        ids_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        return cls(
            scene_ids=ids_map[split],
            image_dir=ds_cfg["image_dir"],
            glass_mask_dir=ds_cfg["glass_mask_dir"],
            dirt_map_dir=ds_cfg["dirt_map_dir"],
            image_size=tuple(ds_cfg["image_size"]),
            transform=transform,
        )

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        scene_id = self.scene_ids[idx]
        image = _load_image_cv2(self._get_image_path(scene_id), self.image_size)
        glass_mask = _load_mask_cv2(
            str(self.glass_mask_dir / f"{scene_id}.png"), self.image_size
        )
        dirt_map = _load_heatmap_cv2(
            str(self.dirt_map_dir / f"{scene_id}.png"), self.image_size
        )

        if self.transform:
            augmented = self.transform(
                image=image, masks=[glass_mask, dirt_map]
            )
            image = augmented["image"]
            glass_mask, dirt_map = augmented["masks"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(glass_mask, np.ndarray):
            glass_mask = torch.from_numpy(glass_mask).unsqueeze(0).float()
        elif isinstance(glass_mask, torch.Tensor) and glass_mask.ndim == 2:
            glass_mask = glass_mask.unsqueeze(0)
        if isinstance(dirt_map, np.ndarray):
            dirt_map = torch.from_numpy(dirt_map).unsqueeze(0).float()
        elif isinstance(dirt_map, torch.Tensor) and dirt_map.ndim == 2:
            dirt_map = dirt_map.unsqueeze(0)

        return {
            "image": image,
            "glass_mask": glass_mask,
            "dirt_map": dirt_map,
            "scene_id": scene_id,
        }
