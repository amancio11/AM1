"""
dataloader.py
=============
DataLoader factory functions for all dataset variants.

Centralises the creation of train/val/test DataLoaders with proper
worker settings, collation, and sampler configuration.
"""

from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import (
    GlassSegmentationDataset,
    DirtEstimationDataset,
    MultitaskDataset,
)
from data.augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)


# ---------------------------------------------------------------------------
# Segmentation DataLoaders
# ---------------------------------------------------------------------------

def build_glass_seg_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for glass segmentation.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    ds_cfg = config["dataset"]
    training_cfg = config["training"]

    train_ds = GlassSegmentationDataset.from_config(
        config, split="train", transform=get_train_transforms(config)
    )
    val_ds = GlassSegmentationDataset.from_config(
        config, split="val", transform=get_val_transforms(config)
    )
    test_ds = GlassSegmentationDataset.from_config(
        config, split="test", transform=get_test_transforms(config)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
        prefetch_factor=ds_cfg.get("prefetch_factor", 2) if ds_cfg.get("num_workers", 4) > 0 else None,
        drop_last=True,
        persistent_workers=ds_cfg.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
        prefetch_factor=ds_cfg.get("prefetch_factor", 2) if ds_cfg.get("num_workers", 4) > 0 else None,
        drop_last=False,
        persistent_workers=ds_cfg.get("num_workers", 4) > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=ds_cfg.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Dirt Estimation DataLoaders
# ---------------------------------------------------------------------------

def build_dirt_est_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for dirt estimation."""
    ds_cfg = config["dataset"]
    training_cfg = config["training"]

    train_ds = DirtEstimationDataset.from_config(
        config, split="train", transform=get_train_transforms(config)
    )
    val_ds = DirtEstimationDataset.from_config(
        config, split="val", transform=get_val_transforms(config)
    )
    test_ds = DirtEstimationDataset.from_config(
        config, split="test", transform=get_test_transforms(config)
    )

    kwargs = _default_loader_kwargs(ds_cfg, training_cfg)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False,
                             batch_size=training_cfg["batch_size"],
                             num_workers=ds_cfg.get("num_workers", 4),
                             pin_memory=ds_cfg.get("pin_memory", True))
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Multi-task DataLoaders
# ---------------------------------------------------------------------------

def build_multitask_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for multi-task learning."""
    ds_cfg = config["dataset"]
    training_cfg = config["training"]

    train_ds = MultitaskDataset.from_config(
        config, split="train", transform=get_train_transforms(config)
    )
    val_ds = MultitaskDataset.from_config(
        config, split="val", transform=get_val_transforms(config)
    )
    test_ds = MultitaskDataset.from_config(
        config, split="test", transform=get_test_transforms(config)
    )

    kwargs = _default_loader_kwargs(ds_cfg, training_cfg)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False,
                             batch_size=training_cfg["batch_size"],
                             num_workers=ds_cfg.get("num_workers", 4),
                             pin_memory=ds_cfg.get("pin_memory", True))
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _default_loader_kwargs(
    ds_cfg: Dict[str, Any], training_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    nw = ds_cfg.get("num_workers", 4)
    return {
        "batch_size": training_cfg["batch_size"],
        "num_workers": nw,
        "pin_memory": ds_cfg.get("pin_memory", True),
        "prefetch_factor": ds_cfg.get("prefetch_factor", 2) if nw > 0 else None,
        "persistent_workers": nw > 0,
    }
