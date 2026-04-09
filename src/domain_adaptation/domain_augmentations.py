"""
domain_augmentations.py
=======================
Heavy-duty augmentation pipeline for domain randomization.

Designed to bridge the gap between clean synthetic renders and real-world
drone imagery, which exhibits:
  - Optical blur (motion, defocus, atmospheric)
  - Specular glass reflections and lens artefacts
  - Sensor noise (Gaussian, ISO, shot noise)
  - JPEG compression artefacts from drone video encoding
  - Illumination variation (brightness, gamma, shadows)
  - Color channel shifts from UAV camera white-balance drift
  - Rain streaks / water droplets on the drone lens

All augmentations are albumentations-based and return plain dicts so they
compose cleanly with the existing `build_transforms()` pipelines.

Usage
-----
    from src.domain_adaptation.domain_augmentations import (
        build_domain_randomization_transform,
        build_real_to_synthetic_transform,
    )

    tfm = build_domain_randomization_transform(cfg["domain_augmentations"])
    sample = tfm(image=image, masks=[glass_mask, dirt_map])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _p(cfg: Dict, key: str, default: float) -> float:
    """Safe probability getter."""
    return float(cfg.get(key, {}).get("p", default))


def _kl(cfg: Dict, key: str, kparam: str, default):
    """Get a named param from a nested augmentation config block."""
    return cfg.get(key, {}).get(kparam, default)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_domain_randomization_transform(
    aug_cfg: Dict,
    image_size: Tuple[int, int] = (512, 512),
) -> A.Compose:
    """
    Aggressive domain-randomization pipeline to make synthetic images
    look more like real drone captures.

    Parameters
    ----------
    aug_cfg : dict
        The ``domain_augmentations`` block from domain_adaptation.yaml.
    image_size : (H, W)
        Spatial dimensions for the final resize.

    Returns
    -------
    albumentations.Compose
        Accepts ``image`` and optional ``masks`` list.
    """
    aug = aug_cfg  # shorthand

    # -- Geometric (mild, already handled in main augmentation) ----------------
    geometric = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=5,
            border_mode=0,    # constant black border
            p=0.4,
        ),
    ]

    # -- Optical blur  ---------------------------------------------------------
    blur_ops: List[A.BasicTransform] = []

    if _kl(aug, "motion_blur", "enabled", True):
        blur_ops.append(
            A.MotionBlur(
                blur_limit=tuple(_kl(aug, "motion_blur", "blur_limit", [3, 15])),
                p=_p(aug, "motion_blur", 0.5),
            )
        )
    if _kl(aug, "gaussian_blur", "enabled", True):
        blur_ops.append(
            A.GaussianBlur(
                blur_limit=tuple(_kl(aug, "gaussian_blur", "blur_limit", [3, 9])),
                p=_p(aug, "gaussian_blur", 0.4),
            )
        )
    if _kl(aug, "defocus_blur", "enabled", True):
        blur_ops.append(
            A.Defocus(
                radius=tuple(_kl(aug, "defocus_blur", "radius", [1, 5])),
                alias_blur=_kl(aug, "defocus_blur", "alias_blur", 0.1),
                p=_p(aug, "defocus_blur", 0.3),
            )
        )
    # Apply at most ONE blur type per sample
    blur_block = A.OneOf(blur_ops, p=0.6) if blur_ops else A.NoOp()

    # -- Lens / lighting artefacts --------------------------------------------
    optics_ops: List[A.BasicTransform] = []

    if _kl(aug, "sun_flare", "enabled", True):
        optics_ops.append(
            A.RandomSunFlare(
                flare_roi=tuple(_kl(aug, "sun_flare", "flare_roi", [0, 0, 1, 0.5])),
                angle_lower=_kl(aug, "sun_flare", "angle_lower", 0.0),
                angle_upper=_kl(aug, "sun_flare", "angle_upper", 1.0),
                num_flare_circles_lower=_kl(aug, "sun_flare", "num_flare_circles_lower", 1),
                num_flare_circles_upper=_kl(aug, "sun_flare", "num_flare_circles_upper", 6),
                src_radius=_kl(aug, "sun_flare", "src_radius", 200),
                p=_p(aug, "sun_flare", 0.2),
            )
        )

    if _kl(aug, "rain_drops", "enabled", False):
        optics_ops.append(
            A.RandomRain(
                slant_lower=_kl(aug, "rain_drops", "slant_lower", -10),
                slant_upper=_kl(aug, "rain_drops", "slant_upper", 10),
                drop_length=_kl(aug, "rain_drops", "drop_length", 10),
                drop_width=_kl(aug, "rain_drops", "drop_width", 1),
                drop_color=tuple(_kl(aug, "rain_drops", "drop_color", [200, 200, 200])),
                rain_type=_kl(aug, "rain_drops", "rain_type", None),
                p=_p(aug, "rain_drops", 0.15),
            )
        )

    optics_block = A.OneOf(optics_ops, p=0.25) if optics_ops else A.NoOp()

    # -- Brightness / color ---------------------------------------------------
    color_ops: List[A.BasicTransform] = [
        A.RandomBrightnessContrast(
            brightness_limit=tuple(
                _kl(aug, "random_brightness", "brightness_limit", [-0.3, 0.3])
            ),
            contrast_limit=tuple(
                _kl(aug, "random_brightness", "contrast_limit", [-0.2, 0.2])
            ),
            p=_p(aug, "random_brightness", 0.7),
        ),
        A.RandomGamma(
            gamma_limit=tuple(_kl(aug, "random_gamma", "gamma_limit", [70, 130])),
            p=_p(aug, "random_gamma", 0.4),
        ),
        A.CLAHE(
            clip_limit=_kl(aug, "clahe", "clip_limit", 4.0),
            tile_grid_size=tuple(_kl(aug, "clahe", "tile_grid_size", [8, 8])),
            p=_p(aug, "clahe", 0.3),
        ),
        A.HueSaturationValue(
            hue_shift_limit=_kl(aug, "hue_saturation_value", "hue_shift_limit", 10),
            sat_shift_limit=_kl(aug, "hue_saturation_value", "sat_shift_limit", 20),
            val_shift_limit=_kl(aug, "hue_saturation_value", "val_shift_limit", 15),
            p=_p(aug, "hue_saturation_value", 0.5),
        ),
        A.RGBShift(
            r_shift_limit=_kl(aug, "rgb_shift", "r_shift_limit", 15),
            g_shift_limit=_kl(aug, "rgb_shift", "g_shift_limit", 15),
            b_shift_limit=_kl(aug, "rgb_shift", "b_shift_limit", 15),
            p=_p(aug, "rgb_shift", 0.3),
        ),
    ]

    # -- Sensor noise ---------------------------------------------------------
    noise_ops: List[A.BasicTransform] = []

    if _kl(aug, "gaussian_noise", "enabled", True):
        noise_ops.append(
            A.GaussNoise(
                var_limit=tuple(_kl(aug, "gaussian_noise", "var_limit", [10, 50])),
                p=_p(aug, "gaussian_noise", 0.4),
            )
        )
    if _kl(aug, "iso_noise", "enabled", True):
        noise_ops.append(
            A.ISONoise(
                color_shift=tuple(_kl(aug, "iso_noise", "color_shift", [0.01, 0.05])),
                intensity=tuple(_kl(aug, "iso_noise", "intensity", [0.1, 0.5])),
                p=_p(aug, "iso_noise", 0.3),
            )
        )
    noise_block = A.OneOf(noise_ops, p=0.5) if noise_ops else A.NoOp()

    # -- JPEG compression -----------------------------------------------------
    jpeg_block = A.ImageCompression(
        quality_lower=_kl(aug, "jpeg_compression", "quality_lower", 60),
        quality_upper=_kl(aug, "jpeg_compression", "quality_upper", 95),
        p=_p(aug, "jpeg_compression", 0.4),
    )

    # -- Normalize + resize ---------------------------------------------------
    final = [
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    pipeline = A.Compose(
        geometric
        + [blur_block, optics_block]
        + color_ops
        + [noise_block, jpeg_block]
        + final,
        additional_targets={},   # caller passes masks list
    )
    return pipeline


def build_real_world_val_transform(
    image_size: Tuple[int, int] = (512, 512),
) -> A.Compose:
    """
    Minimal (deterministic) transform for real-image validation split.
    Only resize + normalize — no stochastic augmentations.
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_synthetic_domain_randomization(
    image_size: Tuple[int, int] = (512, 512),
) -> A.Compose:
    """
    Domain randomization applied to **synthetic** images during training
    so the model sees more realistic inputs without any CycleGAN overhead.

    Lighter than `build_domain_randomization_transform` — focuses on
    the distribution mismatch between perfect renders and real cameras.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.15),
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=12, p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.25),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
