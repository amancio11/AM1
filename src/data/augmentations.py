"""
augmentations.py
================
Albumentations-based augmentation pipelines for training and validation.

All augmentations preserve spatial consistency between the image
and its corresponding masks (glass_mask, dirt_map) since albumentations
applies identical spatial transforms to both.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Normalization constants (ImageNet stats — used with pretrained backbones)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_transforms(
    config: Dict[str, Any],
    split: str = "train",
    to_tensor: bool = True,
) -> A.Compose:
    """
    Build an albumentations Compose pipeline from config.

    Parameters
    ----------
    config : dict
        Full training config dict (the `augmentations` key is read).
    split : str
        "train" or "val".
    to_tensor : bool
        Whether to append ToTensorV2 at the end.

    Returns
    -------
    albumentations.Compose
    """
    aug_cfg = config.get("augmentations", {}).get(split, {})
    norm_cfg = aug_cfg.get("normalize", {})

    mean = norm_cfg.get("mean", list(IMAGENET_MEAN))
    std = norm_cfg.get("std", list(IMAGENET_STD))

    if split == "train":
        transforms = _build_train_transforms(aug_cfg)
    else:
        transforms = []

    transforms.append(
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
    )
    if to_tensor:
        transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        additional_targets={
            "mask": "mask",      # glass mask
            "dirt_map": "mask",  # dirt map (treated as second mask)
        },
    )


def build_multimask_transforms(
    config: Dict[str, Any],
    split: str = "train",
    to_tensor: bool = True,
) -> A.Compose:
    """
    Pipeline variant that supports `masks` (list of masks) via
    albumentations' multi-mask target support.
    """
    aug_cfg = config.get("augmentations", {}).get(split, {})
    norm_cfg = aug_cfg.get("normalize", {})
    mean = norm_cfg.get("mean", list(IMAGENET_MEAN))
    std = norm_cfg.get("std", list(IMAGENET_STD))

    if split == "train":
        transforms = _build_train_transforms(aug_cfg)
    else:
        transforms = []

    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    if to_tensor:
        transforms.append(ToTensorV2())

    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def _build_train_transforms(aug_cfg: Dict[str, Any]):
    transforms = []

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.3))

    if aug_cfg.get("random_rotate_90", True):
        transforms.append(A.RandomRotate90(p=0.3))

    # Geometric distortions
    elastic_cfg = aug_cfg.get("elastic_transform")
    if elastic_cfg:
        transforms.append(
            A.ElasticTransform(
                alpha=elastic_cfg.get("alpha", 120),
                sigma=elastic_cfg.get("sigma", 120),
                # alpha_affine removed in newer albumentations versions
                p=elastic_cfg.get("p", 0.2),
            )
        )

    grid_cfg = aug_cfg.get("grid_distortion")
    if grid_cfg:
        transforms.append(A.GridDistortion(p=grid_cfg.get("p", 0.2)))

    # Photometric augmentations (image only, not masks)
    bc_cfg = aug_cfg.get("random_brightness_contrast")
    if bc_cfg:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc_cfg.get("brightness_limit", 0.3),
                contrast_limit=bc_cfg.get("contrast_limit", 0.3),
                p=bc_cfg.get("p", 0.7),
            )
        )

    noise_cfg = aug_cfg.get("gaussian_noise")
    if noise_cfg:
        transforms.append(
            A.GaussNoise(
                var_limit=tuple(noise_cfg.get("var_limit", [10.0, 50.0])),
                p=noise_cfg.get("p", 0.3),
            )
        )

    blur_cfg = aug_cfg.get("motion_blur")
    if blur_cfg:
        transforms.append(
            A.MotionBlur(
                blur_limit=blur_cfg.get("blur_limit", 7),
                p=blur_cfg.get("p", 0.2),
            )
        )

    jitter_cfg = aug_cfg.get("color_jitter")
    if jitter_cfg:
        transforms.append(
            A.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.2),
                contrast=jitter_cfg.get("contrast", 0.2),
                saturation=jitter_cfg.get("saturation", 0.2),
                hue=jitter_cfg.get("hue", 0.1),
                p=jitter_cfg.get("p", 0.5),
            )
        )

    # Optional: random shadow / sun flare to simulate reflection/glare
    transforms.append(
        A.OneOf(
            [
                A.RandomShadow(p=1.0),
                A.RandomSunFlare(
                    flare_roi=(0.0, 0.0, 1.0, 0.5),
                    angle_lower=0.5,
                    p=1.0,
                ),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, p=1.0),
            ],
            p=0.15,
        )
    )

    # Blur / sharpness
    transforms.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=70, quality_upper=100, p=1.0),
            ],
            p=0.2,
        )
    )

    return transforms


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    return build_multimask_transforms(config, split="train", to_tensor=True)


def get_val_transforms(config: Dict[str, Any]) -> A.Compose:
    return build_multimask_transforms(config, split="val", to_tensor=True)


def get_test_transforms(config: Dict[str, Any]) -> A.Compose:
    return build_multimask_transforms(config, split="val", to_tensor=True)
