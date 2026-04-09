"""
run_adaptation.py
=================
CLI entry point for the full synthetic-to-real domain adaptation pipeline.

Stages (can be run independently or chained)
---------------------------------------------
1. style-transfer   — Train CycleGAN on synthetic vs real images
2. pseudo-label     — Generate pseudo-labels for unlabeled real images
3. finetune         — Fine-tune segmentation/multitask model on mixed data

Usage
-----
    # Full pipeline
    python -m src.domain_adaptation.run_adaptation \\
        --config configs/domain_adaptation.yaml \\
        --model-checkpoint checkpoints/multitask/best.pth \\
        --stages style-transfer pseudo-label finetune

    # Pseudo-label and finetune only (CycleGAN already trained)
    python -m src.domain_adaptation.run_adaptation \\
        --config configs/domain_adaptation.yaml \\
        --model-checkpoint checkpoints/multitask/best.pth \\
        --stages pseudo-label finetune \\
        --cyclegan-checkpoint checkpoints/cyclegan/cyclegan_epoch0050.pth

    # Fine-tune only (pseudo-labels already saved)
    python -m src.domain_adaptation.run_adaptation \\
        --config configs/domain_adaptation.yaml \\
        --model-checkpoint checkpoints/multitask/best.pth \\
        --stages finetune
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage: Style Transfer
# ---------------------------------------------------------------------------

def run_style_transfer(config: Dict, cyclegan_ckpt: Optional[str] = None) -> None:
    """Train (or resume) CycleGAN on synthetic↔real pairs."""
    from torch.utils.data import DataLoader

    from src.domain_adaptation.style_transfer import CycleGANTrainer
    from src.domain_adaptation.real_dataset import RealFacadeDataset, _discover_images
    from src.domain_adaptation.domain_augmentations import build_domain_randomization_transform
    from src.data.dataset import MultitaskDataset

    da_cfg = config.get("domain_adaptation", config)
    st_cfg = da_cfg.get("style_transfer", {})
    ft_cfg = da_cfg.get("finetuning", {})

    real_size = tuple(da_cfg.get("real_dataset", {}).get("image_size", [256, 256]))
    aug_cfg = da_cfg.get("domain_augmentations", {})

    # -- Real images (domain B) ------------------------------------------------
    real_transform = build_domain_randomization_transform(aug_cfg, image_size=real_size)
    real_ds = RealFacadeDataset.from_config(config, split="train", transform=real_transform)

    # -- Synthetic images (domain A) -------------------------------------------
    syn_ds = MultitaskDataset.from_config(config, split="train")

    batch_size = st_cfg.get("batch_size", 1)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    syn_loader = DataLoader(syn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    trainer = CycleGANTrainer.from_config(config, syn_loader, real_loader)

    if cyclegan_ckpt:
        start_epoch = trainer.load_checkpoint(cyclegan_ckpt)
        logger.info(f"Resuming CycleGAN from epoch {start_epoch}")

    logger.info(f"Starting CycleGAN training for {trainer.n_epochs} epochs …")
    trainer.train()
    logger.info("Style transfer training complete.")


# ---------------------------------------------------------------------------
# Stage: Pseudo-Labeling
# ---------------------------------------------------------------------------

def run_pseudo_labeling(
    config: Dict,
    model_ckpt: str,
    force_regen: bool = False,
) -> str:
    """
    Generate pseudo-labels for unlabeled real images using a pretrained model.

    Returns
    -------
    Path to the saved manifest.json.
    """
    da_cfg = config.get("domain_adaptation", config)
    paths = da_cfg.get("paths", {})
    pseudo_dir = paths.get("pseudo_label_dir", "data/pseudo_labels")

    # Load inference predictor
    from src.inference.predictor import Predictor
    predictor = Predictor.from_checkpoints(
        multitask_checkpoint=model_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    from src.domain_adaptation.pseudo_labeling import PseudoLabeler
    labeler = PseudoLabeler.from_config(config, predictor)

    manifest = labeler.generate(force_regen=force_regen)
    logger.info(
        f"Pseudo-labeling: {manifest['stats']['num_accepted']} accepted / "
        f"{manifest['stats']['total']} total "
        f"({manifest['stats']['acceptance_rate']:.1%})"
    )

    manifest_path = str(Path(pseudo_dir) / "manifest.json")
    return manifest_path


# ---------------------------------------------------------------------------
# Stage: Fine-Tuning
# ---------------------------------------------------------------------------

def run_finetuning(
    config: Dict,
    model_ckpt: str,
    pseudo_manifest: Optional[str] = None,
    model_type: str = "multitask",
) -> None:
    """
    Fine-tune the pretrained model on real/pseudo-labeled data.

    Parameters
    ----------
    config : dict
    model_ckpt : str            Path to pretrained model .pth
    pseudo_manifest : str|None  Path to manifest.json from pseudo-labeler
    model_type : str            'multitask' | 'glass_seg' | 'dirt_est'
    """
    from torch.utils.data import DataLoader

    da_cfg = config.get("domain_adaptation", config)
    ft_cfg = da_cfg.get("finetuning", {})
    pl_cfg = da_cfg.get("pseudo_labeling", {})
    paths = da_cfg.get("paths", {})

    real_size = tuple(da_cfg.get("real_dataset", {}).get("image_size", [512, 512]))
    aug_cfg = da_cfg.get("domain_augmentations", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Load model -----------------------------------------------------------
    model = _load_model(model_ckpt, model_type, device)

    # -- Build dataset --------------------------------------------------------
    from src.domain_adaptation.domain_augmentations import (
        build_domain_randomization_transform,
        build_real_world_val_transform,
    )

    train_transform = build_domain_randomization_transform(aug_cfg, real_size)
    val_transform = build_real_world_val_transform(real_size)

    if pseudo_manifest is not None:
        from src.domain_adaptation.pseudo_labeling import load_accepted_paths
        from src.domain_adaptation.real_dataset import PseudoLabeledDataset

        accepted = load_accepted_paths(pseudo_manifest)
        split_n = int(len(accepted) * 0.9)
        train_paths, val_paths = accepted[:split_n], accepted[split_n:]

        train_ds = PseudoLabeledDataset(
            image_paths=train_paths,
            pseudo_label_dir=paths.get("pseudo_label_dir", "data/pseudo_labels"),
            glass_confidence_threshold=pl_cfg.get("glass_confidence_threshold", 0.85),
            image_size=real_size,
            transform=train_transform,
        )
        val_ds = PseudoLabeledDataset(
            image_paths=val_paths,
            pseudo_label_dir=paths.get("pseudo_label_dir", "data/pseudo_labels"),
            image_size=real_size,
            transform=val_transform,
        )
    else:
        # Weakly-labeled or fully-labeled real dataset
        from src.domain_adaptation.real_dataset import RealFacadeDataset
        train_ds = RealFacadeDataset.from_config(config, split="train", transform=train_transform)
        val_ds = RealFacadeDataset.from_config(config, split="val", transform=val_transform)

    # Optionally mix with synthetic data
    if ft_cfg.get("mix_synthetic", True):
        from src.domain_adaptation.real_dataset import MixedDataset
        from src.data.dataset import MultitaskDataset
        syn_ds = MultitaskDataset.from_config(config, split="train")
        mixed_ds = MixedDataset(
            synthetic_dataset=syn_ds,
            real_dataset=train_ds,
            synthetic_ratio=ft_cfg.get("synthetic_ratio", 0.3),
        )
        sampler = mixed_ds.build_weighted_sampler()
        train_loader = DataLoader(
            mixed_ds,
            batch_size=ft_cfg.get("batch_size", 8),
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=ft_cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=ft_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # -- Fine-tune ------------------------------------------------------------
    from src.domain_adaptation.finetuner import DomainAdaptationFinetuner
    finetuner = DomainAdaptationFinetuner.from_config(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    logger.info("Starting fine-tuning …")
    best = finetuner.fit()
    logger.info(f"Fine-tuning complete. Best metrics: {best}")


# ---------------------------------------------------------------------------
# Model loader helper
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, model_type: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_cfg = ckpt.get("config", {})

    if model_type == "multitask":
        from src.models.multitask_model import MultitaskFacadeModel
        model = MultitaskFacadeModel.from_config(model_cfg) if model_cfg else MultitaskFacadeModel()
    elif model_type == "glass_seg":
        from src.models.glass_segmentation import GlassSegmentationModel
        model = GlassSegmentationModel.from_config(model_cfg) if model_cfg else GlassSegmentationModel()
    elif model_type == "dirt_est":
        from src.models.dirt_estimation import DirtEstimationModel
        model = DirtEstimationModel.from_config(model_cfg) if model_cfg else DirtEstimationModel()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.train()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain adaptation pipeline for synthetic→real transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/domain_adaptation.yaml",
        help="Path to domain_adaptation.yaml",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to pretrained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--cyclegan-checkpoint",
        type=str,
        default=None,
        help="(Optional) Resume CycleGAN training from this checkpoint",
    )
    parser.add_argument(
        "--pseudo-manifest",
        type=str,
        default=None,
        help="(Optional) Path to existing pseudo-label manifest.json to skip re-generation",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["style-transfer", "pseudo-label", "finetune"],
        default=["pseudo-label", "finetune"],
        help="Stages to run in order",
    )
    parser.add_argument(
        "--model-type",
        choices=["multitask", "glass_seg", "dirt_est"],
        default="multitask",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Re-generate pseudo-labels even if they already exist",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pseudo_manifest = args.pseudo_manifest

    for stage in args.stages:
        logger.info(f"===== Stage: {stage} =====")

        if stage == "style-transfer":
            run_style_transfer(config, cyclegan_ckpt=args.cyclegan_checkpoint)

        elif stage == "pseudo-label":
            if args.model_checkpoint is None:
                logger.error("--model-checkpoint required for pseudo-label stage")
                sys.exit(1)
            pseudo_manifest = run_pseudo_labeling(
                config,
                model_ckpt=args.model_checkpoint,
                force_regen=args.force_regen,
            )

        elif stage == "finetune":
            if args.model_checkpoint is None:
                logger.error("--model-checkpoint required for finetune stage")
                sys.exit(1)
            run_finetuning(
                config=config,
                model_ckpt=args.model_checkpoint,
                pseudo_manifest=pseudo_manifest,
                model_type=args.model_type,
            )

    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
