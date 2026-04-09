"""
train_glass.py
==============
Training script for the Glass Surface Segmentation model.

Usage:
    python src/training/train_glass.py --config configs/glass_seg_config.yaml
    python src/training/train_glass.py --config configs/glass_seg_config.yaml --resume checkpoints/glass_seg/ckpt_epoch0010_0.8500.pth
"""

import sys
import os
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from data.dataloader import build_glass_seg_dataloaders
from models.glass_segmentation import GlassSegmentationModel
from models.losses import CombinedSegLoss
from training.trainer import Trainer
from training.scheduler import build_optimizer, build_scheduler
from evaluation.metrics import segmentation_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task-specific Trainer subclass
# ---------------------------------------------------------------------------

class GlassSegTrainer(Trainer):
    """Extends Trainer with glass segmentation forward pass."""

    def __init__(self, loss_fn: CombinedSegLoss, threshold: float = 0.5, **kwargs):
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.threshold = threshold

    def _compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        mask = batch["mask"]
        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            logits = self.model(image)
            loss_dict = self.loss_fn(logits, mask)
        return loss_dict

    def _extract_preds_targets(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            logits = self.model(batch["image"])
        preds = (torch.sigmoid(logits) > self.threshold).float()
        return preds, batch["mask"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Glass Segmentation Model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    # -- Dataloaders --
    train_loader, val_loader, test_loader = build_glass_seg_dataloaders(config)
    logger.info(
        f"Dataset: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val / "
        f"{len(test_loader.dataset)} test"
    )

    # -- Model --
    model = GlassSegmentationModel.from_config(config)
    logger.info(
        f"Model: {config['model']['architecture']} | "
        f"Encoder: {config['model']['encoder']} | "
        f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # -- Loss --
    loss_cfg = config["loss"]
    loss_fn = CombinedSegLoss(
        bce_weight=loss_cfg.get("bce_weight", 0.4),
        dice_weight=loss_cfg.get("dice_weight", 0.4),
        focal_weight=loss_cfg.get("focal_weight", 0.2),
        pos_weight=loss_cfg.get("pos_weight"),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )

    # -- Optimizer & Scheduler --
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # -- Metrics callback --
    def metrics_fn(preds, targets):
        return segmentation_metrics(
            preds,
            targets,
            threshold=config["metrics"].get("threshold", 0.5),
        )

    # -- Trainer --
    trainer = GlassSegTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        metrics_fn=metrics_fn,
        experiment_name=config["experiment"]["name"],
        threshold=config["inference"].get("threshold", 0.5),
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # -- Train --
    history = trainer.fit()
    logger.info(f"Best metric: {history['best_metric']:.4f}")
    logger.info(f"Best checkpoint: {history['best_checkpoint']}")


if __name__ == "__main__":
    main()
