"""
train_dirt.py
=============
Training script for the Dirt Estimation (regression) model.

Usage:
    python src/training/train_dirt.py --config configs/dirt_est_config.yaml
"""

import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.dataloader import build_dirt_est_dataloaders
from models.dirt_estimation import DirtEstimationModel
from models.losses import CombinedRegLoss
from training.trainer import Trainer
from training.scheduler import build_optimizer, build_scheduler
from evaluation.metrics import regression_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class DirtEstTrainer(Trainer):
    """Extends Trainer with dirt estimation forward pass."""

    def __init__(self, loss_fn: CombinedRegLoss, mask_loss: bool = True, **kwargs):
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.use_mask_loss = mask_loss

    def _compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        dirt_map = batch["dirt_map"]
        glass_mask = batch.get("glass_mask")

        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            pred = self.model(image, glass_mask)
            loss_dict = self.loss_fn(pred, dirt_map, glass_mask if self.use_mask_loss else None)
        return loss_dict

    def _extract_preds_targets(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            pred = self.model(batch["image"], batch.get("glass_mask"))
        return pred, batch["dirt_map"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dirt Estimation Model")
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

    # Dataloaders
    train_loader, val_loader, test_loader = build_dirt_est_dataloaders(config)
    logger.info(
        f"Dataset: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val / "
        f"{len(test_loader.dataset)} test"
    )

    # Model
    model = DirtEstimationModel.from_config(config)
    logger.info(
        f"Model: UNet / Encoder: {config['model']['encoder']} | "
        f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Loss
    loss_cfg = config["loss"]
    loss_fn = CombinedRegLoss(
        mse_weight=loss_cfg.get("mse_weight", 0.5),
        mae_weight=loss_cfg.get("mae_weight", 0.3),
        ssim_weight=loss_cfg.get("ssim_weight", 0.2),
        mask_loss=loss_cfg.get("mask_loss", True),
        ssim_window_size=loss_cfg.get("ssim_window_size", 11),
    )

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    def metrics_fn(preds, targets):
        return regression_metrics(preds, targets)

    # Trainer
    trainer = DirtEstTrainer(
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
        mask_loss=loss_cfg.get("mask_loss", True),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.fit()
    logger.info(f"Best metric: {history['best_metric']:.4f}")
    logger.info(f"Best checkpoint: {history['best_checkpoint']}")


if __name__ == "__main__":
    main()
