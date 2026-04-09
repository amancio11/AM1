"""
train_multitask.py
==================
Training script for the Multi-Task model (segmentation + dirt estimation).

Usage:
    python src/training/train_multitask.py --config configs/multitask_config.yaml
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

from data.dataloader import build_multitask_dataloaders
from models.multitask_model import MultitaskFacadeModel
from models.losses import CombinedSegLoss, CombinedRegLoss, MultiTaskLoss
from training.trainer import Trainer
from training.scheduler import build_optimizer, build_scheduler
from evaluation.metrics import segmentation_metrics, regression_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class MultitaskTrainer(Trainer):
    """Extends Trainer with multi-task forward pass."""

    def __init__(self, loss_fn: MultiTaskLoss, seg_threshold: float = 0.5, **kwargs):
        super().__init__(loss_fn=loss_fn, **kwargs)
        self.seg_threshold = seg_threshold
        self._all_seg_preds = []
        self._all_seg_targets = []
        self._all_dirt_preds = []
        self._all_dirt_targets = []

    def _compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        glass_mask = batch["glass_mask"]
        dirt_map = batch["dirt_map"]

        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            seg_logits, dirt_pred = self.model(image)
            loss_dict = self.loss_fn(
                seg_pred=seg_logits,
                seg_target=glass_mask,
                reg_pred=dirt_pred,
                reg_target=dirt_map,
                glass_mask=glass_mask,
            )
        return loss_dict

    def _extract_preds_targets(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return seg preds for primary metric; metrics_fn handles both
        with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
            seg_logits, dirt_pred = self.model(batch["image"])
        seg_pred = (torch.sigmoid(seg_logits) > self.seg_threshold).float()
        # We store both task outputs; metrics_fn receives only seg for simplicity
        # Override metrics_fn to handle multitask evaluation
        return seg_pred, batch["glass_mask"]

    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """Override to compute both seg and dirt metrics."""
        self.model.eval()
        total_loss = 0.0
        extra_metrics: Dict[str, float] = {}
        seg_preds_list = []
        seg_targets_list = []
        dirt_preds_list = []
        dirt_targets_list = []
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._batch_to_device(batch)
                with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
                    loss_dict = self._compute_loss(batch)
                    seg_logits, dirt_pred = self.model(batch["image"])

                total_loss += loss_dict["loss"].item()
                for k, v in loss_dict.items():
                    if k != "loss":
                        extra_metrics[k] = extra_metrics.get(k, 0.0) + v.item()

                seg_preds = (torch.sigmoid(seg_logits) > self.seg_threshold).float().cpu()
                seg_targets = batch["glass_mask"].cpu()
                dirt_preds_list.append(dirt_pred.cpu())
                dirt_targets_list.append(batch["dirt_map"].cpu())
                seg_preds_list.append(seg_preds)
                seg_targets_list.append(seg_targets)

        metrics = {"loss": total_loss / n_batches}
        for k, v in extra_metrics.items():
            metrics[k] = v / n_batches

        # Segmentation metrics
        seg_m = segmentation_metrics(
            torch.cat(seg_preds_list, 0), torch.cat(seg_targets_list, 0)
        )
        metrics.update({f"seg_{k}": v for k, v in seg_m.items()})

        # Regression metrics
        reg_m = regression_metrics(
            torch.cat(dirt_preds_list, 0), torch.cat(dirt_targets_list, 0)
        )
        metrics.update({f"reg_{k}": v for k, v in reg_m.items()})

        # Combined score
        iou = metrics.get("seg_iou", 0.0)
        mae = metrics.get("reg_mae", 1.0)
        metrics["combined_score"] = 0.6 * iou + 0.4 * (1.0 - min(mae, 1.0))

        return metrics


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
    parser = argparse.ArgumentParser(description="Train Multi-Task Facade Model")
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

    train_loader, val_loader, test_loader = build_multitask_dataloaders(config)

    model = MultitaskFacadeModel.from_config(config)
    logger.info(
        f"MultitaskFacadeModel | Encoder: {config['model']['encoder']} | "
        f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    loss_cfg = config["loss"]
    seg_loss = CombinedSegLoss(
        bce_weight=loss_cfg.get("seg_bce_weight", 0.4),
        dice_weight=loss_cfg.get("seg_dice_weight", 0.4),
        focal_weight=loss_cfg.get("seg_focal_weight", 0.2),
        pos_weight=loss_cfg.get("seg_pos_weight"),
    )
    reg_loss = CombinedRegLoss(
        mse_weight=loss_cfg.get("reg_mse_weight", 0.5),
        mae_weight=loss_cfg.get("reg_mae_weight", 0.3),
        ssim_weight=loss_cfg.get("reg_ssim_weight", 0.2),
        mask_loss=loss_cfg.get("reg_mask_loss", True),
    )
    loss_fn = MultiTaskLoss(
        seg_loss=seg_loss,
        reg_loss=reg_loss,
        task_balancing=loss_cfg.get("task_balancing", "uncertainty"),
        seg_task_weight=loss_cfg.get("seg_task_weight", 1.0),
        reg_task_weight=loss_cfg.get("reg_task_weight", 1.0),
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    trainer = MultitaskTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        metrics_fn=None,   # handled inside _val_epoch override
        experiment_name=config["experiment"]["name"],
        seg_threshold=config["inference"].get("seg_threshold", 0.5),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    history = trainer.fit()
    logger.info(f"Best metric: {history['best_metric']:.4f}")
    logger.info(f"Best checkpoint: {history['best_checkpoint']}")


if __name__ == "__main__":
    main()
