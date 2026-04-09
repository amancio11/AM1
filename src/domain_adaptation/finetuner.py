"""
finetuner.py
============
Fine-tuning pipeline for bridging the synthetic-to-real domain gap.

Strategy
--------
1. Freeze encoder for the first N epochs (warm-up only the decoder heads).
2. Mix real/pseudo-labeled samples with synthetic samples at configurable ratio.
3. Downweight pseudo-labeled samples by their per-pixel confidence maps.
4. Optionally mix CycleGAN-stylized synthetic images into the real-world batch.
5. Gradually unfreeze encoder layers (progressive fine-tuning).
6. Use very low learning rate for the encoder (10–100× lower than decoder).

Inherits from `src.training.trainer.Trainer` so it gets:
  - AMP mixed-precision
  - Gradient accumulation
  - Checkpoint manager (top-K)
  - TensorBoard logging
  - Early stopping

Usage
-----
    from src.domain_adaptation.finetuner import DomainAdaptationFinetuner

    finetuner = DomainAdaptationFinetuner.from_config(config, model, loaders)
    finetuner.fit()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import Trainer
from src.models.losses import CombinedSegLoss, CombinedRegLoss, MultiTaskLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence-weighted loss helpers
# ---------------------------------------------------------------------------

def _weighted_loss(loss_fn: nn.Module, pred, target, weight_map) -> torch.Tensor:
    """
    Apply a pixel-wise weight map on top of an element-wise loss.

    Parameters
    ----------
    loss_fn : module whose forward returns a scalar loss
    pred    : (B, 1, H, W)
    target  : (B, 1, H, W)
    weight_map : (B, 1, H, W) per-pixel weights ∈ [0, 1]

    Returns
    -------
    Scalar loss tensor (weighted mean).
    """
    # Element-wise loss — requires reduction='none'
    element_loss = torch.abs(pred - target)           # L1 proxy for simplicity
    if weight_map is not None:
        w = weight_map.to(pred.device)
        return (element_loss * w).sum() / (w.sum() + 1e-8)
    return element_loss.mean()


# ---------------------------------------------------------------------------
# DomainAdaptationFinetuner
# ---------------------------------------------------------------------------

class DomainAdaptationFinetuner(Trainer):
    """
    Fine-tunes a MultitaskFacadeModel (or segmentation/dirt model) on
    real-world and pseudo-labeled data.

    Parameters
    ----------
    model : nn.Module
        A MultitaskFacadeModel (or GlassSegmentationModel / DirtEstimationModel).
    train_loader : DataLoader
        Mixed real + optional synthetic data.
    val_loader : DataLoader
        Real images (with weak or pseudo-labels) held out for validation.
    config : dict
        The ``finetuning`` block from domain_adaptation.yaml.
    checkpoint_dir : str
    device : str
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        checkpoint_dir: str,
        device: str = "cuda",
    ):
        ft_cfg = config.get("finetuning", config)

        # Build optimizer from param groups (encoder = low LR, rest = normal LR)
        optimizer = self._build_optimizer(model, ft_cfg)
        scheduler = self._build_scheduler(optimizer, ft_cfg)

        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=checkpoint_dir,
            # Config overrides
            n_epochs=ft_cfg.get("n_epochs", 30),
            grad_accumulation_steps=ft_cfg.get("grad_accumulation_steps", 4),
            amp=ft_cfg.get("amp", True),
            early_stopping_patience=ft_cfg.get("early_stopping_patience", 8),
            monitor_metric=ft_cfg.get("monitor_metric", "val_combined_score"),
            monitor_mode=ft_cfg.get("monitor_mode", "max"),
            log_dir=ft_cfg.get("log_dir", "runs/finetuning"),
        )

        self.ft_cfg = ft_cfg
        self.freeze_encoder_epochs = ft_cfg.get("freeze_encoder_epochs", 3)

        # Detect model type
        self.is_multitask = hasattr(model, "seg_head") and hasattr(model, "reg_head")
        self.is_seg_only = hasattr(model, "segmentation_head") and not self.is_multitask
        self.is_dirt_only = not self.is_multitask and not self.is_seg_only

        # Loss functions
        self.seg_loss_fn = CombinedSegLoss()
        self.reg_loss_fn = CombinedRegLoss()
        if self.is_multitask:
            self.task_loss_fn = MultiTaskLoss()

        # Freeze encoder on startup
        self._freeze_encoder()

    # ------------------------------------------------------------------
    # Trainer abstract method implementations
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        image = batch["image"].to(self.device)
        losses = {}

        glass_mask = self._get_mask(batch, "glass_mask")
        dirt_map = self._get_mask(batch, "dirt_map")
        glass_conf = self._get_mask(batch, "glass_confidence")
        dirt_conf = self._get_mask(batch, "dirt_confidence")

        if self.is_multitask:
            seg_logits, dirt_pred = self.model(image)

            loss_seg = {}
            loss_reg = {}

            if glass_mask is not None:
                loss_seg = self.seg_loss_fn(seg_logits, glass_mask)
                if glass_conf is not None:
                    # Down-weight low-confidence pseudo-label pixels
                    l1_conf = _weighted_loss(None, torch.sigmoid(seg_logits), glass_mask, glass_conf)
                    loss_seg["loss"] = loss_seg["loss"] * 0.7 + l1_conf * 0.3

            if dirt_map is not None and glass_mask is not None:
                loss_reg = self.reg_loss_fn(dirt_pred, dirt_map, glass_mask)
                if dirt_conf is not None:
                    l1_conf = _weighted_loss(None, dirt_pred, dirt_map, dirt_conf)
                    loss_reg["loss"] = loss_reg["loss"] * 0.7 + l1_conf * 0.3

            # Combine via task uncertainty if both active
            if loss_seg and loss_reg:
                total = self.task_loss_fn(
                    loss_seg["loss"],
                    loss_reg["loss"],
                )
                losses = {**{f"seg_{k}": v for k, v in loss_seg.items()},
                          **{f"reg_{k}": v for k, v in loss_reg.items()},
                          "loss": total["loss"]}
            elif loss_seg:
                losses = {**loss_seg}
            elif loss_reg:
                losses = {**loss_reg}
            else:
                # No labels available — skip this batch
                dummy = (seg_logits * 0.0).sum()
                losses = {"loss": dummy}

        elif self.is_seg_only:
            seg_logits = self.model(image)
            if glass_mask is not None:
                losses = self.seg_loss_fn(seg_logits, glass_mask)
            else:
                losses = {"loss": (seg_logits * 0.0).sum()}

        else:  # dirt-only
            dirt_pred = self.model(image)
            if dirt_map is not None and glass_mask is not None:
                losses = self.reg_loss_fn(dirt_pred, dirt_map, glass_mask)
            else:
                losses = {"loss": (dirt_pred * 0.0).sum()}

        return losses["loss"], losses

    def _extract_preds_targets(self, batch: Dict):
        """Return (preds, targets) for metric computation in val loop."""
        image = batch["image"].to(self.device)
        with torch.no_grad():
            if self.is_multitask:
                seg_logits, dirt_pred = self.model(image)
                preds = {"seg": torch.sigmoid(seg_logits), "dirt": dirt_pred}
            elif self.is_seg_only:
                preds = {"seg": torch.sigmoid(self.model(image))}
            else:
                preds = {"dirt": self.model(image)}

        targets = {}
        if "glass_mask" in batch and batch["glass_mask"] is not None:
            targets["seg"] = batch["glass_mask"].to(self.device)
        if "dirt_map" in batch and batch["dirt_map"] is not None:
            targets["dirt"] = batch["dirt_map"].to(self.device)

        return preds, targets

    # ------------------------------------------------------------------
    # Encoder freezing / unfreezing
    # ------------------------------------------------------------------

    def _freeze_encoder(self) -> None:
        encoder = self._get_encoder()
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad_(False)
            logger.info("Encoder frozen for warm-up epochs.")

    def _unfreeze_encoder(self) -> None:
        encoder = self._get_encoder()
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad_(True)
            logger.info("Encoder unfrozen — full fine-tuning active.")

    def _get_encoder(self) -> Optional[nn.Module]:
        """Robustly locate the encoder submodule across model types."""
        for attr in ("encoder", "model.encoder", "backbone"):
            parts = attr.split(".")
            obj = self.model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None

    # ------------------------------------------------------------------
    # Override fit to handle encoder unfreeze at epoch N
    # ------------------------------------------------------------------

    def fit(self) -> Dict:
        """
        Extends Trainer.fit() to unfreeze the encoder after
        `freeze_encoder_epochs` warm-up epochs.
        """
        logger.info(
            f"Fine-tuning for {self.n_epochs} epochs "
            f"(encoder frozen for first {self.freeze_encoder_epochs})."
        )
        for epoch in range(1, self.n_epochs + 1):
            if epoch == self.freeze_encoder_epochs + 1:
                self._unfreeze_encoder()
                # Re-build optimizer to ensure unfrozen params are tracked
                new_opt = self._build_optimizer(self.model, self.ft_cfg)
                self.optimizer = new_opt
                self.scheduler = self._build_scheduler(new_opt, self.ft_cfg)

            self._run_epoch(epoch)

            if self._early_stop():
                logger.info(f"Early stopping at epoch {epoch}.")
                break

        return self.best_metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mask(batch: Dict, key: str) -> Optional[torch.Tensor]:
        val = batch.get(key)
        if val is None or (isinstance(val, torch.Tensor) and val.numel() == 0):
            return None
        if isinstance(val, torch.Tensor):
            return val
        return None

    @staticmethod
    def _build_optimizer(model: nn.Module, ft_cfg: Dict) -> torch.optim.Optimizer:
        lr = ft_cfg.get("lr", 5e-5)
        enc_lr_mult = ft_cfg.get("encoder_lr_multiplier", 0.01)
        weight_decay = ft_cfg.get("weight_decay", 1e-4)

        encoder = None
        for attr in ("encoder",):
            if hasattr(model, attr):
                encoder = getattr(model, attr)
                break

        if encoder is not None:
            enc_ids = set(id(p) for p in encoder.parameters())
            other_params = [p for p in model.parameters() if id(p) not in enc_ids]
            param_groups = [
                {"params": list(encoder.parameters()), "lr": lr * enc_lr_mult},
                {"params": other_params, "lr": lr},
            ]
        else:
            param_groups = [{"params": list(model.parameters()), "lr": lr}]

        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    @staticmethod
    def _build_scheduler(optimizer, ft_cfg: Dict):
        from src.training.scheduler import build_scheduler
        return build_scheduler(
            optimizer,
            scheduler_type=ft_cfg.get("scheduler", "cosine_warmup"),
            n_epochs=ft_cfg.get("n_epochs", 30),
            warmup_epochs=ft_cfg.get("warmup_epochs", 2),
        )

    # ------------------------------------------------------------------
    # Class method factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> "DomainAdaptationFinetuner":
        da_cfg = config.get("domain_adaptation", config)
        paths = da_cfg.get("paths", {})
        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=da_cfg,
            checkpoint_dir=paths.get("finetune_checkpoint_dir", "checkpoints/finetuned"),
        )
