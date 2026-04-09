"""
trainer.py
==========
Generic Trainer class for all models in the pipeline.

Features
--------
- Mixed precision training (torch.amp)
- Gradient accumulation
- Gradient clipping
- LR scheduling (cosine warmup, plateau)
- Checkpoint saving (top-K by metric)
- Early stopping
- TensorBoard + optional W&B logging
- Per-epoch train/val loops
"""

import os
import math
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
)
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Keeps only the top-K checkpoints sorted by metric value."""

    def __init__(
        self,
        checkpoint_dir: str,
        top_k: int = 3,
        mode: str = "max",
        metric_name: str = "val/iou",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.mode = mode
        self.metric_name = metric_name
        self._history: List[Tuple[float, Path]] = []  # (score, path)

    def save(
        self,
        state: Dict[str, Any],
        metric_value: float,
        epoch: int,
    ) -> Optional[Path]:
        filename = self.checkpoint_dir / f"ckpt_epoch{epoch:04d}_{metric_value:.4f}.pth"
        torch.save(state, filename)
        self._history.append((metric_value, filename))
        self._history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Remove worst checkpoints beyond top_k
        while len(self._history) > self.top_k:
            _, worst_path = self._history.pop()
            if worst_path.exists():
                worst_path.unlink()
                logger.debug(f"Removed checkpoint: {worst_path}")

        return filename

    def best_checkpoint(self) -> Optional[Path]:
        if not self._history:
            return None
        return self._history[0][1]

    def best_score(self) -> Optional[float]:
        if not self._history:
            return None
        return self._history[0][0]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Generic trainer for glass segmentation / dirt estimation / multitask models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        metrics_fn: Optional[Callable] = None,
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.metrics_fn = metrics_fn
        self.experiment_name = experiment_name

        training_cfg = config["training"]
        self.epochs = training_cfg["epochs"]
        self.accumulate_grad = training_cfg.get("accumulate_grad_batches", 1)
        self.mixed_precision = training_cfg.get("mixed_precision", True)
        self.grad_clip = training_cfg.get("gradient_clip_val", 1.0)
        self.early_stop_patience = training_cfg.get("early_stopping_patience", 20)
        self.early_stop_metric = training_cfg.get("early_stopping_metric", "val/iou")
        self.early_stop_mode = training_cfg.get("early_stopping_mode", "max")
        self.log_every_n_steps = training_cfg.get("log_every_n_steps", 10)
        self.val_every_n_epochs = training_cfg.get("val_every_n_epochs", 1)
        save_top_k = training_cfg.get("save_top_k", 3)

        paths_cfg = config.get("paths", {})
        self.ckpt_mgr = CheckpointManager(
            checkpoint_dir=paths_cfg.get("checkpoint_dir", f"checkpoints/{experiment_name}"),
            top_k=save_top_k,
            mode=self.early_stop_mode,
            metric_name=self.early_stop_metric,
        )

        log_dir = paths_cfg.get("log_dir", f"logs/{experiment_name}")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.scaler = GradScaler(enabled=self.mixed_precision and device.type == "cuda")
        self._early_stop_counter = 0
        self._best_metric = -math.inf if self.early_stop_mode == "max" else math.inf
        self.current_epoch = 0
        self._global_step = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns
        -------
        dict
            Training history and best metric info.
        """
        logger.info(f"Starting training: {self.experiment_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches:   {len(self.val_loader)}")

        train_history = []
        val_history = []

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            t_epoch = time.perf_counter()

            train_metrics = self._train_epoch(epoch)
            train_history.append(train_metrics)

            val_metrics = {}
            if (epoch + 1) % self.val_every_n_epochs == 0:
                val_metrics = self._val_epoch(epoch)
                val_history.append(val_metrics)
                self._log_metrics(val_metrics, epoch, prefix="val")
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"Train loss: {train_metrics.get('loss', 0):.4f} | "
                    f"Val loss: {val_metrics.get('loss', 0):.4f} | "
                    f"LR: {self._get_lr():.2e} | "
                    f"Time: {time.perf_counter()-t_epoch:.1f}s"
                )

                # Checkpoint
                monitor_metric = val_metrics.get(
                    self.early_stop_metric.replace("val/", ""),
                    val_metrics.get("loss", 0)
                )
                self.ckpt_mgr.save(
                    state=self._build_checkpoint_state(val_metrics),
                    metric_value=monitor_metric,
                    epoch=epoch + 1,
                )

                # Early stopping
                if self._check_early_stopping(monitor_metric):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # LR scheduler step
            self._scheduler_step(val_metrics.get("loss"))

        self.writer.close()
        best_ckpt = self.ckpt_mgr.best_checkpoint()
        logger.info(f"Training complete. Best checkpoint: {best_ckpt}")
        return {
            "train_history": train_history,
            "val_history": val_history,
            "best_checkpoint": str(best_ckpt),
            "best_metric": self.ckpt_mgr.best_score(),
        }

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        extra_metrics: Dict[str, float] = {}
        n_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._batch_to_device(batch)
            loss_dict = self._compute_loss(batch)
            loss = loss_dict["loss"] / self.accumulate_grad

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulate_grad == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss_dict["loss"].item()
            for k, v in loss_dict.items():
                if k != "loss":
                    extra_metrics[k] = extra_metrics.get(k, 0.0) + v.item()

            self._global_step += 1
            if self._global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("train/loss_step", loss_dict["loss"].item(), self._global_step)
                self.writer.add_scalar("train/lr", self._get_lr(), self._global_step)

        avg_loss = total_loss / n_batches
        metrics = {"loss": avg_loss}
        for k, v in extra_metrics.items():
            metrics[k] = v / n_batches

        self._log_metrics(metrics, epoch, prefix="train")
        return metrics

    # ------------------------------------------------------------------
    # Val epoch
    # ------------------------------------------------------------------

    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        extra_metrics: Dict[str, float] = {}
        all_preds = []
        all_targets = []
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._batch_to_device(batch)
                with autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
                    loss_dict = self._compute_loss(batch)

                total_loss += loss_dict["loss"].item()
                for k, v in loss_dict.items():
                    if k != "loss":
                        extra_metrics[k] = extra_metrics.get(k, 0.0) + v.item()

                if self.metrics_fn is not None:
                    preds, targets = self._extract_preds_targets(batch)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

        avg_loss = total_loss / n_batches
        metrics = {"loss": avg_loss}
        for k, v in extra_metrics.items():
            metrics[k] = v / n_batches

        if self.metrics_fn is not None and all_preds:
            all_preds_t = torch.cat(all_preds, dim=0)
            all_targets_t = torch.cat(all_targets, dim=0)
            task_metrics = self.metrics_fn(all_preds_t, all_targets_t)
            metrics.update(task_metrics)

        return metrics

    # ------------------------------------------------------------------
    # Abstract-ish methods (override in subclasses or pass callbacks)
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Override in task-specific trainer subclasses."""
        raise NotImplementedError(
            "Subclasses must implement _compute_loss(). "
            "Or use GlassSegTrainer / DirtEstTrainer / MultitaskTrainer."
        )

    def _extract_preds_targets(
        self, batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override to extract (preds, targets) for metrics computation."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _log_metrics(self, metrics: Dict[str, float], epoch: int, prefix: str) -> None:
        for key, val in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", val, epoch)

    def _scheduler_step(self, val_loss: Optional[float] = None) -> None:
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_loss is not None:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _check_early_stopping(self, metric: float) -> bool:
        improved = (
            metric > self._best_metric
            if self.early_stop_mode == "max"
            else metric < self._best_metric
        )
        if improved:
            self._best_metric = metric
            self._early_stop_counter = 0
        else:
            self._early_stop_counter += 1

        return self._early_stop_counter >= self.early_stop_patience

    def _build_checkpoint_state(self, val_metrics: Dict) -> Dict:
        return {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": self.config,
            "best_metric": self._best_metric,
        }

    def load_checkpoint(self, path: str) -> Dict:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.current_epoch = state["epoch"]
        self._best_metric = state.get("best_metric", self._best_metric)
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
        return state
