"""
scheduler.py
============
Learning rate scheduler factory and cosine warmup implementation.
"""

import math
from typing import Dict, Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LambdaLR,
    _LRScheduler,
)


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> _LRScheduler:
    """
    Build a learning rate scheduler from config.

    Supported types:
      - "cosine_warmup": Linear warmup then cosine annealing
      - "plateau":       ReduceLROnPlateau
      - "step":          StepLR
      - "cosine":        Pure CosineAnnealingLR
    """
    sched_cfg = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_warmup")

    if sched_type == "cosine_warmup":
        warmup_epochs = sched_cfg.get("warmup_epochs", 5)
        T_max = sched_cfg.get("T_max", 95)
        eta_min = sched_cfg.get("eta_min", 1e-7)
        base_lr = optimizer.param_groups[0]["lr"]

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, T_max - warmup_epochs))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min / base_lr + cos * (1.0 - eta_min / base_lr)

        return LambdaLR(optimizer, lr_lambda)

    elif sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=config.get("training", {}).get("early_stopping_mode", "max"),
            patience=sched_cfg.get("patience", 10),
            factor=sched_cfg.get("factor", 0.5),
            min_lr=sched_cfg.get("eta_min", 1e-8),
        )

    elif sched_type == "step":
        return StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )

    elif sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", 100),
            eta_min=sched_cfg.get("eta_min", 1e-8),
        )

    else:
        raise ValueError(f"Unknown scheduler type: '{sched_type}'")


def build_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Build an optimizer with optional encoder/decoder param group split.
    """
    opt_cfg = config.get("optimizer", {})
    opt_type = opt_cfg.get("type", "adamw")
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    encoder_lr_mult = float(opt_cfg.get("encoder_lr_multiplier", 0.1))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    # Try to get task-specific param groups (encoder + decoder)
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(
            encoder_lr=lr * encoder_lr_mult,
            decoder_lr=lr,
        )
    else:
        param_groups = model.parameters()

    if opt_type == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=lr, weight_decay=weight_decay, betas=betas
        )
    elif opt_type == "adam":
        return torch.optim.Adam(
            param_groups, lr=lr, betas=betas
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: '{opt_type}'")
