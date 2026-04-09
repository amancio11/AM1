"""
evaluate.py
===========
Full evaluation script: runs all models on the test split and reports metrics.

Usage:
    python scripts/evaluate.py \\
        --glass-ckpt checkpoints/glass_seg/best.pth \\
        --dirt-ckpt  checkpoints/dirt_est/best.pth \\
        --config     configs/multitask_config.yaml \\
        --output-dir outputs/evaluation/

    # Or multitask:
    python scripts/evaluate.py \\
        --multitask-ckpt checkpoints/multitask/best.pth \\
        --config configs/multitask_config.yaml
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import MultitaskDataset
from data.augmentations import get_test_transforms
from evaluation.metrics import segmentation_metrics, regression_metrics
from evaluation.cleanliness_score import compute_batch_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_glass_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Glass evaluation"):
            images = batch["image"].to(device)
            masks = batch["glass_mask"]
            logits = model(images).cpu()
            all_preds.append(logits)
            all_targets.append(masks)
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    return segmentation_metrics(preds, targets, threshold=threshold)


def evaluate_dirt_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Dirt evaluation"):
            images = batch["image"].to(device)
            glass_mask = batch.get("glass_mask")
            dirt_maps = batch["dirt_map"]
            if glass_mask is not None:
                pred = model(images, glass_mask.to(device)).cpu()
            else:
                pred = model(images).cpu()
            all_preds.append(pred)
            all_targets.append(dirt_maps)
            if glass_mask is not None:
                all_masks.append(glass_mask)
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    masks = torch.cat(all_masks, 0) if all_masks else None
    return regression_metrics(preds, targets, masks)


def evaluate_multitask_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
    seg_threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    seg_preds, seg_targets = [], []
    dirt_preds, dirt_targets, masks_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Multitask evaluation"):
            images = batch["image"].to(device)
            glass_mask = batch["glass_mask"]
            dirt_map = batch["dirt_map"]
            seg_logits, dirt_pred = model(images)
            seg_preds.append(seg_logits.cpu())
            seg_targets.append(glass_mask)
            dirt_preds.append(dirt_pred.cpu())
            dirt_targets.append(dirt_map)
            masks_list.append(glass_mask)

    seg_p = torch.cat(seg_preds, 0)
    seg_t = torch.cat(seg_targets, 0)
    dirt_p = torch.cat(dirt_preds, 0)
    dirt_t = torch.cat(dirt_targets, 0)
    masks = torch.cat(masks_list, 0)

    seg_metrics = segmentation_metrics(seg_p, seg_t, threshold=seg_threshold)
    reg_metrics = regression_metrics(dirt_p, dirt_t, masks)

    glass_probs = torch.sigmoid(seg_p)
    cleanliness_scores = compute_batch_scores(
        dirt_maps=dirt_p,
        glass_masks=glass_probs,
        glass_threshold=seg_threshold,
    )

    return {
        "segmentation": seg_metrics,
        "regression": reg_metrics,
        "cleanliness": {
            "mean_score": float(np.mean(cleanliness_scores)),
            "std_score": float(np.std(cleanliness_scores)),
            "min_score": float(min(cleanliness_scores)),
            "max_score": float(max(cleanliness_scores)),
        },
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Evaluation Results") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    def _print_dict(d, indent=2):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * indent + f"{k}:")
                _print_dict(v, indent + 4)
            else:
                print(" " * indent + f"{k:25s}: {v:.4f}" if isinstance(v, float) else f"{' '*indent}{k:25s}: {v}")

    _print_dict(metrics)
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--glass-ckpt", type=str, default=None)
    parser.add_argument("--dirt-ckpt", type=str, default=None)
    parser.add_argument("--multitask-ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--glass-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = get_test_transforms(config)
    test_ds = MultitaskDataset.from_config(config, split="test", transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 4),
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_ds)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.multitask_ckpt:
        from models.multitask_model import MultitaskFacadeModel
        state = torch.load(args.multitask_ckpt, map_location=device)
        model = MultitaskFacadeModel.from_config(state["config"]).to(device)
        model.load_state_dict(state["model_state_dict"])
        metrics = evaluate_multitask_model(model, test_loader, device, args.glass_threshold)
        print_metrics(metrics, "Multi-Task Model")
        all_results["multitask"] = metrics

    elif args.glass_ckpt:
        from models.glass_segmentation import GlassSegmentationModel
        state = torch.load(args.glass_ckpt, map_location=device)
        glass_model = GlassSegmentationModel.from_config(state["config"]).to(device)
        glass_model.load_state_dict(state["model_state_dict"])
        glass_metrics = evaluate_glass_model(glass_model, test_loader, device, args.glass_threshold)
        print_metrics(glass_metrics, "Glass Segmentation Model")
        all_results["glass_segmentation"] = glass_metrics

        if args.dirt_ckpt:
            from models.dirt_estimation import DirtEstimationModel
            state = torch.load(args.dirt_ckpt, map_location=device)
            dirt_model = DirtEstimationModel.from_config(state["config"]).to(device)
            dirt_model.load_state_dict(state["model_state_dict"])
            dirt_metrics = evaluate_dirt_model(dirt_model, test_loader, device)
            print_metrics(dirt_metrics, "Dirt Estimation Model")
            all_results["dirt_estimation"] = dirt_metrics

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
