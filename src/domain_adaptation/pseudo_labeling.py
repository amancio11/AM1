"""
pseudo_labeling.py
==================
Auto-generate glass segmentation masks and dirt heatmaps for unlabeled
real-world images using a pretrained (or fine-tuned) model.

Pipeline
--------
1. Load unlabeled real images via `RealFacadeDataset`.
2. Run inference with `Predictor` (multitask or two-stage mode).
3. Filter predictions by quality:
   - Glass confidence mask: mean prediction probability > threshold (default 0.85)
   - Entropy filter: mean Shannon entropy < max_entropy (default 0.4)
   - Combined: both conditions must hold
4. Save accepted pseudo-labels to `pseudo_label_dir` as:
     {stem}_glass.png         — binary glass mask [0, 255]
     {stem}_dirt.png          — dirt heatmap [0, 255]
     {stem}_glass_conf.png    — glass probability map [0, 255]
     {stem}_dirt_conf.png     — dirt confidence map [0, 255]
     {stem}_meta.json         — confidence stats, entropy, accept/reject
5. Return a manifest dict of accepted samples.

Usage
-----
    from src.domain_adaptation.pseudo_labeling import PseudoLabeler

    labeler = PseudoLabeler.from_config(cfg, predictor)
    manifest = labeler.generate()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entropy / confidence utilities
# ---------------------------------------------------------------------------

def _binary_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Shannon entropy for a binary-class probability map.

    H = -p*log2(p) - (1-p)*log2(1-p)

    Parameters
    ----------
    prob : ndarray, shape (H, W), values in [0, 1]

    Returns
    -------
    entropy : ndarray, shape (H, W), values in [0, 1]
    """
    p = np.clip(prob, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _compute_glass_confidence(
    glass_prob: np.ndarray,
    strategy: str = "confidence",
) -> Tuple[float, float, np.ndarray]:
    """
    Compute a per-image confidence score for pseudo-label acceptance.

    Returns
    -------
    mean_conf : float
        Mean predicted probability over glass pixels (or all if no glass).
    mean_entropy : float
        Mean entropy of the glass probability map.
    per_pixel_conf : ndarray (H, W)
        Pixel-level confidence weights ∈ [0, 1].
    """
    entropy_map = _binary_entropy(glass_prob)          # 0..1
    mean_entropy = float(entropy_map.mean())

    # Confidence = 1 - entropy (normalized to 0..1)
    conf_map = 1.0 - entropy_map

    # Mean confidence = mean of max(p, 1-p) — peak prediction sharpness
    mean_conf = float(np.maximum(glass_prob, 1.0 - glass_prob).mean())

    return mean_conf, mean_entropy, conf_map.astype(np.float32)


def _passes_filter(
    mean_conf: float,
    mean_entropy: float,
    strategy: str,
    conf_threshold: float,
    entropy_threshold: float,
) -> bool:
    if strategy == "confidence":
        return mean_conf >= conf_threshold
    elif strategy == "entropy":
        return mean_entropy <= entropy_threshold
    elif strategy == "combined":
        return mean_conf >= conf_threshold and mean_entropy <= entropy_threshold
    else:
        raise ValueError(f"Unknown pseudo-label filter strategy: {strategy}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PseudoLabeler:
    """
    Generates and saves pseudo-labels for unlabeled real images.

    Parameters
    ----------
    predictor : Predictor
        Loaded inference object (from src.inference.predictor).
    image_paths : list of Path
        Unlabeled real image files.
    output_dir : str or Path
        Where to write pseudo-label files.
    glass_confidence_threshold : float
        Minimum mean sharpness for acceptance (0.85 recommended).
    max_mean_entropy : float
        Maximum per-image entropy for acceptance (0.4 recommended).
    filter_strategy : str
        One of 'confidence', 'entropy', 'combined'.
    max_pseudo_images : int or None
        Hard cap on accepted samples (None = unlimited).
    batch_size : int
    device : str
    """

    def __init__(
        self,
        predictor,
        image_paths: List[Path],
        output_dir: str,
        glass_confidence_threshold: float = 0.85,
        max_mean_entropy: float = 0.4,
        filter_strategy: str = "combined",
        max_pseudo_images: Optional[int] = None,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        self.predictor = predictor
        self.image_paths = image_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.conf_threshold = glass_confidence_threshold
        self.entropy_threshold = max_mean_entropy
        self.filter_strategy = filter_strategy
        self.max_pseudo_images = max_pseudo_images
        self.batch_size = batch_size
        self.device = device

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict, predictor) -> "PseudoLabeler":
        da_cfg = config.get("domain_adaptation", config)
        pl_cfg = da_cfg.get("pseudo_labeling", {})
        paths = da_cfg.get("paths", {})

        # Collect real image paths
        from src.domain_adaptation.real_dataset import _discover_images
        real_dir = paths.get("real_images_dir", "data/real")
        exts = da_cfg.get("real_dataset", {}).get("extensions", [".png", ".jpg", ".jpeg"])
        image_paths = _discover_images(real_dir, tuple(exts))

        return cls(
            predictor=predictor,
            image_paths=image_paths,
            output_dir=paths.get("pseudo_label_dir", "data/pseudo_labels"),
            glass_confidence_threshold=pl_cfg.get("glass_confidence_threshold", 0.85),
            max_mean_entropy=pl_cfg.get("max_mean_entropy", 0.4),
            filter_strategy=pl_cfg.get("filter_strategy", "combined"),
            max_pseudo_images=pl_cfg.get("max_pseudo_images", None),
            batch_size=pl_cfg.get("batch_size", 8),
        )

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    def generate(self, force_regen: bool = False) -> Dict:
        """
        Run pseudo-label generation over all image paths.

        Returns
        -------
        manifest : dict
            {
                "accepted": [str, ...],
                "rejected": [str, ...],
                "stats": {mean_conf, mean_entropy, acceptance_rate}
            }
        """
        accepted: List[str] = []
        rejected: List[str] = []

        for img_path in self.image_paths:
            # Skip if already labeled (unless force_regen)
            out_glass = self.output_dir / f"{img_path.stem}_glass.png"
            if out_glass.exists() and not force_regen:
                accepted.append(str(img_path))
                continue

            try:
                result = self._process_single(img_path)
            except Exception as exc:
                logger.warning(f"Failed on {img_path.name}: {exc}")
                rejected.append(str(img_path))
                continue

            if result["accepted"]:
                self._save_pseudo_labels(img_path, result)
                accepted.append(str(img_path))
                logger.debug(
                    f"Accepted  {img_path.name}  "
                    f"conf={result['mean_conf']:.3f}  "
                    f"H={result['mean_entropy']:.3f}"
                )
            else:
                rejected.append(str(img_path))
                logger.debug(
                    f"Rejected  {img_path.name}  "
                    f"conf={result['mean_conf']:.3f}  "
                    f"H={result['mean_entropy']:.3f}"
                )

            if self.max_pseudo_images and len(accepted) >= self.max_pseudo_images:
                logger.info(f"Reached max_pseudo_images={self.max_pseudo_images}, stopping.")
                break

        n_total = len(accepted) + len(rejected)
        manifest = {
            "accepted": accepted,
            "rejected": rejected,
            "stats": {
                "total": n_total,
                "num_accepted": len(accepted),
                "num_rejected": len(rejected),
                "acceptance_rate": len(accepted) / max(n_total, 1),
            },
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            f"Pseudo-labeling done: {len(accepted)}/{n_total} accepted "
            f"({manifest['stats']['acceptance_rate']:.1%})"
        )
        return manifest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _process_single(self, img_path: Path) -> Dict:
        """Run predictor on one image and compute quality statistics."""
        import cv2 as _cv2
        image_bgr = _cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(str(img_path))
        image_rgb = _cv2.cvtColor(image_bgr, _cv2.COLOR_BGR2RGB)

        pred = self.predictor.predict_image(image_rgb)

        glass_prob = pred.get("glass_mask")         # (H, W) float [0,1]
        dirt_map = pred.get("dirt_map")             # (H, W) float [0,1]

        if glass_prob is None:
            raise ValueError("Predictor did not return glass_mask")

        if isinstance(glass_prob, torch.Tensor):
            glass_prob = glass_prob.squeeze().cpu().numpy()
        if isinstance(dirt_map, torch.Tensor):
            dirt_map = dirt_map.squeeze().cpu().numpy()

        mean_conf, mean_entropy, conf_map = _compute_glass_confidence(
            glass_prob, self.filter_strategy
        )

        accepted = _passes_filter(
            mean_conf, mean_entropy,
            self.filter_strategy,
            self.conf_threshold,
            self.entropy_threshold,
        )

        # Dirt confidence: 1 - |pred - 0.5| * 2 → uncertainty of regression map
        if dirt_map is not None:
            dirt_conf = 1.0 - np.abs(dirt_map - 0.5) * 2.0
        else:
            dirt_conf = np.ones_like(glass_prob)
            dirt_map = np.zeros_like(glass_prob)

        return {
            "accepted": accepted,
            "mean_conf": mean_conf,
            "mean_entropy": mean_entropy,
            "glass_prob": glass_prob,
            "dirt_map": dirt_map,
            "glass_conf": conf_map,
            "dirt_conf": dirt_conf.astype(np.float32),
        }

    def _save_pseudo_labels(self, img_path: Path, result: Dict) -> None:
        """Write four PNG files to output_dir for one accepted image."""
        stem = img_path.stem

        def _save_float_as_uint8(arr: np.ndarray, fname: str) -> None:
            arr_u8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(self.output_dir / fname), arr_u8)

        # Binary glass mask (threshold at 0.5)
        glass_binary = (result["glass_prob"] > 0.5).astype(np.float32)
        _save_float_as_uint8(glass_binary, f"{stem}_glass.png")
        _save_float_as_uint8(result["dirt_map"], f"{stem}_dirt.png")
        _save_float_as_uint8(result["glass_conf"], f"{stem}_glass_conf.png")
        _save_float_as_uint8(result["dirt_conf"], f"{stem}_dirt_conf.png")

        meta = {
            "image_path": str(img_path),
            "mean_conf": float(result["mean_conf"]),
            "mean_entropy": float(result["mean_entropy"]),
            "glass_pixel_fraction": float(glass_binary.mean()),
        }
        meta_path = self.output_dir / f"{stem}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)


# ---------------------------------------------------------------------------
# Convenience: load accepted paths from a manifest
# ---------------------------------------------------------------------------

def load_accepted_paths(manifest_path: str) -> List[Path]:
    """Return list of accepted image paths from a saved manifest.json."""
    with open(manifest_path, "r") as f:
        data = json.load(f)
    return [Path(p) for p in data.get("accepted", [])]
