"""
predictor.py
============
Core Predictor class — wraps model(s) for inference.

Supports three inference modes:
  1. Two-stage: GlassSegmentationModel → DirtEstimationModel
  2. Single multi-task: MultitaskFacadeModel
  3. Glass-only: just segmentation

Handles:
  - Model loading from checkpoint
  - Image preprocessing
  - Optional test-time augmentation (TTA)
  - Batch inference
  - Output post-processing (sigmoid, threshold, spatial smoothing)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Predictor:
    """
    Unified inference interface for glass segmentation and dirt estimation.
    """

    def __init__(
        self,
        glass_model: Optional[nn.Module],
        dirt_model: Optional[nn.Module] = None,
        multitask_model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (512, 512),
        glass_threshold: float = 0.5,
        smooth_dirt: bool = False,
        smooth_kernel_size: int = 5,
        smooth_sigma: float = 1.0,
        use_tta: bool = False,
        mixed_precision: bool = True,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size      # (H, W)
        self.glass_threshold = glass_threshold
        self.smooth_dirt = smooth_dirt
        self.smooth_kernel_size = smooth_kernel_size
        self.smooth_sigma = smooth_sigma
        self.use_tta = use_tta
        self.mixed_precision = mixed_precision and self.device.type == "cuda"

        # Models
        self.glass_model = self._to_eval(glass_model)
        self.dirt_model = self._to_eval(dirt_model)
        self.multitask_model = self._to_eval(multitask_model)

        self._mode = self._detect_mode()
        logger.info(f"Predictor mode: {self._mode} | Device: {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(
        self, image: Union[np.ndarray, str, Path]
    ) -> Dict[str, Any]:
        """
        Run full inference on a single image.

        Parameters
        ----------
        image : np.ndarray (H,W,3 RGB uint8) or path string

        Returns
        -------
        dict with:
            "glass_mask"  : (H, W) float32 [0,1] glass probability
            "glass_binary": (H, W) uint8 binary glass mask
            "dirt_map"    : (H, W) float32 [0,1] dirt heatmap
            "score"       : float — cleanliness score [0,1]
            "grade"       : str — A/B/C/D/F
            "analysis"    : CleanlinessResult
        """
        if isinstance(image, (str, Path)):
            image = self._load_image(str(image))

        orig_h, orig_w = image.shape[:2]
        tensor = self._preprocess(image)  # (1, 3, H, W)

        if self._mode == "multitask":
            glass_logits, dirt_pred = self._run_multitask(tensor)
        elif self._mode == "two_stage":
            glass_logits = self._run_glass(tensor)
            dirt_pred = self._run_dirt(tensor, glass_logits)
        elif self._mode == "glass_only":
            glass_logits = self._run_glass(tensor)
            dirt_pred = torch.zeros_like(glass_logits)
        else:
            raise RuntimeError(f"Unknown predictor mode: {self._mode}")

        # Convert to (H, W)
        glass_prob = torch.sigmoid(glass_logits).squeeze().cpu()
        dirt_heatmap = dirt_pred.squeeze().cpu()

        # Smooth dirt map
        if self.smooth_dirt:
            dirt_heatmap = self._smooth_map(dirt_heatmap)

        # Resize back to original image size
        glass_prob_orig = self._resize_map(glass_prob.numpy(), (orig_h, orig_w))
        dirt_orig = self._resize_map(dirt_heatmap.numpy(), (orig_h, orig_w))

        # Scores
        from evaluation.cleanliness_score import compute_full_analysis
        result = compute_full_analysis(
            dirt_map=torch.from_numpy(dirt_orig),
            glass_mask=torch.from_numpy(glass_prob_orig),
            glass_threshold=self.glass_threshold,
        )

        glass_binary = (glass_prob_orig > self.glass_threshold).astype(np.uint8) * 255

        return {
            "glass_mask": glass_prob_orig,
            "glass_binary": glass_binary,
            "dirt_map": dirt_orig,
            "score": result.overall_score,
            "grade": result.grade,
            "analysis": result,
        }

    def predict_batch(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch inference: returns glass and dirt tensors at model resolution.

        Parameters
        ----------
        images : (B, 3, H, W) normalized tensor

        Returns
        -------
        (glass_probs, dirt_maps) — both (B, 1, H, W) in [0,1]
        """
        images = images.to(self.device)
        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                if self._mode == "multitask":
                    glass_logits, dirt_pred = self.multitask_model(images)
                elif self._mode == "two_stage":
                    glass_logits = self.glass_model(images)
                    dirt_pred = self.dirt_model(images, torch.sigmoid(glass_logits))
                else:
                    glass_logits = self.glass_model(images)
                    dirt_pred = torch.zeros_like(glass_logits)
        return torch.sigmoid(glass_logits), dirt_pred

    # ------------------------------------------------------------------
    # TTA
    # ------------------------------------------------------------------

    def _run_glass_tta(self, tensor: torch.Tensor) -> torch.Tensor:
        """Horizontal flip TTA for glass segmentation."""
        logits = self._run_glass(tensor)
        flipped = self._run_glass(torch.flip(tensor, dims=[-1]))
        logits_flipped_back = torch.flip(flipped, dims=[-1])
        return (logits + logits_flipped_back) / 2.0

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _run_glass(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                return self.glass_model(tensor.to(self.device))

    def _run_dirt(
        self, tensor: torch.Tensor, glass_logits: torch.Tensor
    ) -> torch.Tensor:
        glass_prob = torch.sigmoid(glass_logits)
        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                try:
                    return self.dirt_model(tensor.to(self.device), glass_prob)
                except TypeError:
                    return self.dirt_model(tensor.to(self.device))

    def _run_multitask(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            with autocast(enabled=self.mixed_precision):
                return self.multitask_model(tensor.to(self.device))

    # ------------------------------------------------------------------
    # Pre/post-processing
    # ------------------------------------------------------------------

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and convert to (1, 3, H, W) tensor."""
        h, w = self.image_size
        img = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _smooth_map(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to the predicted dirt map."""
        k = self.smooth_kernel_size
        sigma = self.smooth_sigma
        tensored = heatmap.unsqueeze(0).unsqueeze(0)
        blurred = F.gaussian_blur(tensored, kernel_size=[k, k], sigma=[sigma, sigma])
        return blurred.squeeze()

    @staticmethod
    def _resize_map(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        if arr.shape[:2] == (h, w):
            return arr
        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)

    def _to_eval(self, model: Optional[nn.Module]) -> Optional[nn.Module]:
        if model is None:
            return None
        model.eval()
        return model.to(self.device)

    def _detect_mode(self) -> str:
        if self.multitask_model is not None:
            return "multitask"
        if self.glass_model is not None and self.dirt_model is not None:
            return "two_stage"
        if self.glass_model is not None:
            return "glass_only"
        raise ValueError(
            "At least one model must be provided (glass_model, dirt_model, or multitask_model)."
        )

    # ------------------------------------------------------------------
    # Class-level factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        glass_ckpt: Optional[str] = None,
        dirt_ckpt: Optional[str] = None,
        multitask_ckpt: Optional[str] = None,
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (512, 512),
        glass_threshold: float = 0.5,
    ) -> "Predictor":
        """
        Load models from checkpoint files.

        Each checkpoint is expected to contain:
          - "model_state_dict"
          - "config"
        """
        import sys
        from pathlib import Path

        _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        glass_model, dirt_model, multitask_model = None, None, None

        if multitask_ckpt:
            from models.multitask_model import MultitaskFacadeModel
            state = torch.load(multitask_ckpt, map_location=_device)
            multitask_model = MultitaskFacadeModel.from_config(state["config"])
            multitask_model.load_state_dict(state["model_state_dict"])
            logger.info(f"Loaded multitask model from: {multitask_ckpt}")
        else:
            if glass_ckpt:
                from models.glass_segmentation import GlassSegmentationModel
                state = torch.load(glass_ckpt, map_location=_device)
                glass_model = GlassSegmentationModel.from_config(state["config"])
                glass_model.load_state_dict(state["model_state_dict"])
                logger.info(f"Loaded glass segmentation model from: {glass_ckpt}")
            if dirt_ckpt:
                from models.dirt_estimation import DirtEstimationModel
                state = torch.load(dirt_ckpt, map_location=_device)
                dirt_model = DirtEstimationModel.from_config(state["config"])
                dirt_model.load_state_dict(state["model_state_dict"])
                logger.info(f"Loaded dirt estimation model from: {dirt_ckpt}")

        return cls(
            glass_model=glass_model,
            dirt_model=dirt_model,
            multitask_model=multitask_model,
            device=_device,
            image_size=image_size,
            glass_threshold=glass_threshold,
        )
