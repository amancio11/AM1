"""
video_inference.py
==================
Video inference pipeline for drone footage.

Processes a video file frame-by-frame and produces:
  - Annotated output video with overlays
  - Per-frame JSON results
  - Temporal smoothing of scores (moving average)
  - Summary statistics

Usage:
    python src/inference/video_inference.py \\
        --input path/to/drone_footage.mp4 \\
        --multitask-ckpt checkpoints/multitask/best.pth \\
        --output-dir outputs/video_results/
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from inference.predictor import Predictor
from evaluation.visualizer import (
    overlay_glass_mask,
    overlay_dirt_heatmap,
    draw_cleanliness_score,
    draw_region_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal smoother
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """Exponential moving average smoother for per-frame scores."""

    def __init__(self, alpha: float = 0.3, window_size: int = 10):
        self.alpha = alpha
        self._ema: Optional[float] = None
        self._window = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._window.append(value)
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1 - self.alpha) * self._ema
        return self._ema

    def moving_avg(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))


# ---------------------------------------------------------------------------
# Main video processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Processes a video file through the glass cleanliness detection pipeline.
    """

    def __init__(
        self,
        predictor: Predictor,
        output_dir: str,
        process_every_n_frames: int = 1,
        temporal_smooth_alpha: float = 0.3,
        show_glass_overlay: bool = True,
        show_dirt_overlay: bool = True,
        show_score: bool = True,
        show_regions: bool = False,
        codec: str = "mp4v",
    ):
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process_every_n = process_every_n_frames
        self.smoother = TemporalSmoother(alpha=temporal_smooth_alpha)
        self.show_glass = show_glass_overlay
        self.show_dirt = show_dirt_overlay
        self.show_score = show_score
        self.show_regions = show_regions
        self.codec = codec

    def process(self, video_path: str) -> Dict[str, Any]:
        """
        Run inference on the full video.

        Returns
        -------
        dict with processing statistics and paths to outputs.
        """
        stem = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video: {video_path} | "
            f"{total_frames} frames @ {fps:.1f} FPS | "
            f"{width}x{height}"
        )

        # Output video writer
        out_video_path = str(self.output_dir / f"{stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # Results storage
        frame_results: List[Dict] = []
        scores_history: List[float] = []
        t_start = time.perf_counter()

        # Cache last prediction for frames we skip
        last_glass_mask: Optional[np.ndarray] = None
        last_dirt_map: Optional[np.ndarray] = None
        last_score: float = 1.0
        last_grade: str = "A"
        last_regions: List = []

        frame_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run prediction every N frames
            if frame_idx % self.process_every_n == 0:
                result = self.predictor.predict_image(frame_rgb)
                last_glass_mask = result["glass_mask"]
                last_dirt_map = result["dirt_map"]
                last_score = result["score"]
                last_grade = result["grade"]
                last_regions = result["analysis"].per_region_scores

                smoothed_score = self.smoother.update(last_score)
                frame_results.append({
                    "frame": frame_idx,
                    "time_s": frame_idx / max(fps, 1.0),
                    "raw_score": last_score,
                    "smoothed_score": smoothed_score,
                    "grade": last_grade,
                    "glass_coverage": result["analysis"].glass_coverage,
                })
                scores_history.append(smoothed_score)

            # Annotate frame
            vis = frame_rgb.copy()
            if last_glass_mask is not None and self.show_glass:
                vis = overlay_glass_mask(vis, last_glass_mask, alpha=0.3)
            if last_dirt_map is not None and self.show_dirt:
                vis = overlay_dirt_heatmap(vis, last_dirt_map, last_glass_mask, alpha=0.45)
            if self.show_regions and last_regions:
                vis = draw_region_scores(vis, last_regions)
            if self.show_score:
                smooth_score = self.smoother.moving_avg()
                vis = draw_cleanliness_score(vis, smooth_score, last_grade)

            # Frame counter
            cv2.putText(
                vis,
                f"Frame {frame_idx}/{total_frames}",
                (width - 200, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            # Write to output
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            frame_idx += 1

            if frame_idx % 100 == 0:
                elapsed = time.perf_counter() - t_start
                fps_proc = frame_idx / elapsed
                logger.info(
                    f"Frame {frame_idx}/{total_frames} | "
                    f"Processing: {fps_proc:.1f} FPS | "
                    f"Score: {self.smoother.moving_avg():.3f}"
                )

        cap.release()
        writer.release()

        # Save frame-level results
        json_path = str(self.output_dir / f"{stem}_frame_results.json")
        with open(json_path, "w") as f:
            json.dump(frame_results, f, indent=2)

        # Summary
        all_scores = [r["smoothed_score"] for r in frame_results]
        summary = {
            "video_path": video_path,
            "total_frames": total_frames,
            "processed_frames": len(frame_results),
            "fps": fps,
            "duration_s": total_frames / max(fps, 1),
            "mean_cleanliness_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "min_score": float(min(all_scores)) if all_scores else 0.0,
            "max_score": float(max(all_scores)) if all_scores else 0.0,
            "std_score": float(np.std(all_scores)) if all_scores else 0.0,
            "output_video": out_video_path,
            "frame_results_json": json_path,
            "processing_time_s": time.perf_counter() - t_start,
        }

        summary_path = str(self.output_dir / f"{stem}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Video processing complete. "
            f"Mean score: {summary['mean_cleanliness_score']:.3f} | "
            f"Output: {out_video_path}"
        )
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Video inference for glass cleanliness detection")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--glass-ckpt", type=str, default=None)
    parser.add_argument("--dirt-ckpt", type=str, default=None)
    parser.add_argument("--multitask-ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/video_inference")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--glass-threshold", type=float, default=0.5)
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Run inference every N frames (1 = every frame)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-glass-overlay", action="store_true")
    parser.add_argument("--no-dirt-overlay", action="store_true")
    parser.add_argument("--show-regions", action="store_true")
    args = parser.parse_args()

    if not args.multitask_ckpt and not args.glass_ckpt:
        parser.error("Provide --multitask-ckpt or at least --glass-ckpt")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = Predictor.from_checkpoints(
        glass_ckpt=args.glass_ckpt,
        dirt_ckpt=args.dirt_ckpt,
        multitask_ckpt=args.multitask_ckpt,
        device=device,
        image_size=tuple(args.image_size),
        glass_threshold=args.glass_threshold,
    )

    processor = VideoProcessor(
        predictor=predictor,
        output_dir=args.output_dir,
        process_every_n_frames=args.skip_frames,
        show_glass_overlay=not args.no_glass_overlay,
        show_dirt_overlay=not args.no_dirt_overlay,
        show_regions=args.show_regions,
    )

    summary = processor.process(args.input)
    logger.info(f"Summary saved to: {summary['frame_results_json']}")


if __name__ == "__main__":
    main()
