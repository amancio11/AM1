"""
image_inference.py
==================
Single-image and batch-image inference pipeline.

Usage:
    python src/inference/image_inference.py \\
        --input path/to/image.png \\
        --glass-ckpt checkpoints/glass_seg/best.pth \\
        --dirt-ckpt  checkpoints/dirt_est/best.pth \\
        --output-dir outputs/results/

    # Or with multitask model:
    python src/inference/image_inference.py \\
        --input path/to/image.png \\
        --multitask-ckpt checkpoints/multitask/best.pth \\
        --output-dir outputs/results/
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from inference.predictor import Predictor
from evaluation.visualizer import (
    make_result_panel,
    draw_region_scores,
    denormalize_image,
    save_visualization,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_image_inference(
    predictor: Predictor,
    image_path: str,
    output_dir: Optional[str] = None,
    save_vis: bool = True,
    save_masks: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Run full inference on a single image and optionally save outputs.

    Parameters
    ----------
    predictor   : initialized Predictor instance
    image_path  : path to input RGB image
    output_dir  : directory to save results (None = don't save)
    save_vis    : save overlay visualization panel
    save_masks  : save glass mask and dirt map as PNG
    save_json   : save result JSON with scores and metadata

    Returns
    -------
    dict with prediction results and metadata
    """
    stem = Path(image_path).stem
    logger.info(f"Processing: {image_path}")

    result = predictor.predict_image(image_path)
    analysis = result["analysis"]

    output = {
        "image_path": str(image_path),
        "cleanliness_score": result["score"],
        "grade": result["grade"],
        "glass_coverage": analysis.glass_coverage,
        "mean_dirt": analysis.mean_dirt_over_glass,
        "max_dirt": analysis.max_dirt_over_glass,
        "per_region": analysis.per_region_scores,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if save_masks:
            glass_path = str(out / f"{stem}_glass_mask.png")
            dirt_path = str(out / f"{stem}_dirt_map.png")
            cv2.imwrite(glass_path, (result["glass_mask"] * 255).astype(np.uint8))
            cv2.imwrite(
                dirt_path,
                cv2.applyColorMap(
                    (result["dirt_map"] * 255).astype(np.uint8),
                    cv2.COLORMAP_JET,
                ),
            )
            output["glass_mask_path"] = glass_path
            output["dirt_map_path"] = dirt_path

        if save_vis:
            # Load the original image for visualization
            img_bgr = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            vis_path = str(out / f"{stem}_result.png")
            save_visualization(
                output_path=vis_path,
                image_rgb=img_rgb,
                glass_mask=result["glass_mask"],
                dirt_map=result["dirt_map"],
                score=result["score"],
                grade=result["grade"],
                region_scores=analysis.per_region_scores,
            )
            output["visualization_path"] = vis_path
            logger.info(f"Saved visualization: {vis_path}")

        if save_json:
            json_path = str(out / f"{stem}_result.json")
            # Convert non-serializable types
            serializable = {
                k: v for k, v in output.items()
                if not isinstance(v, np.ndarray)
            }
            with open(json_path, "w") as f:
                json.dump(serializable, f, indent=2)
            output["json_path"] = json_path

    logger.info(
        f"  Score: {result['score']:.3f} | Grade: {result['grade']} | "
        f"Glass coverage: {analysis.glass_coverage:.1%}"
    )
    return output


def run_batch_folder(
    predictor: Predictor,
    input_dir: str,
    output_dir: str,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
) -> List[Dict[str, Any]]:
    """Process all images in a directory."""
    input_path = Path(input_dir)
    image_files = [
        p for p in sorted(input_path.iterdir())
        if p.suffix.lower() in extensions
    ]
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    results = []
    for img_path in image_files:
        try:
            result = run_image_inference(predictor, str(img_path), output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")

    # Summary statistics
    scores = [r["cleanliness_score"] for r in results]
    if scores:
        logger.info(
            f"Batch complete: {len(scores)} images | "
            f"Mean score: {np.mean(scores):.3f} ± {np.std(scores):.3f} | "
            f"Min: {min(scores):.3f} | Max: {max(scores):.3f}"
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Image inference for glass cleanliness detection")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to input image")
    group.add_argument("--input-dir", type=str, help="Directory of input images")

    ckpt_group = parser.add_argument_group("Checkpoint paths")
    ckpt_group.add_argument("--glass-ckpt", type=str, default=None)
    ckpt_group.add_argument("--dirt-ckpt", type=str, default=None)
    ckpt_group.add_argument("--multitask-ckpt", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default="outputs/inference")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--glass-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    parser.add_argument("--no-masks", action="store_true", help="Skip saving raw masks")
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

    if args.input:
        run_image_inference(
            predictor=predictor,
            image_path=args.input,
            output_dir=args.output_dir,
            save_vis=not args.no_vis,
            save_masks=not args.no_masks,
        )
    else:
        run_batch_folder(
            predictor=predictor,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
