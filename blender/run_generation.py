"""
run_generation.py
=================
Entry point for the synthetic data generation pipeline.

Usage (run from terminal via Blender):
    blender --background --python blender/run_generation.py -- \\
        --config configs/blender_config.yaml \\
        --start 0 \\
        --end 2000 \\
        --workers 1

The script must be executed INSIDE Blender's Python environment.
Run with: blender --background --python blender/run_generation.py -- [args]

Note: args after `--` are passed to this script.
"""

import sys
import os
import argparse
import yaml
import traceback
from pathlib import Path

# Ensure blender/ directory is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# This import works only inside Blender
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender (bpy not available).")
    sys.exit(1)

from render_pipeline import RenderPipeline


def parse_args() -> argparse.Namespace:
    # Blender passes its own args before `--`, so we split on `--`
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Synthetic building facade data generation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/blender_config.yaml",
        help="Path to Blender generation config YAML.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First scene ID (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last scene ID (exclusive). Defaults to config num_scenes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-rendered scenes (based on existing image files).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and report counts without rendering.",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def scene_already_rendered(scene_id: int, output_dir: str) -> bool:
    scene_name = f"scene_{scene_id:06d}"
    img_path = Path(output_dir) / "images" / f"{scene_name}.png"
    return img_path.exists()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  AI Glass Cleanliness — Synthetic Data Generator")
    print("=" * 60)

    config = load_config(args.config)

    if args.output_dir:
        config["render"]["output_dir"] = args.output_dir

    num_scenes = config["generation"]["num_scenes"]
    start_id = args.start
    end_id = args.end if args.end is not None else num_scenes

    print(f"Config:      {args.config}")
    print(f"Output:      {config['render']['output_dir']}")
    print(f"Scenes:      {start_id} → {end_id} ({end_id - start_id} total)")
    print(f"Engine:      {config['render']['engine']} / {config['render']['device']}")
    print(f"Samples:     {config['render']['samples']}")
    print(f"Resolution:  {config['render']['resolution_x']}x{config['render']['resolution_y']}")

    if args.dry_run:
        print("\nDry run — exiting without rendering.")
        return

    pipeline = RenderPipeline(config)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for scene_id in range(start_id, end_id):
        if args.resume and scene_already_rendered(scene_id, config["render"]["output_dir"]):
            skipped_count += 1
            continue

        try:
            metadata = pipeline.render_scene(scene_id)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Scene {scene_id:06d} failed: {e}")
            traceback.print_exc()
            fail_count += 1
            continue

    print("\n" + "=" * 60)
    print(f"  Done. Success: {success_count} | Failed: {fail_count} | Skipped: {skipped_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
