#!/usr/bin/env bash
# =============================================================
# generate_dataset.sh
# Run the Blender synthetic data generation pipeline.
#
# Usage:
#   ./scripts/generate_dataset.sh [start] [end] [--resume]
#
# Requirements:
#   - Blender 4.x installed and on PATH (or modify BLENDER_PATH below)
#   - CUDA-capable GPU recommended
# =============================================================

set -euo pipefail

BLENDER_PATH="${BLENDER_PATH:-blender}"
CONFIG="${CONFIG:-configs/blender_config.yaml}"
SCRIPT="blender/run_generation.py"
START="${1:-0}"
END="${2:-}"
RESUME="${3:-}"

echo "======================================================"
echo "  Synthetic Data Generation Pipeline"
echo "======================================================"
echo "  Blender:  $BLENDER_PATH"
echo "  Config:   $CONFIG"
echo "  Range:    $START → ${END:-config default}"

# Build arguments
ARGS="--config $CONFIG --start $START"
if [ -n "$END" ]; then
  ARGS="$ARGS --end $END"
fi
if [ "$RESUME" = "--resume" ]; then
  ARGS="$ARGS --resume"
fi

echo "  Args:     $ARGS"
echo "======================================================"

$BLENDER_PATH --background --python $SCRIPT -- $ARGS

echo "======================================================"
echo "  Generation complete."
echo "======================================================"
