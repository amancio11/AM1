#!/usr/bin/env bash
# =============================================================
# train_all.sh
# Train all three models sequentially.
#
# Usage:
#   ./scripts/train_all.sh
# =============================================================

set -euo pipefail

PYTHON="${PYTHON:-python}"
SRC="src"

echo "======================================================"
echo "  Training Pipeline — All Models"
echo "======================================================"

echo ""
echo "[1/3] Training Glass Segmentation Model..."
$PYTHON $SRC/training/train_glass.py --config configs/glass_seg_config.yaml
echo "[1/3] Done."

echo ""
echo "[2/3] Training Dirt Estimation Model..."
$PYTHON $SRC/training/train_dirt.py --config configs/dirt_est_config.yaml
echo "[2/3] Done."

echo ""
echo "[3/3] Training Multi-Task Model..."
$PYTHON $SRC/training/train_multitask.py --config configs/multitask_config.yaml
echo "[3/3] Done."

echo ""
echo "======================================================"
echo "  All models trained successfully."
echo "======================================================"
