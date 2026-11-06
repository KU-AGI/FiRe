#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash $0 IMAGE_STAGE_DIR OUTPUT_DIR DETECTOR_DIR [GPU_ID]" >&2
    exit 1
fi
GEN_PATH=$1
OUT_PATH=$2
MODEL_PATH=$3
GPU_ID=${4:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_PATH"

CUDA_VISIBLE_DEVICES="$GPU_ID" python "$SCRIPT_DIR/evaluation/evaluate_images_batch.py" \
    "$GEN_PATH" \
    --folders-per-pass 64 \
    --num-workers 8 \
    --prefetch-batches 4 \
    --outfile "$OUT_PATH/results.jsonl" \
    --model-path "$MODEL_PATH" \
    --options clip_batch=64 clip_workers=8

python "$SCRIPT_DIR/evaluation/summary_scores_v2.py" "$OUT_PATH/results.jsonl"
