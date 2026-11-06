#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 GRID_DIR [GPU_IDS] [RESOLUTION]" >&2
    exit 1
fi
IMAGE_ROOT_PATH=$1
GPU_IDS=${2:-0}
RESOLUTION=${3:-0}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
PROCESSES=${#GPU_LIST[@]}
PORT=${PORT:-29500}
RES_PATH="$IMAGE_ROOT_PATH/dpg-bench_$(date +%Y%m%d-%H%M%S)_results.txt"

LAUNCH_ARGS=(--num_machines 1 --num_processes "$PROCESSES" --mixed_precision fp16 --main_process_port "$PORT")
if [ "$PROCESSES" -gt 1 ]; then
    LAUNCH_ARGS+=(--multi_gpu)
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch "${LAUNCH_ARGS[@]}" \
    "$SCRIPT_DIR/dpg_bench/compute_dpg_bench_refine.py" \
    --image-root-path "$IMAGE_ROOT_PATH" \
    --csv "$SCRIPT_DIR/dpg_bench/dpg_bench.csv" \
    --res-path "$RES_PATH" \
    --resolution "$RESOLUTION" \
    --pic-num 4 \
    --vqa-model mplug
