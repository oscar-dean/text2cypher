#!/usr/bin/env bash
set -euo pipefail

THREAD_COUNT=4
MODEL_PATH="${1:-HuggingFaceTB/SmolLM2-135M-Instruct}"
OUTPUT_PATH="${2:-results/evaluation.json}"

OMP_NUM_THREADS="$THREAD_COUNT" \
MKL_NUM_THREADS="$THREAD_COUNT" \
uv run python evaluate.py \
  --model-path "$MODEL_PATH" \
  --split test \
  --output-path "$OUTPUT_PATH" \
  --num-threads "$THREAD_COUNT"