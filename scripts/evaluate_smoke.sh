#!/usr/bin/env bash
set -euo pipefail

THREAD_COUNT=2
MODEL_PATH="${1:-HuggingFaceTB/SmolLM2-135M-Instruct}"
OUTPUT_PATH="${2:-outputs/evaluation_smoke.json}"

OMP_NUM_THREADS="$THREAD_COUNT" \
MKL_NUM_THREADS="$THREAD_COUNT" \
caffeinate -is \
uv run python evaluate.py \
  --model-path "$MODEL_PATH" \
  --split test \
  --max-samples 2 \
  --output-path "$OUTPUT_PATH" \
  --num-threads "$THREAD_COUNT"