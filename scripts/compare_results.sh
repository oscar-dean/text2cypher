#!/usr/bin/env bash
set -euo pipefail

BASE_RESULTS="${1:-results/base_model_test.json}"
FINE_TUNED_RESULTS="${2:-results/fine_tuned_model_test.json}"
NUM_EXAMPLES="${3:-5}"

uv run python scripts/analysis/compare_results.py \
  --base-results "$BASE_RESULTS" \
  --fine-tuned-results "$FINE_TUNED_RESULTS" \
  --output results/model_comparison.md \
  --num-examples "$NUM_EXAMPLES"
