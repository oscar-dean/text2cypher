#!/usr/bin/env bash
set -euo pipefail

THREAD_COUNT=4

if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -is)
else
  KEEP_AWAKE=()
fi

OMP_NUM_THREADS="$THREAD_COUNT" \
MKL_NUM_THREADS="$THREAD_COUNT" \
"${KEEP_AWAKE[@]}" \
uv run python train.py \
  --epochs 1 \
  --max-train-samples 8 \
  --max-eval-samples 4 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-threads "$THREAD_COUNT" \
  --output-dir outputs/smoke_checkpoints \
  --final-model-dir outputs/smoke_final_model \
  --training-history-path outputs/smoke_training_history.json \
  --training-plot-path outputs/smoke_training_loss.png