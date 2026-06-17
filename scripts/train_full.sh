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
  --epochs 3 \
  --learning-rate 5e-5 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --num-threads "$THREAD_COUNT" \
  --output-dir checkpoints \
  --final-model-dir artifacts/final_model \
  --training-history-path results/training_history.json \
  --training-plot-path results/training_loss.png