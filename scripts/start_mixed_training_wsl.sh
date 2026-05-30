#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate

mkdir -p data/runtime
PID_FILE="data/runtime/mixed_training.pid"
LOG_FILE="data/runtime/mixed_training.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Mixed training is already running. PID: $(cat "$PID_FILE")"
  exit 0
fi

nohup python -u scripts/train_trocr_ru.py \
  --dataset-dir data/training/cyrillic_handwriting \
  --extra-train-dataset-dir data/training/school_notebooks_words \
  --extra-train-dataset-dir data/training/synthetic_ru_words \
  --eval-dataset-dir data/training/school_notebooks_words \
  --base-model models/trocr-cyrillic-finetuned \
  --output-dir models/trocr-cyrillic-mixed \
  --epochs 1 \
  --batch-size 24 \
  --learning-rate 1e-5 \
  --num-workers 4 \
  --max-eval-samples 5000 \
  --augment \
  --log-every 50 \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Mixed training started."
echo "PID: $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
