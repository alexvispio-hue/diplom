#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom

PID_FILE="data/training/logs/full_training.pid"
LOG_FILE="data/training/logs/full_training.log"

if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; then
    ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
  else
    echo "Training process is not running."
  fi
else
  echo "Training PID file is missing."
fi

if [[ -f "$LOG_FILE" ]]; then
  tail -n 30 "$LOG_FILE"
fi

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
