#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom

PID_FILE="data/runtime/mixed_training.pid"
LOG_FILE="data/runtime/mixed_training.log"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Mixed training PID file is missing."
  exit 1
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
else
  echo "Mixed training process is not running. Last PID: $PID"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi

LAST_STEP="$(grep -E 'epoch [0-9]+ step [0-9]+/[0-9]+' "$LOG_FILE" | tail -n 1 || true)"
if [[ -n "$LAST_STEP" ]]; then
  echo
  python3 - "$LAST_STEP" <<'PY'
import re
import sys

match = re.search(r"epoch (\d+) step (\d+)/(\d+)", sys.argv[1])
epoch, step, steps_per_epoch = map(int, match.groups())
total_epochs = 1
completed_steps = (epoch - 1) * steps_per_epoch + step
total_steps = total_epochs * steps_per_epoch
print(f"Progress: {completed_steps}/{total_steps} steps ({completed_steps / total_steps * 100:.2f}%)")
PY
fi

echo
echo "Recent log:"
tail -n 25 "$LOG_FILE"
