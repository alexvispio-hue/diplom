#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom

PID_FILE="data/runtime/mixed_training.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "Mixed training PID file is missing."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Mixed training stopped. PID: $PID"
else
  echo "Mixed training process is not running. Last PID: $PID"
fi

rm -f "$PID_FILE"
