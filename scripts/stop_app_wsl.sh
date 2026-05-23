#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom

stop_process() {
  local name="$1"
  local pid_file="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "$name stopped."
    else
      echo "$name is not running."
    fi
    rm -f "$pid_file"
  else
    echo "$name PID file is missing."
  fi
}

stop_process "Backend" "data/runtime/backend.pid"
stop_process "UI" "data/runtime/ui.pid"

# Also stop development instances started before PID tracking was added.
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000" 2>/dev/null || true
pkill -f "streamlit run ui/streamlit_app.py" 2>/dev/null || true
