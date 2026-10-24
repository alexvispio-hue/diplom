#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${DIPLOM_OCR_VENV:-$HOME/.venvs/diplom-ocr}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating Python environment: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p data/runtime

if [[ -f data/runtime/backend.pid ]] && kill -0 "$(cat data/runtime/backend.pid)" 2>/dev/null; then
  echo "Backend is already running."
else
  nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 \
    > data/runtime/backend.log 2>&1 < /dev/null &
  echo "$!" > data/runtime/backend.pid
  echo "Backend started."
fi

if [[ -f data/runtime/ui.pid ]] && kill -0 "$(cat data/runtime/ui.pid)" 2>/dev/null; then
  echo "UI is already running."
else
  nohup streamlit run ui/streamlit_app.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true \
    > data/runtime/ui.log 2>&1 < /dev/null &
  echo "$!" > data/runtime/ui.pid
  echo "UI started."
fi

sleep 3
echo "UI: http://127.0.0.1:8501"
echo "API docs: http://127.0.0.1:8000/docs"
