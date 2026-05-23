#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
