#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
python -m pytest tests
