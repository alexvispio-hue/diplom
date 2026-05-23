#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
