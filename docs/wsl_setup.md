# WSL Ubuntu Setup

## Окружение

Проверенная конфигурация:

- WSL: Ubuntu 22.04, WSL2;
- Python: 3.10.12;
- виртуальное окружение: `/home/leksandr/.venvs/diplom-ocr`;
- PyTorch: CPU-сборка;
- модель: `microsoft/trocr-base-handwritten`.

## Активация

```bash
cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
```

## Проверки

```bash
python -m pytest tests
python scripts/check_trocr.py
bash scripts/run_tests_wsl.sh
```

## Запуск Backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Запуск UI

Во втором терминале WSL:

```bash
cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Пакетная оценка

```bash
python scripts/evaluate_dataset.py \
  --manifest data/evaluation/manifest.csv \
  --output data/evaluation/results.csv
```

## Примечания

Предупреждение Hugging Face про `HF_TOKEN` не мешает работе. Токен нужен только для более высоких лимитов скачивания и приватных моделей.

Текущая модель TrOCR ориентирована на английский рукописный текст. Для русскоязычного дипломного сценария дальше понадобится либо подобрать модель под кириллицу, либо дообучить OCR-модель на подходящем наборе данных.
