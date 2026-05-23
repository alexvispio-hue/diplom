# Handwriting OCR

Дипломный проект: система распознавания рукописного текста на изображениях с использованием методов глубокого обучения.

## Что делает приложение

- принимает изображение с рукописным текстом;
- выполняет локальную предобработку изображения;
- запускает русскоязычную локальную OCR-модель TrOCR через PyTorch;
- возвращает распознанный текст;
- сохраняет историю распознаваний в SQLite;
- содержит модуль расчета CER/WER для экспериментальной части;
- умеет пакетно оценивать качество распознавания по CSV-манифесту;
- позволяет включать или отключать предобработку изображения в зависимости от качества входа;
- предоставляет REST API и простой Streamlit-интерфейс.

## Архитектура

```text
Пользовательский интерфейс
        |
        v
FastAPI backend
        |
        v
RecognitionService
        |
        +--> ImagePreprocessor
        +--> TrOCRRecognizer
        +--> SQLite history
```

Ключевая идея проекта: распознавание выполняется локально, без внешнего OCR API. По умолчанию используется `kazars24/trocr-base-handwritten-ru`, дообученная под русский рукописный текст. Веса модели загружаются из Hugging Face при первом запуске, после чего инференс выполняется внутри приложения.

## API

- `GET /api/health` — состояние приложения;
- `GET /api/models` — сведения о подключенной OCR-модели;
- `POST /api/recognize` — загрузка изображения и запуск распознавания;
- `GET /api/history` — история распознаваний;
- `GET /api/history/{id}` — отдельный результат.

## Запуск в WSL Ubuntu

Рекомендуемый вариант для нейросетевой части:

```bash
cd /mnt/c/Users/Aleksandr/Desktop/diplom
source ~/.venvs/diplom-ocr/bin/activate
```

Проверка TrOCR:

```bash
python scripts/check_trocr.py
```

Запуск backend:

```bash
bash scripts/run_backend_wsl.sh
```

Запуск интерфейса во втором терминале:

```bash
bash scripts/run_ui_wsl.sh
```

Запуск тестов:

```bash
bash scripts/run_tests_wsl.sh
```

После запуска:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## Установка с нуля

```bash
cd /mnt/c/Users/Aleksandr/Desktop/diplom
python3 -m venv ~/.venvs/diplom-ocr
source ~/.venvs/diplom-ocr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers sentencepiece
```

Первый запуск распознавания может занять больше времени: приложение скачает веса модели в локальный кэш, а затем будет выполнять инференс локально.

## Экспериментальная оценка

Создайте файл `data/evaluation/manifest.csv` по шаблону:

```csv
image_path,reference_text
samples/example_01.png,hello world
```

Путь к изображению указывается относительно папки `data/evaluation`.

Запуск оценки:

```bash
python scripts/evaluate_dataset.py \
  --manifest data/evaluation/manifest.csv \
  --output data/evaluation/results.csv
```

На выходе формируется CSV с распознанным текстом, CER, WER, расстоянием Левенштейна и временем обработки.

Для установленного `Cyrillic Handwriting Dataset` предусмотрен прямой запуск оценки по исходным TSV-файлам:

```bash
python scripts/evaluate_cyrillic_dataset.py --limit 100
```

Инструкция по дообучению модели: `docs/training.md`.

Локально подготовленный набор расположен в `data/training/cyrillic_handwriting` и исключен из GitHub, поскольку содержит большой объем изображений.

## Docker

Backend можно собрать и запустить через Docker:

```bash
docker compose up --build
```

Docker-режим использует CPU-версию PyTorch и сохраняет кэш Hugging Face в отдельном volume.

## Структура проекта

```text
app/
  api/                REST API
  core/               конфигурация
  image_processing/   предобработка изображений
  ocr/                OCR-модели, метрики и постобработка
  services/           бизнес-логика распознавания
  storage/            база данных и модели хранения
scripts/              служебные проверки и запуск
ui/                   пользовательский интерфейс
data/                 загрузки, обработанные изображения, SQLite
models/               локальные модели или заметки по моделям
docs/                 проектная документация
tests/                автоматические проверки
```

## Текущее ограничение

Текущая модель `kazars24/trocr-base-handwritten-ru` поддерживает русский рукописный текст на уровне коротких фрагментов. Для чистых вырезанных строк предобработку можно отключать; для фотографий она доступна как опция. Для распознавания целой страницы конспекта потребуется дополнительно реализовать выделение строк на фотографии страницы.

## Дальнейшее развитие

- провести полный baseline-эксперимент на тестовой части датасета;
- дообучить модель и сравнить CER/WER до и после обучения;
- добавить сегментацию фотографии страницы на отдельные строки;
- добавить поддержку CRNN как второй локальной архитектуры;
- реализовать страницу сравнения моделей;
- добавить экспорт результата в `.docx`.

Подробности по WSL-настройке: `docs/wsl_setup.md`.
