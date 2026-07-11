# Handwriting OCR

Локальное приложение для распознавания русскоязычного рукописного текста на изображениях и многостраничных документах.

## Что делает приложение

- принимает несколько изображений или PDF-документ;
- выполняет локальную предобработку изображения;
- запускает русскоязычную локальную OCR-модель TrOCR через PyTorch;
- умеет распознавать страницу конспекта: выделяет строки и собирает многострочный результат;
- возвращает распознанный текст;
- объединяет страницы в один редактируемый документ;
- экспортирует результат в TXT, JSON и DOCX;
- принимает пользовательскую оценку совпадения и сохраняет проблемные примеры для последующего дообучения;
- сохраняет историю распознаваний в SQLite;
- содержит модуль расчета CER/WER для экспериментальной части;
- умеет пакетно оценивать качество распознавания по CSV-манифесту;
- позволяет включать или отключать предобработку изображения в зависимости от качества входа;
- предоставляет REST API и полноценную локальную рабочую область.

## Быстрый запуск в Windows

Для установки запустите `INSTALL_WINDOWS.bat`, затем открывайте приложение через `START_APP.bat` или созданный ярлык на рабочем столе.

Подробная инструкция: [docs/installation.md](docs/installation.md).

План продуктового развития: [docs/product_roadmap.md](docs/product_roadmap.md).

Действующее ТЗ до версии 1.0: [docs/product_technical_specification.md](docs/product_technical_specification.md).

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
        +--> PageLineSegmenter
        +--> TrOCRRecognizer
        +--> SQLite history
```

Ключевая идея проекта: распознавание выполняется локально, без внешнего OCR API. Приложение автоматически выбирает наиболее обученный локальный checkpoint: сначала `models/trocr-cyrillic-mixed-epoch-2`, затем `models/trocr-cyrillic-mixed`, затем `models/trocr-cyrillic-finetuned`. Если локальных моделей нет, используется `kazars24/trocr-base-handwritten-ru` из Hugging Face.

## API

- `GET /api/health` — состояние приложения;
- `GET /api/models` — сведения о подключенных TrOCR и Qwen-моделях, включая выбранный Qwen preset или ручной override;
- `GET /api/ml-status` — готовность ML-зависимостей для TrOCR, Qwen/LoRA и CUDA;
- `GET /api/experiments/page-summary` — сохранённые сводки экспериментов страниц по CER/WER/времени;
- `POST /api/recognize` — загрузка изображения и запуск распознавания;
- `POST /api/recognize-page` — legacy-распознавание страницы с автоматическим выделением строк;
- `POST /api/recognize/page` — распознавание страницы с выбором режима `trocr_lines`, `qwen_lines`, `qwen_page`, `qwen_layout`, `compare` или `hybrid`;
- `POST /api/recognize/page/qwen` — совместимый маршрут из ТЗ для передачи полной страницы в Qwen целиком;
- `POST /api/recognize/page/compare` — совместимый маршрут из ТЗ для сравнения TrOCR и Qwen по строкам;
- `POST /api/feedback` — сохранение оценки и исправленного текста по `recognition_id`;
- `GET /api/feedback/stats` — агрегированные метрики пользовательских оценок;
- `GET /api/feedback/problems` — очередь проблемных примеров для ручной разметки;
- `POST /api/history/{id}/feedback` — сохранение оценки и исправленного текста для записи истории;
- `GET /api/history` — история распознаваний;
- `GET /api/history/{id}` — отдельный результат.

Для `/api/recognize/page` доступны параметры формы `preprocess`, `mode`, `model`, `max_tokens`, `save_intermediate_images` и `return_coordinates`. Для построчных режимов ответ может содержать `line_image_urls` и `line_coordinates`.

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

Удобный запуск всего приложения в фоне одной командой:

```bash
bash scripts/start_app_wsl.sh
```

Остановка backend и интерфейса:

```bash
bash scripts/stop_app_wsl.sh
```

Запуск тестов:

```bash
bash scripts/run_tests_wsl.sh
```

Проверка распознавания тестовой страницы при запущенном приложении:

```bash
python scripts/smoke_page_api.py
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
Результаты выполненного эксперимента: `docs/experiment_results.md`.

Локально подготовленный набор расположен в `data/training/cyrillic_handwriting` и исключен из GitHub, поскольку содержит большой объем изображений.
Для повторного обучения на реальных фотографиях слов можно подключить распакованный `school_notebooks_RU` и локально сгенерированные синтетические слова. Команды подготовки приведены в `docs/training.md`.

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

Локальная TrOCR-модель распознает отдельные строки. Для страницы приложение автоматически выделяет строки эвристическим OpenCV-сегментатором и передает их модели по очереди. Этот режим подходит для ровных фотографий листа с горизонтальными строками. Для сложных страниц с наклоном, нестандартным расположением текста и пересекающимися строками следующим этапом предусмотрено обучение нейросетевого сегментатора.

## Дальнейшее развитие

- провести полный baseline-эксперимент на тестовой части датасета;
- дообучить модель и сравнить CER/WER до и после обучения;
- обучить сегментатор страниц Kraken на размеченных фотографиях конспектов;
- добавить поддержку CRNN как второй локальной архитектуры;
- реализовать страницу сравнения моделей;
- добавить экспорт результата в `.docx`.

Подробности по WSL-настройке: `docs/wsl_setup.md`.
План развития распознавания страниц: `docs/page_recognition.md`.
