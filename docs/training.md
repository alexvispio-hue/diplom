# Обучение русской модели

## Датасет

В проект подключается `Cyrillic Handwriting Dataset`:

- 72 286 обучающих изображений;
- 1 544 тестовых изображения;
- подписи находятся в `train.tsv` и `test.tsv`;
- каждый пример содержит короткий рукописный фрагмент на русском языке.

Локальное расположение:

```text
data/training/cyrillic_handwriting/
  train.tsv
  test.tsv
  train/
  test/
```

Изображения и архив не публикуются в GitHub.

## Baseline

Приложение использует готовую русскоязычную TrOCR-модель:

```text
kazars24/trocr-base-handwritten-ru
```

Запуск оценки:

```bash
python scripts/evaluate_cyrillic_dataset.py --limit 100
```

Результат сохраняется локально в `data/evaluation/cyrillic_baseline.csv`.

Для оценки влияния предобработки:

```bash
python scripts/evaluate_cyrillic_dataset.py \
  --limit 100 \
  --preprocess \
  --output data/evaluation/cyrillic_preprocessed.csv
```

Проверенный исходный baseline на первых 100 тестовых изображениях:

```text
CER: 0.1140
WER: 0.4325
```

Сравнение на одинаковых первых 30 тестовых изображениях:

| Режим | CER | WER |
| --- | ---: | ---: |
| Без предобработки | 0.0409 | 0.2278 |
| С предобработкой | 0.0542 | 0.2889 |

На уже чистых строках датасета предобработка ухудшила результат, поэтому в интерфейсе она сделана переключаемой. Для фотографий листа или тетради предобработка может быть полезна из-за освещения, шума и геометрических искажений.

## Fine-tuning

Скрипт обучения:

```bash
python scripts/train_trocr_ru.py \
  --dataset-dir data/training/cyrillic_handwriting \
  --epochs 1 \
  --batch-size 4
```

На CPU полное обучение будет очень медленным. Для полноценного эксперимента рекомендуется GPU-среда, например Kaggle Notebook или Google Colab, после чего сохраненную модель можно подключить в приложение через `TROCR_MODEL_NAME`.

Для локальной NVIDIA GPU в WSL:

```bash
python -m pip install --upgrade --force-reinstall torch \
  --index-url https://download.pytorch.org/whl/cu130
```

Скрипт обучения автоматически использует CUDA и mixed precision, если GPU доступна, выводит промежуточный loss и сохраняет checkpoint после каждой эпохи.

Полный локальный запуск на RTX 5070 Ti:

```bash
python -u scripts/train_trocr_ru.py \
  --dataset-dir data/training/cyrillic_handwriting \
  --base-model kazars24/trocr-base-handwritten-ru \
  --output-dir models/trocr-cyrillic-finetuned \
  --epochs 1 \
  --batch-size 32 \
  --num-workers 4 \
  --max-eval-samples 1544 \
  --log-every 100
```

Проверка фонового процесса:

```bash
bash scripts/training_status_wsl.sh
```

Для проверки, что обучающий pipeline запускается, без сохранения checkpoint:

```bash
python scripts/train_trocr_ru.py \
  --max-train-samples 1 \
  --max-eval-samples 1 \
  --batch-size 1 \
  --epochs 1 \
  --skip-save
```

Проверка на одном обучающем и одном тестовом примере выполнена успешно:

```text
train_loss=0.0057
validation_loss=0.0020
```

Полный fine-tuning на RTX 5070 Ti завершен за `98.58` минут. Итоговые метрики приведены в `docs/experiment_results.md`.

## Повторное обучение на школьных тетрадях и синтетических словах

Для повышения качества на фотографиях слов используется смесь:

- исходного `Cyrillic Handwriting Dataset`;
- реальных слов из `ai-forever/school_notebooks_RU`;
- синтетических русских слов с разными шрифтами и фото-аугментациями.

Подготовка вырезок из распакованного школьного датасета:

```bash
python scripts/prepare_school_notebooks.py \
  --source-dir /mnt/c/Users/Aleksandr/Desktop/schooldataset \
  --output-dir data/training/school_notebooks_words
```

Генерация 100 000 синтетических слов:

```bash
python scripts/generate_synthetic_words.py \
  --source-manifests \
    data/training/cyrillic_handwriting/train.tsv \
    data/training/school_notebooks_words/train.tsv \
  --output-dir data/training/synthetic_ru_words \
  --samples 100000
```

Повторное обучение начинается с уже сохраненного checkpoint:

```bash
python -u scripts/train_trocr_ru.py \
  --dataset-dir data/training/cyrillic_handwriting \
  --extra-train-dataset-dir data/training/school_notebooks_words \
  --extra-train-dataset-dir data/training/synthetic_ru_words \
  --eval-dataset-dir data/training/school_notebooks_words \
  --base-model models/trocr-cyrillic-finetuned \
  --output-dir models/trocr-cyrillic-mixed \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --num-workers 4 \
  --max-eval-samples 5000 \
  --augment \
  --log-every 250
```

Скрипт применяет к обучающим изображениям случайные изменения яркости, контраста, небольшой поворот, размытие, шум и JPEG-сжатие. Проверочная выборка не искажается.

Для удобного фонового запуска полного смешанного обучения:

```bash
bash scripts/start_mixed_training_wsl.sh
```

Фоновый сценарий запускает одну полную эпоху. После проверки полученного checkpoint можно повторить обучение или вручную увеличить число эпох.

Просмотр прогресса и загрузки GPU:

```bash
bash scripts/mixed_training_status_wsl.sh
```

Остановка процесса:

```bash
bash scripts/stop_mixed_training_wsl.sh
```

## Пользовательская обратная связь

После распознавания пользователь может оценить совпадение текста с изображением:

- `Полностью совпадает`;
- `Частично совпадает`;
- `Не совпадает`.

Для двух проблемных вариантов приложение автоматически сохраняет исходное изображение и JSON-описание ответа модели:

```text
data/training/user_feedback/problems/
  recognition_49.jpg
  recognition_49.json
```

Эта папка является очередью на последующую разметку. Перед добавлением примеров в обучение нужно указать правильный текст: ошибочный ответ модели нельзя автоматически считать эталоном.

## Подготовка Qwen-VL SFT датасета

Для мультимодального OCR-контурa Qwen данные переводятся в JSONL-формат с сообщениями `user` и `assistant`.

Сначала из `school_notebooks_RU` можно подготовить датасет строк и страниц:

```bash
python scripts/prepare_school_page_lines.py \
  --source-dir data/training/schooldataset \
  --output-dir data/training/page_lines
```

Для быстрой проверки формата без долгой нарезки всего набора:

```bash
python scripts/prepare_school_page_lines.py \
  --source-dir data/training/schooldataset \
  --output-dir data/training/page_lines_smoke \
  --splits train \
  --max-groups-per-split 20
```

Скрипт создает:

```text
data/training/page_lines/train_lines.csv
data/training/page_lines/val_lines.csv
data/training/page_lines/test_lines.csv
data/training/page_lines/train_pages.csv
data/training/page_lines/pages_text/
data/training/page_lines/lines/
```

Подготовка примеров строк:

```bash
python scripts/build_qwen_sft_dataset.py \
  --lines-csv data/training/page_lines/train_lines.csv \
  --output data/training/qwen_sft/lines.jsonl \
  --include-layout
```

Подготовка примеров полных страниц:

```bash
python scripts/build_qwen_sft_dataset.py \
  --pages-csv data/training/page_lines/train_pages.csv \
  --output data/training/qwen_sft/pages.jsonl
```

Можно объединить оба источника в один файл:

```bash
python scripts/build_qwen_sft_dataset.py \
  --lines-csv data/training/page_lines/train_lines.csv \
  --pages-csv data/training/page_lines/train_pages.csv \
  --output data/training/qwen_sft/mixed.jsonl \
  --include-layout
```

Ожидаемый формат CSV для строк:

```csv
image_path,text,page_id,line_number,x1,y1,x2,y2
data/lines/page_001_line_001.jpg,пример рукописной строки,page_001,1,120,80,900,145
```

Ожидаемый формат CSV для страниц:

```csv
image_path,text_path,page_id
data/pages/page_001.jpg,data/pages_text/page_001.txt,page_001
```

Для Qwen рекомендуется не полное обучение модели, а LoRA/QLoRA SFT. Базовая модель выбирается через `QWEN_MODEL_PRESET`:

```text
QWEN_MODEL_PRESET=light
```

Доступные пресеты:

| Preset | Модель | Назначение |
| --- | --- | --- |
| `main` | `Qwen/Qwen3-VL-8B-Instruct` | основной вариант из ТЗ для GPU-среды |
| `light` | `Qwen/Qwen3-VL-2B-Instruct` | быстрые локальные smoke-тесты и отладка pipeline |
| `fallback_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | резервный вариант, если Qwen3-VL недоступна |
| `fallback_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | более лёгкий резервный вариант |

Для ручного эксперимента можно переопределить пресет прямым именем модели:

```text
QWEN_MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
```

После LoRA/QLoRA-обучения адаптер подключается без изменения кода через переменную окружения:

```text
QWEN_LORA_ADAPTER_DIR=models/qwen-vl-ocr-lora
```

При таком запуске приложение загружает базовую модель из `QWEN_MODEL_NAME`, затем накладывает PEFT/LoRA-адаптер из `QWEN_LORA_ADAPTER_DIR`. Если переменная не задана, Qwen работает в базовом режиме.

Qwen-режимы в приложении загружаются лениво: обычный TrOCR-сценарий продолжает работать без загрузки Qwen в память.

Smoke-проверка Qwen на одной строке или странице:

```bash
python scripts/smoke_qwen_ocr.py data/training/page_lines_smoke/lines/train/0_0_line_001.jpg
```

Для полной страницы:

```bash
python scripts/smoke_qwen_ocr.py data/training/schooldataset/images/0_0.jpg --page
```

Оценка режимов распознавания страницы по `pages.csv`:

```bash
python scripts/evaluate_page_modes.py \
  --pages-csv data/training/page_lines/test_pages.csv \
  --mode trocr_lines \
  --output data/evaluation/page_modes/trocr_lines.csv
```

Оценка качества сегментации строк по `lines.csv`:

```bash
python scripts/evaluate_segmentation.py \
  --lines-csv data/training/page_lines/test_lines.csv \
  --output data/evaluation/segmentation/results.csv
```

Скрипт считает число эталонных и найденных строк, долю найденных строк, пропуски, лишние строки, возможные объединения/разделения, средний IoU и среднее отклонение координат.

Qwen и hybrid-режимы:

```bash
python scripts/evaluate_page_modes.py \
  --pages-csv data/training/page_lines/test_pages.csv \
  --mode qwen_page \
  --limit 10 \
  --output data/evaluation/page_modes/qwen_page.csv
```

```bash
python scripts/evaluate_page_modes.py \
  --pages-csv data/training/page_lines/test_pages.csv \
  --mode hybrid \
  --limit 10 \
  --output data/evaluation/page_modes/hybrid.csv
```

Для черновой подготовки разметки можно запросить JSON со строками и координатами:

```bash
python scripts/evaluate_page_modes.py \
  --pages-csv data/training/page_lines/test_pages.csv \
  --mode qwen_layout \
  --limit 10 \
  --output data/evaluation/page_modes/qwen_layout.csv
```

После ручной проверки JSON-ответов Qwen их можно преобразовать в черновой `lines.csv`:

```bash
python scripts/export_qwen_layout_annotations.py \
  --layout-csv data/evaluation/page_modes/qwen_layout.csv \
  --output-csv data/training/qwen_layout_draft/lines.csv
```

Полученный CSV имеет формат `image_path,text,page_id,line_number,x1,y1,x2,y2,source_image` и может использоваться как заготовка для последующей ручной разметки и обучения.

Полный воспроизводимый эксперимент по режимам страницы запускается одной командой:

```bash
python scripts/run_page_experiment.py \
  --pages-csv data/training/page_lines/test_pages.csv \
  --output-dir data/evaluation/page_experiment \
  --modes trocr_lines qwen_lines qwen_page hybrid
```

Скрипт сохраняет отдельный CSV для каждого режима, общий `summary.csv` и отчёт `summary.md` с CER, WER и средним временем обработки.
`summary.csv` формируется в формате из итогового ТЗ:

```text
model,mode,dataset,CER,WER,page_time,line_time,notes
```

Сохранённые `summary.csv` из `data/evaluation/**` доступны в интерфейсе во вкладке `Эксперименты` и через `GET /api/experiments/page-summary`.

Проверка Qwen SFT JSONL без загрузки модели:

```bash
python scripts/train_qwen_vl_lora.py \
  --dataset data/training/qwen_sft_smoke/mixed.jsonl \
  --dry-run
```

LoRA/QLoRA-дообучение Qwen запускается отдельным скриптом. Минимальный запуск на подготовленном JSONL:

```bash
python scripts/train_qwen_vl_lora.py \
  --dataset data/training/qwen_sft/mixed.jsonl \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --output-dir models/qwen-vl-ocr-lora \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 8
```

Для реального обучения нужна GPU-среда и установленные зависимости из `requirements-ml.txt`.

В обучающем pipeline TrOCR при `--augment` используется `HandwritingPhotoAugmenter`. Он имитирует:

- изменение яркости и контраста;
- небольшой поворот;
- перспективные искажения;
- тени и бумажную текстуру;
- шум, размытие и JPEG-сжатие;
- фон тетрадной клетки или линейки.

Перед запуском реального TrOCR/Qwen эксперимента можно проверить окружение:

```bash
python scripts/check_ml_environment.py
```

Та же проверка доступна через backend и отображается в боковой панели Streamlit:

```text
GET /api/ml-status
```

Qwen-режимы страницы выполняют такую проверку до загрузки модели. Если runtime-зависимостей не хватает, API возвращает `503` со списком отсутствующих пакетов. В Streamlit такие режимы дополнительно блокируются в форме распознавания: пользователь видит предупреждение и список недостающих пакетов до отправки изображения.

Для машинного чтения:

```bash
python scripts/check_ml_environment.py --json
```

Единая runtime-проверка Qwen дополнительно сохраняет JSON-отчет в `data/runtime/qwen_runtime_check.json`.
Без `--run-smoke` она проверяет зависимости, подбирает тестовое изображение и оценивает локальные ресурсы
через свободное место на диске и `nvidia-smi`. Для машины с GPU около 16 ГБ VRAM рекомендуемый первый
профиль - `light`, то есть `Qwen/Qwen3-VL-2B-Instruct`; 8B-модель лучше оставлять для более мощной GPU-среды.
С `--run-smoke` скрипт запускает реальный Qwen-инференс:

```bash
python scripts/qwen_runtime_check.py --json
python scripts/qwen_runtime_check.py --run-smoke --page --json
```

После preflight можно собрать воспроизводимый план выполнения всех Qwen-шагов. Скрипт не запускает
тяжелые операции, а формирует JSON и Markdown с командами установки, smoke inference, сравнения моделей,
сборки SFT и LoRA-обучения:

```bash
python scripts/build_qwen_execution_plan.py --json
```

Выполнять шаги плана можно безопасным runner-ом. По умолчанию сетевые шаги блокируются:

```bash
python scripts/run_qwen_execution_plan.py --list
python scripts/run_qwen_execution_plan.py --step build_sft --dry-run
python scripts/run_qwen_execution_plan.py --step install_ml --allow-network
```

Текущую сводку готовности Qwen-части можно получить одной командой:

```bash
python scripts/qwen_project_status.py
python scripts/qwen_project_status.py --json
```

Подготовить команду установки ML-зависимостей без фактического скачивания:

```bash
python scripts/setup_ml_environment.py
```

Реальная установка запускается только явно:

```bash
python scripts/setup_ml_environment.py --profile cuda-cu130 --install
```

Для CPU-only окружения можно использовать:

```bash
python scripts/setup_ml_environment.py --profile cpu --install
```

Если нужен другой индекс PyTorch, можно передать его вручную:

```bash
python scripts/setup_ml_environment.py --extra-index-url <pytorch-wheel-index> --install
```

## Экспорт пользовательских исправлений

Проблемные примеры из интерфейса сохраняются в `data/training/user_feedback/problems`.
Для страничных режимов вместе с исходным изображением сохраняются обработанное изображение страницы и вырезанные строки, а их имена записываются в JSON-метаданные примера.
Исправленный текст можно сохранить сразу после распознавания или позднее из вкладки истории: оба сценария используют один feedback API и попадают в одну очередь разметки.
Очередь можно просмотреть во вкладке Streamlit `Разметка` или через `GET /api/feedback/problems`: там доступны исходное изображение, обработанная версия, строки, координаты, ответ модели и поле для проверенного текста.
Hybrid-режим добавляет в эту очередь спорные строки автоматически: если TrOCR и Qwen существенно расходятся, запись получает `needs_review=true` и `review_notes` с текстом обеих моделей.
После ручной правки текста их можно выгрузить в CSV:

```bash
python scripts/export_feedback_training_data.py \
  --feedback-dir data/training/user_feedback/problems \
  --output-csv data/training/user_feedback/reviewed_examples.csv \
  --only-reviewed
```

Полученный CSV можно использовать как дополнительный источник строковых или страничных примеров. В обучение должны попадать только проверенные записи с `corrected_text`.

Для Qwen можно сразу дополнительно сформировать SFT JSONL:

```bash
python scripts/export_feedback_training_data.py \
  --feedback-dir data/training/user_feedback/problems \
  --output-csv data/training/user_feedback/reviewed_examples.csv \
  --output-qwen-jsonl data/training/user_feedback/qwen_feedback.jsonl \
  --only-reviewed
```
