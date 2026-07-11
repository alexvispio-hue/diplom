# Результаты эксперимента

## Обучение

Fine-tuning модели выполнялся локально на `NVIDIA GeForce RTX 5070 Ti`:

```text
Модель-основа: kazars24/trocr-base-handwritten-ru
Train samples: 72 286
Test samples: 1 544
Epochs: 1
Batch size: 32
Mixed precision: true
Время обучения: 98.58 мин
Train loss: 0.1060
Validation loss: 0.3010
```

## Сравнение моделей

Оценка проведена на всей тестовой выборке `Cyrillic Handwriting Dataset` без дополнительной предобработки.

| Модель | CER | WER |
| --- | ---: | ---: |
| Baseline `kazars24/trocr-base-handwritten-ru` | 0.1002 | 0.4096 |
| Fine-tuned checkpoint (1 epoch) | **0.0959** | **0.4040** |

## Вывод

После одной эпохи дообучения модель показала небольшое улучшение на полной тестовой выборке:

- CER уменьшился на `0.0043`;
- WER уменьшился на `0.0056`.

На короткой подвыборке из 100 примеров результат был менее устойчивым, поэтому итоговый вывод основан на полном test split из 1 544 изображений.

Обученный checkpoint хранится локально в `models/trocr-cyrillic-finetuned` и автоматически используется приложением, если присутствует на машине.

## Смешанное повторное обучение

Для улучшения качества на фотографиях слов выполнена еще одна эпоха обучения от checkpoint `trocr-cyrillic-finetuned`:

```text
Исходный Cyrillic Handwriting Dataset: 72 286 train-фрагментов
school_notebooks_RU: 268 350 train-слов
Локальная синтетика: 100 000 train-слов
Итого: 440 636 train-примеров
Validation: 5 000 школьных слов
Epochs: 1
Batch size: 24
Learning rate: 1e-5
Mixed precision: true
Время обучения: 159.24 мин
Train loss: 0.2121
Validation loss: 0.0737
```

Новая модель хранится локально в `models/trocr-cyrillic-mixed`. Приложение использует ее автоматически, если каталог присутствует.

На пользовательских фотографиях слов получены первые практические улучшения:

| Предыдущая модель | Mixed-модель |
| --- | --- |
| `солице` | `солнце` |
| `шетка` | `клетка` |

## Вторая эпоха смешанного обучения

Обучение продолжено от `models/trocr-cyrillic-mixed` на том же наборе данных. Модель сохранена в `models/trocr-cyrillic-mixed-epoch-2`.

```text
Train samples: 440 636
Validation: 5 000 школьных слов
Epochs: 1
Batch size: 24
Learning rate: 1e-5
Mixed precision: true
Время обучения: 101.10 мин
Train loss: 0.1739
Validation loss: 0.0682
```

По сравнению с первой смешанной эпохой validation loss снизился с `0.0737` до `0.0682`. На шести пользовательских фотографиях слов ухудшений не обнаружено; один из результатов изменился с `фанжевий` на `фанжевый`.

## Qwen-VL page experiment smoke

Для дальнейшего этапа подготовлен ограниченный page/line split из `schooldataset`:

```text
data/training/page_lines_experiment_smoke
test: 12 lines, 1 pages
```

Для этого split выполнена проверка OpenCV-сегментации строк:

```bash
python scripts/evaluate_segmentation.py \
  --lines-csv data/training/page_lines_experiment_smoke/test_lines.csv \
  --output data/evaluation/segmentation_smoke/results.csv \
  --processed-dir data/evaluation/segmentation_smoke/processed \
  --limit-pages 1
```

Результат smoke-оценки:

```text
pages: 1
reference lines: 12
predicted lines: 2
matched lines: 0
found line rate: 0.0000
average IoU: 0.0000
```

Этот результат показывает ограничение текущего OpenCV-baseline на реальной странице и подтверждает необходимость Qwen layout / нейросетевого детектора строк для сложных фотографий.

На его основе собран Qwen SFT JSONL:

```text
data/training/qwen_sft_experiment_smoke/mixed.jsonl
saved examples: 25
```

Dry-run обучающего скрипта Qwen LoRA успешно прочитал первые примеры без загрузки модели:

```bash
python scripts/train_qwen_vl_lora.py \
  --dataset data/training/qwen_sft_experiment_smoke/mixed.jsonl \
  --dry-run \
  --max-samples 4
```

Также проверен экспорт черновой Qwen layout-разметки в формат `lines.csv`:

```bash
python scripts/export_qwen_layout_annotations.py \
  --layout-csv data/evaluation/layout_export_smoke/qwen_layout.csv \
  --output-csv data/evaluation/layout_export_smoke/lines.csv
```

Результат:

```text
saved lines: 1
output: data/evaluation/layout_export_smoke/lines.csv
```

Проверен экспорт пользовательской обратной связи напрямую в Qwen SFT JSONL:

```bash
python scripts/export_feedback_training_data.py \
  --feedback-dir data/evaluation/feedback_export_smoke/problems \
  --output-csv data/evaluation/feedback_export_smoke/reviewed.csv \
  --output-qwen-jsonl data/evaluation/feedback_export_smoke/qwen_feedback.jsonl \
  --only-reviewed
```

Результат:

```text
saved examples: 1
qwen jsonl: data/evaluation/feedback_export_smoke/qwen_feedback.jsonl
```

Реальный OCR/inference-эксперимент TrOCR/Qwen пока не выполнен в текущем Windows `.venv`, потому что ML-зависимости отсутствуют:

```text
missing torch
missing transformers
missing sentencepiece
missing accelerate
missing qwen-vl-utils
missing peft
missing trl
```

Проверка окружения выполняется командой:

```bash
python scripts/check_ml_environment.py
```

## Qwen execution plan offline run

После добавления `scripts/build_qwen_execution_plan.py` и `scripts/run_qwen_execution_plan.py`
выполнены безопасные офлайн-шаги Qwen-плана без сетевой установки ML-зависимостей.

Сборка SFT JSONL:

```bash
python scripts/run_qwen_execution_plan.py --step build_sft --json
```

Результат:

```text
status: passed
saved examples: 3
output: data/training/qwen_sft_light/mixed.jsonl
```

После этого план был пересобран, и `lora_dry_run` начал использовать новый датасет:

```bash
python scripts/build_qwen_execution_plan.py --python python --json
python scripts/run_qwen_execution_plan.py --step lora_dry_run --json
```

Результат:

```text
status: passed
loaded examples: 3
dataset: data/training/qwen_sft_light/mixed.jsonl
model: Qwen/Qwen3-VL-2B-Instruct
```

Эти шаги подтверждают, что подготовка Qwen SFT-датасета и первичная валидация LoRA-обучения
выполняются на текущих локальных данных. Реальные `qwen_smoke`, `page_experiment` с Qwen и `lora_train`
по-прежнему зависят от установки ML-зависимостей и загрузки весов модели.
