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
