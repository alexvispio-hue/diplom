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
