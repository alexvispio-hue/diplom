# Следующие шаги

Полный порядок работ находится в [product_roadmap.md](product_roadmap.md), обязательные требования и критерии приёмки находятся в [product_technical_specification.md](product_technical_specification.md).

Ближайшая последовательность:

1. Добавить CI и разделить статусы smoke, training, evaluation, candidate и production.
2. Добавить в Qwen trainer validation, лучший checkpoint, resume и run manifest.
3. Зафиксировать вручную проверенный gold benchmark и измерить текущие baseline.
4. Выполнить 100-step benchmark на RTX 5070 Ti и уточнить время полной эпохи.
5. Выполнить pilot, 10% run и только затем полное обучение Qwen3-VL-2B.
6. Принять или отклонить модель по CER/WER, latency, VRAM и regression-набору.
7. После принятия модели реализовать background jobs, overlay строк, autosave и searchable PDF.
8. Проверить installer и модельный профиль на чистом компьютере Windows.

Подготовка полного HWR200 уже завершена: 3000 документов, 11 010 страниц, 33 030 вариантов; validation report имеет статус `passed`. Следующий ML-шаг начинается не с повторной подготовки данных, а с доработки trainer и фиксации benchmark.
