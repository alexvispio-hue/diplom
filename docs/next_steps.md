# Следующие шаги

Инженерная часть проекта подготовлена без запуска нового Qwen-дообучения. Текущее состояние фиксируется так:

## Уже сделано

- подготовлен и проверен полный HWR200 page-level dataset;
- добавлены dataset/run manifests, review policy и group-aware sampler;
- trainer получил validation, best-checkpoint, resume, NaN/OOM health checks и generation evaluation;
- добавлены benchmark runner, quality gate, gold-review workflow и regression subset;
- реализованы фоновые OCR jobs, pause/resume/cancel/retry и восстановление после перезапуска;
- реализованы импорт папки, подготовка изображения, overlay строк, редактор документа, autosave, revisions и searchable PDF;
- добавлены model registry, checksum verification, promotion/rollback, model downloads и диагностика;
- добавлены Lite/Full/Developer installation profiles, Inno Setup и reproducible release ZIP;
- CI проверяет ruff, compileall, pytest, dependency audit и release archive.

## Что осталось до запуска обучения

1. В интерфейсе `Gold benchmark` вручную проверить 200 кандидатных страниц и финализировать immutable gold manifest.
2. Запустить baseline suite на финальном gold-наборе для TrOCR, Qwen base и hybrid. Это inference/evaluation, не обучение.
3. Выполнить короткий 100-step performance run, чтобы уточнить скорость, VRAM и размер окна изображения.
4. После отдельного подтверждения пользователя запустить pilot, 10% run и затем полный Qwen3-VL-2B training.
5. Выполнить обязательную evaluation, сравнить CER/WER, exact-page rate, line metrics, latency, VRAM и regression subset.
6. Продвинуть адаптер в `candidate`, затем в `production` только после quality gate.

## Ограничения текущего состояния

- полноценное обучение полного HWR200 в этой сессии не запускалось;
- Qwen-адаптеры в `models` считаются `experimental`, Qwen не назначен production;
- gold benchmark пока не имеет требуемого ручного подтверждения;
- полный pytest и чистая установка на отдельном компьютере должны быть подтверждены в CI/на тестовой машине, потому что текущая среда блокирует доступ к части `%TEMP%`.

Подробные требования: [product_technical_specification.md](product_technical_specification.md). Порядок релизной работы: [product_roadmap.md](product_roadmap.md).
