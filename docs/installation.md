# Установка Handwriting OCR

Приложение работает локально: изображения, распознанный текст, история и пользовательские исправления остаются на компьютере пользователя. Внешний OCR API не используется.

## Профили

- `Lite` — TrOCR для CPU/GPU, минимальный размер установки.
- `Full` — TrOCR и Qwen runtime для распознавания страниц, строк и гибридного режима.
- `Developer` — полный ML-набор, инструменты тестирования и подготовки датасетов. Дообучение запускается только вручную отдельной командой.

Для текущего компьютера с RTX 5070 Ti базовый профиль для разработки страницы — `Full`, модель — `Qwen/Qwen3-VL-2B-Instruct`. Qwen остаётся экспериментальным до завершения полного обучения и gold quality gate.

## Windows

Требования: Windows 10/11 x64 и не менее 10 ГБ свободного места. Для `Lite` достаточно CPU; для Qwen рекомендуется NVIDIA GPU с CUDA. Интернет нужен при установке зависимостей и загрузке весов.

### ZIP-релиз

1. Распакуйте архив в папку пользователя.
2. Запустите `INSTALL_WINDOWS.bat` для профиля `Lite` или передайте профиль: `INSTALL_WINDOWS.bat Full`.
3. После установки запустите `START_APP.bat` или ярлык `Handwriting OCR`.

Скрипт использует установленный Python 3.10+; если Python не найден, он пробует установить Python 3.13 через `winget`. Изолированное окружение создаётся в `.venv-app` проекта.

Пользовательские файлы и модели по умолчанию хранятся отдельно от исходников:

```text
%LOCALAPPDATA%\HandwritingOCR\data
%LOCALAPPDATA%\HandwritingOCR\models
%LOCALAPPDATA%\HandwritingOCR\cache\huggingface
```

Это позволяет обновлять приложение и удалять его без потери документов и скачанных моделей. Для другого расположения используйте `-DataRoot`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1 -Profile Full -DataRoot D:\HandwritingOCR
```

Приложение открывается по адресам `http://127.0.0.1:8501`, API — `http://127.0.0.1:8000/docs`.

### Inno Setup

Файл `installer/handwriting-ocr.iss` задаёт установщик с профилями `Lite`, `Full` и `Developer`. Он не включает датасеты, checkpoint-файлы, историю пользователя или кэш Hugging Face. Пользовательские данные в `%LOCALAPPDATA%\HandwritingOCR` сохраняются при удалении приложения.

## Модели

Модели устанавливаются отдельно, чтобы релиз оставался компактным. Управлять ими можно на странице `Система` или из консоли:

```powershell
\.venv-app\Scripts\python.exe scripts\manage_models.py discover
\.venv-app\Scripts\python.exe scripts\manage_models.py list
\.venv-app\Scripts\python.exe scripts\manage_models.py verify trocr-cyrillic-mixed-epoch-2
```

Скачивание из Hugging Face выполняется только явной командой `install` или через интерфейс менеджера моделей. Qwen не становится production-моделью автоматически: для продвижения требуется отчёт quality gate.

Основные настройки находятся в `.env`:

```dotenv
MAX_UPLOAD_SIZE_MB=10
MAX_DOCUMENT_PAGES=100
MAX_IMAGE_PIXELS=50000000
PDF_RENDER_DPI=160
QWEN_MODEL_PRESET=light
```

После изменения `.env` перезапустите приложение.

## Docker

При установленном Docker Desktop:

```powershell
docker compose up --build
```

Docker-профиль предназначен для локального CPU/API-сценария; каталоги `data` и `models` монтируются отдельно, кэш Hugging Face сохраняется в volume.

## Сборка релиза

Компактный архив без датасетов, весов моделей, истории и временных файлов собирается из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1 -Version 0.4.0-dev
```

В `dist` появятся ZIP и файл его SHA-256. Архив можно прикладывать к GitHub Release. Перед публичной публикацией проверьте лицензии приложения и распространяемых весов моделей.

## Проверка установки

```powershell
\.venv-app\Scripts\python.exe scripts\system_diagnostics.py
\.venv-app\Scripts\python.exe scripts\audit_acceptance.py
```

Полный benchmark, ручная проверка gold-набора и настоящее Qwen-дообучение являются отдельными этапами. В рамках текущей подготовки код, датасеты, manifests, trainer, интерфейс и упаковка подготовлены, но команда обучения не запускалась.
