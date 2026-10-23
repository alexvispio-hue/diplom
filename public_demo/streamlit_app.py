from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from public_demo.ocr import BASE_MODEL, FragmentRecognizer, crop_fragment, read_image

st.set_page_config(page_title="Handwriting OCR · Демо", page_icon="✍️", layout="centered")


@st.cache_resource(show_spinner=False)
def recognizer(model_source: str) -> FragmentRecognizer:
    return FragmentRecognizer(model_source)


st.caption("ДИПЛОМНЫЙ ПРОЕКТ · PYTHON / PYTORCH / TRANSFORMERS")
st.title("Рукописный текст → печатный")
st.write("Загрузите слово или небольшой фрагмент русскоязычного рукописного текста и попробуйте распознавание.")
st.warning(
    "**Распознавание целого листа текста в текущей версии не работает.** "
    "Используйте отдельные фрагменты: слова или небольшие участки текста. "
    "Если на фото целая страница, сначала выделите нужную область."
)
model_source = os.getenv("OCR_DEMO_MODEL", BASE_MODEL)
if model_source == BASE_MODEL:
    st.caption(
        "В онлайн-демо используется публичная базовая модель TrOCR для русского языка. "
        "Дообученные веса локальной версии сюда не перенесены; результат может отличаться."
    )
else:
    st.caption(f"Модель демо: {model_source}")

demo, about = st.tabs(["Попробовать", "О проекте"])
with about:
    st.subheader("Handwriting OCR")
    st.write(
        "Система распознавания русского рукописного текста. Полная локальная версия содержит "
        "FastAPI, Streamlit, предобработку изображений, историю в SQLite, экспорт документов "
        "и инструменты оценки качества CER/WER. Онлайн-демо показывает распознавание фрагментов."
    )
    st.markdown(
        "[Исходный код проекта](https://github.com/alexvispio-hue/diplom) · "
        "[Описание базовой модели](https://huggingface.co/kazars24/trocr-base-handwritten-ru)"
    )
    st.info("Обработка целых страниц находится в разработке. Результат распознавания может содержать ошибки.")

with demo:
    st.caption(
        "PNG, JPG или WEBP · до 5 МБ и 12 мегапикселей. Изображение обрабатывается на сервере демо; "
        "приложение не записывает загрузки и результаты в файлы или общую историю."
    )
    version = st.session_state.setdefault("upload_version", 0)
    uploaded = st.file_uploader("Выберите изображение", type=["png", "jpg", "jpeg", "webp"], key=f"upload_{version}")
    if uploaded is not None:
        payload = uploaded.getvalue()
        try:
            image = read_image(payload)
            signature = hashlib.sha256(payload).hexdigest()
            if st.session_state.get("source_signature") != signature:
                for key in ("result", "result_signature", "recognized_text", "crop_x", "crop_y"):
                    st.session_state.pop(key, None)
                st.session_state["source_signature"] = signature
            st.image(image, caption="Исходное изображение", use_container_width=True)
            with st.expander("Выделить фрагмент", expanded=False):
                horizontal = st.slider("Границы по горизонтали, %", 0, 100, (0, 100), key="crop_x")
                vertical = st.slider("Границы по вертикали, %", 0, 100, (0, 100), key="crop_y")
            fragment = crop_fragment(image, horizontal, vertical)
            current_signature = f"{signature}:{horizontal}:{vertical}"
            if st.session_state.get("result_signature") != current_signature:
                st.session_state.pop("result", None)
                st.session_state.pop("recognized_text", None)
            st.image(fragment, caption="Фрагмент для распознавания", use_container_width=True)
            st.caption("Выбирайте одну строку или слово. Первый запуск загружает модель и может занять несколько минут.")
            recognize_column, clear_column = st.columns([2, 1])
            start = recognize_column.button("Распознать фрагмент", type="primary", use_container_width=True)
            if clear_column.button("Очистить", use_container_width=True):
                for key in (f"upload_{version}", "source_signature", "result", "result_signature", "recognized_text", "crop_x", "crop_y"):
                    st.session_state.pop(key, None)
                st.session_state["upload_version"] = version + 1
                st.rerun()
            if start:
                st.session_state.pop("result", None)
                st.session_state.pop("recognized_text", None)
                try:
                    with st.spinner("Загружаем модель и распознаём фрагмент…"):
                        started = time.monotonic()
                        text = recognizer(model_source).recognize(fragment)
                        elapsed = time.monotonic() - started
                    st.session_state["result"] = (text, elapsed)
                    st.session_state["result_signature"] = current_signature
                except RuntimeError as exc:
                    if str(exc).startswith("Демо сейчас"):
                        st.info(str(exc))
                    else:
                        logging.getLogger(__name__).error("OCR runtime failed: %s", type(exc).__name__)
                        st.error("Не удалось выполнить распознавание. Попробуйте меньший фрагмент или повторите позже.")
                except Exception as exc:
                    logging.getLogger(__name__).error("OCR load/inference failed: %s", type(exc).__name__)
                    st.error("Модель сейчас недоступна. Повторите попытку позже.")
            if "result" in st.session_state:
                text, elapsed = st.session_state["result"]
                st.subheader("Результат")
                if text:
                    edited = st.text_area("Можно исправить текст перед скачиванием", value=text, height=140, key="recognized_text")
                    st.download_button("Скачать TXT", edited.encode("utf-8"), file_name="recognized.txt", mime="text/plain")
                else:
                    st.info("Текст не найден. Попробуйте другой фрагмент или более чёткое изображение.")
                st.caption(f"Время обработки, включая загрузку модели: {elapsed:.1f} с. Проверьте текст перед использованием.")
        except ValueError as exc:
            st.error(str(exc))
    else:
        for key in ("source_signature", "result", "result_signature", "recognized_text", "crop_x", "crop_y"):
            st.session_state.pop(key, None)
        st.info("Начните с чёткого фото одного слова или одной строки на светлом фоне.")

st.divider()
st.markdown("[GitHub · alexvispio-hue/diplom](https://github.com/alexvispio-hue/diplom)")
