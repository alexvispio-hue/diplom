from __future__ import annotations

import html
import os
from dataclasses import replace
from datetime import datetime
from typing import Any

import streamlit as st

from app.core.config import get_settings
from app.services.document_service import (
    DocumentImportError,
    ExportPage,
    PreparedPage,
    build_export_pages,
    expand_document,
    export_docx,
    export_json,
    export_text,
)
from ui.api_client import ApiError, OCRApiClient
from ui.theme import apply_theme


API_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:8000/api")
PAGE_MODE_LABELS = {
    "trocr_lines": "TrOCR по строкам",
    "qwen_page": "Qwen: вся страница",
    "qwen_lines": "Qwen по строкам",
    "hybrid": "Гибридный режим",
    "compare": "Сравнение моделей",
    "qwen_layout": "Qwen: текст и разметка",
}
MODE_LABELS = {
    "fragment": "Слово / фрагмент",
    **PAGE_MODE_LABELS,
}
FEEDBACK_LABELS = {
    "exact": "Точно",
    "partial": "Нужна правка",
    "mismatch": "Неверно",
}
NAVIGATION_LABELS = {
    "workspace": "Рабочая область",
    "history": "История",
    "review": "Проверка",
    "system": "Система",
}
QWEN_MODES = {"qwen_page", "qwen_lines", "hybrid", "compare", "qwen_layout"}


st.set_page_config(
    page_title="Handwriting OCR",
    page_icon="OCR",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

settings = get_settings()
api = OCRApiClient(API_URL)


def api_optional(path: str, **params: Any) -> Any | None:
    try:
        return api.get(path, **params)
    except (ApiError, OSError):
        return None


def init_state() -> None:
    defaults = {
        "workspace_results": [],
        "workspace_document_text": "",
        "workspace_errors": [],
        "workspace_message": "",
        "workspace_page_index": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown('<div class="app-kicker">Handwriting OCR</div>', unsafe_allow_html=True)
    st.markdown(f"<h1>{html.escape(title)}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<p class="app-subtitle">{html.escape(subtitle)}</p>',
            unsafe_allow_html=True,
        )


def status_row(label: str, state: str) -> None:
    css_class = {"ok": "status-ok", "warn": "status-warn"}.get(state, "status-off")
    st.markdown(
        f'<div class="status-row"><span class="status-dot {css_class}"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(health: dict | None, ml_status: dict | None) -> str:
    with st.sidebar:
        st.markdown("### HANDWRITING OCR")
        st.caption("Локальное приложение")
        st.divider()
        section = st.radio(
            "Навигация",
            options=list(NAVIGATION_LABELS),
            format_func=NAVIGATION_LABELS.get,
            label_visibility="collapsed",
        )
        st.divider()
        status_row("Сервер подключён" if health else "Сервер недоступен", "ok" if health else "off")
        if ml_status:
            status_row(
                "TrOCR готов" if ml_status.get("ready_for_trocr") else "TrOCR не готов",
                "ok" if ml_status.get("ready_for_trocr") else "warn",
            )
            status_row(
                "Qwen готов" if ml_status.get("ready_for_qwen_inference") else "Qwen не готов",
                "ok" if ml_status.get("ready_for_qwen_inference") else "warn",
            )
        st.caption(f"Версия {health.get('version', '0.1.0') if health else '0.1.0'}")
        return section


def prepare_uploaded_pages(uploaded_files: list[Any]) -> list[PreparedPage]:
    pages: list[PreparedPage] = []
    for uploaded in uploaded_files:
        payload = uploaded.getvalue()
        if len(payload) > settings.max_upload_size_mb * 10 * 1024 * 1024:
            raise DocumentImportError(
                f"Файл {uploaded.name} превышает лимит документа "
                f"{settings.max_upload_size_mb * 10} МБ."
            )
        pages.extend(
            expand_document(
                uploaded.name,
                payload,
                uploaded.type,
                pdf_render_dpi=settings.pdf_render_dpi,
                max_pages=settings.max_document_pages,
            )
        )
    if len(pages) > settings.max_document_pages:
        raise DocumentImportError(
            f"В очереди {len(pages)} страниц; разрешено не более {settings.max_document_pages}."
        )
    return pages


def current_page_edits(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        widget_key = f"workspace_page_text_{result['id']}"
        if widget_key in st.session_state:
            result["edited_text"] = st.session_state[widget_key]
    return results


def rebuild_document_text(results: list[dict[str, Any]]) -> str:
    pages = build_export_pages(current_page_edits(results))
    return export_text(pages, page_headers=False)


def document_export_pages(results: list[dict[str, Any]], document_text: str) -> list[ExportPage]:
    pages = build_export_pages(current_page_edits(results))
    chunks = [chunk.strip() for chunk in document_text.split("\f")]
    if len(chunks) == len(pages):
        return [replace(page, text=chunks[index]) for index, page in enumerate(pages)]
    return [ExportPage(page_number=1, filename="document.txt", text=document_text)]


def render_result_meta(result: dict[str, Any]) -> None:
    confidence = result.get("confidence")
    confidence_label = "нет оценки" if confidence is None else f"{confidence * 100:.1f}%"
    st.markdown(
        "<div class=\"result-meta\">"
        f"{MODE_LABELS.get(result.get('recognition_mode'), result.get('recognition_mode', ''))} · "
        f"{result.get('model_name', 'модель не указана')} · "
        f"{result.get('processing_time_ms', 0) / 1000:.1f} с · "
        f"уверенность {confidence_label}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_artifacts(result: dict[str, Any]) -> None:
    processed_url = api.asset_url(result.get("processed_image_url"))
    line_urls = result.get("line_image_urls") or []
    coordinates = result.get("line_coordinates") or []
    notes = result.get("review_notes") or []
    if processed_url:
        st.image(processed_url, caption=f"Найдено строк: {result.get('line_count', 0)}", use_container_width=True)
    if line_urls:
        with st.expander(f"Выделенные строки: {len(line_urls)}"):
            columns = st.columns(3)
            for index, path in enumerate(line_urls):
                with columns[index % 3]:
                    st.image(api.asset_url(path), caption=f"Строка {index + 1}", use_container_width=True)
    if coordinates:
        with st.expander("Координаты строк"):
            st.dataframe(coordinates, hide_index=True, use_container_width=True)
    if notes:
        with st.expander(f"Требует проверки: {len(notes)}", expanded=True):
            st.dataframe(notes, hide_index=True, use_container_width=True)


def render_feedback(result: dict[str, Any], corrected_text: str, key_prefix: str) -> None:
    default_rating = result.get("feedback_rating") or ("partial" if result.get("needs_review") else "exact")
    rating = st.segmented_control(
        "Оценка результата",
        options=list(FEEDBACK_LABELS),
        format_func=FEEDBACK_LABELS.get,
        default=default_rating,
        key=f"{key_prefix}_rating_{result['id']}",
    )
    if st.button(
        "Сохранить оценку",
        key=f"{key_prefix}_feedback_{result['id']}",
        use_container_width=True,
    ):
        try:
            saved = api.post_json(
                f"/history/{result['id']}/feedback",
                {"rating": rating or default_rating, "corrected_text": corrected_text},
            )
            result.update(saved)
            st.success("Оценка и исправленный текст сохранены.")
        except ApiError as exc:
            st.error(str(exc))


def render_queue(pages: list[PreparedPage]) -> None:
    if not pages:
        return
    with st.expander(f"Очередь: {len(pages)} страниц", expanded=False):
        rows = [
            {
                "№": index,
                "Страница": page.filename,
                "Источник": page.source_name,
                "Размер, КБ": round(len(page.data) / 1024, 1),
            }
            for index, page in enumerate(pages, start=1)
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)


def recognize_queue(
    pages: list[PreparedPage],
    *,
    document_target: str,
    page_mode: str,
    preprocess: bool,
    max_tokens: int,
    model_override: str,
    save_intermediate_images: bool,
    return_coordinates: bool,
) -> None:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    progress = st.progress(0, text="Подготовка очереди")
    for index, page in enumerate(pages, start=1):
        progress.progress((index - 1) / len(pages), text=f"Страница {index} из {len(pages)}")
        try:
            result = api.recognize(
                filename=page.filename,
                content_type=page.content_type,
                data=page.data,
                document_target=document_target,
                page_mode=page_mode,
                preprocess=preprocess,
                max_tokens=max_tokens,
                model_override=model_override,
                save_intermediate_images=save_intermediate_images,
                return_coordinates=return_coordinates,
            )
            result.update(
                {
                    "document_page_number": index,
                    "source_name": page.source_name,
                    "source_data": page.data,
                    "source_content_type": page.content_type,
                    "edited_text": result.get("corrected_text") or result.get("recognized_text", ""),
                }
            )
            results.append(result)
        except ApiError as exc:
            errors.append(f"{page.filename}: {exc}")
        st.session_state["workspace_results"] = results
    progress.progress(1.0, text="Обработка завершена")
    st.session_state["workspace_errors"] = errors
    st.session_state["workspace_document_text"] = rebuild_document_text(results)
    st.session_state["workspace_page_index"] = 0
    st.session_state["workspace_message"] = (
        f"Обработано страниц: {len(results)}" if results else "Распознавание не выполнено."
    )


def render_document_result(results: list[dict[str, Any]]) -> None:
    results = current_page_edits(results)
    total_time = sum(result.get("processing_time_ms", 0) for result in results)
    review_count = sum(bool(result.get("needs_review")) for result in results)
    metric_pages, metric_time, metric_review = st.columns(3)
    metric_pages.metric("Страниц", len(results))
    metric_time.metric("Время", f"{total_time / 1000:.1f} с")
    metric_review.metric("Требуют проверки", review_count)

    text_tab, page_tab = st.tabs(["Документ", "Страницы"])
    with text_tab:
        document_title = st.text_input("Название", value="Распознанный документ", key="workspace_title")
        document_text = st.text_area(
            "Текст документа",
            key="workspace_document_text",
            height=430,
        )
        export_pages = document_export_pages(results, document_text)
        download_txt, download_docx, download_json, rebuild = st.columns([1, 1, 1, 1])
        with download_txt:
            st.download_button(
                "TXT",
                data=document_text.encode("utf-8"),
                file_name="recognized_document.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with download_docx:
            st.download_button(
                "DOCX",
                data=export_docx(export_pages, title=document_title),
                file_name="recognized_document.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        with download_json:
            st.download_button(
                "JSON",
                data=export_json(export_pages, title=document_title),
                file_name="recognized_document.json",
                mime="application/json",
                use_container_width=True,
            )
        with rebuild:
            if st.button("Собрать из страниц", use_container_width=True):
                st.session_state["workspace_document_text"] = rebuild_document_text(results)
                st.rerun()

    with page_tab:
        options = list(range(len(results)))
        selected_index = st.selectbox(
            "Страница",
            options=options,
            index=min(st.session_state.get("workspace_page_index", 0), len(options) - 1),
            format_func=lambda index: f"{index + 1}. {results[index]['filename']}",
        )
        st.session_state["workspace_page_index"] = selected_index
        result = results[selected_index]
        image_column, text_column = st.columns([0.9, 1.1], gap="large")
        with image_column:
            st.image(result["source_data"], caption=result["filename"], use_container_width=True)
            render_artifacts(result)
        with text_column:
            page_text = st.text_area(
                "Текст страницы",
                value=result.get("edited_text") or result.get("recognized_text", ""),
                height=360,
                key=f"workspace_page_text_{result['id']}",
            )
            result["edited_text"] = page_text
            render_result_meta(result)
            render_feedback(result, page_text, "workspace")


def render_workspace(health: dict | None, ml_status: dict | None) -> None:
    page_header("Рабочая область", "Распознавание изображений и многостраничных документов")
    uploaded_files = st.file_uploader(
        "Документы",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    pages: list[PreparedPage] = []
    if uploaded_files:
        try:
            pages = prepare_uploaded_pages(uploaded_files)
        except DocumentImportError as exc:
            st.error(str(exc))
    render_queue(pages)

    target_column, mode_column = st.columns([0.9, 1.4])
    with target_column:
        document_target = st.segmented_control(
            "Тип",
            options=["fragment", "page"],
            format_func=lambda value: "Фрагмент" if value == "fragment" else "Страница",
            default="page",
        )
    with mode_column:
        page_mode = st.selectbox(
            "Режим",
            options=list(PAGE_MODE_LABELS),
            format_func=PAGE_MODE_LABELS.get,
            disabled=document_target != "page",
        )
    preprocess = st.toggle("Предобработка изображения", value=True)

    max_tokens = 1024
    model_override = ""
    save_intermediate_images = True
    return_coordinates = True
    with st.expander("Дополнительные параметры"):
        advanced_one, advanced_two = st.columns(2)
        with advanced_one:
            save_intermediate_images = st.toggle("Сохранять найденные строки", value=True)
            return_coordinates = st.toggle("Сохранять координаты", value=True)
        with advanced_two:
            max_tokens = st.number_input("Лимит токенов Qwen", 128, 8192, 1024, 128)
            model_override = st.text_input("Qwen model id", placeholder="Локальный путь или Hugging Face id").strip()

    qwen_selected = document_target == "page" and page_mode in QWEN_MODES
    qwen_ready = bool(ml_status and ml_status.get("ready_for_qwen_inference"))
    if qwen_selected and not qwen_ready:
        st.warning("Qwen-режим пока недоступен. Состояние окружения показано в разделе «Система».")
    if st.button(
        "Распознать очередь",
        type="primary",
        use_container_width=True,
        disabled=not pages or not health or (qwen_selected and not qwen_ready),
    ):
        recognize_queue(
            pages,
            document_target=document_target or "page",
            page_mode=page_mode,
            preprocess=preprocess,
            max_tokens=int(max_tokens),
            model_override=model_override,
            save_intermediate_images=save_intermediate_images,
            return_coordinates=return_coordinates,
        )
        st.rerun()

    message = st.session_state.pop("workspace_message", "")
    if message:
        st.success(message)
    for error in st.session_state.get("workspace_errors", []):
        st.error(error)
    results = st.session_state.get("workspace_results", [])
    if results:
        st.divider()
        render_document_result(results)
    elif not uploaded_files:
        st.info("Добавьте изображения или PDF в рабочую область.")


def history_matches(item: dict[str, Any], query: str, mode: str, status: str) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("original_filename", "recognized_text", "corrected_text", "model_name")
    ).lower()
    if query and query.lower() not in haystack:
        return False
    if mode != "all" and item.get("recognition_mode") != mode:
        return False
    if status == "review" and not item.get("needs_review"):
        return False
    if status == "rated" and not item.get("feedback_rating"):
        return False
    if status == "unrated" and item.get("feedback_rating"):
        return False
    return True


def render_history() -> None:
    page_header("История", "Сохранённые результаты и пользовательские исправления")
    history = api_optional("/history", limit=200)
    if history is None:
        st.error("Не удалось получить историю с локального сервера.")
        return
    if not history:
        st.info("История распознавания пока пуста.")
        return

    search_column, mode_column, status_column = st.columns([1.5, 1, 1])
    with search_column:
        query = st.text_input("Поиск", placeholder="Имя файла, текст или модель")
    with mode_column:
        modes = ["all"] + sorted({item.get("recognition_mode", "fragment") for item in history})
        mode = st.selectbox(
            "Режим",
            modes,
            format_func=lambda value: "Все режимы" if value == "all" else MODE_LABELS.get(value, value),
        )
    with status_column:
        status = st.selectbox(
            "Статус",
            ["all", "review", "rated", "unrated"],
            format_func={
                "all": "Все",
                "review": "На проверке",
                "rated": "Оценённые",
                "unrated": "Без оценки",
            }.get,
        )
    filtered = [item for item in history if history_matches(item, query, mode, status)]
    total_metric, review_metric, rated_metric = st.columns(3)
    total_metric.metric("Найдено", len(filtered))
    review_metric.metric("На проверке", sum(bool(item.get("needs_review")) for item in filtered))
    rated_metric.metric("С оценкой", sum(bool(item.get("feedback_rating")) for item in filtered))

    for item in filtered:
        created = format_datetime(item.get("created_at"))
        marker = " · требуется проверка" if item.get("needs_review") else ""
        with st.expander(f"{item['original_filename']} · {created}{marker}"):
            image_column, result_column = st.columns([0.75, 1.25], gap="large")
            with image_column:
                st.image(api.asset_url(item.get("original_image_url")), use_container_width=True)
                processed_url = api.asset_url(item.get("processed_image_url"))
                if processed_url and processed_url != api.asset_url(item.get("original_image_url")):
                    st.image(processed_url, caption="После обработки", use_container_width=True)
            with result_column:
                text = st.text_area(
                    "Текст",
                    value=item.get("corrected_text") or item.get("recognized_text", ""),
                    height=220,
                    key=f"history_text_{item['id']}",
                )
                render_result_meta(item)
                actions = st.columns(2)
                with actions[0]:
                    st.download_button(
                        "TXT",
                        data=text.encode("utf-8"),
                        file_name=f"recognition_{item['id']}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key=f"history_download_{item['id']}",
                    )
                with actions[1]:
                    render_feedback(item, text, "history")
            render_artifacts(item)


def render_review() -> None:
    page_header("Проверка", "Очередь спорных и исправленных примеров")
    problems = api_optional("/feedback/problems")
    if problems is None:
        st.error("Не удалось получить очередь проверки.")
        return
    if not problems:
        st.success("Очередь проверки пуста.")
        return

    query = st.text_input("Поиск", placeholder="Имя файла или распознанный текст")
    visible = [
        item
        for item in problems
        if not query
        or query.lower()
        in f"{item.get('original_filename', '')} {item.get('recognized_text', '')}".lower()
    ]
    st.metric("В очереди", len(visible))
    for item in visible:
        title = f"#{item['recognition_id']} · {item['original_filename']} · {FEEDBACK_LABELS.get(item['rating'], item['rating'])}"
        with st.expander(title):
            image_column, text_column = st.columns([0.85, 1.15], gap="large")
            with image_column:
                st.image(api.asset_url(item.get("image_url")), use_container_width=True)
                if item.get("processed_image_url"):
                    st.image(api.asset_url(item["processed_image_url"]), caption="После обработки", use_container_width=True)
            with text_column:
                st.text_area(
                    "Ответ модели",
                    value=item.get("recognized_text", ""),
                    height=150,
                    disabled=True,
                    key=f"review_original_{item['recognition_id']}",
                )
                corrected = st.text_area(
                    "Проверенный текст",
                    value=item.get("corrected_text") or "",
                    height=190,
                    key=f"review_corrected_{item['recognition_id']}",
                )
                if st.button(
                    "Сохранить проверку",
                    type="primary",
                    use_container_width=True,
                    key=f"review_save_{item['recognition_id']}",
                ):
                    try:
                        api.post_json(
                            "/feedback",
                            {
                                "recognition_id": item["recognition_id"],
                                "rating": item["rating"],
                                "corrected_text": corrected,
                            },
                        )
                        st.success("Проверенный текст сохранён.")
                    except ApiError as exc:
                        st.error(str(exc))
            if item.get("review_notes"):
                st.dataframe(item["review_notes"], hide_index=True, use_container_width=True)


def render_system(health: dict | None, ml_status: dict | None) -> None:
    page_header("Система", "Локальные модели, окружение и техническое состояние")
    models = api_optional("/models") or []
    qwen_status = api_optional("/qwen-status")
    packages = (ml_status or {}).get("packages", [])

    server_metric, trocr_metric, qwen_metric, cuda_metric = st.columns(4)
    server_metric.metric("Сервер", "Готов" if health else "Недоступен")
    trocr_metric.metric("TrOCR", "Готов" if ml_status and ml_status.get("ready_for_trocr") else "Не готов")
    qwen_metric.metric("Qwen", "Готов" if ml_status and ml_status.get("ready_for_qwen_inference") else "Не готов")
    cuda_value = "Нет данных"
    if ml_status and ml_status.get("torch_cuda_available") is not None:
        cuda_value = f"{ml_status.get('torch_device_count', 0)} GPU" if ml_status["torch_cuda_available"] else "CPU"
    cuda_metric.metric("Вычисления", cuda_value)

    model_tab, environment_tab, diagnostics_tab = st.tabs(["Модели", "Окружение", "Диагностика"])
    with model_tab:
        if models:
            st.dataframe(
                [
                    {
                        "Backend": item["backend"],
                        "Модель": item["model_name"],
                        "Страницы": item.get("supports_page", False),
                        "Обучение": item.get("supports_training", False),
                        "Локально": item.get("local_inference", False),
                    }
                    for item in models
                ],
                hide_index=True,
                use_container_width=True,
            )
        if qwen_status:
            st.caption(f"Qwen runtime: {qwen_status.get('runtime_status', 'unknown')}")
            st.caption(f"SFT-примеров: {(qwen_status.get('sft_dataset') or {}).get('examples', 0)}")
            missing = qwen_status.get("missing_packages") or []
            if missing:
                st.warning("Не установлены: " + ", ".join(missing))
    with environment_tab:
        if packages:
            st.dataframe(
                [
                    {
                        "Пакет": item["package"],
                        "Установлен": item["installed"],
                        "Версия": item.get("version") or "",
                    }
                    for item in packages
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Статус окружения недоступен.")
    with diagnostics_tab:
        summaries = api_optional("/experiments/page-summary") or []
        audit = api_optional("/acceptance-audit") or []
        if summaries:
            st.subheader("Эксперименты")
            st.dataframe(summaries, hide_index=True, use_container_width=True)
        if audit:
            st.subheader("Техническая проверка")
            st.dataframe(audit, hide_index=True, use_container_width=True)
        if not summaries and not audit:
            st.info("Диагностические отчёты пока не созданы.")


def format_datetime(value: str | None) -> str:
    if not value:
        return "дата не указана"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return value


init_state()
health_status = api_optional("/health")
ml_environment = api_optional("/ml-status", timeout=15)
selected_section = render_sidebar(health_status, ml_environment)

if selected_section == "workspace":
    render_workspace(health_status, ml_environment)
elif selected_section == "history":
    render_history()
elif selected_section == "review":
    render_review()
else:
    render_system(health_status, ml_environment)
