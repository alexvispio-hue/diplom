from __future__ import annotations

import html
import io
import json
import os
from dataclasses import replace
from datetime import date, datetime
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.image_processing.quality_analysis import ImageQualityAnalyzer
from app.services.document_service import (
    DocumentImportError,
    PreparedPage,
    expand_document,
    transform_image,
)
from ui.api_client import ApiError, OCRApiClient
from ui.theme import apply_theme


API_URL = os.getenv("OCR_API_URL", "http://127.0.0.1:8000/api")
PAGE_MODE_LABELS = {
    "auto": "Автоматически",
    "trocr_lines": "TrOCR по строкам",
    "qwen_page": "Qwen: вся страница",
    "qwen_lines": "Qwen по строкам",
    "hybrid": "Гибридный режим",
    "compare": "Сравнение моделей",
    "qwen_layout": "Qwen: текст и разметка",
}
MODE_LABELS = {"fragment": "Слово / фрагмент", **PAGE_MODE_LABELS}
FEEDBACK_LABELS = {"exact": "Точно", "partial": "Нужна правка", "mismatch": "Неверно"}
JOB_STATUS_LABELS = {
    "queued": "В очереди",
    "running": "Выполняется",
    "paused": "Приостановлено",
    "completed": "Завершено",
    "failed": "Ошибка",
    "cancelled": "Отменено",
}
NAVIGATION_LABELS = {
    "workspace": "Импорт",
    "jobs": "Очередь",
    "documents": "Документы",
    "history": "История",
    "review": "Проверка",
    "system": "Система",
}
QWEN_MODES = {"qwen_page", "qwen_lines", "hybrid", "compare", "qwen_layout"}
QUALITY_ISSUES = {
    "blur": "размытие",
    "underexposed": "темное изображение",
    "overexposed": "пересвет",
    "low_contrast": "низкий контраст",
    "low_resolution": "низкое разрешение",
}


st.set_page_config(
    page_title="Handwriting OCR",
    page_icon="OCR",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()

settings = get_settings()
api = OCRApiClient(API_URL)
quality_analyzer = ImageQualityAnalyzer()


def api_optional(path: str, **params: Any) -> Any | None:
    try:
        return api.get(path, **params)
    except (ApiError, OSError):
        return None


def init_state() -> None:
    defaults = {
        "navigation": "workspace",
        "workspace_pages": [],
        "workspace_original_pages": [],
        "workspace_signature": None,
        "workspace_page_index": 0,
        "active_job_id": None,
        "selected_draft_id": None,
        "document_notice": None,
        "redo_revision": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def page_header(title: str, detail: str | None = None) -> None:
    st.markdown('<div class="app-kicker">Handwriting OCR</div>', unsafe_allow_html=True)
    st.markdown(f"<h1>{html.escape(title)}</h1>", unsafe_allow_html=True)
    if detail:
        st.markdown(f'<p class="app-subtitle">{html.escape(detail)}</p>', unsafe_allow_html=True)


def status_row(label: str, state: str) -> None:
    css_class = {"ok": "status-ok", "warn": "status-warn"}.get(state, "status-off")
    st.markdown(
        f'<div class="status-row"><span class="status-dot {css_class}"></span>{html.escape(label)}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(health: dict | None, ml_status: dict | None) -> str:
    with st.sidebar:
        st.markdown("### HANDWRITING OCR")
        st.caption("Локальный профиль")
        st.divider()
        section = st.radio(
            "Навигация",
            options=list(NAVIGATION_LABELS),
            format_func=NAVIGATION_LABELS.get,
            key="navigation",
            label_visibility="collapsed",
        )
        st.divider()
        status_row("API подключен" if health else "API недоступен", "ok" if health else "off")
        if ml_status:
            status_row(
                "TrOCR готов" if ml_status.get("ready_for_trocr") else "TrOCR не готов",
                "ok" if ml_status.get("ready_for_trocr") else "warn",
            )
            status_row(
                "Qwen готов" if ml_status.get("ready_for_qwen_inference") else "Qwen не готов",
                "ok" if ml_status.get("ready_for_qwen_inference") else "warn",
            )
        st.caption(f"Версия {(health or {}).get('version', settings.app_version)}")
        return section


def _upload_signature(uploaded_files: list[Any]) -> tuple[tuple[str, int], ...]:
    return tuple((uploaded.name, len(uploaded.getvalue())) for uploaded in uploaded_files)


def prepare_uploaded_pages(uploaded_files: list[Any]) -> list[PreparedPage]:
    pages: list[PreparedPage] = []
    for uploaded in uploaded_files:
        payload = uploaded.getvalue()
        pages.extend(
            expand_document(
                uploaded.name,
                payload,
                uploaded.type,
                pdf_render_dpi=settings.pdf_render_dpi,
                max_pages=settings.max_document_pages - len(pages),
                max_file_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
                max_image_pixels=settings.max_image_pixels,
            )
        )
    if len(pages) > settings.max_document_pages:
        raise DocumentImportError(
            f"В очереди {len(pages)} страниц; разрешено не более {settings.max_document_pages}."
        )
    return _renumber_pages(pages)


def _renumber_pages(pages: list[PreparedPage]) -> list[PreparedPage]:
    return [replace(page, page_number=index) for index, page in enumerate(pages, start=1)]


def sync_uploaded_pages(uploaded_files: list[Any]) -> None:
    signature = _upload_signature(uploaded_files)
    if signature == st.session_state["workspace_signature"]:
        return
    pages = prepare_uploaded_pages(uploaded_files)
    st.session_state["workspace_signature"] = signature
    st.session_state["workspace_pages"] = pages
    st.session_state["workspace_original_pages"] = list(pages)
    st.session_state["workspace_page_index"] = 0


def _move_workspace_page(index: int, offset: int) -> None:
    target = index + offset
    pages = list(st.session_state["workspace_pages"])
    originals = list(st.session_state["workspace_original_pages"])
    if target < 0 or target >= len(pages):
        return
    pages[index], pages[target] = pages[target], pages[index]
    originals[index], originals[target] = originals[target], originals[index]
    st.session_state["workspace_pages"] = _renumber_pages(pages)
    st.session_state["workspace_original_pages"] = _renumber_pages(originals)
    st.session_state["workspace_page_index"] = target


def render_page_preparation() -> None:
    pages: list[PreparedPage] = st.session_state["workspace_pages"]
    if not pages:
        return
    selected = st.selectbox(
        "Страница",
        options=list(range(len(pages))),
        index=min(st.session_state["workspace_page_index"], len(pages) - 1),
        format_func=lambda index: f"{index + 1}. {pages[index].filename}",
        key="workspace_prepare_index",
    )
    st.session_state["workspace_page_index"] = selected
    actions = st.columns([1, 1, 1, 5])
    if actions[0].button(
        "Вверх",
        icon=":material/arrow_upward:",
        disabled=selected == 0,
        use_container_width=True,
    ):
        _move_workspace_page(selected, -1)
        st.rerun()
    if actions[1].button(
        "Вниз",
        icon=":material/arrow_downward:",
        disabled=selected == len(pages) - 1,
        use_container_width=True,
    ):
        _move_workspace_page(selected, 1)
        st.rerun()
    if actions[2].button("Сбросить", icon=":material/restart_alt:", use_container_width=True):
        pages[selected] = st.session_state["workspace_original_pages"][selected]
        st.session_state["workspace_pages"] = pages
        st.rerun()

    current = pages[selected]
    original = st.session_state["workspace_original_pages"][selected]
    preview_one, preview_two = st.columns(2, gap="large")
    with preview_one:
        st.caption("Исходник")
        st.image(original.data, use_container_width=True)
    with preview_two:
        st.caption("Подготовленная страница")
        st.image(current.data, use_container_width=True)

    quality = quality_analyzer.analyze_bytes(current.data)
    metric_quality, metric_size, metric_issues = st.columns(3)
    metric_quality.metric("Качество", f"{quality.score * 100:.0f}%")
    metric_size.metric("Размер", f"{quality.width} x {quality.height}")
    issue_label = ", ".join(QUALITY_ISSUES.get(item, item) for item in quality.issues) or "нет"
    metric_issues.metric("Замечания", len(quality.issues))
    metric_issues.caption(issue_label)

    with st.expander("Подготовка изображения", expanded=bool(quality.issues)):
        rotation = st.segmented_control(
            "Поворот",
            options=[0, 90, 180, 270],
            format_func=lambda value: f"{value}°",
            default=0,
            key=f"rotation_{selected}",
        )
        crop_columns = st.columns(4)
        crop_left = crop_columns[0].number_input("Слева, %", 0, 45, 0, key=f"crop_l_{selected}")
        crop_top = crop_columns[1].number_input("Сверху, %", 0, 45, 0, key=f"crop_t_{selected}")
        crop_right = crop_columns[2].number_input("Справа, %", 0, 45, 0, key=f"crop_r_{selected}")
        crop_bottom = crop_columns[3].number_input("Снизу, %", 0, 45, 0, key=f"crop_b_{selected}")
        preset_columns = st.columns(5)
        perspective = preset_columns[0].toggle("Лист", value=False, key=f"perspective_{selected}")
        deskew = preset_columns[1].toggle("Deskew", value=False, key=f"deskew_{selected}")
        contrast = preset_columns[2].toggle(
            "Контраст", value="low_contrast" in quality.issues, key=f"contrast_{selected}"
        )
        denoise = preset_columns[3].toggle("Шум", value=False, key=f"denoise_{selected}")
        remove_grid = preset_columns[4].toggle("Убрать сетку", value=False, key=f"grid_{selected}")
        if st.button(
            "Применить к странице",
            type="primary",
            icon=":material/tune:",
            use_container_width=True,
        ):
            with Image.open(io.BytesIO(current.data)) as image:
                width, height = image.size
            if rotation in {90, 270}:
                width, height = height, width
            crop_box = (
                round(width * crop_left / 100),
                round(height * crop_top / 100),
                round(width * (1 - crop_right / 100)),
                round(height * (1 - crop_bottom / 100)),
            )
            transformed = transform_image(
                current.data,
                rotation_degrees=int(rotation or 0),
                crop_box=crop_box,
                correct_perspective=perspective,
                deskew=deskew,
                enhance_contrast=contrast,
                denoise=denoise,
                remove_grid=remove_grid,
            )
            pages[selected] = replace(
                current,
                filename=str(current.filename.rsplit(".", 1)[0]) + ".png",
                content_type="image/png",
                data=transformed,
            )
            st.session_state["workspace_pages"] = pages
            st.rerun()


def render_workspace(health: dict | None, ml_status: dict | None) -> None:
    page_header("Импорт")
    upload_tab, folder_tab = st.tabs(["Файлы", "Локальная папка"])
    with upload_tab:
        uploaded_files = st.file_uploader(
            "Документы",
            type=["png", "jpg", "jpeg", "tif", "tiff", "webp", "pdf"],
            accept_multiple_files=True,
            max_upload_size=settings.max_upload_size_mb,
            label_visibility="collapsed",
        )
        if uploaded_files:
            try:
                sync_uploaded_pages(uploaded_files)
            except DocumentImportError as exc:
                st.error(str(exc))
        pages: list[PreparedPage] = st.session_state["workspace_pages"]
        if pages:
            if st.button("Очистить список", icon=":material/clear_all:"):
                st.session_state["workspace_pages"] = []
                st.session_state["workspace_original_pages"] = []
                st.session_state["workspace_signature"] = None
                st.rerun()
            render_page_preparation()
            st.divider()
            title = st.text_input("Название документа", value="Новый документ", key="upload_job_title")
            target_column, mode_column, preprocess_column = st.columns([1, 1.5, 1])
            document_target = target_column.segmented_control(
                "Тип",
                options=["fragment", "page"],
                format_func=lambda value: "Фрагмент" if value == "fragment" else "Страница",
                default="page",
            )
            page_mode = mode_column.selectbox(
                "Режим",
                options=list(PAGE_MODE_LABELS),
                format_func=PAGE_MODE_LABELS.get,
                disabled=document_target != "page",
            )
            preprocess = preprocess_column.toggle("OCR-предобработка", value=True)
            with st.expander("Параметры модели"):
                model_column, token_column = st.columns([2, 1])
                model_override = model_column.text_input(
                    "Qwen model id",
                    placeholder="Пусто: production registry",
                ).strip()
                max_tokens = token_column.number_input("Токены", 128, 8192, 1024, 128)
            qwen_selected = document_target == "page" and page_mode in QWEN_MODES
            qwen_ready = bool(ml_status and ml_status.get("ready_for_qwen_inference"))
            if qwen_selected and not qwen_ready:
                st.warning("Выбранный Qwen-режим недоступен в текущем runtime.")
            if st.button(
                f"Поставить в очередь: {len(pages)} стр.",
                type="primary",
                icon=":material/play_arrow:",
                disabled=not health or (qwen_selected and not qwen_ready),
                use_container_width=True,
            ):
                try:
                    job = api.create_job(
                        [(page.filename, page.data, page.content_type) for page in pages],
                        title=title,
                        document_target=document_target or "page",
                        page_mode=page_mode,
                        preprocess=preprocess,
                        max_tokens=int(max_tokens),
                        model_override=model_override,
                    )
                    st.session_state["active_job_id"] = job["id"]
                    st.session_state["navigation"] = "jobs"
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
        else:
            st.info("Выберите изображения или PDF.")

    with folder_tab:
        with st.form("folder_job_form"):
            folder = st.text_input("Путь к папке", placeholder=r"C:\Users\...\Pages")
            folder_title = st.text_input("Название", value="Документ из папки")
            folder_columns = st.columns(3)
            recursive = folder_columns[0].toggle("Вложенные папки", value=False)
            sort_by = folder_columns[1].selectbox(
                "Сортировка",
                ["name", "modified", "created"],
                format_func={"name": "По имени", "modified": "По изменению", "created": "По созданию"}.get,
            )
            folder_mode = folder_columns[2].selectbox(
                "Режим",
                options=list(PAGE_MODE_LABELS),
                format_func=PAGE_MODE_LABELS.get,
            )
            folder_preprocess = st.toggle("OCR-предобработка", value=True, key="folder_preprocess")
            submitted = st.form_submit_button(
                "Поставить в очередь",
                type="primary",
                icon=":material/folder_open:",
                use_container_width=True,
                disabled=not health,
            )
        if submitted:
            try:
                job = api.post_json(
                    "/jobs/folder",
                    {
                        "folder": folder,
                        "title": folder_title,
                        "recursive": recursive,
                        "sort_by": sort_by,
                        "document_target": "page",
                        "mode": folder_mode,
                        "preprocess": folder_preprocess,
                        "max_tokens": 1024,
                    },
                    timeout=1800,
                )
                st.session_state["active_job_id"] = job["id"]
                st.session_state["navigation"] = "jobs"
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))


def _job_action(path: str) -> None:
    try:
        api.post(path, timeout=60)
        st.rerun()
    except ApiError as exc:
        st.error(str(exc))


def render_jobs() -> None:
    page_header("Очередь")
    jobs = api_optional("/jobs", limit=100)
    if jobs is None:
        st.error("Очередь недоступна.")
        return
    if not jobs:
        st.info("Очередь пуста.")
        return
    if st.button("Обновить", icon=":material/refresh:"):
        st.rerun()
    ids = [job["id"] for job in jobs]
    preferred = st.session_state.get("active_job_id")
    selected_id = st.selectbox(
        "Задание",
        ids,
        index=ids.index(preferred) if preferred in ids else 0,
        format_func=lambda job_id: next(
            f"{item['title']} · {JOB_STATUS_LABELS.get(item['status'], item['status'])}" for item in jobs if item["id"] == job_id
        ),
    )
    st.session_state["active_job_id"] = selected_id
    job = api_optional(f"/jobs/{selected_id}")
    if not job:
        st.error("Не удалось обновить задание.")
        return

    total = max(1, int(job["total_pages"]))
    finished = int(job["completed_pages"]) + int(job["failed_pages"])
    st.progress(min(1.0, finished / total), text=f"{finished} из {job['total_pages']} страниц")
    metrics = st.columns(4)
    metrics[0].metric("Статус", JOB_STATUS_LABELS.get(job["status"], job["status"]))
    metrics[1].metric("Готово", job["completed_pages"])
    metrics[2].metric("Ошибки", job["failed_pages"])
    metrics[3].metric("Режим", PAGE_MODE_LABELS.get(job["recognition_mode"], job["recognition_mode"]))

    actions = st.columns(5)
    if actions[0].button(
        "Пауза",
        icon=":material/pause:",
        disabled=job["status"] not in {"queued", "running"},
        use_container_width=True,
    ):
        _job_action(f"/jobs/{selected_id}/pause")
    if actions[1].button(
        "Продолжить",
        icon=":material/play_arrow:",
        disabled=job["status"] != "paused",
        use_container_width=True,
    ):
        _job_action(f"/jobs/{selected_id}/resume")
    if actions[2].button(
        "Отменить",
        icon=":material/stop:",
        disabled=job["status"] in {"completed", "failed", "cancelled"},
        use_container_width=True,
    ):
        _job_action(f"/jobs/{selected_id}/cancel")
    if actions[3].button(
        "Повторить",
        icon=":material/replay:",
        disabled=job["status"] not in {"failed", "cancelled", "completed"} or not job["failed_pages"],
        use_container_width=True,
    ):
        _job_action(f"/jobs/{selected_id}/retry")
    if actions[4].button(
        "В документ",
        icon=":material/description:",
        disabled=not job["completed_pages"],
        use_container_width=True,
    ):
        try:
            draft = api.post_json(f"/jobs/{selected_id}/document", {"title": job["title"]})
            st.session_state["selected_draft_id"] = draft["id"]
            st.session_state["navigation"] = "documents"
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

    if job.get("error"):
        st.error(job["error"])
    rows = []
    for page in job["pages"]:
        rows.append(
            {
                "№": page["page_number"],
                "Файл": page["original_filename"],
                "Статус": JOB_STATUS_LABELS.get(page["status"], page["status"]),
                "Надежность": page.get("reliability_score"),
                "Проверка": page.get("needs_review", False),
                "Модель": page.get("model_name") or "",
                "Ошибка": page.get("error") or "",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    completed_pages = [page for page in job["pages"] if page.get("recognized_text")]
    if completed_pages:
        with st.expander("Результаты страниц"):
            page_number = st.selectbox(
                "Страница",
                [page["page_number"] for page in completed_pages],
                key=f"job_page_{selected_id}",
            )
            page = next(item for item in completed_pages if item["page_number"] == page_number)
            st.text_area(
                "Текст",
                value=page.get("corrected_text") or page.get("recognized_text") or "",
                height=260,
                disabled=True,
            )
            if page.get("routing_reason"):
                st.caption(page["routing_reason"])


def _save_document(
    draft_id: str,
    revision: int,
    *,
    title: str | None = None,
    pages: list[dict[str, Any]] | None = None,
    document_text: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"expected_revision": revision}
    if title is not None:
        payload["title"] = title
    if pages is not None:
        payload["pages"] = pages
    if document_text is not None:
        payload["document_text"] = document_text
    try:
        result = api.patch_json(f"/documents/{draft_id}", payload)
        st.session_state["document_notice"] = "Сохранено"
        return result
    except ApiError as exc:
        st.session_state["document_notice"] = str(exc)
        return None


def _autosave_title(draft_id: str, revision: int, widget_key: str) -> None:
    _save_document(draft_id, revision, title=st.session_state[widget_key])


def _autosave_page(
    draft_id: str,
    revision: int,
    pages: list[dict[str, Any]],
    page_index: int,
    widget_key: str,
) -> None:
    updated = [dict(page) for page in pages]
    updated[page_index]["text"] = st.session_state[widget_key]
    document_text = "\n\n\f\n\n".join(str(page.get("text") or "") for page in updated)
    _save_document(draft_id, revision, pages=updated, document_text=document_text)


def _autosave_line(
    draft_id: str,
    revision: int,
    pages: list[dict[str, Any]],
    page_index: int,
    line_index: int,
    widget_key: str,
) -> None:
    updated = [dict(page) for page in pages]
    lines = str(updated[page_index].get("text") or "").splitlines()
    if 0 <= line_index < len(lines):
        lines[line_index] = st.session_state[widget_key]
        updated[page_index]["text"] = "\n".join(lines)
        document_text = "\n\n\f\n\n".join(str(page.get("text") or "") for page in updated)
        _save_document(draft_id, revision, pages=updated, document_text=document_text)


def _autosave_document_text(
    draft_id: str,
    revision: int,
    pages: list[dict[str, Any]],
    widget_key: str,
) -> None:
    document_text = st.session_state[widget_key]
    chunks = [chunk.strip() for chunk in document_text.split("\f")]
    updated = [dict(page) for page in pages]
    if len(chunks) == len(updated):
        for page, text in zip(updated, chunks, strict=False):
            page["text"] = text
    _save_document(draft_id, revision, pages=updated, document_text=document_text)


def _overlay_image(data: bytes, coordinates: list[dict[str, Any]], selected_line: int | None) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 350))
    for coordinate in coordinates:
        number = int(coordinate.get("line_number", 0))
        color = "#d62839" if number == selected_line else "#167d5a"
        box = (
            int(coordinate["x1"]),
            int(coordinate["y1"]),
            int(coordinate["x2"]),
            int(coordinate["y2"]),
        )
        draw.rectangle(box, outline=color, width=line_width)
        draw.text((box[0] + 3, box[1] + 2), str(number), fill=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _document_page_strip(draft_id: str, pages: list[dict[str, Any]], selected: int) -> int:
    if not pages:
        return 0
    start = max(0, min(selected - 2, len(pages) - min(6, len(pages))))
    visible = list(enumerate(pages))[start : start + 6]
    columns = st.columns(len(visible))
    for column, (index, page) in zip(columns, visible, strict=False):
        with column:
            recognition_id = page.get("recognition_id")
            if recognition_id:
                st.image(
                    api.asset_url(f"/api/history/{recognition_id}/image/original"),
                    use_container_width=True,
                )
            if st.button(
                f"{index + 1}",
                key=f"page_strip_{draft_id}_{index}",
                type="primary" if index == selected else "secondary",
                use_container_width=True,
            ):
                return index
    return selected


def _reorder_document_page(draft: dict[str, Any], selected: int, offset: int) -> None:
    pages = [dict(page) for page in draft["pages"]]
    target = selected + offset
    if target < 0 or target >= len(pages):
        return
    pages[selected], pages[target] = pages[target], pages[selected]
    for index, page in enumerate(pages, start=1):
        page["page_number"] = index
    text = "\n\n\f\n\n".join(str(page.get("text") or "") for page in pages)
    if _save_document(draft["id"], draft["revision"], pages=pages, document_text=text):
        st.session_state[f"selected_page_{draft['id']}"] = target
        st.rerun()


def render_document_editor(draft: dict[str, Any]) -> None:
    draft_id = draft["id"]
    revision = int(draft["revision"])
    pages = draft["pages"]
    notice = st.session_state.pop("document_notice", None)
    if notice:
        st.caption(notice)

    title_key = f"draft_title_{draft_id}_{revision}"
    st.text_input(
        "Название",
        value=draft["title"],
        key=title_key,
        on_change=_autosave_title,
        args=(draft_id, revision, title_key),
    )
    document_tab, page_tab, revisions_tab = st.tabs(["Документ", "Страницы", "Версии"])
    with document_tab:
        text_key = f"draft_text_{draft_id}_{revision}"
        st.text_area(
            "Текст документа",
            value=draft["document_text"],
            height=430,
            key=text_key,
            on_change=_autosave_document_text,
            args=(draft_id, revision, pages, text_key),
        )
        with st.expander("Поиск и замена"):
            search_columns = st.columns([2, 2, 1])
            query = search_columns[0].text_input("Найти", key=f"find_{draft_id}")
            replacement = search_columns[1].text_input("Заменить на", key=f"replace_{draft_id}")
            case_sensitive = search_columns[2].toggle("Учитывать регистр", value=False)
            if st.button("Заменить все", icon=":material/find_replace:", disabled=not query):
                updated_pages = [dict(page) for page in pages]
                for page in updated_pages:
                    source = str(page.get("text") or "")
                    if case_sensitive:
                        page["text"] = source.replace(query, replacement)
                    else:
                        import re

                        page["text"] = re.sub(re.escape(query), replacement, source, flags=re.IGNORECASE)
                updated_text = "\n\n\f\n\n".join(str(page.get("text") or "") for page in updated_pages)
                if _save_document(draft_id, revision, pages=updated_pages, document_text=updated_text):
                    st.rerun()

        export_columns = st.columns(4)
        for column, file_format, icon in zip(
            export_columns,
            ["txt", "docx", "json", "pdf"],
            [":material/article:", ":material/description:", ":material/data_object:", ":material/picture_as_pdf:"],
            strict=False,
        ):
            column.link_button(
                file_format.upper(),
                f"{API_URL}/documents/{draft_id}/export?format={file_format}",
                icon=icon,
                use_container_width=True,
            )

    with page_tab:
        if not pages:
            st.info("В документе нет страниц.")
        else:
            selected_key = f"selected_page_{draft_id}"
            st.session_state.setdefault(selected_key, 0)
            selected = min(int(st.session_state[selected_key]), len(pages) - 1)
            strip_selected = _document_page_strip(draft_id, pages, selected)
            if strip_selected != selected:
                st.session_state[selected_key] = strip_selected
                st.rerun()
            selected = strip_selected
            order_actions = st.columns([1, 1, 4])
            if order_actions[0].button(
                "Влево",
                icon=":material/arrow_back:",
                disabled=selected == 0,
                use_container_width=True,
            ):
                _reorder_document_page(draft, selected, -1)
            if order_actions[1].button(
                "Вправо",
                icon=":material/arrow_forward:",
                disabled=selected == len(pages) - 1,
                use_container_width=True,
            ):
                _reorder_document_page(draft, selected, 1)

            page = pages[selected]
            recognition_id = page.get("recognition_id")
            record = api_optional(f"/history/{recognition_id}") if recognition_id else None
            image_column, text_column = st.columns([0.95, 1.05], gap="large")
            selected_line = None
            with image_column:
                if record:
                    image_data = api.asset_bytes(record.get("processed_image_url") or record["original_image_url"])
                    coordinates = record.get("line_coordinates") or []
                    line_options = [int(item["line_number"]) for item in coordinates]
                    selected_line = st.selectbox(
                        "Строка на странице",
                        [None, *line_options],
                        format_func=lambda value: "Все строки" if value is None else f"Строка {value}",
                        key=f"selected_line_{draft_id}_{selected}_{revision}",
                    )
                    st.image(
                        _overlay_image(image_data, coordinates, selected_line),
                        use_container_width=True,
                    )
                    quality = record.get("image_quality") or {}
                    if quality:
                        st.caption(
                            f"Качество {float(quality.get('score', 0)) * 100:.0f}% · "
                            + (", ".join(quality.get("issues") or []) or "без замечаний")
                        )
            with text_column:
                page_key = f"draft_page_{draft_id}_{selected}_{revision}"
                page_text = str(page.get("text") or "")
                page_lines = page_text.splitlines()
                if selected_line is not None and 0 < selected_line <= len(page_lines):
                    selected_line_key = (
                        f"draft_selected_line_{draft_id}_{selected}_{selected_line}_{revision}"
                    )
                    st.text_input(
                        f"Строка {selected_line}",
                        value=page_lines[selected_line - 1],
                        key=selected_line_key,
                        on_change=_autosave_line,
                        args=(
                            draft_id,
                            revision,
                            pages,
                            selected,
                            selected_line - 1,
                            selected_line_key,
                        ),
                    )
                st.text_area(
                    "Текст страницы",
                    value=page_text,
                    height=360,
                    key=page_key,
                    on_change=_autosave_page,
                    args=(draft_id, revision, pages, selected, page_key),
                )
                if record:
                    render_result_meta(record)
                    with st.expander("Исходный OCR"):
                        st.text(record.get("recognized_text") or "")
                    render_feedback(record, page_text, f"draft_{draft_id}")

                lines = page_text.splitlines()
                if lines:
                    with st.expander("Порядок строк"):
                        edited_lines = st.data_editor(
                            [{"Порядок": index, "Текст": text} for index, text in enumerate(lines, start=1)],
                            hide_index=True,
                            num_rows="dynamic",
                            key=f"line_order_{draft_id}_{selected}_{revision}",
                        )
                        if st.button("Применить строки", icon=":material/reorder:"):
                            line_records = (
                                edited_lines.to_dict("records")
                                if hasattr(edited_lines, "to_dict")
                                else list(edited_lines)
                            )
                            ordered = sorted(line_records, key=lambda item: int(item["Порядок"]))
                            updated_pages = [dict(item) for item in pages]
                            updated_pages[selected]["text"] = "\n".join(str(item["Текст"]) for item in ordered)
                            updated_text = "\n\n\f\n\n".join(
                                str(item.get("text") or "") for item in updated_pages
                            )
                            if _save_document(
                                draft_id,
                                revision,
                                pages=updated_pages,
                                document_text=updated_text,
                            ):
                                st.rerun()

    with revisions_tab:
        revisions = api_optional(f"/documents/{draft_id}/revisions", limit=100) or []
        history_actions = st.columns(2)
        if history_actions[0].button(
            "Отменить",
            icon=":material/undo:",
            disabled=not revisions,
            use_container_width=True,
        ):
            target = int(revisions[0]["revision"])
            try:
                api.post_json(
                    f"/documents/{draft_id}/restore/{target}",
                    {"expected_revision": revision},
                )
                st.session_state["redo_revision"][draft_id] = revision
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
        redo_target = st.session_state["redo_revision"].get(draft_id)
        if history_actions[1].button(
            "Повторить",
            icon=":material/redo:",
            disabled=redo_target is None,
            use_container_width=True,
        ):
            try:
                api.post_json(
                    f"/documents/{draft_id}/restore/{redo_target}",
                    {"expected_revision": revision},
                )
                st.session_state["redo_revision"].pop(draft_id, None)
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
        if revisions:
            st.dataframe(
                [
                    {
                        "Версия": item["revision"],
                        "Дата": format_datetime(item["created_at"]),
                        "Название": item["title"],
                        "Страниц": len(item["pages"]),
                    }
                    for item in revisions
                ],
                hide_index=True,
                use_container_width=True,
            )


def render_documents() -> None:
    page_header("Документы")
    drafts = api_optional("/documents", limit=100)
    if drafts is None:
        st.error("Черновики недоступны.")
        return
    if not drafts:
        st.info("Черновиков пока нет. Создайте документ из готового задания.")
        return
    ids = [draft["id"] for draft in drafts]
    preferred = st.session_state.get("selected_draft_id")
    draft_id = st.selectbox(
        "Документ",
        ids,
        index=ids.index(preferred) if preferred in ids else 0,
        format_func=lambda item_id: next(item["title"] for item in drafts if item["id"] == item_id),
    )
    st.session_state["selected_draft_id"] = draft_id
    draft = api_optional(f"/documents/{draft_id}")
    if draft:
        render_document_editor(draft)
        with st.expander("Удаление"):
            confirmed = st.checkbox("Подтверждаю удаление черновика", key=f"delete_confirm_{draft_id}")
            if st.button(
                "Удалить документ",
                icon=":material/delete:",
                disabled=not confirmed,
            ):
                try:
                    api.delete(f"/documents/{draft_id}")
                    st.session_state["selected_draft_id"] = None
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))


def render_result_meta(result: dict[str, Any]) -> None:
    reliability = result.get("reliability_score")
    confidence = result.get("confidence")
    score = reliability if reliability is not None else confidence
    score_label = "нет оценки" if score is None else f"{float(score) * 100:.1f}%"
    vram_label = (
        f" · VRAM {format_bytes(result.get('peak_vram_bytes'))}"
        if result.get("peak_vram_bytes") is not None
        else ""
    )
    st.markdown(
        '<div class="result-meta">'
        f"{html.escape(MODE_LABELS.get(result.get('recognition_mode'), result.get('recognition_mode', '')))} · "
        f"{html.escape(result.get('model_name', 'модель не указана'))} · "
        f"{result.get('processing_time_ms', 0) / 1000:.1f} с · надежность {score_label}{vram_label}"
        "</div>",
        unsafe_allow_html=True,
    )
    if result.get("routing_reason"):
        st.caption(result["routing_reason"])


def render_feedback(result: dict[str, Any], corrected_text: str, key_prefix: str) -> None:
    default_rating = result.get("feedback_rating") or ("partial" if result.get("needs_review") else "exact")
    rating = st.segmented_control(
        "Оценка",
        options=list(FEEDBACK_LABELS),
        format_func=FEEDBACK_LABELS.get,
        default=default_rating,
        key=f"{key_prefix}_rating_{result['id']}",
    )
    include_for_training = st.checkbox(
        "Разрешить использовать этот исправленный пример в будущих training exports",
        value=bool(result.get("feedback_training_approved")),
        key=f"{key_prefix}_training_{result['id']}",
    )
    feedback_actions = st.columns(2)
    if feedback_actions[0].button(
        "Сохранить проверку",
        icon=":material/save:",
        key=f"{key_prefix}_feedback_{result['id']}",
        use_container_width=True,
    ):
        try:
            saved = api.post_json(
                f"/history/{result['id']}/feedback",
                {
                    "rating": rating or default_rating,
                    "corrected_text": corrected_text,
                    "include_for_training": include_for_training,
                },
            )
            result.update(saved)
            st.success("Проверка сохранена.")
        except ApiError as exc:
            st.error(str(exc))
    if feedback_actions[1].button(
        "Исключить из обучения",
        icon=":material/block:",
        key=f"{key_prefix}_exclude_{result['id']}",
        disabled=not result.get("feedback_saved_for_training"),
        use_container_width=True,
    ):
        try:
            result.update(api.post(f"/feedback/{result['id']}/exclude"))
            st.success("Пример исключен из training exports.")
        except ApiError as exc:
            st.error(str(exc))


def history_matches(
    item: dict[str, Any],
    query: str,
    mode: str,
    status: str,
    model: str,
    date_from: date,
    date_to: date,
) -> bool:
    haystack = " ".join(
        str(item.get(key) or "")
        for key in ("original_filename", "recognized_text", "corrected_text", "model_name")
    ).casefold()
    if query and query.casefold() not in haystack:
        return False
    if mode != "all" and item.get("recognition_mode") != mode:
        return False
    if model != "all" and item.get("model_name") != model:
        return False
    if status == "review" and not item.get("needs_review"):
        return False
    if status == "rated" and not item.get("feedback_rating"):
        return False
    if status == "unrated" and item.get("feedback_rating"):
        return False
    created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")).date()
    return date_from <= created <= date_to


def render_history() -> None:
    page_header("История")
    history = api_optional("/history", limit=500)
    if history is None:
        st.error("История недоступна.")
        return
    if not history:
        st.info("История пуста.")
        return
    first_date = min(datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")).date() for item in history)
    filter_one = st.columns([2, 1, 1, 1])
    query = filter_one[0].text_input("Поиск", placeholder="Файл, текст или модель")
    modes = ["all", *sorted({item.get("recognition_mode", "fragment") for item in history})]
    mode = filter_one[1].selectbox(
        "Режим",
        modes,
        format_func=lambda value: "Все" if value == "all" else MODE_LABELS.get(value, value),
    )
    models = ["all", *sorted({item.get("model_name", "") for item in history})]
    model = filter_one[2].selectbox("Модель", models, format_func=lambda value: "Все" if value == "all" else value)
    status = filter_one[3].selectbox(
        "Статус",
        ["all", "review", "rated", "unrated"],
        format_func={"all": "Все", "review": "Проверить", "rated": "Оценено", "unrated": "Без оценки"}.get,
    )
    date_columns = st.columns(2)
    date_from = date_columns[0].date_input("С", value=first_date)
    date_to = date_columns[1].date_input("По", value=date.today())
    filtered = [item for item in history if history_matches(item, query, mode, status, model, date_from, date_to)]
    st.metric("Найдено", len(filtered))
    for item in filtered:
        marker = " · проверить" if item.get("needs_review") else ""
        with st.expander(f"{item['original_filename']} · {format_datetime(item['created_at'])}{marker}"):
            image_column, result_column = st.columns([0.8, 1.2], gap="large")
            with image_column:
                st.image(api.asset_url(item["original_image_url"]), use_container_width=True)
                if item.get("processed_image_url"):
                    st.image(api.asset_url(item["processed_image_url"]), caption="После обработки", use_container_width=True)
            with result_column:
                text = st.text_area(
                    "Текст",
                    value=item.get("corrected_text") or item.get("recognized_text") or "",
                    height=260,
                    key=f"history_text_{item['id']}",
                )
                render_result_meta(item)
                render_feedback(item, text, "history")


def render_feedback_review() -> None:
    problems = api_optional("/feedback/problems")
    if problems is None:
        st.error("Очередь недоступна.")
        return
    if not problems:
        st.success("Очередь пользовательских примеров пуста.")
        return
    query = st.text_input("Поиск", placeholder="Файл или текст", key="feedback_review_search")
    visible = [
        item
        for item in problems
        if not query
        or query.casefold() in f"{item.get('original_filename', '')} {item.get('recognized_text', '')}".casefold()
    ]
    st.metric("В очереди", len(visible))
    for item in visible:
        with st.expander(f"#{item['recognition_id']} · {item['original_filename']}"):
            image_column, text_column = st.columns([0.85, 1.15], gap="large")
            image_column.image(api.asset_url(item["image_url"]), use_container_width=True)
            with text_column:
                st.text_area("Ответ модели", value=item.get("recognized_text") or "", disabled=True)
                corrected = st.text_area(
                    "Проверенный текст",
                    value=item.get("corrected_text") or "",
                    key=f"review_corrected_{item['recognition_id']}",
                    height=220,
                )
                consent = st.checkbox(
                    "Разрешить training export",
                    value=bool(item.get("training_approved")),
                    key=f"review_consent_{item['recognition_id']}",
                )
                if st.button(
                    "Сохранить",
                    type="primary",
                    icon=":material/save:",
                    key=f"review_save_{item['recognition_id']}",
                    use_container_width=True,
                ):
                    try:
                        api.post_json(
                            "/feedback",
                            {
                                "recognition_id": item["recognition_id"],
                                "rating": item["rating"],
                                "corrected_text": corrected,
                                "include_for_training": consent,
                            },
                        )
                        st.success("Сохранено.")
                    except ApiError as exc:
                        st.error(str(exc))


def render_gold_review() -> None:
    summary = api_optional("/gold-review/summary")
    items = api_optional("/gold-review")
    if summary is None or items is None:
        st.error("Gold subset недоступен.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Всего", summary["total"])
    metrics[1].metric("Ожидают", summary["pending"])
    metrics[2].metric("Подтверждено", summary["verified"])
    metrics[3].metric("Отклонено", summary["rejected"])
    status_filter = st.segmented_control(
        "Статус",
        ["all", "pending_human_review", "verified", "rejected"],
        format_func={
            "all": "Все",
            "pending_human_review": "Ожидают",
            "verified": "Подтверждены",
            "rejected": "Отклонены",
        }.get,
        default="pending_human_review",
    )
    visible = items if status_filter == "all" else [item for item in items if item["review_status"] == status_filter]
    if not visible:
        st.info("В выбранной группе нет страниц.")
        return
    benchmark_id = st.selectbox(
        "Страница",
        [item["benchmark_id"] for item in visible],
        format_func=lambda item_id: f"{item_id} · {next(item.get('source_dataset', 'hwr200') for item in visible if item['benchmark_id'] == item_id)}",
    )
    item = next(record for record in visible if record["benchmark_id"] == benchmark_id)
    image_column, review_column = st.columns([0.9, 1.1], gap="large")
    image_column.image(api.asset_url(f"/api/gold-review/{benchmark_id}/image"), use_container_width=True)
    with review_column:
        reference = st.text_area(
            "Эталонный текст",
            value=item.get("reference_text") or "",
            height=430,
            key=f"gold_text_{benchmark_id}_{item.get('reviewed_at', 'new')}",
        )
        review_status = st.segmented_control(
            "Решение",
            ["verified", "rejected", "pending_human_review"],
            format_func={"verified": "Подтвердить", "rejected": "Отклонить", "pending_human_review": "Отложить"}.get,
            default=item["review_status"],
        )
        reviewer = st.text_input("Проверяющий", value=item.get("reviewer") or "")
        notes = st.text_input("Примечание", value=item.get("review_notes") or "")
        if st.button("Сохранить решение", type="primary", icon=":material/fact_check:", use_container_width=True):
            try:
                api.patch_json(
                    f"/gold-review/{benchmark_id}",
                    {
                        "review_status": review_status,
                        "reference_text": reference,
                        "reviewer": reviewer,
                        "review_notes": notes,
                    },
                )
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
    if summary["ready_to_finalize"]:
        if st.button("Зафиксировать gold v1", icon=":material/lock:"):
            try:
                api.post("/gold-review/finalize")
                st.success("Gold benchmark зафиксирован.")
            except ApiError as exc:
                st.error(str(exc))


def render_review() -> None:
    page_header("Проверка")
    feedback_tab, gold_tab = st.tabs(["Пользовательские примеры", "Gold benchmark"])
    with feedback_tab:
        render_feedback_review()
    with gold_tab:
        render_gold_review()


def render_model_manager() -> None:
    installed = api_optional("/models/installed") or []
    action_columns = st.columns(3)
    if action_columns[0].button("Найти локальные адаптеры", icon=":material/search:", use_container_width=True):
        try:
            api.post("/models/discover")
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    if action_columns[1].button("Rollback TrOCR", icon=":material/settings_backup_restore:", use_container_width=True):
        try:
            api.post_json("/models/rollback", {"backend": "trocr"})
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    if action_columns[2].button("Rollback Qwen", icon=":material/settings_backup_restore:", use_container_width=True):
        try:
            api.post_json("/models/rollback", {"backend": "qwen"})
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))

    for model in installed:
        production = " · production" if model.get("is_production") else ""
        with st.expander(f"{model['id']} · {model['backend']} · {model['stage']}{production}"):
            info = st.columns(4)
            info[0].metric("Размер", format_bytes(model.get("size_bytes")))
            info[1].metric("Revision", model.get("revision") or "не указана")
            info[2].metric("Установлена", "Да" if model.get("installed") else "Нет")
            info[3].metric("Checksum", "Есть" if model.get("checksum_sha256") else "Нет")
            st.caption(model.get("local_path") or model.get("base_model") or "")
            controls = st.columns(3)
            if controls[0].button(
                "Проверить",
                icon=":material/verified:",
                key=f"verify_{model['id']}",
                use_container_width=True,
            ):
                result = api_optional(f"/models/{model['id']}/verify")
                if result and result.get("valid"):
                    st.success("Целостность подтверждена.")
                else:
                    st.error((result or {}).get("reason", "Проверка не выполнена."))
            if controls[1].button(
                "Production",
                icon=":material/publish:",
                key=f"promote_{model['id']}",
                disabled=model.get("is_production", False),
                use_container_width=True,
            ):
                try:
                    api.post(f"/models/{model['id']}/promote")
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))
            delete_files = controls[2].checkbox("Удалить файлы", key=f"delete_files_{model['id']}")
            if controls[2].button(
                "Удалить",
                icon=":material/delete:",
                key=f"delete_model_{model['id']}",
                disabled=model.get("is_production", False),
                use_container_width=True,
            ):
                try:
                    api.delete(f"/models/{model['id']}", delete_files=delete_files)
                    st.rerun()
                except ApiError as exc:
                    st.error(str(exc))

    with st.expander("Установить модель"):
        st.warning("Эта операция подключается к Hugging Face и загружает файлы только после подтверждения.")
        with st.form("install_model_form"):
            install_columns = st.columns(2)
            model_id = install_columns[0].text_input("Локальный id", placeholder="qwen3-vl-2b")
            repo_id = install_columns[1].text_input("Hugging Face repo", value="Qwen/Qwen3-VL-2B-Instruct")
            option_columns = st.columns(3)
            backend = option_columns[0].selectbox("Backend", ["qwen", "trocr"])
            revision = option_columns[1].text_input("Revision")
            expected_sha = option_columns[2].text_input("SHA-256")
            install = st.form_submit_button(
                "Начать загрузку",
                type="primary",
                icon=":material/download:",
                use_container_width=True,
            )
        if install:
            try:
                api.post_json(
                    "/models/install",
                    {
                        "id": model_id,
                        "repo_id": repo_id,
                        "backend": backend,
                        "revision": revision or None,
                        "expected_sha256": expected_sha or None,
                        "stage": "experimental",
                    },
                    timeout=60,
                )
                st.success("Загрузка поставлена в очередь.")
            except ApiError as exc:
                st.error(str(exc))
    downloads = api_optional("/models/downloads") or []
    if downloads:
        st.subheader("Загрузки")
        for item in downloads[:10]:
            total = item.get("total_bytes") or 0
            progress = item["downloaded_bytes"] / total if total else 0.0
            st.progress(
                min(1.0, progress),
                text=f"{item['model_id']} · {item['status']} · {format_bytes(item['downloaded_bytes'])}",
            )
            if item.get("error"):
                st.error(item["error"])


def render_dictionary() -> None:
    dictionary = api_optional("/dictionary") or {"words": []}
    words = dictionary["words"]
    st.metric("Слов", len(words))
    add_columns = st.columns([3, 1])
    word = add_columns[0].text_input("Новое слово", key="dictionary_word")
    if add_columns[1].button("Добавить", icon=":material/add:", use_container_width=True, disabled=not word.strip()):
        try:
            api.post_json("/dictionary", {"word": word})
            st.rerun()
        except ApiError as exc:
            st.error(str(exc))
    if words:
        selected = st.selectbox("Словарь", words)
        if st.button("Удалить слово", icon=":material/delete:"):
            try:
                api.delete(f"/dictionary/{selected}")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))


def render_system(health: dict | None, ml_status: dict | None) -> None:
    page_header("Система")
    diagnostics = api_optional("/diagnostics")
    qwen_status = api_optional("/qwen-status")
    metrics = st.columns(4)
    metrics[0].metric("API", "Готов" if health else "Недоступен")
    metrics[1].metric("TrOCR", "Готов" if ml_status and ml_status.get("ready_for_trocr") else "Не готов")
    metrics[2].metric("Qwen", "Готов" if ml_status and ml_status.get("ready_for_qwen_inference") else "Не готов")
    gpu_devices = (ml_status or {}).get("gpu_devices") or []
    metrics[3].metric("GPU", gpu_devices[0]["name"] if gpu_devices else "CPU")
    model_tab, environment_tab, diagnostics_tab, dictionary_tab = st.tabs(
        ["Модели", "Окружение", "Диагностика", "Словарь"]
    )
    with model_tab:
        render_model_manager()
        if qwen_status:
            st.caption(
                f"Pipeline: {qwen_status.get('pipeline_maturity')} · production ready: "
                f"{qwen_status.get('production_ready')}"
            )
            for reason in qwen_status.get("blocking_reasons") or []:
                st.warning(reason)
    with environment_tab:
        packages = (ml_status or {}).get("packages") or []
        if packages:
            st.dataframe(
                [
                    {"Пакет": item["package"], "Установлен": item["installed"], "Версия": item.get("version") or ""}
                    for item in packages
                ],
                hide_index=True,
                use_container_width=True,
            )
        if gpu_devices:
            st.dataframe(gpu_devices, hide_index=True, use_container_width=True)
    with diagnostics_tab:
        if diagnostics:
            storage = diagnostics["storage"]
            diagnostic_metrics = st.columns(3)
            diagnostic_metrics[0].metric("Свободно", format_bytes(storage.get("free_bytes")))
            diagnostic_metrics[1].metric("Python", diagnostics["application"].get("python"))
            diagnostic_metrics[2].metric("CUDA", (diagnostics["runtime"].get("cuda_version") or "нет"))
            st.json({"storage": storage, "ports": diagnostics["ports"], "recent_errors": diagnostics["recent_errors"]})
            st.download_button(
                "Экспорт диагностики",
                data=json.dumps(diagnostics, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="handwriting_ocr_diagnostics.json",
                mime="application/json",
                icon=":material/download:",
            )
        audit = api_optional("/acceptance-audit") or []
        if audit:
            st.dataframe(audit, hide_index=True, use_container_width=True)
    with dictionary_tab:
        render_dictionary()


def format_datetime(value: str | None) -> str:
    if not value:
        return "дата не указана"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return value


def format_bytes(value: int | None) -> str:
    if value is None:
        return "нет данных"
    amount = float(value)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if amount < 1024 or unit == "ТБ":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} ТБ"


init_state()
health_status = api_optional("/health")
ml_environment = api_optional("/ml-status", timeout=15)
selected_section = render_sidebar(health_status, ml_environment)

if selected_section == "workspace":
    render_workspace(health_status, ml_environment)
elif selected_section == "jobs":
    render_jobs()
elif selected_section == "documents":
    render_documents()
elif selected_section == "history":
    render_history()
elif selected_section == "review":
    render_review()
else:
    render_system(health_status, ml_environment)
