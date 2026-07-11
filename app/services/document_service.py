from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.enum.text import WD_BREAK


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class DocumentImportError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedPage:
    source_name: str
    page_number: int
    filename: str
    content_type: str
    data: bytes


@dataclass(frozen=True)
class ExportPage:
    page_number: int
    filename: str
    text: str
    recognition_id: int | None = None
    model_name: str | None = None
    recognition_mode: str | None = None
    processing_time_ms: int | None = None
    needs_review: bool = False


def expand_document(
    filename: str,
    data: bytes,
    content_type: str | None = None,
    *,
    pdf_render_dpi: int = 160,
    max_pages: int = 100,
) -> list[PreparedPage]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        return _render_pdf_pages(
            filename,
            data,
            pdf_render_dpi=pdf_render_dpi,
            max_pages=max_pages,
        )
    if suffix not in IMAGE_SUFFIXES:
        raise DocumentImportError(f"Неподдерживаемый формат файла: {suffix or 'без расширения'}.")
    if not data:
        raise DocumentImportError(f"Файл {filename} пуст.")
    return [
        PreparedPage(
            source_name=filename,
            page_number=1,
            filename=filename,
            content_type=content_type or _image_content_type(suffix),
            data=data,
        )
    ]


def build_export_pages(results: Iterable[dict[str, Any]]) -> list[ExportPage]:
    pages = []
    for index, result in enumerate(results, start=1):
        pages.append(
            ExportPage(
                page_number=int(result.get("document_page_number") or index),
                filename=str(result.get("filename") or result.get("original_filename") or f"page_{index}.png"),
                text=str(result.get("edited_text") or result.get("corrected_text") or result.get("recognized_text") or ""),
                recognition_id=result.get("id"),
                model_name=result.get("model_name"),
                recognition_mode=result.get("recognition_mode"),
                processing_time_ms=result.get("processing_time_ms"),
                needs_review=bool(result.get("needs_review", False)),
            )
        )
    return pages


def export_text(pages: Iterable[ExportPage], *, page_headers: bool = True) -> str:
    blocks = []
    for page in pages:
        text = page.text.strip()
        if page_headers:
            blocks.append(f"Страница {page.page_number}\n\n{text}".rstrip())
        else:
            blocks.append(text)
    return "\n\n\f\n\n".join(blocks).strip()


def export_json(pages: Iterable[ExportPage], *, title: str = "Распознанный документ") -> bytes:
    payload = {
        "title": title,
        "pages": [asdict(page) for page in pages],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_docx(pages: Iterable[ExportPage], *, title: str = "Распознанный документ") -> bytes:
    page_list = list(pages)
    document = Document()
    document.core_properties.title = title
    document.add_heading(title, level=0)

    for index, page in enumerate(page_list):
        document.add_heading(f"Страница {page.page_number}", level=1)
        for line in page.text.splitlines() or [""]:
            document.add_paragraph(line)
        if index + 1 < len(page_list):
            document.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)

    output = io.BytesIO()
    document.save(output)
    return output.getvalue()


def _render_pdf_pages(
    filename: str,
    data: bytes,
    *,
    pdf_render_dpi: int,
    max_pages: int,
) -> list[PreparedPage]:
    try:
        import fitz
    except ImportError as exc:
        raise DocumentImportError(
            "Для открытия PDF установите зависимость pymupdf."
        ) from exc

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentImportError(f"Не удалось открыть PDF {filename}.") from exc

    try:
        if document.page_count == 0:
            raise DocumentImportError(f"PDF {filename} не содержит страниц.")
        if document.page_count > max_pages:
            raise DocumentImportError(
                f"PDF содержит {document.page_count} страниц; разрешено не более {max_pages}."
            )
        scale = max(72, pdf_render_dpi) / 72
        matrix = fitz.Matrix(scale, scale)
        stem = Path(filename).stem
        return [
            PreparedPage(
                source_name=filename,
                page_number=index + 1,
                filename=f"{stem}_page_{index + 1:03d}.png",
                content_type="image/png",
                data=document.load_page(index).get_pixmap(matrix=matrix, alpha=False).tobytes("png"),
            )
            for index in range(document.page_count)
        ]
    finally:
        document.close()


def _image_content_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
