import json
from io import BytesIO

import fitz
import pytest
from docx import Document
from PIL import Image

from app.services.document_service import (
    DocumentImportError,
    ExportPage,
    build_export_pages,
    expand_document,
    export_docx,
    export_json,
    export_searchable_pdf,
    export_text,
    load_image_folder,
    transform_image,
)


def test_expand_document_keeps_image_as_one_page() -> None:
    payload = BytesIO()
    Image.new("RGB", (20, 10), "white").save(payload, format="PNG")

    pages = expand_document("note.png", payload.getvalue(), "image/png")

    assert len(pages) == 1
    assert pages[0].filename == "note.png"
    assert pages[0].data == payload.getvalue()


def test_expand_document_rejects_unknown_format() -> None:
    with pytest.raises(DocumentImportError, match="Неподдерживаемый формат"):
        expand_document("note.txt", b"text", "text/plain")


def test_expand_document_renders_pdf_pages_in_order() -> None:
    source = fitz.open()
    source.new_page().insert_text((72, 72), "Page one")
    source.new_page().insert_text((72, 72), "Page two")
    payload = source.tobytes()
    source.close()

    pages = expand_document("notes.pdf", payload, "application/pdf", pdf_render_dpi=96)

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.filename for page in pages] == ["notes_page_001.png", "notes_page_002.png"]
    assert all(page.data.startswith(b"\x89PNG") for page in pages)


def test_build_exports_uses_edited_text() -> None:
    pages = build_export_pages(
        [
            {
                "id": 7,
                "filename": "page.png",
                "recognized_text": "draft",
                "edited_text": "verified",
                "model_name": "local-model",
            }
        ]
    )

    assert pages[0].text == "verified"
    assert "Страница 1" in export_text(pages)
    assert "verified" in export_text(pages)
    assert json.loads(export_json(pages))["pages"][0]["recognition_id"] == 7


def test_export_docx_contains_pages_and_text() -> None:
    content = export_docx(
        [
            ExportPage(page_number=1, filename="one.png", text="Первая строка"),
            ExportPage(page_number=2, filename="two.png", text="Вторая строка"),
        ],
        title="Конспект",
    )

    document = Document(BytesIO(content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Конспект" in text
    assert "Страница 1" in text
    assert "Первая строка" in text
    assert "Страница 2" in text
    assert "Вторая строка" in text


def test_load_image_folder_uses_natural_order(tmp_path) -> None:
    for name in ("page_10.png", "page_2.png", "page_1.png"):
        Image.new("RGB", (12, 8), "white").save(tmp_path / name)

    pages = load_image_folder(tmp_path)

    assert [page.filename for page in pages] == ["page_1.png", "page_2.png", "page_10.png"]
    assert [page.page_number for page in pages] == [1, 2, 3]


def test_transform_image_rotates_and_crops() -> None:
    payload = BytesIO()
    Image.new("RGB", (30, 20), "white").save(payload, format="PNG")

    transformed = transform_image(
        payload.getvalue(),
        rotation_degrees=90,
        crop_box=(2, 3, 18, 27),
    )

    with Image.open(BytesIO(transformed)) as image:
        assert image.size == (16, 24)


def test_export_searchable_pdf_keeps_image_and_text_layer() -> None:
    payload = BytesIO()
    Image.new("RGB", (240, 120), "white").save(payload, format="PNG")
    pages = [
        ExportPage(
            page_number=1,
            filename="page.png",
            text="Searchable page text",
            line_coordinates=[
                {"line_number": 1, "x1": 10, "y1": 10, "x2": 220, "y2": 40}
            ],
        )
    ]

    content = export_searchable_pdf(pages, {1: payload.getvalue()})

    document = fitz.open(stream=content, filetype="pdf")
    try:
        assert document.page_count == 1
        assert "Searchable page text" in document[0].get_text().replace("\xa0", " ")
        assert document[0].get_images(full=True)
    finally:
        document.close()
