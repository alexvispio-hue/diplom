import io
from threading import Lock
from unittest.mock import patch

import pytest
from PIL import Image

from public_demo.ocr import MAX_BYTES, FragmentRecognizer, crop_fragment, read_image


def image_bytes(size=(200, 80), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_limits_and_formats():
    assert read_image(image_bytes()).size == (200, 80)
    for payload in (b"", b"not an image", b"x" * (MAX_BYTES + 1)):
        with pytest.raises(ValueError):
            read_image(payload)
    with pytest.raises(ValueError, match="12 мегапикселей"):
        read_image(image_bytes((4000, 3100)))
    buffer = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buffer, format="GIF")
    with pytest.raises(ValueError, match="PNG"):
        read_image(buffer.getvalue())


def test_transparency_and_orientation():
    assert read_image(image_bytes(mode="RGBA")).getpixel((0, 0)) == (255, 255, 255)
    buffer = io.BytesIO()
    image = Image.new("RGB", (80, 40), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(buffer, format="JPEG", exif=exif)
    assert read_image(buffer.getvalue()).size == (40, 80)


def test_crop_bounds():
    image = Image.new("RGB", (200, 100))
    assert crop_fragment(image, (25, 75), (10, 90)).size == (100, 80)
    for horizontal, vertical in (((50, 50), (0, 100)), ((0, 101), (0, 100)), ((0, 1), (0, 100))):
        with pytest.raises(ValueError):
            crop_fragment(image, horizontal, vertical)


def test_concurrent_request_is_rejected_without_running_model():
    recognizer = FragmentRecognizer.__new__(FragmentRecognizer)
    recognizer.lock = Lock()
    recognizer.lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="другой запрос"):
            recognizer.recognize(Image.new("RGB", (20, 20)))
    finally:
        recognizer.lock.release()


def test_streamlit_upload_recognize_edit_and_crop_reset():
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    st.cache_resource.clear()
    upload = io.BytesIO(image_bytes())
    with patch("streamlit.file_uploader", return_value=upload):
        app = AppTest.from_file("public_demo/streamlit_app.py").run(timeout=20)
        assert not app.exception
        assert "целого листа" in app.warning[0].value
        with patch("public_demo.ocr.FragmentRecognizer.recognize", return_value="тестовый текст"), patch(
            "public_demo.ocr.FragmentRecognizer.__init__", return_value=None
        ):
            app.button[0].click().run()
        assert not app.exception
        assert app.text_area[0].value == "тестовый текст"
        app.text_area[0].set_value("исправленный текст").run()
        assert app.text_area[0].value == "исправленный текст"
        app.slider[0].set_value((10, 90)).run()
        assert not app.exception
        assert len(app.text_area) == 0
    st.cache_resource.clear()


def test_streamlit_empty_start_and_bad_file():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("public_demo/streamlit_app.py").run(timeout=20)
    assert not app.exception
    assert "целого листа" in app.warning[0].value
    with patch("streamlit.file_uploader", return_value=io.BytesIO(b"broken")):
        app = AppTest.from_file("public_demo/streamlit_app.py").run(timeout=20)
        assert not app.exception
        assert "прочитать" in app.error[0].value
