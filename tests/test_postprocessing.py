from app.ocr.postprocessing import normalize_recognized_text


def test_normalize_recognized_text_compacts_spaces() -> None:
    assert normalize_recognized_text(" hello   world ") == "hello world"


def test_normalize_recognized_text_limits_blank_lines() -> None:
    assert normalize_recognized_text("a\n\n\n\nb") == "a\n\nb"
