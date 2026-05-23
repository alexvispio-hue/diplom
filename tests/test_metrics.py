from app.ocr.metrics import (
    calculate_metrics,
    character_error_rate,
    levenshtein_distance,
    word_error_rate,
)


def test_levenshtein_distance_for_strings() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3


def test_character_error_rate() -> None:
    assert character_error_rate("abcd", "abxd") == 0.25


def test_word_error_rate() -> None:
    assert word_error_rate("hello brave world", "hello world") == 1 / 3


def test_calculate_metrics() -> None:
    metrics = calculate_metrics("hello world", "hello word")

    assert metrics.character_distance == 1
    assert metrics.word_distance == 1
    assert metrics.character_error_rate > 0
    assert metrics.word_error_rate == 0.5
