import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextRecognitionMetrics:
    character_error_rate: float
    word_error_rate: float
    character_distance: int
    word_distance: int


def levenshtein_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    if reference == hypothesis:
        return 0

    previous = list(range(len(hypothesis) + 1))
    for i, reference_item in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_item in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (reference_item != hypothesis_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = _normalize_metric_text(reference)
    hypothesis = _normalize_metric_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _tokenize_words(reference)
    hypothesis_words = _tokenize_words(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def calculate_metrics(reference: str, hypothesis: str) -> TextRecognitionMetrics:
    normalized_reference = _normalize_metric_text(reference)
    normalized_hypothesis = _normalize_metric_text(hypothesis)
    reference_words = _tokenize_words(reference)
    hypothesis_words = _tokenize_words(hypothesis)

    return TextRecognitionMetrics(
        character_error_rate=character_error_rate(reference, hypothesis),
        word_error_rate=word_error_rate(reference, hypothesis),
        character_distance=levenshtein_distance(normalized_reference, normalized_hypothesis),
        word_distance=levenshtein_distance(reference_words, hypothesis_words),
    )


def _normalize_metric_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize_words(text: str) -> list[str]:
    normalized = _normalize_metric_text(text)
    return normalized.split() if normalized else []
