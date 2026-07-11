from scripts.prepare_hwr200_full_qwen import (
    estimate_line_period,
    align_repaired_documents,
    find_component_line_peaks,
    find_line_peaks,
    normalize_alignment_text,
    select_draft_repair_keys,
    similarity,
    split_reference_by_drafts,
)

import numpy as np


def test_split_reference_by_page_drafts_preserves_full_reference() -> None:
    reference = "Первая страница заканчивается здесь. Вторая страница начинается отсюда."
    drafts = [
        "Первая странипа заканчивается здесь",
        "Вторая страница начинается отсюда",
    ]

    pages = split_reference_by_drafts(reference, drafts)

    assert pages == [
        "Первая страница заканчивается здесь.",
        "Вторая страница начинается отсюда.",
    ]
    assert " ".join(pages) == reference


def test_split_reference_supports_page_break_inside_sentence() -> None:
    reference = "Один два три четыре пять шесть семь восемь."
    pages = split_reference_by_drafts(reference, ["Один два три четыре", "пять шесть семь восемь"])

    assert pages[0].endswith("четыре")
    assert pages[1].startswith("пять")
    assert similarity(" ".join(pages), reference) == 1.0


def test_alignment_normalization_ignores_case_and_punctuation() -> None:
    assert normalize_alignment_text("Привет, МИР!") == "привет мир"
    assert similarity("Привет мир", "привет, мир!") == 1.0


def test_projection_finds_periodic_handwriting_lines_on_noisy_background() -> None:
    projection = np.full(500, 20.0)
    expected = list(range(40, 480, 40))
    for center in expected:
        projection[center - 3 : center + 4] += np.array([10, 25, 45, 60, 45, 25, 10])

    period = estimate_line_period(projection)
    peaks = find_line_peaks(projection, period)

    assert 36 <= period <= 44
    assert len(peaks) == len(expected)
    assert all(abs(actual - target) <= 3 for actual, target in zip(peaks, expected))


def test_component_peaks_ignore_narrow_noise_at_page_edge() -> None:
    binary = np.zeros((500, 400), dtype=np.uint8)
    for center in (80, 140, 200):
        binary[center - 8 : center + 8, 30:300] = 255
    for center in range(40, 460, 30):
        binary[center - 5 : center + 5, 385:399] = 255

    peaks = find_component_line_peaks(binary)

    assert len(peaks) == 3
    assert all(abs(actual - target) <= 3 for actual, target in zip(peaks, (80, 140, 200)))


def test_select_draft_repair_keys_finds_outliers_and_short_nonfinal_pages() -> None:
    inventory = [
        {"author_id": 4, "document_name": "reuse1", "scan_pages": 3},
    ]
    drafts = [
        {"author_id": 4, "document_name": "reuse1", "page_index": 1, "line_count": 65},
        {"author_id": 4, "document_name": "reuse1", "page_index": 2, "line_count": 4},
        {"author_id": 4, "document_name": "reuse1", "page_index": 3, "line_count": 2},
    ]

    selected = select_draft_repair_keys(inventory, drafts, max_lines=60, min_lines=5)

    assert selected == {(4, "reuse1", 1), (4, "reuse1", 2)}


def test_align_repaired_documents_replaces_only_affected_document(tmp_path) -> None:
    inventory = [
        {
            "author_id": 1,
            "document_name": "reuse1",
            "full_text": "первая страница вторая страница",
            "conditions": {"Сканы": ["one.jpg", "two.jpg"]},
        },
        {
            "author_id": 2,
            "document_name": "reuse2",
            "full_text": "без изменений",
            "conditions": {"Сканы": ["three.jpg"]},
        },
    ]
    drafts = [
        {"author_id": 1, "document_name": "reuse1", "page_index": 1, "draft_text": "первая страница", "line_drafts": ["первая страница"], "error": None},
        {"author_id": 1, "document_name": "reuse1", "page_index": 2, "draft_text": "вторая страница", "line_drafts": ["вторая страница"], "error": None},
    ]
    old_affected = {"author_id": 1, "document_name": "reuse1", "page_index": 1, "target_text": "old"}
    untouched = {"author_id": 2, "document_name": "reuse2", "page_index": 1, "target_text": "без изменений"}

    merged = align_repaired_documents(
        inventory,
        drafts,
        [old_affected, untouched],
        {(1, "reuse1", 1)},
        tmp_path / "labels.jsonl",
        tmp_path / "review.jsonl",
        0.35,
    )

    assert merged[0]["target_text"] == "первая страница"
    assert merged[1] is untouched
