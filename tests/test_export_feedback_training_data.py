import csv
import json
from pathlib import Path

from scripts.export_feedback_training_data import iter_feedback_examples, write_csv, write_qwen_jsonl


def test_exports_only_reviewed_feedback_examples(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "problems"
    feedback_dir.mkdir()
    (feedback_dir / "recognition_1.jpg").write_bytes(b"image")
    (feedback_dir / "recognition_1.json").write_text(
        json.dumps(
            {
                "recognition_id": 1,
                "image_filename": "recognition_1.jpg",
                "recognized_text": "ошибка",
                "corrected_text": "правильный текст",
                "recognition_mode": "qwen_page",
                "model_name": "qwen",
                "rating": "partial",
                "training_approved": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (feedback_dir / "recognition_2.jpg").write_bytes(b"image")
    (feedback_dir / "recognition_2.json").write_text(
        json.dumps(
            {
                "recognition_id": 2,
                "image_filename": "recognition_2.jpg",
                "recognized_text": "без проверки",
                "corrected_text": None,
                "rating": "mismatch",
                "training_approved": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    examples = iter_feedback_examples(feedback_dir, only_reviewed=True)

    assert len(examples) == 1
    assert examples[0]["text"] == "правильный текст"
    assert examples[0]["recognition_mode"] == "qwen_page"


def test_writes_feedback_csv(tmp_path: Path) -> None:
    output_csv = tmp_path / "reviewed_examples.csv"
    write_csv(
        [
            {
                "image_path": "problems/recognition_1.jpg",
                "text": "правка",
                "recognized_text": "ошибка",
                "recognition_id": 1,
                "recognition_mode": "trocr_lines",
                "model_name": "trocr",
                "rating": "partial",
                "training_approved": True,
            }
        ],
        output_csv,
    )

    with output_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["text"] == "правка"
    assert rows[0]["rating"] == "partial"


def test_writes_qwen_jsonl_from_reviewed_feedback(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "feedback.jsonl"
    write_qwen_jsonl(
        [
            {
                "image_path": "problems/recognition_1.jpg",
                "text": "corrected text",
                "recognized_text": "wrong text",
                "recognition_id": 1,
                "recognition_mode": "qwen_page",
                "model_name": "qwen",
                "rating": "partial",
                "training_approved": True,
            }
        ],
        output_jsonl,
    )

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"][0]["image"] == "problems/recognition_1.jpg"
    assert rows[0]["messages"][1]["content"] == "corrected text"


def test_exports_feedback_metadata_with_utf8_sig(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "problems"
    feedback_dir.mkdir()
    (feedback_dir / "recognition_1.jpg").write_bytes(b"image")
    (feedback_dir / "recognition_1.json").write_text(
        json.dumps(
            {
                "recognition_id": 1,
                "image_filename": "recognition_1.jpg",
                "recognized_text": "wrong",
                "corrected_text": "corrected",
                "recognition_mode": "qwen_page",
                "model_name": "qwen",
                "rating": "partial",
            }
        ),
        encoding="utf-8-sig",
    )

    examples = iter_feedback_examples(feedback_dir, only_reviewed=True)

    assert len(examples) == 1
    assert examples[0]["text"] == "corrected"


def test_feedback_export_skips_explicitly_excluded_sample(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "problems"
    feedback_dir.mkdir()
    (feedback_dir / "recognition_1.jpg").write_bytes(b"image")
    (feedback_dir / "recognition_1.json").write_text(
        json.dumps(
            {
                "recognition_id": 1,
                "image_filename": "recognition_1.jpg",
                "corrected_text": "text",
                "training_approved": True,
                "training_excluded": True,
            }
        ),
        encoding="utf-8",
    )

    assert iter_feedback_examples(feedback_dir) == []
