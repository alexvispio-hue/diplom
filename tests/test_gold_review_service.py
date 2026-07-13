import json

import pytest

from app.services.gold_review_service import GoldReviewService


def test_gold_review_keeps_candidates_immutable_and_tracks_review_state(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    original = {
        "benchmark_id": "page-1",
        "image_path": "image.png",
        "reference_text": "draft",
        "review_status": "pending_human_review",
    }
    candidates.write_text(json.dumps(original) + "\n", encoding="utf-8")
    state = tmp_path / "state.json"
    service = GoldReviewService(candidates, state)

    reviewed = service.update(
        "page-1",
        review_status="verified",
        reference_text="verified text",
        reviewer="reviewer",
        review_notes=None,
    )

    assert reviewed["reference_text"] == "verified text"
    assert service.summary()["verified"] == 1
    assert json.loads(candidates.read_text(encoding="utf-8"))["reference_text"] == "draft"


def test_gold_review_cannot_finalize_incomplete_subset(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "benchmark_id": "page-1",
                "image_path": "image.png",
                "reference_text": "draft",
                "review_status": "pending_human_review",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    service = GoldReviewService(candidates, tmp_path / "state.json")

    with pytest.raises(ValueError, match="not ready"):
        service.finalize(tmp_path / "gold.jsonl")
