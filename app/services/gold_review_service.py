from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import BASE_DIR
from app.training.gold_benchmark import read_jsonl, write_jsonl
from app.training.run_manifest import atomic_write_json


REVIEW_STATUSES = {"pending_human_review", "verified", "rejected"}


class GoldReviewService:
    def __init__(
        self,
        candidates_path: Path = BASE_DIR / "manifests" / "benchmarks" / "gold-candidates-v1.jsonl",
        state_path: Path = BASE_DIR / "data" / "runtime" / "gold-review-v1.json",
    ) -> None:
        self.candidates_path = candidates_path
        self.state_path = state_path

    def list(self, *, status: str | None = None) -> list[dict[str, Any]]:
        state = self._read_state().get("reviews", {})
        result = []
        for candidate in read_jsonl(self.candidates_path):
            merged = dict(candidate)
            merged.update(state.get(candidate["benchmark_id"], {}))
            if status is None or merged.get("review_status") == status:
                result.append(merged)
        return result

    def get(self, benchmark_id: str) -> dict[str, Any]:
        for item in self.list():
            if item["benchmark_id"] == benchmark_id:
                return item
        raise ValueError(f"Gold candidate was not found: {benchmark_id}")

    def update(
        self,
        benchmark_id: str,
        *,
        review_status: str,
        reference_text: str,
        reviewer: str | None,
        review_notes: str | None,
    ) -> dict[str, Any]:
        if review_status not in REVIEW_STATUSES:
            raise ValueError(f"Unsupported gold review status: {review_status}")
        candidate = self.get(benchmark_id)
        normalized_text = reference_text.strip()
        if review_status == "verified" and not normalized_text:
            raise ValueError("Verified gold page must contain reference text.")
        state = self._read_state()
        state.setdefault("reviews", {})[benchmark_id] = {
            "review_status": review_status,
            "reference_text": normalized_text,
            "reviewer": (reviewer or "").strip() or None,
            "review_notes": (review_notes or "").strip() or None,
            "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        state["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        atomic_write_json(self.state_path, state)
        candidate.update(state["reviews"][benchmark_id])
        return candidate

    def summary(self) -> dict[str, Any]:
        items = self.list()
        counts = {status: 0 for status in REVIEW_STATUSES}
        sources: dict[str, int] = {}
        for item in items:
            counts[item.get("review_status", "pending_human_review")] += 1
            source = str(item.get("source_dataset") or "hwr200")
            sources[source] = sources.get(source, 0) + 1
        return {
            "total": len(items),
            "pending": counts["pending_human_review"],
            "verified": counts["verified"],
            "rejected": counts["rejected"],
            "sources": sources,
            "ready_to_finalize": counts["pending_human_review"] == 0 and counts["verified"] >= 200,
        }

    def image_path(self, benchmark_id: str) -> Path:
        candidate = self.get(benchmark_id)
        path = (BASE_DIR / candidate["image_path"]).resolve()
        try:
            path.relative_to(BASE_DIR.resolve())
        except ValueError as exc:
            raise ValueError("Gold candidate image path leaves the project directory.") from exc
        if not path.is_file():
            raise ValueError(f"Gold candidate image was not found: {path}")
        return path

    def finalize(self, output_path: Path) -> dict[str, Any]:
        summary = self.summary()
        if not summary["ready_to_finalize"]:
            raise ValueError(
                "Gold benchmark is not ready: every candidate must be reviewed and at least 200 verified pages are required."
            )
        verified = [item for item in self.list() if item["review_status"] == "verified"]
        write_jsonl(output_path, verified)
        checksum = _sha256(output_path)
        lock_path = output_path.with_suffix(output_path.suffix + ".lock.json")
        lock = {
            "schema_version": 1,
            "benchmark": output_path.name,
            "pages": len(verified),
            "sha256": checksum,
            "candidates_sha256": _sha256(self.candidates_path),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        atomic_write_json(lock_path, lock)
        return lock

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema_version": 1, "reviews": {}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
