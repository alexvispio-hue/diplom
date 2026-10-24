import csv
from dataclasses import dataclass
from pathlib import Path

from app.core.config import BASE_DIR


@dataclass(frozen=True)
class ExperimentSummaryRow:
    summary_path: str
    model: str
    mode: str
    dataset: str
    cer: float | None
    wer: float | None
    page_time: float | None
    line_time: float | None
    notes: str


class ExperimentResultsService:
    """Reads saved experiment summary.csv files from data/evaluation."""

    def __init__(self, evaluation_dir: Path | None = None) -> None:
        self.evaluation_dir = evaluation_dir or BASE_DIR / "data" / "evaluation"

    def list_page_summaries(self) -> list[ExperimentSummaryRow]:
        if not self.evaluation_dir.exists():
            return []

        rows: list[ExperimentSummaryRow] = []
        for summary_path in sorted(self.evaluation_dir.rglob("summary.csv")):
            rows.extend(self._read_summary(summary_path))
        return rows

    def _read_summary(self, summary_path: Path) -> list[ExperimentSummaryRow]:
        if not self._is_safe_path(summary_path):
            return []

        with summary_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            return [
                ExperimentSummaryRow(
                    summary_path=str(summary_path.relative_to(self.evaluation_dir)),
                    model=row.get("model", ""),
                    mode=row.get("mode", ""),
                    dataset=row.get("dataset", ""),
                    cer=self._parse_float(row.get("CER")),
                    wer=self._parse_float(row.get("WER")),
                    page_time=self._parse_float(row.get("page_time")),
                    line_time=self._parse_float(row.get("line_time")),
                    notes=row.get("notes", ""),
                )
                for row in reader
            ]

    def _is_safe_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.evaluation_dir.resolve())
        except ValueError:
            return False
        return path.name == "summary.csv"

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
