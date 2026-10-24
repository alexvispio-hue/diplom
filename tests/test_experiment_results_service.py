import csv
from pathlib import Path

from app.services.experiment_results_service import ExperimentResultsService


def test_experiment_results_service_reads_page_summary(tmp_path: Path) -> None:
    summary_dir = tmp_path / "page_experiment"
    summary_dir.mkdir()
    summary_path = summary_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["model", "mode", "dataset", "CER", "WER", "page_time", "line_time", "notes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "stub",
                "mode": "trocr_lines",
                "dataset": "pages.csv",
                "CER": "0.100000",
                "WER": "0.200000",
                "page_time": "42.00",
                "line_time": "21.00",
                "notes": "items=1",
            }
        )

    rows = ExperimentResultsService(evaluation_dir=tmp_path).list_page_summaries()

    assert len(rows) == 1
    assert rows[0].summary_path.replace("\\", "/") == "page_experiment/summary.csv"
    assert rows[0].model == "stub"
    assert rows[0].mode == "trocr_lines"
    assert rows[0].cer == 0.1
    assert rows[0].wer == 0.2
    assert rows[0].page_time == 42.0
    assert rows[0].line_time == 21.0
