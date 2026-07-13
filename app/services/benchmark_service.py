from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable

from app.ocr.metrics import calculate_metrics


@dataclass(frozen=True)
class BenchmarkSummary:
    pages: int
    mean_cer: float
    mean_wer: float
    exact_page_rate: float
    exact_line_rate: float
    line_detection_recall: float
    line_detection_precision: float
    empty_response_rate: float
    truncated_response_rate: float
    median_latency_ms: float | None
    p95_latency_ms: float | None
    peak_vram_bytes: int | None
    backend_memory_growth_bytes: int | None
    review_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualityGateResult:
    passed: bool
    checks: dict[str, bool]
    baseline: BenchmarkSummary
    candidate: BenchmarkSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


def summarize_predictions(records: Iterable[dict[str, Any]]) -> BenchmarkSummary:
    items = list(records)
    if not items:
        raise ValueError("Benchmark requires at least one prediction record.")

    cer_values = []
    wer_values = []
    exact_pages = 0
    exact_lines = 0
    reference_lines_total = 0
    predicted_lines_total = 0
    matched_line_slots = 0
    empty_responses = 0
    truncated_responses = 0
    latency_values = []
    vram_values = []
    backend_memory_values = []
    review_count = 0
    for item in items:
        reference = str(item.get("reference_text") or "")
        prediction = str(item.get("prediction_text") or "")
        metrics = calculate_metrics(reference, prediction)
        cer_values.append(metrics.character_error_rate)
        wer_values.append(metrics.word_error_rate)
        if metrics.character_distance == 0:
            exact_pages += 1
        reference_lines = reference.splitlines() or ([reference] if reference else [])
        predicted_lines = prediction.splitlines() or ([prediction] if prediction else [])
        reference_lines_total += len(reference_lines)
        predicted_lines_total += len(predicted_lines)
        matched_line_slots += min(len(reference_lines), len(predicted_lines))
        exact_lines += sum(
            expected.strip() == actual.strip()
            for expected, actual in zip(reference_lines, predicted_lines, strict=False)
        )
        if not prediction.strip():
            empty_responses += 1
        if item.get("truncated"):
            truncated_responses += 1
        if item.get("latency_ms") is not None:
            latency_values.append(float(item["latency_ms"]))
        if item.get("peak_vram_bytes") is not None:
            vram_values.append(int(item["peak_vram_bytes"]))
        if item.get("backend_memory_bytes") is not None:
            backend_memory_values.append(int(item["backend_memory_bytes"]))
        if item.get("needs_review"):
            review_count += 1

    return BenchmarkSummary(
        pages=len(items),
        mean_cer=sum(cer_values) / len(cer_values),
        mean_wer=sum(wer_values) / len(wer_values),
        exact_page_rate=exact_pages / len(items),
        exact_line_rate=exact_lines / reference_lines_total if reference_lines_total else 0.0,
        line_detection_recall=(
            matched_line_slots / reference_lines_total if reference_lines_total else 0.0
        ),
        line_detection_precision=(
            matched_line_slots / predicted_lines_total if predicted_lines_total else 0.0
        ),
        empty_response_rate=empty_responses / len(items),
        truncated_response_rate=truncated_responses / len(items),
        median_latency_ms=median(latency_values) if latency_values else None,
        p95_latency_ms=_percentile(latency_values, 0.95) if latency_values else None,
        peak_vram_bytes=max(vram_values) if vram_values else None,
        backend_memory_growth_bytes=(
            backend_memory_values[-1] - backend_memory_values[0]
            if len(backend_memory_values) >= 2
            else None
        ),
        review_rate=review_count / len(items),
    )


def evaluate_quality_gate(
    baseline: BenchmarkSummary,
    candidate: BenchmarkSummary,
    *,
    minimum_relative_cer_improvement: float = 0.10,
    maximum_relative_wer_regression: float = 0.02,
    maximum_empty_or_truncated_rate: float = 0.01,
    maximum_p95_latency_ms: float = 30_000,
    maximum_vram_bytes: int = 16 * 1024**3,
    school_baseline: BenchmarkSummary | None = None,
    school_candidate: BenchmarkSummary | None = None,
    maximum_school_relative_cer_regression: float = 0.03,
    regression_baseline: BenchmarkSummary | None = None,
    regression_candidate: BenchmarkSummary | None = None,
) -> QualityGateResult:
    cer_target = baseline.mean_cer * (1 - minimum_relative_cer_improvement)
    wer_limit = baseline.mean_wer * (1 + maximum_relative_wer_regression)
    empty_or_truncated = candidate.empty_response_rate + candidate.truncated_response_rate
    checks = {
        "cer_improved": candidate.mean_cer <= cer_target,
        "wer_not_regressed": candidate.mean_wer <= wer_limit,
        "responses_complete": empty_or_truncated < maximum_empty_or_truncated_rate,
        "latency_within_limit": (
            candidate.p95_latency_ms is not None
            and candidate.p95_latency_ms <= maximum_p95_latency_ms
        ),
        "vram_within_limit": (
            candidate.peak_vram_bytes is not None
            and candidate.peak_vram_bytes <= maximum_vram_bytes
        ),
    }
    if school_baseline is not None and school_candidate is not None:
        checks["school_subset_not_regressed"] = school_candidate.mean_cer <= (
            school_baseline.mean_cer * (1 + maximum_school_relative_cer_regression)
        )
    if regression_baseline is not None and regression_candidate is not None:
        checks["user_regression_not_worse"] = regression_candidate.mean_cer <= regression_baseline.mean_cer
    return QualityGateResult(
        passed=all(checks.values()),
        checks=checks,
        baseline=baseline,
        candidate=candidate,
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty list.")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]
