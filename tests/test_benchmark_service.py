from app.services.benchmark_service import evaluate_quality_gate, summarize_predictions


def test_benchmark_summary_and_quality_gate() -> None:
    baseline = summarize_predictions(
        [
            {"reference_text": "один два", "prediction_text": "один три", "latency_ms": 1000, "peak_vram_bytes": 8},
            {"reference_text": "текст", "prediction_text": "", "latency_ms": 1500, "peak_vram_bytes": 8},
        ]
    )
    candidate = summarize_predictions(
        [
            {"reference_text": "один два", "prediction_text": "один два", "latency_ms": 900, "peak_vram_bytes": 8},
            {"reference_text": "текст", "prediction_text": "текст", "latency_ms": 1200, "peak_vram_bytes": 8},
        ]
    )

    gate = evaluate_quality_gate(
        baseline,
        candidate,
        maximum_empty_or_truncated_rate=0.01,
        maximum_vram_bytes=16,
    )

    assert baseline.empty_response_rate == 0.5
    assert candidate.exact_page_rate == 1.0
    assert candidate.p95_latency_ms == 1200
    assert gate.passed is True
