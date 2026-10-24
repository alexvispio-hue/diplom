from app.image_processing.quality_analysis import ImageQualityReport
from app.services.model_router_service import ModelRouterService


def _quality(score: float) -> ImageQualityReport:
    return ImageQualityReport(score, 100, 150, 50, 1000, 1000, [], False)


def test_auto_router_is_explainable_and_respects_runtime() -> None:
    router = ModelRouterService()

    assert router.choose("auto", _quality(0.8), qwen_available=True).selected_mode == "trocr_lines"
    assert router.choose("auto", _quality(0.6), qwen_available=True).selected_mode == "hybrid"
    assert router.choose("auto", _quality(0.3), qwen_available=True).selected_mode == "qwen_page"
    fallback = router.choose("auto", _quality(0.3), qwen_available=False)
    assert fallback.selected_mode == "trocr_lines"
    assert "unavailable" in fallback.reason
