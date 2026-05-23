from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.ocr.base import BaseRecognizer, RecognitionResult
from app.ocr.postprocessing import normalize_recognized_text


class TrOCRRecognizer(BaseRecognizer):
    """Local transformer-based recognizer using Hugging Face TrOCR."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.model_name = model_name or settings.trocr_model_name
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"

    def recognize(self, image_path: Path) -> RecognitionResult:
        self._ensure_loaded()

        image = Image.open(image_path).convert("RGB")
        pixel_values = self._processor(images=image, return_tensors="pt").pixel_values.to(self._device)

        with self._torch.inference_mode():
            generated_ids = self._model.generate(pixel_values)

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return RecognitionResult(
            text=normalize_recognized_text(text),
            confidence=None,
            model_name=self.model_name,
        )

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "ML-зависимости не установлены. Установите их командой "
                "`python -m pip install -r requirements-ml.txt`."
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.generation_config.max_length = self.settings.generation_max_new_tokens
        self._model.eval()
