from pathlib import Path

from PIL import Image

from app.core.config import BASE_DIR, get_settings
from app.ocr.base import BaseRecognizer, RecognitionResult
from app.ocr.postprocessing import normalize_recognized_text


class QwenVLRecognizer(BaseRecognizer):
    """Lazy local recognizer for Qwen-VL image-to-text OCR experiments."""

    def __init__(self, model_name: str | None = None, prompt: str | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.model_source = model_name or settings.qwen_effective_model_name
        self.model_name = self.model_source
        self.prompt = prompt or "Recognize Russian handwritten text in the image. Return only the text."
        self._processor = None
        self._model = None
        self._torch = None

    def recognize(self, image_path: Path, max_new_tokens: int | None = None) -> RecognitionResult:
        self._ensure_loaded()
        text = self._generate(image_path, self.prompt, max_new_tokens=max_new_tokens)
        return RecognitionResult(
            text=normalize_recognized_text(text),
            confidence=None,
            model_name=self.model_name,
        )

    def recognize_page(self, image_path: Path, max_new_tokens: int | None = None) -> RecognitionResult:
        self._ensure_loaded()
        prompt = (
            "Recognize all Russian handwritten text on the page. Preserve line order and line breaks. "
            "Return only the final text."
        )
        text = self._generate(image_path, prompt, max_new_tokens=max_new_tokens)
        return RecognitionResult(
            text=normalize_recognized_text(text),
            confidence=None,
            model_name=self.model_name,
        )

    def recognize_layout(self, image_path: Path, max_new_tokens: int | None = None) -> RecognitionResult:
        self._ensure_loaded()
        prompt = (
            "Find all handwritten Russian text lines on this page. Return only valid JSON array. "
            "Each item must contain line_number, text, and box fields. "
            "box must be [x1, y1, x2, y2] in image pixel coordinates."
        )
        text = self._generate(image_path, prompt, max_new_tokens=max_new_tokens).strip()
        return RecognitionResult(
            text=text,
            confidence=None,
            model_name=self.model_name,
        )

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are not installed. Install transformers, torch, accelerate, "
                "and qwen-vl-utils before running a Qwen mode."
            ) from exc

        model_class = None
        import_errors: list[Exception] = []
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            try:
                module = __import__("transformers", fromlist=[class_name])
                model_class = getattr(module, class_name)
                break
            except (ImportError, AttributeError) as exc:
                import_errors.append(exc)
        if model_class is None:
            raise RuntimeError("The installed transformers version does not support Qwen-VL models.") from import_errors[-1]

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_source, trust_remote_code=True)
        model_kwargs = {"trust_remote_code": True}
        if self.settings.qwen_device_map:
            model_kwargs["device_map"] = self.settings.qwen_device_map
        if self.settings.qwen_torch_dtype and self.settings.qwen_torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, self.settings.qwen_torch_dtype)
        elif self.settings.qwen_torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        self._model = model_class.from_pretrained(self.model_source, **model_kwargs)
        self._model = self._apply_lora_adapter(self._model)
        self._model.eval()

    def _resolve_lora_adapter_dir(self) -> Path | None:
        adapter_dir = self.settings.qwen_lora_adapter_dir
        if adapter_dir is None:
            return None

        adapter_path = Path(adapter_dir)
        if not adapter_path.is_absolute():
            adapter_path = BASE_DIR / adapter_path

        if not adapter_path.exists():
            raise RuntimeError(f"Qwen LoRA adapter directory does not exist: {adapter_path}")
        if not adapter_path.is_dir():
            raise RuntimeError(f"Qwen LoRA adapter path is not a directory: {adapter_path}")
        return adapter_path

    def _apply_lora_adapter(self, model):
        adapter_dir = self._resolve_lora_adapter_dir()
        if adapter_dir is None:
            return model

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen LoRA adapter is configured, but peft is not installed. "
                "Install requirements-ml.txt before loading the adapter."
            ) from exc

        adapted_model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model_name = f"{self.model_source}+lora:{adapter_dir.name}"
        return adapted_model

    def _generate(self, image_path: Path, prompt: str, max_new_tokens: int | None = None) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")
        inputs = self._move_inputs_to_model(inputs)

        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.settings.generation_max_new_tokens,
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = generated_ids[:, input_length:]
        return self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _move_inputs_to_model(self, inputs):
        try:
            device = next(self._model.parameters()).device
        except StopIteration:
            return inputs
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
