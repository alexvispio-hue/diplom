import sys
import types
from pathlib import Path

import pytest

from app.core.config import QWEN_MODEL_PRESETS, Settings
from app.ocr.qwen_vl_recognizer import QwenVLRecognizer


def build_recognizer(monkeypatch: pytest.MonkeyPatch, adapter_dir: Path | None) -> QwenVLRecognizer:
    settings = Settings(qwen_lora_adapter_dir=adapter_dir)
    monkeypatch.setattr("app.ocr.qwen_vl_recognizer.get_settings", lambda: settings)
    return QwenVLRecognizer()


def test_qwen_lora_adapter_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = build_recognizer(monkeypatch, None)

    assert recognizer._resolve_lora_adapter_dir() is None
    assert recognizer._apply_lora_adapter("base-model") == "base-model"
    assert recognizer.model_name == "Qwen/Qwen3-VL-2B-Instruct"


def test_qwen_lora_adapter_missing_path_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recognizer = build_recognizer(monkeypatch, tmp_path / "missing-adapter")

    with pytest.raises(RuntimeError, match="does not exist"):
        recognizer._resolve_lora_adapter_dir()


def test_qwen_lora_adapter_wraps_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path: str):
            return {"model": model, "adapter_path": adapter_path}

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))
    recognizer = build_recognizer(monkeypatch, adapter_dir)

    adapted = recognizer._apply_lora_adapter("base-model")

    assert adapted == {"model": "base-model", "adapter_path": str(adapter_dir)}
    assert recognizer.model_name == "Qwen/Qwen3-VL-2B-Instruct+lora:adapter"


def test_qwen_model_preset_resolves_light_model() -> None:
    settings = Settings(qwen_model_preset="light")

    assert settings.qwen_effective_model_name == QWEN_MODEL_PRESETS["light"]


def test_qwen_model_name_overrides_preset() -> None:
    settings = Settings(qwen_model_preset="main", qwen_model_name="local/qwen-test")

    assert settings.qwen_effective_model_name == "local/qwen-test"


def test_unknown_qwen_model_preset_fails() -> None:
    settings = Settings(qwen_model_preset="unknown")

    with pytest.raises(ValueError, match="Unknown Qwen model preset"):
        _ = settings.qwen_effective_model_name
