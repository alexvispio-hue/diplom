from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings


class UserDictionaryService:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().user_dictionary_path

    def list_words(self) -> list[str]:
        if not self.path.is_file():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return sorted(
            {str(word).strip() for word in payload.get("words", []) if str(word).strip()},
            key=str.casefold,
        )

    def add(self, word: str) -> list[str]:
        normalized = word.strip()
        if not normalized:
            raise ValueError("Слово не может быть пустым.")
        words = self.list_words()
        if normalized.casefold() not in {item.casefold() for item in words}:
            words.append(normalized)
        return self._write(words)

    def remove(self, word: str) -> list[str]:
        normalized = word.strip().casefold()
        words = [item for item in self.list_words() if item.casefold() != normalized]
        return self._write(words)

    def _write(self, words: list[str]) -> list[str]:
        ordered = sorted(set(words), key=str.casefold)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "words": ordered}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return ordered
