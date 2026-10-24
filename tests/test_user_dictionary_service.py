from app.services.user_dictionary_service import UserDictionaryService


def test_dictionary_adds_deduplicates_and_removes_words(tmp_path) -> None:
    service = UserDictionaryService(tmp_path / "dictionary.json")

    assert service.add("Алгоритм") == ["Алгоритм"]
    assert service.add("алгоритм") == ["Алгоритм"]
    assert service.add("Qwen") == ["Qwen", "Алгоритм"]
    assert service.remove("АЛГОРИТМ") == ["Qwen"]
