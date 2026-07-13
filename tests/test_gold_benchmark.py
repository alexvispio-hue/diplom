import json
from pathlib import Path

from app.training.gold_benchmark import build_gold_candidates, build_sft_gold_candidates


def _example(path: str, text: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": path}]},
                {"role": "assistant", "content": text},
            ]
        },
        ensure_ascii=False,
    )


def test_gold_candidates_are_unique_and_author_balanced(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    test_path = tmp_path / "test.jsonl"
    metadata = []
    examples = []
    for author in range(3):
        for page in range(2):
            path = f"images/test/author_{author:03d}_doc_page_{page:03d}_scan.jpg"
            metadata.append(
                {
                    "image_path": path,
                    "split": "test",
                    "author_id": author,
                    "document_name": "doc",
                    "page_index": page,
                    "condition": "Сканы",
                    "source_member": path,
                    "needs_review": False,
                }
            )
            examples.append(_example(path, f"text {author} {page}"))
    metadata_path.write_text("\n".join(map(json.dumps, metadata)), encoding="utf-8")
    test_path.write_text("\n".join(examples), encoding="utf-8")

    candidates = build_gold_candidates(metadata_path, test_path, count=6)

    assert len(candidates) == 6
    assert {item["author_id"] for item in candidates} == {0, 1, 2}
    assert len({item["benchmark_id"] for item in candidates}) == 6
    assert all(item["review_status"] == "pending_human_review" for item in candidates)


def test_sft_gold_candidates_skip_layout_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    text_example = json.loads(_example("images/page.jpg", "page text"))
    text_example["messages"][0]["content"].append({"type": "text", "text": "Recognize all text."})
    layout_example = json.loads(_example("images/page.jpg", "[]"))
    layout_example["messages"][0]["content"].append({"type": "text", "text": "Return JSON layout."})
    path.write_text(
        json.dumps(text_example, ensure_ascii=False) + "\n" + json.dumps(layout_example, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    candidates = build_sft_gold_candidates(path, count=1, dataset_name="school")

    assert candidates[0]["benchmark_id"] == "school-page"
    assert candidates[0]["reference_text"] == "page text"
