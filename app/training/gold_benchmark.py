from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def extract_image_path(example: dict[str, Any]) -> str:
    for item in example["messages"][0]["content"]:
        if item.get("type") == "image":
            return str(item["image"])
    raise ValueError("Qwen example does not contain an image path.")


def build_gold_candidates(
    metadata_path: Path,
    test_jsonl_path: Path,
    *,
    count: int = 200,
    include_review: bool = False,
) -> list[dict[str, Any]]:
    metadata = read_jsonl(metadata_path)
    examples = {
        extract_image_path(example): example["messages"][1]["content"]
        for example in read_jsonl(test_jsonl_path)
    }

    preferred = []
    fallback = []
    for item in metadata:
        if item.get("split") != "test":
            continue
        if item.get("needs_review") and not include_review:
            continue
        image_path = str(item["image_path"])
        if image_path not in examples:
            continue
        target = preferred if image_path.lower().endswith("_scan.jpg") else fallback
        target.append(item)

    selected_variants: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in [*preferred, *fallback]:
        page_key = (
            int(item["author_id"]),
            str(item["document_name"]),
            int(item["page_index"]),
        )
        selected_variants.setdefault(page_key, item)

    by_author: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    for item in sorted(
        selected_variants.values(),
        key=lambda value: (int(value["author_id"]), str(value["document_name"]), int(value["page_index"])),
    ):
        by_author[int(item["author_id"])].append(item)

    candidates = []
    author_ids = sorted(by_author)
    while len(candidates) < count and author_ids:
        next_authors = []
        for author_id in author_ids:
            if len(candidates) >= count:
                break
            queue = by_author[author_id]
            if not queue:
                continue
            item = queue.popleft()
            image_path = str(item["image_path"])
            candidates.append(
                {
                    "benchmark_id": (
                        f"hwr200-a{author_id:03d}-{item['document_name']}-"
                        f"p{int(item['page_index']):03d}"
                    ),
                    "image_path": image_path,
                    "reference_text": examples[image_path],
                    "author_id": author_id,
                    "document_name": item["document_name"],
                    "page_index": int(item["page_index"]),
                    "condition": item.get("condition"),
                    "source_member": item.get("source_member"),
                    "source_needs_review": bool(item.get("needs_review")),
                    "review_status": "pending_human_review",
                    "reviewer": None,
                    "review_notes": None,
                }
            )
            if queue:
                next_authors.append(author_id)
        author_ids = next_authors

    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} unique gold candidates are available; requested {count}.")
    return candidates


def build_sft_gold_candidates(
    test_jsonl_path: Path,
    *,
    count: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    candidates = []
    seen_images = set()
    for example in read_jsonl(test_jsonl_path):
        image_path = extract_image_path(example)
        prompt_parts = example["messages"][0].get("content", [])
        prompt = " ".join(
            str(part.get("text") or "")
            for part in prompt_parts
            if part.get("type") == "text"
        ).lower()
        if "json" in prompt or image_path in seen_images:
            continue
        seen_images.add(image_path)
        candidates.append(
            {
                "benchmark_id": f"{dataset_name}-{Path(image_path).stem}",
                "image_path": image_path,
                "reference_text": example["messages"][1]["content"],
                "source_dataset": dataset_name,
                "review_status": "pending_human_review",
                "reviewer": None,
                "review_notes": None,
            }
        )
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} unique {dataset_name} candidates are available; requested {count}.")
    return candidates


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
