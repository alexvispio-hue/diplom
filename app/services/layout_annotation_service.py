import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayoutLine:
    line_number: int
    text: str
    box: tuple[int, int, int, int]


def parse_layout_json(text: str) -> list[LayoutLine]:
    payload = _extract_json_payload(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Layout response is not valid JSON.") from exc
    if not isinstance(data, list):
        raise ValueError("Layout response must be a JSON array.")

    lines = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Layout item {index} must be an object.")
        line_number = int(item.get("line_number") or index)
        line_text = str(item.get("text") or "").strip()
        box = item.get("box")
        if not line_text:
            continue
        if not isinstance(box, list | tuple) or len(box) != 4:
            raise ValueError(f"Layout item {index} must contain box [x1, y1, x2, y2].")
        x1, y1, x2, y2 = [int(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Layout item {index} has invalid box coordinates.")
        lines.append(LayoutLine(line_number=line_number, text=line_text, box=(x1, y1, x2, y2)))
    return sorted(lines, key=lambda line: (line.line_number, line.box[1], line.box[0]))


class LayoutAnnotationExporter:
    """Converts qwen_layout evaluation CSV rows into a draft lines.csv manifest."""

    def export(
        self,
        layout_csv_path: Path,
        output_csv_path: Path,
    ) -> int:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written = 0
        with layout_csv_path.open("r", encoding="utf-8-sig", newline="") as source, output_csv_path.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.DictReader(source)
            writer = csv.writer(target)
            writer.writerow(["image_path", "text", "page_id", "line_number", "x1", "y1", "x2", "y2", "source_image"])
            for row in reader:
                image_path = row["image_path"]
                page_id = row.get("page_id") or Path(image_path).stem
                for line in parse_layout_json(row["recognized_text"]):
                    x1, y1, x2, y2 = line.box
                    writer.writerow([image_path, line.text, page_id, line.line_number, x1, y1, x2, y2, image_path])
                    rows_written += 1
        return rows_written


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped
