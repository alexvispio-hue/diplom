import argparse
import io
import json
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.postprocessing import normalize_recognized_text


PAGE_PROMPT = (
    "Recognize all Russian handwritten text on this page. "
    "Preserve the reading order and line breaks. Return only the final text."
)
CONDITIONS = ("Сканы", "ФотоСветлое", "ФотоТемное")
CONDITION_SLUGS = {"Сканы": "scan", "ФотоСветлое": "light", "ФотоТемное": "dark"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MEMBER_PATTERN = re.compile(r"hw_dataset/(\d+)/([^/]+)/([^/]+)/([^/]+)$")
TEXT_MEMBER_PATTERN = re.compile(r"hw_dataset/(\d+)/([^/]+)\.txt$")
ANNOTATION_PATTERN = re.compile(r"annotations/(?:Originals|Reuse|FPR)/(\d+)/([^/]+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all HWR200 pages for Qwen SFT by aligning page OCR drafts to document transcripts."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("data/training/hwr200_hf"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/training/hwr200_page_alignment"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/hwr200_qwen_pages_full"))
    parser.add_argument("--model", type=Path, default=Path("models/trocr-cyrillic-mixed-epoch-2"))
    parser.add_argument(
        "--stage",
        choices=("inventory", "draft", "repair", "align", "export", "all"),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-authors", type=int)
    parser.add_argument("--author-ids", type=int, nargs="+")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--review-threshold", type=float, default=0.35)
    parser.add_argument("--repair-max-lines", type=int, default=60)
    parser.add_argument("--repair-min-lines", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def split_for_author(author_id: int) -> str:
    if author_id < 160:
        return "train"
    if author_id < 180:
        return "val"
    return "test"


def build_message(image_path: str, answer: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": PAGE_PROMPT},
                ],
            },
            {"role": "assistant", "content": answer},
        ]
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_annotations(path: Path) -> dict[tuple[int | None, str], str]:
    annotations = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = ANNOTATION_PATTERN.fullmatch(name)
            if not match:
                continue
            payload = json.loads(archive.read(name))
            text = payload.get("full_text", "").replace("\r\n", "\n").strip()
            if text:
                folder_id = int(match.group(1))
                document_name = match.group(2)
                annotations[(folder_id, document_name)] = text
                if document_name.startswith(("original", "fpr")):
                    annotations[(None, document_name)] = text
    return annotations


def build_inventory(source_dir: Path, output_path: Path) -> list[dict]:
    annotations = load_annotations(source_dir / "annotations.zip")
    documents: dict[tuple[int, str], dict] = {}
    archive_transcripts: dict[tuple[int, str], str] = {}
    for archive_path in sorted(source_dir.glob("hwr200_*.zip"), key=lambda path: natural_key(path.name)):
        if not re.fullmatch(r"hwr200_\d+_\d+\.zip", archive_path.name):
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                text_match = TEXT_MEMBER_PATTERN.fullmatch(name)
                if text_match:
                    document_name = text_match.group(2).strip()
                    text = archive.read(name).decode("utf-8-sig", errors="replace").replace("\r\n", "\n").strip()
                    if text:
                        archive_transcripts[(int(text_match.group(1)), document_name)] = text
                    continue
                match = MEMBER_PATTERN.fullmatch(name)
                if not match or Path(match.group(4)).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                author_id = int(match.group(1))
                document_name = match.group(2)
                condition = match.group(3)
                if condition not in CONDITIONS:
                    continue
                key = (author_id, document_name)
                row = documents.setdefault(
                    key,
                    {
                        "author_id": author_id,
                        "document_name": document_name,
                        "archive": archive_path.as_posix(),
                        "full_text": "",
                        "conditions": defaultdict(list),
                    },
                )
                row["conditions"][condition].append(name)

    rows = []
    for key in sorted(documents, key=lambda item: (item[0], natural_key(item[1]))):
        row = documents[key]
        row["full_text"] = (
            archive_transcripts.get(key)
            or annotations.get(key)
            or annotations.get((None, row["document_name"]), "")
        )
        row["conditions"] = {
            condition: sorted(row["conditions"].get(condition, []), key=natural_key)
            for condition in CONDITIONS
        }
        row["scan_pages"] = len(row["conditions"]["Сканы"])
        row["condition_page_counts_match"] = len(
            {len(row["conditions"][condition]) for condition in CONDITIONS}
        ) == 1
        row["has_annotation"] = bool(row["full_text"])
        rows.append(row)
    write_jsonl(output_path, rows)
    return rows


def filter_inventory(
    rows: list[dict],
    max_authors: int | None,
    max_documents: int | None,
    author_ids: list[int] | None = None,
) -> list[dict]:
    if author_ids is not None:
        allowed = set(author_ids)
        rows = [row for row in rows if row["author_id"] in allowed]
    if max_authors is not None:
        author_ids = sorted({row["author_id"] for row in rows})[:max_authors]
        allowed = set(author_ids)
        rows = [row for row in rows if row["author_id"] in allowed]
    if max_documents is not None:
        rows = rows[:max_documents]
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(int(window), len(values)))
    return np.convolve(values, np.ones(window, dtype=np.float64) / window, mode="same")


def estimate_line_period(row_projection: np.ndarray) -> int:
    height = len(row_projection)
    smoothed = moving_average(row_projection.astype(np.float64), 5)
    background_window = max(31, (height // 15) | 1)
    detrended = smoothed - moving_average(smoothed, background_window)
    detrended -= detrended.mean()
    minimum_lag = max(12, height // 150)
    maximum_lag = min(180, max(minimum_lag + 5, height // 5))
    scores = []
    for lag in range(minimum_lag, maximum_lag):
        left = detrended[:-lag]
        right = detrended[lag:]
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        scores.append(float(np.dot(left, right) / denominator) if denominator else 0.0)
    if not scores:
        return max(12, height // 40)
    maximum_score = max(scores)
    candidates = []
    for index in range(2, len(scores) - 2):
        if scores[index] >= max(scores[index - 2 : index + 3]) and scores[index] >= max(0.2, maximum_score * 0.65):
            candidates.append(minimum_lag + index)
    selected_period = candidates[0] if candidates else minimum_lag + int(np.argmax(scores))
    half_period = selected_period // 2
    if selected_period >= 40 and half_period >= minimum_lag:
        half_score = scores[half_period - minimum_lag]
        if half_score > 0.1:
            selected_period = half_period
    return selected_period


def line_period_confidence(row_projection: np.ndarray, period: int) -> float:
    if period <= 0 or period >= len(row_projection):
        return 0.0
    height = len(row_projection)
    smoothed = moving_average(row_projection.astype(np.float64), 5)
    background_window = max(31, (height // 15) | 1)
    detrended = smoothed - moving_average(smoothed, background_window)
    detrended -= detrended.mean()
    left = detrended[:-period]
    right = detrended[period:]
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def find_line_peaks(row_projection: np.ndarray, period: int) -> list[int]:
    smoothed = moving_average(row_projection.astype(np.float64), 5)
    radius = max(4, int(period * 0.45))
    candidates = []
    for index in range(radius, len(smoothed) - radius):
        if smoothed[index] < smoothed[index - 1] or smoothed[index] < smoothed[index + 1]:
            continue
        left_valley = float(np.min(smoothed[index - radius : index]))
        right_valley = float(np.min(smoothed[index + 1 : index + radius + 1]))
        prominence = float(smoothed[index] - (left_valley + right_valley) / 2)
        candidates.append((prominence, float(smoothed[index]), index))
    if not candidates:
        return []
    prominence_threshold = max(2.0, float(np.percentile([item[0] for item in candidates], 55)))
    ink_threshold = float(np.percentile(smoothed, 45))
    candidates = [
        item for item in candidates if item[0] >= prominence_threshold and item[1] >= ink_threshold
    ]
    selected = []
    minimum_distance = max(6, int(period * 0.65))
    for candidate in sorted(candidates, reverse=True):
        if all(abs(candidate[2] - existing[2]) >= minimum_distance for existing in selected):
            selected.append(candidate)
    edge_margin = max(4, period // 2)
    return sorted(
        item[2]
        for item in selected
        if edge_margin <= item[2] < len(smoothed) - edge_margin
    )


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(1.0, float(sigma))
    radius = max(2, int(sigma * 3))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(positions * positions) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def find_component_line_peaks(binary: np.ndarray, minimum_spacing_hint: int | None = None) -> list[int]:
    height, width = binary.shape
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(20, height * width // 100_000)
    minimum_component_height = max(5, round(height / 180))
    components = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if area < minimum_area:
            continue
        if component_height < minimum_component_height or component_height >= height * 0.08:
            continue
        components.append(
            (x, y, component_width, component_height, area, float(centroids[index, 1]))
        )
    if not components:
        return []

    median_height = float(np.median([component[3] for component in components]))
    center_histogram = np.zeros(height, dtype=np.float64)
    for _x, _y, component_width, _component_height, _area, center_y in components:
        center = min(height - 1, max(0, int(round(center_y))))
        center_histogram[center] += min(component_width, width * 0.25)
    smoothed = gaussian_smooth(center_histogram, max(2.0, median_height * 0.35))
    radius = max(4, int(median_height * 0.8))
    candidates = []
    for index in range(radius, height - radius):
        if smoothed[index] < smoothed[index - 1] or smoothed[index] < smoothed[index + 1]:
            continue
        left_valley = float(np.min(smoothed[index - radius : index]))
        right_valley = float(np.min(smoothed[index + 1 : index + radius + 1]))
        prominence = float(smoothed[index] - (left_valley + right_valley) / 2)
        candidates.append((prominence, float(smoothed[index]), index))
    if not candidates:
        return []

    prominence_threshold = max(
        float(np.percentile([candidate[0] for candidate in candidates], 40)),
        float(np.max(smoothed) * 0.02),
    )
    positive_values = smoothed[smoothed > 0]
    ink_threshold = float(np.percentile(positive_values, 25)) if positive_values.size else 0.0
    candidates = [
        candidate
        for candidate in candidates
        if candidate[0] >= prominence_threshold and candidate[1] >= ink_threshold
    ]
    selected = []
    minimum_distance = max(8, int(median_height * 1.1), minimum_spacing_hint or 0)
    for candidate in sorted(candidates, reverse=True):
        if all(abs(candidate[2] - existing[2]) >= minimum_distance for existing in selected):
            selected.append(candidate)

    peaks = []
    for _prominence, _value, peak in sorted(selected, key=lambda item: item[2]):
        nearby = [
            component
            for component in components
            if abs(component[5] - peak) <= median_height * 0.8
        ]
        if not nearby:
            continue
        horizontal_span = (
            max(component[0] + component[2] for component in nearby)
            - min(component[0] for component in nearby)
        ) / width
        if horizontal_span >= 0.12:
            peaks.append(peak)
    return peaks


def remove_page_guides(binary: np.ndarray) -> np.ndarray:
    """Remove long notebook rules before estimating handwriting line centers."""
    height, width = binary.shape
    horizontal_length = max(25, width // 20)
    vertical_length = max(25, height // 20)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horizontal_length, 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, vertical_length)
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    guides = cv2.bitwise_or(horizontal, vertical)
    line_segments = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, width // 12),
        minLineLength=max(40, width // 8),
        maxLineGap=max(8, width // 100),
    )
    if line_segments is not None:
        hough_guides = np.zeros_like(binary)
        thickness = max(2, min(height, width) // 700)
        for segment in line_segments[:, 0]:
            x1, y1, x2, y2 = (int(value) for value in segment)
            delta_x = abs(x2 - x1)
            delta_y = abs(y2 - y1)
            nearly_horizontal = delta_y <= max(3, round(delta_x * 0.05))
            nearly_vertical = delta_x <= max(3, round(delta_y * 0.05))
            if nearly_horizontal or nearly_vertical:
                cv2.line(hough_guides, (x1, y1), (x2, y2), 255, thickness)
        guides = cv2.bitwise_or(guides, hough_guides)
    return cv2.bitwise_and(binary, cv2.bitwise_not(guides))


def segment_page_lines(image: Image.Image) -> list[Image.Image]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    normalized = cv2.divide(gray, background, scale=255)
    binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    detection_binary = remove_page_guides(binary)
    row_projection = np.count_nonzero(detection_binary, axis=1).astype(np.float64)
    period = estimate_line_period(row_projection)
    confidence = line_period_confidence(row_projection, period)
    spacing_hint = round(period * 0.55) if confidence >= 0.6 else None
    peaks = find_component_line_peaks(detection_binary, spacing_hint)
    if not peaks:
        peaks = find_line_peaks(row_projection, period)
    if not peaks:
        return []

    if len(peaks) > 1:
        period = max(8, round(float(np.median(np.diff(peaks)))))

    lines = []
    for index, peak in enumerate(peaks):
        previous_peak = peaks[index - 1] if index else max(0, peak - period)
        next_peak = peaks[index + 1] if index + 1 < len(peaks) else min(binary.shape[0], peak + period)
        y_start = max(0, (previous_peak + peak) // 2)
        y_end = min(binary.shape[0], (peak + next_peak) // 2)
        if y_end - y_start < 6:
            continue
        band = detection_binary[y_start:y_end, :]
        column_projection = np.count_nonzero(band, axis=0)
        active_columns = np.flatnonzero(column_projection >= max(2, int((y_end - y_start) * 0.03)))
        if active_columns.size == 0:
            continue
        x_start = max(0, int(active_columns[0]) - max(8, binary.shape[1] // 120))
        x_end = min(binary.shape[1], int(active_columns[-1]) + 1 + max(8, binary.shape[1] // 120))
        y_start = max(0, y_start - max(5, binary.shape[0] // 250))
        y_end = min(binary.shape[0], y_end + max(5, binary.shape[0] // 250))
        crop = gray[y_start:y_end, x_start:x_end]
        if crop.shape[1] >= 20 and crop.shape[0] >= 8:
            lines.append(Image.fromarray(crop).convert("RGB"))
    return lines


class BatchedTrOCREngine:
    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(str(model_path), local_files_only=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(model_path), local_files_only=True, dtype=dtype
        ).to(self.device)
        self.model.eval()

    def recognize(self, images: list[Image.Image], batch_size: int) -> list[str]:
        results = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    inputs,
                    max_length=128,
                    num_beams=1,
                    do_sample=False,
                )
            results.extend(
                normalize_recognized_text(text)
                for text in self.processor.batch_decode(generated, skip_special_tokens=True)
            )
        return results


def build_drafts(
    inventory: list[dict],
    output_path: Path,
    model_path: Path,
    batch_size: int,
    resume: bool,
) -> list[dict]:
    existing = read_jsonl(output_path) if resume else []
    completed = {(row["author_id"], row["document_name"], row["page_index"]) for row in existing}
    mode = "a" if resume and output_path.exists() else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = BatchedTrOCREngine(model_path)
    processed_pages = len(existing)

    with output_path.open(mode, encoding="utf-8", newline="\n") as stream:
        current_archive_path = None
        current_archive = None
        try:
            for document in inventory:
                if not document["has_annotation"]:
                    continue
                pending = [
                    (index, member)
                    for index, member in enumerate(document["conditions"]["Сканы"], start=1)
                    if (document["author_id"], document["document_name"], index) not in completed
                ]
                if not pending:
                    continue
                archive_path = Path(document["archive"])
                if archive_path != current_archive_path:
                    if current_archive is not None:
                        current_archive.close()
                    current_archive = zipfile.ZipFile(archive_path)
                    current_archive_path = archive_path

                page_lines = []
                page_errors = {}
                all_line_images = []
                for page_index, member in pending:
                    try:
                        image = Image.open(io.BytesIO(current_archive.read(member)))
                        lines = segment_page_lines(image)
                        if not lines:
                            raise ValueError("line segmentation returned no lines")
                        page_lines.append((page_index, member, len(all_line_images), len(lines)))
                        all_line_images.extend(lines)
                    except Exception as exc:
                        page_errors[page_index] = str(exc)
                        page_lines.append((page_index, member, len(all_line_images), 0))

                recognized_lines = engine.recognize(all_line_images, batch_size) if all_line_images else []
                for page_index, member, offset, line_count in page_lines:
                    lines = recognized_lines[offset : offset + line_count]
                    row = {
                        "author_id": document["author_id"],
                        "document_name": document["document_name"],
                        "page_index": page_index,
                        "scan_member": member,
                        "line_count": line_count,
                        "line_drafts": lines,
                        "draft_text": "\n".join(text for text in lines if text),
                        "error": page_errors.get(page_index),
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    existing.append(row)
                    processed_pages += 1
                    if processed_pages % 25 == 0:
                        print(f"draft pages: {processed_pages}", flush=True)
        finally:
            if current_archive is not None:
                current_archive.close()
    return existing


def select_draft_repair_keys(
    inventory: list[dict],
    draft_rows: list[dict],
    max_lines: int,
    min_lines: int,
) -> set[tuple[int, str, int]]:
    page_counts = {
        (row["author_id"], row["document_name"]): int(row["scan_pages"])
        for row in inventory
    }
    selected = set()
    for row in draft_rows:
        document_key = (row["author_id"], row["document_name"])
        page_index = int(row["page_index"])
        line_count = int(row.get("line_count", 0))
        is_nonfinal_short_page = (
            line_count <= min_lines
            and page_index < page_counts.get(document_key, page_index)
        )
        if line_count > max_lines or is_nonfinal_short_page:
            selected.add((*document_key, page_index))
    return selected


def repair_drafts(
    inventory: list[dict],
    output_path: Path,
    model_path: Path,
    batch_size: int,
    max_lines: int,
    min_lines: int,
) -> tuple[list[dict], set[tuple[int, str, int]]]:
    existing = read_jsonl(output_path)
    backup_path = output_path.with_name(f"{output_path.stem}_before_guide_repair{output_path.suffix}")
    previous_rows = read_jsonl(backup_path)
    existing_by_key = {
        (row["author_id"], row["document_name"], row["page_index"]): row
        for row in existing
    }
    previously_repaired = {
        (row["author_id"], row["document_name"], row["page_index"])
        for row in previous_rows
        if existing_by_key.get((row["author_id"], row["document_name"], row["page_index"])) != row
    }
    detected_keys = select_draft_repair_keys(
        inventory,
        existing,
        max_lines=max_lines,
        min_lines=min_lines,
    )
    repair_keys = detected_keys - previously_repaired
    affected_keys = previously_repaired | repair_keys
    if not repair_keys:
        print(
            f"OCR draft repair: no pending pages; previously repaired: {len(previously_repaired)}",
            flush=True,
        )
        return existing, affected_keys

    if not backup_path.exists():
        shutil.copy2(output_path, backup_path)
    retained = [
        row
        for row in existing
        if (row["author_id"], row["document_name"], row["page_index"]) not in repair_keys
    ]
    write_jsonl(output_path, retained)
    print(f"OCR draft repair pages: {len(repair_keys)}", flush=True)
    repaired = build_drafts(
        inventory,
        output_path,
        model_path,
        batch_size,
        resume=True,
    )
    return repaired, affected_keys


def align_repaired_documents(
    inventory: list[dict],
    draft_rows: list[dict],
    existing_aligned: list[dict],
    repair_keys: set[tuple[int, str, int]],
    output_path: Path,
    review_path: Path,
    review_threshold: float,
) -> list[dict]:
    affected_documents = {(author_id, document_name) for author_id, document_name, _ in repair_keys}
    if not affected_documents:
        return existing_aligned
    selected_inventory = [
        row
        for row in inventory
        if (row["author_id"], row["document_name"]) in affected_documents
    ]
    temporary_output = output_path.with_name(f"{output_path.stem}_repair_tmp{output_path.suffix}")
    temporary_review = review_path.with_name(f"{review_path.stem}_repair_tmp{review_path.suffix}")
    repaired_rows = align_documents(
        selected_inventory,
        draft_rows,
        temporary_output,
        temporary_review,
        review_threshold,
    )
    updates = {
        (row["author_id"], row["document_name"], row["page_index"]): row
        for row in repaired_rows
    }
    merged = []
    seen = set()
    for row in existing_aligned:
        key = (row["author_id"], row["document_name"], row["page_index"])
        merged.append(updates.get(key, row))
        seen.add(key)
    merged.extend(row for key, row in updates.items() if key not in seen)
    write_jsonl(output_path, merged)
    write_jsonl(review_path, [row for row in merged if row.get("needs_review")])
    temporary_output.unlink(missing_ok=True)
    temporary_review.unlink(missing_ok=True)
    print(
        f"realigned documents: {len(affected_documents)}, pages: {len(repaired_rows)}",
        flush=True,
    )
    return merged


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    normalized = []
    source_positions = []
    previous_space = True
    for index, char in enumerate(text):
        if char.isalnum():
            normalized.append(char.lower())
            source_positions.append(index)
            previous_space = False
        elif char.isspace() and not previous_space:
            normalized.append(" ")
            source_positions.append(index)
            previous_space = True
    if normalized and normalized[-1] == " ":
        normalized.pop()
        source_positions.pop()
    return "".join(normalized), source_positions


def normalize_alignment_text(text: str) -> str:
    return normalize_with_map(text)[0]


def map_draft_position(position: int, blocks, draft_length: int, reference_length: int) -> int:
    nonempty = [block for block in blocks if block.size]
    for block in nonempty:
        if block.a <= position <= block.a + block.size:
            return block.b + min(position - block.a, block.size)
    previous = next((block for block in reversed(nonempty) if block.a + block.size < position), None)
    following = next((block for block in nonempty if block.a > position), None)
    if previous and following:
        left_a = previous.a + previous.size
        left_b = previous.b + previous.size
        ratio = (position - left_a) / max(1, following.a - left_a)
        return round(left_b + ratio * (following.b - left_b))
    if previous:
        return min(reference_length, previous.b + previous.size + position - previous.a - previous.size)
    if following:
        return max(0, following.b - (following.a - position))
    return round(position / max(1, draft_length) * reference_length)


def snap_boundary(text: str, position: int, lower: int, upper: int, radius: int = 80) -> int:
    position = max(lower + 1, min(position, upper - 1))
    candidates = [
        index
        for index in range(max(lower + 1, position - radius), min(upper, position + radius + 1))
        if text[index - 1].isspace() or text[index].isspace()
    ]
    return min(candidates, key=lambda index: abs(index - position)) if candidates else position


def split_reference_by_drafts(reference: str, drafts: list[str]) -> list[str]:
    if len(drafts) <= 1:
        return [reference.strip()]
    normalized_reference, reference_map = normalize_with_map(reference)
    normalized_drafts = [normalize_alignment_text(draft) for draft in drafts]
    typical_length = round(median([len(text) for text in normalized_drafts if text])) if any(normalized_drafts) else 100
    normalized_drafts = [text if text else "¤" * max(1, typical_length) for text in normalized_drafts]
    combined = "".join(normalized_drafts)
    positions = []
    cursor = 0
    for draft in normalized_drafts[:-1]:
        cursor += len(draft)
        positions.append(cursor)

    matcher = SequenceMatcher(None, combined, normalized_reference, autojunk=False)
    if matcher.ratio() < 0.35:
        mapped = [
            round(position / max(1, len(combined)) * len(normalized_reference))
            for position in positions
        ]
    else:
        mapped = [
            map_draft_position(position, matcher.get_matching_blocks(), len(combined), len(normalized_reference))
            for position in positions
        ]
    source_boundaries = [0]
    for normalized_position in mapped:
        if reference_map:
            source_position = reference_map[min(max(normalized_position, 0), len(reference_map) - 1)]
        else:
            source_position = round(normalized_position / max(1, len(normalized_reference)) * len(reference))
        source_boundaries.append(
            snap_boundary(reference, source_position, source_boundaries[-1], len(reference))
        )
    source_boundaries.append(len(reference))
    segments = [
        reference[source_boundaries[index] : source_boundaries[index + 1]].strip()
        for index in range(len(drafts))
    ]
    if all(normalize_alignment_text(segment) for segment in segments):
        return segments

    weights = [max(1, len(text)) for text in normalized_drafts]
    total_weight = sum(weights)
    fallback_boundaries = [0]
    consumed = 0
    for weight in weights[:-1]:
        consumed += weight
        estimated = round(consumed / total_weight * len(reference))
        fallback_boundaries.append(
            snap_boundary(reference, estimated, fallback_boundaries[-1], len(reference))
        )
    fallback_boundaries.append(len(reference))
    return [
        reference[fallback_boundaries[index] : fallback_boundaries[index + 1]].strip()
        for index in range(len(drafts))
    ]


def similarity(left: str, right: str) -> float:
    left_normalized = normalize_alignment_text(left)
    right_normalized = normalize_alignment_text(right)
    if not left_normalized and not right_normalized:
        return 1.0
    return SequenceMatcher(None, left_normalized, right_normalized, autojunk=False).ratio()


def align_documents(
    inventory: list[dict],
    draft_rows: list[dict],
    output_path: Path,
    review_path: Path,
    review_threshold: float,
) -> list[dict]:
    drafts_by_document = defaultdict(list)
    for row in draft_rows:
        drafts_by_document[(row["author_id"], row["document_name"])].append(row)

    aligned = []
    review = []
    for document in inventory:
        key = (document["author_id"], document["document_name"])
        pages = sorted(drafts_by_document.get(key, []), key=lambda row: row["page_index"])
        expected_pages = len(document["conditions"]["Сканы"])
        if not document["full_text"] or len(pages) != expected_pages:
            review.append({
                "author_id": key[0],
                "document_name": key[1],
                "reason": "missing annotation or OCR draft pages",
                "expected_pages": expected_pages,
                "draft_pages": len(pages),
            })
            continue
        page_segments = split_reference_by_drafts(document["full_text"], [page["draft_text"] for page in pages])
        document_score = similarity("\n".join(page["draft_text"] for page in pages), document["full_text"])
        for page, segment in zip(pages, page_segments):
            line_drafts = page["line_drafts"] or [page["draft_text"]]
            line_segments = split_reference_by_drafts(segment, line_drafts)
            line_target = "\n".join(part for part in line_segments if part).strip()
            target_text = (
                line_target
                if normalize_alignment_text(line_target) == normalize_alignment_text(segment)
                else segment
            )
            page_score = similarity(page["draft_text"], segment)
            row = {
                **page,
                "target_text": target_text,
                "page_confidence": round(page_score, 6),
                "document_confidence": round(document_score, 6),
                "needs_review": bool(page["error"] or page_score < review_threshold),
            }
            aligned.append(row)
            if row["needs_review"]:
                review.append(row)
    write_jsonl(output_path, aligned)
    write_jsonl(review_path, review)
    return aligned


def export_dataset(
    inventory: list[dict],
    aligned: list[dict],
    output_dir: Path,
) -> dict:
    labels = {
        (row["author_id"], row["document_name"], row["page_index"]): row
        for row in aligned
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    streams = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in ("train", "val", "test")
    }
    metadata_stream = (output_dir / "page_metadata.jsonl").open("w", encoding="utf-8", newline="\n")
    stats = {"train": 0, "val": 0, "test": 0, "missing_labels": 0, "condition_mismatches": 0, "needs_review": 0}
    current_archive_path = None
    current_archive = None
    try:
        for document in inventory:
            archive_path = Path(document["archive"])
            if archive_path != current_archive_path:
                if current_archive is not None:
                    current_archive.close()
                current_archive = zipfile.ZipFile(archive_path)
                current_archive_path = archive_path
            split = split_for_author(document["author_id"])
            scan_count = len(document["conditions"]["Сканы"])
            if not document["condition_page_counts_match"]:
                stats["condition_mismatches"] += 1
            for condition in CONDITIONS:
                members = document["conditions"][condition]
                if len(members) != scan_count:
                    continue
                for page_index, member in enumerate(members, start=1):
                    label = labels.get((document["author_id"], document["document_name"], page_index))
                    if not label:
                        stats["missing_labels"] += 1
                        continue
                    suffix = Path(member).suffix.lower()
                    image_name = (
                        f"author_{document['author_id']:03d}_{document['document_name']}_"
                        f"page_{page_index:03d}_{CONDITION_SLUGS[condition]}{suffix}"
                    )
                    image_path = images_dir / split / image_name
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    if not image_path.exists():
                        image_path.write_bytes(current_archive.read(member))
                    streams[split].write(
                        json.dumps(build_message(image_path.as_posix(), label["target_text"]), ensure_ascii=False) + "\n"
                    )
                    metadata = {
                        "image_path": image_path.as_posix(),
                        "split": split,
                        "author_id": document["author_id"],
                        "document_name": document["document_name"],
                        "page_index": page_index,
                        "condition": condition,
                        "source_member": member,
                        "page_confidence": label["page_confidence"],
                        "document_confidence": label["document_confidence"],
                        "needs_review": label["needs_review"],
                    }
                    metadata_stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    stats[split] += 1
                    stats["needs_review"] += int(label["needs_review"])
                    total = stats["train"] + stats["val"] + stats["test"]
                    if total % 500 == 0:
                        print(f"exported images: {total}", flush=True)
    finally:
        if current_archive is not None:
            current_archive.close()
        for stream in streams.values():
            stream.close()
        metadata_stream.close()
    (output_dir / "conversion_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.work_dir / "inventory.jsonl"
    drafts_path = args.work_dir / "page_ocr_drafts.jsonl"
    aligned_path = args.work_dir / "page_labels.jsonl"
    review_path = args.work_dir / "review_queue.jsonl"

    if args.stage in {"inventory", "all"} or not inventory_path.exists():
        inventory = build_inventory(args.source_dir, inventory_path)
        print(f"inventory documents: {len(inventory)}", flush=True)
    else:
        inventory = read_jsonl(inventory_path)
    inventory = filter_inventory(inventory, args.max_authors, args.max_documents, args.author_ids)

    if args.stage in {"draft", "all"}:
        drafts = build_drafts(
            inventory,
            drafts_path,
            args.model,
            args.batch_size,
            resume=not args.no_resume,
        )
        print(f"OCR draft pages: {len(drafts)}", flush=True)
    elif args.stage == "repair":
        drafts, repair_keys = repair_drafts(
            inventory,
            drafts_path,
            args.model,
            args.batch_size,
            max_lines=args.repair_max_lines,
            min_lines=args.repair_min_lines,
        )
        print(f"OCR draft pages after repair: {len(drafts)}", flush=True)
    else:
        drafts = read_jsonl(drafts_path)

    if args.stage in {"align", "all"}:
        aligned = align_documents(
            inventory,
            drafts,
            aligned_path,
            review_path,
            args.review_threshold,
        )
        print(f"aligned pages: {len(aligned)}", flush=True)
    elif args.stage == "repair":
        aligned = align_repaired_documents(
            inventory,
            drafts,
            read_jsonl(aligned_path),
            repair_keys,
            aligned_path,
            review_path,
            args.review_threshold,
        )
        print(f"aligned pages after repair: {len(aligned)}", flush=True)
    else:
        aligned = read_jsonl(aligned_path)

    if args.stage in {"export", "repair", "all"}:
        stats = export_dataset(inventory, aligned, args.output_dir)
        print(" ".join(f"{key}={value}" for key, value in stats.items()), flush=True)


if __name__ == "__main__":
    main()
