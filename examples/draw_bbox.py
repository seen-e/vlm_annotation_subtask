from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


DEFAULT_CONTEXT_ROOT = Path("examples/qwen3.5b_batch_predictions")
DEFAULT_PROCESSED_ROOT = Path("outputs/processed_frames")
DEFAULT_OUTPUT_DIR = Path("examples/qwen3.5b_bbox_visualizations")


COLORS = [
    (0, 80, 255),
    (0, 180, 80),
    (255, 120, 0),
    (220, 80, 220),
    (80, 180, 255),
    (80, 220, 220),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_id_from_context(context: dict[str, Any], context_path: Path) -> str:
    video_id = context.get("input", {}).get("video_id")
    if video_id:
        return str(video_id)

    match = re.search(r"(episode_\d+)", context_path.parent.name)
    if match:
        return match.group(1)

    raise ValueError(f"cannot infer episode_id from {context_path}")


def _iter_context_files(context_root: Path) -> list[Path]:
    if context_root.is_file():
        return [context_root]
    return sorted(context_root.glob("*/context.json"))


def _find_frame_in_dir(stage_dir: Path, sample_index: int) -> Path | None:
    stem = f"frame_{sample_index:06d}"
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = stage_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _find_frame(
    *,
    context_path: Path,
    context: dict[str, Any],
    processed_root: Path,
    episode_id: str,
    stage_name: str,
    sample_index: int,
) -> Path | None:
    stage_record = context.get("stages", {}).get(stage_name, {})
    video_meta = stage_record.get("video_meta", {}) if isinstance(stage_record, dict) else {}

    candidates: list[Path] = [
        context_path.parent / "stages" / stage_name / "processed_frames",
    ]
    if isinstance(video_meta, dict) and video_meta.get("processed_output_path"):
        candidates.append(Path(str(video_meta["processed_output_path"])))
    candidates.append(processed_root / episode_id / stage_name)

    for stage_dir in candidates:
        frame_path = _find_frame_in_dir(stage_dir, sample_index)
        if frame_path is not None:
            return frame_path
    return None


def _normalize_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    return x1, y1, x2, y2


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, round(x1 / 1000.0 * width))),
        max(0, min(height - 1, round(y1 / 1000.0 * height))),
        max(0, min(width - 1, round(x2 / 1000.0 * width))),
        max(0, min(height - 1, round(y2 / 1000.0 * height))),
    )


def _draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_h - baseline - 6)
    right = min(image.shape[1] - 1, x + text_w + 8)
    cv2.rectangle(image, (x, top), (right, top + text_h + baseline + 6), color, -1)
    cv2.putText(image, text, (x + 4, top + text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _collect_boxes(context: dict[str, Any], stage_name: str) -> dict[int, list[dict[str, Any]]]:
    stage = context.get("stages", {}).get(stage_name, {})
    output = stage.get("output", {})
    objects = output.get("interaction_objects", [])
    boxes_by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for obj in objects:
        obs = obj.get("first_observation")
        if not isinstance(obs, dict):
            continue
        sample_index = obs.get("sample_index")
        bbox = _normalize_bbox(obs.get("bbox_xyxy_normalized") or obs.get("bbox_xyxy"))
        if sample_index is None or bbox is None:
            continue
        try:
            sample_index_int = int(sample_index)
        except (TypeError, ValueError):
            continue
        boxes_by_sample[sample_index_int].append(
            {
                "object_id": str(obj.get("object_id", "unknown")),
                "category": str(obj.get("category", "")),
                "bbox": bbox,
                "confidence": obs.get("bbox_confidence"),
            }
        )

    return dict(boxes_by_sample)


def draw_context(
    context_path: Path,
    *,
    processed_root: Path,
    output_dir: Path,
    stage_name: str,
    draw_labels: bool,
) -> tuple[int, list[str]]:
    context = _read_json(context_path)
    episode_id = _episode_id_from_context(context, context_path)
    boxes_by_sample = _collect_boxes(context, stage_name)
    warnings: list[str] = []
    written = 0

    for sample_index, boxes in sorted(boxes_by_sample.items()):
        frame_path = _find_frame(
            context_path=context_path,
            context=context,
            processed_root=processed_root,
            episode_id=episode_id,
            stage_name=stage_name,
            sample_index=sample_index,
        )
        if frame_path is None:
            warnings.append(f"missing frame: episode={episode_id}, stage={stage_name}, sample_index={sample_index}")
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            warnings.append(f"failed to read image: {frame_path}")
            continue

        height, width = image.shape[:2]
        for idx, box in enumerate(boxes):
            color = COLORS[idx % len(COLORS)]
            x1, y1, x2, y2 = _bbox_to_pixels(box["bbox"], width, height)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if draw_labels:
                label = box["object_id"]
                if box["category"]:
                    label = f"{label} ({box['category']})"
                _draw_label(image, label, x1, y1, color)

        episode_dir = output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        out_path = episode_dir / f"{episode_id}_sample_{sample_index:06d}.jpg"
        cv2.imwrite(str(out_path), image)
        written += 1

    return written, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw interaction object bboxes on processed scene frames.")
    parser.add_argument("--contexts-root", type=Path, default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", default="scene")
    parser.add_argument("--draw-labels", action="store_true", help="Draw object_id/category labels above bboxes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context_files = _iter_context_files(args.contexts_root)
    if not context_files:
        raise FileNotFoundError(f"no context.json found under {args.contexts_root}")

    total_written = 0
    all_warnings: list[str] = []
    for context_path in context_files:
        written, warnings = draw_context(
            context_path,
            processed_root=args.processed_root,
            output_dir=args.output_dir,
            stage_name=args.stage,
            draw_labels=args.draw_labels,
        )
        total_written += written
        all_warnings.extend(warnings)

    print(f"processed contexts: {len(context_files)}")
    print(f"written images: {total_written}")
    print(f"output_dir: {args.output_dir.resolve()}")
    if all_warnings:
        print("warnings:")
        for warning in all_warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
