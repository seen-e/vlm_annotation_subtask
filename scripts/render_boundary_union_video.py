"""Render the union of per-executor boundaries onto a video clip."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from render_executor_annotation_video import COLORS, _fit_text, _font


def _segment_times(segment: dict[str, Any]) -> tuple[float, float]:
    if "start_time" in segment and "end_time" in segment:
        return float(segment["start_time"]), float(segment["end_time"])
    return float(segment["start_time_hint"]), float(segment["end_time_hint"])


def _active_segment(
    segments: list[dict[str, Any]], timestamp: float
) -> tuple[int, dict[str, Any] | None]:
    for index, segment in enumerate(segments):
        start, end = _segment_times(segment)
        if start <= timestamp < end:
            return index, segment
    return -1, None


def _action_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(action["subtask_id"]): action
        for timeline in data.get("executor_timelines") or []
        for action in timeline.get("actions") or []
        if action.get("subtask_id")
    }


def _action_label(subtask_id: Any, actions: dict[str, dict[str, Any]]) -> str:
    if not subtask_id:
        return "未覆盖"
    action = actions.get(str(subtask_id), {})
    action_name = action.get("action") or "unknown"
    object_id = action.get("object") or "null"
    target_id = action.get("target")
    relation = str(object_id) if not target_id else f"{object_id} -> {target_id}"
    return f"{subtask_id} | {action_name} | {relation}"


def _boundary_label(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "无"
    return ", ".join(
        f"{source.get('executor')}:{source.get('boundary_type')}"
        for source in sources
    )


def _draw_panel(
    frame: np.ndarray,
    *,
    timestamp: float,
    duration: float,
    segments: list[dict[str, Any]],
    actions: dict[str, dict[str, Any]],
    video_height: int,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    title_font = _font(22)
    body_font = _font(15)
    small_font = _font(13)
    panel_top = video_height
    draw.rectangle((0, panel_top, width, height), fill=(24, 27, 32))

    segment_index, segment = _active_segment(segments, timestamp)
    draw.text(
        (16, panel_top + 9),
        f"双臂合并时间线  |  t={timestamp:05.2f}s",
        font=title_font,
        fill=(245, 245, 245),
    )
    if segment is None:
        segment_text = "当前区间：未覆盖"
        arm_1_text = "arm_1：未覆盖"
        arm_2_text = "arm_2：未覆盖"
        source_text = "起点边界来源：无"
    else:
        start, end = _segment_times(segment)
        segment_text = f"{segment.get('segment_id')}  [{start:.2f}, {end:.2f})s"
        covered = segment.get("covered_subtasks") or {
            executor: (segment.get(executor) or {}).get("subtask_id")
            for executor in ("arm_1", "arm_2")
        }
        arm_1 = segment.get("arm_1") or {}
        arm_2 = segment.get("arm_2") or {}
        arm_1_text = arm_1.get("description") or (
            "arm_1：" + _action_label(covered.get("arm_1"), actions)
        )
        arm_2_text = arm_2.get("description") or (
            "arm_2：" + _action_label(covered.get("arm_2"), actions)
        )
        if segment.get("segment_description"):
            source_text = "合并描述：" + str(segment["segment_description"])
        else:
            source_text = "起点边界来源：" + _boundary_label(
                segment.get("start_boundary_from") or []
            )

    draw.text((16, panel_top + 42), segment_text, font=body_font, fill=(241, 196, 15))
    draw.text(
        (16, panel_top + 70),
        _fit_text(draw, arm_1_text, body_font, width - 32),
        font=body_font,
        fill=(52, 152, 219),
    )
    draw.text(
        (16, panel_top + 98),
        _fit_text(draw, arm_2_text, body_font, width - 32),
        font=body_font,
        fill=(46, 204, 113),
    )
    draw.text(
        (16, panel_top + 126),
        _fit_text(draw, source_text, small_font, width - 32),
        font=small_font,
        fill=(205, 205, 205),
    )

    timeline_left = 18
    timeline_right = width - 18
    timeline_top = panel_top + 159
    timeline_bottom = timeline_top + 28
    timeline_width = timeline_right - timeline_left
    draw.rectangle(
        (timeline_left, timeline_top, timeline_right, timeline_bottom),
        fill=(70, 73, 79),
        outline=(180, 180, 180),
        width=1,
    )
    for index, item in enumerate(segments):
        item_start, item_end = _segment_times(item)
        start = max(0.0, min(duration, item_start))
        end = max(0.0, min(duration, item_end))
        x0 = timeline_left + round(timeline_width * start / duration)
        x1 = timeline_left + round(timeline_width * end / duration)
        if x1 > x0:
            color = COLORS[index % len(COLORS)]
            if index != segment_index:
                color = tuple(max(35, int(channel * 0.58)) for channel in color)
            draw.rectangle((x0, timeline_top + 1, x1, timeline_bottom - 1), fill=color)

    marker_x = timeline_left + round(timeline_width * min(timestamp, duration) / duration)
    draw.line((marker_x, timeline_top - 5, marker_x, timeline_bottom + 5), fill=(255, 255, 255), width=3)
    for tick in range(0, int(duration) + 1, 5):
        x = timeline_left + round(timeline_width * tick / duration)
        draw.text((x - 8, timeline_bottom + 7), f"{tick}s", font=small_font, fill=(205, 205, 205))

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(
    *,
    video_path: Path,
    analysis_path: Path,
    output_path: Path,
    duration: float,
    scale: int,
) -> None:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    segments = data.get("merged_timeline") or data.get("boundary_union")
    if not isinstance(segments, list) or not segments:
        raise ValueError("input must contain a non-empty merged_timeline or boundary_union list")
    actions = _action_index(data)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or source_width <= 0 or source_height <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")

    video_width = source_width * scale
    video_height = source_height * scale
    panel_height = 235
    output_height = video_height + panel_height
    max_frames = int(round(duration * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="boundary_union_annotation_") as temp_dir:
        intermediate = Path(temp_dir) / "annotated.avi"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (video_width, output_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("failed to create intermediate annotation video")

        frame_index = 0
        try:
            while frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
                canvas = np.zeros((output_height, video_width, 3), dtype=np.uint8)
                canvas[:video_height] = frame
                timestamp = frame_index / fps
                writer.write(
                    _draw_panel(
                        canvas,
                        timestamp=timestamp,
                        duration=duration,
                        segments=segments,
                        actions=actions,
                        video_height=video_height,
                    )
                )
                frame_index += 1
        finally:
            writer.release()
            capture.release()

        if frame_index == 0:
            raise RuntimeError(f"no frames decoded from video: {video_path}")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    render(
        video_path=args.video,
        analysis_path=args.analysis,
        output_path=args.output,
        duration=args.duration,
        scale=args.scale,
    )
    print(args.output)


if __name__ == "__main__":
    main()
