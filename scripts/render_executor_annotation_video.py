"""Render per-executor stage annotations onto a short video clip."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
COLORS = [
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (230, 126, 34),
    (155, 89, 182),
    (26, 188, 156),
    (231, 76, 60),
    (149, 165, 166),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    value = text
    while value and draw.textlength(value + suffix, font=font) > max_width:
        value = value[:-1]
    return value + suffix


def _action_times(action: dict[str, Any]) -> tuple[float, float]:
    if "start_time" in action and "end_time" in action:
        return float(action["start_time"]), float(action["end_time"])
    return float(action["start_time_hint"]), float(action["end_time_hint"])


def _active_action(actions: list[dict[str, Any]], timestamp: float) -> tuple[int, dict[str, Any] | None]:
    for index, action in enumerate(actions):
        start, end = _action_times(action)
        if start <= timestamp < end:
            return index, action
    return -1, None


def _draw_panel(
    frame: np.ndarray,
    *,
    timestamp: float,
    duration: float,
    executor: str,
    actions: list[dict[str, Any]],
    video_height: int,
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    title_font = _font(23)
    body_font = _font(18)
    small_font = _font(14)

    panel_top = video_height
    draw.rectangle((0, panel_top, width, height), fill=(24, 27, 32))
    action_index, action = _active_action(actions, timestamp)

    header = f"{executor}  |  t={timestamp:05.2f}s"
    draw.text((16, panel_top + 10), header, font=title_font, fill=(245, 245, 245))
    if action is None:
        detail = "当前区间：未覆盖"
        subtask = "Analysis 未输出对应 subtask"
        active_color = (110, 110, 110)
    else:
        action_name = str(action.get("action") or "unknown")
        object_id = action.get("object") or "null"
        target_id = action.get("target") or "null"
        start, end = _action_times(action)
        detail = f"动作：{action_name}    object：{object_id}    target：{target_id}"
        subtask = f"{action.get('subtask_id', 'subtask')}    [{start:.2f}, {end:.2f})s"
        active_color = COLORS[action_index % len(COLORS)]

    draw.text(
        (16, panel_top + 45),
        _fit_text(draw, detail, body_font, width - 32),
        font=body_font,
        fill=active_color,
    )
    draw.text(
        (16, panel_top + 73),
        _fit_text(draw, subtask, small_font, width - 32),
        font=small_font,
        fill=(215, 215, 215),
    )

    timeline_left = 18
    timeline_right = width - 18
    timeline_top = panel_top + 107
    timeline_bottom = timeline_top + 28
    draw.rectangle(
        (timeline_left, timeline_top, timeline_right, timeline_bottom),
        fill=(70, 73, 79),
        outline=(180, 180, 180),
        width=1,
    )
    timeline_width = timeline_right - timeline_left
    for index, item in enumerate(actions):
        item_start, item_end = _action_times(item)
        start = max(0.0, min(duration, item_start))
        end = max(0.0, min(duration, item_end))
        x0 = timeline_left + round(timeline_width * start / duration)
        x1 = timeline_left + round(timeline_width * end / duration)
        if x1 > x0:
            draw.rectangle((x0, timeline_top + 1, x1, timeline_bottom - 1), fill=COLORS[index % len(COLORS)])

    marker_x = timeline_left + round(timeline_width * min(timestamp, duration) / duration)
    draw.line((marker_x, timeline_top - 5, marker_x, timeline_bottom + 5), fill=(255, 255, 255), width=3)
    for tick in (0, 5, 10, 15):
        if tick > duration:
            continue
        x = timeline_left + round(timeline_width * tick / duration)
        label = f"{tick}s"
        draw.text((x - 8, timeline_bottom + 7), label, font=small_font, fill=(205, 205, 205))

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render(
    *,
    video_path: Path,
    analysis_path: Path,
    executor: str,
    output_path: Path,
    duration: float,
    scale: int,
) -> None:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    timelines = data.get("executor_timelines") or []
    timeline = next((item for item in timelines if item.get("executor") == executor), None)
    if timeline is None:
        raise ValueError(f"executor timeline not found: {executor}")
    actions = timeline.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"{executor}.actions must be a list")

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
    panel_height = 180
    output_height = video_height + panel_height
    max_frames = int(round(duration * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="executor_annotation_") as temp_dir:
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
                        executor=executor,
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
    parser.add_argument("--executor", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    render(
        video_path=args.video,
        analysis_path=args.analysis,
        executor=args.executor,
        output_path=args.output,
        duration=args.duration,
        scale=args.scale,
    )
    print(args.output)


if __name__ == "__main__":
    main()
