"""Build standardized trajectory summaries from refinement outputs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class TrajectorySummaryError(RuntimeError):
    """Raised when trajectory summary generation or saving fails."""


PHASE_MAPPING: dict[str, str] = {
    "接近": "approach",
    "靠近": "approach",
    "接触": "contact",
    "抓取": "grasp",
    "抓住": "grasp",
    "夹取": "grasp",
    "夹持": "grasp",
    "拿起": "lift",
    "提起": "lift",
    "搬运": "transport",
    "携带移动": "transport",
    "移动": "move",
    "移动机械臂": "move",
    "放置": "place",
    "释放": "release",
    "松开": "release",
    "撤回": "retract",
    "离开": "retract",
    "对齐": "align",
    "对准": "align",
    "旋转": "rotate",
    "插入": "insert",
    "拔出": "remove",
    "推动": "push",
    "拉动": "pull",
    "按压": "press",
    "打开": "open",
    "关闭": "close",
    "固定": "hold",
    "支撑": "support",
    "交接": "handover",
}

SUCCESS_VALUES = {"完成", "成功", "completed", "complete", "success", "succeeded"}
FAILED_VALUES = {"失败", "未完成", "failed", "failure", "incomplete"}
UNKNOWN_TASK_VALUES = {"", "unknown task", "unknown", "n/a", "na", "none", "null"}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_unknown_task(value: Any) -> bool:
    text = _clean_text(value)
    return text is None or text.lower() in UNKNOWN_TASK_VALUES


def _clamp(value: int, lower: int, upper: int) -> int:
    if upper < lower:
        upper = lower
    return max(lower, min(upper, value))


def _frame_bounds(video_meta: dict[str, Any], start_time: float, end_time: float) -> tuple[int, int]:
    fps = _to_float(video_meta.get("original_fps")) or 0.0
    start_frame = _to_int(video_meta.get("effective_frame_start"))
    end_frame = _to_int(video_meta.get("effective_frame_end"))
    if start_frame is None:
        start_frame = max(0, int(round(start_time * fps))) if fps > 0 else 0
    if end_frame is None:
        end_frame = max(start_frame, int(round(end_time * fps))) if fps > 0 else start_frame
    if end_frame < start_frame:
        end_frame = start_frame
    return start_frame, end_frame


def timestamp_to_original_frame(timestamp: Any, video_meta: dict[str, Any], mode: str = "nearest") -> int:
    """Convert an original-video timestamp to an original frame number."""
    if not isinstance(video_meta, dict):
        raise TrajectorySummaryError("video_meta must be a JSON object")
    ts = _to_float(timestamp)
    if ts is None:
        ts = 0.0

    start_time, end_time = _episode_time_bounds(video_meta)
    start_frame, end_frame = _frame_bounds(video_meta, start_time, end_time)
    sampled_timestamps = video_meta.get("sampled_timestamps")
    sampled_frame_indices = video_meta.get("sampled_frame_indices")

    frame: int | None = None
    if (
        isinstance(sampled_timestamps, list)
        and isinstance(sampled_frame_indices, list)
        and sampled_timestamps
        and len(sampled_timestamps) == len(sampled_frame_indices)
    ):
        best_index = min(
            range(len(sampled_timestamps)),
            key=lambda idx: abs((_to_float(sampled_timestamps[idx]) or 0.0) - ts),
        )
        frame = _to_int(sampled_frame_indices[best_index])

    if frame is None:
        fps = _to_float(video_meta.get("original_fps")) or 0.0
        frame = int(round(ts * fps)) if fps > 0 else start_frame

    return _clamp(frame, start_frame, end_frame)


def build_keyframes(start_time: float, end_time: float, video_meta: dict[str, Any]) -> list[int]:
    start_frame, end_frame = _frame_bounds(video_meta, start_time, end_time)
    middle_time = (start_time + end_time) / 2.0
    frames = [
        start_frame,
        timestamp_to_original_frame(middle_time, video_meta),
        end_frame,
    ]
    return sorted({_clamp(frame, start_frame, end_frame) for frame in frames})


def keyframes_from_segments(segments: dict[str, list[dict[str, Any]]]) -> list[int]:
    frames: set[int] = set()
    for executor_segments in segments.values():
        if not isinstance(executor_segments, list):
            continue
        for segment in executor_segments:
            if not isinstance(segment, dict):
                continue
            keyframes = segment.get("keyframes")
            if not isinstance(keyframes, list):
                continue
            for frame in keyframes:
                parsed = _to_int(frame)
                if parsed is not None:
                    frames.add(parsed)
    return sorted(frames)


def normalize_phase(action: Any) -> str:
    text = _clean_text(action)
    if text is None:
        return "unknown"
    if re.fullmatch(r"[A-Za-z0-9 _\-/]+", text):
        phase = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
        return phase or "unknown"
    return text


def build_caption(action: dict[str, Any], executor: Any = None) -> str | None:
    for key in ("caption", "description"):
        text = _clean_text(action.get(key))
        if text:
            return text

    executor_text = _clean_text(action.get("executor")) or _clean_text(executor)
    action_text = _clean_text(action.get("action"))
    object_text = _clean_text(action.get("object"))
    target_text = _clean_text(action.get("target"))
    caption = " ".join(part for part in [executor_text, action_text, object_text, target_text] if part)
    return caption or None


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def build_step_to_subtask_map(refinement_output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    subtasks = refinement_output.get("subtasks")
    if not isinstance(subtasks, list):
        return mapping
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            continue
        source_step_ids = subtask.get("source_step_ids")
        if not isinstance(source_step_ids, list):
            continue
        for step_id in source_step_ids:
            mapping[str(step_id)] = subtask
    return mapping


def _timeline_actions(refinement_output: dict[str, Any]) -> list[tuple[dict[str, Any], Any, str]]:
    """Return action-like records from old and current refinement schemas."""
    records: list[tuple[dict[str, Any], Any, str]] = []
    timelines = refinement_output.get("timed_executor_timelines")
    if not isinstance(timelines, list):
        timelines = refinement_output.get("executor_timelines")
    if isinstance(timelines, list):
        for timeline in timelines:
            if not isinstance(timeline, dict):
                continue
            executor = timeline.get("executor")
            for key, source in (("actions", "action"), ("supplemented_actions", "supplemented_action")):
                timeline_actions = timeline.get(key)
                if not isinstance(timeline_actions, list):
                    continue
                for action in timeline_actions:
                    if isinstance(action, dict):
                        records.append((action, executor, source))

    refined_segments = refinement_output.get("refined_segments")
    if isinstance(refined_segments, list):
        for action in refined_segments:
            if isinstance(action, dict):
                records.append((action, action.get("executor"), "refined_segment"))
    return records


def resolve_segment_success(
    action: dict[str, Any],
    step_to_subtask: dict[str, dict[str, Any]],
    refinement_output: dict[str, Any],
) -> int | None:
    step_id = action.get("step_id")
    subtask = step_to_subtask.get(str(step_id)) if step_id is not None else None
    status = subtask.get("completion_status") if isinstance(subtask, dict) else None
    value = _success_value(status)
    if value is not None:
        return value
    return _success_value(refinement_output.get("success"))


def _success_value(status: Any) -> int | None:
    text = _clean_text(status)
    if text is None:
        return None
    lowered = text.lower()
    if text in SUCCESS_VALUES or lowered in SUCCESS_VALUES:
        return 1
    if text in FAILED_VALUES or lowered in FAILED_VALUES:
        return 0
    return None


def build_segments(refinement_output: dict[str, Any], video_meta: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(refinement_output, dict):
        raise TrajectorySummaryError("refinement_output must be a JSON object")
    if not isinstance(video_meta, dict):
        raise TrajectorySummaryError("video_meta must be a JSON object")

    episode_start, episode_end = _episode_time_bounds(video_meta)
    step_to_subtask = build_step_to_subtask_map(refinement_output)
    actions: list[tuple[float, float, str, str, dict[str, Any], Any, str]] = []

    for action, executor, source in _timeline_actions(refinement_output):
        start = _to_float(action.get("start_time"))
        end = _to_float(action.get("end_time"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        clipped_start = max(episode_start, start)
        clipped_end = min(episode_end, end)
        if clipped_end < clipped_start:
            continue
        actions.append((clipped_start, clipped_end, str(executor or ""), str(action.get("step_id", "")), action, executor, source))

    actions.sort(key=lambda item: (item[2], item[0], item[1], item[3], item[6]))
    segments_by_executor: dict[str, list[dict[str, Any]]] = {}
    for start, end, executor_key, _step_id, action, executor, source in actions:
        start_frame = timestamp_to_original_frame(start, video_meta)
        end_frame = timestamp_to_original_frame(end, video_meta)
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        keyframes = sorted(
            {
                _clamp(frame, start_frame, end_frame)
                for frame in [
                    start_frame,
                    timestamp_to_original_frame((start + end) / 2.0, video_meta),
                    end_frame,
                ]
                }
            )
        executor_id = _clean_text(action.get("executor")) or _clean_text(executor) or "unknown"
        executor_segments = segments_by_executor.setdefault(executor_id, [])
        executor_segments.append(
            {
                "segment_id": len(executor_segments),
                "phase": normalize_phase(action.get("action")),
                "start_time": start,
                "end_time": end,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "keyframes": keyframes,
                "caption": build_caption(action, executor),
                "success": resolve_segment_success(action, step_to_subtask, refinement_output),
                "executor": executor_id,
                "object": _clean_text(action.get("object")),
                "target": _clean_text(action.get("target")),
                "evidence": _clean_text(action.get("evidence")),
                "source": source,
            }
        )
    return segments_by_executor


def build_trajectory(
    *,
    task: dict[str, Any],
    refinement_output: dict[str, Any],
    video_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build the final standardized trajectory summary from in-memory values."""
    if not isinstance(task, dict):
        raise TrajectorySummaryError("task must be a JSON object")
    if not isinstance(refinement_output, dict):
        raise TrajectorySummaryError("refinement_output must be a JSON object")
    if not isinstance(video_meta, dict):
        raise TrajectorySummaryError("video_meta must be a JSON object")

    start_time, end_time = _episode_time_bounds(video_meta)
    start_frame, end_frame = _frame_bounds(video_meta, start_time, end_time)
    l1_task = _resolve_l1_task(task, refinement_output)
    segments = build_segments(refinement_output, video_meta)
    keyframes = keyframes_from_segments(segments) or build_keyframes(start_time, end_time, video_meta)
    return {
        "episode_id": _clean_text(task.get("episode_id")),
        "robot_name": _clean_text(task.get("robot_name")),
        "original_dataset_id": _clean_text(task.get("original_dataset_id")),
        "duration": max(0.0, end_time - start_time),
        "start_time": start_time,
        "end_time": end_time,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "keyframes": keyframes,
        "L1_task": l1_task,
        "sense": _first_text(refinement_output, ["sense", "scene_description", "scene_summary", "overall_summary"]),
        "category": _first_text(task, ["category"]) or _first_text(refinement_output, ["category", "task_category"]),
        "segments": segments,
        "quality": _build_quality(refinement_output),
    }


def _episode_time_bounds(video_meta: dict[str, Any]) -> tuple[float, float]:
    start = _to_float(video_meta.get("effective_start_time"))
    end = _to_float(video_meta.get("effective_end_time"))
    if start is None:
        start = 0.0
    if end is None:
        duration = _to_float(video_meta.get("effective_duration"))
        if duration is None:
            duration = _to_float(video_meta.get("original_duration")) or 0.0
        end = start + duration
    if end < start:
        end = start
    return start, end


def _first_text(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        text = _clean_text(data.get(key))
        if text:
            return text
    return None


def _resolve_l1_task(task: dict[str, Any], refinement_output: dict[str, Any]) -> str | None:
    task_text = task.get("task") if not _is_unknown_task(task.get("task")) else None
    if task_text:
        return str(task_text).strip()
    text = _first_text(refinement_output, ["L1_task", "task_summary", "overall_task", "overall_summary"])
    if text:
        return text
    subtasks = refinement_output.get("subtasks")
    if isinstance(subtasks, list):
        descriptions = [
            _clean_text(subtask.get("description"))
            for subtask in subtasks
            if isinstance(subtask, dict) and _clean_text(subtask.get("description"))
        ]
        if descriptions:
            return "；".join(descriptions)
    return None


def _build_quality(refinement_output: dict[str, Any]) -> dict[str, Any]:
    quality = refinement_output.get("quality")
    if not isinstance(quality, dict):
        quality = {}
    return {
        "caption_video_match": _quality_float(quality.get("caption_video_match")),
        "object_presence_rate": _quality_float(quality.get("object_presence_rate")),
        "status": _quality_status(quality.get("status")),
    }


def _quality_float(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def _quality_status(value: Any) -> str:
    text = _clean_text(value)
    if text is None:
        return "UNKNOWN"
    upper = text.upper()
    return upper if upper in {"PASSED", "FAILED", "UNKNOWN"} else "UNKNOWN"


def resolve_summary_output_path(
    task: dict[str, Any],
    default_summary_path: str | Path,
    *,
    task_base_dir: str | Path | None = None,
) -> Path:
    trajectory_path = task.get("trajectory_path") if isinstance(task, dict) else None
    text = _clean_text(trajectory_path)
    path = Path(text) if text else Path(default_summary_path)
    if not path.is_absolute() and task_base_dir is not None:
        path = Path(task_base_dir) / path
    return path


def register_unique_output_path(
    output_path: str | Path,
    owner: str,
    used_paths: dict[str, str],
) -> None:
    """Record one output path and fail if another episode already claimed it."""
    path = Path(output_path)
    key = str(path.resolve()).casefold()
    if key in used_paths:
        raise TrajectorySummaryError(
            f"trajectory output path conflict: {path} is already used by {used_paths[key]}"
        )
    used_paths[key] = owner


def save_trajectory(trajectory: dict[str, Any], output_path: str | Path) -> None:
    save_json_atomic(output_path, trajectory)


def save_json_atomic(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=str(output_path.parent),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(data, tmp_file, ensure_ascii=False, indent=2, default=str)
                tmp_file.write("\n")
            Path(tmp_name).replace(output_path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except Exception as exc:
        raise TrajectorySummaryError(f"Failed to save trajectory to {output_path}: {exc}") from exc
