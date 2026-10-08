"""Run one configured lightweight VLM stage."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .json_utils import extract_json
from .model_client import call_vlm_with_metadata
from .prompt_utils import (
    _prompt_package_from_module,
    load_stage_prompt,
    render_template,
    resolve_executor_ids,
    resolve_input_fields,
    resolve_robot_type_prompt,
)
from .video_process import build_video_inputs


class StageRunnerError(RuntimeError):
    """Raised when a stage fails to run."""


_VIDEO_SAVE_KEYS = {"save_processed", "processed_output_path", "output_path", "save_processed_path"}
_EPISODE_FIELDS_KEYS = ("episode_fields", "episode_input_fields")


def _model_cfg(config: dict[str, Any], stage_cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model") or {})
    generation = dict(stage_cfg.get("generation") or {})
    model_cfg.update(generation)
    return model_cfg


def _dry_run_output(
    stage_name: str,
    context: dict[str, Any] | None = None,
    executors: list[str] | None = None,
) -> dict[str, Any]:
    segment = (context or {}).get("segment") or {}
    segment_start = float(segment.get("start_time_sec", 0.0))
    segment_end = float(segment.get("end_time_sec", max(segment_start + 1.0, 1.0)))
    active_executors = list(executors or ["arm_1", "arm_2"])

    def arm_start(executor: str) -> float:
        return float(segment.get(f"{executor}_annotation_start_time_sec", segment_start))

    if stage_name == "scene":
        return {
            "scene_summary": "dry_run scene output",
            "executors": [
                {"executor_id": executor, "description": f"dry-run {executor} executor"}
                for executor in active_executors
            ],
            "objects": [{"object_id": "object", "description": "dry-run object"}],
            "interaction_objects": [
                {
                    "object_id": "object",
                    "interaction_role": "被执行物体",
                    "interaction_reason": "dry-run interaction reason",
                    "contact_executors": ["arm_1"],
                }
            ],
            "touched_objects": [{"object_id": "object", "description": "dry-run object"}],
            "background_objects": [],
            "best_observation_views": [],
        }
    if stage_name == "analysis":
        return {
            "executor_timelines": [
                {
                    "executor": executor,
                    "actions": [
                        {
                            "start_time_hint": arm_start(executor),
                            "end_time_hint": segment_end,
                            "start_evidence": "dry-run start evidence",
                            "process_evidence": "dry-run process evidence",
                            "end_evidence": "dry-run end evidence",
                            "action": "等待",
                            "object": None,
                            "target": None,
                        }
                    ],
                }
                for executor in active_executors
            ],
            "uncertainties": [],
        }
    if stage_name.startswith("wrist_view"):
        return {
            "gripper_location": None,
            "object_categories": [],
            "objects": [],
            "interaction_objects": [],
            "gripper_state": [],
            "gripper_object_interactions": [],
        }
    if stage_name == "refinement":
        return {
            "executor_timelines": [
                {
                    "executor": executor,
                    "actions": [
                        {
                            "start_time": arm_start(executor),
                            "end_time": segment_end,
                            "start_boundary_evidence": "dry-run start evidence",
                            "end_boundary_evidence": "dry-run end evidence",
                            "action": "等待",
                            "object": None,
                            "target": None,
                        }
                    ],
                }
                for executor in active_executors
            ],
            "uncertainties": [],
        }
    if stage_name == "fusion":
        return {
            "robot_type": "dry_run",
            "fusion_summary": "dry_run fusion output",
            "executor_timelines": [
                {
                    "executor": "arm_1",
                    "actions": [
                        {
                            "source": {"analysis_action_indices": [], "wrist_evidence": []},
                            "fusion_status": "kept",
                            "start_time_hint": 0.0,
                            "end_time_hint": 1.0,
                            "local_observation": "dry-run local observation",
                            "evidence": "dry-run evidence",
                            "object": "object",
                            "target": None,
                            "action": "grasp",
                        }
                    ],
                    "dropped_actions": [],
                }
            ],
            "supplemented_actions": [],
            "added_actions": [],
            "conflicts": [],
            "uncertainties": [],
        }
    return {"stage": stage_name, "status": "dry_run_ok"}


def _ensure_context(context: dict[str, Any]) -> None:
    context.setdefault("input", {})
    context.setdefault("stages", {})
    if "video_path" not in context["input"]:
        raise StageRunnerError("context['input']['video_path'] is required")


def _resolve_dotted_field(source: Any, field_path: str) -> Any:
    if not field_path:
        raise StageRunnerError("episode field path must not be empty")
    parts = str(field_path).split(".")
    current = source
    index = 0
    while index < len(parts):
        if isinstance(current, dict):
            matched = False
            for end in range(len(parts), index, -1):
                key = ".".join(parts[index:end])
                if key in current:
                    current = current[key]
                    index = end
                    matched = True
                    break
            if not matched:
                missing = ".".join(parts[: index + 1])
                raise StageRunnerError(f"episode field path not found: {field_path!r}; missing {missing!r}")
        elif isinstance(current, (list, tuple)):
            try:
                list_index = int(parts[index])
            except ValueError as exc:
                raise StageRunnerError(
                    f"episode field path {field_path!r} expected list index, got {parts[index]!r}"
                ) from exc
            try:
                current = current[list_index]
            except IndexError as exc:
                raise StageRunnerError(
                    f"episode field path {field_path!r} list index out of range: {list_index}"
                ) from exc
            index += 1
        else:
            raise StageRunnerError(
                f"cannot resolve episode field {field_path!r} through {type(current).__name__}"
            )
    return current


def _assign_dotted_field(target: dict[str, Any], field_path: str, value: Any) -> None:
    parts = str(field_path).split(".")
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise StageRunnerError(f"episode field path conflict while assigning {field_path!r}")
        current = existing
    leaf = parts[-1]
    if leaf in current and isinstance(current[leaf], dict) and not isinstance(value, dict):
        raise StageRunnerError(f"episode field path conflict while assigning {field_path!r}")
    current[leaf] = copy.deepcopy(value)


def _episode_source(context: dict[str, Any]) -> dict[str, Any]:
    source = context.get("episode_source")
    if isinstance(source, dict):
        return source
    input_ctx = context.get("input") or {}
    source = input_ctx.get("episode") if isinstance(input_ctx, dict) else None
    if isinstance(source, dict):
        return source
    return input_ctx if isinstance(input_ctx, dict) else {}


def _stage_episode_fields(stage_cfg: dict[str, Any]) -> Any:
    for key in _EPISODE_FIELDS_KEYS:
        if key in stage_cfg:
            return stage_cfg.get(key)
    return None


def _prepare_episode_context(context: dict[str, Any], episode_fields: Any) -> dict[str, Any] | None:
    if episode_fields is None or episode_fields is False:
        return None
    source = _episode_source(context)
    if episode_fields is True or episode_fields == "*":
        return copy.deepcopy(source)
    if isinstance(episode_fields, str):
        episode_fields = [episode_fields]
    if not isinstance(episode_fields, list):
        raise StageRunnerError("episode_fields must be a list of field paths, '*', or true")

    episode_context: dict[str, Any] = {}
    for field_path in episode_fields:
        if not isinstance(field_path, str):
            raise StageRunnerError(f"episode_fields entries must be strings, got {type(field_path).__name__}")
        value = _resolve_dotted_field(source, field_path)
        _assign_dotted_field(episode_context, field_path, value)
    return episode_context


def _configured_robot_type(config: dict[str, Any]) -> str:
    robot_cfg = config.get("robot") or {}
    if isinstance(robot_cfg, dict) and robot_cfg.get("type"):
        return str(robot_cfg["type"])
    return str(config.get("robot_type") or "bimanual")


def _safe_path_name(value: Any, *, default: str) -> str:
    invalid = '<>:"/\\|?*'
    text = str(value or default)
    cleaned = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in text)
    return cleaned.strip(" .") or default


def _resolve_processed_output_path(
    *,
    stage_name: str,
    video_cfg: dict[str, Any],
    run_dir: str | Path | None,
) -> Path | None:
    if not bool(video_cfg.get("save_processed", False)):
        return None
    if run_dir is None:
        return None
    return Path(run_dir) / "stages" / _safe_path_name(stage_name, default="stage") / "processed_frames"


def _stage_dir(run_dir: str | Path | None, stage_name: str) -> Path:
    if run_dir is not None:
        return Path(run_dir) / "stages" / _safe_path_name(stage_name, default="stage")
    return Path(tempfile.mkdtemp(prefix=f"{_safe_path_name(stage_name, default='stage')}_"))


def _selected_view_names(video_cfg: dict[str, Any], video_path: Any, video_segments: dict[str, Any]) -> list[str]:
    configured = video_cfg.get("view_names")
    if configured:
        return [str(name) for name in configured]
    if isinstance(video_path, dict) and video_path:
        return [str(name) for name in video_path.keys()]
    return [str(name) for name in video_segments.keys()]


def _cut_video_segment_by_frame(
    *,
    source_path: str | Path,
    output_path: str | Path,
    start_frame: int,
    end_frame: int,
    fps: Any = None,
) -> None:
    if start_frame < 0:
        raise StageRunnerError(f"video segment start_frame must be >= 0, got {start_frame}")
    if end_frame < start_frame:
        raise StageRunnerError(f"video segment end_frame must be >= start_frame, got {start_frame}-{end_frame}")

    try:
        import ffmpeg
    except ImportError as exc:
        raise StageRunnerError("ffmpeg-python is required to cut video_segments; install package 'ffmpeg-python'") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = (
        ffmpeg
        .input(str(source_path))
        .filter("select", f"between(n,{start_frame},{end_frame})")
        .filter("setpts", "PTS-STARTPTS")
    )
    output_kwargs: dict[str, Any] = {
        "an": None,
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
    }
    if fps is not None:
        output_kwargs["r"] = fps
    try:
        (
            ffmpeg
            .output(stream, str(output), **output_kwargs)
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        detail = (stderr or "").strip()
        raise StageRunnerError(f"ffmpeg failed to cut video segment {source_path}: {detail}") from exc
    if not output.exists() or output.stat().st_size <= 0:
        raise StageRunnerError(f"ffmpeg did not create a valid video clip: {output}")


def _cut_video_segment_by_time(
    *,
    source_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    fps: Any = None,
) -> None:
    if start_time < 0:
        raise StageRunnerError(f"video segment start_time must be >= 0, got {start_time}")
    if end_time <= start_time:
        raise StageRunnerError(f"video segment end_time must be > start_time, got {start_time}-{end_time}")

    try:
        import ffmpeg
    except ImportError as exc:
        raise StageRunnerError("ffmpeg-python is required to cut video_segments; install package 'ffmpeg-python'") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = float(end_time) - float(start_time)
    stream = (
        ffmpeg
        .input(str(source_path), ss=float(start_time), t=duration)
        .filter("setpts", "PTS-STARTPTS")
    )
    output_kwargs: dict[str, Any] = {
        "an": None,
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
    }
    if fps is not None:
        output_kwargs["r"] = fps
    try:
        (
            ffmpeg
            .output(stream, str(output), **output_kwargs)
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        detail = (stderr or "").strip()
        raise StageRunnerError(f"ffmpeg failed to cut video segment {source_path}: {detail}") from exc
    if not output.exists() or output.stat().st_size <= 0:
        raise StageRunnerError(f"ffmpeg did not create a valid video clip: {output}")


def _is_minus_one(value: Any) -> bool:
    try:
        return float(value) == -1.0
    except (TypeError, ValueError):
        return False


def _has_segment_pair(segment: dict[str, Any], start_key: str, end_key: str) -> bool:
    return start_key in segment and end_key in segment


def _is_full_video_pair(segment: dict[str, Any], start_key: str, end_key: str) -> bool:
    return _has_segment_pair(segment, start_key, end_key) and _is_minus_one(segment[start_key]) and _is_minus_one(segment[end_key])


def _should_use_full_video(segment: dict[str, Any]) -> bool:
    if _has_segment_pair(segment, "start_time", "end_time"):
        return _is_full_video_pair(segment, "start_time", "end_time")
    if _has_segment_pair(segment, "start_frame", "end_frame"):
        return _is_full_video_pair(segment, "start_frame", "end_frame")
    return False


def _register_stage_video_clip_cleanup(
    context: dict[str, Any],
    *,
    clip_paths: list[Path],
    cleanup_dirs: list[Path],
) -> None:
    if not clip_paths and not cleanup_dirs:
        return
    pipeline_state = context.setdefault("pipeline", {})
    entries = pipeline_state.setdefault("temporary_video_segments", [])
    entries.append(
        {
            "clip_paths": [str(path) for path in clip_paths],
            "cleanup_dirs": [str(path) for path in cleanup_dirs],
        }
    )


def _video_segment_cache(context: dict[str, Any]) -> dict[str, Any]:
    pipeline_state = context.setdefault("pipeline", {})
    cache = pipeline_state.setdefault("video_segment_clip_cache", {})
    if not isinstance(cache, dict):
        cache = {}
        pipeline_state["video_segment_clip_cache"] = cache
    return cache


def _shared_clip_root(context: dict[str, Any], run_dir: str | Path | None) -> tuple[Path, bool]:
    pipeline_state = context.setdefault("pipeline", {})
    clip_storage = str(pipeline_state.get("video_segment_clip_storage") or "output-dir")
    if clip_storage == "output-dir" and run_dir is not None:
        return Path(run_dir) / "input_clips" / "video_segments", False

    root = pipeline_state.get("temporary_video_segment_clip_root")
    if root:
        return Path(root), False

    clip_root = Path(tempfile.mkdtemp(prefix="video_segment_input_clips_"))
    pipeline_state["temporary_video_segment_clip_root"] = str(clip_root)
    return clip_root, True


def _cache_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 9)
    return value


def _segment_clip_cache_key(
    *,
    source_path: str | Path,
    view_name: str,
    segment_mode: str,
    start_frame: Any,
    end_frame: Any,
    start_time: Any,
    end_time: Any,
    fps: Any,
) -> str:
    payload = {
        "source_path": str(source_path),
        "view_name": str(view_name),
        "segment_mode": str(segment_mode),
        "start_frame": _cache_value(start_frame),
        "end_frame": _cache_value(end_frame),
        "start_time": _cache_value(start_time),
        "end_time": _cache_value(end_time),
        "fps": _cache_value(fps),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _clip_filename(view_name: str, segment_mode: str, cache_key: str) -> str:
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    safe_view_name = _safe_path_name(view_name, default="view")
    return f"{safe_view_name}_{segment_mode}_{digest}.mp4"


def _prepare_stage_video_input(
    *,
    stage_name: str,
    context: dict[str, Any],
    video_cfg: dict[str, Any],
    run_dir: str | Path | None,
) -> tuple[Any, list[Path], list[Path], list[dict[str, Any]]]:
    video_segments = context.get("input", {}).get("video_segments")
    if not isinstance(video_segments, dict) or not video_segments:
        return context["input"]["video_path"], [], [], []

    source_video_path = context["input"]["video_path"]
    view_names = _selected_view_names(video_cfg, source_video_path, video_segments)
    missing = [name for name in view_names if name not in video_segments]
    if missing:
        raise StageRunnerError(
            f"stage {stage_name!r} requested view_names not present in input.video_segments: {missing}"
        )

    clip_paths: list[Path] = []
    cleanup_dirs: list[Path] = []
    stage_video_path: dict[str, str] = {}
    clip_meta: list[dict[str, Any]] = []
    clip_cache = _video_segment_cache(context)
    for view_name in view_names:
        segment = video_segments[view_name]
        if not isinstance(segment, dict):
            raise StageRunnerError(f"input.video_segments[{view_name!r}] must be a JSON object")
        source_path = segment.get("video_path")
        if source_path is None and isinstance(source_video_path, dict):
            source_path = source_video_path.get(view_name)
        elif source_path is None:
            source_path = source_video_path
        if source_path is None:
            raise StageRunnerError(f"input.video_segments[{view_name!r}] missing video_path")

        if _should_use_full_video(segment):
            stage_video_path[view_name] = str(source_path)
            clip_meta.append(
                {
                    "view_name": view_name,
                    "source_video_path": str(source_path),
                    "temporary_clip_path": None,
                    "start_time": segment.get("start_time"),
                    "end_time": segment.get("end_time"),
                    "start_frame": segment.get("start_frame"),
                    "end_frame": segment.get("end_frame"),
                    "fps": segment.get("fps"),
                    "used_full_video": True,
                    "segment_mode": "full",
                    "deleted_after_pipeline": False,
                }
            )
            continue

        segment_mode = ""
        start_frame = segment.get("start_frame")
        end_frame = segment.get("end_frame")
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        if _has_segment_pair(segment, "start_frame", "end_frame") and not _is_full_video_pair(
            segment, "start_frame", "end_frame"
        ):
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            segment_mode = "frame"
        elif _has_segment_pair(segment, "start_time", "end_time") and not _is_full_video_pair(
            segment, "start_time", "end_time"
        ):
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            segment_mode = "time"
        else:
            raise StageRunnerError(
                f"input.video_segments[{view_name!r}] must provide start_frame/end_frame or start_time/end_time"
            )

        cache_key = _segment_clip_cache_key(
            source_path=source_path,
            view_name=view_name,
            segment_mode=segment_mode,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            fps=segment.get("fps"),
        )
        cached = clip_cache.get(cache_key)
        reused_cached_clip = False
        if isinstance(cached, dict) and cached.get("clip_path") and Path(cached["clip_path"]).exists():
            clip_path = Path(cached["clip_path"])
            reused_cached_clip = True
        else:
            clip_root, cleanup_root = _shared_clip_root(context, run_dir)
            clip_path = clip_root / _clip_filename(view_name, segment_mode, cache_key)
            if segment_mode == "frame":
                _cut_video_segment_by_frame(
                    source_path=source_path,
                    output_path=clip_path,
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    fps=segment.get("fps"),
                )
            else:
                _cut_video_segment_by_time(
                    source_path=source_path,
                    output_path=clip_path,
                    start_time=float(start_time),
                    end_time=float(end_time),
                    fps=segment.get("fps"),
                )
            clip_cache[cache_key] = {
                "clip_path": str(clip_path),
                "source_video_path": str(source_path),
                "view_name": str(view_name),
                "segment_mode": segment_mode,
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": segment.get("fps"),
            }
            clip_paths.append(clip_path)
            if cleanup_root:
                cleanup_dirs.append(clip_root)

        stage_video_path[view_name] = str(clip_path)
        clip_meta.append(
            {
                "view_name": view_name,
                "source_video_path": str(source_path),
                "temporary_clip_path": str(clip_path),
                "segment_cache_key": cache_key,
                "reused_cached_clip": reused_cached_clip,
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": segment.get("fps"),
                "used_full_video": False,
                "segment_mode": segment_mode,
                "deleted_after_pipeline": True,
            }
        )

    return stage_video_path, clip_paths, cleanup_dirs, clip_meta


def _cleanup_stage_video_clips(clip_paths: list[Path], cleanup_dirs: list[Path]) -> None:
    for path in clip_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    for directory in sorted(set(cleanup_dirs), key=lambda path: len(path.parts), reverse=True):
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except Exception:
            pass


def cleanup_temporary_video_segments(context: dict[str, Any]) -> None:
    pipeline_state = context.get("pipeline")
    if not isinstance(pipeline_state, dict):
        return
    entries = pipeline_state.pop("temporary_video_segments", [])
    if not isinstance(entries, list):
        return
    if not entries:
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        clip_paths = [Path(path) for path in entry.get("clip_paths", [])]
        cleanup_dirs = [Path(path) for path in entry.get("cleanup_dirs", [])]
        _cleanup_stage_video_clips(clip_paths, cleanup_dirs)
    pipeline_state["temporary_video_segments_deleted"] = True


def _save_failure_debug(
    *,
    stage_name: str,
    run_dir: str | Path,
    exc: Exception,
    local_values: dict[str, Any],
) -> None:
    """Persist enough stage state to debug failures before normal result saving."""
    try:
        from .result_io import save_json, save_text

        stage_dir = Path(run_dir) / "stages" / stage_name
        error: dict[str, Any] = {
            "stage": stage_name,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }
        api_debug = getattr(exc, "debug_info", None)
        if api_debug:
            save_json(api_debug, stage_dir / "api_debug.json")
            error["api_debug_file"] = "api_debug.json"

        if "system_prompt" in local_values:
            save_text(local_values.get("system_prompt", ""), stage_dir / "system_prompt.txt")
        if "user_prompt" in local_values:
            save_text(local_values.get("user_prompt", ""), stage_dir / "user_prompt.txt")
        if "raw_text" in local_values:
            save_text(local_values.get("raw_text", ""), stage_dir / "raw_text.txt")
        if "video_meta" in local_values:
            save_json(local_values.get("video_meta", {}), stage_dir / "video_meta.json")
        if "current_video_layout" in local_values:
            save_text(local_values.get("current_video_layout", ""), stage_dir / "video_layout.txt")

        save_json(error, stage_dir / "error.json")
    except Exception:
        pass


def build_video_layout_description(video_cfg: dict[str, Any], video_meta: dict[str, Any]) -> str:
    """Build a prompt-facing description of the actual visual input layout."""
    view_names = [str(name) for name in video_meta.get("view_names") or video_cfg.get("view_names") or []]
    merge_views = bool(video_meta.get("merge_views", video_cfg.get("merge_views", False)))
    merge_mode = str(video_meta.get("merge_mode", video_cfg.get("merge_mode", "per_frame")))
    merge_length = int(video_meta.get("merge_length", video_cfg.get("merge_length", 0)) or 0)
    effective_merge_length = int(video_meta.get("effective_merge_length", 0) or 0)
    draw_timestamps = bool(video_meta.get("draw_timestamps", video_cfg.get("draw_timestamps", True)))
    draw_view_names = bool(video_meta.get("draw_view_names", video_cfg.get("draw_view_names", True)))
    draw_montage_axes = bool(video_meta.get("draw_montage_axes", video_cfg.get("draw_montage_axes", False)))
    sampled_count = video_meta.get("num_sampled_frames")
    output_count = video_meta.get("num_output_parts")
    primary_view = video_meta.get("primary_view")
    input_mode = str(video_meta.get("input_mode", video_cfg.get("input_mode", "image_sequence")))
    separate_view_inputs = bool(video_meta.get("separate_view_inputs", False))
    add_frame_tags = bool(video_meta.get("add_frame_tags", False))
    view_outputs = video_meta.get("view_outputs") or {}
    if merge_mode == "timeline_grid" and effective_merge_length <= 0:
        if merge_length < 1 and sampled_count is not None:
            effective_merge_length = int(sampled_count)
        else:
            effective_merge_length = merge_length

    lines = [
        "当前视频输入布局说明：",
        f"- 输入模式：{input_mode}。",
    ]
    try:
        effective_start = float(video_meta["effective_start_time"])
        effective_end = float(video_meta["effective_end_time"])
    except (KeyError, TypeError, ValueError):
        effective_start = None
        effective_end = None
    if effective_start is not None and effective_end is not None and effective_end > effective_start:
        lines.append(
            f"- 当前视觉输入的有效时间区间为 [{effective_start:.2f}s, {effective_end:.2f}s)。"
            "右端点是区间的结束边界，可能没有对应的采样图像。"
        )
    if sampled_count is not None and output_count is not None:
        media_name = "张图像" if input_mode == "image_sequence" else "个视频"
        if separate_view_inputs:
            sampled_summary = "、".join(
                f"{name}={output.get('num_sampled_frames', 0)}"
                for name, output in view_outputs.items()
            )
            lines.append(f"- 各视角采样点数：{sampled_summary}；最终送入模型 {output_count} {media_name}。")
        else:
            lines.append(f"- 共采样 {sampled_count} 个主时间点，最终送入模型 {output_count} {media_name}。")
    if view_names:
        lines.append(f"- 视角顺序：{', '.join(view_names)}。")
    if primary_view and not separate_view_inputs:
        lines.append(f"- 主时间轴基于视角 {primary_view}；其他视角按同一时间戳对齐。")
    elif primary_view:
        lines.append(f"- {primary_view} 仍是 primary_view，但每个视角使用各自原始时间轴独立采样。")

    if merge_views and len(view_names) > 1:
        lines.append("- 同一时间点的多个视角会按视角顺序自上而下拼接到同一张图中。")
        lines.append("- 上下相邻通常表示不同摄像机视角，不表示真实世界中物体一定上下相邻。")
    elif separate_view_inputs:
        lines.append("- 未启用多视角拼图；所有配置视角都会分别处理，不会因较短视角而截断其他视角。")
        lines.append("- message 中每组图像序列或视频前都有文字标签，明确标识其所属视角。")
    else:
        lines.append("- 每个采样时间点只有一个视角。")

    if merge_mode == "timeline_grid" and effective_merge_length > 1:
        if merge_length < 1:
            lines.append(
                "- timeline_grid 会将全部连续时间点合成为一个时间序列 montage；"
                "时间按从左到右的列方向推进。"
            )
        else:
            lines.append(
                f"- timeline_grid 会将每 {effective_merge_length} 个连续时间点合成为一个时间序列 montage；"
                "时间按从左到右的列方向推进。"
            )
        if merge_views and draw_montage_axes:
            lines.append("- montage 外侧会绘制坐标轴标签：顶部列标签表示时间戳，左侧行标签表示视角名称。")
    elif merge_mode == "timeline_grid":
        lines.append("- 当前配置选择 timeline_grid，但每个 montage 只包含一个时间点。")
    else:
        if input_mode == "video":
            lines.append("- per_frame 模式保留独立采样帧，并按时间顺序编码到视频中。")
        else:
            lines.append("- per_frame 模式未启用时间维度 montage；每张图像对应一个采样时间点。")

    if draw_timestamps:
        lines.append("- 图像上绘制了 t=...s 时间戳，动作顺序和边界判断应优先参考这些时间戳。")
    else:
        lines.append("- 图像上没有绘制时间戳，需要根据输入顺序和上下文估计时间推进。")
    if draw_view_names:
        lines.append("- 图像上绘制了视角名称，可用来区分不同摄像机来源。")
    elif separate_view_inputs:
        lines.append("- 媒体画面内没有绘制视角名称，但每组媒体前的 message 文字标签会标明视角来源。")
    elif len(view_names) > 1:
        lines.append("- 图像上没有绘制视角名称，需要按上述视角顺序理解不同视角来源。")
    if add_frame_tags:
        lines.append(
            "- 每张 image_sequence 图像前都有独立文字标签；<t=...s> 表示原始视频时间轴，"
            "<frame_index=...> 表示原始视频帧号，<view_name> 表示拍摄视角。"
        )

    return "\n".join(lines)


def _segment_time_contract(segment: dict[str, Any]) -> str:
    start = float(segment["start_time_sec"])
    end = float(segment["end_time_sec"])
    duration = segment.get("episode_duration_sec")
    duration_text = f"{float(duration):.6f}" if duration is not None else "unknown"
    return (
        "\n\n<segment_time_contract>\n"
        f"当前视觉输入只是完整 episode 的全局时间窗 [{start:.6f}, {end:.6f}) 秒；"
        f"完整 episode 时长为 {duration_text} 秒。\n"
        "画面标签中的时间戳是原 episode 的全局时间，禁止将当前片段重新从 0.0 开始计时。\n"
        "除每个 executor 最后一个 action 的结束时间必须等于其标注契约右端点外，"
        "所有输出时间字段必须等于当前实际发送图像标签中出现的某个时间戳，"
        "不得估算、插值或生成未出现的时间。\n"

        "Analysis 和 Refinement 中的每个 Scene executor 必须服从其对应的标注契约；"
        "某机械臂标注起点之前的画面只用于观察边界上下文，不得为该机械臂输出 action。\n"
        "</segment_time_contract>"
    )


def _annotation_contract(
    executor: str,
    *,
    requested_start: float,
    video_start: float,
    video_end: float,
    video_meta: dict[str, Any],
) -> str:
    timestamps = [float(value) for value in video_meta.get("sampled_timestamps") or []]
    frame_indices = [int(value) for value in video_meta.get("sampled_frame_indices") or []]
    if len(timestamps) != len(frame_indices) or not timestamps:
        raise StageRunnerError("annotation contracts require sampled timestamps and frame indices")
    nearest_index = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - requested_start))
    timestamp = timestamps[nearest_index]
    if abs(timestamp - requested_start) > 0.011:
        raise StageRunnerError(
            f"{executor} annotation start {requested_start} has no matching sampled frame"
        )
    frame_index = frame_indices[nearest_index]
    view_name = str(video_meta.get("primary_view") or (video_meta.get("view_names") or ["view"])[0])
    tag_time = f"{timestamp:.2f}"
    end_time = f"{video_end:.6f}"
    lines = [
        f"{executor}：",
        f"- 标注起始采样帧：<t={tag_time}s> <frame_index={frame_index}> <{view_name}>。",
        f"- 该机械臂必须输出的全视频时间区间为 [{tag_time}s, {end_time}s)。",
        f"- 第一个 action 的开始时间必须为 {tag_time}，并对应上述 frame_index={frame_index} 的图像。",
        f"- 最后一个 action 的结束时间必须为 {end_time}。",
    ]
    if requested_start > video_start + 1e-9:
        lines.append(
            f"- 视频窗开头 [{video_start:.6f}s, {tag_time}s) 仅作为该机械臂的边界上下文，"
            "不得输出该区间内的 action。"
        )
    return "\n".join(lines)


def _prompt_render_context(
    context: dict[str, Any],
    segment: dict[str, Any] | None,
    video_meta: dict[str, Any],
    executors: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(segment, dict):
        render_context = context
        prompt_segment = segment
        render_context["segment"] = prompt_segment
    else:
        render_context = copy.deepcopy(context)
        prompt_segment = {
            "start_time_sec": float(video_meta.get("effective_start_time", 0.0)),
            "end_time_sec": float(video_meta.get("effective_end_time", 0.0)),
        }
        render_context["segment"] = prompt_segment
    video_start = float(prompt_segment["start_time_sec"])
    video_end = float(prompt_segment["end_time_sec"])
    executor_contracts: list[str] = []
    for executor in executors or ["arm_1", "arm_2"]:
        requested_start = float(
            prompt_segment.get(f"{executor}_annotation_start_time_sec", video_start)
        )
        contract = _annotation_contract(
            executor,
            requested_start=requested_start,
            video_start=video_start,
            video_end=video_end,
            video_meta=video_meta,
        )
        prompt_segment[f"{executor}_annotation_contract"] = contract
        executor_contracts.append(f"{executor} 标注契约：\n{contract}")
    prompt_segment["executor_annotation_contracts"] = "\n\n".join(executor_contracts)
    return render_context


def _analysis_prompt_context(context: dict[str, Any]) -> dict[str, Any]:
    """Enrich Analysis objects while removing Scene arm-contact hints."""
    render_context = dict(context)
    stages = context.get("stages")
    if not isinstance(stages, dict):
        return render_context
    scene = stages.get("scene")
    if not isinstance(scene, dict):
        return render_context
    output = scene.get("output")
    if not isinstance(output, dict):
        return render_context
    interaction_objects = output.get("interaction_objects")
    if not isinstance(interaction_objects, list):
        return render_context

    object_fields = (
        "object_id",
        "category_id",
        "first_view_time",
        "category",
        "best_view_time",
        "description",
    )
    scene_objects = output.get("objects")
    objects_by_id = {
        str(item["object_id"]): item
        for item in scene_objects or []
        if isinstance(item, dict) and item.get("object_id") is not None
    }

    sanitized_interaction_objects: list[Any] = []
    for item in interaction_objects:
        if isinstance(item, dict):
            object_id = item.get("object_id")
            scene_object = objects_by_id.get(str(object_id)) if object_id is not None else None
            sanitized_item: dict[str, Any] = {}
            if isinstance(scene_object, dict):
                for field in object_fields:
                    if field in scene_object:
                        sanitized_item[field] = copy.deepcopy(scene_object[field])
            for field, value in item.items():
                if field != "contact_executors" and field not in sanitized_item:
                    sanitized_item[field] = copy.deepcopy(value)
            sanitized_interaction_objects.append(sanitized_item)
        else:
            sanitized_interaction_objects.append(copy.deepcopy(item))

    output_copy = dict(output)
    output_copy["interaction_objects"] = sanitized_interaction_objects
    scene_copy = dict(scene)
    scene_copy["output"] = output_copy
    stages_copy = dict(stages)
    stages_copy["scene"] = scene_copy
    render_context["stages"] = stages_copy
    return render_context


def run_stage(
    stage_name: str,
    context: dict[str, Any],
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    save_result: bool = False,
) -> dict[str, Any]:
    """Run one configured VLM stage and store its result in context.

    The stage behavior is fully defined by ``config['stages'][stage_name]``.
    Prompt placeholders can directly reference ``ctx.*`` and ``prompt.*``.
    """
    try:
        _ensure_context(context)
        stages_cfg = config.get("stages") or {}
        if stage_name not in stages_cfg:
            raise StageRunnerError(f"stage {stage_name!r} not found in config['stages']")
        stage_cfg = stages_cfg[stage_name]
        video_cfg = dict(stage_cfg.get("video") or {})
        if not video_cfg:
            raise StageRunnerError(f"stage {stage_name!r} missing video config")
        processed_output_path = _resolve_processed_output_path(
            stage_name=stage_name,
            video_cfg=video_cfg,
            run_dir=run_dir,
        )
        video_build_cfg = {key: value for key, value in video_cfg.items() if key not in _VIDEO_SAVE_KEYS}
        segment = context.get("segment")
        if isinstance(segment, dict):
            video_build_cfg["time_start_sec"] = segment.get("start_time_sec")
            video_build_cfg["time_end_sec"] = segment.get("end_time_sec")
            frame_cache = context.get("frame_cache")
            if isinstance(frame_cache, dict):
                manifest = (frame_cache.get("stage_manifests") or {}).get(stage_name)
                if manifest:
                    video_build_cfg["frame_cache_manifest"] = manifest
        if processed_output_path is not None:
            video_build_cfg["save_processed_path"] = processed_output_path

        stage_video_path, clip_paths, cleanup_dirs, clip_meta = _prepare_stage_video_input(
            stage_name=stage_name,
            context=context,
            video_cfg=video_cfg,
            run_dir=run_dir,
        )
        _register_stage_video_clip_cleanup(context, clip_paths=clip_paths, cleanup_dirs=cleanup_dirs)
        image_parts, video_meta = build_video_inputs(stage_video_path, **video_build_cfg)
        if clip_meta:
            video_meta["source_video_segments"] = clip_meta
        if processed_output_path is not None:
            video_meta["processed_output_path"] = str(processed_output_path)
        current_video_layout = build_video_layout_description(video_cfg, video_meta)
        context["current_video_layout"] = current_video_layout
        episode_context = _prepare_episode_context(context, _stage_episode_fields(stage_cfg))
        if episode_context is None:
            context.pop("episode", None)
        else:
            context["episode"] = episode_context
        extra_vars = resolve_input_fields(context, stage_cfg.get("input_fields"))
        system_template, user_template = load_stage_prompt(stage_cfg, stage_name=stage_name)
        prompt_cfg = stage_cfg.get("prompt") or {}
        prompt_module = str(prompt_cfg.get("module") or "") if isinstance(prompt_cfg, dict) else ""
        prompt_package = _prompt_package_from_module(prompt_module)
        robot_type = _configured_robot_type(config)
        context["robot_type"] = robot_type
        context["robot_type_prompt"] = resolve_robot_type_prompt(robot_type, prompt_package=prompt_package)
        executors = resolve_executor_ids(config)
        render_context = context
        if stage_name in {"analysis", "refinement"}:
            render_context = _prompt_render_context(
                context,
                segment,
                video_meta,
                executors=executors,
            )
        if stage_name == "analysis":
            render_context = _analysis_prompt_context(render_context)
        system_prompt = render_template(system_template, context=render_context, extra_vars=extra_vars, prompt_package=prompt_package)
        user_prompt = render_template(user_template, context=render_context, extra_vars=extra_vars, prompt_package=prompt_package)
        if isinstance(segment, dict):
            contract = _segment_time_contract(segment)
            system_prompt += contract
            user_prompt += contract

        usage: dict[str, Any] = {}
        if dry_run:
            parsed_json = _dry_run_output(stage_name, context, executors=executors)
            raw_text = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        else:
            model_cfg = _model_cfg(config, stage_cfg)
            result = call_vlm_with_metadata(
                base_url=str(model_cfg.get("base_url") or ""),
                api_key=str(model_cfg.get("api_key") or "EMPTY"),
                model=str(model_cfg.get("model") or model_cfg.get("name") or ""),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_parts=image_parts,
                max_tokens=int(model_cfg.get("max_tokens", stage_cfg.get("max_tokens", 4096)) or 4096),
                temperature=float(model_cfg.get("temperature", 0.0) or 0.0),
                top_p=model_cfg.get("top_p"),
                top_k=model_cfg.get("top_k"),
                timeout=model_cfg.get("timeout"),
                extra_body=model_cfg.get("extra_body"),
                max_retries=int(model_cfg.get("max_retries", 3) or 3),
            )
            raw_text = str(result.get("raw_text") or "")
            usage = dict(result.get("usage") or {})
            parsed_json = extract_json(raw_text)

        context["stages"][stage_name] = {
            "output": parsed_json,
            "raw_text": raw_text,
            "system_prompt": system_prompt,
            "prompt": user_prompt,
            "video_meta": video_meta,
            "video_layout": current_video_layout,
            "usage": usage,
        }
        if not dry_run:
            context["stages"][stage_name]["model"] = result.get("model")
            context["stages"][stage_name]["finish_reason"] = result.get("finish_reason")

        if save_result and run_dir is not None:
            from .result_io import save_stage_result

            save_stage_result(context, stage_name, run_dir)
        return parsed_json
    except StageRunnerError:
        raise
    except Exception as exc:
        if save_result and run_dir is not None:
            _save_failure_debug(stage_name=stage_name, run_dir=run_dir, exc=exc, local_values=locals().copy())
        raise StageRunnerError(f"stage {stage_name!r} failed: {exc}") from exc
