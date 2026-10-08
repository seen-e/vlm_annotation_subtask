"""Outer episode loop for repeatedly running the configured stage pipeline."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

from .frame_cache import cleanup_episode_frame_cache, prepare_episode_frame_cache
from .pipeline import run_pipeline
from .postprocessors.analysis_boundary_union import (
    postprocess_refinement_output,
    postprocess_refinement_timelines,
)
from .prompt_utils import resolve_executor_ids
from .result_io import make_run_dir
from .segment_postprocessor import process_segment_result, stitch_executor_timelines
from .video_process import normalize_video_input, read_video_info


class EpisodeRunnerError(RuntimeError):
    """Raised when a segmented episode cannot finish safely."""


def _atomic_save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EpisodeRunnerError(f"Expected a JSON object in {path}")
    return value


def is_episode_complete(
    run_dir: str | Path,
    *,
    require_merged_timeline: bool = True,
    allow_legacy_final_subtasks: bool = True,
) -> bool:
    output_path = Path(run_dir) / "output.json"
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            value = json.loads(output_path.read_text(encoding="utf-8"))
            timelines_complete = (
                isinstance(value, dict)
                and isinstance(value.get("executor_timelines"), list)
            )
            if not timelines_complete:
                return False
            return not require_merged_timeline or isinstance(value.get("merged_timeline"), list)
        except (OSError, json.JSONDecodeError):
            return False
    if not allow_legacy_final_subtasks:
        return False
    final_path = Path(run_dir) / "final_subtasks.json"
    if not final_path.exists() or final_path.stat().st_size == 0:
        return False
    try:
        return isinstance(json.loads(final_path.read_text(encoding="utf-8")), list)
    except (OSError, json.JSONDecodeError):
        return False


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _duration_from_metadata(episode_context: dict[str, Any]) -> float | None:
    input_context = episode_context.get("input") or {}
    sources = [input_context]
    episode_source = input_context.get("episode")
    if isinstance(episode_source, dict):
        sources.append(episode_source)

    for source in sources:
        duration = _positive_number(source.get("duration_sec"))
        if duration is not None:
            return duration
    for source in sources:
        timestamp_end = _positive_number(source.get("timestamp_end"))
        if timestamp_end is not None:
            return timestamp_end
    for source in sources:
        length = _positive_number(source.get("length"))
        fps = _positive_number(source.get("fps"))
        if length is not None and fps is not None:
            return length / fps
    return None


def _configured_view_names(config: dict[str, Any], video_path: Any) -> list[str] | None:
    if not isinstance(video_path, dict):
        return None
    selected: list[str] = []
    for stage_name in config.get("workflow") or []:
        stage_cfg = (config.get("stages") or {}).get(stage_name) or {}
        for view_name in (stage_cfg.get("video") or {}).get("view_names") or []:
            if view_name in video_path and view_name not in selected:
                selected.append(view_name)
    return selected or None


def _episode_duration(episode_context: dict[str, Any], config: dict[str, Any]) -> float:
    metadata_duration = _duration_from_metadata(episode_context)
    if metadata_duration is not None:
        return metadata_duration

    input_context = episode_context.get("input") or {}
    if "video_path" not in input_context:
        raise EpisodeRunnerError("context['input']['video_path'] is required to determine episode duration")
    video_path = input_context["video_path"]
    view_map, _ = normalize_video_input(video_path, view_names=_configured_view_names(config, video_path))
    durations = [float(read_video_info(path)["duration"]) for path in view_map.values()]
    if not durations:
        raise EpisodeRunnerError("No video views are available to determine episode duration")
    return min(durations)


def _segment_context(
    episode_context: dict[str, Any],
    segment_window: dict[str, Any],
    frame_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = copy.deepcopy(episode_context)
    context.pop("pipeline", None)
    context.pop("run_dir", None)
    context.pop("episode_loop", None)
    context["stages"] = {}
    context["segment"] = copy.deepcopy(segment_window)
    if frame_cache:
        context["frame_cache"] = copy.deepcopy(frame_cache)
    input_context = context.get("input")
    if isinstance(input_context, dict):
        input_context.pop("video_segments", None)
    return context


def _initial_progress(
    episode_context: dict[str, Any],
    duration: float,
    executors: list[str],
) -> dict[str, Any]:
    return {
        "episode_id": str((episode_context.get("input") or {}).get("video_id") or "episode"),
        "episode_duration_sec": duration,
        "next_segment_index": 0,
        "next_start_time_sec": 0.0,
        "previous_start_time_sec": None,
        "committed_subtasks": [],
        "provisional_tail": None,
        "annotation_start_times": {executor: 0.0 for executor in executors},
        "accumulated_output": {
            "executor_timelines": [
                {"executor": executor, "actions": []} for executor in executors
            ],
            "uncertainties": [],
        },
        "completed": False,
        "failed": False,
        "retry_count": 0,
        "visited_start_times": [],
        "last_decision": {},
    }


def _record_failure(progress: dict[str, Any], progress_path: Path, reason: str) -> None:
    progress["failed"] = True
    progress["completed"] = False
    progress["failure_reason"] = reason
    _atomic_save_json(progress, progress_path)


def run_episode_pipeline(
    episode_context: dict[str, Any],
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    run_name: str,
    dry_run: bool = False,
    skip_existing: bool = False,
    **existing_run_options: Any,
) -> dict[str, Any]:
    """Run every segment through a fresh full workflow until post-processing finishes."""
    segment_cfg = config.get("segment_loop") or {}
    if not bool(segment_cfg.get("enabled", False)):
        raise EpisodeRunnerError("run_episode_pipeline requires segment_loop.enabled=true")

    duration = _episode_duration(episode_context, config)
    window_duration = _positive_number(segment_cfg.get("window_duration_sec"))
    if window_duration is None:
        raise EpisodeRunnerError("segment_loop.window_duration_sec must be finite and > 0")

    postprocessor_cfg = segment_cfg.get("postprocessor") or {}
    bimanual_mode = postprocessor_cfg.get("policy") == "bimanual_last_action_start"
    merge_after_episode = bool(segment_cfg.get("merge_after_episode", True))
    try:
        executors = resolve_executor_ids(config)
    except ValueError as exc:
        raise EpisodeRunnerError(str(exc)) from exc
    run_dir = make_run_dir(output_dir, run_name=run_name, video_id=(episode_context.get("input") or {}).get("video_id"))
    progress_path = run_dir / "progress.json"
    final_path = run_dir / ("output.json" if bimanual_mode else "final_subtasks.json")
    if skip_existing and is_episode_complete(
        run_dir,
        require_merged_timeline=bimanual_mode and merge_after_episode,
        allow_legacy_final_subtasks=not bimanual_mode,
    ):
        result = copy.deepcopy(episode_context)
        result["run_dir"] = str(run_dir)
        final_value = json.loads(final_path.read_text(encoding="utf-8"))
        if bimanual_mode:
            result["output"] = final_value
        else:
            result["final_subtasks"] = final_value
        result["episode_loop"] = {"completed": True, "resumed_from_final": True}
        return result

    if skip_existing and progress_path.exists():
        progress = _load_json_object(progress_path)
        if progress.get("completed") and not is_episode_complete(
            run_dir,
            require_merged_timeline=bimanual_mode and merge_after_episode,
            allow_legacy_final_subtasks=not bimanual_mode,
        ):
            raise EpisodeRunnerError(f"Completed progress has no valid final result: {progress_path}")
    else:
        progress = _initial_progress(episode_context, duration, executors)
        _atomic_save_json(progress, progress_path)

    frame_cache_cfg = segment_cfg.get("frame_cache") or {}
    frame_cache_state: dict[str, Any] | None = None
    if bool(frame_cache_cfg.get("enabled", False)):
        try:
            input_context = episode_context.get("input") or {}
            frame_cache_state = prepare_episode_frame_cache(
                input_context.get("video_path"),
                config,
                run_dir / ".frame_cache",
                jpeg_quality=int(frame_cache_cfg.get("jpeg_quality", 95)),
                video_segments=input_context.get("video_segments"),
            )
            progress["frame_cache"] = {
                "root": frame_cache_state["root"],
                "reused": bool(frame_cache_state.get("reused", False)),
            }
            _atomic_save_json(progress, progress_path)
        except Exception as exc:
            reason = f"failed to prepare episode frame cache: {exc}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason) from exc

    current_start = float(progress.get("next_start_time_sec", 0.0))
    segment_index = int(progress.get("next_segment_index", 0))
    committed_subtasks = list(progress.get("committed_subtasks") or [])
    accumulated_output = copy.deepcopy(progress.get("accumulated_output") or {})
    annotation_start_times = {
        executor: float((progress.get("annotation_start_times") or {}).get(executor, current_start))
        for executor in executors
    }
    visited_starts = [float(value) for value in progress.get("visited_start_times") or []]
    retry_count = int(progress.get("retry_count", 0))
    max_retries = int(segment_cfg.get("max_retries_per_segment", 2))
    min_advance = _positive_number((segment_cfg.get("postprocessor") or {}).get("min_advance_sec")) or 1.0
    default_max_segments = max(1, int(math.ceil(duration / min_advance)) + 1)
    max_segments = int(segment_cfg.get("max_segments", default_max_segments))
    if max_retries < 0 or max_segments < 1:
        raise EpisodeRunnerError("segment retry and count limits must be non-negative")
    progress["max_segments"] = max_segments

    while segment_index < max_segments:
        if not (0 <= current_start < duration):
            reason = f"invalid segment start: {current_start}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason)
        if current_start not in visited_starts:
            visited_starts.append(current_start)

        segment_window = {
            "segment_index": segment_index,
            "start_time_sec": current_start,
            "end_time_sec": min(current_start + window_duration, duration),
            "episode_duration_sec": duration,
        }
        if bimanual_mode:
            for executor in executors:
                segment_window[f"{executor}_annotation_start_time_sec"] = annotation_start_times[executor]
        progress["current_start_time_sec"] = current_start
        context = _segment_context(episode_context, segment_window, frame_cache_state)
        segment_run_name = f"segment_{segment_index:04d}"
        context = run_pipeline(
            context,
            config,
            dry_run=dry_run,
            skip_existing=False,
            output_dir=run_dir / "segments",
            run_name=segment_run_name,
            save_results=True,
            video_segment_clip_storage=str(existing_run_options.get("video_segment_clip_storage") or "output-dir"),
        )
        decision = process_segment_result(context, segment_window, duration, segment_cfg)
        status = decision.get("status")
        progress["last_decision"] = decision
        progress["provisional_tail"] = decision.get("provisional_tail")
        progress["visited_start_times"] = visited_starts

        if status == "retry":
            retry_count += 1
            progress["retry_count"] = retry_count
            _atomic_save_json(progress, progress_path)
            if retry_count > max_retries:
                reason = f"segment {segment_index} exceeded retry limit: {decision.get('reason')}"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason)
            continue

        if status == "fail":
            reason = f"segment {segment_index} postprocessor failed: {decision.get('reason')}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason)

        if bimanual_mode:
            segment_output = decision.get("segment_output")
            if not isinstance(segment_output, dict):
                reason = "bimanual postprocessor returned no segment_output"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason)
            try:
                accumulated_output = stitch_executor_timelines(
                    accumulated_output,
                    segment_output,
                    annotation_start_times,
                    executors=executors,
                )
            except (KeyError, TypeError, ValueError) as exc:
                reason = f"failed to stitch segment {segment_index}: {exc}"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason) from exc
            progress["accumulated_output"] = copy.deepcopy(accumulated_output)

        if not bimanual_mode:
            new_committed = decision.get("committed_subtasks") or []
            if not isinstance(new_committed, list):
                reason = "postprocessor committed_subtasks must be a list"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason)
            committed_subtasks.extend(copy.deepcopy(new_committed))

        if status == "finish":
            if bimanual_mode:
                try:
                    if merge_after_episode:
                        final_value = postprocess_refinement_output(accumulated_output)
                    else:
                        final_value = postprocess_refinement_timelines(accumulated_output)
                except Exception as exc:
                    operation = "build final merged_timeline" if merge_after_episode else "finalize timelines"
                    reason = f"failed to {operation}: {exc}"
                    _record_failure(progress, progress_path, reason)
                    raise EpisodeRunnerError(reason) from exc
            else:
                final_value = committed_subtasks
            _atomic_save_json(final_value, final_path)
            progress.update(
                {
                    "next_segment_index": segment_index + 1,
                    "next_start_time_sec": None,
                    "committed_subtasks": committed_subtasks,
                    "accumulated_output": accumulated_output,
                    "annotation_start_times": annotation_start_times,
                    "merge_after_episode": merge_after_episode,
                    "provisional_tail": None,
                    "completed": True,
                    "failed": False,
                    "retry_count": 0,
                }
            )
            _atomic_save_json(progress, progress_path)
            if bool(frame_cache_cfg.get("cleanup_on_complete", False)):
                cleanup_episode_frame_cache(frame_cache_state)
            result = copy.deepcopy(episode_context)
            result["run_dir"] = str(run_dir)
            if bimanual_mode:
                result["output"] = final_value
            else:
                result["final_subtasks"] = final_value
            result["episode_loop"] = copy.deepcopy(progress)
            return result

        if status != "continue":
            reason = f"unsupported postprocessor status: {status!r}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason)

        try:
            next_start = float(decision["next_start_time_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            reason = "continue decision has no numeric next_start_time_sec"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason) from exc
        if not math.isfinite(next_start) or next_start <= current_start or next_start >= duration:
            reason = f"next segment start is not a valid monotonic advance: {current_start} -> {next_start}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason)
        if any(math.isclose(next_start, value, abs_tol=1e-9) for value in visited_starts):
            reason = f"next segment start was already visited: {next_start}"
            _record_failure(progress, progress_path, reason)
            raise EpisodeRunnerError(reason)

        previous_start = current_start
        current_start = next_start
        if bimanual_mode:
            raw_annotation_starts = decision.get("next_annotation_start_times")
            if not isinstance(raw_annotation_starts, dict):
                reason = "continue decision has no next_annotation_start_times"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason)
            try:
                annotation_start_times = {
                    executor: float(raw_annotation_starts[executor]) for executor in executors
                }
            except (KeyError, TypeError, ValueError) as exc:
                reason = "continue decision has invalid per-executor annotation starts"
                _record_failure(progress, progress_path, reason)
                raise EpisodeRunnerError(reason) from exc
        segment_index += 1
        retry_count = 0
        progress.update(
            {
                "next_segment_index": segment_index,
                "next_start_time_sec": current_start,
                "previous_start_time_sec": previous_start,
                "committed_subtasks": committed_subtasks,
                "accumulated_output": accumulated_output,
                "annotation_start_times": annotation_start_times,
                "completed": False,
                "failed": False,
                "retry_count": retry_count,
            }
        )
        _atomic_save_json(progress, progress_path)

    reason = f"segment count exceeded max_segments={max_segments}"
    _record_failure(progress, progress_path, reason)
    raise EpisodeRunnerError(reason)
