"""Pure post-processing for converting a segment timeline into a scheduler decision."""

from __future__ import annotations

import copy
import math
from typing import Any


class _TimestampValidationError(ValueError):
    pass


def _decision(status: str, reason: str, **values: Any) -> dict[str, Any]:
    result = {
        "status": status,
        "committed_subtasks": [],
        "provisional_tail": None,
        "next_start_time_sec": None,
        "reason": reason,
    }
    result.update(values)
    return result


def _executor_names(config: dict[str, Any]) -> list[str]:
    values = config.get("executors") or ["arm_1", "arm_2"]
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError("segment_loop.executors must be a non-empty list of executor ids")
    if len(set(values)) != len(values):
        raise ValueError("segment_loop.executors must not contain duplicates")
    return values


def _resolve_path(value: Any, field_path: str | None) -> Any:
    if field_path is None:
        return value
    current = value
    for part in str(field_path).split("."):
        if not part:
            raise KeyError("timeline_field contains an empty path component")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(f"timeline path component not found: {part!r}")
    return current


def _finite_time(value: Any, *, field: str, index: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"timeline[{index}].{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeline[{index}].{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"timeline[{index}].{field} must be finite")
    return number


def _time_fields(item: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    configured_start = config.get("start_time_field")
    configured_end = config.get("end_time_field")
    if configured_start or configured_end:
        return str(configured_start or "start_time_sec"), str(configured_end or "end_time_sec")
    if "start_time_sec" in item or "end_time_sec" in item:
        return "start_time_sec", "end_time_sec"
    return "start_time", "end_time"


def _stage_sampled_timestamps(stage_record: dict[str, Any]) -> list[float]:
    video_meta = stage_record.get("video_meta") or {}
    values: list[Any] = list(video_meta.get("sampled_timestamps") or [])
    for view_output in (video_meta.get("view_outputs") or {}).values():
        values.extend(view_output.get("sampled_timestamps") or [])
    timestamps: set[float] = set()
    for value in values:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp):
            timestamps.add(timestamp)
    return sorted(timestamps)


def _canonical_timestamp(
    value: float,
    sampled_timestamps: list[float],
    *,
    tolerance: float,
    field: str,
    index: int,
    episode_duration_sec: float,
    allow_episode_end: bool,
    allowed_right_boundary: float | None = None,
) -> float:
    if sampled_timestamps:
        nearest = min(sampled_timestamps, key=lambda timestamp: abs(timestamp - value))
        if abs(nearest - value) <= tolerance:
            return nearest
    if allow_episode_end and abs(value - episode_duration_sec) <= tolerance:
        return episode_duration_sec
    if allowed_right_boundary is not None and abs(value - allowed_right_boundary) <= tolerance:
        return allowed_right_boundary
    raise _TimestampValidationError(
        f"timeline[{index}].{field}={value} does not match any sampled timestamp"
    )


def _validated_timeline(
    timeline: Any,
    *,
    episode_duration_sec: float,
    config: dict[str, Any],
    sampled_timestamps: list[float] | None = None,
    timestamp_tolerance_sec: float = 0.0,
    allowed_right_boundary: float | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[float]]:
    if not isinstance(timeline, list):
        raise ValueError(f"timeline must be a list, got {type(timeline).__name__}")

    validated: list[tuple[float, float, dict[str, Any]]] = []
    for index, raw_item in enumerate(timeline):
        if not isinstance(raw_item, dict):
            raise ValueError(f"timeline[{index}] must be an object")
        start_field, end_field = _time_fields(raw_item, config)
        if start_field not in raw_item or end_field not in raw_item:
            raise ValueError(
                f"timeline[{index}] must contain {start_field!r} and {end_field!r}"
            )
        start = _finite_time(raw_item[start_field], field=start_field, index=index)
        end = _finite_time(raw_item[end_field], field=end_field, index=index)
        item = copy.deepcopy(raw_item)
        if sampled_timestamps is not None:
            start = _canonical_timestamp(
                start,
                sampled_timestamps,
                tolerance=timestamp_tolerance_sec,
                field=start_field,
                index=index,
                episode_duration_sec=episode_duration_sec,
                allow_episode_end=False,
            )
            end = _canonical_timestamp(
                end,
                sampled_timestamps,
                tolerance=timestamp_tolerance_sec,
                field=end_field,
                index=index,
                episode_duration_sec=episode_duration_sec,
                allow_episode_end=bool(config.get("allow_episode_end_boundary", True)),
                allowed_right_boundary=allowed_right_boundary,
            )
            item[start_field] = start
            item[end_field] = end
        if start < 0 or end < 0 or start > episode_duration_sec or end > episode_duration_sec:
            raise ValueError(
                f"timeline[{index}] range [{start}, {end}] is outside episode [0, {episode_duration_sec}]"
            )
        if end < start:
            raise ValueError(f"timeline[{index}] ends before it starts: [{start}, {end}]")
        validated.append((start, end, item))

    warnings: list[str] = []
    if any(validated[index][0] < validated[index - 1][0] for index in range(1, len(validated))):
        validated.sort(key=lambda value: (value[0], value[1]))
        warnings.append("timeline_sorted_by_start_time")
    return [item for _, _, item in validated], warnings, [start for start, _, _ in validated]


def _bimanual_segment_decision(
    pipeline_context: dict[str, Any],
    segment_window: dict[str, Any],
    episode_duration_sec: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate two independent arm timelines and choose the next overlap window."""
    nested = config.get("postprocessor")
    postprocessor_cfg = nested if isinstance(nested, dict) else {}
    validation_cfg = dict(config)
    validation_cfg.update(postprocessor_cfg)
    validation_cfg["start_time_field"] = str(postprocessor_cfg.get("start_time_field", "start_time"))
    validation_cfg["end_time_field"] = str(postprocessor_cfg.get("end_time_field", "end_time"))

    try:
        duration = float(episode_duration_sec)
        segment_start = float(segment_window["start_time_sec"])
        segment_end = float(segment_window["end_time_sec"])
        if not all(math.isfinite(value) for value in (duration, segment_start, segment_end)):
            raise ValueError("episode and segment times must be finite")
        executors = _executor_names(config)
        result_stage = str(config.get("result_stage") or "refinement")
        stage_record = pipeline_context["stages"][result_stage]
        stage_output = stage_record["output"]
        if not isinstance(stage_output, dict):
            raise ValueError(f"{result_stage} output must be a JSON object")
        raw_timelines = stage_output.get("executor_timelines")
        if not isinstance(raw_timelines, list):
            raise ValueError(f"{result_stage} output must contain executor_timelines")
        by_executor: dict[str, dict[str, Any]] = {}
        for timeline in raw_timelines:
            if not isinstance(timeline, dict) or not isinstance(timeline.get("executor"), str):
                raise ValueError("each executor timeline must be a JSON object with executor")
            executor = timeline["executor"]
            if executor in by_executor:
                raise ValueError(f"duplicate executor timeline: {executor}")
            by_executor[executor] = timeline
        if set(by_executor) != set(executors):
            raise ValueError(
                f"executor timelines must exactly match {executors}, got {list(by_executor)}"
            )

        require_samples = bool(postprocessor_cfg.get("require_sampled_timestamps", True))
        sampled_timestamps = _stage_sampled_timestamps(stage_record) if require_samples else None
        if require_samples and not sampled_timestamps:
            raise _TimestampValidationError("result stage has no sampled_timestamps metadata")
        tolerance = float(postprocessor_cfg.get("timestamp_tolerance_sec", 0.011))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("timestamp_tolerance_sec must be finite and >= 0")

        validated_timelines: list[dict[str, Any]] = []
        last_action_starts: dict[str, float] = {}
        action_counts: dict[str, int] = {}
        warnings: list[str] = []
        for executor in executors:
            annotation_start = float(
                segment_window.get(f"{executor}_annotation_start_time_sec", segment_start)
            )
            timeline = copy.deepcopy(by_executor[executor])
            actions, action_warnings, starts = _validated_timeline(
                timeline.get("actions"),
                episode_duration_sec=duration,
                config=validation_cfg,
                sampled_timestamps=sampled_timestamps,
                timestamp_tolerance_sec=tolerance,
                allowed_right_boundary=segment_end,
            )
            if not actions:
                raise ValueError(f"{executor}.actions must not be empty")
            start_field, end_field = _time_fields(actions[0], validation_cfg)
            first_start = float(actions[0][start_field])
            last_end = float(actions[-1][end_field])
            if not math.isclose(first_start, annotation_start, abs_tol=tolerance):
                raise ValueError(
                    f"{executor} must start at its annotation boundary {annotation_start}, got {first_start}"
                )
            if not math.isclose(last_end, segment_end, abs_tol=tolerance):
                raise ValueError(f"{executor} must end at segment boundary {segment_end}, got {last_end}")
            actions[0][start_field] = annotation_start
            actions[-1][end_field] = segment_end
            for action_index, (previous, current) in enumerate(zip(actions, actions[1:]), start=1):
                previous_end = float(previous[_time_fields(previous, validation_cfg)[1]])
                current_start = float(current[_time_fields(current, validation_cfg)[0]])
                if not math.isclose(previous_end, current_start, abs_tol=tolerance):
                    raise ValueError(
                        f"{executor}.actions has a gap or overlap before index {action_index}: "
                        f"{previous_end} != {current_start}"
                    )
                current[_time_fields(current, validation_cfg)[0]] = previous_end
            timeline["actions"] = actions
            validated_timelines.append(timeline)
            action_counts[executor] = len(actions)
            last_action_starts[executor] = float(starts[-1])
            warnings.extend(f"{executor}:{warning}" for warning in action_warnings)

        segment_output = copy.deepcopy(stage_output)
        segment_output["executor_timelines"] = validated_timelines
    except _TimestampValidationError as exc:
        return _decision("retry", "timestamp_not_sampled", error=str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        return _decision("retry", "invalid_executor_timelines", error=str(exc))

    if segment_end >= duration - 1e-9:
        return _decision(
            "finish",
            "episode_end",
            segment_output=segment_output,
            annotation_start_times={
                executor: float(segment_window.get(f"{executor}_annotation_start_time_sec", segment_start))
                for executor in executors
            },
            warnings=warnings,
        )

    try:
        min_advance = float(postprocessor_cfg.get("min_advance_sec", 0.25))
        fallback_overlap = float(postprocessor_cfg.get("fallback_overlap_sec", 2.0))
        if not math.isfinite(min_advance) or min_advance <= 0:
            raise ValueError("min_advance_sec must be finite and > 0")
        if not math.isfinite(fallback_overlap) or fallback_overlap < 0:
            raise ValueError("fallback_overlap_sec must be finite and >= 0")
    except (TypeError, ValueError) as exc:
        return _decision("fail", "invalid_postprocessor_config", error=str(exc), warnings=warnings)

    multi_subtask_executors = [
        executor for executor in executors if action_counts[executor] > 1
    ]
    if not multi_subtask_executors:
        next_start = segment_end
        next_annotation_starts = {executor: segment_end for executor in executors}
        reason = "both_executors_single_subtask_segment_end"
    elif len(multi_subtask_executors) == 1:
        boundary_executor = multi_subtask_executors[0]
        next_start = last_action_starts[boundary_executor]
        next_annotation_starts = {executor: next_start for executor in executors}
        reason = "single_subtask_executor_ignored"
    else:
        next_annotation_starts = dict(last_action_starts)
        next_start = min(last_action_starts.values())
        reason = "earliest_arm_last_action_start"

    if multi_subtask_executors and next_start - segment_start < min_advance:
        next_start = segment_end - fallback_overlap
        reason = "fallback_fixed_overlap"
        if sampled_timestamps:
            candidates = [
                timestamp
                for timestamp in sampled_timestamps
                if segment_start < timestamp < segment_end and timestamp < duration
            ]
            if candidates:
                next_start = min(candidates, key=lambda timestamp: abs(timestamp - next_start))
        if not (segment_start < next_start < segment_end and next_start < duration):
            return _decision("fail", "fallback_cannot_advance", warnings=warnings)
        next_annotation_starts = {executor: next_start for executor in executors}

    return _decision(
        "continue",
        reason,
        segment_output=segment_output,
        next_start_time_sec=next_start,
        next_annotation_start_times=next_annotation_starts,
        last_action_start_times=last_action_starts,
        action_counts=action_counts,
        warnings=warnings,
    )


def stitch_executor_timelines(
    accumulated_output: dict[str, Any] | None,
    segment_output: dict[str, Any],
    annotation_start_times: dict[str, float],
    *,
    executors: list[str] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Replace each arm's provisional tail with its newly predicted continuation."""
    expected = executors or ["arm_1", "arm_2"]
    previous = accumulated_output or {"executor_timelines": [], "uncertainties": []}
    previous_map = {
        timeline["executor"]: timeline
        for timeline in previous.get("executor_timelines") or []
        if isinstance(timeline, dict) and isinstance(timeline.get("executor"), str)
    }
    segment_map = {
        timeline["executor"]: timeline
        for timeline in segment_output.get("executor_timelines") or []
        if isinstance(timeline, dict) and isinstance(timeline.get("executor"), str)
    }
    if set(segment_map) != set(expected):
        raise ValueError(f"segment executor timelines must exactly match {expected}")

    timelines: list[dict[str, Any]] = []
    for executor in expected:
        boundary = float(annotation_start_times[executor])
        retained: list[dict[str, Any]] = []
        for action in (previous_map.get(executor) or {}).get("actions") or []:
            action_start = float(action["start_time"])
            action_end = float(action["end_time"])
            if action_end <= boundary + tolerance:
                retained.append(copy.deepcopy(action))
            elif action_start < boundary - tolerance < action_end:
                prefix = copy.deepcopy(action)
                prefix["end_time"] = boundary
                retained.append(prefix)
        incoming = copy.deepcopy(segment_map[executor].get("actions") or [])
        if not incoming:
            raise ValueError(f"{executor} segment actions must not be empty")
        if not math.isclose(float(incoming[0]["start_time"]), boundary, abs_tol=tolerance):
            raise ValueError(
                f"{executor} segment starts at {incoming[0]['start_time']}, expected {boundary}"
            )
        timelines.append({"executor": executor, "actions": retained + incoming})

    uncertainties = copy.deepcopy(previous.get("uncertainties") or [])
    uncertainties.extend(copy.deepcopy(segment_output.get("uncertainties") or []))
    return {"executor_timelines": timelines, "uncertainties": uncertainties}


def process_segment_result(
    pipeline_context: dict[str, Any],
    segment_window: dict[str, Any],
    episode_duration_sec: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the configured final timeline and return a scheduler-only decision."""
    nested_postprocessor = config.get("postprocessor")
    postprocessor_cfg = nested_postprocessor if isinstance(nested_postprocessor, dict) else config
    if postprocessor_cfg.get("policy") == "bimanual_last_action_start":
        return _bimanual_segment_decision(
            pipeline_context,
            segment_window,
            episode_duration_sec,
            config,
        )
    validation_cfg = dict(config)
    validation_cfg.update(postprocessor_cfg)
    if postprocessor_cfg.get("policy", "last_subtask_start") != "last_subtask_start":
        return _decision("fail", "unsupported_policy")

    sampled_timestamps: list[float] | None = None
    try:
        duration = float(episode_duration_sec)
        segment_start = float(segment_window["start_time_sec"])
        segment_end = float(segment_window["end_time_sec"])
        if not all(math.isfinite(value) for value in (duration, segment_start, segment_end)):
            raise ValueError("episode and segment times must be finite")
        if duration <= 0 or segment_start < 0 or segment_end <= segment_start or segment_end > duration:
            raise ValueError("invalid episode duration or segment window")

        result_stage = config["result_stage"]
        stage_record = pipeline_context["stages"][result_stage]
        stage_output = stage_record["output"]
        timeline = _resolve_path(stage_output, config.get("timeline_field"))
        require_sampled_timestamps = bool(postprocessor_cfg.get("require_sampled_timestamps", False))
        sampled_timestamps = _stage_sampled_timestamps(stage_record) if require_sampled_timestamps else None
        if require_sampled_timestamps and not sampled_timestamps:
            raise _TimestampValidationError("result stage has no sampled_timestamps metadata")
        timestamp_tolerance = float(postprocessor_cfg.get("timestamp_tolerance_sec", 0.011))
        if not math.isfinite(timestamp_tolerance) or timestamp_tolerance < 0:
            raise ValueError("timestamp_tolerance_sec must be finite and >= 0")
        timeline, warnings, start_times = _validated_timeline(
            timeline,
            episode_duration_sec=duration,
            config=validation_cfg,
            sampled_timestamps=sampled_timestamps,
            timestamp_tolerance_sec=timestamp_tolerance,
        )
    except _TimestampValidationError as exc:
        return _decision("retry", "timestamp_not_sampled", error=str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        return _decision("retry", "invalid_timeline", error=str(exc))

    if not timeline:
        return _decision("retry", "empty_timeline", warnings=warnings)

    if segment_end >= duration:
        return _decision(
            "finish",
            "episode_end",
            committed_subtasks=timeline,
            warnings=warnings,
        )

    provisional_tail = timeline[-1]
    candidate = start_times[-1]
    try:
        min_advance = float(postprocessor_cfg.get("min_advance_sec", 1.0))
        fallback_overlap = float(postprocessor_cfg.get("fallback_overlap_sec", 2.0))
        if not math.isfinite(min_advance) or min_advance <= 0:
            raise ValueError("min_advance_sec must be finite and > 0")
        if not math.isfinite(fallback_overlap) or fallback_overlap < 0:
            raise ValueError("fallback_overlap_sec must be finite and >= 0")
    except (TypeError, ValueError) as exc:
        return _decision("fail", "invalid_postprocessor_config", error=str(exc), warnings=warnings)

    reason = "last_subtask_start"
    next_start = candidate
    if candidate - segment_start < min_advance:
        next_start = segment_end - fallback_overlap
        reason = "fallback_fixed_overlap"
        if sampled_timestamps:
            fallback_timestamps = [
                timestamp
                for timestamp in sampled_timestamps
                if segment_start < timestamp < segment_end and timestamp < duration
            ]
            if fallback_timestamps:
                next_start = min(fallback_timestamps, key=lambda timestamp: abs(timestamp - next_start))
        if not (segment_start < next_start < segment_end and next_start < duration):
            return _decision(
                "fail",
                "fallback_cannot_advance",
                provisional_tail=provisional_tail,
                warnings=warnings,
            )

    return _decision(
        "continue",
        reason,
        committed_subtasks=timeline[:-1],
        provisional_tail=provisional_tail,
        next_start_time_sec=next_start,
        warnings=warnings,
    )
