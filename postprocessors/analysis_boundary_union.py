"""Build a mechanical union of per-executor action boundaries."""

from __future__ import annotations

import copy
import math
from typing import Any


class AnalysisBoundaryUnionError(ValueError):
    """Raised when an Analysis timeline cannot be post-processed safely."""


def _finite_time(value: Any, *, field: str, executor: str, action_index: int) -> float:
    if isinstance(value, bool):
        raise AnalysisBoundaryUnionError(
            f"{executor}.actions[{action_index}].{field} must be numeric"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisBoundaryUnionError(
            f"{executor}.actions[{action_index}].{field} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise AnalysisBoundaryUnionError(
            f"{executor}.actions[{action_index}].{field} must be finite"
        )
    return number


def _subtask_id(executor: str, action: dict[str, Any], action_index: int) -> str:
    existing = action.get("subtask_id")
    if existing is not None:
        if not isinstance(existing, str) or not existing.strip():
            raise AnalysisBoundaryUnionError(
                f"{executor}.actions[{action_index}].subtask_id must be a non-empty string"
            )
        return existing
    return f"{executor}_subtask_{action_index + 1:04d}"


def _postprocess_output(
    stage_output: dict[str, Any],
    *,
    start_field: str,
    end_field: str,
    stage_name: str,
) -> dict[str, Any]:
    """Return stage output plus stable subtask IDs and a boundary-union timeline.

    The function is deliberately semantic-free: it copies stage actions without
    changing their action/object/target fields and only derives boundary provenance.
    """
    if not isinstance(stage_output, dict):
        raise AnalysisBoundaryUnionError(f"{stage_name} output must be a JSON object")

    result = copy.deepcopy(stage_output)
    timelines = result.get("executor_timelines")
    if not isinstance(timelines, list):
        raise AnalysisBoundaryUnionError(
            f"{stage_name} output must contain executor_timelines list"
        )

    seen_executors: set[str] = set()
    seen_subtask_ids: set[str] = set()
    boundary_sources: dict[float, list[dict[str, str]]] = {}
    executor_actions: dict[str, list[tuple[float, float, str]]] = {}

    for timeline_index, timeline in enumerate(timelines):
        if not isinstance(timeline, dict):
            raise AnalysisBoundaryUnionError(
                f"executor_timelines[{timeline_index}] must be a JSON object"
            )
        executor = timeline.get("executor")
        if not isinstance(executor, str) or not executor.strip():
            raise AnalysisBoundaryUnionError(
                f"executor_timelines[{timeline_index}].executor must be a non-empty string"
            )
        if executor in seen_executors:
            raise AnalysisBoundaryUnionError(f"duplicate executor timeline: {executor}")
        seen_executors.add(executor)

        actions = timeline.get("actions")
        if not isinstance(actions, list):
            raise AnalysisBoundaryUnionError(f"{executor}.actions must be a list")

        indexed_actions: list[tuple[float, float, str]] = []
        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise AnalysisBoundaryUnionError(
                    f"{executor}.actions[{action_index}] must be a JSON object"
                )
            if start_field not in action or end_field not in action:
                raise AnalysisBoundaryUnionError(
                    f"{executor}.actions[{action_index}] must contain {start_field} and {end_field}"
                )
            start = _finite_time(
                action[start_field],
                field=start_field,
                executor=executor,
                action_index=action_index,
            )
            end = _finite_time(
                action[end_field],
                field=end_field,
                executor=executor,
                action_index=action_index,
            )
            if start < 0 or end <= start:
                raise AnalysisBoundaryUnionError(
                    f"{executor}.actions[{action_index}] has invalid interval [{start}, {end}]"
                )

            subtask_id = _subtask_id(executor, action, action_index)
            if subtask_id in seen_subtask_ids:
                raise AnalysisBoundaryUnionError(f"duplicate subtask_id: {subtask_id}")
            seen_subtask_ids.add(subtask_id)
            action["subtask_id"] = subtask_id
            indexed_actions.append((start, end, subtask_id))

            boundary_sources.setdefault(start, []).append(
                {"executor": executor, "subtask_id": subtask_id, "boundary_type": "start"}
            )
            boundary_sources.setdefault(end, []).append(
                {"executor": executor, "subtask_id": subtask_id, "boundary_type": "end"}
            )

        ordered_actions = sorted(indexed_actions, key=lambda item: (item[0], item[1]))
        for previous, current in zip(ordered_actions, ordered_actions[1:]):
            if current[0] < previous[1]:
                raise AnalysisBoundaryUnionError(
                    f"{executor} has overlapping subtasks {previous[2]} and {current[2]}"
                )
        executor_actions[executor] = ordered_actions

    boundary_times = sorted(boundary_sources)
    boundary_union: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(zip(boundary_times, boundary_times[1:]), start=1):
        covered_subtasks: dict[str, str | None] = {}
        for executor, actions in executor_actions.items():
            covered = [
                subtask_id
                for action_start, action_end, subtask_id in actions
                if action_start <= start and action_end >= end
            ]
            if len(covered) > 1:
                raise AnalysisBoundaryUnionError(
                    f"{executor} has multiple subtasks covering interval [{start}, {end}]"
                )
            covered_subtasks[executor] = covered[0] if covered else None

        boundary_union.append(
            {
                "segment_id": f"segment_{segment_index:04d}",
                start_field: start,
                end_field: end,
                "start_boundary_from": copy.deepcopy(boundary_sources[start]),
                "end_boundary_from": copy.deepcopy(boundary_sources[end]),
                "covered_subtasks": covered_subtasks,
            }
        )

    result["boundary_union"] = boundary_union
    return result


def postprocess_analysis_output(analysis_output: dict[str, Any]) -> dict[str, Any]:
    """Legacy helper for Analysis-style ``*_time_hint`` boundaries."""
    return _postprocess_output(
        analysis_output,
        start_field="start_time_hint",
        end_field="end_time_hint",
        stage_name="Analysis",
    )


def _action_phrase(action: dict[str, Any]) -> str:
    action_name = str(action.get("action") or "未知动作")
    object_id = action.get("object")
    target_id = action.get("target")
    direction = action.get("direction")

    phrase = action_name
    if object_id:
        phrase += f" {object_id}"
    if target_id:
        phrase += f"，目标为 {target_id}"
    if isinstance(direction, dict) and direction.get("description"):
        phrase += f"，方向为{direction['description']}"
    return phrase


def _merged_phase(*, starts_here: bool, finishes_here: bool) -> str:
    if starts_here and finishes_here:
        return "start_and_finish"
    if starts_here:
        return "start"
    if finishes_here:
        return "continue_and_finish"
    return "continue"


def _format_duration(duration: float) -> str:
    return f"{duration:.2f}".rstrip("0").rstrip(".")


def _phase_description(
    executor: str,
    action: dict[str, Any],
    phase: str,
    *,
    segment_duration: float,
) -> str:
    verb = "开始" if phase in {"start", "start_and_finish"} else "继续"
    description = f"{executor}{verb}执行{_action_phrase(action)}"
    if phase == "start_and_finish":
        description += f"，持续{_format_duration(segment_duration)}秒后结束该动作"
    elif phase == "continue_and_finish":
        description += f"，{_format_duration(segment_duration)}秒后结束该动作"
    return description


def _build_merged_timeline(result: dict[str, Any]) -> list[dict[str, Any]]:
    action_by_id: dict[str, dict[str, Any]] = {}
    executor_order: list[str] = []
    for timeline in result["executor_timelines"]:
        executor = str(timeline["executor"])
        executor_order.append(executor)
        for action in timeline["actions"]:
            action_by_id[str(action["subtask_id"])] = action

    merged_timeline: list[dict[str, Any]] = []
    for segment in result["boundary_union"]:
        segment_start = float(segment["start_time"])
        segment_end = float(segment["end_time"])
        arm_entries: dict[str, dict[str, Any]] = {}

        for executor in executor_order:
            subtask_id = segment["covered_subtasks"].get(executor)
            if subtask_id is None:
                raise AnalysisBoundaryUnionError(
                    f"{executor} has no subtask covering merged interval "
                    f"[{segment_start}, {segment_end}]"
                )
            action = action_by_id[subtask_id]
            starts_here = math.isclose(
                float(action["start_time"]), segment_start, abs_tol=1e-9
            )
            finishes_here = math.isclose(
                float(action["end_time"]), segment_end, abs_tol=1e-9
            )
            phase = _merged_phase(
                starts_here=starts_here,
                finishes_here=finishes_here,
            )
            arm_entries[executor] = {
                "subtask_id": subtask_id,
                "action": action.get("action"),
                "phase": phase,
                "description": _phase_description(
                    executor,
                    action,
                    phase,
                    segment_duration=segment_end - segment_start,
                ),
            }

        description_order = sorted(
            executor_order,
            key=lambda executor: (
                0 if arm_entries[executor]["phase"] in {"start", "start_and_finish"} else 1,
                executor_order.index(executor),
            ),
        )
        merged_segment: dict[str, Any] = {
            "segment_id": segment["segment_id"],
            "start_time": segment_start,
            "end_time": segment_end,
        }
        merged_segment.update(arm_entries)
        merged_segment["segment_description"] = "；".join(
            arm_entries[executor]["description"] for executor in description_order
        ) + "。"
        merged_timeline.append(merged_segment)

    return merged_timeline


def postprocess_refinement_output(refinement_output: dict[str, Any]) -> dict[str, Any]:
    """Add stable IDs and a described merged timeline to refined timelines."""
    result = _postprocess_output(
        refinement_output,
        start_field="start_time",
        end_field="end_time",
        stage_name="Refinement",
    )
    result["merged_timeline"] = _build_merged_timeline(result)
    result.pop("boundary_union", None)
    return result


def postprocess_refinement_timelines(refinement_output: dict[str, Any]) -> dict[str, Any]:
    """Add stable IDs to complete refined timelines without building a merge."""
    result = _postprocess_output(
        refinement_output,
        start_field="start_time",
        end_field="end_time",
        stage_name="Refinement",
    )
    result.pop("boundary_union", None)
    result.pop("merged_timeline", None)
    return result
