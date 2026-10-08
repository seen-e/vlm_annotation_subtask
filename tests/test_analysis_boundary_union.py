from __future__ import annotations

import copy

import pytest

from postprocessors.analysis_boundary_union import (
    AnalysisBoundaryUnionError,
    postprocess_analysis_output,
    postprocess_refinement_output,
    postprocess_refinement_timelines,
)
from result_io import load_stage_into_context, save_stage_result


def _action(start: float, end: float, action: str) -> dict:
    return {
        "start_time_hint": start,
        "end_time_hint": end,
        "start_evidence": "start evidence",
        "process_evidence": "process evidence",
        "end_evidence": "end evidence",
        "object": "object_1",
        "target": None,
        "action": action,
    }


def _refined_action(start: float, end: float, action: str) -> dict:
    return {
        "start_time": start,
        "end_time": end,
        "start_boundary_evidence": "start evidence",
        "end_boundary_evidence": "end evidence",
        "object": "object_1",
        "target": None,
        "action": action,
    }


def test_builds_boundary_union_without_mutating_input() -> None:
    source = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_action(0.0, 3.0, "action_a")]},
            {"executor": "arm_2", "actions": [_action(1.0, 5.0, "action_b")]},
        ],
        "uncertainties": [],
    }
    original = copy.deepcopy(source)

    result = postprocess_analysis_output(source)

    assert source == original
    assert result["executor_timelines"][0]["actions"][0]["subtask_id"] == "arm_1_subtask_0001"
    assert [(item["start_time_hint"], item["end_time_hint"]) for item in result["boundary_union"]] == [
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 5.0),
    ]
    assert result["boundary_union"][1]["covered_subtasks"] == {
        "arm_1": "arm_1_subtask_0001",
        "arm_2": "arm_2_subtask_0001",
    }
    assert result["boundary_union"][1]["start_boundary_from"] == [
        {
            "executor": "arm_2",
            "subtask_id": "arm_2_subtask_0001",
            "boundary_type": "start",
        }
    ]


def test_keeps_all_sources_for_a_shared_boundary() -> None:
    result = postprocess_analysis_output(
        {
            "executor_timelines": [
                {
                    "executor": "arm_1",
                    "actions": [_action(0.0, 2.0, "action_a"), _action(2.0, 4.0, "action_b")],
                },
                {"executor": "arm_2", "actions": [_action(2.0, 5.0, "action_c")]},
            ]
        }
    )

    shared = result["boundary_union"][1]["start_boundary_from"]
    assert {(item["subtask_id"], item["boundary_type"]) for item in shared} == {
        ("arm_1_subtask_0001", "end"),
        ("arm_1_subtask_0002", "start"),
        ("arm_2_subtask_0001", "start"),
    }


def test_represents_inactive_executor_as_null() -> None:
    result = postprocess_analysis_output(
        {
            "executor_timelines": [
                {"executor": "arm_1", "actions": [_action(0.0, 1.0, "action_a")]},
                {"executor": "arm_2", "actions": []},
            ]
        }
    )

    assert result["boundary_union"][0]["covered_subtasks"] == {
        "arm_1": "arm_1_subtask_0001",
        "arm_2": None,
    }


def test_rejects_overlapping_subtasks_for_one_executor() -> None:
    with pytest.raises(AnalysisBoundaryUnionError, match="overlapping subtasks"):
        postprocess_analysis_output(
            {
                "executor_timelines": [
                    {
                        "executor": "arm_1",
                        "actions": [_action(0.0, 3.0, "action_a"), _action(2.0, 4.0, "action_b")],
                    }
                ]
            }
        )


def test_preserves_existing_subtask_id() -> None:
    action = _action(0.0, 1.0, "action_a")
    action["subtask_id"] = "custom_id"

    result = postprocess_analysis_output(
        {"executor_timelines": [{"executor": "arm_1", "actions": [action]}]}
    )

    assert result["executor_timelines"][0]["actions"][0]["subtask_id"] == "custom_id"
    assert result["boundary_union"][0]["covered_subtasks"]["arm_1"] == "custom_id"


def test_refinement_postprocessor_uses_refined_time_fields() -> None:
    result = postprocess_refinement_output(
        {
            "executor_timelines": [
                {
                    "executor": "arm_1",
                    "actions": [
                        _refined_action(0.0, 2.0, "action_a"),
                        _refined_action(2.0, 3.0, "action_c"),
                    ],
                },
                {
                    "executor": "arm_2",
                    "actions": [
                        _refined_action(0.0, 1.0, "action_d"),
                        _refined_action(1.0, 3.0, "action_b"),
                    ],
                },
            ],
            "uncertainties": [],
        }
    )

    assert result["executor_timelines"][0]["actions"][0]["subtask_id"] == "arm_1_subtask_0001"
    assert "boundary_union" not in result
    assert [(item["start_time"], item["end_time"]) for item in result["merged_timeline"]] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]
    middle = result["merged_timeline"][1]
    assert middle["arm_1"]["phase"] == "continue_and_finish"
    assert middle["arm_2"]["phase"] == "start"
    assert middle["segment_description"].startswith("arm_2开始执行action_b object_1")
    assert "arm_1继续执行action_a object_1，1秒后结束该动作" in middle[
        "segment_description"
    ]


def test_refinement_timelines_adds_ids_without_merge() -> None:
    result = postprocess_refinement_timelines(
        {
            "executor_timelines": [
                {"executor": "arm_1", "actions": [_refined_action(0.0, 3.0, "action_a")]},
                {"executor": "arm_2", "actions": [_refined_action(0.0, 3.0, "action_b")]},
            ],
            "uncertainties": [],
        }
    )

    assert result["executor_timelines"][0]["actions"][0]["subtask_id"] == "arm_1_subtask_0001"
    assert result["executor_timelines"][1]["actions"][0]["subtask_id"] == "arm_2_subtask_0001"
    assert "boundary_union" not in result
    assert "merged_timeline" not in result


def test_merged_timeline_marks_same_arm_start_and_finish() -> None:
    result = postprocess_refinement_output(
        {
            "executor_timelines": [
                {"executor": "arm_1", "actions": [_refined_action(0.0, 4.0, "保持")]},
                {
                    "executor": "arm_2",
                    "actions": [
                        _refined_action(0.0, 1.0, "等待"),
                        _refined_action(1.0, 2.25, "夹持"),
                        _refined_action(2.25, 4.0, "移动"),
                    ],
                },
            ]
        }
    )

    middle = result["merged_timeline"][1]
    assert middle["arm_1"]["phase"] == "continue"
    assert middle["arm_2"]["phase"] == "start_and_finish"
    assert middle["segment_description"] == (
        "arm_2开始执行夹持 object_1，持续1.25秒后结束该动作；"
        "arm_1继续执行保持 object_1。"
    )


def test_refinement_postprocessing_is_saved_in_output_json(tmp_path) -> None:
    original = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_refined_action(0.0, 1.0, "action_a")]}
        ]
    }
    processed = postprocess_refinement_output(original)
    context = {
        "stages": {
            "refinement": {
                "output": processed,
            }
        }
    }

    save_stage_result(context, "refinement", tmp_path)

    stage_dir = tmp_path / "stages" / "refinement"
    assert (stage_dir / "output.json").exists()
    assert not (stage_dir / "output_postprocessed.json").exists()
    loaded_context: dict = {}
    load_stage_into_context(loaded_context, "refinement", tmp_path)
    assert loaded_context["stages"]["refinement"]["output"] == processed
    assert "output_postprocessed" not in loaded_context["stages"]["refinement"]
