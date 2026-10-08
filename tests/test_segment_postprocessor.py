from __future__ import annotations

import pytest

from segment_postprocessor import process_segment_result, stitch_executor_timelines


def _config(**postprocessor):
    return {
        "result_stage": "final",
        "timeline_field": "payload.timeline",
        "postprocessor": {
            "policy": "last_subtask_start",
            "min_advance_sec": 1.0,
            "fallback_overlap_sec": 2.0,
            **postprocessor,
        },
    }


def _context(timeline):
    return {"stages": {"final": {"output": {"payload": {"timeline": timeline}}}}}


def _context_with_timestamps(timeline, timestamps):
    return {
        "stages": {
            "final": {
                "output": {"payload": {"timeline": timeline}},
                "video_meta": {"sampled_timestamps": timestamps},
            }
        }
    }


def _item(name: str, start: float, end: float):
    return {"name": name, "start_time_sec": start, "end_time_sec": end}


def test_non_final_segment_commits_all_but_provisional_tail() -> None:
    timeline = [_item("a", 0.0, 5.0), _item("b", 5.0, 10.0)]

    decision = process_segment_result(
        _context(timeline),
        {"start_time_sec": 0.0, "end_time_sec": 15.0},
        30.0,
        _config(),
    )

    assert decision["status"] == "continue"
    assert decision["committed_subtasks"] == [timeline[0]]
    assert decision["provisional_tail"] == timeline[1]
    assert decision["next_start_time_sec"] == 5.0
    assert decision["reason"] == "last_subtask_start"


def test_final_segment_commits_entire_timeline() -> None:
    timeline = [_item("a", 21.0, 25.0), _item("b", 25.0, 30.0)]

    decision = process_segment_result(
        _context(timeline),
        {"start_time_sec": 21.0, "end_time_sec": 30.0},
        30.0,
        _config(),
    )

    assert decision["status"] == "finish"
    assert decision["committed_subtasks"] == timeline
    assert decision["provisional_tail"] is None


def test_empty_and_invalid_timelines_request_retry() -> None:
    empty = process_segment_result(
        _context([]), {"start_time_sec": 0.0, "end_time_sec": 15.0}, 30.0, _config()
    )
    invalid = process_segment_result(
        _context([_item("bad", 5.0, float("nan"))]),
        {"start_time_sec": 0.0, "end_time_sec": 15.0},
        30.0,
        _config(),
    )

    assert empty["status"] == "retry"
    assert empty["reason"] == "empty_timeline"
    assert invalid["status"] == "retry"
    assert invalid["reason"] == "invalid_timeline"


def test_out_of_order_timeline_is_sorted_with_warning() -> None:
    timeline = [_item("late", 5.0, 9.0), _item("early", 0.0, 5.0)]

    decision = process_segment_result(
        _context(timeline), {"start_time_sec": 0.0, "end_time_sec": 15.0}, 30.0, _config()
    )

    assert decision["committed_subtasks"][0]["name"] == "early"
    assert decision["provisional_tail"]["name"] == "late"
    assert decision["warnings"] == ["timeline_sorted_by_start_time"]


def test_small_advance_uses_fixed_overlap_without_committing_tail() -> None:
    timeline = [_item("tail", 10.2, 15.0)]

    decision = process_segment_result(
        _context(timeline), {"start_time_sec": 10.0, "end_time_sec": 25.0}, 40.0, _config()
    )

    assert decision["status"] == "continue"
    assert decision["committed_subtasks"] == []
    assert decision["next_start_time_sec"] == 23.0
    assert decision["reason"] == "fallback_fixed_overlap"


def test_impossible_fallback_fails() -> None:
    decision = process_segment_result(
        _context([_item("tail", 10.2, 15.0)]),
        {"start_time_sec": 10.0, "end_time_sec": 11.0},
        40.0,
        _config(fallback_overlap_sec=2.0),
    )

    assert decision["status"] == "fail"
    assert decision["reason"] == "fallback_cannot_advance"


def test_current_business_schema_start_time_alias_is_supported() -> None:
    timeline = [{"name": "a", "start_time": 0.0, "end_time": 8.0}]

    decision = process_segment_result(
        _context(timeline), {"start_time_sec": 0.0, "end_time_sec": 15.0}, 30.0, _config()
    )

    assert decision["status"] == "continue"
    assert decision["provisional_tail"] == timeline[0]


def test_boundaries_are_canonicalized_to_actual_sampled_timestamps() -> None:
    timeline = [_item("a", 5.001, 8.009), _item("b", 8.0, 10.0)]
    config = _config(require_sampled_timestamps=True, timestamp_tolerance_sec=0.011)

    decision = process_segment_result(
        _context_with_timestamps(timeline, [5.0, 8.0, 10.0]),
        {"start_time_sec": 5.0, "end_time_sec": 15.0},
        30.0,
        config,
    )

    assert decision["status"] == "continue"
    assert decision["committed_subtasks"][0]["start_time_sec"] == 5.0
    assert decision["committed_subtasks"][0]["end_time_sec"] == 8.0
    assert decision["next_start_time_sec"] == 8.0


def test_boundary_not_present_in_sent_images_requests_retry() -> None:
    config = _config(require_sampled_timestamps=True, timestamp_tolerance_sec=0.01)

    decision = process_segment_result(
        _context_with_timestamps([_item("bad", 5.2, 8.0)], [5.0, 8.0, 10.0]),
        {"start_time_sec": 5.0, "end_time_sec": 15.0},
        30.0,
        config,
    )

    assert decision["status"] == "retry"
    assert decision["reason"] == "timestamp_not_sampled"


def test_fixed_overlap_fallback_uses_a_cached_timestamp() -> None:
    config = _config(
        require_sampled_timestamps=True,
        timestamp_tolerance_sec=0.01,
        fallback_overlap_sec=2.0,
    )

    decision = process_segment_result(
        _context_with_timestamps([_item("tail", 10.2, 22.9)], [10.2, 22.9, 24.8]),
        {"start_time_sec": 10.0, "end_time_sec": 25.0},
        40.0,
        config,
    )

    assert decision["status"] == "continue"
    assert decision["reason"] == "fallback_fixed_overlap"
    assert decision["next_start_time_sec"] == 22.9


def _refined_action(name: str, start: float, end: float) -> dict:
    return {"action": name, "start_time": start, "end_time": end}


def test_bimanual_policy_uses_earlier_last_action_start_and_keeps_arm_starts() -> None:
    output = {
        "executor_timelines": [
            {
                "executor": "arm_1",
                "actions": [
                    _refined_action("等待", 0.0, 11.0),
                    _refined_action("移动", 11.0, 15.0),
                ],
            },
            {
                "executor": "arm_2",
                "actions": [
                    _refined_action("等待", 0.0, 13.0),
                    _refined_action("夹持", 13.0, 15.0),
                ],
            },
        ],
        "uncertainties": [],
    }
    context = {
        "stages": {
            "refinement": {
                "output": output,
                "video_meta": {"sampled_timestamps": [0.0, 11.0, 13.0]},
            }
        }
    }
    config = {
        "result_stage": "refinement",
        "executors": ["arm_1", "arm_2"],
        "postprocessor": {
            "policy": "bimanual_last_action_start",
            "require_sampled_timestamps": True,
            "timestamp_tolerance_sec": 0.011,
        },
    }

    decision = process_segment_result(
        context,
        {
            "start_time_sec": 0.0,
            "end_time_sec": 15.0,
            "arm_1_annotation_start_time_sec": 0.0,
            "arm_2_annotation_start_time_sec": 0.0,
        },
        30.0,
        config,
    )

    assert decision["status"] == "continue"
    assert decision["next_start_time_sec"] == 11.0
    assert decision["next_annotation_start_times"] == {"arm_1": 11.0, "arm_2": 13.0}
    assert "merged_timeline" not in decision["segment_output"]


def test_bimanual_policy_ignores_the_arm_with_only_one_subtask() -> None:
    output = {
        "executor_timelines": [
            {
                "executor": "arm_1",
                "actions": [_refined_action("等待", 0.0, 15.0)],
            },
            {
                "executor": "arm_2",
                "actions": [
                    _refined_action("等待", 0.0, 9.0),
                    _refined_action("移动", 9.0, 15.0),
                ],
            },
        ],
        "uncertainties": [],
    }
    context = {
        "stages": {
            "refinement": {
                "output": output,
                "video_meta": {"sampled_timestamps": [0.0, 9.0]},
            }
        }
    }
    config = {
        "result_stage": "refinement",
        "executors": ["arm_1", "arm_2"],
        "postprocessor": {
            "policy": "bimanual_last_action_start",
            "require_sampled_timestamps": True,
        },
    }

    decision = process_segment_result(
        context,
        {
            "start_time_sec": 0.0,
            "end_time_sec": 15.0,
            "arm_1_annotation_start_time_sec": 0.0,
            "arm_2_annotation_start_time_sec": 0.0,
        },
        30.0,
        config,
    )

    assert decision["status"] == "continue"
    assert decision["reason"] == "single_subtask_executor_ignored"
    assert decision["next_start_time_sec"] == 9.0
    assert decision["next_annotation_start_times"] == {"arm_1": 9.0, "arm_2": 9.0}
    assert decision["action_counts"] == {"arm_1": 1, "arm_2": 2}


def test_bimanual_policy_uses_segment_end_when_both_arms_have_one_subtask() -> None:
    output = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_refined_action("等待", 0.0, 15.0)]},
            {"executor": "arm_2", "actions": [_refined_action("保持", 0.0, 15.0)]},
        ],
        "uncertainties": [],
    }
    context = {
        "stages": {
            "refinement": {
                "output": output,
                "video_meta": {"sampled_timestamps": [0.0]},
            }
        }
    }
    config = {
        "result_stage": "refinement",
        "executors": ["arm_1", "arm_2"],
        "postprocessor": {
            "policy": "bimanual_last_action_start",
            "require_sampled_timestamps": True,
        },
    }

    decision = process_segment_result(
        context,
        {
            "start_time_sec": 0.0,
            "end_time_sec": 15.0,
            "arm_1_annotation_start_time_sec": 0.0,
            "arm_2_annotation_start_time_sec": 0.0,
        },
        30.0,
        config,
    )

    assert decision["status"] == "continue"
    assert decision["reason"] == "both_executors_single_subtask_segment_end"
    assert decision["next_start_time_sec"] == 15.0
    assert decision["next_annotation_start_times"] == {"arm_1": 15.0, "arm_2": 15.0}
    assert decision["action_counts"] == {"arm_1": 1, "arm_2": 1}


def test_stitch_replaces_each_arms_tail_at_its_own_boundary() -> None:
    previous = {
        "executor_timelines": [
            {
                "executor": "arm_1",
                "actions": [
                    _refined_action("a", 0.0, 11.0),
                    _refined_action("old_arm_1_tail", 11.0, 15.0),
                ],
            },
            {
                "executor": "arm_2",
                "actions": [
                    _refined_action("b", 0.0, 13.0),
                    _refined_action("old_arm_2_tail", 13.0, 15.0),
                ],
            },
        ],
        "uncertainties": [],
    }
    current = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_refined_action("new_1", 11.0, 26.0)]},
            {"executor": "arm_2", "actions": [_refined_action("new_2", 13.0, 26.0)]},
        ],
        "uncertainties": ["current segment"],
    }

    stitched = stitch_executor_timelines(
        previous,
        current,
        {"arm_1": 11.0, "arm_2": 13.0},
    )

    assert [item["action"] for item in stitched["executor_timelines"][0]["actions"]] == [
        "a",
        "new_1",
    ]
    assert [item["action"] for item in stitched["executor_timelines"][1]["actions"]] == [
        "b",
        "new_2",
    ]
    assert stitched["uncertainties"] == ["current segment"]


def test_stitch_keeps_prefix_when_fallback_boundary_splits_a_long_action() -> None:
    previous = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_refined_action("等待", 0.0, 15.0)]},
            {"executor": "arm_2", "actions": [_refined_action("等待", 0.0, 15.0)]},
        ],
        "uncertainties": [],
    }
    current = {
        "executor_timelines": [
            {"executor": "arm_1", "actions": [_refined_action("等待", 13.0, 28.0)]},
            {"executor": "arm_2", "actions": [_refined_action("等待", 13.0, 28.0)]},
        ],
        "uncertainties": [],
    }

    stitched = stitch_executor_timelines(
        previous,
        current,
        {"arm_1": 13.0, "arm_2": 13.0},
    )

    for timeline in stitched["executor_timelines"]:
        assert [(item["start_time"], item["end_time"]) for item in timeline["actions"]] == [
            (0.0, 13.0),
            (13.0, 28.0),
        ]
