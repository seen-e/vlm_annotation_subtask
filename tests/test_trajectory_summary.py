from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajectory_summary import (
    TrajectorySummaryError,
    build_trajectory,
    register_unique_output_path,
    resolve_summary_output_path,
    save_trajectory,
    timestamp_to_original_frame,
)


def _video_meta(**overrides):
    meta = {
        "original_fps": 30.0,
        "original_duration": 10.0,
        "effective_duration": 10.0,
        "effective_start_time": 0.0,
        "effective_end_time": 10.0,
        "effective_frame_start": 0,
        "effective_frame_end": 300,
        "sampled_timestamps": [0.0, 2.0, 5.0, 9.0],
        "sampled_frame_indices": [0, 60, 150, 270],
    }
    meta.update(overrides)
    return meta


def _refinement_output():
    return {
        "subtasks": [
            {
                "description": "将 bottle_1 放到桌面左侧",
                "source_step_ids": [1, 2],
                "completion_status": "完成",
            },
            {
                "description": "调整 bottle_2",
                "source_step_ids": [3],
                "completion_status": "无法判断",
            },
        ],
        "timed_executor_timelines": [
            {
                "executor": "right",
                "actions": [
                    {
                        "step_id": 2,
                        "action": "放置",
                        "object": "bottle_1",
                        "target": "桌面左侧",
                        "start_time": 4.0,
                        "end_time": 6.0,
                    },
                    {
                        "step_id": 1,
                        "action": "抓取",
                        "object": "bottle_1",
                        "target": None,
                        "start_time": 1.0,
                        "end_time": 3.0,
                        "start_boundary_evidence": "右臂接触 bottle_1",
                    },
                ],
            },
            {
                "executor": "left",
                "actions": [
                    {
                        "step_id": 3,
                        "action": "接近",
                        "object": "bottle_2",
                        "start_time": 2.0,
                        "end_time": 5.0,
                        "caption": "left 接近 bottle_2",
                    }
                ],
            },
        ],
    }


def _current_refinement_output():
    return {
        "video_duration": 10.0,
        "executor_timelines": [
            {
                "executor": "left",
                "actions": [
                    {
                        "action": "approach",
                        "object": "bottle_1",
                        "target": None,
                        "start_time": 1.0,
                        "end_time": 2.0,
                        "evidence": "left approaches bottle_1",
                    }
                ],
                "supplemented_actions": [
                    {
                        "action": "release",
                        "object": "bottle_1",
                        "target": "table",
                        "start_time": 3.0,
                        "end_time": 4.0,
                        "evidence": "left releases bottle_1",
                    }
                ],
                "remaining_intervals": [
                    {
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "interval_type": "idle",
                        "evidence": "idle",
                    }
                ],
            }
        ],
        "uncertainties": [],
    }


def test_resolve_summary_output_path_prefers_task_trajectory_path(tmp_path: Path) -> None:
    task = {"trajectory_path": str(tmp_path / "custom" / "trajectory.json")}
    default = tmp_path / "default.json"

    assert resolve_summary_output_path(task, default) == tmp_path / "custom" / "trajectory.json"


@pytest.mark.parametrize("value", [None, ""])
def test_resolve_summary_output_path_falls_back_to_default(tmp_path: Path, value) -> None:
    assert resolve_summary_output_path({"trajectory_path": value}, tmp_path / "default.json") == tmp_path / "default.json"


def test_timestamp_to_original_frame_uses_sampled_mapping_not_sample_index() -> None:
    assert timestamp_to_original_frame(9.0, _video_meta()) == 270


def test_timestamp_to_original_frame_falls_back_to_fps_when_sampling_missing() -> None:
    assert timestamp_to_original_frame(2.0, _video_meta(sampled_timestamps=[], sampled_frame_indices=[])) == 60


def test_build_trajectory_episode_fields_and_keyframes_from_video_meta() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(effective_start_time=1.0, effective_end_time=9.0, effective_frame_start=30, effective_frame_end=270),
    )

    assert trajectory["episode_id"] == "episode_000001"
    assert trajectory["duration"] == pytest.approx(8.0)
    assert trajectory["start_frame"] == 30
    assert trajectory["end_frame"] == 270
    assert trajectory["keyframes"] == [30, 60, 150]


def test_episode_keyframes_fall_back_to_episode_bounds_when_no_segments() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output={"executor_timelines": []},
        video_meta=_video_meta(effective_start_time=1.0, effective_end_time=9.0, effective_frame_start=30, effective_frame_end=270),
    )

    assert trajectory["keyframes"] == [30, 150, 270]


def test_actions_are_flattened_sorted_and_not_merged() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    segments = trajectory["segments"]
    assert set(segments) == {"left", "right"}
    assert [segment["segment_id"] for segment in segments["right"]] == [0, 1]
    assert [segment["segment_id"] for segment in segments["left"]] == [0]
    assert [segment["start_time"] for segment in segments["right"]] == [1.0, 4.0]
    assert [segment["start_time"] for segment in segments["left"]] == [2.0]
    assert [segment["phase"] for segment in segments["right"]] == ["抓取", "放置"]
    assert [segment["phase"] for segment in segments["left"]] == ["接近"]


def test_current_refinement_executor_timelines_are_converted_to_segments() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_current_refinement_output(),
        video_meta=_video_meta(),
    )

    segments = trajectory["segments"]
    assert set(segments) == {"left"}
    left_segments = segments["left"]
    assert [segment["segment_id"] for segment in left_segments] == [0, 1]
    assert [segment["phase"] for segment in left_segments] == ["approach", "release"]
    assert [segment["source"] for segment in left_segments] == ["action", "supplemented_action"]
    assert [segment["executor"] for segment in left_segments] == ["left", "left"]
    assert left_segments[0]["object"] == "bottle_1"
    assert left_segments[1]["target"] == "table"
    assert left_segments[0]["evidence"] == "left approaches bottle_1"


def test_unknown_task_falls_back_to_subtask_descriptions() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Unknown task"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    assert trajectory["L1_task"] == "将 bottle_1 放到桌面左侧；调整 bottle_2"


def test_success_uses_subtask_source_step_ids() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    assert trajectory["segments"]["right"][0]["success"] == 1
    assert trajectory["segments"]["left"][0]["success"] is None


def test_fallback_caption_uses_core_action_fields_without_evidence() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    assert trajectory["segments"]["right"][0]["caption"] == "right 抓取 bottle_1"
    assert "开始证据" not in trajectory["segments"]["right"][0]["caption"]


def test_missing_quality_defaults_to_unknown_values() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    assert trajectory["quality"] == {
        "caption_video_match": None,
        "object_presence_rate": None,
        "status": "UNKNOWN",
    }


def test_save_trajectory_uses_utf8_without_ascii_escaping(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "trajectory.json"
    save_trajectory({"caption": "中文动作"}, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "中文动作" in text
    assert json.loads(text)["caption"] == "中文动作"


def test_register_unique_output_path_rejects_same_path_conflict(tmp_path: Path) -> None:
    used: dict[str, str] = {}
    output_path = tmp_path / "trajectory.json"
    register_unique_output_path(output_path, "episode_000001", used)

    with pytest.raises(TrajectorySummaryError, match="trajectory output path conflict"):
        register_unique_output_path(output_path, "episode_000002", used)


def test_final_trajectory_does_not_keep_old_summary_debug_fields() -> None:
    trajectory = build_trajectory(
        task={"episode_id": "episode_000001", "task": "Pick bottle"},
        refinement_output=_refinement_output(),
        video_meta=_video_meta(),
    )

    forbidden = {"status", "run_dir", "workflow", "selected_stages", "executed_stages", "final_output"}
    assert forbidden.isdisjoint(trajectory.keys())


def test_invalid_refinement_output_raises() -> None:
    with pytest.raises(TrajectorySummaryError):
        build_trajectory(task={}, refinement_output=[], video_meta=_video_meta())
