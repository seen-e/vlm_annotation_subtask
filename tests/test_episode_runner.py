from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

episode_runner = importlib.import_module(f"{PACKAGE_DIR.name}.episode_runner")


def _context() -> dict[str, Any]:
    return {
        "input": {
            "video_path": "/data/original.mp4",
            "video_segments": {"camera": {"video_path": "/data/source.mp4"}},
            "video_id": "episode_1",
            "duration_sec": 30.0,
        },
        "stages": {"stale": {"output": {"bad": True}}},
    }


def _config() -> dict[str, Any]:
    return {
        "workflow": ["scene", "analysis", "final"],
        "stages": {"scene": {}, "analysis": {}, "final": {}},
        "segment_loop": {
            "enabled": True,
            "window_duration_sec": 15.0,
            "result_stage": "final",
            "timeline_field": "timeline",
            "postprocessor": {
                "policy": "last_subtask_start",
                "min_advance_sec": 1.0,
                "fallback_overlap_sec": 2.0,
            },
        },
    }


def test_episode_runs_full_pipeline_for_fresh_serial_segments(tmp_path: Path, monkeypatch) -> None:
    pipeline_windows: list[tuple[float, float]] = []
    pipeline_context_ids: list[int] = []

    def fake_pipeline(context: dict[str, Any], config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        assert context["stages"] == {}
        assert "video_segments" not in context["input"]
        assert kwargs["skip_existing"] is False
        pipeline_context_ids.append(id(context["stages"]))
        window = context["segment"]
        pipeline_windows.append((window["start_time_sec"], window["end_time_sec"]))
        context["stages"] = {name: {"output": {"ok": True}} for name in config["workflow"]}
        return context

    decisions = {
        0: {
            "status": "continue",
            "committed_subtasks": [{"name": "a"}],
            "provisional_tail": {"name": "tail_0"},
            "next_start_time_sec": 10.0,
            "reason": "last_subtask_start",
        },
        1: {
            "status": "continue",
            "committed_subtasks": [{"name": "b"}],
            "provisional_tail": {"name": "tail_1"},
            "next_start_time_sec": 21.0,
            "reason": "last_subtask_start",
        },
        2: {
            "status": "finish",
            "committed_subtasks": [{"name": "c"}],
            "provisional_tail": None,
            "next_start_time_sec": None,
            "reason": "episode_end",
        },
    }

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        episode_runner,
        "process_segment_result",
        lambda context, window, duration, config: decisions[window["segment_index"]],
    )

    result = episode_runner.run_episode_pipeline(
        _context(), _config(), output_dir=tmp_path, run_name="run", skip_existing=False
    )

    assert pipeline_windows == [(0.0, 15.0), (10.0, 25.0), (21.0, 30.0)]
    assert len(set(pipeline_context_ids)) == 3
    assert result["final_subtasks"] == [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    assert json.loads((tmp_path / "run" / "final_subtasks.json").read_text(encoding="utf-8")) == result[
        "final_subtasks"
    ]
    progress = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed"] is True
    assert progress["next_start_time_sec"] is None


def test_episode_resume_continues_from_progress(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    progress = {
        "episode_id": "episode_1",
        "episode_duration_sec": 30.0,
        "next_segment_index": 1,
        "next_start_time_sec": 10.0,
        "committed_subtasks": [{"name": "a"}],
        "provisional_tail": {"name": "tail_0"},
        "completed": False,
        "failed": False,
        "retry_count": 0,
        "visited_start_times": [0.0],
        "last_decision": {},
    }
    (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    seen_indices: list[int] = []

    def fake_pipeline(context: dict[str, Any], config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen_indices.append(context["segment"]["segment_index"])
        return context

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        episode_runner,
        "process_segment_result",
        lambda *args: {
            "status": "finish",
            "committed_subtasks": [{"name": "b"}],
            "provisional_tail": None,
            "next_start_time_sec": None,
            "reason": "episode_end",
        },
    )

    result = episode_runner.run_episode_pipeline(
        _context(), _config(), output_dir=tmp_path, run_name="run", skip_existing=True
    )

    assert seen_indices == [1]
    assert result["final_subtasks"] == [{"name": "a"}, {"name": "b"}]


def test_episode_rejects_non_monotonic_decision_and_saves_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(episode_runner, "run_pipeline", lambda context, config, **kwargs: context)
    monkeypatch.setattr(
        episode_runner,
        "process_segment_result",
        lambda *args: {
            "status": "continue",
            "committed_subtasks": [],
            "provisional_tail": {"name": "tail"},
            "next_start_time_sec": 0.0,
            "reason": "last_subtask_start",
        },
    )

    with pytest.raises(episode_runner.EpisodeRunnerError, match="monotonic"):
        episode_runner.run_episode_pipeline(
            _context(), _config(), output_dir=tmp_path, run_name="run", skip_existing=False
        )

    progress = json.loads((tmp_path / "run" / "progress.json").read_text(encoding="utf-8"))
    assert progress["failed"] is True
    assert progress["completed"] is False


def test_completed_final_result_is_reused_without_pipeline_call(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final = [{"name": "done"}]
    (run_dir / "final_subtasks.json").write_text(json.dumps(final), encoding="utf-8")
    monkeypatch.setattr(
        episode_runner,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pipeline should not run")),
    )

    result = episode_runner.run_episode_pipeline(
        _context(), _config(), output_dir=tmp_path, run_name="run", skip_existing=True
    )

    assert result["final_subtasks"] == final
    assert result["episode_loop"]["resumed_from_final"] is True


def test_retry_limit_saves_failed_progress(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    config["segment_loop"]["max_retries_per_segment"] = 1
    calls = 0

    def fake_pipeline(context: dict[str, Any], config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return context

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        episode_runner,
        "process_segment_result",
        lambda *args: {
            "status": "retry",
            "committed_subtasks": [],
            "provisional_tail": None,
            "next_start_time_sec": None,
            "reason": "empty_timeline",
        },
    )

    with pytest.raises(episode_runner.EpisodeRunnerError, match="retry limit"):
        episode_runner.run_episode_pipeline(
            _context(), config, output_dir=tmp_path, run_name="retry", skip_existing=False
        )

    assert calls == 2
    progress = json.loads((tmp_path / "retry" / "progress.json").read_text(encoding="utf-8"))
    assert progress["failed"] is True
    assert progress["last_decision"]["reason"] == "empty_timeline"


def test_episode_prepares_cache_once_and_passes_it_to_every_segment(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    config["segment_loop"]["frame_cache"] = {
        "enabled": True,
        "jpeg_quality": 91,
        "cleanup_on_complete": True,
    }
    prepared: list[dict[str, Any]] = []
    cleaned: list[dict[str, Any]] = []
    cache_state = {
        "root": str(tmp_path / "run" / ".frame_cache"),
        "stage_manifests": {"scene": "/cache/scene.json"},
        "reused": False,
    }

    def fake_prepare(video_path: Any, config: dict[str, Any], cache_root: Path, **kwargs: Any):
        prepared.append({"video_path": video_path, "cache_root": cache_root, **kwargs})
        return cache_state

    def fake_pipeline(context: dict[str, Any], config: dict[str, Any], **kwargs: Any):
        assert context["frame_cache"] == cache_state
        assert context["segment"]["episode_duration_sec"] == 30.0
        return context

    monkeypatch.setattr(episode_runner, "prepare_episode_frame_cache", fake_prepare)
    monkeypatch.setattr(episode_runner, "cleanup_episode_frame_cache", lambda state: cleaned.append(state))
    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(
        episode_runner,
        "process_segment_result",
        lambda *args: {
            "status": "finish",
            "committed_subtasks": [{"name": "done"}],
            "provisional_tail": None,
            "next_start_time_sec": None,
            "reason": "episode_end",
        },
    )

    episode_runner.run_episode_pipeline(
        _context(), config, output_dir=tmp_path, run_name="run", skip_existing=False
    )

    assert len(prepared) == 1
    assert prepared[0]["jpeg_quality"] == 91
    assert prepared[0]["video_segments"] == _context()["input"]["video_segments"]
    assert cleaned == [cache_state]


def test_bimanual_episode_stitches_full_timelines_then_merges_once(tmp_path: Path, monkeypatch) -> None:
    config = {
        "workflow": ["refinement"],
        "stages": {"refinement": {}},
        "segment_loop": {
            "enabled": True,
            "window_duration_sec": 15.0,
            "result_stage": "refinement",
            "executors": ["arm_1", "arm_2"],
            "postprocessor": {
                "policy": "bimanual_last_action_start",
                "require_sampled_timestamps": False,
                "min_advance_sec": 0.25,
                "fallback_overlap_sec": 2.0,
            },
        },
    }
    context = _context()
    context["input"]["duration_sec"] = 26.0
    seen_windows: list[dict[str, Any]] = []

    def action(name: str, start: float, end: float) -> dict[str, Any]:
        return {
            "start_time": start,
            "end_time": end,
            "action": name,
            "object": None,
            "target": None,
        }

    def fake_pipeline(segment_context: dict[str, Any], pipeline_config: dict[str, Any], **kwargs: Any):
        window = dict(segment_context["segment"])
        seen_windows.append(window)
        if window["segment_index"] == 0:
            output = {
                "executor_timelines": [
                    {
                        "executor": "arm_1",
                        "actions": [action("等待", 0.0, 11.0), action("移动", 11.0, 15.0)],
                    },
                    {
                        "executor": "arm_2",
                        "actions": [action("等待", 0.0, 13.0), action("夹持", 13.0, 15.0)],
                    },
                ],
                "uncertainties": [],
            }
        else:
            output = {
                "executor_timelines": [
                    {
                        "executor": "arm_1",
                        "actions": [action("移动", 11.0, 20.0), action("等待", 20.0, 26.0)],
                    },
                    {
                        "executor": "arm_2",
                        "actions": [action("夹持", 13.0, 18.0), action("放置", 18.0, 26.0)],
                    },
                ],
                "uncertainties": [],
            }
        segment_context["stages"] = {
            "refinement": {"output": output, "video_meta": {"sampled_timestamps": []}}
        }
        return segment_context

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)

    result = episode_runner.run_episode_pipeline(
        context,
        config,
        output_dir=tmp_path,
        run_name="bimanual",
        skip_existing=False,
    )

    assert seen_windows[0]["arm_1_annotation_start_time_sec"] == 0.0
    assert seen_windows[0]["arm_2_annotation_start_time_sec"] == 0.0
    assert seen_windows[1]["start_time_sec"] == 11.0
    assert seen_windows[1]["arm_1_annotation_start_time_sec"] == 11.0
    assert seen_windows[1]["arm_2_annotation_start_time_sec"] == 13.0

    output = result["output"]
    assert [
        (item["start_time"], item["end_time"])
        for item in output["executor_timelines"][0]["actions"]
    ] == [(0.0, 11.0), (11.0, 20.0), (20.0, 26.0)]
    assert [
        (item["start_time"], item["end_time"])
        for item in output["executor_timelines"][1]["actions"]
    ] == [(0.0, 13.0), (13.0, 18.0), (18.0, 26.0)]
    assert output["merged_timeline"]
    assert all("subtask_id" in action for timeline in output["executor_timelines"] for action in timeline["actions"])
    saved = json.loads((tmp_path / "bimanual" / "output.json").read_text(encoding="utf-8"))
    assert saved == output


def test_bimanual_episode_can_skip_final_merge(tmp_path: Path, monkeypatch) -> None:
    config = {
        "workflow": ["refinement"],
        "stages": {"refinement": {}},
        "segment_loop": {
            "enabled": True,
            "window_duration_sec": 15.0,
            "result_stage": "refinement",
            "executors": ["arm_1", "arm_2"],
            "merge_after_episode": False,
            "postprocessor": {
                "policy": "bimanual_last_action_start",
                "require_sampled_timestamps": False,
                "min_advance_sec": 0.25,
                "fallback_overlap_sec": 2.0,
            },
        },
    }
    context = _context()
    context["input"]["duration_sec"] = 10.0

    def fake_pipeline(segment_context: dict[str, Any], pipeline_config: dict[str, Any], **kwargs: Any):
        output = {
            "executor_timelines": [
                {
                    "executor": "arm_1",
                    "actions": [{"start_time": 0.0, "end_time": 10.0, "action": "等待"}],
                },
                {
                    "executor": "arm_2",
                    "actions": [{"start_time": 0.0, "end_time": 10.0, "action": "静止"}],
                },
            ],
            "uncertainties": [],
        }
        segment_context["stages"] = {
            "refinement": {"output": output, "video_meta": {"sampled_timestamps": []}}
        }
        return segment_context

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)

    result = episode_runner.run_episode_pipeline(
        context,
        config,
        output_dir=tmp_path,
        run_name="no_merge",
        skip_existing=False,
    )

    output = result["output"]
    assert "merged_timeline" not in output
    assert all(
        "subtask_id" in action
        for timeline in output["executor_timelines"]
        for action in timeline["actions"]
    )
    run_dir = tmp_path / "no_merge"
    assert episode_runner.is_episode_complete(run_dir, require_merged_timeline=False) is True
    assert episode_runner.is_episode_complete(run_dir, require_merged_timeline=True) is False
    assert json.loads((run_dir / "output.json").read_text(encoding="utf-8")) == output


def test_single_arm_episode_uses_one_timeline_and_keeps_recursive_overlap(
    tmp_path: Path, monkeypatch
) -> None:
    config = {
        "robot_type": "single_arm",
        "workflow": ["refinement"],
        "stages": {"refinement": {}},
        "segment_loop": {
            "enabled": True,
            "window_duration_sec": 15.0,
            "result_stage": "refinement",
            "executors": ["arm_1"],
            "postprocessor": {
                "policy": "bimanual_last_action_start",
                "require_sampled_timestamps": False,
                "min_advance_sec": 0.25,
                "fallback_overlap_sec": 2.0,
            },
        },
    }
    context = _context()
    context["input"]["duration_sec"] = 26.0
    seen_windows: list[dict[str, Any]] = []

    def action(name: str, start: float, end: float) -> dict[str, Any]:
        return {
            "start_time": start,
            "end_time": end,
            "action": name,
            "object": None,
            "target": None,
        }

    def fake_pipeline(segment_context: dict[str, Any], pipeline_config: dict[str, Any], **kwargs: Any):
        window = dict(segment_context["segment"])
        seen_windows.append(window)
        if window["segment_index"] == 0:
            actions = [action("等待", 0.0, 11.0), action("移动", 11.0, 15.0)]
        else:
            actions = [action("移动", 11.0, 20.0), action("等待", 20.0, 26.0)]
        segment_context["stages"] = {
            "refinement": {
                "output": {
                    "executor_timelines": [{"executor": "arm_1", "actions": actions}],
                    "uncertainties": [],
                },
                "video_meta": {"sampled_timestamps": []},
            }
        }
        return segment_context

    monkeypatch.setattr(episode_runner, "run_pipeline", fake_pipeline)

    result = episode_runner.run_episode_pipeline(
        context,
        config,
        output_dir=tmp_path,
        run_name="single_arm",
        skip_existing=False,
    )

    assert seen_windows[0]["arm_1_annotation_start_time_sec"] == 0.0
    assert "arm_2_annotation_start_time_sec" not in seen_windows[0]
    assert seen_windows[1]["start_time_sec"] == 11.0
    assert seen_windows[1]["arm_1_annotation_start_time_sec"] == 11.0
    assert [item["executor"] for item in result["output"]["executor_timelines"]] == ["arm_1"]
    assert all("arm_2" not in item for item in result["output"]["merged_timeline"])
