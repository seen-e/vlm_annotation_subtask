from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

stage_runner = importlib.import_module(f"{PACKAGE_DIR.name}.stage_runner")


def test_segment_window_only_overrides_runtime_time_bounds(tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("segment", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_build(video_path: Any, **kwargs: Any):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(stage_runner, "build_video_inputs", fake_build)
    context = {
        "input": {"video_path": "/data/source.mp4"},
        "segment": {
            "segment_index": 2,
            "start_time_sec": 10.8,
            "end_time_sec": 25.8,
            "episode_duration_sec": 40.0,
        },
        "frame_cache": {"stage_manifests": {"probe": "/cache/probe/manifest.json"}},
        "stages": {},
    }
    video_config = {
        "fps": 7.0,
        "max_frames": 123,
        "resize_width": 64,
        "jpeg_quality": 90,
        "view_names": ["camera_top"],
    }
    config = {
        "workflow": ["probe"],
        "stages": {
            "probe": {
                "prompt_file": str(prompt),
                "system_prompt": "system",
                "video": video_config,
            }
        },
    }

    stage_runner.run_stage("probe", context, config, dry_run=True)

    assert captured["time_start_sec"] == 10.8
    assert captured["time_end_sec"] == 25.8
    assert captured["fps"] == 7.0
    assert captured["max_frames"] == 123
    assert captured["view_names"] == ["camera_top"]
    assert captured["frame_cache_manifest"] == "/cache/probe/manifest.json"
    assert "全局时间窗 [10.800000, 25.800000)" in context["stages"]["probe"]["prompt"]
    assert "禁止将当前片段重新从 0.0 开始计时" in context["stages"]["probe"]["system_prompt"]


def test_stage_without_segment_does_not_add_runtime_time_bounds(tmp_path: Path, monkeypatch) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("full", encoding="utf-8")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        stage_runner,
        "build_video_inputs",
        lambda video_path, **kwargs: (captured.update(kwargs) or [], {}),
    )
    context = {"input": {"video_path": "/data/source.mp4"}, "stages": {}}
    config = {
        "workflow": ["probe"],
        "stages": {
            "probe": {
                "prompt_file": str(prompt),
                "system_prompt": "system",
                "video": {"fps": 3.0, "max_frames": 10, "resize_width": 32, "jpeg_quality": 80},
            }
        },
    }

    stage_runner.run_stage("probe", context, config, dry_run=True)

    assert "time_start_sec" not in captured
    assert "time_end_sec" not in captured


def test_annotation_contract_uses_actual_sampled_frame_and_per_arm_start() -> None:
    context = {"input": {}, "stages": {}}
    segment = {
        "start_time_sec": 11.0,
        "end_time_sec": 26.0,
        "arm_1_annotation_start_time_sec": 11.0,
        "arm_2_annotation_start_time_sec": 13.0,
    }
    video_meta = {
        "sampled_timestamps": [11.0, 11.25, 12.75, 13.0, 13.25],
        "sampled_frame_indices": [330, 338, 383, 390, 398],
        "primary_view": "camera_top",
    }

    render_context = stage_runner._prompt_render_context(context, segment, video_meta)

    arm_1 = render_context["segment"]["arm_1_annotation_contract"]
    arm_2 = render_context["segment"]["arm_2_annotation_contract"]
    assert "<t=11.00s> <frame_index=330> <camera_top>" in arm_1
    assert "[11.00s, 26.000000s)" in arm_1
    assert "<t=13.00s> <frame_index=390> <camera_top>" in arm_2
    assert "[11.000000s, 13.00s) 仅作为" in arm_2


def test_single_arm_prompt_context_only_contains_arm_1_contract() -> None:
    context = {"input": {}, "stages": {}}
    segment = {
        "start_time_sec": 0.0,
        "end_time_sec": 5.0,
        "arm_1_annotation_start_time_sec": 0.0,
    }
    video_meta = {
        "sampled_timestamps": [0.0, 0.25],
        "sampled_frame_indices": [0, 6],
        "primary_view": "camera_top",
    }

    render_context = stage_runner._prompt_render_context(
        context,
        segment,
        video_meta,
        executors=["arm_1"],
    )

    prompt_segment = render_context["segment"]
    assert "arm_1 标注契约" in prompt_segment["executor_annotation_contracts"]
    assert "arm_2" not in prompt_segment["executor_annotation_contracts"]
    assert "arm_2_annotation_contract" not in prompt_segment


def test_analysis_prompt_context_removes_contact_executors_without_mutating_scene() -> None:
    context = {
        "stages": {
            "scene": {
                "output": {
                    "objects": [
                        {
                            "object_id": "object_1",
                            "category_id": "block",
                            "first_view_time": "0.00s",
                            "category": "building block",
                            "best_view_time": "1.00s",
                            "description": "purple block near the center",
                        },
                        {
                            "object_id": "target_1",
                            "category_id": "tray",
                            "first_view_time": "0.00s",
                            "category": "tray",
                            "best_view_time": "0.00s",
                            "description": "tray on the right",
                        },
                    ],
                    "interaction_objects": [
                        {
                            "object_id": "object_1",
                            "interaction_role": "manipulated_object",
                            "interaction_reason": "visible contact",
                            "contact_executors": ["arm_1"],
                        },
                        {
                            "object_id": "target_1",
                            "interaction_role": "target",
                            "interaction_reason": "receives object",
                            "contact_executors": None,
                        },
                    ]
                }
            }
        }
    }

    render_context = stage_runner._analysis_prompt_context(context)
    rendered_objects = render_context["stages"]["scene"]["output"]["interaction_objects"]
    original_objects = context["stages"]["scene"]["output"]["interaction_objects"]

    assert rendered_objects == [
        {
            "object_id": "object_1",
            "category_id": "block",
            "first_view_time": "0.00s",
            "category": "building block",
            "best_view_time": "1.00s",
            "description": "purple block near the center",
            "interaction_role": "manipulated_object",
            "interaction_reason": "visible contact",
        },
        {
            "object_id": "target_1",
            "category_id": "tray",
            "first_view_time": "0.00s",
            "category": "tray",
            "best_view_time": "0.00s",
            "description": "tray on the right",
            "interaction_role": "target",
            "interaction_reason": "receives object",
        },
    ]
    assert original_objects[0]["contact_executors"] == ["arm_1"]
    assert "contact_executors" in original_objects[1]
