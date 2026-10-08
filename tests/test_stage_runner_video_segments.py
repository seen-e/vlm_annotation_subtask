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


def test_video_layout_describes_effective_interval_and_frame_tags() -> None:
    description = stage_runner.build_video_layout_description(
        {
            "input_mode": "image_sequence",
            "view_names": ["camera_top"],
            "add_frame_tags": True,
        },
        {
            "input_mode": "image_sequence",
            "view_names": ["camera_top"],
            "primary_view": "camera_top",
            "effective_start_time": 0.0,
            "effective_end_time": 5.0,
            "num_sampled_frames": 21,
            "num_output_parts": 21,
            "add_frame_tags": True,
        },
    )

    assert "[0.00s, 5.00s)" in description
    assert "右端点是区间的结束边界" in description
    assert "<frame_index=...>" in description


def _fake_context(video_segments: dict[str, Any], video_path: Any = "/data/source.mp4") -> dict[str, Any]:
    return {
        "input": {
            "video_path": video_path,
            "video_segments": video_segments,
        },
        "stages": {},
    }


def test_prepare_stage_video_input_supports_time_only_segments(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_cut_by_time(**kwargs: Any) -> None:
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")

    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_time", fake_cut_by_time)

    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_time": 1.5,
                "end_time": 3.25,
                "fps": 5.0,
            }
        }
    )

    stage_video_path, clip_paths, cleanup_dirs, clip_meta = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )

    assert calls
    assert calls[0]["source_path"] == "/data/big_camera_top.mp4"
    assert calls[0]["start_time"] == 1.5
    assert calls[0]["end_time"] == 3.25
    assert calls[0]["fps"] == 5.0
    assert stage_video_path == {"camera_top": str(clip_paths[0])}
    assert clip_paths[0].exists()
    assert cleanup_dirs == []
    assert clip_meta[0]["segment_mode"] == "time"
    assert clip_meta[0]["used_full_video"] is False
    assert clip_meta[0]["reused_cached_clip"] is False


def test_prepare_stage_video_input_uses_full_video_for_minus_one_time_pair(tmp_path: Path) -> None:
    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_time": -1,
                "end_time": -1,
            }
        }
    )

    stage_video_path, clip_paths, cleanup_dirs, clip_meta = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )

    assert stage_video_path == {"camera_top": "/data/big_camera_top.mp4"}
    assert clip_paths == []
    assert cleanup_dirs == []
    assert clip_meta[0]["segment_mode"] == "full"
    assert clip_meta[0]["used_full_video"] is True


def test_prepare_stage_video_input_prefers_frame_segments_when_both_are_present(tmp_path: Path, monkeypatch) -> None:
    frame_calls: list[dict[str, Any]] = []
    time_calls: list[dict[str, Any]] = []

    def fake_cut_by_frame(**kwargs: Any) -> None:
        frame_calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")

    def fake_cut_by_time(**kwargs: Any) -> None:
        time_calls.append(kwargs)

    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_frame", fake_cut_by_frame)
    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_time", fake_cut_by_time)

    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_time": 1.0,
                "end_time": 4.0,
                "start_frame": 5,
                "end_frame": 20,
                "fps": 5.0,
            }
        }
    )

    _, clip_paths, _, clip_meta = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )

    assert frame_calls
    assert time_calls == []
    assert frame_calls[0]["start_frame"] == 5
    assert frame_calls[0]["end_frame"] == 20
    assert clip_paths[0].name.startswith("camera_top_frame_")
    assert clip_meta[0]["segment_mode"] == "frame"


def test_prepare_stage_video_input_uses_time_when_frame_pair_is_minus_one(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_cut_by_time(**kwargs: Any) -> None:
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")

    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_time", fake_cut_by_time)

    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_time": 2.0,
                "end_time": 5.0,
                "start_frame": -1,
                "end_frame": -1,
                "fps": 5.0,
            }
        }
    )

    _, _, _, clip_meta = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )

    assert calls
    assert calls[0]["start_time"] == 2.0
    assert calls[0]["end_time"] == 5.0
    assert clip_meta[0]["segment_mode"] == "time"


def test_prepare_stage_video_input_reuses_cached_clip_across_stages(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_cut_by_frame(**kwargs: Any) -> None:
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")

    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_frame", fake_cut_by_frame)

    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_frame": 5,
                "end_frame": 20,
                "fps": 5.0,
            }
        }
    )

    first_video_path, first_clip_paths, _, first_meta = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )
    second_video_path, second_clip_paths, second_cleanup_dirs, second_meta = stage_runner._prepare_stage_video_input(
        stage_name="analysis",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path,
    )

    assert len(calls) == 1
    assert first_clip_paths
    assert second_clip_paths == []
    assert second_cleanup_dirs == []
    assert first_video_path == second_video_path
    assert first_meta[0]["temporary_clip_path"] == second_meta[0]["temporary_clip_path"]
    assert first_meta[0]["reused_cached_clip"] is False
    assert second_meta[0]["reused_cached_clip"] is True


def test_prepare_stage_video_input_can_store_shared_clip_in_system_temp(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    temp_root = tmp_path / "system_temp_clips"

    def fake_mkdtemp(**kwargs: Any) -> str:
        temp_root.mkdir(parents=True, exist_ok=True)
        return str(temp_root)

    def fake_cut_by_frame(**kwargs: Any) -> None:
        calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")

    monkeypatch.setattr(stage_runner.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(stage_runner, "_cut_video_segment_by_frame", fake_cut_by_frame)

    context = _fake_context(
        {
            "camera_top": {
                "video_path": "/data/big_camera_top.mp4",
                "start_frame": 5,
                "end_frame": 20,
                "fps": 5.0,
            }
        }
    )
    context["pipeline"] = {"video_segment_clip_storage": "temp"}

    _, clip_paths, cleanup_dirs, _ = stage_runner._prepare_stage_video_input(
        stage_name="scene",
        context=context,
        video_cfg={"view_names": ["camera_top"]},
        run_dir=tmp_path / "outputs" / "episode",
    )

    assert calls
    assert clip_paths[0].is_relative_to(temp_root)
    assert not clip_paths[0].is_relative_to(tmp_path / "outputs" / "episode")
    assert cleanup_dirs == [temp_root]
