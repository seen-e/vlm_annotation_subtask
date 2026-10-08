from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PACKAGE_DIR.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

frame_cache = importlib.import_module(f"{PACKAGE_DIR.name}.frame_cache")
video_process = importlib.import_module(f"{PACKAGE_DIR.name}.video_process")


def test_default_pipeline_enables_frame_cache_and_timestamp_contract() -> None:
    config = yaml.safe_load((PACKAGE_DIR / "config" / "config_pipeline.yaml").read_text(encoding="utf-8"))

    assert config["segment_loop"]["frame_cache"]["enabled"] is True
    assert config["segment_loop"]["postprocessor"]["require_sampled_timestamps"] is True


def _make_video(path: Path, *, duration: float = 4.0, fps: float = 10.0) -> Path:
    size = (32, 24)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for index in range(int(round(duration * fps))):
            writer.write(np.full((size[1], size[0], 3), index % 255, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _config() -> dict:
    return {
        "workflow": ["scene", "analysis", "refinement"],
        "stages": {
            "scene": {"video": {"fps": 1.0, "view_names": ["camera_a"]}},
            "analysis": {"video": {"fps": 2.0, "view_names": ["camera_a", "camera_b"]}},
            "refinement": {"video": {"fps": 2.0, "view_names": ["camera_a"]}},
        },
    }


def test_episode_cache_decodes_shared_view_fps_once_and_reuses_manifest(tmp_path: Path) -> None:
    videos = {
        "camera_a": _make_video(tmp_path / "a.avi"),
        "camera_b": _make_video(tmp_path / "b.avi"),
    }
    cache_root = tmp_path / ".frame_cache"

    state = frame_cache.prepare_episode_frame_cache(videos, _config(), cache_root)

    assert state["reused"] is False
    analysis = json.loads(Path(state["stage_manifests"]["analysis"]).read_text(encoding="utf-8"))
    refinement = json.loads(Path(state["stage_manifests"]["refinement"]).read_text(encoding="utf-8"))
    analysis_a = analysis["views"]["camera_a"]["entries"]
    refinement_a = refinement["views"]["camera_a"]["entries"]
    assert analysis_a == refinement_a
    assert analysis_a[0]["timestamp_sec"] == 0.0
    assert all(
        analysis_a[index]["timestamp_sec"] < analysis_a[index + 1]["timestamp_sec"]
        for index in range(len(analysis_a) - 1)
    )
    assert all((cache_root / entry["path"]).exists() for entry in analysis_a)

    reused = frame_cache.prepare_episode_frame_cache(videos, _config(), cache_root)
    assert reused["reused"] is True
    assert reused["stage_manifests"] == state["stage_manifests"]


def test_episode_cache_uses_ffmpeg_decoder_for_av1(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path / "source.avi")
    calls: list[list[int]] = []

    def fake_ffmpeg_decoder(
        path: Path,
        frames_dir: Path,
        wanted_indices: list[int],
        *,
        jpeg_quality: int,
        source_start_time_sec: float = 0.0,
    ) -> None:
        assert source_start_time_sec == 0.0
        calls.append(list(wanted_indices))
        for frame_index in wanted_indices:
            frame = np.full((24, 32, 3), frame_index % 255, dtype=np.uint8)
            assert cv2.imwrite(str(frames_dir / f"frame_{frame_index:09d}.jpg"), frame)

    monkeypatch.setattr(frame_cache, "_probe_video_codec", lambda path: "av1")
    monkeypatch.setattr(frame_cache, "_decode_frames_with_ffmpeg", fake_ffmpeg_decoder)
    monkeypatch.setattr(
        frame_cache,
        "_decode_frames_with_opencv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("OpenCV must not decode AV1")),
    )

    state = frame_cache.prepare_episode_frame_cache(
        {"camera_a": video},
        {
            "workflow": ["scene"],
            "stages": {"scene": {"video": {"fps": 2.0, "view_names": ["camera_a"]}}},
        },
        tmp_path / ".frame_cache",
    )

    assert calls
    manifest = json.loads(Path(state["stage_manifests"]["scene"]).read_text(encoding="utf-8"))
    assert all((tmp_path / ".frame_cache" / entry["path"]).exists() for entry in manifest["views"]["camera_a"]["entries"])


def test_episode_cache_honors_source_frame_segment_and_uses_local_timeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video = _make_video(tmp_path / "segmented.avi")
    monkeypatch.setattr(frame_cache, "_probe_video_codec", lambda path: "mjpeg")
    state = frame_cache.prepare_episode_frame_cache(
        {"camera_a": video},
        {
            "workflow": ["scene"],
            "stages": {"scene": {"video": {"fps": 2.0, "view_names": ["camera_a"]}}},
        },
        tmp_path / ".frame_cache",
        video_segments={
            "camera_a": {
                "video_path": str(video),
                "start_frame": 10,
                "end_frame": 30,
                "fps": 10.0,
            }
        },
    )

    manifest = json.loads(Path(state["stage_manifests"]["scene"]).read_text(encoding="utf-8"))
    view = manifest["views"]["camera_a"]
    assert view["video_info"]["source_start_frame"] == 10
    assert view["video_info"]["source_end_frame"] == 30
    assert [entry["frame_index"] for entry in view["entries"]] == [0, 5, 10, 15, 19]
    assert [entry["timestamp_sec"] for entry in view["entries"]] == [0.0, 0.5, 1.0, 1.5, 1.9]

    first_frame = cv2.imread(str(tmp_path / ".frame_cache" / view["entries"][0]["path"]))
    assert first_frame is not None
    assert abs(float(first_frame.mean()) - 10.0) < 3.0


def test_segment_reads_cached_multiview_sequence_without_source_decode(tmp_path: Path, monkeypatch) -> None:
    videos = {
        "camera_a": _make_video(tmp_path / "a.avi"),
        "camera_b": _make_video(tmp_path / "b.avi"),
    }
    state = frame_cache.prepare_episode_frame_cache(videos, _config(), tmp_path / ".frame_cache")
    monkeypatch.setattr(
        video_process,
        "_read_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source video must not be decoded")),
    )

    parts, meta = video_process.build_video_inputs(
        videos,
        fps=2.0,
        max_frames=100,
        resize_width=32,
        jpeg_quality=80,
        min_api_frames=1,
        view_names=["camera_a", "camera_b"],
        merge_views=False,
        input_mode="image_sequence",
        add_frame_tags=True,
        time_start_sec=1.0,
        time_end_sec=3.0,
        frame_cache_manifest=state["stage_manifests"]["analysis"],
    )

    assert parts
    assert meta["frame_cache_used"] is True
    assert meta["separate_view_inputs"] is True
    for output in meta["view_outputs"].values():
        assert min(output["sampled_timestamps"]) >= 1.0
        assert max(output["sampled_timestamps"]) < 3.0
