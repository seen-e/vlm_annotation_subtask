from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video_process import VideoProcessError, build_video_inputs


def _make_video(path: Path, *, duration: float, fps: float = 10.0) -> Path:
    size = (32, 24)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for index in range(int(round(duration * fps))):
            writer.write(np.full((size[1], size[0], 3), index % 255, dtype=np.uint8))
    finally:
        writer.release()
    return path


def _build(video_path, **overrides):
    config = {
        "fps": 2.0,
        "max_frames": 100,
        "resize_width": 32,
        "jpeg_quality": 80,
        "max_time": -1,
        "draw_timestamps": False,
        "draw_view_names": False,
        "min_api_frames": 1,
        "merge_views": False,
        "merge_mode": "per_frame",
        "merge_length": 0,
        "input_mode": "image_sequence",
    }
    config.update(overrides)
    return build_video_inputs(video_path, **config)


def test_runtime_window_samples_half_open_global_interval(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "window.avi", duration=20.0)

    _, meta = _build(video, time_start_sec=10.0, time_end_sec=15.0)

    assert meta["effective_start_time"] == pytest.approx(10.0)
    assert meta["effective_end_time"] == pytest.approx(15.0)
    assert min(meta["sampled_timestamps"]) >= 10.0
    assert max(meta["sampled_timestamps"]) < 15.0
    assert meta["sampled_timestamps"][0] == pytest.approx(10.0)


def test_runtime_window_intersects_max_time_and_frame_range(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "intersection.avi", duration=20.0)

    _, meta = _build(
        video,
        time_start_sec=10.0,
        time_end_sec=15.0,
        max_time=13.0,
        frame_start=110,
    )

    assert meta["effective_start_time"] == pytest.approx(11.0)
    assert meta["effective_end_time"] == pytest.approx(13.0)
    assert min(meta["sampled_timestamps"]) >= 11.0
    assert max(meta["sampled_timestamps"]) < 13.0


def test_runtime_window_keeps_stage_specific_fps(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "fps.avi", duration=16.0)

    _, slow = _build(video, fps=1.0, time_start_sec=10.0, time_end_sec=15.0)
    _, fast = _build(video, fps=4.0, time_start_sec=10.0, time_end_sec=15.0)

    assert slow["target_fps"] == pytest.approx(1.0)
    assert fast["target_fps"] == pytest.approx(4.0)
    assert len(fast["sampled_timestamps"]) > len(slow["sampled_timestamps"])


def test_runtime_window_uses_common_multiview_end(tmp_path: Path) -> None:
    long_view = _make_video(tmp_path / "long.avi", duration=20.0)
    short_view = _make_video(tmp_path / "short.avi", duration=12.0)

    _, meta = _build(
        {"long": long_view, "short": short_view},
        view_names=["long", "short"],
        merge_views=True,
        time_start_sec=10.0,
        time_end_sec=15.0,
    )

    assert meta["effective_start_time"] == pytest.approx(10.0)
    assert meta["effective_end_time"] == pytest.approx(12.0)
    assert max(meta["sampled_timestamps"]) < 12.0


def test_omitting_runtime_window_preserves_full_range(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "full.avi", duration=3.0)

    _, meta = _build(video)

    assert meta["effective_start_time"] == pytest.approx(0.0)
    assert meta["effective_end_time"] == pytest.approx(3.0)
    assert meta["configured_time_start_sec"] is None
    assert meta["configured_time_end_sec"] is None


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (float("nan"), 2.0)],
)
def test_invalid_runtime_window_is_rejected(tmp_path: Path, start: float, end: float) -> None:
    video = _make_video(tmp_path / "invalid.avi", duration=4.0)

    with pytest.raises(VideoProcessError):
        _build(video, time_start_sec=start, time_end_sec=end)
