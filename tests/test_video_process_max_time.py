from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from video_process import VideoProcessError, build_video_inputs


def _make_video(path: Path, *, fps: float = 10.0, duration: float = 5.0, size: tuple[int, int] = (64, 48)) -> Path:
    frame_count = int(round(fps * duration))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), f"failed to open test video writer: {path}"
    try:
        for idx in range(frame_count):
            frame = np.full((size[1], size[0], 3), idx % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _build(path, **overrides):
    cfg = {
        "fps": 1.0,
        "max_frames": 100,
        "resize_width": 32,
        "jpeg_quality": 80,
        "draw_timestamps": False,
        "draw_view_names": False,
        "min_api_frames": 1,
        "merge_views": False,
        "merge_mode": "per_frame",
        "merge_length": 0,
        "input_mode": "image_sequence",
    }
    cfg.update(overrides)
    return build_video_inputs(path, **cfg)


def test_max_time_minus_one_uses_full_video(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "full.avi", duration=5.0)
    _, meta = _build(video, max_time=-1)

    assert meta["configured_max_time"] == -1
    assert meta["effective_duration"] == pytest.approx(5.0)
    assert meta["was_time_limited"] is False
    assert max(meta["sampled_timestamps"]) < 5.0


def test_max_time_truncates_long_video_before_sampling(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "long.avi", duration=5.0)
    _, meta = _build(video, max_time=2.0)

    assert meta["configured_max_time"] == pytest.approx(2.0)
    assert meta["effective_duration"] == pytest.approx(2.0)
    assert meta["was_time_limited"] is True
    assert max(meta["sampled_timestamps"]) < 2.0
    assert max(meta["sampled_frame_indices"]) < 20


def test_max_time_longer_than_video_uses_full_video(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "short.avi", duration=2.0)
    _, meta = _build(video, max_time=5.0)

    assert meta["configured_max_time"] == pytest.approx(5.0)
    assert meta["effective_duration"] == pytest.approx(2.0)
    assert meta["was_time_limited"] is False
    assert max(meta["sampled_timestamps"]) < 2.0


@pytest.mark.parametrize("bad_max_time", [0, -2])
def test_invalid_max_time_fails_validation(tmp_path: Path, bad_max_time: int) -> None:
    video = _make_video(tmp_path / f"bad_{bad_max_time}.avi", duration=1.0)

    with pytest.raises(VideoProcessError, match="max_time must be -1 or > 0"):
        _build(video, max_time=bad_max_time)


def test_multiview_uses_common_effective_time_without_padding_short_view(tmp_path: Path) -> None:
    top = _make_video(tmp_path / "top.avi", duration=5.0)
    front = _make_video(tmp_path / "front.avi", duration=2.0)

    _, meta = _build(
        {"camera_top": top, "camera_front": front},
        max_time=4.0,
        merge_views=True,
        view_names=["camera_top", "camera_front"],
    )

    assert meta["original_duration"] == pytest.approx(5.0)
    assert meta["effective_duration"] == pytest.approx(2.0)
    assert meta["was_time_limited"] is True
    assert meta["was_limited_by_view_duration"] is True
    assert max(meta["sampled_timestamps"]) < 2.0
    front_indices = [
        group["per_view_frame_indices"]["camera_front"]
        for group in meta["output_groups"]
        if "per_view_frame_indices" in group
    ]
    assert front_indices
    assert max(front_indices) < 20


def test_max_time_combines_with_fps_and_max_frames(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "combined.avi", duration=10.0)
    _, meta = _build(video, max_time=5.0, fps=4.0, max_frames=3)

    assert meta["effective_duration"] == pytest.approx(5.0)
    assert meta["num_sampled_frames"] == 3
    assert len(meta["sampled_timestamps"]) == 3
    assert max(meta["sampled_timestamps"]) < 5.0


def test_timestamps_remain_on_original_time_axis_with_frame_start(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "offset.avi", duration=6.0)
    _, meta = _build(video, max_time=4.0, fps=1.0, frame_start=20)

    assert meta["effective_start_time"] == pytest.approx(2.0)
    assert meta["effective_end_time"] == pytest.approx(4.0)
    assert meta["effective_duration"] == pytest.approx(2.0)
    assert min(meta["sampled_timestamps"]) >= 2.0
    assert max(meta["sampled_timestamps"]) < 4.0
