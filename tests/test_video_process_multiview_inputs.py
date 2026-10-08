from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from model_client import _build_messages
from video_process import build_video_inputs


def _make_video(path: Path, *, duration: float, fps: float = 10.0) -> Path:
    size = (64, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    try:
        for index in range(int(round(duration * fps))):
            frame = np.full((size[1], size[0], 3), index % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _build(video_path, **overrides):
    config = {
        "fps": 1.0,
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


def test_single_view_non_merge_keeps_legacy_message_shape(tmp_path: Path) -> None:
    top = _make_video(tmp_path / "top.avi", duration=2.0)
    front = _make_video(tmp_path / "front.avi", duration=2.0)

    parts, meta = _build(
        {"camera_top": top, "camera_front": front},
        view_names=["camera_top"],
    )

    assert parts
    assert all(part["type"] == "image_url" for part in parts)
    assert meta["separate_view_inputs"] is False
    assert meta["message_view_labels"] is False
    assert meta["view_names"] == ["camera_top"]
    assert meta["num_message_parts"] == meta["num_output_parts"]


def test_single_view_video_mode_has_no_extra_view_label(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "single_video.avi", duration=1.0)

    parts, meta = _build(video, view_names=["camera_top"], input_mode="video", fps=2.0)

    assert [part["type"] for part in parts] == ["video_url"]
    assert meta["separate_view_inputs"] is False
    assert meta["video_encoding"]["container"] == "mp4"


def test_image_sequence_can_add_timestamp_frame_index_and_view_tag_before_each_image(
    tmp_path: Path,
) -> None:
    video = _make_video(tmp_path / "tagged.avi", duration=1.0)

    parts, meta = _build(
        video,
        view_names=["camera_top"],
        fps=1.0,
        add_frame_tags=True,
    )

    assert meta["add_frame_tags_requested"] is True
    assert meta["add_frame_tags"] is True
    assert len(parts) == meta["num_output_parts"] * 2
    assert [part["type"] for part in parts] == ["text", "image_url", "text", "image_url"]
    assert parts[0]["text"] == "<t=0.00s> <frame_index=0> <camera_top>"
    assert parts[2]["text"] == "<t=0.90s> <frame_index=9> <camera_top>"
    assert meta["output_groups"][0]["message_tag_part_index"] == 0
    assert meta["output_groups"][0]["message_media_part_index"] == 1


def test_frame_tags_list_all_views_for_merged_image(tmp_path: Path) -> None:
    front = _make_video(tmp_path / "merged_front.avi", duration=1.0)
    top = _make_video(tmp_path / "merged_top.avi", duration=1.0)

    parts, meta = _build(
        {"camera_front": front, "camera_top": top},
        view_names=["camera_front", "camera_top"],
        merge_views=True,
        max_frames=1,
        add_frame_tags=True,
    )

    assert [part["type"] for part in parts] == ["text", "image_url"]
    assert parts[0]["text"] == "<t=0.00s> <frame_index=0> <camera_front> <camera_top>"
    assert meta["num_output_parts"] == 1


def test_independent_multiview_sequences_keep_group_and_frame_tags(tmp_path: Path) -> None:
    front = _make_video(tmp_path / "tagged_front.avi", duration=1.0)
    top = _make_video(tmp_path / "tagged_top.avi", duration=1.0)

    parts, meta = _build(
        {"camera_front": front, "camera_top": top},
        view_names=["camera_front", "camera_top"],
        max_frames=1,
        add_frame_tags=True,
    )

    assert [part["type"] for part in parts] == [
        "text", "text", "image_url", "text", "text", "image_url"
    ]
    assert "camera_front" in parts[0]["text"]
    assert parts[1]["text"] == "<t=0.00s> <frame_index=0> <camera_front>"
    assert "camera_top" in parts[3]["text"]
    assert parts[4]["text"] == "<t=0.00s> <frame_index=0> <camera_top>"
    assert meta["num_output_parts"] == 2
    assert meta["num_message_parts"] == 6
    assert meta["output_groups"][0]["message_tag_part_index"] == 1
    assert meta["output_groups"][0]["message_media_part_index"] == 2
    assert meta["output_groups"][1]["message_tag_part_index"] == 4
    assert meta["output_groups"][1]["message_media_part_index"] == 5


def test_frame_tags_are_ignored_for_video_mode(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "video_tags.avi", duration=1.0)

    parts, meta = _build(video, input_mode="video", add_frame_tags=True)

    assert [part["type"] for part in parts] == ["video_url"]
    assert meta["add_frame_tags_requested"] is True
    assert meta["add_frame_tags"] is False


def test_multiview_non_merge_processes_each_view_independently(tmp_path: Path) -> None:
    front = _make_video(tmp_path / "front.avi", duration=3.0)
    top = _make_video(tmp_path / "top.avi", duration=1.0)
    save_dir = tmp_path / "processed"

    parts, meta = _build(
        {"camera_front": front, "camera_top": top},
        view_names=["camera_front", "camera_top"],
        max_time=2.0,
        save_processed_path=save_dir,
    )

    front_meta = meta["view_outputs"]["camera_front"]
    top_meta = meta["view_outputs"]["camera_top"]
    assert meta["separate_view_inputs"] is True
    assert front_meta["effective_duration"] == 2.0
    assert top_meta["effective_duration"] == 1.0
    assert front_meta["num_sampled_frames"] > top_meta["num_sampled_frames"]
    assert meta["was_limited_by_view_duration"] is False
    assert parts[front_meta["message_label_part_index"]]["type"] == "text"
    assert "camera_front" in parts[front_meta["message_label_part_index"]]["text"]
    assert parts[top_meta["message_label_part_index"]]["type"] == "text"
    assert "camera_top" in parts[top_meta["message_label_part_index"]]["text"]
    assert [group["output_index"] for group in meta["output_groups"]] == list(range(meta["num_output_parts"]))
    assert (save_dir / "camera_front" / "frame_000000.jpg").exists()
    assert (save_dir / "camera_top" / "frame_000000.jpg").exists()
    assert not (save_dir / "frame_000000.jpg").exists()


def test_multiview_video_mode_emits_labeled_video_groups(tmp_path: Path) -> None:
    front = _make_video(tmp_path / "front.avi", duration=1.0)
    top = _make_video(tmp_path / "top.avi", duration=1.0)

    parts, meta = _build(
        {"camera_front": front, "camera_top": top},
        view_names=["camera_front", "camera_top"],
        input_mode="video",
        fps=2.0,
    )

    assert [part["type"] for part in parts] == ["text", "video_url", "text", "video_url"]
    assert meta["num_output_parts"] == 2
    assert meta["num_message_parts"] == 4
    for part in (parts[1], parts[3]):
        url = part["video_url"]["url"]
        assert url.startswith("data:video/mp4;base64,")
        payload = base64.b64decode(url.split(",", 1)[1])
        assert len(payload) > 100
        assert payload[4:8] == b"ftyp"


def test_mixed_view_labels_are_preserved_when_building_messages() -> None:
    media_parts = [
        {"type": "text", "text": "camera_front"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
    ]

    messages = _build_messages("system", "user", media_parts, model="qwen-test")

    assert messages[0]["content"][1:] == media_parts
