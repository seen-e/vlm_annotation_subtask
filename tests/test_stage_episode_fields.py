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


def _base_context() -> dict[str, Any]:
    return {
        "input": {
            "video_path": "/data/episode.mp4",
            "episode": {
                "fps": 5.0,
                "length": None,
                "video_path": {
                    "camera_front": "/data/front.mp4",
                    "observation.images.image_0": "/data/image_0.mp4",
                },
            },
            "instruction": "test",
        },
        "stages": {},
    }


def _stage_cfg(prompt_path: Path, *, episode_fields: Any = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "prompt_file": str(prompt_path),
        "system_prompt": "system",
        "video": {
            "fps": 1.0,
            "max_frames": 1,
            "resize_width": 32,
            "jpeg_quality": 80,
        },
    }
    if episode_fields is not None:
        cfg["episode_fields"] = episode_fields
    return cfg


def test_stage_episode_fields_are_available_in_ctx_episode(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        (
            "fps={{ ctx.episode.fps }} "
            "length={{ ctx.episode.length }} "
            "front={{ ctx.episode.video_path.camera_front }}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(stage_runner, "build_video_inputs", lambda *args, **kwargs: ([], {}))
    context = _base_context()
    config = {
        "workflow": ["probe"],
        "stages": {
            "probe": _stage_cfg(
                prompt_path,
                episode_fields=["fps", "length", "video_path.camera_front"],
            )
        },
    }

    stage_runner.run_stage("probe", context, config, dry_run=True)

    assert context["stages"]["probe"]["prompt"] == "fps=5.0 length=null front=/data/front.mp4"
    assert context["episode"] == {
        "fps": 5.0,
        "length": None,
        "video_path": {"camera_front": "/data/front.mp4"},
    }


def test_stage_episode_fields_support_direct_keys_that_contain_dots(tmp_path: Path, monkeypatch) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "image0={{ ctx.episode.video_path.observation.images.image_0 }}",
        encoding="utf-8",
    )
    monkeypatch.setattr(stage_runner, "build_video_inputs", lambda *args, **kwargs: ([], {}))
    context = _base_context()
    config = {
        "workflow": ["probe"],
        "stages": {
            "probe": _stage_cfg(
                prompt_path,
                episode_fields=["video_path.observation.images.image_0"],
            )
        },
    }

    stage_runner.run_stage("probe", context, config, dry_run=True)

    assert context["stages"]["probe"]["prompt"] == "image0=/data/image_0.mp4"


def test_stage_without_episode_fields_clears_previous_ctx_episode(tmp_path: Path, monkeypatch) -> None:
    prompt_with_episode = tmp_path / "with_episode.txt"
    prompt_with_episode.write_text("fps={{ ctx.episode.fps }}", encoding="utf-8")
    prompt_without_episode = tmp_path / "without_episode.txt"
    prompt_without_episode.write_text("plain", encoding="utf-8")
    monkeypatch.setattr(stage_runner, "build_video_inputs", lambda *args, **kwargs: ([], {}))
    context = _base_context()
    config = {
        "workflow": ["with_episode", "without_episode"],
        "stages": {
            "with_episode": _stage_cfg(prompt_with_episode, episode_fields=["fps"]),
            "without_episode": _stage_cfg(prompt_without_episode),
        },
    }

    stage_runner.run_stage("with_episode", context, config, dry_run=True)
    assert context["episode"] == {"fps": 5.0}

    stage_runner.run_stage("without_episode", context, config, dry_run=True)

    assert "episode" not in context
    assert context["stages"]["without_episode"]["prompt"] == "plain"
