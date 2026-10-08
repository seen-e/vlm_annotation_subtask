"""Episode-level image sequence cache shared by segmented stage runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2

from .video_process import compute_sample_timestamps, normalize_video_input, read_video_info


CACHE_VERSION = 1


class FrameCacheError(RuntimeError):
    """Raised when the episode frame cache cannot be prepared."""


def _safe_name(value: Any) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in str(value))
    return cleaned.strip(" .") or "stage"


def _source_key(path: Path, segment: dict[str, Any] | None = None) -> str:
    payload = {
        "path": str(path.resolve()),
        "segment": segment or None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _fps_key(fps: float) -> str:
    return hashlib.sha256(f"{fps:.12g}".encode("ascii")).hexdigest()[:12]


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _cache_spec(
    video_path: Any,
    config: dict[str, Any],
    video_segments: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workflow = config.get("workflow") or []
    stages_cfg = config.get("stages") or {}
    stage_specs: dict[str, Any] = {}
    source_specs: dict[str, Any] = {}

    for stage_name in workflow:
        stage_cfg = stages_cfg.get(stage_name) or {}
        video_cfg = stage_cfg.get("video") or {}
        try:
            target_fps = float(video_cfg["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameCacheError(f"stage {stage_name!r} requires a valid video.fps for frame caching") from exc
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise FrameCacheError(f"stage {stage_name!r} video.fps must be finite and > 0")

        view_map, _ = normalize_video_input(video_path, view_names=video_cfg.get("view_names"))
        stage_views: dict[str, Any] = {}
        for view_name, source_path in view_map.items():
            segment = None
            if isinstance(video_segments, dict) and view_name in video_segments:
                raw_segment = video_segments[view_name]
                if not isinstance(raw_segment, dict):
                    raise FrameCacheError(f"video segment for view {view_name!r} must be an object")
                segment = {
                    key: raw_segment[key]
                    for key in ("start_frame", "end_frame", "start_time", "end_time", "fps")
                    if key in raw_segment
                }
                if raw_segment.get("video_path") is not None:
                    source_path = Path(raw_segment["video_path"])
            resolved = source_path.resolve()
            key = _source_key(resolved, segment)
            source = source_specs.setdefault(
                key,
                {
                    "path": resolved,
                    "signature": _source_signature(resolved),
                    "segment": segment,
                    "fps_requests": {},
                },
            )
            source["fps_requests"][_fps_key(target_fps)] = target_fps
            stage_views[view_name] = {"source_key": key, "source_path": str(resolved)}
        stage_specs[stage_name] = {"fps": target_fps, "views": stage_views}

    fingerprint_data = {
        "version": CACHE_VERSION,
        "workflow": list(workflow),
        "stages": stage_specs,
        "sources": {
            key: {
                "signature": value["signature"],
                "segment": value.get("segment"),
                "fps_requests": value["fps_requests"],
            }
            for key, value in source_specs.items()
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {"fingerprint": fingerprint, **fingerprint_data}, source_specs


def _valid_existing_cache(cache_root: Path, spec: dict[str, Any]) -> bool:
    index_path = cache_root / "index.json"
    if not index_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("fingerprint") != spec["fingerprint"]:
            return False
        for manifest_path in (index.get("stage_manifests") or {}).values():
            manifest_file = Path(manifest_path)
            if not manifest_file.exists():
                return False
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            for view in (manifest.get("views") or {}).values():
                for entry in view.get("entries") or []:
                    if not (cache_root / entry["path"]).exists():
                        return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _probe_video_codec(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    codec = completed.stdout.strip().splitlines()
    return codec[0].lower() if codec else None


def _fraction(value: Any) -> float:
    text = str(value or "0")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(text)


def _read_video_info_with_ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise FrameCacheError(f"ffprobe is unavailable for {path}: {exc}") from exc
    if completed.returncode != 0:
        raise FrameCacheError(f"ffprobe failed for {path}: {completed.stderr.strip()[-1000:]}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
        duration = float(stream.get("duration") or (payload.get("format") or {}).get("duration") or 0.0)
        frame_count = int(stream.get("nb_frames") or 0)
        if frame_count <= 0 and fps > 0 and duration > 0:
            frame_count = int(round(duration * fps))
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrameCacheError(f"invalid ffprobe metadata for {path}") from exc
    if fps <= 0 or frame_count <= 0:
        raise FrameCacheError(
            f"ffprobe returned invalid fps/frame_count for {path}: fps={fps}, frames={frame_count}"
        )
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def _episode_video_info(
    path: Path,
    source: dict[str, Any],
    *,
    codec: str | None,
) -> tuple[dict[str, Any], int]:
    full_info = _read_video_info_with_ffprobe(path) if codec == "av1" else read_video_info(path)
    segment = source.get("segment")
    if not isinstance(segment, dict) or not segment:
        return full_info, 0

    source_fps = float(segment.get("fps") or full_info["fps"])
    if source_fps <= 0:
        raise FrameCacheError(f"video segment fps must be > 0 for {path}")
    if "start_frame" in segment and "end_frame" in segment:
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
    elif "start_time" in segment and "end_time" in segment:
        start_frame = int(round(float(segment["start_time"]) * source_fps))
        end_frame = int(round(float(segment["end_time"]) * source_fps))
    else:
        raise FrameCacheError(
            f"video segment for {path} must provide start_frame/end_frame or start_time/end_time"
        )
    if start_frame == -1 and end_frame == -1:
        return full_info, 0
    if start_frame < 0 or end_frame <= start_frame:
        raise FrameCacheError(f"invalid video segment frame range for {path}: [{start_frame}, {end_frame})")
    if end_frame > int(full_info["frame_count"]):
        raise FrameCacheError(
            f"video segment end frame {end_frame} exceeds source frame count "
            f"{full_info['frame_count']} for {path}"
        )
    frame_count = end_frame - start_frame
    return {
        "path": str(path),
        "fps": source_fps,
        "frame_count": frame_count,
        "width": int(full_info.get("width") or 0),
        "height": int(full_info.get("height") or 0),
        "duration": frame_count / source_fps,
        "source_start_frame": start_frame,
        "source_end_frame": end_frame,
    }, start_frame


def _decode_frames_with_opencv(
    path: Path,
    frames_dir: Path,
    wanted_indices: list[int],
    *,
    jpeg_quality: int,
    source_start_frame: int = 0,
) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FrameCacheError(f"failed to open source video for caching: {path}")
    try:
        if source_start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        wanted = iter(wanted_indices)
        target_index = next(wanted, None)
        frame_index = 0
        while target_index is not None:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise FrameCacheError(f"failed to decode frame {frame_index} from {path}")
            if frame_index == target_index:
                output_path = frames_dir / f"frame_{frame_index:09d}.jpg"
                if not cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
                ):
                    raise FrameCacheError(f"failed to save cached frame {frame_index} from {path}")
                target_index = next(wanted, None)
            frame_index += 1
    finally:
        cap.release()


def _decode_frames_with_ffmpeg(
    path: Path,
    frames_dir: Path,
    wanted_indices: list[int],
    *,
    jpeg_quality: int,
    source_start_time_sec: float = 0.0,
) -> None:
    if not wanted_indices:
        return
    selected_dir = frames_dir.parent / ".ffmpeg_selected"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    selected_dir.mkdir(parents=True)
    select_expression = "+".join(f"eq(n\\,{index})" for index in wanted_indices)
    qscale = max(2, min(31, round(2 + (100 - int(jpeg_quality)) * 29 / 99)))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    if source_start_time_sec > 0:
        command.extend(["-ss", f"{source_start_time_sec:.12f}"])
    command.extend(
        [
            "-i",
            str(path),
            "-vf",
            f"select={select_expression}",
            "-vsync",
            "0",
            "-q:v",
            str(qscale),
            "-start_number",
            "0",
            str(selected_dir / "selected_%09d.jpg"),
        ]
    )
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise FrameCacheError(f"ffmpeg is unavailable for decoding {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise FrameCacheError(f"ffmpeg failed to decode {path}: {detail}")

    selected_frames = sorted(selected_dir.glob("selected_*.jpg"))
    if len(selected_frames) != len(wanted_indices):
        raise FrameCacheError(
            f"ffmpeg decoded {len(selected_frames)} selected frames from {path}; "
            f"expected {len(wanted_indices)}"
        )
    for selected_path, frame_index in zip(selected_frames, wanted_indices, strict=True):
        selected_path.replace(frames_dir / f"frame_{frame_index:09d}.jpg")
    selected_dir.rmdir()


def _decode_source_once(
    source_key: str,
    source: dict[str, Any],
    build_root: Path,
    *,
    jpeg_quality: int,
) -> tuple[dict[str, Any], dict[str, tuple[list[int], list[float]]]]:
    path = Path(source["path"])
    codec = _probe_video_codec(path)
    info, source_start_frame = _episode_video_info(path, source, codec=codec)
    source_start_time_sec = source_start_frame / float(info["fps"])
    sequences: dict[str, tuple[list[int], list[float]]] = {}
    wanted_indices: set[int] = set()
    for fps_key, target_fps in source["fps_requests"].items():
        indices, timestamps = compute_sample_timestamps(
            original_fps=float(info["fps"]),
            frame_count=int(info["frame_count"]),
            target_fps=float(target_fps),
            max_frames=int(info["frame_count"]),
            min_api_frames=1,
        )
        sequences[fps_key] = (indices, timestamps)
        wanted_indices.update(indices)

    frames_dir = build_root / "sources" / source_key / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sorted_indices = sorted(wanted_indices)
    if codec == "av1":
        _decode_frames_with_ffmpeg(
            path,
            frames_dir,
            sorted_indices,
            jpeg_quality=jpeg_quality,
            source_start_time_sec=source_start_time_sec,
        )
    else:
        try:
            _decode_frames_with_opencv(
                path,
                frames_dir,
                sorted_indices,
                jpeg_quality=jpeg_quality,
                source_start_frame=source_start_frame,
            )
        except FrameCacheError as opencv_error:
            shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True)
            try:
                _decode_frames_with_ffmpeg(
                    path,
                    frames_dir,
                    sorted_indices,
                    jpeg_quality=jpeg_quality,
                    source_start_time_sec=source_start_time_sec,
                )
            except FrameCacheError as ffmpeg_error:
                raise FrameCacheError(
                    f"{opencv_error}; ffmpeg fallback also failed: {ffmpeg_error}"
                ) from ffmpeg_error
    return info, sequences


def prepare_episode_frame_cache(
    video_path: Any,
    config: dict[str, Any],
    cache_root: str | Path,
    *,
    jpeg_quality: int = 95,
    video_segments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode each episode-local source range once and build timestamped stage manifests."""
    quality = int(jpeg_quality)
    if not 1 <= quality <= 100:
        raise FrameCacheError(f"frame cache jpeg_quality must be in [1, 100], got {quality}")
    root = Path(cache_root)
    spec, source_specs = _cache_spec(video_path, config, video_segments)
    if _valid_existing_cache(root, spec):
        index = json.loads((root / "index.json").read_text(encoding="utf-8"))
        return {"root": str(root), "stage_manifests": index["stage_manifests"], "reused": True}

    build_root = root.with_name(f".{root.name}.build.{os.getpid()}")
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    try:
        decoded: dict[str, Any] = {}
        for source_key, source in source_specs.items():
            info, sequences = _decode_source_once(
                source_key,
                source,
                build_root,
                jpeg_quality=quality,
            )
            decoded[source_key] = {"video_info": info, "sequences": sequences}

        stage_manifests: dict[str, str] = {}
        for stage_name, stage_spec in spec["stages"].items():
            fps_key = _fps_key(float(stage_spec["fps"]))
            manifest: dict[str, Any] = {
                "version": CACHE_VERSION,
                "stage_name": stage_name,
                "target_fps": stage_spec["fps"],
                "views": {},
            }
            for view_name, view_spec in stage_spec["views"].items():
                source_key = view_spec["source_key"]
                info = decoded[source_key]["video_info"]
                indices, timestamps = decoded[source_key]["sequences"][fps_key]
                entries = [
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp,
                        "path": f"sources/{source_key}/frames/frame_{frame_index:09d}.jpg",
                    }
                    for frame_index, timestamp in zip(indices, timestamps, strict=True)
                ]
                manifest["views"][view_name] = {
                    "source_path": view_spec["source_path"],
                    "video_info": info,
                    "entries": entries,
                }
            relative_manifest = Path("stages") / _safe_name(stage_name) / "manifest.json"
            _write_json(manifest, build_root / relative_manifest)
            stage_manifests[stage_name] = str(root / relative_manifest)

        index = {
            **spec,
            "jpeg_quality": quality,
            "stage_manifests": stage_manifests,
        }
        _write_json(index, build_root / "index.json")
        if root.exists():
            shutil.rmtree(root)
        build_root.replace(root)
    except Exception:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise
    return {"root": str(root), "stage_manifests": stage_manifests, "reused": False}


def cleanup_episode_frame_cache(cache_state: dict[str, Any] | None) -> None:
    if not cache_state:
        return
    root_value = cache_state.get("root")
    if not root_value:
        return
    root = Path(root_value)
    if root.name != ".frame_cache":
        raise FrameCacheError(f"refusing to remove unexpected cache path: {root}")
    if root.exists():
        shutil.rmtree(root)
