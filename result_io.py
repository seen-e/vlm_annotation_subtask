"""Lightweight result persistence for VLM stage experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultIOError(RuntimeError):
    """Raised for result save/load failures."""


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(value: Any) -> str:
    return str(value)


def save_json(data: Any, path: str | Path) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    except Exception as exc:
        raise ResultIOError(f"Failed to save JSON to {path}: {exc}") from exc


def load_json(path: str | Path) -> Any:
    try:
        p = Path(path)
        if not p.exists():
            raise ResultIOError(f"JSON file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    except ResultIOError:
        raise
    except Exception as exc:
        raise ResultIOError(f"Failed to load JSON from {path}: {exc}") from exc


def save_text(text: str | None, path: str | Path) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("" if text is None else str(text), encoding="utf-8")
    except Exception as exc:
        raise ResultIOError(f"Failed to save text to {path}: {exc}") from exc


def load_text(path: str | Path) -> str:
    try:
        p = Path(path)
        if not p.exists():
            raise ResultIOError(f"Text file not found: {p}")
        return p.read_text(encoding="utf-8")
    except ResultIOError:
        raise
    except Exception as exc:
        raise ResultIOError(f"Failed to load text from {path}: {exc}") from exc


def _safe_name(value: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in str(value))
    return cleaned.strip(" .") or "run"


def make_run_dir(output_dir: str | Path = "outputs", *, run_name: str | None = None, video_id: str | None = None) -> Path:
    name = run_name or video_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return ensure_dir(Path(output_dir) / _safe_name(name))


def _stage_dir(run_dir: str | Path, stage_name: str) -> Path:
    return Path(run_dir) / "stages" / stage_name


def save_stage_result(context: dict[str, Any], stage_name: str, run_dir: str | Path) -> None:
    if stage_name not in context.get("stages", {}):
        raise ResultIOError(f"Cannot save missing stage result: {stage_name}")
    record = context["stages"][stage_name]
    stage_dir = ensure_dir(_stage_dir(run_dir, stage_name))
    save_text(record.get("system_prompt", ""), stage_dir / "system_prompt.txt")
    save_text(record.get("prompt", ""), stage_dir / "user_prompt.txt")
    save_text(record.get("raw_text", ""), stage_dir / "raw_text.txt")
    if "output" not in record:
        raise ResultIOError(f"Stage result has no output: {stage_name}")
    save_json(record.get("output"), stage_dir / "output.json")
    if "output_postprocessed" in record:
        save_json(record.get("output_postprocessed"), stage_dir / "output_postprocessed.json")
    save_json(record.get("video_meta", {}), stage_dir / "video_meta.json")
    if "usage" in record:
        save_json(record.get("usage", {}), stage_dir / "usage.json")


def save_context(context: dict[str, Any], run_dir: str | Path) -> None:
    run_path = ensure_dir(run_dir)
    save_json(context.get("input", {}), run_path / "input.json")
    save_json(context, run_path / "context.json")


def load_stage_output(stage_name: str, run_dir: str | Path) -> dict[str, Any]:
    output = load_json(_stage_dir(run_dir, stage_name) / "output.json")
    if not isinstance(output, dict):
        raise ResultIOError(f"Stage output must be a JSON object for {stage_name}, got {type(output).__name__}")
    return output


def load_stage_into_context(context: dict[str, Any], stage_name: str, run_dir: str | Path) -> None:
    stage_dir = _stage_dir(run_dir, stage_name)
    if not stage_dir.exists():
        raise ResultIOError(f"Stage directory not found: {stage_dir}")
    output_path = stage_dir / "output.json"
    if not output_path.exists():
        raise ResultIOError(f"Stage output.json not found: {output_path}")
    record: dict[str, Any] = {
        "output": load_json(output_path),
        "loaded_from": str(output_path),
    }
    postprocessed_path = stage_dir / "output_postprocessed.json"
    if postprocessed_path.exists():
        record["output_postprocessed"] = load_json(postprocessed_path)
    optional_text = {
        "system_prompt": "system_prompt.txt",
        "prompt": "user_prompt.txt",
        "raw_text": "raw_text.txt",
    }
    for key, filename in optional_text.items():
        path = stage_dir / filename
        if path.exists():
            record[key] = load_text(path)
    meta_path = stage_dir / "video_meta.json"
    if meta_path.exists():
        record["video_meta"] = load_json(meta_path)
    usage_path = stage_dir / "usage.json"
    if usage_path.exists():
        record["usage"] = load_json(usage_path)
    context.setdefault("stages", {})[stage_name] = record


def load_outputs_into_context(context: dict[str, Any], run_dir: str | Path, stage_names: list[str]) -> None:
    for stage_name in stage_names:
        load_stage_into_context(context, stage_name, run_dir)
