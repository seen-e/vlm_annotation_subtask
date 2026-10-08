"""Merge per-episode trajectory.json files into chunk-level trajectory.json files."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_TRAJECTORY_ROOT = Path(r"C:\Users\34927\Desktop\trajectories")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_atomic(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
        Path(tmp_name).replace(output_path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def iter_chunk_dirs(root: str | Path) -> list[Path]:
    trajectory_root = Path(root)
    if not trajectory_root.exists():
        raise FileNotFoundError(f"Trajectory root not found: {trajectory_root}")

    chunk_dirs: list[Path] = []
    for task_dir in sorted(path for path in trajectory_root.iterdir() if path.is_dir()):
        for chunk_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            chunk_dirs.append(chunk_dir)
    return chunk_dirs


def merge_chunk(chunk_dir: str | Path, *, output_name: str = "trajectory.json") -> tuple[Path, int]:
    chunk_path = Path(chunk_dir)
    episode_records: dict[str, dict[str, Any]] = {}
    for episode_dir in sorted(path for path in chunk_path.iterdir() if path.is_dir()):
        trajectory_path = episode_dir / "trajectory.json"
        if not trajectory_path.exists():
            continue
        data = load_json(trajectory_path)
        if not isinstance(data, dict):
            raise ValueError(f"Episode trajectory must be a JSON object: {trajectory_path}")
        episode_id = data.get("episode_id") or episode_dir.name
        episode_key = str(episode_id)
        if episode_key in episode_records:
            raise ValueError(f"Duplicate episode_id {episode_key!r} under {chunk_path}")
        episode_records[episode_key] = data

    output_path = chunk_path / output_name
    save_json_atomic(output_path, episode_records)
    return output_path, len(episode_records)


def merge_all(root: str | Path, *, output_name: str = "trajectory.json") -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    for chunk_dir in iter_chunk_dirs(root):
        output_path, count = merge_chunk(chunk_dir, output_name=output_name)
        results.append((output_path, count))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge episode trajectory.json files into each chunk directory.")
    parser.add_argument("--root", default=str(DEFAULT_TRAJECTORY_ROOT), help="Root trajectory directory.")
    parser.add_argument("--output-name", default="trajectory.json", help="Merged JSON filename written under each chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = merge_all(args.root, output_name=args.output_name)
    total = 0
    for output_path, count in results:
        total += count
        print(f"Merged {count:4d} episodes -> {output_path}")
    print(f"Done. chunks={len(results)}, episodes={total}")


if __name__ == "__main__":
    main()
