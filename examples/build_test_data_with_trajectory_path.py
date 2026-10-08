r"""Build per-dataset task JSON files with per-episode trajectory output paths.

Default usage from the repository root:

    python examples/build_test_data_with_trajectory_path.py

The script scans robot_mind2 dataset folders like:

    <root>/<dataset>/videos/<chunk>/observation.rgb_images.<view>/episode_000000.mp4

and writes one JSON file per dataset to ``examples/test_data/<dataset>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = Path(r"C:\Users\34927\Desktop\robot_mind2")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test_data"
DEFAULT_TRAJECTORY_ROOT = SCRIPT_DIR / "trajectories"
VIEW_PREFIX = "observation.rgb_images."


def _json_default(value: Any) -> str:
    return str(value)


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _view_name(view_dir: Path) -> str | None:
    name = view_dir.name
    if not name.startswith(VIEW_PREFIX):
        return None
    view = name[len(VIEW_PREFIX) :].strip()
    return view or None


def _safe_name(value: Any) -> str:
    invalid = '<>:"/\\|?*'
    text = str(value or "item")
    cleaned = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in text)
    return cleaned.strip(" ._") or "item"


def _infer_task(dataset_name: str, *, task_from_name: bool) -> str:
    if not task_from_name:
        return "Unknown task"
    text = dataset_name
    prefixes = ["robogene_twoArm_franka_", "robogene_"]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_", " ").strip()
    return text or "Unknown task"


def resolve_trajectory_path(base_path: str | Path, *, dataset_name: str, chunk_name: str, episode_id: str, total_count: int) -> Path:
    """Return a unique trajectory path for one episode.

    If ``base_path`` is a directory, write under
    ``<base>/<dataset>/<chunk>/<episode_id>/trajectory.json``.
    If it is a JSON file and there is only one record, use it directly.
    If it is a JSON file for multiple records, expand it to sibling files with
    dataset/chunk/episode suffixes to avoid path conflicts in batch main.py.
    """
    base = Path(base_path)
    dataset = _safe_name(dataset_name)
    chunk = _safe_name(chunk_name)
    episode = _safe_name(episode_id)
    if base.suffix.lower() == ".json":
        if total_count == 1:
            return base
        return base.with_name(f"{base.stem}_{dataset}_{chunk}_{episode}{base.suffix}")
    return base / dataset / chunk / episode / "trajectory.json"


def discover_episode_groups(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        videos_dir = dataset_dir / "videos"
        if not videos_dir.exists():
            continue
        for chunk_dir in sorted(path for path in videos_dir.iterdir() if path.is_dir()):
            view_dirs = [path for path in sorted(chunk_dir.iterdir()) if path.is_dir()]
            for view_dir in view_dirs:
                view = _view_name(view_dir)
                if view is None:
                    continue
                for video_file in sorted(view_dir.glob("episode_*.mp4")):
                    episode_id = video_file.stem
                    key = (dataset_dir.name, chunk_dir.name, episode_id)
                    group = groups.setdefault(
                        key,
                        {
                            "dataset_name": dataset_dir.name,
                            "chunk_name": chunk_dir.name,
                            "episode_id": episode_id,
                            "video_path": {},
                        },
                    )
                    group["video_path"][view] = str(video_file.resolve())

    return sorted(groups.values(), key=lambda item: (item["dataset_name"], item["chunk_name"], item["episode_id"]))


def build_task_records(
    *,
    data_root: str | Path,
    trajectory_path: str | Path,
    task_from_name: bool = False,
) -> list[dict[str, Any]]:
    groups = discover_episode_groups(data_root)
    total_count = len(groups)
    records: list[dict[str, Any]] = []
    seen_paths: dict[str, str] = {}
    for group in groups:
        output_path = resolve_trajectory_path(
            trajectory_path,
            dataset_name=group["dataset_name"],
            chunk_name=group["chunk_name"],
            episode_id=group["episode_id"],
            total_count=total_count,
        )
        output_key = str(output_path.resolve()).casefold()
        if output_key in seen_paths:
            raise ValueError(f"trajectory_path conflict for {group['episode_id']}: {output_path}")
        seen_paths[output_key] = group["episode_id"]

        video_path = dict(sorted(group["video_path"].items()))
        records.append(
            {
                "episode_id": group["episode_id"],
                "dataset_name": group["dataset_name"],
                "chunk": group["chunk_name"],
                "video_path": video_path,
                "task": _infer_task(group["dataset_name"], task_from_name=task_from_name),
                "trajectory_path": str(output_path.resolve()),
            }
        )
    return records


def build_task_records_by_dataset(
    *,
    data_root: str | Path,
    trajectory_path: str | Path,
    task_from_name: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    records = build_task_records(
        data_root=data_root,
        trajectory_path=trajectory_path,
        task_from_name=task_from_name,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["dataset_name"]), []).append(record)
    return dict(sorted(grouped.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test_data JSON with trajectory_path for robot_mind2 videos.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory containing dataset folders.")
    parser.add_argument(
        "--trajectory-path",
        default=str(DEFAULT_TRAJECTORY_ROOT),
        help="Trajectory output directory, or a .json file for a single/suffixed output path.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for per-dataset JSON files.")
    parser.add_argument(
        "--task-from-name",
        action="store_true",
        help="Use dataset folder names as simple task text instead of 'Unknown task'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_records = build_task_records_by_dataset(
        data_root=args.data_root,
        trajectory_path=args.trajectory_path,
        task_from_name=args.task_from_name,
    )
    output_dir = Path(args.output_dir)
    total_count = 0
    for dataset_name, records in grouped_records.items():
        total_count += len(records)
        output_path = output_dir / f"{_safe_name(dataset_name)}.json"
        save_json(records, output_path)
        print(f"Wrote {len(records):4d} episodes -> {output_path.resolve()}")
    print(f"Found {total_count} episodes in {len(grouped_records)} dataset folders")


if __name__ == "__main__":
    main()
