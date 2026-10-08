"""Export episode video clips for completed scene predictions.

This utility scans an output directory for runs with ``stages/scene/output.json``,
matches each completed episode against a BridgeData paths JSON, and cuts the
per-view episode clips into a destination directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_TASK_DIR = "/data/xuchacha/bridgedata_task"
DEFAULT_PATHS_JSON = (
    "/home/xuchacha/vlm_auto_annotation_refactor_gpt/examples/test_data/"
    "bridgedata2_lerobot_v3_2stage_input_111.198.58.150_paths.json"
)
DEFAULT_OUTPUT_DIR = "/data/xuchacha/bridgedata_video_episode"


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [item for item in data if isinstance(item, dict)]


def _safe_name(value: Any, *, default: str = "item") -> str:
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or default))
    text = re.sub(r"\s+", "_", text).strip(" ._")
    return text or default


def _episode_id_from_run_name(run_name: str) -> str:
    match = re.search(r"(episode_\d+)", run_name)
    return match.group(1) if match else ""


def _find_completed_scene_runs(task_dir: Path) -> list[dict[str, str]]:
    proc = subprocess.run(
        ["find", str(task_dir), "-path", "*/stages/scene/output.json", "-type", "f", "-size", "+0c"],
        check=True,
        capture_output=True,
        text=True,
    )
    runs: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        scene_output_path = Path(line)
        run_dir = scene_output_path.parents[2]
        episode_id = _episode_id_from_run_name(run_dir.name)
        if not episode_id:
            input_path = run_dir / "input.json"
            if input_path.exists():
                try:
                    input_data = json.loads(input_path.read_text(encoding="utf-8-sig"))
                    episode = input_data.get("episode") if isinstance(input_data, dict) else {}
                    episode_id = str(
                        input_data.get("video_id")
                        or input_data.get("episode_id")
                        or (episode.get("episode_id") if isinstance(episode, dict) else "")
                    )
                except Exception:
                    episode_id = ""
        if episode_id:
            runs.append(
                {
                    "episode_id": episode_id,
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir),
                    "scene_output_path": str(scene_output_path),
                }
            )
    runs.sort(key=lambda item: item["run_name"])
    return runs


def _task_label(item: dict[str, Any]) -> str:
    task = item.get("task")
    if task:
        return str(task)
    tasks = item.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return "unknown_task"


def _select_runs(
    *,
    completed_runs: list[dict[str, str]],
    by_episode: dict[str, dict[str, Any]],
    limit_episodes: int | None,
    sample_episodes: int | None,
    per_task_limit: int | None,
    seed: int,
    excluded_tasks: set[str] | None = None,
) -> list[dict[str, str]]:
    if sample_episodes is None:
        return completed_runs[:limit_episodes] if limit_episodes is not None else completed_runs
    if limit_episodes is not None:
        raise ValueError("--limit-episodes and --sample-episodes cannot be used together")
    if sample_episodes < 0:
        raise ValueError("--sample-episodes must be >= 0")
    if per_task_limit is not None and per_task_limit < 1:
        raise ValueError("--per-task-limit must be >= 1")

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = {}
    for run in completed_runs:
        item = by_episode.get(run["episode_id"])
        if item is None:
            continue
        task_label = _task_label(item)
        if excluded_tasks and task_label in excluded_tasks:
            continue
        grouped.setdefault(task_label, []).append(run)

    for runs in grouped.values():
        rng.shuffle(runs)

    task_labels = list(grouped)
    rng.shuffle(task_labels)
    selected: list[dict[str, str]] = []
    if per_task_limit is not None:
        eligible_task_labels = [label for label in task_labels if len(grouped[label]) >= per_task_limit]
        full_group_count, remainder = divmod(sample_episodes, per_task_limit)
        needed_task_groups = full_group_count + (1 if remainder else 0)
        if len(eligible_task_labels) < needed_task_groups:
            raise ValueError(
                "Not enough task groups with at least "
                f"{per_task_limit} episodes: need {needed_task_groups}, got {len(eligible_task_labels)}"
            )
        for index, task_label in enumerate(eligible_task_labels[:needed_task_groups]):
            take_count = per_task_limit if index < full_group_count else remainder
            selected.extend(grouped[task_label][:take_count])
        selected.sort(key=lambda item: item["run_name"])
        return selected

    cap = sample_episodes
    for task_label in task_labels:
        if len(selected) >= sample_episodes:
            break
        remaining = sample_episodes - len(selected)
        selected.extend(grouped[task_label][: min(cap, remaining)])

    selected.sort(key=lambda item: item["run_name"])
    return selected


def _build_jobs(
    *,
    selected_runs: list[dict[str, str]],
    by_episode: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    jobs: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for run in selected_runs:
        item = by_episode.get(run["episode_id"])
        if item is None:
            missing.append(run)
            continue
        segments = item.get("video_segments")
        if not isinstance(segments, dict) or not segments:
            missing.append(run)
            continue
        episode_dir = output_dir / _safe_name(run["run_name"], default=run["episode_id"])
        for view_name, segment in segments.items():
            if not isinstance(segment, dict):
                continue
            source_path = segment.get("video_path")
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")
            if source_path is None or start_time is None or end_time is None:
                continue
            view_file = f"{_safe_name(view_name, default='view')}.mp4"
            jobs.append(
                {
                    "episode_id": run["episode_id"],
                    "run_name": run["run_name"],
                    "scene_output_path": run["scene_output_path"],
                    "task": _task_label(item),
                    "view_name": str(view_name),
                    "source_path": str(source_path),
                    "output_path": str(episode_dir / view_file),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "start_frame": segment.get("start_frame"),
                    "end_frame": segment.get("end_frame"),
                    "fps": segment.get("fps") or item.get("fps"),
                    "num_frames": segment.get("num_frames"),
                }
            )
    return jobs, missing


def _cut_one(job: dict[str, Any], *, overwrite: bool, codec_mode: str) -> dict[str, Any]:
    output_path = Path(job["output_path"])
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return {**job, "status": "skipped_existing", "bytes": output_path.stat().st_size}

    source_path = Path(job["source_path"])
    if not source_path.exists():
        return {**job, "status": "missing_source", "error": f"source not found: {source_path}"}

    start_time = float(job["start_time"])
    end_time = float(job["end_time"])
    if end_time <= start_time:
        return {**job, "status": "invalid_segment", "error": f"invalid time range: {start_time}-{end_time}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start_time:.9f}",
        "-i",
        str(source_path),
        "-t",
        f"{(end_time - start_time):.9f}",
        "-map",
        "0:v:0",
        "-an",
    ]
    if codec_mode == "copy":
        cmd.extend(["-c", "copy"])
    else:
        cmd.extend(
            [
                "-vf",
                "setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if job.get("fps") is not None:
            cmd.extend(["-r", str(job["fps"])])
    cmd.append(str(output_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            **job,
            "status": "ffmpeg_error",
            "returncode": proc.returncode,
            "error": (proc.stderr or proc.stdout or "").strip(),
        }
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return {**job, "status": "empty_output", "error": f"empty output: {output_path}"}
    return {**job, "status": "exported", "bytes": output_path.stat().st_size}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_excluded_tasks(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        path = Path(value)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        records = data if isinstance(data, list) else [data]
        for record in records:
            if isinstance(record, dict) and record.get("task"):
                excluded.add(str(record["task"]))
    return excluded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BridgeData episode clips with completed scene outputs.")
    parser.add_argument("--task-dir", default=DEFAULT_TASK_DIR)
    parser.add_argument("--paths-json", default=DEFAULT_PATHS_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--sample-episodes", type=int, default=None)
    parser.add_argument("--per-task-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--exclude-selected-json",
        action="append",
        default=[],
        help="Path to a selected_episodes.json file whose task labels should be excluded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--codec-mode",
        choices=("copy", "h264"),
        default="copy",
        help="Use fast stream copy, or re-encode to H.264 like the stage runner.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.limit_episodes is not None and args.limit_episodes < 0:
        raise ValueError("--limit-episodes must be >= 0")

    started_at = time.perf_counter()
    task_dir = Path(args.task_dir)
    paths_json = Path(args.paths_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _load_json_list(paths_json)
    by_episode = {str(item.get("episode_id")): item for item in items if item.get("episode_id")}
    completed_runs = _find_completed_scene_runs(task_dir)
    excluded_tasks = _load_excluded_tasks(args.exclude_selected_json)
    selected_runs = _select_runs(
        completed_runs=completed_runs,
        by_episode=by_episode,
        limit_episodes=args.limit_episodes,
        sample_episodes=args.sample_episodes,
        per_task_limit=args.per_task_limit,
        seed=args.seed,
        excluded_tasks=excluded_tasks,
    )
    jobs, missing = _build_jobs(
        selected_runs=selected_runs,
        by_episode=by_episode,
        output_dir=output_dir,
    )

    summary_path = output_dir / "export_summary.json"
    manifest_path = output_dir / "manifest.jsonl"
    summary = {
        "task_dir": str(task_dir),
        "paths_json": str(paths_json),
        "output_dir": str(output_dir),
        "paths_items": len(items),
        "completed_scene_episodes": len(completed_runs),
        "selected_episodes": len(selected_runs),
        "sample_episodes": args.sample_episodes,
        "per_task_limit": args.per_task_limit,
        "seed": args.seed,
        "excluded_tasks": sorted(excluded_tasks),
        "jobs": len(jobs),
        "missing_episode_records": len(missing),
        "workers": args.workers,
        "overwrite": args.overwrite,
        "codec_mode": args.codec_mode,
        "dry_run": args.dry_run,
    }
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "selected_episodes.json",
        [
            {
                **run,
                "task": _task_label(by_episode[run["episode_id"]]),
                "episode_index": by_episode[run["episode_id"]].get("episode_index"),
            }
            for run in selected_runs
            if run["episode_id"] in by_episode
        ],
    )
    if missing:
        _write_json(output_dir / "missing_episode_records.json", missing)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        for job in jobs[:20]:
            print(json.dumps(job, ensure_ascii=False), flush=True)
        return

    counts: dict[str, int] = {}
    pending_records: list[dict[str, Any]] = []
    processed = 0
    last_report = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(_cut_one, job, overwrite=args.overwrite, codec_mode=args.codec_mode): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            record = future.result()
            processed += 1
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            pending_records.append(record)
            if len(pending_records) >= 100:
                _append_jsonl(manifest_path, pending_records)
                pending_records.clear()
            now = time.perf_counter()
            if processed == len(jobs) or processed % args.progress_every == 0 or now - last_report >= 30:
                elapsed = now - started_at
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"progress {processed}/{len(jobs)} counts={counts} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s",
                    flush=True,
                )
                last_report = now
    _append_jsonl(manifest_path, pending_records)

    summary.update(
        {
            "status_counts": counts,
            "elapsed_seconds": time.perf_counter() - started_at,
            "manifest_path": str(manifest_path),
        }
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
