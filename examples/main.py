"""Batch prediction entrypoint for robot_mind2 task JSON.

Default usage from the repository root:

    python examples/main.py

For a quick pipeline check without calling the VLM:

    python examples/main.py --dry-run --limit 1

Per-task outputs are saved under ``examples/vlm_annotation_batch_predictions``.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PACKAGE_PARENT = PACKAGE_DIR.parent
PACKAGE_NAME = PACKAGE_DIR.name
DEFAULT_TASKS_PATH = "/home/xuchacha/vlm_auto_annotation_refactor/examples/data/dual_arm/robogene_twoArm_franka_adjust_black_computer_stand.json"


def _ensure_imports() -> Any:
    """Import pipeline helpers from the current package directory name."""
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

    pipeline_module = importlib.import_module(f"{PACKAGE_NAME}.pipeline")
    return pipeline_module.run_pipeline


def _ensure_episode_imports() -> Any:
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

    episode_module = importlib.import_module(f"{PACKAGE_NAME}.episode_runner")
    return episode_module.run_episode_pipeline


def _segment_loop_enabled(config: dict[str, Any]) -> bool:
    return bool((config.get("segment_loop") or {}).get("enabled", False))


def _segment_episode_complete(run_dir: Path, config: dict[str, Any]) -> bool:
    segment_config = config.get("segment_loop") or {}
    merge_after_episode = bool(segment_config.get("merge_after_episode", True))
    bimanual_mode = (
        (segment_config.get("postprocessor") or {}).get("policy")
        == "bimanual_last_action_start"
    )
    output_path = run_dir / "output.json"
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
            has_timelines = isinstance(output, dict) and isinstance(
                output.get("executor_timelines"), list
            )
            has_required_merge = not merge_after_episode or isinstance(
                output.get("merged_timeline"), list
            )
            if has_timelines and has_required_merge:
                return True
        except (OSError, json.JSONDecodeError):
            pass

    if bimanual_mode:
        return False

    # Compatibility with generic or older segment-loop outputs.
    final_path = run_dir / "final_subtasks.json"
    if not final_path.exists() or final_path.stat().st_size == 0:
        return False
    try:
        return isinstance(json.loads(final_path.read_text(encoding="utf-8")), list)
    except (OSError, json.JSONDecodeError):
        return False


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Task file must contain a JSON list: {path}")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Task item #{index} must be a JSON object")
    return data


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value))
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return value or "run"


def _infer_dataset_name(video_path: Any) -> str:
    if isinstance(video_path, dict) and video_path:
        first_path = next(iter(video_path.values()))
    elif isinstance(video_path, (list, tuple)) and video_path:
        first_path = video_path[0]
    else:
        first_path = video_path

    parts = Path(str(first_path)).parts
    if "videos" in parts:
        idx = parts.index("videos")
        if idx > 0:
            return parts[idx - 1]
    return "task"


def _resolve_video_path(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {name: _resolve_video_path(path, base_dir) for name, path in value.items()}
    if isinstance(value, list):
        return [_resolve_video_path(path, base_dir) for path in value]
    if isinstance(value, tuple):
        return [_resolve_video_path(path, base_dir) for path in value]

    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_video_segment_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if key == "video_path" and isinstance(item, (str, Path)):
                resolved[key] = _resolve_video_path(item, base_dir)
            else:
                resolved[key] = _resolve_video_segment_paths(item, base_dir)
        return resolved
    if isinstance(value, list):
        return [_resolve_video_segment_paths(item, base_dir) for item in value]
    return value


def _build_episode_source(item: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    episode = copy.deepcopy(item)
    if "video_path" in episode:
        episode["video_path"] = _resolve_video_path(episode["video_path"], base_dir)
    if "video_segments" in episode:
        episode["video_segments"] = _resolve_video_segment_paths(episode["video_segments"], base_dir)
    return episode


def _slice_tasks(tasks: list[dict[str, Any]], *, start_index: int, limit: int | None) -> list[tuple[int, dict[str, Any]]]:
    end_index = len(tasks) if limit is None else min(len(tasks), start_index + limit)
    return list(enumerate(tasks[start_index:end_index], start=start_index))


def _build_run_name(item: dict[str, Any], index: int) -> str:
    dataset_name = _infer_dataset_name(item.get("video_path"))
    episode_id = str(item.get("episode_id") or f"item_{index:05d}")
    return _safe_name(f"{index:05d}_{dataset_name}_{episode_id}")


def _selected_stage_names(config: dict[str, Any], *, start_from: str | None, stop_after: str | None) -> list[str]:
    workflow = config.get("workflow")
    if not isinstance(workflow, list) or not workflow:
        raise ValueError("config workflow must be a non-empty list")

    start = 0
    end = len(workflow)
    if start_from is not None:
        if start_from not in workflow:
            raise ValueError(f"start_from stage not in workflow: {start_from!r}")
        start = workflow.index(start_from)
    if stop_after is not None:
        if stop_after not in workflow:
            raise ValueError(f"stop_after stage not in workflow: {stop_after!r}")
        end = workflow.index(stop_after) + 1
    if start >= end:
        raise ValueError(f"Invalid pipeline range: start_from={start_from!r}, stop_after={stop_after!r}")
    return list(workflow[start:end])


def _existing_stage_outputs(run_dir: Path, stage_names: list[str]) -> list[str]:
    """Return the contiguous completed prefix of the configured workflow.

    Resume must follow config.workflow order. If analysis is missing, any stale
    refinement output is ignored and refinement will be regenerated after
    analysis. This avoids mixing outputs from incompatible partial runs.
    """
    existing: list[str] = []
    for stage_name in stage_names:
        output_path = run_dir / "stages" / stage_name / "output.json"
        if output_path.exists() and output_path.stat().st_size > 0:
            existing.append(stage_name)
            continue
        break
    return existing


def _load_existing_stage_outputs(context: dict[str, Any], run_dir: Path, stage_names: list[str]) -> None:
    if not stage_names:
        return
    result_io_module = importlib.import_module(f"{PACKAGE_NAME}.result_io")
    result_io_module.load_outputs_into_context(context, run_dir, stage_names)


def _scan_task_resume_state(
    *,
    item: dict[str, Any],
    index: int,
    config: dict[str, Any],
    output_dir: str,
    run_options: dict[str, Any],
) -> dict[str, Any]:
    episode_id = str(item.get("episode_id") or f"item_{index:05d}")
    run_name = _build_run_name(item, index)
    run_dir = Path(output_dir) / run_name
    selected_stage_names = _selected_stage_names(
        config,
        start_from=run_options.get("start_from"),
        stop_after=run_options.get("stop_after"),
    )
    segment_mode = _segment_loop_enabled(config)
    existing_stage_names = [] if segment_mode else _existing_stage_outputs(run_dir, selected_stage_names)
    is_complete = (
        _segment_episode_complete(run_dir, config)
        if segment_mode
        else bool(selected_stage_names) and len(existing_stage_names) == len(selected_stage_names)
    )
    return {
        "index": index,
        "episode_id": episode_id,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "selected_stage_names": selected_stage_names,
        "existing_stage_names": existing_stage_names,
        "is_complete": is_complete,
    }


def _collect_resume_states_from_output_dir(output_dir: str, stage_names: list[str], queue: Any) -> None:
    resume_states: dict[str, list[str]] = {}
    output_root = Path(output_dir)
    if output_root.exists():
        for run_dir in output_root.iterdir():
            if not run_dir.is_dir():
                continue
            existing_stage_names = _existing_stage_outputs(run_dir, stage_names)
            if existing_stage_names:
                resume_states[run_dir.name] = existing_stage_names
    queue.put(resume_states)


def _scan_selected_tasks(
    selected_tasks: list[tuple[int, dict[str, Any]]],
    *,
    config: dict[str, Any],
    output_dir: str,
    run_options: dict[str, Any],
    scan_workers: int,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    if not selected_tasks or not bool(run_options.get("skip_existing", False)):
        return selected_tasks, []

    selected_stage_names = _selected_stage_names(
        config,
        start_from=run_options.get("start_from"),
        stop_after=run_options.get("stop_after"),
    )
    if not selected_stage_names:
        return selected_tasks, []

    output_root = Path(output_dir)
    scan_started_at = time.perf_counter()
    existing_run_names: set[str] = set()
    if output_root.exists():
        for run_dir in output_root.iterdir():
            if run_dir.is_dir():
                existing_run_names.add(run_dir.name)

    existing_tasks: list[tuple[int, dict[str, Any]]] = []
    not_started_tasks: list[tuple[int, dict[str, Any]]] = []
    for index, item in selected_tasks:
        run_name = _build_run_name(item, index)
        if run_name in existing_run_names:
            existing_tasks.append((index, item))
        else:
            not_started_tasks.append((index, item))

    print(
        f"Resume census: existing_run_dirs={len(existing_run_names)}, "
        f"selected_existing={len(existing_tasks)}, not_started={len(not_started_tasks)}, "
        f"workflow={selected_stage_names!r}"
    )

    executable_tasks: list[tuple[int, dict[str, Any]]] = list(not_started_tasks)
    skipped_complete: list[dict[str, Any]] = []
    partial_count = 0
    checked_count = 0
    max_workers = max(1, min(scan_workers, len(existing_tasks))) if existing_tasks else 1
    if existing_tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _scan_task_resume_state,
                    item=item,
                    index=index,
                    config=config,
                    output_dir=output_dir,
                    run_options=run_options,
                ): (index, item)
                for index, item in existing_tasks
            }
            for future in as_completed(future_to_task):
                index, item = future_to_task[future]
                state = future.result()
                checked_count += 1
                if state["is_complete"]:
                    skipped_complete.append(state)
                else:
                    if state.get("existing_stage_names"):
                        partial_count += 1
                    executable_tasks.append((index, item))
                if checked_count % 1000 == 0 or checked_count == len(existing_tasks):
                    elapsed = time.perf_counter() - scan_started_at
                    print(
                        f"Resume census progress: checked_existing={checked_count}/{len(existing_tasks)}, "
                        f"complete={len(skipped_complete)}, partial={partial_count}, "
                        f"elapsed={elapsed:.2f}s"
                    )

    executable_tasks.sort(key=lambda pair: pair[0])
    skipped_complete.sort(key=lambda state: state["index"])
    elapsed = time.perf_counter() - scan_started_at
    print(
        f"Full resume census done: existing_run_dirs={len(existing_run_names)}, "
        f"complete={len(skipped_complete)}, partial={partial_count}, "
        f"not_started={len(not_started_tasks)}, to_execute={len(executable_tasks)}, "
        f"scan_workers={max_workers}, elapsed={elapsed:.2f}s"
    )
    return executable_tasks, skipped_complete

def _build_context(item: dict[str, Any], index: int, *, task_base_dir: Path) -> dict[str, Any]:
    if "video_path" not in item:
        raise ValueError(f"Task item #{index} missing 'video_path'")
    instruction = item.get("task") or item.get("instruction")
    if not instruction:
        raise ValueError(f"Task item #{index} missing 'task' or 'instruction'")

    episode_id = str(item.get("episode_id") or f"item_{index:05d}")
    input_context = {
        "video_path": _resolve_video_path(item["video_path"], task_base_dir),
        "episode": _build_episode_source(item, task_base_dir),
        "instruction": str(instruction),
        "video_id": episode_id,
        "task_index": index,
    }
    if "video_segments" in item:
        input_context["video_segments"] = _resolve_video_segment_paths(item["video_segments"], task_base_dir)
    return {
        "input": input_context,
        "stages": {},
    }


def _run_one_task_worker(
    *,
    item: dict[str, Any],
    index: int,
    total_tasks: int,
    task_base_dir: str,
    config: dict[str, Any],
    output_dir: str,
    run_options: dict[str, Any],
) -> dict[str, Any]:
    """Run one episode pipeline in a worker process."""
    started_at = time.perf_counter()

    episode_id = str(item.get("episode_id") or f"item_{index:05d}")
    run_name = _build_run_name(item, index)
    run_dir = Path(output_dir) / run_name
    if _segment_loop_enabled(config):
        if bool(run_options.get("skip_existing", False)) and _segment_episode_complete(run_dir, config):
            return {
                "index": index,
                "total_tasks": total_tasks,
                "episode_id": episode_id,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "elapsed_seconds": time.perf_counter() - started_at,
                "status": "skipped_complete",
                "existing_stage_names": [],
                "executed_stage_names": [],
            }

        context = _build_context(item, index, task_base_dir=Path(task_base_dir))
        run_episode_pipeline = _ensure_episode_imports()
        context = run_episode_pipeline(
            context,
            config,
            output_dir=Path(output_dir),
            run_name=run_name,
            dry_run=bool(run_options.get("dry_run", False)),
            skip_existing=bool(run_options.get("skip_existing", False)),
            video_segment_clip_storage=str(run_options.get("video_segment_clip_storage") or "output-dir"),
        )
        return {
            "index": index,
            "total_tasks": total_tasks,
            "episode_id": episode_id,
            "run_name": run_name,
            "run_dir": context.get("run_dir"),
            "elapsed_seconds": time.perf_counter() - started_at,
            "status": "completed",
            "existing_stage_names": [],
            "executed_stage_names": [],
        }

    selected_stage_names = _selected_stage_names(
        config,
        start_from=run_options.get("start_from"),
        stop_after=run_options.get("stop_after"),
    )
    existing_stage_names = (
        _existing_stage_outputs(run_dir, selected_stage_names)
        if bool(run_options.get("skip_existing", False))
        else []
    )

    if existing_stage_names and len(existing_stage_names) == len(selected_stage_names):
        return {
            "index": index,
            "total_tasks": total_tasks,
            "episode_id": episode_id,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "elapsed_seconds": time.perf_counter() - started_at,
            "status": "skipped_complete",
            "existing_stage_names": existing_stage_names,
            "executed_stage_names": [],
        }

    context = _build_context(item, index, task_base_dir=Path(task_base_dir))
    if existing_stage_names:
        _load_existing_stage_outputs(context, run_dir, existing_stage_names)

    run_pipeline = _ensure_imports()
    run_pipeline(
        context,
        config,
        dry_run=bool(run_options.get("dry_run", False)),
        start_from=run_options.get("start_from"),
        stop_after=run_options.get("stop_after"),
        skip_existing=bool(run_options.get("skip_existing", False)),
        output_dir=Path(output_dir),
        run_name=run_name,
        save_results=True,
        video_segment_clip_storage=str(run_options.get("video_segment_clip_storage") or "output-dir"),
    )

    return {
        "index": index,
        "total_tasks": total_tasks,
        "episode_id": episode_id,
        "run_name": run_name,
        "run_dir": context.get("run_dir"),
        "elapsed_seconds": time.perf_counter() - started_at,
        "status": "completed",
        "existing_stage_names": existing_stage_names,
        "executed_stage_names": context.get("pipeline", {}).get("executed_stages", []),
    }


def _handle_task_success(result: dict[str, Any]) -> None:
    elapsed_seconds = result.get("elapsed_seconds")
    if elapsed_seconds is not None:
        print(f"  elapsed -> {float(elapsed_seconds):.2f}s")
    existing_stage_names = result.get("existing_stage_names") or []
    if result.get("status") == "skipped_complete":
        print(f"  skipped complete -> {result.get('run_dir')}")
        print(f"  existing stages -> {existing_stage_names}")
        return
    if existing_stage_names:
        print(f"  resumed from existing stages -> {existing_stage_names}")
    print(f"  ok -> {result.get('run_dir')}")


def _print_batch_progress(
    *,
    ok_count: int,
    error_count: int,
    skipped_count: int,
    selected_count: int,
    total_started_at: float,
) -> None:
    processed_count = ok_count + error_count + skipped_count
    total_elapsed = time.perf_counter() - total_started_at
    average_elapsed = total_elapsed / processed_count if processed_count else 0.0
    print(
        f"  progress -> processed={processed_count}/{selected_count}, "
        f"ok={ok_count}, skipped={skipped_count}, error={error_count}, "
        f"total_elapsed={total_elapsed:.2f}s, avg_per_finished={average_elapsed:.2f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run configured VLM pipeline on robot_mind2 tasks.")
    parser.add_argument("--config", default=str(PACKAGE_DIR / "config" / "config_pipeline.yaml"), help="Path to config.yaml.")
    parser.add_argument(
        "--tasks",
        default=str(DEFAULT_TASKS_PATH),
        help="Path to task JSON list.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "vlm_annotation_batch_predictions"),
        help="Directory for per-task outputs.",
    )
    parser.add_argument(
        "--summary",
        default=str(SCRIPT_DIR / "trajectory.json"),
        help="Deprecated compatibility option; trajectory.json is no longer written.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run video/prompt pipeline without real model calls.")
    parser.add_argument("--start-index", type=int, default=0, help="Start task index, inclusive.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of tasks to process.")
    parser.add_argument("--start-from", default=None, help="Optional pipeline start stage.")
    parser.add_argument("--stop-after", default=None, help="Optional pipeline stop stage.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Resume from output-dir: skip an episode when all selected stages already have output.json, "
            "or load completed stages and run only missing stages."
        ),
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of episode processes to run concurrently.")
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=32,
        help=(
            "Deprecated compatibility option. With --skip-existing, resume scan now lists existing run dirs "
            "and workers verify each run's workflow state."
        ),
    )
    parser.add_argument(
        "--video-segment-clip-storage",
        choices=("output-dir", "temp"),
        default="output-dir",
        help=(
            "Where to store temporary clips cut from input.video_segments: "
            "'output-dir' stores them under each run directory; 'temp' stores them in the system temp directory."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed task instead of writing an error record and continuing.",
    )
    return parser.parse_args()


def main() -> None:
    total_started_at = time.perf_counter()
    args = parse_args()
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.scan_workers < 1:
        raise ValueError("--scan-workers must be >= 1")

    _ensure_imports()

    config_path = Path(args.config)
    tasks_path = Path(args.tasks)
    output_dir = Path(args.output_dir)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tasks = _load_json_list(tasks_path)
    task_base_dir = tasks_path.resolve().parent
    selected_tasks = _slice_tasks(tasks, start_index=args.start_index, limit=args.limit)
    selected_count = len(selected_tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    error_count = 0
    skipped_count = 0

    print(f"Loaded config: {config_path}")
    print(f"Loaded tasks: {tasks_path} ({len(tasks)} total, {selected_count} selected)")
    print(f"Saving per-task outputs to: {output_dir}")
    print(f"Workers: {args.workers}")

    run_options = {
        "dry_run": args.dry_run,
        "start_from": args.start_from,
        "stop_after": args.stop_after,
        "skip_existing": args.skip_existing,
        "video_segment_clip_storage": args.video_segment_clip_storage,
    }

    if args.skip_existing:
        scan_started_at = time.perf_counter()
        selected_tasks, skipped_complete = _scan_selected_tasks(
            selected_tasks,
            config=config,
            output_dir=str(output_dir),
            run_options=run_options,
            scan_workers=args.scan_workers,
        )
        skipped_count = len(skipped_complete)
        scan_elapsed = time.perf_counter() - scan_started_at
        print(
            f"Resume scan: skipped_complete={skipped_count}, "
            f"remaining={len(selected_tasks)}, scan_workers={args.scan_workers}, "
            f"elapsed={scan_elapsed:.2f}s"
        )

    def print_task_header(index: int, item: dict[str, Any]) -> None:
        episode_id = str(item.get("episode_id") or f"item_{index:05d}")
        run_name = _build_run_name(item, index)
        print(f"\n[{index + 1}/{len(tasks)}] {run_name}")

    def build_worker_kwargs(index: int, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "item": item,
            "index": index,
            "total_tasks": len(tasks),
            "task_base_dir": str(task_base_dir),
            "config": config,
            "output_dir": str(output_dir),
            "run_options": run_options,
        }

    if args.workers == 1:
        for index, item in selected_tasks:
            print_task_header(index, item)
            try:
                result = _run_one_task_worker(**build_worker_kwargs(index, item))
                _handle_task_success(result)
                if result.get("status") == "skipped_complete":
                    skipped_count += 1
                else:
                    ok_count += 1
                _print_batch_progress(
                    ok_count=ok_count,
                    error_count=error_count,
                    skipped_count=skipped_count,
                    selected_count=selected_count,
                    total_started_at=total_started_at,
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                error_count += 1
                print(f"  error -> {error_text}")
                _print_batch_progress(
                    ok_count=ok_count,
                    error_count=error_count,
                    skipped_count=skipped_count,
                    selected_count=selected_count,
                    total_started_at=total_started_at,
                )
                if args.fail_fast:
                    traceback.print_exc()
                    raise
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    _run_one_task_worker,
                    **build_worker_kwargs(index, item),
                ): (index, item)
                for index, item in selected_tasks
            }
            for future in as_completed(future_to_task):
                index, item = future_to_task[future]
                print_task_header(index, item)
                try:
                    result = future.result()
                    _handle_task_success(result)
                    if result.get("status") == "skipped_complete":
                        skipped_count += 1
                    else:
                        ok_count += 1
                    _print_batch_progress(
                        ok_count=ok_count,
                        error_count=error_count,
                        skipped_count=skipped_count,
                        selected_count=selected_count,
                        total_started_at=total_started_at,
                    )
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    error_count += 1
                    print(f"  error -> {error_text}")
                    _print_batch_progress(
                        ok_count=ok_count,
                        error_count=error_count,
                        skipped_count=skipped_count,
                        selected_count=selected_count,
                        total_started_at=total_started_at,
                    )
                    if args.fail_fast:
                        for pending in future_to_task:
                            pending.cancel()
                        traceback.print_exc()
                        raise

    total_elapsed = time.perf_counter() - total_started_at
    processed_count = ok_count + error_count + skipped_count
    average_elapsed = total_elapsed / processed_count if processed_count else 0.0
    print(
        f"\nDone. ok={ok_count}, skipped={skipped_count}, error={error_count}, "
        f"total_elapsed={total_elapsed:.2f}s, avg_per_episode={average_elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
