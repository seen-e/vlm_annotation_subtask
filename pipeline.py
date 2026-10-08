"""Sequential pipeline runner for lightweight VLM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .stage_runner import cleanup_temporary_video_segments, run_stage


class PipelineError(RuntimeError):
    """Raised when pipeline configuration or execution fails."""


def _get_workflow(config: dict[str, Any], workflow: list[str] | None) -> list[str]:
    stages = workflow if workflow is not None else config.get("workflow")
    if not stages:
        raise PipelineError("workflow is required and must be non-empty")
    if not isinstance(stages, list):
        raise PipelineError("workflow must be a list of stage names")
    known = set((config.get("stages") or {}).keys())
    missing = [name for name in stages if name not in known]
    if missing:
        raise PipelineError(f"workflow contains stages not present in config['stages']: {missing}")
    return list(stages)


def _slice_workflow(stages: list[str], start_from: str | None, stop_after: str | None) -> list[str]:
    start = 0
    end = len(stages)
    if start_from is not None:
        if start_from not in stages:
            raise PipelineError(f"start_from stage not in workflow: {start_from!r}")
        start = stages.index(start_from)
    if stop_after is not None:
        if stop_after not in stages:
            raise PipelineError(f"stop_after stage not in workflow: {stop_after!r}")
        end = stages.index(stop_after) + 1
    if start >= end:
        raise PipelineError(f"Invalid pipeline range: start_from={start_from!r}, stop_after={stop_after!r}, workflow={stages}")
    return stages[start:end]


def run_pipeline(
    context: dict[str, Any],
    config: dict[str, Any],
    *,
    workflow: list[str] | None = None,
    dry_run: bool = False,
    start_from: str | None = None,
    stop_after: str | None = None,
    skip_existing: bool = False,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    save_results: bool = False,
    video_segment_clip_storage: str = "output-dir",
) -> dict[str, Any]:
    """Run configured stages sequentially and return the updated context."""
    try:
        if video_segment_clip_storage not in {"output-dir", "temp"}:
            raise PipelineError(
                "video_segment_clip_storage must be 'output-dir' or 'temp', "
                f"got {video_segment_clip_storage!r}"
            )
        all_stages = _get_workflow(config, workflow)
        selected_stages = _slice_workflow(all_stages, start_from, stop_after)
        context.setdefault("stages", {})
        context.setdefault("pipeline", {})
        context["pipeline"]["workflow"] = all_stages
        context["pipeline"]["selected_stages"] = selected_stages
        context["pipeline"]["video_segment_clip_storage"] = video_segment_clip_storage

        run_dir: Path | None = None
        if save_results:
            from .result_io import make_run_dir

            run_dir = make_run_dir(output_dir or "outputs", run_name=run_name, video_id=context.get("input", {}).get("video_id"))
            context["run_dir"] = str(run_dir)

        executed: list[str] = []
        skipped: list[str] = []
        for stage_name in selected_stages:
            if skip_existing and stage_name in context.get("stages", {}):
                skipped.append(stage_name)
                continue
            try:
                run_stage(stage_name, context, config, dry_run=dry_run, run_dir=run_dir, save_result=save_results)
                executed.append(stage_name)
            except Exception as exc:
                raise PipelineError(f"Pipeline failed at stage {stage_name!r}: {exc}") from exc

        context["pipeline"]["executed_stages"] = executed
        context["pipeline"]["skipped_stages"] = skipped

        cleanup_temporary_video_segments(context)
        if save_results and run_dir is not None:
            from .result_io import save_context

            save_context(context, run_dir)
        return context
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"Pipeline execution failed: {exc}") from exc
    finally:
        cleanup_temporary_video_segments(context)
