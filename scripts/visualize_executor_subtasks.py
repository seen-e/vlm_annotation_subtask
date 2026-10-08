"""Render one Analysis subtask timeline image per executor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _label(action: dict[str, Any]) -> str:
    parts = [str(action.get("subtask_id") or "subtask"), str(action.get("action") or "unknown")]
    object_id = action.get("object")
    target_id = action.get("target")
    if object_id:
        relation = str(object_id)
        if target_id:
            relation += f" -> {target_id}"
        parts.append(relation)
    return " | ".join(parts)


def _render_executor(
    *,
    episode_name: str,
    executor: str,
    actions: list[dict[str, Any]],
    duration: float,
    output_path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    row_count = max(1, len(actions))
    fig_height = max(2.8, 0.72 * row_count + 1.8)
    fig, ax = plt.subplots(figsize=(14, fig_height), dpi=160)
    palette = plt.get_cmap("tab20")

    if actions:
        for row, action in enumerate(actions):
            start = float(action["start_time_hint"])
            end = float(action["end_time_hint"])
            width = max(0.0, end - start)
            color = palette(row % 20)
            ax.barh(row, width, left=start, height=0.56, color=color, edgecolor="#202124", linewidth=0.8)
            ax.text(
                start + width / 2,
                row,
                f"{action.get('action', 'unknown')}\n{start:.2f}-{end:.2f}s",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
                clip_on=True,
            )
        ax.set_yticks(range(len(actions)), labels=[_label(action) for action in actions])
        ax.invert_yaxis()
    else:
        ax.text(duration / 2, 0, "No subtasks", ha="center", va="center", fontsize=12, color="#666666")
        ax.set_yticks([])

    ax.set_xlim(0.0, duration)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"{episode_name} | {executor} subtasks | [0, {duration:g})s")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for boundary in range(int(duration) + 1):
        ax.axvline(boundary, color="#eeeeee", linewidth=0.45, zorder=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize(input_path: Path, output_dir: Path, duration: float) -> list[Path]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    timelines = data.get("executor_timelines")
    if not isinstance(timelines, list):
        raise ValueError("input must contain executor_timelines list")

    episode_name = input_path.parents[2].name
    outputs: list[Path] = []
    for timeline in timelines:
        executor = str(timeline.get("executor") or "unknown_executor")
        actions = timeline.get("actions")
        if not isinstance(actions, list):
            raise ValueError(f"{executor}.actions must be a list")
        output_path = output_dir / f"{executor}_subtasks.png"
        _render_executor(
            episode_name=episode_name,
            executor=executor,
            actions=actions,
            duration=duration,
            output_path=output_path,
        )
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    for output_path in visualize(args.input, args.output_dir, args.duration):
        print(output_path)


if __name__ == "__main__":
    main()
