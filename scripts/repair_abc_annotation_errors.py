#!/usr/bin/env python3
"""Repair known recoverable ABC annotation pipeline failures.

This script scans a batch log, locates failed episode directories, and repairs
only the failure modes that can be fixed deterministically:

1. scene/output.json is a bare list of executors, but downstream expects
   {"executors": [...], "objects": [...]}.
2. analysis/output.json is a bare list of subtasks, but downstream expects
   {"subtasks": [...]}.
3. analysis/output.json is missing optional refinement inputs such as
   added_actions / uncertainties.
4. refinement/raw_text.txt is complete JSON with small syntax glitches:
   duplicated quote before a field name, missing opening quote before a field
   name, bad outer "{ { ... } }" wrapper, or a few safe comma fixes.

It deliberately does not repair truncated model outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ERROR_BLOCK_RE = re.compile(
    r"\[(\d+)/(\d+)\]\s+([^\n]+)\n\s+error -> (.*?)(?=\n\s+progress ->|\n\[\d+/|\Z)",
    re.S,
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.S)


@dataclass
class RepairResult:
    episode: str
    stage: str
    action: str
    changed: bool = False
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_fence(text: str) -> str:
    text = text.strip()
    match = FENCE_RE.search(text)
    return match.group(1).strip() if match else text


def try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def ensure_inside(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if not (resolved == root or root in resolved.parents):
        raise RuntimeError(f"Unsafe path outside root: {resolved}")
    return resolved


def remove_dir(root: Path, path: Path, *, allowed_names: set[str]) -> bool:
    path = ensure_inside(root, path)
    if not path.exists():
        return False
    if path.name not in allowed_names:
        raise RuntimeError(f"Refusing to remove unexpected directory: {path}")
    shutil.rmtree(path)
    return True


def remove_file(root: Path, path: Path) -> bool:
    path = ensure_inside(root, path)
    if not path.exists():
        return False
    path.unlink()
    return True


def repair_refinement_json_text(raw_text: str) -> tuple[Any | None, str]:
    """Return repaired JSON object and repair name, or (None, reason)."""
    base = strip_fence(raw_text)

    candidates: list[tuple[str, str]] = []

    # Model glitch: `"        "target_state": null`
    field_quote = re.sub(r'"\s+"([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'"\1":', base)
    # Model glitch: `       object_target_relation": "..."`
    field_quote = re.sub(
        r'(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*")\s*:',
        r'\1"\2:',
        field_quote,
    )
    candidates.append(("field_quote_glitch", field_quote))

    # Same as above plus conservative trailing comma removal.
    candidates.append(
        (
            "field_quote_glitch+trailing_commas",
            re.sub(r",\s*([}\]])", r"\1", field_quote),
        )
    )

    # Model glitch: `{ { "subtasks": ... } }`
    bad_outer = re.sub(r"^\s*\{\s*\{", "{", base, count=1)
    bad_outer = re.sub(r"\n\s*\}\s*$", "\n", bad_outer, count=1)
    candidates.append(("bad_outer_wrapper", bad_outer))

    # Some responses wrap object as `{ { ... } ] }`.
    bad_outer_array = re.sub(r"^\s*\{\s*\{", "{", base, count=1)
    bad_outer_array = re.sub(
        r"\n\s*\}\s*\n\s*\]\s*\n\s*\}\s*$",
        "\n}\n",
        bad_outer_array,
        count=1,
    )
    candidates.append(("bad_outer_array_wrapper", bad_outer_array))

    for name, candidate in candidates:
        parsed = try_parse_json(candidate)
        if isinstance(parsed, dict) and "subtasks" in parsed:
            return parsed, name

    return None, "no_safe_repair_matched"


def iter_log_errors(log_path: Path) -> list[dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = ANSI_RE.sub("", text)
    rows = []
    for match in ERROR_BLOCK_RE.finditer(text):
        err = " ".join(match.group(4).split())
        stage_match = re.search(r"stage '([^']+)' failed:", err)
        rows.append(
            {
                "idx": int(match.group(1)),
                "total": int(match.group(2)),
                "episode": match.group(3).strip(),
                "stage": stage_match.group(1) if stage_match else "unknown",
                "error": err,
            }
        )
    return rows


def repair_scene_bare_list(root: Path, ep_dir: Path, apply: bool) -> RepairResult:
    out_path = ep_dir / "stages" / "scene" / "output.json"
    if not out_path.exists():
        return RepairResult(ep_dir.name, "scene", "wrap_scene_executors", detail="missing scene/output.json")
    data = read_json(out_path)
    if not isinstance(data, list):
        return RepairResult(ep_dir.name, "scene", "wrap_scene_executors", detail=f"already {type(data).__name__}")

    fixed = {"executors": data, "objects": []}
    if apply:
        write_json(out_path, fixed)
        remove_dir(root, ep_dir / "stages" / "analysis", allowed_names={"analysis"})
    return RepairResult(
        ep_dir.name,
        "scene",
        "wrap_scene_executors_and_remove_analysis",
        changed=apply,
        detail=f"executors={len(data)}",
    )


def repair_analysis_bare_list(root: Path, ep_dir: Path, apply: bool) -> RepairResult:
    out_path = ep_dir / "stages" / "analysis" / "output.json"
    if not out_path.exists():
        return RepairResult(ep_dir.name, "analysis", "wrap_analysis_subtasks", detail="missing analysis/output.json")
    data = read_json(out_path)
    if not isinstance(data, list):
        return RepairResult(ep_dir.name, "analysis", "wrap_analysis_subtasks", detail=f"already {type(data).__name__}")

    fixed = {"subtasks": data}
    if apply:
        write_json(out_path, fixed)
        remove_dir(root, ep_dir / "stages" / "refinement", allowed_names={"refinement"})
    return RepairResult(
        ep_dir.name,
        "analysis",
        "wrap_analysis_subtasks_and_remove_refinement",
        changed=apply,
        detail=f"subtasks={len(data)}",
    )


def repair_missing_analysis_optional_fields(root: Path, ep_dir: Path, apply: bool) -> RepairResult:
    out_path = ep_dir / "stages" / "analysis" / "output.json"
    if not out_path.exists():
        return RepairResult(ep_dir.name, "analysis", "add_analysis_optional_fields", detail="missing analysis/output.json")
    data = read_json(out_path)
    if not isinstance(data, dict):
        return RepairResult(ep_dir.name, "analysis", "add_analysis_optional_fields", detail=f"not dict: {type(data).__name__}")

    changed_fields = []
    for key in ("added_actions", "uncertainties"):
        if key not in data:
            data[key] = []
            changed_fields.append(key)

    if apply and changed_fields:
        write_json(out_path, data)
        remove_dir(root, ep_dir / "stages" / "refinement", allowed_names={"refinement"})

    return RepairResult(
        ep_dir.name,
        "analysis",
        "add_analysis_optional_fields_and_remove_refinement",
        changed=apply and bool(changed_fields),
        detail=",".join(changed_fields) if changed_fields else "nothing_missing",
    )


def repair_refinement_json(root: Path, ep_dir: Path, apply: bool) -> RepairResult:
    stage_dir = ep_dir / "stages" / "refinement"
    raw_path = stage_dir / "raw_text.txt"
    if not raw_path.exists():
        return RepairResult(ep_dir.name, "refinement", "repair_refinement_json", detail="missing raw_text.txt")

    raw = raw_path.read_text(encoding="utf-8", errors="replace")
    fixed, repair_name = repair_refinement_json_text(raw)
    if fixed is None:
        return RepairResult(ep_dir.name, "refinement", "repair_refinement_json", detail=repair_name)

    if apply:
        write_json(stage_dir / "output.json", fixed)
        remove_file(root, stage_dir / "error.json")

    return RepairResult(
        ep_dir.name,
        "refinement",
        f"repair_refinement_json:{repair_name}",
        changed=apply,
        detail=f"subtasks={len(fixed.get('subtasks', []))}",
    )


def classify_and_repair(root: Path, row: dict[str, Any], apply: bool) -> RepairResult | None:
    ep_dir = ensure_inside(root, root / row["episode"])
    err = row["error"]

    if "Expected list index in context path 'stages.scene.output.executors'" in err:
        return repair_scene_bare_list(root, ep_dir, apply)
    if "Expected list index in context path 'stages.analysis.output.subtasks'" in err:
        return repair_analysis_bare_list(root, ep_dir, apply)
    if "Context path not found: 'stages.analysis.output.added_actions'" in err:
        return repair_missing_analysis_optional_fields(root, ep_dir, apply)
    if row["stage"] == "refinement" and "Failed to extract valid JSON" in err:
        return repair_refinement_json(root, ep_dir, apply)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Batch log path.")
    parser.add_argument("--data-root", required=True, type=Path, help="Split output dir containing episode dirs.")
    parser.add_argument("--apply", action="store_true", help="Actually modify files. Without this, dry-run only.")
    args = parser.parse_args()

    root = args.data_root.resolve()
    rows = iter_log_errors(args.log)
    results: list[RepairResult] = []
    skipped = 0

    for row in rows:
        result = classify_and_repair(root, row, args.apply)
        if result is None:
            skipped += 1
            continue
        results.append(result)

    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"log_errors={len(rows)}")
    print(f"repair_candidates={len(results)}")
    print(f"skipped_not_repairable={skipped}")

    grouped: dict[str, int] = {}
    for result in results:
        grouped[result.action] = grouped.get(result.action, 0) + 1
    print("summary:")
    for action, count in sorted(grouped.items()):
        print(f"  {count:4d} {action}")

    print("details:")
    for result in results:
        changed = "CHANGED" if result.changed else "DRYRUN"
        print(f"{changed} {result.stage} {result.episode} {result.action} {result.detail}")


if __name__ == "__main__":
    main()
