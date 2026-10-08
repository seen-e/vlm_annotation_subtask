"""Run one lightweight stage.

Example:
python examples/run_stage.py \
  --stage scene \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --dry-run \
  --save-results
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PACKAGE_PARENT = PACKAGE_DIR.parent
PACKAGE_NAME = PACKAGE_DIR.name
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

import yaml

result_io_module = importlib.import_module(f"{PACKAGE_NAME}.result_io")
stage_runner_module = importlib.import_module(f"{PACKAGE_NAME}.stage_runner")
make_run_dir = result_io_module.make_run_dir
save_context = result_io_module.save_context
run_stage = stage_runner_module.run_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_DIR / "config" / "config.yaml"))
    parser.add_argument("--stage", default="scene")
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--video-id", default="demo_episode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    context = {
        "input": {"video_path": args.video, "instruction": args.instruction, "video_id": args.video_id},
        "stages": {},
    }
    run_dir = None
    if args.save_results:
        run_dir = make_run_dir(args.output_dir, run_name=args.run_name, video_id=args.video_id)
        context["run_dir"] = str(run_dir)

    output = run_stage(args.stage, context, config, dry_run=args.dry_run, run_dir=run_dir, save_result=args.save_results)
    if run_dir is not None:
        save_context(context, run_dir)

    record = context["stages"][args.stage]
    print("stage:", args.stage)
    print("output:", json.dumps(output, ensure_ascii=False, indent=2))
    print("prompt preview:\n", record.get("prompt", "")[:1200])
    print("video_meta:", json.dumps(record.get("video_meta", {}), ensure_ascii=False, indent=2)[:1200])
    if run_dir:
        print("run_dir:", run_dir)


if __name__ == "__main__":
    main()
