"""Run the lightweight configurable pipeline.

Example:
python examples/run_pipeline.py \
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

pipeline_module = importlib.import_module(f"{PACKAGE_NAME}.pipeline")
result_io_module = importlib.import_module(f"{PACKAGE_NAME}.result_io")
run_pipeline = pipeline_module.run_pipeline
load_outputs_into_context = result_io_module.load_outputs_into_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PACKAGE_DIR / "config" / "config.yaml"))
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--video-id", default="demo_episode")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-from", default=None)
    parser.add_argument("--stop-after", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--load-run-dir", default=None)
    parser.add_argument("--load-stages", nargs="*", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    context = {
        "input": {"video_path": args.video, "instruction": args.instruction, "video_id": args.video_id},
        "stages": {},
    }
    if args.load_run_dir and args.load_stages:
        load_outputs_into_context(context, args.load_run_dir, args.load_stages)

    run_pipeline(
        context,
        config,
        dry_run=args.dry_run,
        start_from=args.start_from,
        stop_after=args.stop_after,
        skip_existing=args.skip_existing,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_results=args.save_results,
    )

    print("workflow:", config.get("workflow"))
    print("executed:", context.get("pipeline", {}).get("executed_stages"))
    print("skipped:", context.get("pipeline", {}).get("skipped_stages"))
    for stage_name, record in context.get("stages", {}).items():
        print("\n===", stage_name, "===")
        print("output:", json.dumps(record.get("output"), ensure_ascii=False, indent=2)[:1200])
        print("prompt preview:\n", record.get("prompt", "")[:1000])
        print("video_meta:", json.dumps(record.get("video_meta", {}), ensure_ascii=False, indent=2)[:1000])
    if context.get("run_dir"):
        print("\nrun_dir:", context["run_dir"])


if __name__ == "__main__":
    main()
