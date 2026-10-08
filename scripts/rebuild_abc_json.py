from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("/mnt/data/yuluo/data/abc_130k_v3_train")
OUTPUT_DIR = Path("/mnt/workspace/abc_130k_v3")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    info = json.loads((DATASET_ROOT / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps") or 30)
    chunks_size = int(info.get("chunks_size") or 1000)
    video_template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    video_keys = [
        key
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    primary_video_key = "observation.images.top" if "observation.images.top" in video_keys else video_keys[0]
    video_keys = [primary_video_key]

    tasks_df = pd.read_parquet(DATASET_ROOT / "meta" / "tasks.parquet")
    task_map = {int(row["task_index"]): str(index) for index, row in tasks_df.iterrows()}

    stats: dict[int, dict[str, int | float]] = {}
    files = sorted((DATASET_ROOT / "data").glob("**/*.parquet"))
    started = time.perf_counter()
    for file_index, path in enumerate(files, 1):
        df = pd.read_parquet(
            path,
            columns=["episode_index", "task_index", "timestamp", "frame_index"],
        )
        grouped = df.groupby("episode_index", sort=False).agg(
            task_index=("task_index", "first"),
            min_timestamp=("timestamp", "min"),
            max_timestamp=("timestamp", "max"),
            min_frame=("frame_index", "min"),
            max_frame=("frame_index", "max"),
            frame_count=("frame_index", "count"),
        )
        for episode_index, row in grouped.iterrows():
            episode_index = int(episode_index)
            current = stats.get(episode_index)
            if current is None:
                stats[episode_index] = {
                    "task_index": int(row.task_index),
                    "min_timestamp": float(row.min_timestamp),
                    "max_timestamp": float(row.max_timestamp),
                    "min_frame": int(row.min_frame),
                    "max_frame": int(row.max_frame),
                    "frame_count": int(row.frame_count),
                }
                continue

            current["min_timestamp"] = min(float(current["min_timestamp"]), float(row.min_timestamp))
            current["max_timestamp"] = max(float(current["max_timestamp"]), float(row.max_timestamp))
            current["min_frame"] = min(int(current["min_frame"]), int(row.min_frame))
            current["max_frame"] = max(int(current["max_frame"]), int(row.max_frame))
            current["frame_count"] = int(current["frame_count"]) + int(row.frame_count)

        if file_index % 25 == 0 or file_index == len(files):
            elapsed = time.perf_counter() - started
            print(f"parquet progress {file_index}/{len(files)} episodes={len(stats)} elapsed={elapsed:.1f}s", flush=True)

    video_files = sorted((DATASET_ROOT / "videos" / primary_video_key).glob("chunk-*/*.mp4"))
    if not video_files:
        raise RuntimeError(f"No video files found for primary video key: {primary_video_key}")

    def probe_frame_count(video_file: Path) -> int:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=nk=1:nw=1",
                str(video_file),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return int(probe.stdout.strip())

    video_frame_counts: list[int] = [0] * len(video_files)
    probed_count = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_index = {
            executor.submit(probe_frame_count, video_file): index
            for index, video_file in enumerate(video_files)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            video_frame_counts[index] = future.result()
            probed_count += 1
            if probed_count % 500 == 0 or probed_count == len(video_files):
                print(f"ffprobe progress {probed_count}/{len(video_files)}", flush=True)

    items = []
    episode_video_index = 0
    episode_start_frame_in_video = 0
    for episode_index in sorted(stats):
        stat = stats[episode_index]
        task_index = int(stat["task_index"])
        frame_count = int(stat["max_frame"]) - int(stat["min_frame"]) + 1
        if frame_count <= 0:
            frame_count = int(stat["frame_count"])
        while episode_video_index < len(video_frame_counts) and (
            episode_start_frame_in_video + frame_count > video_frame_counts[episode_video_index]
        ):
            episode_video_index += 1
            episode_start_frame_in_video = 0
        if episode_video_index >= len(video_files):
            raise RuntimeError(f"Ran out of video files at episode {episode_index}")

        primary_video = video_files[episode_video_index]
        rel_video_parts = primary_video.relative_to(DATASET_ROOT / "videos" / primary_video_key).parts
        chunk_name = rel_video_parts[0]
        file_name = rel_video_parts[1]
        video_path = {
            key: str(DATASET_ROOT / "videos" / key / chunk_name / file_name)
            for key in video_keys
        }
        video_segments = {
            key: {
                "video_path": path,
                "start_frame": episode_start_frame_in_video,
                "end_frame": episode_start_frame_in_video + frame_count,
                "fps": fps,
            }
            for key, path in video_path.items()
        }

        duration_sec = frame_count / fps
        task = task_map.get(task_index, str(task_index))
        items.append(
            {
                "dataset": "abc_130k_v3_train",
                "episode_id": f"abc_130k_v3_train__episode_{episode_index:06d}",
                "episode_index": episode_index,
                "task_index": task_index,
                "task": task,
                "instruction": task,
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": duration_sec,
                "timestamp_start": float(stat["min_timestamp"]),
                "timestamp_end": float(stat["max_timestamp"]),
                "video_path": video_path,
                "video_segments": video_segments,
            }
        )
        episode_start_frame_in_video += frame_count

    duration_0_90s = [item for item in items if item["duration_sec"] <= 90]
    duration_90_150s = [item for item in items if 90 < item["duration_sec"] <= 150]
    duration_150s_plus = [item for item in items if item["duration_sec"] > 150]

    outputs = [
        ("abc_130k_v3_train.json", items),
        ("duration_0_90s.json", duration_0_90s),
        ("duration_90_150s.json", duration_90_150s),
        ("duration_150s_plus.json", duration_150s_plus),
    ]
    for name, data in outputs:
        path = OUTPUT_DIR / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path} count={len(data)} size={path.stat().st_size}", flush=True)

    print(
        json.dumps(
            {
                "total": len(items),
                "0_90": len(duration_0_90s),
                "90_150": len(duration_90_150s),
                "150_plus": len(duration_150s_plus),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
