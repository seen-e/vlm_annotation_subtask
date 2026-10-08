# stage_runner.py

## 文件职责

`stage_runner.py` 执行单个配置化 stage。

它把以下模块串起来：

```text
video_process.py -> prompt_utils.py -> model_client.py -> json_utils.py
```

它不负责多 stage 顺序调度，多 stage 调度由 `pipeline.py` 负责。

## 异常类

### `StageRunnerError`

stage 配置缺失、视频处理失败、prompt 渲染失败、模型调用失败或 JSON 提取失败时抛出。

## 主要函数

### `run_stage(stage_name, context, config, *, dry_run=False, run_dir=None, save_result=False) -> dict`

执行一个 stage。

#### 参数

| 参数 | 类型 | 说明 |
|---|---:|---|
| `stage_name` | str | 要执行的 stage 名称，必须存在于 `config["stages"]` |
| `context` | dict | 运行上下文，至少包含 `input.video_path` |
| `config` | dict | 读取后的 `config.yaml` |
| `dry_run` | bool | true 时不调用模型，仍执行视频处理和 prompt 渲染 |
| `run_dir` | str/Path/null | 保存结果的运行目录 |
| `save_result` | bool | 是否调用 `result_io.save_stage_result` |

当 stage 的 `video.save_processed=true` 且传入了 `run_dir` 时，当前 stage 送入模型的处理后媒体会保存到：

```text
<run_dir>/stages/<stage_name>/processed_frames/
```

单视角或 `merge_views=true` 时，帧和可选 `processed.mp4` 位于该目录根层。`merge_views=false` 且选中多个视角时，分别保存到 `processed_frames/<view_name>/`。

`build_video_layout_description()` 会把实际消息布局注入 prompt：多视角非合并时明确说明各视角独立采样，并说明每组图像序列或视频前均有视角文字标签。

当 `video.add_frame_tags=true` 且 `input_mode=image_sequence` 时，布局说明还会明确 `<t=...s>` 是原始视频时间戳、`<view_name>` 是拍摄视角；逐帧标签位于对应 `image_url` 前。

该路径不再通过 `config.yaml` 中的 `processed_output_path` 配置。

#### context 输入格式

```python
context = {
  "input": {
    "video_path": "...",
    "instruction": "...",
    "video_id": "optional"
  },
  "stages": {}
}
```

可选的 `input.video_segments` 用于描述 episode 的某个视角只是大视频中的一段。外层 key 是视角名，应与
`input.video_path` 或 stage 配置里的 `video.view_names` 对齐：

```python
context = {
  "input": {
    "video_path": {
      "observation.images.image_0": "/data/videos/image_0/file-000.mp4"
    },
    "video_segments": {
      "observation.images.image_0": {
        "video_path": "/data/videos/image_0/file-000.mp4",
        "start_time": 4.6,
        "end_time": 9.2,
        "fps": 5.0
      }
    },
    "instruction": "flip cup upright"
  },
  "stages": {}
}
```

切片规则：

- 如果 `start_frame/end_frame` 存在且不是 `-1/-1`，优先按闭区间帧号切片，并在 `video_meta.source_video_segments`
  记录 `segment_mode="frame"`。
- 否则如果 `start_time/end_time` 存在且不是 `-1/-1`，按秒切片，并记录 `segment_mode="time"`。
- 如果 `start_time/end_time` 是 `-1/-1`，直接使用完整原视频，并记录 `segment_mode="full"`；对只提供
  frame pair 的旧格式，`start_frame/end_frame=-1/-1` 也表示完整原视频。
- 如果某个 segment 没有写 `video_path`，runner 会回退到匹配视角的 `input.video_path`。

每个 stage 还可以通过 `episode_fields` 把输入 JSON 中的字段传给 prompt：

```yaml
scene:
  episode_fields:
    - fps
    - length
    - video_path.camera_front
```

这些字段会在 prompt 中挂到 `ctx.episode`：

```text
{{ ctx.episode.fps }}
{{ ctx.episode.length }}
{{ ctx.episode.video_path.camera_front }}
```

字段值为 JSON `null` 时会渲染为 `null`。未配置 `episode_fields` 的 stage 不会继承其他 stage 的
`ctx.episode`。

#### context 输出格式

```python
context["stages"][stage_name] = {
  "output": parsed_json,
  "raw_text": raw_text,
  "system_prompt": system_prompt,
  "prompt": user_prompt,
  "video_meta": video_meta,
  "usage": usage,
  "model": model,                 # 非 dry-run 且模型返回时写入
  "finish_reason": finish_reason, # 非 dry-run 且模型返回时写入
}
```

## 内部函数

| 函数 | 作用 |
|---|---|
| `_model_cfg(config, stage_cfg)` | 合并顶层 `model` 参数和 stage 内 `generation` 覆盖参数 |
| `_dry_run_output(stage_name)` | 为 dry-run 生成合法的假输出 |
| `_ensure_context(context)` | 检查并初始化 `context` 必需字段 |

## dry-run 说明

`dry_run=True` 仍会处理视频、解析 prompt 占位符并写入 context，只跳过真实 VLM 请求。适合检查：

1. stage 配置是否正确；
2. prompt 是否能渲染；
3. 上游字段路径是否存在；
4. pipeline 是否能串联。
