# Dataflow

本文档说明一次完整 pipeline 的数据流，以及 `context`、prompt 占位符、dry-run 和结果保存如何协作。

## 总体流程

```text
config.yaml
  ↓
pipeline.run_pipeline
  ↓
stage_runner.run_stage
  ↓
video_process.build_video_inputs
  ↓
prompt_utils 渲染 system/user prompt
  ↓
model_client.call_vlm_with_metadata
  ↓
json_utils.extract_json
  ↓
context["stages"][stage_name]
  ↓
result_io 保存结果
```

真实执行顺序：

```text
examples/run_pipeline.py
  -> 读取 config.yaml
  -> 构造 context["input"]
  -> pipeline.run_pipeline(context, config)
       -> 校验 workflow
       -> 裁剪 selected_stages
       -> 对每个 stage 调用 stage_runner.run_stage
            -> video_process.build_video_inputs
            -> prompt_utils.load_stage_prompt
            -> prompt_utils.render_template
            -> dry_run ?
                 yes: 使用 _dry_run_output
                 no : call_vlm_with_metadata -> extract_json
            -> 写入 context["stages"][stage_name]
       -> 可选 save_context
```

## 视频输入数据流

`stage_runner.py` 调用：

```python
media_parts, video_meta = build_video_inputs(context["input"]["video_path"], **video_cfg)
```

`video_process.py` 内部流程：

```text
video_path
  ↓
normalize_video_input
  ↓
read_video_info
  ↓
_effective_time_window  # max_time / frame range / view duration 交集
  ↓
compute_sample_timestamps  # 合并时用 primary 时间轴；非合并多视角时分别计算
  ↓
_build_timepoint_frames
  ↓
resize_keep_aspect
  ↓
draw_overlay
  ↓
merge_view_frames          # merge_views=True
  ↓
_apply_temporal_merge      # 每个待编码序列内部执行 timeline_grid
  ↓
encode_frame_to_image_part / encode_frames_to_video_part
  ↓
_frame_message_tag         # image_sequence + add_frame_tags=true
  ↓
视角文字标签 + media_parts + video_meta  # 非合并多视角
```

重要细节：

1. `video_path` 可是单路径、路径 list 或 `{view_name: path}` dict。
2. dict 输入时，`view_names` 可选择和排序视角。
3. 第一个视角仍作为 primary view。
4. `merge_views=true` 时，其他视角按 primary timestamp 映射到自己的帧；`merge_views=false` 且有多个视角时，每个视角按自己的原始时间轴独立处理并全部发送。
5. `max_time` 是每个 stage 独立的视频时长上限，在抽帧、缩放、多视角合并、图像序列生成和编码之前生效。
6. `max_time=-1` 表示不限制；`max_time>0` 表示只处理原始时间轴 `[0, min(max_time, original_duration))`；`0` 或 `<-1` 会抛配置异常。
7. 如果同时有 `frame_start/frame_end`，最终处理区间是各限制的交集；非合并多视角对每个视角分别计算，合并多视角再取共同视角交集。
8. 多视角输入时不会循环、补帧、减速或修改时间戳；采样时间戳仍对应原始视频起点。
9. `input_mode="image_sequence"` 生成 JPEG `image_url`；`input_mode="video"` 将处理后帧编码为 MP4 `video_url`，目标服务必须支持视频 content part。
10. 非合并多视角的 message 在每组媒体前插入文字标签说明视角；`video_meta.view_outputs` 保存逐视角采样信息，`output_groups` 保存全局媒体顺序。
11. `add_frame_tags=true` 时，每个 image_url 前还会插入 `<t=...s> <view_name>` text part；时间来自原始视频时间轴，video 模式不启用逐帧标签。

## `context` 结构

最小输入：

```python
context = {
    "input": {
        "video_path": "/path/to/video.mp4",
        "instruction": "pick up the cup",
        "video_id": "demo_episode"
    },
    "stages": {}
}
```

`context["input"]` 当前常见字段：

| 字段 | 作用 |
|---|---|
| `video_path` | 传给 `video_process.build_video_inputs()`；必需。 |
| `instruction` | 被默认 prompts 通过 `{{ ctx.input.instruction }}` 引用。 |
| `video_id` | 保存结果时可作为默认 run name。 |

`pipeline.run_pipeline()` 会补充：

```python
context["pipeline"] = {
    "workflow": [...],
    "selected_stages": [...],
    "executed_stages": [...],
    "skipped_stages": [...]
}
```

如果 `save_results=True`，还会写入：

```python
context["run_dir"] = "outputs/<run_name>"
```

## `context["stages"][stage_name]`

每个 stage 运行后写入：

```python
context["stages"][stage_name] = {
    "output": parsed_json,
    "raw_text": raw_text,
    "system_prompt": system_prompt,
    "prompt": user_prompt,
    "video_meta": video_meta,
    "usage": usage,
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `output` | 模型输出解析后的 JSON，或 dry-run 假输出。 |
| `raw_text` | 模型原文，或 dry-run JSON 字符串。 |
| `system_prompt` | 渲染后的 system prompt。 |
| `prompt` | 渲染后的 user prompt。 |
| `video_meta` | 抽帧、视角、拼接、编码等 metadata。 |
| `usage` | 模型返回的 token usage metadata；dry-run 时为空 dict。 |

`video_meta` 还会包含用于确认有效处理范围的字段：

```json
{
  "original_duration": 120.0,
  "configured_max_time": 30.0,
  "effective_duration": 30.0,
  "was_time_limited": true
}
```

## 上游输出如何传给下游 stage

上游输出存在：

```python
context["stages"][stage_name]["output"]
```

下游 prompt 通过占位符引用：

```text
{{ ctx.stages.scene.output }}
{{ ctx.stages.scene.output.executors }}
{{ ctx.stages.analysis.output.action_sequence }}
```

当前默认 prompt 依赖关系：

```text
scene
  ↓
analysis: 引用 ctx.stages.scene.output
  ↓
refinement: 引用 ctx.stages.scene.output 和 ctx.stages.analysis.output.action_sequence
```

## `{{ ctx.stages.scene.output.xxx }}` 如何解析

`prompt_utils.render_template()` 发现 `ctx.` 前缀后调用：

```python
resolve_context_path(context, path)
```

解析规则：

1. dict：路径片段作为 key。
2. list/tuple：路径片段必须是整数下标。
3. 普通对象：尝试属性访问。
4. 任一路径不存在会抛 `PromptRenderError`。

示例：

```text
{{ ctx.stages.analysis.output.action_sequence.0.action }}
```

会依次访问：

```text
stages -> analysis -> output -> action_sequence -> 0 -> action
```

## `{{ prompt.common.xxx }}` 如何解析

`prompt.` 前缀会触发：

```python
resolve_prompt_path("common.JSON_ONLY_RULE")
```

解析过程：

1. 动态导入 `prompts.common` / `vlm_auto_annotation_refactor.prompts.common`。
2. 读取变量。
3. 用 `_to_text()` 转成 prompt 文本。

常见引用：

```python
{{ prompt.common.JSON_ONLY_RULE }}
{{ prompt.common.ACTION_VOCABULARY }}
{{ prompt.common.TIME_BOUNDARY_RULE }}
```

list/dict 会被转成 pretty JSON 字符串。

## `dry_run=True`

仍会执行：

1. context 检查。
2. stage 配置读取。
3. 视频处理：`build_video_inputs()`。
4. prompt 加载和渲染。
5. context 写入。
6. 可选结果保存。

会跳过：

1. `model_client.call_vlm_with_metadata()`。
2. `json_utils.extract_json()`。

替代输出来自 `_dry_run_output(stage_name)`：

| stage | dry-run output |
|---|---|
| `scene` | `scene_summary`、`executors`、`touched_objects`、`background_objects`、`best_observation_views` |
| `analysis` | `action_sequence` |
| `refinement` | `refined_segments` |
| 其他 stage | `{"stage": stage_name, "status": "dry_run_ok"}` |

注意：dry-run 不需要真实模型服务，但仍需要视频文件存在，并且需要 `opencv-python` / `numpy` 可用。

## `skip_existing=True`

`pipeline.run_pipeline()` 对每个 selected stage 检查：

```python
if skip_existing and stage_name in context.get("stages", {}):
    skipped.append(stage_name)
    continue
```

用途：先用 `result_io.load_outputs_into_context()` 加载上游结果，然后跳过已有 stage，只跑下游。

示例：

```python
load_outputs_into_context(context, "outputs/demo", ["scene", "analysis"])
run_pipeline(context, config, start_from="scene", stop_after="refinement", skip_existing=True)
```

如果 selected stages 只有下游，例如 `start_from="refinement"`，则不一定需要 `skip_existing=True`。

## 保存结果

`save_results=True` 时：

```text
pipeline.run_pipeline
  -> result_io.make_run_dir
  -> stage_runner.run_stage(..., save_result=True)
       -> result_io.save_stage_result
  -> result_io.save_context
```

目录结构：

```text
outputs/<run_name>/
  input.json
  context.json
  stages/<stage_name>/
    processed_frames/
      frame_000000.jpg
      frame_000001.jpg
      video_meta.json
    system_prompt.txt
    user_prompt.txt
    raw_text.txt
    output.json
    video_meta.json
    usage.json
```
