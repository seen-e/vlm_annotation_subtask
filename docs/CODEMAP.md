# Codemap

本文档逐文件说明当前代码结构、主要函数、输入输出、调用关系和外部调用边界。

## `config.yaml`

职责：主配置文件，控制 workflow、模型参数、stage prompt 和视频处理参数。

主要结构：

| 字段 | 说明 |
|---|---|
| `workflow` | stage 执行顺序，当前为 `scene -> analysis -> refinement`。 |
| `model` | 默认模型调用参数：`base_url`、`api_key`、`model`、`max_tokens`、`temperature`、`top_p`、`top_k`、`max_retries`。 |
| `stages.<stage>.prompt` | prompt 模块路径和变量名。 |
| `stages.<stage>.output_key` | 语义标识；当前核心代码不使用它决定 context key。 |
| `stages.<stage>.video` | 当前 stage 的视频读取、抽帧、拼接、编码参数。 |
| `stages.<stage>.generation` | 可选，覆盖顶层 `model` 参数。 |

依赖：被 examples 读取后传给 `pipeline.run_pipeline()` 或 `stage_runner.run_stage()`。

## `video_process.py`

职责：standalone 视频预处理层。输入单视频/list/dict 多视角视频，输出 OpenAI-compatible image/video media parts 和 `video_meta`。

主要函数：

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `normalize_video_input(video_path, view_names=None)` | 单路径、路径 list 或 `{view: path}` dict；可选视角顺序 | `(view_map, input_type)` | `build_video_inputs()` | 可用于调试输入归一化 |
| `read_video_info(path)` | 视频路径 | FPS、帧数、尺寸、时长 dict | `build_video_inputs()` | 可用于调试视频 metadata |
| `compute_sample_timestamps(original_fps, frame_count, target_fps, max_frames, min_api_frames=1, frame_start=0, frame_end=None)` | 主视角 FPS/帧数、采样参数和帧范围 | `(indices, timestamps)` | `build_video_inputs()` | 可用于调试抽帧策略 |
| `resize_keep_aspect(frame, resize_width)` | OpenCV BGR frame、目标宽度 | 缩放后的 frame | `_build_timepoint_frames()` | 工具函数 |
| `draw_overlay(frame, timestamp=None, view_name=None, draw_timestamps=True, draw_view_names=True)` | frame、时间戳、视角名和绘制开关 | 带标签的 frame copy | `_build_timepoint_frames()` | 工具函数 |
| `merge_view_frames(frames, separator=4)` | 同一时间点的多视角 frames | 纵向拼接 frame | `_build_timepoint_frames()` | 工具函数 |
| `merge_temporal_frames(frames, columns=None)` | 连续时间点 frames | 横向时间序列 montage frame | `_apply_temporal_merge()` | 工具函数 |
| `encode_frame_to_image_part(frame, jpeg_quality)` | BGR frame、JPEG 质量 | OpenAI-compatible `image_url` part | `build_video_inputs()` | 可用于单帧编码 |
| `encode_frames_to_video_part(frames, timestamps, fallback_fps)` | 处理后帧、真实时间戳和回退 FPS | MP4 `video_url` part、payload 和编码 metadata | `_encode_processed_media()` | 视频编码入口 |
| `save_processed_frames(...)` | 最终 frames、逐视角 frames、MP4 payload、metadata、目录 | `None` | `build_video_inputs()` | 调试保存入口 |
| `build_video_inputs(video_path, *, fps, max_frames, resize_width, jpeg_quality, max_time=-1, draw_timestamps=True, draw_view_names=True, min_api_frames=1, frame_start=0, frame_end=None, merge_views=False, merge_mode="per_frame", merge_length=0, view_names=None, input_mode="image_sequence", add_frame_tags=False, save_processed_path=None)` | 视频输入和全部视频参数 | `(parts, video_meta)` | `stage_runner.run_stage()` | 主要公开入口 |

内部函数，不建议外部直接调用：

| 函数 | 作用 |
|---|---|
| `_normalize_jpeg_quality(value)` | JPEG quality 转 int 并校验 1-100。 |
| `_normalize_max_time(value)` | 将 `max_time` 归一化为 float，并校验只能是 `-1` 或 `>0`。 |
| `_validate_video_config(...)` | 校验 fps、max_frames、max_time、resize_width、input_mode、merge_mode、min_api_frames。 |
| `_effective_time_window(...)` | 计算 `max_time`、多视角原始时长、`frame_start/frame_end` 的交集，并生成处理区间 metadata。 |
| `_read_frame(path, frame_index)` | 用 OpenCV 读取单帧。 |
| `_pad_to_size(frame, target_h, target_w)` | padding 到指定尺寸。 |
| `_build_timepoint_frames(...)` | 根据采样时间点读取各视角帧、缩放、绘制标签、按需合并视角。 |
| `_apply_temporal_merge(frames, groups, merge_length)` | 按 `merge_length` 把连续输出 frame 合成横向时间序列 montage。 |
| `_process_view_set(...)` | 对单视角或共同时间轴的合并视角统一执行时间窗、采样和图像处理。 |
| `_encode_processed_media(...)` | 根据 `input_mode` 编码图像序列或 MP4。 |
| `_frame_message_tag(...)` | 为单张输出图像生成 `<t=...s> <view_name>` message 标签。 |
| `_view_message_label(...)` | 为非合并多视角的每组媒体生成视角说明文字 part。 |

依赖：`cv2`、`numpy`、标准库。当前不依赖旧项目代码。

重要行为：

1. 多视角 dict 输入可用 `view_names` 选择和排序视角。
2. 第一个视角是 primary view；合并多视角时采样时间轴基于它。
3. 非合并多视角时，每个视角独立采样和编码，并通过 message 文字 part 标识来源。
4. `max_time` 是 stage 级视频参数，在抽帧和后续视频处理前统一生效；输出时间戳仍对应原始视频时间轴。
5. 多视角输入会对各视角使用同一个 `max_time` 上限，但不会补齐较短视角。
6. `input_mode="video"` 编码 MP4 `video_url`；模型服务不支持该 content part 时，错误会发生在真实 API 调用阶段。

## `model_client.py`

职责：封装 OpenAI-compatible VLM 调用。

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `call_vlm(...)` | base_url、api_key、model、system/user prompt、image_parts、生成参数 | `raw_text` 字符串 | 兼容/简化调用场景 | 需要纯文本时使用 |
| `call_vlm_with_metadata(...)` | 同上 | dict：`raw_text`、`model`、`usage`、`finish_reason` | `stage_runner.run_stage()`、`call_vlm()` | stage 执行入口默认使用 |

内部函数：

| 函数 | 作用 |
|---|---|
| `_is_qwen_model(model)` | 判断 Qwen/QVQ 模型。 |
| `_build_messages(system_prompt, user_prompt, image_parts, *, model)` | 构造 chat messages；Qwen 系会把 system/user 合并进 user message。 |

依赖：运行真实模型调用时需要 `openai`。

## `json_utils.py`

职责：从 VLM 文本中抽取 JSON。

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `extract_json(raw_text)` | 模型原始文本 | dict 或 list | `stage_runner.run_stage()` | 公开入口 |

内部函数：

| 函数 | 作用 |
|---|---|
| `_strip_thinking(text)` | 去除 `<think>...</think>`。 |
| `_try_parse(text)` | 尝试 `json.loads`。 |
| `_find_balanced_span(text, open_char, close_char)` | 找平衡 JSON object/array 范围。 |
| `_extract_fenced_blocks(text)` | 提取 Markdown code block 内容。 |
| `_extract_balanced_json(text)` | 从文本中提取第一个完整 JSON。 |

## `prompt_utils.py`

职责：加载 prompt 模块并渲染占位符。

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `resolve_context_path(context, path)` | context、点分路径 | 任意值 | `render_template()`、`resolve_input_fields()` | 可用于调试 `ctx.*` |
| `resolve_prompt_path(path)` | 如 `common.JSON_ONLY_RULE` | prompt 变量值 | `render_template()` | 可用于调试 `prompt.*` |
| `render_template(template, *, context, extra_vars=None)` | 模板、context、兼容旧变量 | 渲染后字符串 | `stage_runner.run_stage()` | 公开入口 |
| `load_stage_prompt(stage_cfg, *, stage_name=None)` | stage 配置 | `(system_template, user_template)` | `stage_runner.run_stage()` | 公开入口 |
| `resolve_input_fields(context, input_fields)` | context 和旧式 input_fields | dict | `stage_runner.run_stage()` | 兼容旧配置 |

内部函数：

| 函数 | 作用 |
|---|---|
| `_to_text(value)` | dict/list/tuple 转 pretty JSON，None 转 `null`。 |
| `_import_prompt_module(module_name)` | 按候选路径动态导入 prompt 模块。 |

## `stage_runner.py`

职责：执行单个 stage。

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `run_stage(stage_name, context, config, *, dry_run=False, run_dir=None, save_result=False)` | stage 名、context、config、dry-run 和保存参数 | parsed JSON 输出 | `pipeline.run_pipeline()`、`examples/run_stage.py` | 单 stage 调试入口 |

内部函数：

| 函数 | 作用 |
|---|---|
| `_model_cfg(config, stage_cfg)` | 合并顶层 `model` 和 stage `generation`。 |
| `_dry_run_output(stage_name)` | 生成内置假输出。 |
| `_ensure_context(context)` | 初始化并检查 `context["input"]["video_path"]`。 |
| `_resolve_processed_output_path(...)` | 当 `video.save_processed=true` 且存在 `run_dir` 时，返回 `<run_dir>/stages/<stage_name>/processed_frames`。 |

调用链：

```text
run_stage
  -> video_process.build_video_inputs
  -> prompt_utils.resolve_input_fields
  -> prompt_utils.load_stage_prompt
  -> prompt_utils.render_template
  -> model_client.call_vlm_with_metadata  # dry_run=False
  -> json_utils.extract_json      # dry_run=False
  -> result_io.save_stage_result  # save_result=True
```

## `pipeline.py`

职责：按顺序 workflow 调度多个 stage。

| 函数 | 输入参数 | 返回值 | 被谁调用 | 外部调用建议 |
|---|---|---|---|---|
| `run_pipeline(context, config, *, workflow=None, dry_run=False, start_from=None, stop_after=None, skip_existing=False, output_dir=None, run_name=None, save_results=False)` | context、config、workflow 覆盖、范围控制、保存参数 | 更新后的 context | `examples/run_pipeline.py` | 完整 pipeline 入口 |

内部函数：

| 函数 | 作用 |
|---|---|
| `_get_workflow(config, workflow)` | 获取并校验 workflow。 |
| `_slice_workflow(stages, start_from, stop_after)` | 根据起止 stage 裁剪执行范围。 |

调用关系：

```text
pipeline.run_pipeline
  -> stage_runner.run_stage
  -> result_io.make_run_dir   # save_results=True
  -> result_io.save_context   # save_results=True
```

## `result_io.py`

职责：轻量级结果保存和加载。

| 函数 | 输入参数 | 返回值 | 外部调用建议 |
|---|---|---|---|
| `ensure_dir(path)` | 路径 | Path | 可用 |
| `save_json(data, path)` | 数据和路径 | `None` | 可用 |
| `load_json(path)` | JSON 路径 | Python 值 | 可用 |
| `save_text(text, path)` | 文本和路径 | `None` | 可用 |
| `load_text(path)` | 文本路径 | str | 可用 |
| `make_run_dir(output_dir="outputs", *, run_name=None, video_id=None)` | 输出根目录、run 名或 video id | Path | 可用 |
| `save_stage_result(context, stage_name, run_dir)` | context、stage、run 目录 | `None` | 可用 |
| `save_context(context, run_dir)` | context、run 目录 | `None` | 可用 |
| `load_stage_output(stage_name, run_dir)` | stage、run 目录 | output dict | 可用 |
| `load_stage_into_context(context, stage_name, run_dir)` | context、stage、run 目录 | `None` | 可用 |
| `load_outputs_into_context(context, run_dir, stage_names)` | context、run 目录、stage 列表 | `None` | 可用 |

内部函数：`_json_default()`、`_safe_name()`、`_stage_dir()`。

## `prompts/*.py`

| 文件 | 职责 | 当前占位符 |
|---|---|---|
| `prompts/common.py` | 公共 prompt 片段和动作词表 | 无 |
| `prompts/scene.py` | scene prompt，输出场景上下文 | `prompt.common.JSON_ONLY_RULE`、`ctx.input.instruction` |
| `prompts/analysis.py` | analysis prompt，输出动作序列 | `prompt.common.JSON_ONLY_RULE`、`ctx.input.instruction`、`ctx.stages.scene.output`、`prompt.common.ACTION_VOCABULARY` |
| `prompts/refinement.py` | refinement prompt，输出精修时间段 | `prompt.common.JSON_ONLY_RULE`、`ctx.input.instruction`、`ctx.stages.scene.output`、`ctx.stages.analysis.output.action_sequence`、`prompt.common.TIME_BOUNDARY_RULE` |

## `examples/*.py`

| 文件 | 职责 |
|---|---|
| `examples/run_stage.py` | 单 stage 调试入口。 |
| `examples/run_pipeline.py` | 完整 pipeline 或从中间 stage 继续执行的示例入口。 |

## 总体依赖图

```text
examples/run_pipeline.py
  -> pipeline.py
       -> stage_runner.py
            -> video_process.py
            -> prompt_utils.py
            -> model_client.py
            -> json_utils.py
            -> result_io.py
       -> result_io.py

examples/run_stage.py
  -> stage_runner.py
  -> result_io.py

prompt_utils.py
  -> prompts/*.py dynamic import
```

## 建议后续优化

1. 统一目录名和 Python 包名，避免 examples 中 `vlm_auto_annotation_refactor` 导入失败。
2. 当前 `stage_runner.py` 已写入 `call_vlm_with_metadata()` 返回的 usage；后续如调整 usage 字段口径，应同步 `result_io.py` 和文档。
3. `input_mode="video"` 的 `video_url` 兼容性取决于实际 OpenAI-compatible 模型服务，部署切换时应先做小样本验证。
4. `merge_mode="timeline_grid"` 会按 `merge_length` 合并时间点；`merge_length < 1` 表示合并全部时间点。
