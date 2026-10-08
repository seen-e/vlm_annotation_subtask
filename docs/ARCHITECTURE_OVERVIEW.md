# Architecture Overview

本文档基于当前代码重新梳理 `vlm_auto_annotation_refactor` 的整体架构。当前版本是 standalone 轻量实验框架：核心视频处理逻辑已内置在 `video_process.py`，不再依赖旧项目中的 `flows/`、`utils/`、`annotation_pipeline/` 等模块。

## 设计目标

当前框架服务于多阶段 VLM 视频标注实验，默认 workflow 是：

```text
scene -> analysis -> refinement
```

设计目标：

1. 用 `config.yaml` 控制 stage 顺序、模型参数、prompt 模块和视频处理参数。
2. 用 `prompts/*.py` 管理 system/user prompt 模板和公共 prompt 片段。
3. 用运行时 `context` 在 stage 之间传递结构化 JSON 输出。
4. 用 `result_io.py` 保存和加载 stage 中间结果，支持固定上游结果复跑下游 stage。
5. 用 standalone `video_process.py` 直接完成视频读取、抽帧、多视角拼接、时间帧 montage 和 base64 JPEG 编码。

## 为什么是轻量实验框架

这个项目不是复杂平台架构，而是面向实验迭代的轻量框架。它把核心流程拆成几个单文件模块，每个模块只处理一个明确环节：

```text
视频输入 -> prompt 渲染 -> VLM 调用 -> JSON 抽取 -> context 写入 -> 结果保存
```

轻量化的好处是：

1. 新增 stage 通常只需要新增 `prompts/<stage>.py` 并修改 `config.yaml`。
2. 默认 stage 名称不写死在 `pipeline.py`，workflow 可配置。
3. 下游 stage 通过 `{{ ctx.stages.xxx.output.xxx }}` 引用上游结果，不需要新增传参代码。
4. 输出是普通 JSON/text 文件，便于人工检查和实验比较。
5. 视频处理已内聚到本仓库，不需要旧项目运行环境。

## 核心模块职责

| 模块 | 一句话职责 |
|---|---|
| `config.yaml` | 定义 workflow、顶层模型参数、每个 stage 的 prompt 配置、视频参数和可选 generation 覆盖。 |
| `video_process.py` | standalone 视频输入层，读取单视频/多视角视频，抽帧、缩放、绘制标签、按需拼接，并编码为 OpenAI-compatible `image_url` / `video_url` media parts。 |
| `model_client.py` | OpenAI-compatible VLM client，构造 chat messages 并返回模型文本。 |
| `json_utils.py` | 从模型文本中提取 JSON object 或 JSON array。 |
| `prompt_utils.py` | 加载 prompt 模块，并渲染 `{{ ctx.* }}`、`{{ prompt.* }}`、旧式 `extra_vars` 占位符。 |
| `stage_runner.py` | 执行单个 stage，串联视频处理、prompt 渲染、模型调用、JSON 解析和 context 写入。 |
| `pipeline.py` | 按 workflow 顺序执行多个 stage，支持 `start_from`、`stop_after`、`skip_existing` 和保存结果。 |
| `result_io.py` | 保存/加载 stage output、raw_text、prompt、video_meta、usage 和完整 context。 |
| `prompts/common.py` | 定义公共 prompt 片段，如 JSON-only 规则、动作词表、时间边界规则。 |
| `prompts/scene.py` | 定义 scene 阶段 prompt，用于提取场景上下文。 |
| `prompts/analysis.py` | 定义 analysis 阶段 prompt，用于从视频和 scene 输出生成动作序列。 |
| `prompts/refinement.py` | 定义 refinement 阶段 prompt，用于根据 analysis 输出精修动作时间边界。 |

## 当前不包含的复杂模块

当前框架刻意不包含：

1. schema registry：没有集中注册、版本化 stage 输出 schema。
2. artifact store：没有独立 artifact 存储服务，只保存本地 JSON/text/JPEG。
3. DAG workflow：`pipeline.py` 只支持顺序 workflow，不支持任意依赖图。
4. plugin system：stage/prompt 是普通配置和 Python 模块，没有插件注册系统。
5. complex CLI：`examples/run_pipeline.py` 和 `examples/run_stage.py` 是示例入口，不是完整命令行产品。
6. 并发调度或队列系统：pipeline 当前串行执行。
7. 强业务 schema 校验：`json_utils.py` 只负责提取 JSON，不验证业务字段。

## 适合的使用场景

1. 多阶段 VLM 视频标注实验。
2. prompt 消融实验。
3. stage 插入 / 删除实验。
4. 固定上游结果复跑下游 stage。
5. 视频参数实验，例如 max_time、fps、max_frames、resize_width、jpeg_quality、多视角拼接和 montage 设置。

## 重要实现备注

1. 当前 `video_process.py` 是 standalone 实现，不再依赖旧 `utils.video_utils`。
2. `input_mode="image_sequence"` 输出 JPEG parts；`input_mode="video"` 输出处理后 MP4 `video_url` parts，真实调用要求服务端支持该格式。
3. `merge_mode` 只控制时间维度：`per_frame` 逐时间点输出，`timeline_grid` 按 `merge_length` 从左到右合并时间点；`merge_length < 1` 表示全部时间点合成一张图。
4. `max_time` 是 stage 级视频处理上限，在抽帧、缩放、多视角合并和编码前统一生效；它限制原始时间轴上的有效区间，不改变 FPS 或时间戳。
5. `merge_views=false` 时，单视角保持原有消息形状；多视角会分别处理并全部发送，每组媒体前带视角文字标签。
5. `dry_run=True` 仍会执行视频处理和 prompt 渲染，只跳过真实 VLM 调用。
6. 当前目录名是 `vlm_auto_annotation_refactor_gpt`，但 README、examples 和 config 中的包名是 `vlm_auto_annotation_refactor`。如果没有安装/映射同名包，直接运行 examples 可能导入失败。建议后续统一目录名和包名。
