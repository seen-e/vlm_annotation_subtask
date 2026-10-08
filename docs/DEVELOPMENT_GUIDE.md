# Development Guide

本文档说明后续开发当前 standalone 轻量框架时应遵守的规则。

## 开发原则

1. 不要把 `scene`、`analysis`、`refinement` 写死进核心代码；它们只是默认配置。
2. 新增 stage 优先通过 `config.yaml + prompts/*.py` 完成。
3. 不要轻易修改 `pipeline.py` 和 `stage_runner.py` 的核心流程。
4. prompt 实验优先修改 `prompts/*.py`。
5. 视频参数实验优先修改 `config.yaml`。
6. 上游输出字段传递优先通过 `{{ ctx.stages.xxx.output.xxx }}` 完成。
7. 公共 prompt 片段优先放到 `prompts/common.py`。
8. 新增功能优先保持单文件轻量模块，不要引入复杂目录结构。
9. `video_process.py` 当前是 standalone 视频层；不要重新引入旧项目 `utils.video_utils` 依赖。
10. `json_utils.py` 只负责 JSON 抽取；业务 schema 校验如需增加，建议做独立轻量模块。

## 新增 stage 的方法

以 `object_state` 为例。

### 1. 新增 `prompts/<stage_name>.py`

新增：

```text
prompts/object_state.py
```

至少定义：

```python
SYSTEM_PROMPT = """
你是机器人操作视频物体状态分析助手。
{{ prompt.common.JSON_ONLY_RULE }}
"""

USER_PROMPT_TEMPLATE = """
任务指令：
{{ ctx.input.instruction }}

Scene 阶段输出：
{{ ctx.stages.scene.output }}

请输出物体状态变化。

严格返回 JSON：
{
  "object_states": []
}
"""
```

### 2. 在 `config.yaml` 的 `stages` 下新增配置

```yaml
object_state:
  prompt:
    module: "vlm_auto_annotation_refactor.prompts.object_state"
    system: "SYSTEM_PROMPT"
    user: "USER_PROMPT_TEMPLATE"
  output_key: "object_state"
  video:
    input_mode: "image_sequence"
    max_time: -1
    fps: 1.0
    max_frames: 64
    resize_width: 336
    jpeg_quality: 85
    draw_timestamps: true
    draw_view_names: true
    min_api_frames: 2
    merge_views: true
    merge_mode: "per_frame"
    merge_length: 8
    view_names:
      - camera_front
      - camera_top
```

注意：`output_key` 当前不决定 context key。下游仍通过 stage 名称引用：

```text
{{ ctx.stages.object_state.output }}
```

### 3. 在 `workflow` 中插入 stage

```yaml
workflow:
  - scene
  - object_state
  - analysis
  - refinement
```

### 4. 在下游 prompt 中引用结果

```python
Object state 阶段输出：
{{ ctx.stages.object_state.output }}
```

或引用具体字段：

```python
{{ ctx.stages.object_state.output.object_states }}
```

### 5. 用 `dry_run=True` 验证 prompt 渲染

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --dry-run \
  --save-results
```

检查：

```text
outputs/<run>/stages/object_state/user_prompt.txt
outputs/<run>/stages/analysis/user_prompt.txt
```

### 6. 再真实调用模型

确认 prompt 和视频处理都正常后，去掉 `--dry-run`。

## 修改 stage 间字段传递

只改 prompt 即可。示例：

```python
{{ ctx.stages.scene.output.executors }}
{{ ctx.stages.analysis.output.action_sequence }}
{{ ctx.stages.object_state.output.object_states }}
```

支持 list 下标：

```python
{{ ctx.stages.analysis.output.action_sequence.0.action }}
```

路径不存在时会抛 `PromptRenderError`，通常说明上游没有执行/加载，或字段名与实际 output 不一致。

## 从 prompt 文件引用公共片段

在 `prompts/common.py` 中放公共片段：

```python
JSON_ONLY_RULE = "..."
ACTION_VOCABULARY = [...]
TIME_BOUNDARY_RULE = "..."
```

在其他 prompt 中引用：

```python
{{ prompt.common.JSON_ONLY_RULE }}
{{ prompt.common.ACTION_VOCABULARY }}
{{ prompt.common.TIME_BOUNDARY_RULE }}
```

list/dict 会自动转成 pretty JSON 字符串。

## 视频处理开发规则

当前 `video_process.py` 已经承担 standalone 视频层职责：

1. 输入归一化：`normalize_video_input()`。
2. 视频信息读取：`read_video_info()`。
3. 有效时间区间计算：`_effective_time_window()` 统一处理 `max_time`、多视角时长和 `frame_start/frame_end` 的交集。
4. 抽帧策略：`compute_sample_timestamps()`。
5. 帧处理：缩放、绘制时间戳/视角名。
6. 多视角拼接：`merge_view_frames()`。
7. 时间 montage：`merge_temporal_frames()` / `_apply_temporal_merge()`。
8. JPEG base64 编码：`encode_frame_to_image_part()`。

开发时优先在这个文件内保持清晰函数边界，不要把视频逻辑散落到 `stage_runner.py`。

`max_time` 属于 stage 级 `video` 配置，新增或调整视频时长实验时应优先改 `config.yaml`。不要在 `scene`、`analysis`、`refinement` 的 prompt 或 stage 逻辑里分别实现截断；统一入口应保持在 `build_video_inputs()` 内。

## 何时应修改核心流程

只有这些情况才考虑修改 `pipeline.py` 或 `stage_runner.py`：

1. 所有 stage 都需要共享的新横切行为。
2. 顺序 workflow 无法表达需求，需要 DAG 或条件执行。
3. 需要改变 dry-run、错误处理、保存策略等核心行为。
4. 需要改变模型 metadata/usage 写入 context 的方式。

如果只是新增 stage、改 prompt、改字段传递或改视频参数，不应修改核心流程。

## 建议后续优化

1. 统一目录名和包名，避免 `vlm_auto_annotation_refactor_gpt` 与 `vlm_auto_annotation_refactor` 不一致。
2. 当前 `stage_runner.py` 已写入 `model_client.call_vlm_with_metadata()` 返回的 usage；如果字段口径变化，应同步 `result_io.py` 和相关文档。
3. 明确 `merge_mode="timeline_grid"` 在 standalone 版本中的语义。
4. `input_mode="video"` 已通过 MP4 `video_url` 实现，但不同模型服务的支持范围可能不同；更换部署时应先验证服务端兼容性。
