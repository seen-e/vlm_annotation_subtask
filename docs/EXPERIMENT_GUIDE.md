# Experiment Guide

本文档说明如何基于当前 standalone 轻量框架做实验。

## 实验 1：只改 analysis prompt

目标：固定 `scene` 输出，只比较不同 `analysis` prompt。

步骤：

1. 先跑并保存 scene：

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --stop-after scene \
  --save-results \
  --run-name scene_baseline
```

2. 修改 `prompts/analysis.py`。

3. 加载已有 scene，只跑 analysis：

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --load-run-dir outputs/scene_baseline \
  --load-stages scene \
  --start-from analysis \
  --stop-after analysis \
  --save-results \
  --run-name analysis_variant_001
```

4. 对比：

```text
outputs/analysis_variant_001/stages/analysis/user_prompt.txt
outputs/analysis_variant_001/stages/analysis/output.json
outputs/analysis_variant_001/stages/analysis/video_meta.json
```

## 实验 2：插入 object_state stage

目标 workflow：

```text
scene -> object_state -> analysis -> refinement
```

步骤：

1. 新增 `prompts/object_state.py`。
2. 在 `config.yaml` 的 `stages` 下新增 `object_state`。
3. 修改 workflow：

```yaml
workflow:
  - scene
  - object_state
  - analysis
  - refinement
```

4. 在下游 prompt 中引用：

```python
{{ ctx.stages.object_state.output }}
{{ ctx.stages.object_state.output.object_states }}
```

5. dry-run 检查：

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --dry-run \
  --save-results \
  --run-name object_state_dry_run
```

6. 检查 prompt 和 output 后再真实调用模型。

## 实验 3：只复跑 refinement

加载已有：

```text
scene output
analysis output
```

只运行：

```text
refinement
```

命令：

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --load-run-dir outputs/baseline_run \
  --load-stages scene analysis \
  --start-from refinement \
  --stop-after refinement \
  --save-results \
  --run-name refinement_variant_001
```

此时 `prompts/refinement.py` 中的这些路径可解析：

```python
{{ ctx.stages.scene.output }}
{{ ctx.stages.analysis.output.action_sequence }}
```

## 实验 4：修改视频处理参数

在 `config.yaml` 的 stage 下调整：

```yaml
video:
  max_time: 30
  fps: 1.0
  max_frames: 128
  resize_width: 336
  jpeg_quality: 75
  merge_views: true
  merge_mode: "per_frame"
  merge_length: 8
  draw_timestamps: true
  draw_view_names: true
```

参数意义：

| 参数 | 影响 |
|---|---|
| `max_time` | 当前 stage 最大原始视频处理时长，单位秒；`-1` 不限制，`>0` 只处理 `[0, max_time)`，在抽帧和后续视频处理前生效。 |
| `fps` | 抽帧密度，基于 primary view 原始时间轴。 |
| `max_frames` | temporal montage 前的最大采样时间点数。 |
| `resize_width` | 每个视角缩放宽度，影响细节和 payload。 |
| `jpeg_quality` | JPEG 质量，影响图像细节和传输体积。 |
| `merge_views` | true 时同一时间点多视角按行纵向拼接；false 时单视角保持原行为，多视角分别处理并全部发送。 |
| `merge_mode` | `per_frame` 逐时间点输出；`timeline_grid` 按时间从左到右合并；多视角非合并时对每个视角独立生效。 |
| `merge_length` | timeline_grid 分组长度；小于 1 表示全部时间点合成一张图。 |
| `draw_timestamps` | 是否绘制时间戳，refinement 通常建议开启。 |
| `draw_view_names` | 是否绘制视角名，多视角输入建议开启。 |
| `draw_montage_axes` | 多视角 timeline montage 外侧是否绘制顶部时间戳和左侧视角标签。 |
| `add_frame_tags` | image_sequence 下是否在每张图像前插入 `<t=...s> <view_name>` message 标签；不改变图像像素。 |

对比重点：

```text
outputs/<run>/stages/<stage>/video_meta.json
outputs/<run>/stages/<stage>/output.json
outputs/<run>/stages/<stage>/user_prompt.txt
```

`input_mode: "video"` 会把处理后帧编码为 MP4 `video_url`。使用前应确认当前模型服务支持视频 content part；框架侧编码成功不等于服务端一定接受该格式。

当 `merge_views=false` 且配置多个 `view_names` 时，对比实验应查看 `video_meta.view_outputs`：每个视角独立受 `fps`、`max_frames`、`max_time` 和帧范围限制，message 中每组媒体前会带视角名称。

`max_time` 与 `fps`、`max_frames`、`frame_start/frame_end` 同时配置时，采样只发生在最终有效区间内；输出帧时间戳仍然是原始视频时间轴上的真实时间戳。多视角实验中，较短视角不会被补齐，建议同时查看 `video_meta.json` 中的 `original_duration`、`effective_duration`、`per_view_effective_durations` 和 `was_time_limited`。

## 实验 5：dry_run 检查 prompt 注入

目标：不调用模型，检查 prompt 渲染、上游字段解析和公共片段插入。

命令：

```bash
python vlm_auto_annotation_refactor/examples/run_pipeline.py \
  --config vlm_auto_annotation_refactor/config.yaml \
  --video /path/to/video.mp4 \
  --instruction "pick up the cup" \
  --dry-run \
  --save-results \
  --run-name prompt_injection_check
```

检查：

```text
outputs/prompt_injection_check/stages/scene/user_prompt.txt
outputs/prompt_injection_check/stages/analysis/user_prompt.txt
outputs/prompt_injection_check/stages/refinement/user_prompt.txt
```

重点确认这些占位符已被替换：

```python
{{ ctx.input.instruction }}
{{ ctx.stages.scene.output }}
{{ ctx.stages.analysis.output.action_sequence }}
{{ prompt.common.JSON_ONLY_RULE }}
{{ prompt.common.ACTION_VOCABULARY }}
{{ prompt.common.TIME_BOUNDARY_RULE }}
```

注意：dry-run 仍会读取和处理视频，所以视频路径必须有效，且需要安装 `opencv-python` 和 `numpy`。

## Python 调用方式

```python
from vlm_auto_annotation_refactor.pipeline import run_pipeline
from vlm_auto_annotation_refactor.result_io import load_outputs_into_context

context = {
    "input": {
        "video_path": "/path/to/video.mp4",
        "instruction": "pick up the cup",
        "video_id": "demo_episode",
    },
    "stages": {},
}

load_outputs_into_context(context, "outputs/baseline_run", ["scene", "analysis"])
run_pipeline(
    context,
    config,
    start_from="refinement",
    stop_after="refinement",
    save_results=True,
    run_name="refinement_variant_001",
)
```

## 实验记录建议

1. 每个实验使用明确 `--run-name`。
2. prompt 实验重点保存和对比 `user_prompt.txt`。
3. 视频参数实验重点保存和对比 `video_meta.json`。
4. 固定上游复跑下游时记录加载了哪些 stage。
5. prompt 路径解析失败时，先检查上游 `output.json` 的真实字段结构。
