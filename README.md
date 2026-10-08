# vlm_auto_annotation_refactor

这是一个轻量级 VLM 多阶段实验框架。该目录是 standalone 重构版，只包含新框架代码，不依赖原始项目中的 `flows/`、`utils/`、`annotation_pipeline/` 等旧模块。

## 依赖

运行真实视频处理和模型调用通常需要：

```bash
pip install opencv-python numpy pyyaml openai
```

其中 `openai` 只在真实调用 OpenAI-compatible VLM 时需要；`dry-run` 不需要真实模型服务。

核心文件：

- `video_process.py`：视频/多视角视频 -> VLM media parts。
- `model_client.py`：OpenAI-compatible VLM 调用。
- `json_utils.py`：从模型响应中提取 JSON。
- `prompt_utils.py`：渲染 `{{ ctx.* }}` 和 `{{ prompt.* }}` 占位符。
- `stage_runner.py`：执行单个配置化 stage。
- `pipeline.py`：按 `config.yaml` 中的 `workflow` 顺序执行多个 stage。
- `result_io.py`：保存和加载每个 stage 的中间结果。

## Prompt 占位符

在 `prompts/*.py` 中可以直接写：

```text
{{ ctx.input.instruction }}
{{ ctx.stages.scene.output.executors }}
{{ ctx.stages.analysis.output.action_sequence.0.action }}
{{ prompt.common.JSON_ONLY_RULE }}
```

- `ctx.*` 从运行时 `context` 取值。
- `prompt.*` 从 `vlm_auto_annotation_refactor/prompts/*.py` 取 prompt 片段或列表。

## 最小运行

```bash
python -m compileall vlm_auto_annotation_refactor
python vlm_auto_annotation_refactor/examples/run_pipeline.py --config vlm_auto_annotation_refactor/config.yaml --video /path/to/video.mp4 --instruction "pick up the cup" --dry-run
```

## 固定上游结果复跑

```python
from vlm_auto_annotation_refactor.result_io import load_outputs_into_context
from vlm_auto_annotation_refactor.pipeline import run_pipeline

load_outputs_into_context(context, "outputs/demo", ["scene"])
run_pipeline(context, config, start_from="analysis", skip_existing=True)
```

## 文档

详细参数和函数说明见 `docs/`：

- `docs/config.md`：`config.yaml` 的参数含义、可填写内容和自定义 stage 方法。
- `docs/video_process.md`：视频处理接口说明。
- `docs/stage_runner.md`：单 stage 执行接口说明。
- `docs/pipeline.md`：workflow 串联说明。
- `docs/result_io.md`：中间结果保存与加载说明。
