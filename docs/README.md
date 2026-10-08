# 轻量实验框架文档索引

本目录为 `vlm_auto_annotation_refactor/` 内的核心文件提供同名说明文档，重点解释每个文件的职责、内部函数、参数和典型用法。

## 配置与入口

- [config.md](config.md)：`config.yaml` 参数说明。
- [README.md](README.md)：当前文档索引。

## 核心模块

- [video_process.md](video_process.md)：视频/多视角视频转 VLM 输入。
- [model_client.md](model_client.md)：OpenAI-compatible VLM 请求。
- [json_utils.md](json_utils.md)：模型响应中的 JSON 提取。
- [prompt_utils.md](prompt_utils.md)：`{{ ctx.* }}` 与 `{{ prompt.* }}` 占位符渲染。
- [stage_runner.md](stage_runner.md)：单个 stage 执行。
- [pipeline.md](pipeline.md)：多个 stage 顺序执行。
- [result_io.md](result_io.md)：中间结果保存与加载。
- [__init__.md](__init__.md)：包初始化说明。

## Prompt 文件

- [prompts/common.md](prompts/common.md)：公共 prompt 片段和词表。
- [prompts/scene.md](prompts/scene.md)：scene 阶段 prompt。
- [prompts/analysis.md](prompts/analysis.md)：analysis 阶段 prompt。
- [prompts/refinement.md](prompts/refinement.md)：refinement 阶段 prompt。
- [prompts/__init__.md](prompts/__init__.md)：prompt 包说明。

## 示例文件

- [examples/run_stage.md](examples/run_stage.md)：单 stage 运行示例。
- [examples/run_pipeline.md](examples/run_pipeline.md)：完整 pipeline 运行示例。
