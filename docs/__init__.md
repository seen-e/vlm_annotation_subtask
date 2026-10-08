# __init__.py

## 文件职责

`__init__.py` 将 `vlm_auto_annotation_refactor` 声明为 Python 包，并通过 `__all__` 暴露主要轻量模块。

## 导出模块

| 名称 | 说明 |
|---|---|
| `video_process` | 视频处理模块 |
| `model_client` | VLM 调用模块 |
| `json_utils` | JSON 提取模块 |
| `prompt_utils` | prompt 渲染模块 |
| `stage_runner` | 单 stage 执行模块 |
| `pipeline` | 多 stage 顺序执行模块 |
| `result_io` | 结果保存和加载模块 |

## 参数说明

本文件没有函数参数。修改时通常只需要在新增核心模块后更新 `__all__`。
