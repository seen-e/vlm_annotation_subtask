# result_io.py

## 文件职责

`result_io.py` 提供轻量级中间结果保存与加载能力，便于固定上游结果、复跑下游 stage。

## 输出目录结构

```text
outputs/<run_name>/
  input.json
  context.json
  stages/<stage_name>/
    system_prompt.txt
    user_prompt.txt
    raw_text.txt
    output.json
    video_meta.json
    usage.json
```

## 异常类

### `ResultIOError`

保存或加载失败时抛出。

## 主要函数

| 函数 | 参数 | 作用 |
|---|---|---|
| `ensure_dir(path)` | `path` | 创建目录并返回 Path |
| `save_json(data, path)` | JSON 数据、路径 | 保存 JSON |
| `load_json(path)` | 路径 | 读取 JSON |
| `save_text(text, path)` | 文本、路径 | 保存文本 |
| `load_text(path)` | 路径 | 读取文本 |
| `make_run_dir(output_dir="outputs", run_name=None, video_id=None)` | 输出根目录、运行名、视频 id | 创建运行目录 |
| `save_stage_result(context, stage_name, run_dir)` | context、stage 名、运行目录 | 保存单个 stage 结果 |
| `save_context(context, run_dir)` | context、运行目录 | 保存 `input.json` 和 `context.json` |
| `load_stage_output(stage_name, run_dir)` | stage 名、运行目录 | 读取某个 stage 的 `output.json` |
| `load_stage_into_context(context, stage_name, run_dir)` | context、stage 名、运行目录 | 将已有 stage 输出加载进 context |
| `load_outputs_into_context(context, run_dir, stage_names)` | context、运行目录、stage 列表 | 批量加载多个 stage 输出 |

## 典型用法

```python
load_outputs_into_context(context, "outputs/demo", ["scene", "analysis"])
run_pipeline(context, config, start_from="refinement", skip_existing=True)
```
