# pipeline.py

## 文件职责

`pipeline.py` 按 `workflow` 顺序执行多个 stage。

它不关心具体 stage 是 scene、analysis 还是 refinement，只根据 `config["workflow"]` 中的 stage 名称列表执行。

## 异常类

### `PipelineError`

workflow 配置错误、范围裁剪错误或某个 stage 执行失败时抛出。

## 主要函数

### `run_pipeline(context, config, *, workflow=None, dry_run=False, start_from=None, stop_after=None, skip_existing=False, output_dir=None, run_name=None, save_results=False) -> dict`

执行一条顺序 pipeline。

#### 参数

| 参数 | 类型 | 说明 |
|---|---:|---|
| `context` | dict | 运行上下文 |
| `config` | dict | 读取后的配置 |
| `workflow` | list[str]/null | 可选覆盖 `config["workflow"]` |
| `dry_run` | bool | 传给 `run_stage` |
| `start_from` | str/null | 从某个 stage 开始 |
| `stop_after` | str/null | 执行到某个 stage 后停止 |
| `skip_existing` | bool | context 中已有同名 stage 时跳过 |
| `output_dir` | str/Path/null | 保存结果的根目录 |
| `run_name` | str/null | 当前运行名称 |
| `save_results` | bool | 是否保存 stage 结果和完整 context |

#### 返回值

更新后的 `context`。

## 内部函数

| 函数 | 作用 |
|---|---|
| `_get_workflow(config, workflow)` | 获取并校验 workflow 列表 |
| `_slice_workflow(stages, start_from, stop_after)` | 根据 start/stop 裁剪执行范围 |

## 使用场景

### 完整执行

```python
run_pipeline(context, config)
```

### 只执行 analysis 之后

```python
run_pipeline(context, config, start_from="analysis")
```

### 固定上游结果，只跑下游

```python
load_outputs_into_context(context, run_dir, ["scene"])
run_pipeline(context, config, start_from="analysis", skip_existing=True)
```
