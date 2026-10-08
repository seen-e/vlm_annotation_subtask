# prompt_utils.py

## 文件职责

`prompt_utils.py` 负责加载 Python prompt 模块，并渲染 prompt 占位符。

核心能力：

```text
{{ ctx.input.instruction }}
{{ ctx.stages.scene.output.executors }}
{{ prompt.common.JSON_ONLY_RULE }}
{{ prompt.common.ACTION_VOCABULARY }}
```

## 异常类

### `PromptRenderError`

prompt 模块加载失败、变量缺失、占位符无法解析时抛出。

## 主要函数

### `resolve_context_path(context, path)`

从运行时 `context` 中解析路径。

| 参数 | 类型 | 说明 |
|---|---:|---|
| `context` | dict | pipeline 运行上下文 |
| `path` | str | 点分路径，例如 `stages.scene.output.executors` |

支持 list index：

```text
stages.analysis.output.action_sequence.0.action
```

### `resolve_prompt_path(path)`

从 `vlm_auto_annotation_refactor/prompts/*.py` 中读取变量。

| 参数 | 类型 | 说明 |
|---|---:|---|
| `path` | str | 例如 `common.JSON_ONLY_RULE` |

### `render_template(template, *, context, extra_vars=None)`

渲染 `{{ ... }}` 占位符。

| 参数 | 类型 | 说明 |
|---|---:|---|
| `template` | str | prompt 模板字符串 |
| `context` | dict | 运行上下文 |
| `extra_vars` | dict/null | 兼容旧 `input_fields` 的额外变量 |

占位符规则：

| 格式 | 来源 |
|---|---|
| `{{ ctx.xxx }}` | 从 `context` 中取值 |
| `{{ prompt.xxx }}` | 从 prompt Python 模块中取值 |
| `{{ old_var }}` | 从 `extra_vars` 中取值 |

### `load_stage_prompt(stage_cfg, *, stage_name=None)`

从 stage 配置加载 system/user prompt 模板。

支持新格式：

```yaml
prompt:
  module: "vlm_auto_annotation_refactor.prompts.analysis"
  system: "SYSTEM_PROMPT"
  user: "USER_PROMPT_TEMPLATE"
```

也兼容旧的 `prompt_file` 方式。

### `resolve_input_fields(context, input_fields)`

兼容旧 `input_fields` 配置，将字段路径解析成 `extra_vars`。

## 内部函数

| 函数 | 作用 |
|---|---|
| `_to_text(value)` | 将 dict/list/None 等转换成 prompt 文本 |
| `_import_prompt_module(module_name)` | 动态导入 prompt 模块 |
