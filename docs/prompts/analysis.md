# prompts/analysis.py

## 文件职责

定义 analysis 阶段 prompt。analysis 阶段根据视频和 scene 输出生成动作序列。

## 变量

| 变量 | 类型 | 说明 |
|---|---:|---|
| `SYSTEM_PROMPT` | str | analysis 阶段 system prompt 模板 |
| `USER_PROMPT_TEMPLATE` | str | analysis 阶段 user prompt 模板 |

## 占位符

当前使用：

```text
{{ prompt.common.JSON_ONLY_RULE }}
{{ ctx.input.instruction }}
{{ ctx.stages.scene.output }}
{{ prompt.common.ACTION_VOCABULARY }}
```

你可以按实验需要改为更细粒度字段，例如：

```text
{{ ctx.stages.scene.output.executors }}
{{ ctx.stages.scene.output.touched_objects }}
```

## 预期输出

```json
{
  "action_sequence": [
    {
      "step_id": 1,
      "executor": "...",
      "action": "...",
      "object": "...",
      "evidence": "...",
      "confidence": 0.0
    }
  ]
}
```
