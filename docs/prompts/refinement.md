# prompts/refinement.py

## 文件职责

定义 refinement 阶段 prompt。refinement 阶段根据带时间戳视频和 analysis 输出精修动作边界。

## 变量

| 变量 | 类型 | 说明 |
|---|---:|---|
| `SYSTEM_PROMPT` | str | refinement 阶段 system prompt 模板 |
| `USER_PROMPT_TEMPLATE` | str | refinement 阶段 user prompt 模板 |

## 占位符

当前使用：

```text
{{ prompt.common.JSON_ONLY_RULE }}
{{ ctx.input.instruction }}
{{ ctx.stages.scene.output }}
{{ ctx.stages.analysis.output.action_sequence }}
{{ prompt.common.TIME_BOUNDARY_RULE }}
```

## 预期输出

```json
{
  "refined_segments": [
    {
      "step_id": 1,
      "executor": "...",
      "action": "...",
      "object": "...",
      "start_time": 0.0,
      "end_time": 1.0,
      "evidence": "...",
      "confidence": 0.0
    }
  ]
}
```
