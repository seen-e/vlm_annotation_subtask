# prompts/scene.py

## 文件职责

定义 scene 阶段的 system prompt 和 user prompt。scene 阶段用于提取稳定场景上下文，不做动作时间边界定位。

## 变量

| 变量 | 类型 | 说明 |
|---|---:|---|
| `SYSTEM_PROMPT` | str | scene 阶段 system prompt 模板 |
| `USER_PROMPT_TEMPLATE` | str | scene 阶段 user prompt 模板 |

## 占位符

当前使用：

```text
{{ prompt.common.JSON_ONLY_RULE }}
{{ ctx.input.instruction }}
```

## 预期输出

```json
{
  "scene_summary": "...",
  "executors": [],
  "touched_objects": [],
  "background_objects": [],
  "best_observation_views": []
}
```
