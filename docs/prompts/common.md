# prompts/common.py

## 文件职责

定义可被任意 prompt 复用的公共规则、动作词表和边界判断规则。

## 变量

| 变量 | 类型 | 作用 |
|---|---:|---|
| `JSON_ONLY_RULE` | str | 要求模型只输出 JSON |
| `VIDEO_LAYOUT_RULE` | str | 要求模型结合布局说明和 message 视角标签理解拼图、独立视角序列或视频 |
| `ACTION_VOCABULARY` | list[str] | 可选动作词表，可在 analysis prompt 中引用 |
| `TIME_BOUNDARY_RULE` | str | refinement 阶段时间边界判断规则 |

## 在 prompt 中引用

```text
{{ prompt.common.JSON_ONLY_RULE }}
{{ prompt.common.VIDEO_LAYOUT_RULE }}
{{ prompt.common.ACTION_VOCABULARY }}
{{ prompt.common.TIME_BOUNDARY_RULE }}
```

如果变量是 list 或 dict，渲染时会自动转为 pretty JSON 字符串。
