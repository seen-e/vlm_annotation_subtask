# json_utils.py

## 文件职责

`json_utils.py` 从 VLM 的原始文本响应中提取合法 JSON object 或 JSON array。

它只解决“文本转 JSON”问题，不负责业务字段校验。

## 异常类

### `JSONExtractError`

无法从模型文本中提取合法 JSON 时抛出。

## 主要函数

### `extract_json(raw_text: str) -> Any`

支持以下输入格式：

1. 纯 JSON object；
2. 纯 JSON array；
3. ```json fenced code block；
4. 普通 code block；
5. 前后夹杂解释文本但中间包含完整 JSON；
6. 带 `<think>...</think>` 的 thinking 输出。

#### 参数

| 参数 | 类型 | 说明 |
|---|---:|---|
| `raw_text` | str | 模型返回的原始文本 |

#### 返回值

- `dict` 或 `list`，由 JSON 内容决定。

## 内部函数

| 函数 | 作用 |
|---|---|
| `_strip_thinking(text)` | 去除 `<think>...</think>` 内容 |
| `_try_parse(text)` | 尝试 `json.loads`，失败返回 None |
| `_find_balanced_span(text, open_char, close_char)` | 扫描平衡括号，忽略字符串内部括号 |
| `_extract_fenced_blocks(text)` | 提取 Markdown code block 内容 |
| `_extract_balanced_json(text)` | 从文本中提取第一个完整 JSON object/array |

## 典型用法

```python
raw_text = call_vlm(...)
parsed = extract_json(raw_text)
```
