# model_client.py

## 文件职责

`model_client.py` 封装 OpenAI-compatible VLM 调用。它只负责把 prompt 和 media parts 发给模型，并返回模型文本响应。

它不负责视频处理、prompt 渲染、JSON 提取或 schema 校验。

## 异常类

### `VLMClientError`

模型请求构造失败、请求失败或响应格式异常时抛出。

## 主要函数

### `call_vlm(...) -> str`

调用模型并只返回文本内容。

#### 参数

| 参数 | 类型 | 说明 |
|---|---:|---|
| `base_url` | str | OpenAI-compatible API 地址 |
| `api_key` | str | API key，本地 vLLM 常用 `EMPTY` |
| `model` | str | 模型 ID |
| `system_prompt` | str | system prompt |
| `user_prompt` | str | user prompt |
| `image_parts` | list[dict]/null | `video_process.build_video_inputs` 返回的 media parts；名称为兼容旧 API，内容可包含视角 text、`image_url` 和 `video_url` parts |
| `max_tokens` | int | 最大输出 token |
| `temperature` | float | 采样温度 |
| `top_p` | float/null | 可选采样参数 |
| `top_k` | int/null | 可选采样参数，通过 `extra_body` 传递 |
| `timeout` | float/null | 请求超时 |
| `extra_body` | dict/null | 传给 OpenAI client 的额外参数 |
| `max_retries` | int | 失败重试次数 |

### `call_vlm_with_metadata(...) -> dict`

调用模型并返回更多信息：

```python
{
  "raw_text": "...",
  "model": "...",
  "usage": {...},
  "finish_reason": "..."
}
```

参数同 `call_vlm`。

## 内部函数

| 函数 | 作用 |
|---|---|
| `_is_qwen_model(model)` | 判断是否为 Qwen/QVQ 系模型 |
| `_build_messages(...)` | 构造 OpenAI-compatible chat messages |

## 说明

对 Qwen 系模型，当前实现会把 `system_prompt` 和 `user_prompt` 合并到 user message 中，并把图像 parts 放在文本前面，以适配部分 Qwen-VL 服务端的多模态输入习惯。
