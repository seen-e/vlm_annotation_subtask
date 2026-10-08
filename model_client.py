"""Small OpenAI-compatible VLM client wrapper."""

from __future__ import annotations

import time
from typing import Any


class VLMClientError(RuntimeError):
    """Raised when VLM request construction or execution fails."""

    def __init__(self, message: str, *, debug_info: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.debug_info = debug_info or {}


def _is_qwen_model(model: str) -> bool:
    m = (model or "").lower()
    return "qwen" in m or "qvq" in m


def _build_messages(system_prompt: str, user_prompt: str, image_parts: list[dict[str, Any]] | None, *, model: str) -> list[dict[str, Any]]:
    parts = list(image_parts or [])
    if _is_qwen_model(model):
        text = f"{system_prompt.strip()}\n\n{user_prompt.strip()}" if system_prompt else user_prompt.strip()
        return [{"role": "user", "content": [{"type": "text", "text": text}, *parts]}]
    return [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}, *parts]},
    ]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)
    return ""


def _usage_dict(usage_obj: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
    }


def _message_debug(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", None)
    debug = {
        "role": getattr(message, "role", None),
        "content_type": type(content).__name__,
        "content_preview": repr(content)[:1000],
    }
    for name in ("reasoning_content", "tool_calls", "function_call", "refusal"):
        value = getattr(message, name, None)
        if value is not None:
            debug[name] = repr(value)[:1000]
    return debug


def call_vlm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_parts: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    timeout: float | None = None,
    extra_body: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> str:
    """Call an OpenAI-compatible VLM endpoint and return response text."""
    result = call_vlm_with_metadata(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_parts=image_parts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        extra_body=extra_body,
        max_retries=max_retries,
    )
    return str(result["raw_text"])


def call_vlm_with_metadata(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_parts: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    timeout: float | None = None,
    extra_body: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    if not base_url:
        raise VLMClientError("base_url is required")
    if not model:
        raise VLMClientError("model is required")
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - dependency/environment dependent
        raise VLMClientError("openai package is required for real VLM calls") from exc

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout)
    messages = _build_messages(system_prompt, user_prompt, image_parts, model=model)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        request_kwargs["top_p"] = top_p
    body = dict(extra_body or {})
    if top_k is not None and top_k > 0:
        body["top_k"] = top_k
    if body:
        request_kwargs["extra_body"] = body

    last_error: Exception | None = None
    for attempt in range(max(max_retries, 1)):
        try:
            response = client.chat.completions.create(**request_kwargs)
            if not getattr(response, "choices", None):
                raise VLMClientError("response has no choices")
            choice = response.choices[0]
            message = choice.message
            content = _content_to_text(getattr(message, "content", None))
            usage = _usage_dict(getattr(response, "usage", None))
            if not content.strip():
                raise VLMClientError(
                    "response message content is empty",
                    debug_info={
                        "model": model,
                        "base_url": base_url,
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "usage": usage,
                        "message": _message_debug(message),
                    },
                )
            return {
                "raw_text": content.strip(),
                "model": model,
                "usage": usage,
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            if attempt + 1 < max(max_retries, 1):
                time.sleep(min(2 ** attempt, 8))
    debug_info = getattr(last_error, "debug_info", {}) if last_error is not None else {}
    raise VLMClientError(f"VLM call failed model={model!r} base_url={base_url!r}: {last_error}", debug_info=debug_info)
