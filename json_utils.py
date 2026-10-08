"""JSON extraction helpers for VLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


class JSONExtractError(RuntimeError):
    """Raised when no valid JSON object/array can be extracted."""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _try_parse(text: str) -> Any | None:
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _find_balanced_span(text: str, open_char: str, close_char: str) -> tuple[int, int] | None:
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_char and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return start, i + 1
    return None


def _extract_fenced_blocks(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)]


def _extract_balanced_json(text: str) -> str | None:
    candidates: list[tuple[int, int]] = []
    for pair in (("{", "}"), ("[", "]")):
        span = _find_balanced_span(text, pair[0], pair[1])
        if span:
            candidates.append(span)
    if not candidates:
        return None
    start, end = sorted(candidates, key=lambda s: s[0])[0]
    return text[start:end]


def extract_json(raw_text: str) -> Any:
    """Extract a JSON object or array from model text.

    Supports pure JSON, fenced ```json blocks, generic code blocks, and text
    surrounding the first balanced JSON object/array. Returns a dict or list.
    """
    if raw_text is None:
        raise JSONExtractError("raw_text is None")
    text = _strip_thinking(str(raw_text).strip())
    if not text:
        raise JSONExtractError("raw_text is empty")

    parsed = _try_parse(text)
    if parsed is not None:
        return parsed

    for block in _extract_fenced_blocks(text):
        parsed = _try_parse(block)
        if parsed is not None:
            return parsed
        balanced = _extract_balanced_json(block)
        if balanced:
            parsed = _try_parse(balanced)
            if parsed is not None:
                return parsed

    balanced = _extract_balanced_json(text)
    if balanced:
        parsed = _try_parse(balanced)
        if parsed is not None:
            return parsed

    preview = text[:500].replace("\n", "\\n")
    raise JSONExtractError(f"Failed to extract valid JSON from response. preview={preview!r}")
