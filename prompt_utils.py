"""Prompt module loading and placeholder rendering utilities."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any


class PromptRenderError(RuntimeError):
    """Raised when prompt loading or rendering fails."""


_PLACEHOLDER_RE = re.compile(r"{{\s*([^{}]+?)\s*}}")


def _to_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


def resolve_context_path(context: dict[str, Any], path: str) -> Any:
    """Resolve a simple dotted path against context.

    Supports dict keys and list indices, e.g.
    ``stages.analysis.output.action_sequence.0.action``.
    """
    if not path:
        raise PromptRenderError("context path is empty")
    current: Any = context
    consumed: list[str] = []
    for part in path.split("."):
        consumed.append(part)
        if isinstance(current, dict):
            if part not in current:
                raise PromptRenderError(f"Context path not found: {path!r}; missing {'.'.join(consumed)!r}")
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(part)
            except ValueError as exc:
                raise PromptRenderError(f"Expected list index in context path {path!r}, got {part!r}") from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise PromptRenderError(f"List index out of range in context path {path!r}: {index}") from exc
        else:
            if not hasattr(current, part):
                raise PromptRenderError(f"Cannot resolve {part!r} in context path {path!r} from {type(current).__name__}")
            current = getattr(current, part)
    return current


def _import_prompt_module(module_name: str):
    module_name = module_name.strip()
    candidates: list[str] = []

    def add_candidate(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    package_name = (__package__ or "").split(".", 1)[0]
    if package_name and not module_name.startswith(f"{package_name}."):
        add_candidate(f"{package_name}.{module_name}")
    add_candidate(module_name)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:
            last_error = exc
    raise PromptRenderError(f"Failed to import prompt module {module_name!r}: {last_error}")


def _prompt_package_from_module(module_name: str | None) -> str | None:
    if not module_name:
        return None
    text = str(module_name).strip()
    if "." not in text:
        return None
    return text.rsplit(".", 1)[0] or None


def resolve_prompt_path(path: str, *, prompt_package: str | None = None) -> Any:
    """Resolve a prompt variable path like ``common.JSON_ONLY_RULE``."""
    if "." not in path:
        raise PromptRenderError(f"Prompt path must be '<module>.<variable>', got {path!r}")
    module_name, attr_path = path.split(".", 1)
    candidates: list[str] = []
    if prompt_package:
        candidates.append(f"{prompt_package}.{module_name}")
    candidates.append(f"prompts.{module_name}")

    module = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            module = _import_prompt_module(candidate)
            break
        except Exception as exc:
            last_error = exc
    if module is None:
        raise PromptRenderError(f"Failed to import prompt path {path!r}: {last_error}")

    current: Any = module
    consumed = [module_name]
    for part in attr_path.split("."):
        consumed.append(part)
        if isinstance(current, dict):
            if part not in current:
                raise PromptRenderError(f"Prompt path not found: {path!r}; missing {'.'.join(consumed)!r}")
            current = current[part]
        elif isinstance(current, (list, tuple)):
            try:
                current = current[int(part)]
            except Exception as exc:
                raise PromptRenderError(f"Invalid list index in prompt path {path!r}: {part!r}") from exc
        else:
            if not hasattr(current, part):
                raise PromptRenderError(f"Prompt variable not found: {path!r}; missing {'.'.join(consumed)!r}")
            current = getattr(current, part)
    return current


def resolve_robot_type_prompt(robot_type: str, *, prompt_package: str | None = None) -> str:
    """Resolve the configured robot type through the active prompt package."""
    resolver = resolve_prompt_path("robot_type.get_robot_type_prompt", prompt_package=prompt_package)
    if not callable(resolver):
        raise PromptRenderError("prompt.robot_type.get_robot_type_prompt must be callable")
    return _to_text(resolver(str(robot_type)))


def resolve_executor_ids(config: dict[str, Any]) -> list[str]:
    """Resolve the executor set shared by prompts and the episode loop."""
    robot_cfg = config.get("robot") or {}
    robot_type = (
        str(robot_cfg.get("type"))
        if isinstance(robot_cfg, dict) and robot_cfg.get("type")
        else str(config.get("robot_type") or "bimanual")
    )
    expected_by_robot_type = {
        "single_arm": ["arm_1"],
        "bimanual": ["arm_1", "arm_2"],
    }
    expected = expected_by_robot_type.get(robot_type)
    configured = (config.get("segment_loop") or {}).get("executors")
    if configured is None:
        return list(expected or ["arm_1", "arm_2"])
    if (
        not isinstance(configured, list)
        or not configured
        or not all(isinstance(value, str) and value for value in configured)
        or len(set(configured)) != len(configured)
    ):
        raise ValueError("segment_loop.executors must be a non-empty list of unique executor ids")
    executors = list(configured)
    if expected is not None and executors != expected:
        raise ValueError(
            f"robot_type={robot_type!r} requires segment_loop.executors={expected}, got {executors}"
        )
    return executors


def render_template(
    template: str,
    *,
    context: dict[str, Any],
    extra_vars: dict[str, Any] | None = None,
    prompt_package: str | None = None,
) -> str:
    """Render ``{{ ... }}`` placeholders.

    Supported forms:
    - ``{{ ctx.input.instruction }}`` from pipeline context
    - ``{{ prompt.common.JSON_ONLY_RULE }}`` from the current prompt package, falling back to ``prompts/common.py``
    - ``{{ old_var }}`` from ``extra_vars`` for backward compatibility
    """
    extra_vars = extra_vars or {}

    def replace(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if expr.startswith("ctx."):
            return _to_text(resolve_context_path(context, expr[len("ctx.") :]))
        if expr.startswith("prompt."):
            return _to_text(resolve_prompt_path(expr[len("prompt.") :], prompt_package=prompt_package))
        if expr in extra_vars:
            return _to_text(extra_vars[expr])
        raise PromptRenderError(f"Unresolved prompt placeholder: {{{{ {expr} }}}}")

    return _PLACEHOLDER_RE.sub(replace, template)


def load_stage_prompt(stage_cfg: dict[str, Any], *, stage_name: str | None = None) -> tuple[str, str]:
    """Load system and user prompt templates from a prompt module or legacy txt file."""
    prompt_cfg = stage_cfg.get("prompt") or {}
    label = f" for stage {stage_name}" if stage_name else ""
    if isinstance(prompt_cfg, dict) and prompt_cfg.get("module"):
        module = _import_prompt_module(str(prompt_cfg["module"]))
        system_attr = str(prompt_cfg.get("system", "SYSTEM_PROMPT"))
        user_attr = str(prompt_cfg.get("user", "USER_PROMPT_TEMPLATE"))
        try:
            system_template = getattr(module, system_attr)
            user_template = getattr(module, user_attr)
        except AttributeError as exc:
            raise PromptRenderError(
                f"Prompt module {prompt_cfg['module']!r}{label} missing {system_attr!r} or {user_attr!r}"
            ) from exc
        if not isinstance(system_template, str) or not isinstance(user_template, str):
            raise PromptRenderError(f"Prompt variables must be strings in module {prompt_cfg['module']!r}{label}")
        return system_template, user_template

    # Backward-compatible txt prompt support.
    prompt_file = stage_cfg.get("prompt_file")
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            # Resolve relative to the refactor package root.
            path = Path(__file__).resolve().parent / prompt_file
        if not path.exists():
            raise PromptRenderError(f"prompt_file not found{label}: {prompt_file}")
        return str(stage_cfg.get("system_prompt", "")), path.read_text(encoding="utf-8")

    raise PromptRenderError(f"No prompt.module or prompt_file configured{label}")


def resolve_input_fields(context: dict[str, Any], input_fields: dict[str, str] | None) -> dict[str, Any]:
    """Resolve legacy stage input_fields into extra prompt variables."""
    resolved: dict[str, Any] = {}
    for name, path in (input_fields or {}).items():
        resolved[name] = resolve_context_path(context, str(path))
    return resolved
