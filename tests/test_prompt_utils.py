from __future__ import annotations

import importlib
import sys

from prompt_utils import (
    _prompt_package_from_module,
    render_template,
    resolve_executor_ids,
    resolve_robot_type_prompt,
)
from prompts.prompts_3_stages.scene import SCENE_USER_PROMPT


def test_prompt_placeholder_prefers_current_prompt_package(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "prompts_custom"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "actionbase.py").write_text('ACTION_VOCABULARY = "custom actions"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("prompts_custom.actionbase", None)

    rendered = render_template(
        "{{ prompt.actionbase.ACTION_VOCABULARY }}",
        context={},
        prompt_package="prompts_custom",
    )

    assert rendered == "custom actions"


def test_prompt_package_is_derived_from_stage_prompt_module() -> None:
    assert _prompt_package_from_module("prompts_4stage.gripper_prompt") == "prompts_4stage"
    assert _prompt_package_from_module("prompts.scene_dual") == "prompts"
    assert _prompt_package_from_module("prompts.prompts_3_stages.scene") == "prompts.prompts_3_stages"


def test_resolve_robot_type_prompt_uses_configured_type() -> None:
    prompt_package = "prompts.prompts_3_stages"
    single_prompt = resolve_robot_type_prompt("single_arm", prompt_package=prompt_package)
    bimanual_prompt = resolve_robot_type_prompt("bimanual", prompt_package=prompt_package)

    assert 'executor 固定使用 "arm_1"' in single_prompt
    assert 'executor 集合固定为 "arm_1" 和 "arm_2"' in bimanual_prompt


def test_scene_prompt_declares_single_and_bimanual_executor_sets() -> None:
    assert "single_arm 对应 executors=[arm_1]" in SCENE_USER_PROMPT
    assert "bimanual 对应 executors=[arm_1, arm_2]" in SCENE_USER_PROMPT
    assert "single_arm 固定只输出 arm_1" in SCENE_USER_PROMPT


def test_executor_ids_follow_robot_type_and_reject_conflicts() -> None:
    assert resolve_executor_ids({"robot_type": "single_arm"}) == ["arm_1"]
    assert resolve_executor_ids({"robot_type": "bimanual"}) == ["arm_1", "arm_2"]

    try:
        resolve_executor_ids(
            {
                "robot_type": "single_arm",
                "segment_loop": {"executors": ["arm_1", "arm_2"]},
            }
        )
    except ValueError as exc:
        assert "requires segment_loop.executors=['arm_1']" in str(exc)
    else:
        raise AssertionError("single_arm executor mismatch must fail")
