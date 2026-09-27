"""Tests for the leakage-safe Computer Agent Arena text probe."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_computer_agent_arena_text import (
    canonical_tool,
    load_cases,
    prompt_for_instruction,
)


def test_canonical_tool_projection_is_explicit() -> None:
    assert canonical_tool("left_click") == "click"
    assert canonical_tool("doubleClick") == "double_click"
    assert canonical_tool("typewrite") == "type_text"
    assert canonical_tool("hotkey") == "key_press"
    assert canonical_tool("screenshot") == "screenshot"
    assert canonical_tool("not_supported") is None


def test_load_cases_excludes_thoughts_and_images(tmp_path: Path) -> None:
    source = tmp_path / "arena.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "instruction": "Open the browser",
                "human_eval_correctness": 1,
                "traj": [
                    {
                        "image": "secret.png",
                        "value": {
                            "thought": "Do not expose this reasoning",
                            "code": "pyautogui.click(1, 2)",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_cases(source, limit=4, seed=2027)
    assert cases == [
        {
            "task_id": "task-1",
            "instruction": "Open the browser",
            "gold_tool": "click",
            "gold_family": "pointer",
            "model": "<missing>",
            "human_eval_correctness": 1,
        }
    ]
    prompt = prompt_for_instruction(cases[0]["instruction"])
    assert "Open the browser" in prompt
    assert "secret.png" not in prompt
    assert "reasoning" not in prompt
