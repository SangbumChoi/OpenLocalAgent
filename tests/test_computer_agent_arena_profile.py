"""Tests for the bounded Computer Agent Arena metadata audit."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.profile_computer_agent_arena import action_family, profile


def test_action_family_is_conservative() -> None:
    assert action_family("click") == "pointer"
    assert action_family("left_click") == "pointer"
    assert action_family("double_click") == "pointer"
    assert action_family("typewrite") == "type"
    assert action_family("type") == "type"
    assert action_family("hotkey") == "keyboard"
    assert action_family("unknown") == "unknown"


def test_profile_counts_rows_without_executing_code(tmp_path: Path) -> None:
    source = tmp_path / "arena.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "a",
                        "model": "demo",
                        "human_eval_correctness": 1,
                        "traj": [
                            {
                                "image": "images/a_1.png",
                                "value": {
                                    "thought": "Click the button.",
                                    "code": "pyautogui.click(10, 20)",
                                },
                            },
                            {
                                "value": {
                                    "thought": "Type text.",
                                    "code": "pyautogui.typewrite('safe')",
                                }
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "task_id": "b",
                        "model": "demo2",
                        "human_eval_correctness": 0,
                        "traj": [
                            {"value": {"observation": "No screenshot", "code": "computer.wait(1)"}}
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = profile(source, revision="test-revision")
    assert result["coverage"]["trajectories"] == 2
    assert result["coverage"]["steps"] == 3
    assert result["coverage"]["image_reference_steps"] == 1
    assert result["coverage"]["thought_steps"] == 2
    assert result["coverage"]["observation_steps"] == 1
    assert result["coverage"]["parseable_action_steps"] == 3
    assert result["action_families"] == {"pointer": 1, "type": 1, "wait": 1}
    assert result["human_eval_correct_rate"] == 0.5
    assert result["source"]["image_archives_downloaded"] is False
