"""Tests for the AndroidWorld public task-metadata profile."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.profile_androidworld_metadata import _category, profile


def test_category_prefixes_are_stable() -> None:
    assert _category("SimpleSmsSend") == "SimpleSms"
    assert _category("TurnOnWifiAndOpenApp") == "TurnOnWifiAndOpenApp"
    assert _category("UnlistedTask") == "other"


def test_profile_is_metadata_only_and_validates_task_rows(tmp_path: Path) -> None:
    source = tmp_path / "task_metadata.json"
    source.write_text(
        json.dumps(
            [
                {
                    "task_name": "SimpleSmsSend",
                    "task_template": "Send {message} to a contact.",
                    "difficulty": "easy",
                    "tags": ["data_entry", "parameterized"],
                    "optimal_steps": "4",
                },
                {
                    "task_name": "SystemWifiTurnOn",
                    "task_template": "Turn on Wi-Fi.",
                    "difficulty": "medium",
                    "tags": ["verification"],
                    "optimal_steps": "2",
                },
            ]
        ),
        encoding="utf-8",
    )
    result = profile(source, revision="test-revision")
    assert result["coverage"]["task_templates"] == 2
    assert result["coverage"]["parameterized_tasks"] == 1
    assert result["coverage"]["optimal_steps"]["min"] == 2
    assert result["source"]["emulator_executed"] is False
    assert result["source"]["adb_invoked"] is False
    assert result["task_inventory"][0]["template_has_parameters"] is True
