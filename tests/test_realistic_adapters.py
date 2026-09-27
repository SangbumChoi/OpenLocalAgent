import pytest

from localagent.data.realistic_adapters import normalize_mobile_row


def _row() -> dict:
    return {
        "record_id": "episode-1",
        "goal": "Open the mail app and scroll down.",
        "steps": [
            {
                "instruction": "Open mail.",
                "accessibility_tree": "[node text='Mail']",
                "action": {"action_type": "open_app", "app_name": "Mail"},
                "next_observation": "Mail list",
            },
            {
                "instruction": "Scroll down.",
                "accessibility_tree": "[list]",
                "action": {"action_type": "scroll", "direction": "down"},
            },
        ],
    }


def test_normalize_mobile_row_produces_localagent_v1_shape() -> None:
    normalized = normalize_mobile_row(
        _row(), family="androidcontrol", source_revision="abc123"
    )
    assert normalized["record_id"] == "episode-1"
    assert normalized["quality"]["text_first"] is True
    assert [message["role"] for message in normalized["messages"]] == [
        "user",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert {tool["name"] for tool in normalized["tools"]} == {
        "mobile_open_app",
        "mobile_scroll",
    }


def test_screenshot_only_rows_fail_closed() -> None:
    row = _row()
    row["steps"][0].pop("accessibility_tree")
    row["steps"][0]["screenshot"] = "base64-bytes"
    with pytest.raises(ValueError, match="screenshot-only"):
        normalize_mobile_row(row, family="android_in_the_wild", source_revision="rev")


def test_unknown_action_and_wrong_arguments_fail_closed() -> None:
    row = _row()
    row["steps"][0]["action"] = {"action_type": "delete_everything"}
    with pytest.raises(ValueError, match="unsupported action_type"):
        normalize_mobile_row(row, family="androidcontrol", source_revision="rev")

    row = _row()
    row["steps"][0]["action"] = {"action_type": "scroll", "direction": "down", "x": 1}
    with pytest.raises(ValueError, match="arguments"):
        normalize_mobile_row(row, family="androidcontrol", source_revision="rev")


def test_androidcontrol_wait_accepts_the_official_empty_argument_object() -> None:
    row = _row()
    row["steps"] = [
        {
            "instruction": "Wait for the app to settle.",
            "accessibility_tree": "[loading]",
            "action": {"action_type": "wait"},
        }
    ]
    normalized = normalize_mobile_row(row, family="androidcontrol", source_revision="rev")
    wait_tool = next(tool for tool in normalized["tools"] if tool["name"] == "mobile_wait")
    assert wait_tool["parameters"]["required"] == []
    assert normalized["messages"][2]["tool_calls"] == [
        {"name": "mobile_wait", "arguments": {}}
    ]
