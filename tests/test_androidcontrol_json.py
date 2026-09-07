import json

import pytest

from openlocalagent.data.adapters.androidcontrol_json import canonical_action_from_conversation, json_row_to_conversation


def _row(action: dict[str, object]) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "mobile parser"},
            {"role": "user", "content": "<image>Tap the button"},
            {"role": "assistant", "content": json.dumps(action)},
        ],
        "images": ["and_ctrl/out_episode_1_step_0.png"],
    }


def test_json_row_preserves_action_and_marks_visual_omission() -> None:
    conversation = json_row_to_conversation(
        _row({"action_type": "click", "x": 11, "y": 22}),
        source_revision="mirror-sha",
        split="train",
        row_index=7,
    )
    assert canonical_action_from_conversation(conversation) == ("mobile_click", {"x": 11, "y": 22})
    assert conversation.meta["visual_input_omitted"] is True
    assert conversation.meta["grounding_evaluable"] is False
    assert conversation.meta["source_row_index"] == 7


def test_json_row_supports_test_only_navigate_home() -> None:
    conversation = json_row_to_conversation(
        _row({"action_type": "navigate_home"}),
        source_revision="mirror-sha",
        split="test",
        row_index=0,
    )
    assert canonical_action_from_conversation(conversation) == ("mobile_navigate_home", {})


def test_json_row_rejects_unknown_or_extra_arguments() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        json_row_to_conversation(
            _row({"action_type": "drag", "x": 1}),
            source_revision="mirror-sha",
            split="train",
            row_index=0,
        )
    with pytest.raises(ValueError, match="exactly"):
        json_row_to_conversation(
            _row({"action_type": "wait", "seconds": 1}),
            source_revision="mirror-sha",
            split="train",
            row_index=0,
        )
