from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.train.function_masking import (
    FUNCTION_MASKING_KIND,
    augment_conversations,
)


def _conversation(index: int = 0) -> Conversation:
    return Conversation(
        tools=[
            ToolSpec(
                name="send_email",
                description="Send an email.",
                parameters={
                    "type": "object",
                    "properties": {"recipient": {"type": "string"}},
                    "required": ["recipient"],
                },
            ),
            ToolSpec(
                name="notion_write",
                description="Write a note.",
                parameters={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            ),
        ],
        messages=[
            Message(role=Role.user, content=f"Save report {index} to Notion"),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(
                        name="notion_write",
                        arguments={"content": f"report {index}"},
                    )
                ],
            ),
        ],
    )


def test_function_masking_is_deterministic_and_schema_preserving() -> None:
    rows, source_indices, audit = augment_conversations(
        [_conversation(0), _conversation(1)],
        {"mask_fraction": 1.0, "variants": 1, "name_prefix": "opaque"},
        seed=17,
    )
    repeat, repeat_indices, repeat_audit = augment_conversations(
        [_conversation(0), _conversation(1)],
        {"mask_fraction": 1.0, "variants": 1, "name_prefix": "opaque"},
        seed=17,
    )
    assert source_indices == [0, 0, 1, 1]
    assert repeat_indices == source_indices
    assert [row.to_json() for row in rows] == [row.to_json() for row in repeat]
    assert audit == repeat_audit
    assert audit["kind"] == FUNCTION_MASKING_KIND
    assert audit["masked_rows"] == 2
    assert audit["output_rows"] == 4

    original, masked = rows[0], rows[1]
    assert [tool.name for tool in original.tools] == ["send_email", "notion_write"]
    masked_names = [tool.name for tool in masked.tools]
    assert all(name.startswith("opaque_") for name in masked_names)
    assert masked_names != [tool.name for tool in original.tools]
    assert masked.tools[1].parameters == original.tools[1].parameters
    assert masked.messages[1].tool_calls[0].name == masked_names[1]
    assert masked.messages[1].tool_calls[0].arguments == {"content": "report 0"}


def test_function_masking_keeps_unselected_rows_and_supports_multiple_variants() -> None:
    rows, source_indices, audit = augment_conversations(
        [_conversation()],
        {"mask_fraction": 0.0, "variants": 4},
        seed=1,
    )
    assert rows == [_conversation()]
    assert source_indices == [0]
    assert audit["enabled"] is True
    assert audit["masked_rows"] == 0
    assert audit["output_rows"] == 1

    rows, source_indices, audit = augment_conversations(
        [_conversation()],
        {"mask_fraction": 1.0, "variants": 2},
        seed=1,
    )
    assert len(rows) == 3
    assert source_indices == [0, 0, 0]
    assert audit["variants"] == 2
    assert rows[1].tools[0].name != rows[2].tools[0].name


def test_function_masking_rejects_dangling_calls_and_unknown_config() -> None:
    broken = _conversation()
    broken.messages[1].tool_calls[0].name = "missing"
    try:
        augment_conversations([broken], {"mask_fraction": 1.0}, seed=0)
    except ValueError as exc:
        assert "without a catalog entry" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("dangling tool call was accepted")

    try:
        augment_conversations([_conversation()], {"typo": True}, seed=0)
    except ValueError as exc:
        assert "unknown keys" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown function-masking config key was accepted")
