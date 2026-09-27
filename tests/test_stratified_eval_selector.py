from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from openlocalagent.data.conversation_artifact import (
    canonical_json_bytes,
    conversation_semantic_sha256,
)
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.data.decontam.stratified_eval_selector import (
    InsufficientStratumCapacityError,
    conversation_eval_strata,
    select_stratified_eval_subset,
)


def _tool(
    name: str,
    *,
    properties: dict[str, dict[str, object]] | None = None,
    required: list[str] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Call {name}.",
        parameters={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    )


def _single(
    category: str,
    prompt: str,
    *,
    calls: list[ToolCall] | None = None,
    content: str = "",
    tools: list[ToolSpec] | None = None,
    meta: dict[str, object] | None = None,
) -> Conversation:
    return Conversation(
        messages=[
            Message(Role.user, prompt),
            Message(Role.assistant, content, tool_calls=calls or []),
        ],
        tools=tools or [],
        meta={"category": category, **(meta or {})},
    )


def _episode(
    kind: str,
    prompt: str,
    *,
    first_call: ToolCall,
    response: str,
    second_call: ToolCall | None = None,
    final: str = "Done.",
    tools: list[ToolSpec] | None = None,
    meta: dict[str, object] | None = None,
) -> Conversation:
    messages = [
        Message(Role.user, prompt),
        Message(Role.assistant, tool_calls=[first_call]),
        Message(Role.tool, tool_response=response),
    ]
    if second_call is not None:
        messages.extend(
            [
                Message(Role.assistant, tool_calls=[second_call]),
                Message(Role.tool, tool_response="ok"),
            ]
        )
    messages.append(Message(Role.assistant, final))
    return Conversation(
        messages=messages,
        tools=tools or [],
        meta={"kind": kind, **(meta or {})},
    )


def _strata_dict(selection) -> dict[str, dict[str, dict[str, int]]]:
    return selection.audit.as_dict()["strata"]


def test_selector_covers_all_observed_capability_and_schema_strata() -> None:
    scroll = _tool(
        "scroll",
        properties={
            "direction": {"type": "string", "enum": ["up", "down"]},
            "amount": {"type": "number"},
            "smooth": {"type": "boolean"},
        },
        required=["direction"],
    )
    screenshot = _tool("screenshot")
    patch = _tool(
        "edit_file",
        properties={"path": {"type": "string"}},
        required=["path"],
    )
    rows = [
        _single(
            "parallel",
            "Scroll precisely and capture.",
            calls=[
                ToolCall(
                    "scroll",
                    {"direction": "down", "amount": 1.5, "smooth": True},
                ),
                ToolCall("screenshot", {}),
            ],
            tools=[scroll, screenshot],
            meta={"stratum": "schema_precision"},
        ),
        _single("no_tool", "Do something unsafe.", content="I cannot do that."),
        _single(
            "edit_file",
            "Patch the file.",
            calls=[ToolCall("edit_file", {"path": "src/app.py"})],
            tools=[patch],
        ),
        _episode(
            "workflow_episode",
            "Inspect first.",
            first_call=ToolCall("screenshot", {}),
            response="<screen>",
            tools=[screenshot],
        ),
        _episode(
            "error_recovery",
            "Repair and retry.",
            first_call=ToolCall("edit_file", {"path": "broken.py"}),
            response="FAILED: invalid syntax",
            second_call=ToolCall("edit_file", {"path": "fixed.py"}),
            tools=[patch],
            meta={"stratum": "scripted_failure_recovery"},
        ),
        _episode(
            "planner_episode",
            "Do not over-plan.",
            first_call=ToolCall("screenshot", {}),
            response="<screen>",
            tools=[screenshot],
            meta={"plan_len": 0},
        ),
        _episode(
            "planner_episode",
            "Make a two-step plan.",
            first_call=ToolCall("screenshot", {}),
            response="<screen>",
            second_call=ToolCall("edit_file", {"path": "plan.py"}),
            tools=[screenshot, patch],
            meta={"plan_len": 2},
        ),
    ]

    selection = select_stratified_eval_subset(rows, max_rows=len(rows))
    strata = _strata_dict(selection)

    assert selection.source_row_numbers == tuple(range(1, len(rows) + 1))
    assert selection.conversations == tuple(rows)
    assert set(strata["single_turn_category"]) == {
        "edit_file",
        "no_tool",
        "parallel",
    }
    assert set(strata["multi_turn_kind"]) == {
        "error_recovery",
        "planner_episode",
        "workflow_episode",
    }
    assert set(strata["planner_plan_len"]) == {"0", "2"}
    assert set(strata["tool_name"]) == {"edit_file", "screenshot", "scroll"}
    assert {
        "abstention",
        "parallel",
        "recovery",
        "schema_argument_type_boolean",
        "schema_argument_type_number",
        "schema_argument_type_string",
        "schema_enum_argument",
        "schema_episode",
        "schema_multiple_arguments",
        "schema_optional_argument",
        "text",
    } <= set(strata["behavior"])
    assert all(
        counts["selected_rows"] >= 1 for family in strata.values() for counts in family.values()
    )

    audit = selection.audit.as_dict()
    expected_decisions = sum(
        message.role == Role.assistant for conversation in rows for message in conversation.messages
    )
    assert audit["source"]["rows"] == len(rows)
    assert audit["source"]["assistant_decisions"] == expected_decisions
    assert audit["selected"]["assistant_decisions"] == expected_decisions
    assert audit["source"]["semantic_set_sha256"] == audit["selected"]["semantic_set_sha256"]
    audit_sha256 = audit.pop("audit_sha256")
    assert audit_sha256 == hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    assert selection.audit.canonical_bytes().endswith(b"\n")


def test_selector_uses_semantic_sha_for_coverage_and_fill_then_restores_source_order() -> None:
    tool = _tool(
        "lookup",
        properties={"query": {"type": "string"}},
        required=["query"],
    )
    rows = [
        _single(
            "lookup",
            prompt,
            calls=[ToolCall("lookup", {"query": prompt})],
            tools=[tool],
        )
        for prompt in ("zeta", "alpha", "gamma", "beta")
    ]
    semantic = [conversation_semantic_sha256(row) for row in rows]
    expected_indices = sorted(sorted(range(len(rows)), key=lambda index: semantic[index])[:2])

    selection = select_stratified_eval_subset(rows, max_rows=2)

    assert selection.source_row_numbers == tuple(index + 1 for index in expected_indices)
    assert list(selection.conversations) == [rows[index] for index in expected_indices]
    assert selection.audit.coverage_rows == 1
    assert selection.audit.fill_rows == 1
    assert selection.audit.as_dict() == selection.audit.as_dict()
    assert selection.audit.canonical_bytes() == selection.audit.canonical_bytes()

    reversed_selection = select_stratified_eval_subset(list(reversed(rows)), max_rows=2)
    assert {conversation_semantic_sha256(row) for row in selection.conversations} == {
        conversation_semantic_sha256(row) for row in reversed_selection.conversations
    }
    assert (
        selection.audit.source_semantic_set_sha256
        == reversed_selection.audit.source_semantic_set_sha256
    )
    assert (
        selection.audit.selected_semantic_set_sha256
        == reversed_selection.audit.selected_semantic_set_sha256
    )


def test_selector_fails_closed_when_capacity_cannot_cover_mandatory_categories() -> None:
    rows = [
        _single(category, f"{category} request", content=f"{category} response")
        for category in ("alpha", "beta", "gamma")
    ]

    with pytest.raises(
        InsufficientStratumCapacityError,
        match=r"max_rows=2, required_rows=3",
    ) as raised:
        select_stratified_eval_subset(rows, max_rows=2)

    assert raised.value.max_rows == 2
    assert raised.value.required_rows == 3
    assert raised.value.mandatory_strata == 4  # three categories plus text behavior


@pytest.mark.parametrize("max_rows", [0, -1, True, 1.5])
def test_selector_rejects_invalid_capacity(max_rows: object) -> None:
    row = _single("text", "hello", content="hi")
    with pytest.raises(ValueError, match="positive integer"):
        select_stratified_eval_subset([row], max_rows=max_rows)  # type: ignore[arg-type]


def test_strata_extraction_fails_closed_on_noncanonical_metadata() -> None:
    missing_category = Conversation(
        messages=[Message(Role.user, "hello"), Message(Role.assistant, "hi")]
    )
    with pytest.raises(ValueError, match="meta.category"):
        conversation_eval_strata(missing_category)

    multi = _episode(
        "planner_episode",
        "plan",
        first_call=ToolCall("step", {}),
        response="ok",
        meta={"plan_len": 1},
    )
    with pytest.raises(ValueError, match="meta.kind"):
        conversation_eval_strata(replace(multi, meta={"plan_len": 1}))

    with pytest.raises(ValueError, match="non-negative integer"):
        conversation_eval_strata(replace(multi, meta={"kind": "planner_episode", "plan_len": True}))

    with pytest.raises(ValueError, match="empty source"):
        select_stratified_eval_subset([], max_rows=1)
