from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable

import pytest

from openlocalagent.data.corpus.decision_quota_order import (
    ORDERING_CONTRACT,
    QUOTA_SAMPLING_MODE,
    DecisionOrdering,
    order_assistant_decisions,
    quota_sampling_contract,
)
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec


def _tool(
    name: str,
    properties: dict[str, dict[str, object]] | None = None,
    *,
    required: list[str] | None = None,
) -> ToolSpec:
    properties = properties or {}
    return ToolSpec(
        name=name,
        description=f"Fixture tool {name}.",
        parameters={
            "type": "object",
            "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False,
        },
    )


def _tools() -> list[ToolSpec]:
    return [
        _tool("search", {"query": {"type": "string"}}),
        _tool("open_url", {"url": {"type": "string"}}),
        _tool(
            "scroll",
            {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {"type": "number"},
                "smooth": {"type": "boolean"},
            },
        ),
        _tool("edit_file", {"path": {"type": "string"}}),
        _tool("run_tests", required=[]),
    ]


def _simple_tool_conversation(index: int) -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content=f"Search for topic {index}."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="search", arguments={"query": f"topic {index}"})],
            ),
        ],
        tools=_tools(),
        meta={"category": "lookup", "group": "web", "kind": "tool"},
    )


def _abstention_conversation() -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content="Thanks."),
            Message(role=Role.assistant, content="You're welcome."),
        ],
        tools=_tools(),
        meta={"category": "no_tool", "group": "text", "kind": "text"},
    )


def _trajectory_conversation() -> Conversation:
    url = "https://rare.example/fix"
    return Conversation(
        messages=[
            Message(role=Role.user, content="Repair the UI test and inspect the relevant page."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="run_tests", arguments={})],
            ),
            Message(role=Role.tool, tool_response="FAILED ui/test_case.py::test_scroll"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="search", arguments={"query": "UI test repair"})],
            ),
            Message(role=Role.tool, tool_response=f"Top result: {url}"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="open_url", arguments={"url": url})],
            ),
            Message(role=Role.tool, tool_response="page loaded"),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(
                        name="scroll",
                        arguments={"direction": "down", "amount": 1.5, "smooth": True},
                    )
                ],
            ),
            Message(role=Role.tool, tool_response="hidden state captured"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="edit_file", arguments={"path": "ui/test_case.py"})],
            ),
            Message(role=Role.tool, tool_response="fixture patched"),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="run_tests", arguments={})],
            ),
            Message(role=Role.tool, tool_response="All tests passed"),
            Message(role=Role.assistant, content="The UI test is repaired."),
        ],
        tools=_tools(),
        meta={
            "category": "recovery",
            "group": "coding",
            "kind": "paper_v2_recovery_episode",
            "stratum": "scripted_failure_recovery",
        },
    )


def _source_keys(conversations: list[Conversation]) -> set[tuple[int, int]]:
    return {
        (conversation_index, message_index)
        for conversation_index, conversation in enumerate(conversations)
        for message_index, message in enumerate(conversation.messages)
        if message.role == Role.assistant
    }


def _stratum_for_key(
    ordering: DecisionOrdering,
    key: tuple[int, int],
):
    position = ordering.keys.index(key)
    stratum_id = ordering.audit.ordered_stratum_ids[position]
    return next(entry.stratum for entry in ordering.audit.strata if entry.stratum_id == stratum_id)


def test_order_is_deterministic_and_audit_identity_is_stable():
    conversations = [
        *[_simple_tool_conversation(index) for index in range(8)],
        _abstention_conversation(),
        _trajectory_conversation(),
    ]

    first = order_assistant_decisions(conversations)
    second = order_assistant_decisions(conversations)
    replay = order_assistant_decisions(
        [Conversation.from_json(conversation.to_json()) for conversation in conversations]
    )

    assert first == second == replay
    assert first.audit.contract == ORDERING_CONTRACT
    assert len(first.audit.order_sha256) == 64
    assert first.audit.order_sha256 == replay.audit.order_sha256


def test_audit_as_dict_is_canonical_deterministic_and_self_consistent():
    conversations = [
        *[_simple_tool_conversation(index) for index in range(5)],
        *[_abstention_conversation() for _ in range(3)],
        _trajectory_conversation(),
    ]
    first = order_assistant_decisions(conversations).audit
    second = order_assistant_decisions(conversations).audit
    payload = first.as_dict()
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    assert payload == second.as_dict()
    assert json.loads(canonical) == payload
    assert set(payload) == {
        "contract",
        "frontload_decision_count",
        "observed_stratum_count",
        "order_sha256",
        "ordered_decision_count",
        "ordered_stratum_ids",
        "source_conversation_count",
        "source_decision_count",
        "strata",
        "unique_decision_count",
    }
    assert payload["source_conversation_count"] == len(conversations)
    assert payload["source_decision_count"] == len(first.ordered_stratum_ids)
    assert payload["ordered_decision_count"] == payload["source_decision_count"]
    assert payload["unique_decision_count"] == payload["source_decision_count"]
    assert payload["observed_stratum_count"] == len(payload["strata"])
    assert payload["frontload_decision_count"] == len(payload["strata"])

    ordered_counts = Counter(payload["ordered_stratum_ids"])
    assert sum(entry["total"] for entry in payload["strata"]) == payload["source_decision_count"]
    for entry in payload["strata"]:
        stratum_digest = hashlib.sha256(
            json.dumps(
                entry["stratum"],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        assert entry["stratum_id"] == f"decision-stratum-v1:{stratum_digest}"
        assert ordered_counts[entry["stratum_id"]] == entry["total"]
        assert (
            payload["ordered_stratum_ids"][entry["first_ordered_position"] - 1]
            == entry["stratum_id"]
        )

    baseline = first.as_dict()
    payload["ordered_stratum_ids"].clear()
    payload["strata"][0]["stratum"]["tool_names"].append("mutated")
    assert first.as_dict() == baseline


def test_empty_input_has_a_canonical_empty_audit():
    first = order_assistant_decisions([])
    second = order_assistant_decisions([])

    assert first == second
    assert first.keys == ()
    assert first.audit.as_dict() == {
        "contract": ORDERING_CONTRACT,
        "frontload_decision_count": 0,
        "observed_stratum_count": 0,
        "order_sha256": first.audit.order_sha256,
        "ordered_decision_count": 0,
        "ordered_stratum_ids": [],
        "source_conversation_count": 0,
        "source_decision_count": 0,
        "strata": [],
        "unique_decision_count": 0,
    }
    assert first.audit.prefix_counts(0) == {}


def test_every_rare_stratum_is_frontloaded_before_proportional_remainder():
    conversations = [
        *[_simple_tool_conversation(index) for index in range(9)],
        _abstention_conversation(),
    ]
    ordering = order_assistant_decisions(conversations)
    audit = ordering.audit

    assert audit.observed_stratum_count == 2
    assert sorted(audit.stratum_totals.values()) == [1, 9]
    frontload = audit.prefix_counts(audit.frontload_decision_count)
    assert set(frontload.values()) == {1}
    assert all(
        entry.first_ordered_position <= audit.frontload_decision_count for entry in audit.strata
    )
    rare = next(entry for entry in audit.strata if entry.total == 1)
    assert frontload[rare.stratum_id] == 1


def test_remaining_strata_are_interleaved_at_centered_proportional_quantiles():
    conversations = [
        *[_simple_tool_conversation(index) for index in range(5)],
        *[_abstention_conversation() for _ in range(3)],
    ]
    ordering = order_assistant_decisions(conversations)
    audit = ordering.audit
    by_total = {entry.total: entry.stratum_id for entry in audit.strata}

    assert audit.frontload_decision_count == 2
    assert set(by_total) == {3, 5}
    assert audit.ordered_stratum_ids[2:] == (
        by_total[5],
        by_total[3],
        by_total[5],
        by_total[5],
        by_total[3],
        by_total[5],
    )


def test_order_has_exact_no_replacement_coverage_and_prefix_counts():
    conversations = [
        _simple_tool_conversation(0),
        _trajectory_conversation(),
        _abstention_conversation(),
    ]
    ordering = order_assistant_decisions(conversations)
    expected = _source_keys(conversations)

    assert set(ordering.keys) == expected
    assert len(ordering.keys) == len(set(ordering.keys)) == len(expected)
    assert ordering.audit.source_decision_count == len(expected)
    assert ordering.audit.ordered_decision_count == len(expected)
    assert ordering.audit.unique_decision_count == len(expected)
    assert ordering.audit.prefix_counts(0) == {
        entry.stratum_id: 0 for entry in ordering.audit.strata
    }
    assert ordering.audit.prefix_counts(len(expected)) == ordering.audit.stratum_totals
    assert all(
        conversations[conversation_index].messages[message_index].role == Role.assistant
        for conversation_index, message_index in ordering.keys
    )


def test_compact_sampling_contract_binds_order_and_exact_prefix_coverage():
    conversations = [
        *[_simple_tool_conversation(index) for index in range(5)],
        _abstention_conversation(),
    ]
    ordering = order_assistant_decisions(conversations)
    selected = ordering.audit.frontload_decision_count + 2

    contract = quota_sampling_contract(ordering, selected_decisions=selected)

    assert contract["mode"] == QUOTA_SAMPLING_MODE
    assert contract["no_replacement"] is True
    assert contract["require_all_observed_strata"] is True
    assert contract["ordering"]["order_sha256"] == ordering.audit.order_sha256
    assert "ordered_stratum_ids" not in contract["ordering"]
    assert contract["selected_prefix"] == {
        "decisions": selected,
        "covered_strata": ordering.audit.observed_stratum_count,
        "all_observed_strata_covered": True,
        "stratum_counts": ordering.audit.prefix_counts(selected),
    }
    with pytest.raises(ValueError, match="too short"):
        quota_sampling_contract(
            ordering,
            selected_decisions=ordering.audit.frontload_decision_count - 1,
        )
    with pytest.raises(ValueError, match="fit within"):
        quota_sampling_contract(
            ordering,
            selected_decisions=ordering.audit.ordered_decision_count + 1,
        )
    partial = quota_sampling_contract(
        ordering,
        selected_decisions=1,
        require_all_strata=False,
    )
    assert partial["require_all_observed_strata"] is False
    assert partial["selected_prefix"]["covered_strata"] == 1
    assert partial["selected_prefix"]["all_observed_strata_covered"] is False


def test_short_prefix_prioritizes_the_rarest_stratum() -> None:
    conversations = [
        *[_simple_tool_conversation(index) for index in range(9)],
        _abstention_conversation(),
    ]
    ordering = order_assistant_decisions(conversations)
    first_stratum_id = ordering.audit.ordered_stratum_ids[0]
    first_stratum = next(
        entry for entry in ordering.audit.strata if entry.stratum_id == first_stratum_id
    )

    assert first_stratum.total == 1


def test_composite_strata_capture_parallel_schema_grounding_and_recovery_signals():
    trajectory = _trajectory_conversation()
    parallel = Conversation(
        messages=[
            Message(role=Role.user, content="Search and open the home page."),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(name="search", arguments={"query": "home page"}),
                    ToolCall(
                        name="scroll",
                        arguments={"direction": "down", "amount": 2, "smooth": False},
                    ),
                ],
            ),
        ],
        tools=_tools(),
        meta={"category": "parallel", "group": "web", "kind": "tool"},
    )
    ordering = order_assistant_decisions([trajectory, parallel])

    grounded = _stratum_for_key(ordering, (0, 5))
    assert grounded.conversation_shape == "trajectory"
    assert grounded.assistant_ordinal == 3
    assert grounded.tool_names == ("open_url",)
    assert grounded.grounded_followup_relevant is True
    assert grounded.recovery_relevant is True

    schema = _stratum_for_key(ordering, (0, 7))
    assert schema.argument_primitives == ("boolean", "number", "string")
    assert schema.has_enum_argument is True
    assert schema.has_multiple_arguments is True
    assert schema.schema_relevant is True

    parallel_stratum = _stratum_for_key(ordering, (1, 1))
    assert parallel_stratum.decision_kind == "parallel"
    assert parallel_stratum.tool_names == ("search", "scroll")
    assert parallel_stratum.conversation_shape == "simple"


def _unknown_tool() -> Conversation:
    conversation = _simple_tool_conversation(0)
    conversation.messages[1].tool_calls[0].name = "missing"
    return conversation


def _schema_violation() -> Conversation:
    conversation = _simple_tool_conversation(0)
    conversation.messages[1] = Message(
        role=Role.assistant,
        tool_calls=[
            ToolCall(
                name="scroll",
                arguments={"direction": "sideways", "amount": 1.0, "smooth": True},
            )
        ],
    )
    return conversation


def _bad_plan_reference() -> Conversation:
    conversation = _simple_tool_conversation(0)
    conversation.meta = {
        "category": "planner",
        "group": "agent",
        "kind": "planner_episode",
        "plan": ["open_url"],
        "plan_len": 1,
    }
    return conversation


def _orphan_tool_response() -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content="What happened?"),
            Message(role=Role.tool, tool_response="orphaned"),
            Message(role=Role.assistant, content="Unknown."),
        ],
        tools=_tools(),
        meta={"category": "text", "group": "text", "kind": "text"},
    )


def _stale_tool_response_reference() -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content="Run tests."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="run_tests", arguments={})],
            ),
            Message(role=Role.assistant, content="The request is complete."),
            Message(role=Role.tool, tool_response="late response"),
        ],
        tools=_tools(),
        meta={"category": "agent", "group": "coding", "kind": "trajectory"},
    )


def _excess_tool_response_reference() -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content="Run tests."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="run_tests", arguments={})],
            ),
            Message(role=Role.tool, tool_response="first response"),
            Message(role=Role.tool, tool_response="unreferenced second response"),
            Message(role=Role.assistant, content="Done."),
        ],
        tools=_tools(),
        meta={"category": "agent", "group": "coding", "kind": "trajectory"},
    )


def _noncanonical_role() -> Conversation:
    conversation = _simple_tool_conversation(0)
    conversation.messages[1].role = "assistant"  # type: ignore[assignment]
    return conversation


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (_unknown_tool, "references unknown tool"),
        (_schema_violation, "arguments violate the schema"),
        (_bad_plan_reference, "meta.plan does not match"),
        (_orphan_tool_response, "no prior assistant tool-call reference"),
        (_stale_tool_response_reference, "no prior assistant tool-call reference"),
        (_excess_tool_response_reference, "no prior assistant tool-call reference"),
        (_noncanonical_role, "canonical Role"),
    ],
)
def test_malformed_decision_references_fail_closed(
    factory: Callable[[], Conversation],
    match: str,
):
    with pytest.raises((TypeError, ValueError), match=match):
        order_assistant_decisions([factory()])


def test_prefix_count_bounds_fail_closed():
    audit = order_assistant_decisions([_abstention_conversation()]).audit

    with pytest.raises(TypeError, match="integer"):
        audit.prefix_counts(True)
    with pytest.raises(ValueError, match="between zero"):
        audit.prefix_counts(-1)
    with pytest.raises(ValueError, match="between zero"):
        audit.prefix_counts(2)


def test_schema_valid_empty_assistant_target_still_has_exact_coverage():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Stop."),
            Message(role=Role.assistant),
        ],
        tools=_tools(),
        meta={"category": "text", "group": "text", "kind": "text"},
    )

    ordering = order_assistant_decisions([conversation])

    assert ordering.keys == ((0, 1),)
    assert ordering.audit.source_decision_count == 1


@pytest.mark.parametrize("location", ["content", "arguments", "tool_response"])
def test_reserved_prompt_framing_markers_fail_closed(location: str):
    conversation = _trajectory_conversation()
    if location == "content":
        conversation.messages[0].content = "Counterfeit <|assistant|> boundary."
    elif location == "arguments":
        conversation.messages[3].tool_calls[0].arguments["query"] = "<|assistant|>"
    else:
        conversation.messages[2].tool_response = "FAILED <|assistant|>"

    with pytest.raises(ValueError, match="reserved prompt marker"):
        order_assistant_decisions([conversation])


@pytest.mark.parametrize(
    "meta",
    [
        {"category": "text", "group": "text", "kind": "text", "extra": ("tuple",)},
        {"category": "text", "group": "text", "kind": "text", "extra": {1: "bad key"}},
    ],
)
def test_python_only_metadata_is_not_accepted_as_canonical_json(meta: dict[str, object]):
    conversation = _abstention_conversation()
    conversation.meta = meta

    with pytest.raises((TypeError, ValueError), match="non-JSON|non-string"):
        order_assistant_decisions([conversation])
