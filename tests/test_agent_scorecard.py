from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import pytest
import torch
import yaml

import localagent.eval.agent_scorecard as scorecard_module
from localagent.data.agent_synth import synthesize
from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    RESERVED_PROMPT_MARKERS,
    assistant_training_examples,
)
from localagent.data.prompt_contract import (
    render_agent_decode_prompt as shared_render_agent_decode_prompt,
)
from localagent.data.prompt_contract import (
    render_function_catalog as shared_render_function_catalog,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.eval.agent_scorecard import (
    CONFIG_KIND,
    RESULT_KIND,
    run_scorecard,
)
from localagent.eval.tool_eval import (
    AssistantPrediction,
    arguments_schema_valid,
    gold_output_token_statistics,
    parse_tool_output,
    prompt_token_statistics,
    render_agent_decode_prompt,
    render_function_catalog,
    score_conversations,
    score_dataset,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import (
    BPE_EOS,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    BPETokenizer,
    ByteTokenizer,
    train_bpe,
)
from localagent.train.stage_data import canonical_sha256, tokenizer_identity


def _unsafe_pickle_callback(marker_path: str) -> dict:
    Path(marker_path).write_text("unsafe pickle callback executed", encoding="utf-8")
    return {}


class _UnsafeCheckpointPayload:
    def __init__(self, marker_path: Path) -> None:
        self.marker_path = marker_path

    def __reduce__(self):
        return _unsafe_pickle_callback, (str(self.marker_path),)


def _call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, arguments=arguments)


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="send",
            description="Send",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                },
                "required": ["to", "subject"],
            },
        ),
    ]


def _tool_text(*calls: ToolCall) -> str:
    return "".join(
        "<tool_call>"
        + json.dumps(
            {"name": call.name, "arguments": call.arguments},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "</tool_call>"
        for call in calls
    )


def test_strict_tool_output_parser_rejects_repair_and_mixed_content():
    valid = parse_tool_output(
        _tool_text(_call("search", query="alpha"), _call("send", to="Ada", subject="Hi"))
    )
    assert valid.format_valid is True
    assert [call.name for call in valid.calls] == ["search", "send"]

    assert parse_tool_output("No tool is needed.").format_valid is True
    assert parse_tool_output("No tool is needed.").tool_syntax_present is False
    assert parse_tool_output("<tool_call>{bad}</tool_call>").format_valid is False
    assert (
        parse_tool_output(
            '<tool_call>{"name":"search","name":"send","arguments":{}}</tool_call>'
        ).format_valid
        is False
    )
    assert (
        parse_tool_output(
            '<tool_call>{"name":"search","arguments":{"query":NaN}}</tool_call>'
        ).format_valid
        is False
    )
    assert (
        parse_tool_output(
            '<tool_call>{"name":"search","arguments":{"query":1e400}}</tool_call>'
        ).format_valid
        is False
    )
    assert (
        parse_tool_output(
            _tool_text(_call("search", query="alpha")) + " trailing explanation"
        ).format_valid
        is False
    )
    for marker in RESERVED_PROMPT_MARKERS:
        if marker in {TOOL_CALL_OPEN, TOOL_CALL_CLOSE}:
            continue
        spilled = parse_tool_output(marker + "role or boundary spill")
        assert spilled.format_valid is False
        assert f"reserved_prompt_marker:{marker}" in spilled.errors
        assert (
            parse_tool_output(
                _tool_text(_call("search", query=f"value {marker} spill"))
            ).format_valid
            is False
        )
    assert (
        parse_tool_output(_tool_text(_call("search", query=TOOL_CALL_OPEN))).format_valid is False
    )
    assert (
        parse_tool_output(
            '<tool_call>{"name":"search","arguments":{"query":"<\\u007cuser\\u007c>"}}</tool_call>'
        ).format_valid
        is False
    )


def test_decode_prompt_preserves_system_and_full_unseen_or_renamed_function_catalog():
    parameters = {
        "type": "object",
        "description": "A described top-level telemetry request.",
        "properties": {
            "satellite": {"type": "string", "description": "Satellite identifier."},
            "window": {
                "type": "object",
                "description": "Sampling window.",
                "properties": {
                    "start": {"type": "string", "description": "Window start."},
                    "samples": {
                        "type": "array",
                        "description": "Requested samples.",
                        "items": {"type": "number", "description": "Sample value."},
                    },
                },
                "required": ["start", "samples"],
                "additionalProperties": False,
            },
        },
        "required": ["satellite", "window"],
        "additionalProperties": False,
    }
    unseen = ToolSpec(
        name="lookup_satellite_telemetry_v9",
        description="Fetch telemetry for a satellite and sampling window.",
        parameters=parameters,
    )
    messages = [
        Message(role=Role.system, content="Never invent telemetry."),
        Message(role=Role.user, content="Inspect Asteria."),
    ]

    prompt = render_agent_decode_prompt(messages, [unseen])
    catalog_text = prompt.removeprefix("<|tool_catalog|>").split(
        "</|tool_catalog|>",
        maxsplit=1,
    )[0]
    catalog = json.loads(catalog_text)

    assert catalog == {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": unseen.name,
                    "description": unseen.description,
                    "parameters": parameters,
                },
            }
        ]
    }
    assert (
        "</|tool_catalog|><|end|><|system|>Never invent telemetry."
        "<|user|>Inspect Asteria.<|assistant|>"
    ) in prompt

    renamed = ToolSpec(
        name="fetch_orbit_samples",
        description=unseen.description,
        parameters=parameters,
    )
    renamed_prompt = render_agent_decode_prompt(messages, [renamed])
    assert renamed_prompt != prompt
    assert '"name":"fetch_orbit_samples"' in renamed_prompt
    assert '"name":"lookup_satellite_telemetry_v9"' not in renamed_prompt


def test_evaluator_prompts_are_exact_shared_training_examples_with_eos_boundaries():
    tools = _tools()
    conversation = Conversation(
        messages=[
            Message(role=Role.system, content="Use only supplied tools."),
            Message(role=Role.user, content="Find alpha."),
            Message(role=Role.assistant, tool_calls=[_call("search", query="alpha")]),
            Message(role=Role.tool, tool_response='{"result":"alpha"}'),
            Message(role=Role.user, content="Send the result."),
            Message(
                role=Role.assistant,
                tool_calls=[_call("send", to="Ada", subject="alpha")],
            ),
        ],
        tools=tools,
        meta={"kind": "planner_episode"},
    )
    examples = assistant_training_examples(conversation)
    expected_outputs = {example.prompt: example.body for example in examples}
    observed_prompts: list[str] = []

    def predictor(prompt: str, _tools: list[ToolSpec] | tuple[ToolSpec, ...]) -> str:
        observed_prompts.append(prompt)
        return expected_outputs[prompt]

    result = score_conversations([conversation], predictor)

    assert render_agent_decode_prompt is shared_render_agent_decode_prompt
    assert render_function_catalog is shared_render_function_catalog
    assert observed_prompts == [example.prompt for example in examples]
    assert result["metrics"]["whole_call_exact"]["accuracy"] == 1.0
    catalog_boundary = shared_render_function_catalog(tools) + BPE_EOS
    assert all(example.prompt.startswith(catalog_boundary) for example in examples)
    assert examples[0].prompt.count(BPE_EOS) == 1
    assert examples[1].prompt.count(BPE_EOS) == 2
    assert (
        "<|assistant|>"
        + _tool_text(_call("search", query="alpha"))
        + BPE_EOS
        + '<|tool|><tool_response>{"result":"alpha"}</tool_response>'
        in examples[1].prompt
    )


def test_recursive_schema_validation_and_unsupported_constructs_fail_closed(
    tmp_path: Path,
):
    nested_schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["red", "blue"]},
                    },
                    "scores": {"type": "array", "items": {"type": "number"}},
                    "enabled": {"type": "boolean"},
                },
                "required": ["tags", "scores", "enabled"],
                "additionalProperties": False,
            }
        },
        "required": ["payload"],
        "additionalProperties": False,
    }
    valid = {
        "payload": {
            "tags": ["red", "blue"],
            "scores": [1, 2.5],
            "enabled": True,
        }
    }
    assert arguments_schema_valid(valid, nested_schema) is True
    assert (
        arguments_schema_valid(
            {**valid, "payload": {**valid["payload"], "tags": ["green"]}},
            nested_schema,
        )
        is False
    )
    assert (
        arguments_schema_valid(
            {**valid, "payload": {**valid["payload"], "scores": [True]}},
            nested_schema,
        )
        is False
    )
    assert (
        arguments_schema_valid(
            {**valid, "payload": {**valid["payload"], "unexpected": 1}},
            nested_schema,
        )
        is False
    )
    nested_enum_schema = {
        "type": "object",
        "properties": {
            "array_value": {"enum": [[1]]},
            "object_value": {"enum": [{"x": 1}]},
        },
        "required": ["array_value", "object_value"],
        "additionalProperties": False,
    }
    assert (
        arguments_schema_valid(
            {"array_value": [True], "object_value": {"x": True}},
            nested_enum_schema,
        )
        is False
    )
    assert (
        arguments_schema_valid(
            {"array_value": [1.0], "object_value": {"x": 1}},
            nested_enum_schema,
        )
        is True
    )

    tool = ToolSpec("nested_action", "Exercise nested arguments.", nested_schema)
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Run the nested action."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name=tool.name, arguments=valid)],
            ),
        ],
        tools=[tool],
        meta={"category": "nested"},
    )
    dataset = tmp_path / "nested.jsonl"
    dataset.write_text(conversation.to_json() + "\n", encoding="utf-8")
    invalid_nested = {
        **valid,
        "payload": {**valid["payload"], "scores": ["not-a-number"]},
    }
    result = score_dataset(
        dataset,
        lambda _prompt, _tools: _tool_text(ToolCall(name=tool.name, arguments=invalid_nested)),
    )
    assert result["metrics"]["format_validity"]["accuracy"] == 1.0
    assert result["metrics"]["schema_validity_on_tool_attempts"]["accuracy"] == 0.0
    assert result["metrics"]["tool_format_validity_on_tool_decisions"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }
    assert result["metrics"]["schema_validity_on_tool_decisions"] == {
        "correct": 0,
        "total": 1,
        "accuracy": 0.0,
    }
    assert result["metrics"]["whole_call_exact"]["accuracy"] == 0.0

    unsupported = ToolSpec(
        "unsupported",
        "Unsupported union.",
        {
            "type": "object",
            "properties": {"value": {"oneOf": [{"type": "string"}, {"type": "number"}]}},
        },
    )
    with pytest.raises(ValueError, match="unsupported JSON Schema keywords"):
        render_agent_decode_prompt([], [unsupported])
    with pytest.raises(ValueError, match="array schema must declare items"):
        arguments_schema_valid({"values": []}, {"type": "array"})


def test_no_tool_text_exact_is_literal_while_abstention_is_structural(tmp_path: Path):
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Do not call a tool."),
            Message(role=Role.assistant, content="Acknowledged."),
        ],
        tools=_tools(),
        meta={"category": "no_tool"},
    )
    dataset = tmp_path / "literal.jsonl"
    dataset.write_text(conversation.to_json() + "\n", encoding="utf-8")

    result = score_dataset(dataset, lambda _prompt, _tools: "Acknowledged. ")

    assert result["metrics"]["abstention"]["accuracy"] == 1.0
    assert result["metrics"]["no_tool_text_exact"]["accuracy"] == 0.0
    assert result["metrics"]["assistant_response_exact"]["accuracy"] == 0.0


def test_tool_conditioned_metrics_reject_abstention_only_action_credit():
    tools = _tools()
    tool_conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"Find item {index}."),
                Message(
                    role=Role.assistant,
                    tool_calls=[_call("search", query=f"item {index}")],
                ),
            ],
            tools=tools,
            meta={"category": "tool"},
        )
        for index in range(611)
    ]
    no_tool_conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"Do not call a tool for item {index}."),
                Message(role=Role.assistant, content="Acknowledged."),
            ],
            tools=tools,
            meta={"category": "no_tool"},
        )
        for index in range(209)
    ]

    result = score_conversations(
        [*tool_conversations, *no_tool_conversations],
        lambda _prompt, _tools: "Acknowledged.",
    )

    assert result["metrics"]["action_exact"] == {
        "correct": 209,
        "total": 820,
        "accuracy": 209 / 820,
    }
    assert result["metrics"]["tool_format_validity_on_tool_decisions"] == {
        "correct": 0,
        "total": 611,
        "accuracy": 0.0,
    }
    assert result["metrics"]["schema_validity_on_tool_decisions"] == {
        "correct": 0,
        "total": 611,
        "accuracy": 0.0,
    }


def test_length_capped_exact_looking_outputs_are_always_inexact():
    tools = _tools()
    conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content="Find alpha."),
                Message(role=Role.assistant, tool_calls=[_call("search", query="alpha")]),
            ],
            tools=tools,
            meta={"category": "tool"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Do not call a tool."),
                Message(role=Role.assistant, content="Acknowledged."),
            ],
            tools=tools,
            meta={"category": "no_tool"},
        ),
    ]
    outputs = {
        example.prompt: example.body
        for conversation in conversations
        for example in assistant_training_examples(conversation)
    }

    capped = score_conversations(
        conversations,
        lambda prompt, _tools: AssistantPrediction(
            text=outputs[prompt],
            finish_reason="length",
        ),
    )

    assert capped["metrics"]["generation_completion"] == {
        "correct": 0,
        "total": 2,
        "accuracy": 0.0,
    }
    assert capped["metrics"]["format_validity"]["accuracy"] == 0.0
    assert capped["metrics"]["action_exact"]["accuracy"] == 0.0
    assert capped["metrics"]["assistant_response_exact"]["accuracy"] == 0.0
    assert capped["metrics"]["whole_call_exact"]["accuracy"] == 0.0
    assert capped["metrics"]["abstention"]["accuracy"] == 0.0
    assert capped["predictions"]["finish_reasons"] == {"length": 2}
    assert capped["predictions"]["complete"] == 0
    assert capped["predictions"]["terminated_by_eos"] == 0

    eos_terminated = score_conversations(
        conversations,
        lambda prompt, _tools: AssistantPrediction(
            text=outputs[prompt],
            finish_reason="eos",
        ),
    )

    assert eos_terminated["metrics"]["generation_completion"]["accuracy"] == 1.0
    assert eos_terminated["metrics"]["action_exact"]["accuracy"] == 1.0
    assert eos_terminated["metrics"]["assistant_response_exact"]["accuracy"] == 1.0
    assert eos_terminated["predictions"]["finish_reasons"] == {"eos": 2}
    assert eos_terminated["predictions"]["terminated_by_eos"] == 2


def test_score_dataset_propagates_model_decode_termination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Reply briefly."),
            Message(role=Role.assistant, content="OK"),
        ],
        tools=_tools(),
        meta={"category": "termination"},
    )
    dataset = tmp_path / "termination.jsonl"
    dataset.write_text(conversation.to_json() + "\n", encoding="utf-8")

    class Stats:
        new_tokens = 3

    def capped_generate(*_args, **_kwargs):
        return "OK", Stats()

    monkeypatch.setattr("localagent.inference.generate.generate", capped_generate)
    result = score_dataset(
        dataset,
        object(),
        ByteTokenizer(),
        max_new_tokens=3,
    )

    assert result["gold_output_budget"]["fits_generation_budget"] is True
    assert result["metrics"]["generation_completion"]["accuracy"] == 0.0
    assert result["metrics"]["action_exact"]["accuracy"] == 0.0
    assert result["predictions"]["finish_reasons"] == {"length": 1}


def test_gold_output_budget_counts_every_body_plus_eos_without_retaining_lengths():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="First."),
            Message(role=Role.assistant, content="A"),
            Message(role=Role.user, content="Second."),
            Message(role=Role.assistant, content="BC"),
        ],
        tools=_tools(),
        meta={"category": "budget"},
    )

    exact_fit = gold_output_token_statistics(
        [conversation],
        ByteTokenizer(),
        max_new_tokens=3,
    )

    assert exact_fit["assistant_decisions"] == 2
    assert exact_fit["gold_body_plus_eos_tokens"] == {
        "minimum": 2,
        "p50_nearest_rank": 2,
        "p95_nearest_rank": 3,
        "p99_nearest_rank": 3,
        "maximum": 3,
        "total": 5,
        "ordered_values_sha256": exact_fit["gold_body_plus_eos_tokens"]["ordered_values_sha256"],
    }
    assert exact_fit["outputs_at_limit"] == 1
    assert exact_fit["outputs_over_max_new_tokens"] == 0
    assert exact_fit["outputs_with_embedded_eos"] == 0
    assert exact_fit["fits_generation_budget"] is True

    too_short = gold_output_token_statistics(
        [conversation],
        ByteTokenizer(),
        max_new_tokens=2,
    )
    assert too_short["outputs_at_limit"] == 1
    assert too_short["outputs_over_max_new_tokens"] == 1
    assert too_short["fits_generation_budget"] is False
    assert too_short["over_budget_case_ids_sha256"] != exact_fit["over_budget_case_ids_sha256"]


def test_case_set_identity_binds_category_metadata():
    messages = [
        Message(role=Role.user, content="No tool."),
        Message(role=Role.assistant, content="Done."),
    ]
    alpha = Conversation(messages=messages, tools=_tools(), meta={"category": "alpha"})
    beta = Conversation(messages=messages, tools=_tools(), meta={"category": "beta"})

    alpha_score = score_conversations([alpha], lambda _prompt, _tools: "Done.")
    beta_score = score_conversations([beta], lambda _prompt, _tools: "Done.")
    combined = score_conversations([alpha, beta], lambda _prompt, _tools: "Done.")

    assert alpha_score["case_set"]["sha256"] != beta_score["case_set"]["sha256"]
    assert combined["case_set"]["assistant_decisions"] == 2
    assert set(combined["by_category"]) == {"alpha", "beta"}


def test_scorecard_case_selection_is_optional_and_fails_closed_on_audit_drift():
    rows = [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"Prompt {index}."),
                Message(role=Role.assistant, content=f"Answer {index}."),
            ],
            tools=[],
            meta={"category": "text"},
        )
        for index in range(4)
    ]
    expected_selection = select_stratified_eval_subset(rows, max_rows=2)
    expected_audit = expected_selection.audit.as_dict()
    contract = {
        "algorithm": expected_audit["algorithm"],
        "max_rows": expected_audit["capacity"]["max_rows"],
        "expected_source_rows": expected_audit["source"]["rows"],
        "expected_source_assistant_decisions": expected_audit["source"][
            "assistant_decisions"
        ],
        "expected_source_semantic_set_sha256": expected_audit["source"][
            "semantic_set_sha256"
        ],
        "expected_selected_rows": expected_audit["selected"]["rows"],
        "expected_selected_assistant_decisions": expected_audit["selected"][
            "assistant_decisions"
        ],
        "expected_selected_semantic_set_sha256": expected_audit["selected"][
            "semantic_set_sha256"
        ],
        "expected_audit_sha256": expected_audit["audit_sha256"],
    }

    all_rows, no_audit = scorecard_module._select_scorecard_cases(rows, None)
    selected_rows, observed_audit = scorecard_module._select_scorecard_cases(
        rows,
        contract,
    )

    assert all_rows is rows
    assert no_audit is None
    assert tuple(selected_rows) == expected_selection.conversations
    assert observed_audit == expected_audit

    drifted = {**contract, "expected_audit_sha256": "0" * 64}
    with pytest.raises(ValueError, match="scorecard case selection contract mismatch"):
        scorecard_module._select_scorecard_cases(rows, drifted)

    incomplete = dict(contract)
    incomplete.pop("expected_selected_assistant_decisions")
    with pytest.raises(ValueError, match=r"missing=\['expected_selected_assistant_decisions'\]"):
        scorecard_module._select_scorecard_cases(rows, incomplete)


def test_teacher_forced_tool_multi_turn_metric_explicitly_excludes_no_tool_turns():
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Wait."),
            Message(role=Role.assistant, content="Waiting."),
            Message(role=Role.user, content="Now find alpha."),
            Message(role=Role.assistant, tool_calls=[_call("search", query="alpha")]),
        ],
        tools=_tools(),
        meta={"category": "mixed_episode"},
    )
    examples = assistant_training_examples(conversation)
    outputs = {
        examples[0].prompt: "Wrong no-tool response.",
        examples[1].prompt: examples[1].body,
    }

    result = score_conversations(
        [conversation],
        lambda prompt, _tools: outputs[prompt],
    )

    assert result["metrics"]["action_exact"]["accuracy"] == 1.0
    assert result["metrics"]["assistant_response_exact"]["accuracy"] == 0.5
    assert result["metrics"]["no_tool_text_exact"]["accuracy"] == 0.0
    assert result["metrics"]["teacher_forced_tool_multi_turn"] == {
        "tool_step_exact": {"correct": 1, "total": 1, "accuracy": 1.0},
        "tool_episode_exact": {"correct": 1, "total": 1, "accuracy": 1.0},
    }
    assert "multi_turn" not in result["contract"]


def test_score_dataset_reports_decomposed_parallel_abstention_and_multiturn_metrics(
    tmp_path: Path,
):
    tools = _tools()
    conversations = [
        Conversation(
            messages=[
                Message(role=Role.system, content="Use only the supplied function catalog."),
                Message(role=Role.user, content="Find alpha"),
                Message(role=Role.assistant, tool_calls=[_call("search", query="alpha")]),
            ],
            tools=tools,
            meta={"category": "single"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Send Ada Hi"),
                Message(
                    role=Role.assistant,
                    tool_calls=[_call("send", to="Ada", subject="Hi")],
                ),
            ],
            tools=tools,
            meta={"category": "multi_argument"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Find beta and send Ada"),
                Message(
                    role=Role.assistant,
                    tool_calls=[
                        _call("search", query="beta"),
                        _call("send", to="Ada", subject="Found"),
                    ],
                ),
            ],
            tools=tools,
            meta={"category": "parallel"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Tell me a joke"),
                Message(role=Role.assistant, content="I cannot help with that."),
            ],
            tools=tools,
            meta={"category": "no_tool"},
        ),
        Conversation(
            messages=[
                Message(role=Role.user, content="Find gamma"),
                Message(role=Role.assistant, tool_calls=[_call("search", query="gamma")]),
                Message(role=Role.tool, tool_response="found"),
                Message(role=Role.user, content="Send it"),
                Message(
                    role=Role.assistant,
                    tool_calls=[_call("send", to="Ada", subject="Found")],
                ),
            ],
            tools=tools,
            meta={"kind": "planner_episode"},
        ),
    ]
    predictions: dict[str, str] = {}
    for conversation in conversations:
        for index, message in enumerate(conversation.messages):
            if message.role != Role.assistant:
                continue
            prompt = render_agent_decode_prompt(
                conversation.messages[:index],
                conversation.tools,
            )
            predictions[prompt] = (
                _tool_text(*message.tool_calls) if message.tool_calls else message.content
            )
    predictions[
        render_agent_decode_prompt(conversations[1].messages[:1], conversations[1].tools)
    ] = _tool_text(_call("send", to="Ada", subject="Wrong"))
    predictions[
        render_agent_decode_prompt(conversations[2].messages[:1], conversations[2].tools)
    ] = _tool_text(
        _call("send", to="Ada", subject="Found"),
        _call("search", query="beta"),
    )
    predictions[
        render_agent_decode_prompt(conversations[3].messages[:1], conversations[3].tools)
    ] = "<tool_call>{bad}</tool_call>"
    predictions[
        render_agent_decode_prompt(conversations[4].messages[:4], conversations[4].tools)
    ] = _tool_text(_call("search", query="wrong"))

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "".join(conversation.to_json() + "\n" for conversation in conversations),
        encoding="utf-8",
    )

    result = score_dataset(dataset, lambda prompt, _available_tools: predictions[prompt])

    assert result["contract"]["official_bfcl"] is False
    assert result["dataset"]["sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["case_set"] == {
        "sha256": result["case_set"]["sha256"],
        "conversations": 5,
        "assistant_decisions": 6,
        "tool_decisions": 5,
        "no_tool_decisions": 1,
    }
    metrics = result["metrics"]
    assert metrics["format_validity"] == {"correct": 5, "total": 6, "accuracy": 5 / 6}
    assert metrics["schema_validity_on_tool_attempts"] == {
        "correct": 5,
        "total": 6,
        "accuracy": 5 / 6,
    }
    assert metrics["tool_format_validity_on_tool_decisions"] == {
        "correct": 5,
        "total": 5,
        "accuracy": 1.0,
    }
    assert metrics["schema_validity_on_tool_decisions"] == {
        "correct": 5,
        "total": 5,
        "accuracy": 1.0,
    }
    assert metrics["tool_name"]["case_exact"] == {
        "correct": 4,
        "total": 5,
        "accuracy": 0.8,
    }
    assert metrics["tool_name"]["matched_reference_calls"] == 5
    assert metrics["arguments"]["exact_calls_given_matched_name"] == {
        "correct": 4,
        "total": 5,
        "accuracy": 0.8,
    }
    assert metrics["whole_call_exact"] == {"correct": 3, "total": 5, "accuracy": 0.6}
    assert metrics["abstention"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert metrics["parallel_whole_call_exact"] == {
        "correct": 1,
        "total": 1,
        "accuracy": 1.0,
    }
    assert metrics["multi_argument_whole_call_exact"] == {
        "correct": 1,
        "total": 3,
        "accuracy": 1 / 3,
    }
    assert "multi_turn" not in metrics
    assert metrics["teacher_forced_tool_multi_turn"]["tool_step_exact"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert metrics["teacher_forced_tool_multi_turn"]["tool_episode_exact"] == {
        "correct": 0,
        "total": 1,
        "accuracy": 0.0,
    }


def _scorecard_fixture(
    tmp_path: Path,
    *,
    max_seq_len: int = 16_384,
    max_new_tokens: int = 96,
    tokenizer_kind: str = "bpe",
    atomic_bpe: bool = True,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if tokenizer_kind == "byte":
        tokenizer_config: dict[str, str] = {"kind": "byte"}
        tokenizer = tokenizer_identity("byte", vocab_size=256)
        vocab_size = 256
    elif tokenizer_kind == "bpe":
        tokenizers_package = pytest.importorskip("tokenizers")
        tokenizer_path = tmp_path / "tokenizer.json"
        if atomic_bpe:
            trained = train_bpe(
                [
                    (
                        "agent scorecard catalog system user assistant tool search send "
                        "query subject result alpha beta gamma weather calculator email "
                        "calendar reminder translate summarize supplied function description"
                    )
                ],
                tokenizer_path,
                vocab_size=300,
                min_frequency=1,
            )
            vocab_size = trained.vocab_size
        else:
            backend = tokenizers_package.Tokenizer(
                tokenizers_package.models.WordLevel(
                    {BPE_EOS: 0, "[UNK]": 1},
                    unk_token="[UNK]",
                )
            )
            backend.save(str(tokenizer_path))
            vocab_size = backend.get_vocab_size()
        tokenizer_config = {"kind": "bpe", "path": str(tokenizer_path)}
        tokenizer = tokenizer_identity(
            "bpe",
            vocab_size=vocab_size,
            path=tokenizer_path,
        )
    else:
        raise ValueError(f"unsupported test tokenizer kind: {tokenizer_kind}")

    model_path = tmp_path / "model.yaml"
    model_mapping = {
        "name": "scorecard-fixture",
        "vocab_size": vocab_size,
        "d_model": 16,
        "embed_dim": 16,
        "n_layers": 1,
        "n_loops": 1,
        "n_heads": 1,
        "n_kv_heads": 1,
        "ffn_hidden": 32,
        "max_seq_len": max_seq_len,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
        "tie_embeddings": True,
        "dropout": 0.0,
        "qk_norm": False,
        "conv_kernel": 3,
        "layer_types": None,
    }
    model_path.write_text(yaml.safe_dump(model_mapping, sort_keys=True), encoding="utf-8")
    model_config = ModelConfig(**model_mapping)

    training_path = tmp_path / "sft.yaml"
    training_mapping = {
        "stage": "sft",
        "model_config": str(model_path),
        "data": {
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "tokenizer": tokenizer_config,
        },
        "runtime": {"resume": False, "seed": 7},
    }
    training_path.write_text(
        yaml.safe_dump(training_mapping, sort_keys=True),
        encoding="utf-8",
    )
    normalized_training = json.loads(json.dumps(training_mapping))
    normalized_training["runtime"].pop("resume")
    checkpoint_path = tmp_path / "latest.pt"
    model = LocalAgentLM(model_config)
    torch.save(
        {
            "cfg": model_config.__dict__,
            "state_dict": model.state_dict(),
            "stage": "sft",
            "step": 0,
            "tokenizer": {
                "kind": tokenizer_kind,
                "path": tokenizer_config.get("path"),
                "sha256": tokenizer["sha256"],
            },
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "lineage": {
                "version": 1,
                "stage": "sft",
                "config_sha256": canonical_sha256(normalized_training),
                "model_config_sha256": canonical_sha256(model_config.__dict__),
                "data_sha256": hashlib.sha256(b"training corpus").hexdigest(),
                "tokenizer_sha256": tokenizer["sha256"],
                "parent_checkpoint_sha256": hashlib.sha256(b"parent").hexdigest(),
                "git": {
                    "commit": "a" * 40,
                    "repository_sha256": hashlib.sha256(b"repository").hexdigest(),
                    "dirty": False,
                    "worktree_sha256": hashlib.sha256(b"worktree").hexdigest(),
                },
            },
        },
        checkpoint_path,
    )

    cases_path = tmp_path / "eval.jsonl"
    synth_config_path = tmp_path / "agent-eval.yaml"
    synth_config = {
        "out": str(cases_path),
        "n_samples": 12,
        "seed": 19,
        "level": 5,
        "split": "eval",
        "generator": {"backend": "deterministic_templates"},
        "complexity": {"multi_turn": 0},
        "irrelevance_fraction": 0.15,
        "verification": {"rule_based": True, "model_based": False},
    }
    synth_config_path.write_text(
        yaml.safe_dump(synth_config, sort_keys=True),
        encoding="utf-8",
    )
    synthesize(str(synth_config_path))
    manifest_path = cases_path.with_suffix(cases_path.suffix + ".manifest.json")

    scorecard_path = tmp_path / "scorecard.yaml"
    scorecard_config = {
        "kind": CONFIG_KIND,
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "training_config": str(training_path),
        "model_config": str(model_path),
        "tokenizer": tokenizer_config,
        "cases": {
            "path": str(cases_path),
            "manifest": str(manifest_path),
            "generator_config": str(synth_config_path),
            "expected_split": "eval",
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
        "generation": {"device": "cpu", "max_new_tokens": max_new_tokens},
    }
    scorecard_path.write_text(
        yaml.safe_dump(scorecard_config, sort_keys=True),
        encoding="utf-8",
    )
    return scorecard_path, training_path


def _stub_checkpoint_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def score(conversations, _predictor):
        return {
            "contract": {},
            "case_set": {
                "sha256": "b" * 64,
                "conversations": len(conversations),
                "assistant_decisions": len(conversations),
                "tool_decisions": 0,
                "no_tool_decisions": len(conversations),
            },
            "metrics": {},
            "by_category": {},
            "predictions": {
                "sha256": "c" * 64,
                "records": len(conversations),
                "raw_outputs_retained": False,
            },
        }

    monkeypatch.setattr(scorecard_module, "score_conversations", score)


def _rewrite_scorecard_context_limit(scorecard_path: Path, *, delta: int) -> int:
    config = yaml.safe_load(scorecard_path.read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.from_file(config["tokenizer"]["path"])
    cases_path = Path(config["cases"]["path"])
    conversations = [
        Conversation.from_json(line) for line in cases_path.read_text(encoding="utf-8").splitlines()
    ]
    max_new_tokens = config["generation"]["max_new_tokens"]
    statistics = prompt_token_statistics(
        conversations,
        tokenizer,
        max_new_tokens=max_new_tokens,
        model_max_seq_len=10**9,
    )
    required = statistics["required_context_tokens"]
    assert isinstance(required, int)
    requested_limit = required + delta
    assert requested_limit > 0

    model_path = Path(config["model_config"])
    model_mapping = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_mapping["max_seq_len"] = requested_limit
    model_path.write_text(
        yaml.safe_dump(model_mapping, sort_keys=True),
        encoding="utf-8",
    )
    model_config = ModelConfig(**model_mapping)

    checkpoint_path = Path(config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["cfg"] = model_config.__dict__
    checkpoint["lineage"]["model_config_sha256"] = canonical_sha256(model_config.__dict__)
    torch.save(checkpoint, checkpoint_path)
    return required


def test_checkpoint_scorecard_binds_all_artifacts_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scorecard_path, training_path = _scorecard_fixture(tmp_path)
    _stub_checkpoint_scoring(monkeypatch)

    result = run_scorecard(scorecard_path)

    assert result["kind"] == RESULT_KIND
    assert result["benchmark"]["official_bfcl"] is False
    assert (
        result["provenance"]["checkpoint"]["sha256"]
        == hashlib.sha256(
            Path(yaml.safe_load(scorecard_path.read_text())["checkpoint"]).read_bytes()
        ).hexdigest()
    )
    assert result["provenance"]["training_corpus"] == {
        "checkpoint_lineage_data_sha256": hashlib.sha256(b"training corpus").hexdigest(),
        "independently_reconstructed_by_scorecard": False,
    }
    assert (
        result["provenance"]["cases"]["case_set_sha256"]
        == result["scorecard"]["case_set"]["sha256"]
    )
    assert result["scorecard"]["case_set"]["conversations"] == 12
    evaluator = result["provenance"]["evaluator"]
    assert evaluator["source_tree"]["worktree_sha256"]
    assert set(evaluator["modules"]) == {
        "agent_scorecard",
        "prompt_contract",
        "stratified_eval_selector",
        "tool_eval",
    }
    for identity in evaluator["modules"].values():
        assert identity["sha256"] == hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
    runtime = result["provenance"]["generation"]
    assert runtime["resolved_device"] == "cpu"
    assert runtime["resolved_dtype"] == "fp32"
    assert runtime["torch_version"] == torch.__version__
    assert runtime["python_version"]
    assert runtime["platform"]
    assert runtime["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert runtime["truncation"] == "forbidden"
    assert runtime["generation_reserve_tokens"] == 96
    assert runtime["serial_generation_calls"] == 12
    assert runtime["serial_prefill_calls"] == 12
    assert runtime["generation_batch_size"] == 1
    assert runtime["maximum_non_eos_new_tokens"] == 1_152
    assert "reset before the next row" in runtime["kv_cache_scope"]
    assert "length-capped output is always inexact" in runtime["termination"]
    prompt_budget = runtime["prompt_budget"]
    assert prompt_budget["contract"] == OPENAI_FULL_CATALOG_V1
    assert prompt_budget["truncation"] == "forbidden"
    assert prompt_budget["generation_reserve_tokens"] == 96
    assert prompt_budget["assistant_decisions"] == 12
    assert prompt_budget["unique_catalogs"] == 1
    assert prompt_budget["required_context_tokens"] <= 16_384
    assert prompt_budget["fits_model_context"] is True
    assert prompt_budget["catalog_tokens"]["ordered_values_sha256"]
    assert prompt_budget["prompt_tokens"]["ordered_values_sha256"]
    gold_output_budget = runtime["gold_output_budget"]
    assert gold_output_budget["assistant_decisions"] == 12
    assert gold_output_budget["max_new_tokens"] == 96
    assert gold_output_budget["outputs_over_max_new_tokens"] == 0
    assert gold_output_budget["outputs_with_embedded_eos"] == 0
    assert gold_output_budget["fits_generation_budget"] is True
    assert gold_output_budget["gold_body_plus_eos_tokens"]["ordered_values_sha256"]
    unsigned = dict(result)
    self_hash = unsigned.pop("result_self_sha256")
    assert self_hash == hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()

    training = yaml.safe_load(training_path.read_text())
    training["runtime"]["seed"] = 8
    training_path.write_text(yaml.safe_dump(training, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="training config does not match checkpoint lineage"):
        run_scorecard(scorecard_path)


def test_checkpoint_scorecard_records_imported_bpe_runtime_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tokenizers_package = pytest.importorskip("tokenizers")
    scorecard_path, _training_path = _scorecard_fixture(
        tmp_path,
        tokenizer_kind="bpe",
    )
    _stub_checkpoint_scoring(monkeypatch)

    result = run_scorecard(scorecard_path)

    assert result["provenance"]["tokenizer"]["runtime_package"] == {
        "name": "tokenizers",
        "version": tokenizers_package.__version__,
    }


def test_checkpoint_scorecard_allows_exact_fit_and_rejects_one_short_without_truncation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scorecard_path, _training_path = _scorecard_fixture(tmp_path)
    _stub_checkpoint_scoring(monkeypatch)
    required = _rewrite_scorecard_context_limit(scorecard_path, delta=0)

    result = run_scorecard(scorecard_path)

    prompt_budget = result["provenance"]["generation"]["prompt_budget"]
    assert prompt_budget["required_context_tokens"] == required
    assert prompt_budget["model_max_seq_len"] == required
    assert prompt_budget["context_headroom_tokens"] == 0
    assert prompt_budget["fits_model_context"] is True
    assert prompt_budget["truncation"] == "forbidden"

    _rewrite_scorecard_context_limit(scorecard_path, delta=-1)
    with pytest.raises(ValueError, match="scorecard context budget exceeded") as error:
        run_scorecard(scorecard_path)

    assert '"context_headroom_tokens":-1' in str(error.value)
    assert '"fits_model_context":false' in str(error.value)
    assert '"truncation":"forbidden"' in str(error.value)


def test_checkpoint_scorecard_rejects_gold_outputs_that_cannot_fit_decode_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scorecard_path, _training_path = _scorecard_fixture(
        tmp_path,
        max_new_tokens=1,
    )
    _stub_checkpoint_scoring(monkeypatch)

    with pytest.raises(ValueError, match="scorecard gold output budget exceeded") as error:
        run_scorecard(scorecard_path)

    assert '"assistant_decisions":12' in str(error.value)
    assert '"fits_generation_budget":false' in str(error.value)
    assert '"outputs_over_max_new_tokens":12' in str(error.value)
    assert '"truncation":"forbidden"' in str(error.value)


def test_checkpoint_scorecard_rejects_missing_legacy_or_conflicting_prompt_contract(
    tmp_path: Path,
):
    scorecard_path, training_path = _scorecard_fixture(tmp_path)
    original_training = yaml.safe_load(training_path.read_text(encoding="utf-8"))

    missing_training = json.loads(json.dumps(original_training))
    missing_training["data"].pop("conversation_prompt_contract")
    training_path.write_text(
        yaml.safe_dump(missing_training, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data.conversation_prompt_contract must be"):
        run_scorecard(scorecard_path)

    legacy_training = json.loads(json.dumps(original_training))
    legacy_training["data"]["conversation_prompt_contract"] = "legacy"
    training_path.write_text(
        yaml.safe_dump(legacy_training, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="got 'legacy'"):
        run_scorecard(scorecard_path)

    training_path.write_text(
        yaml.safe_dump(original_training, sort_keys=True),
        encoding="utf-8",
    )
    config = yaml.safe_load(scorecard_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(config["checkpoint"])
    original_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    missing_checkpoint = dict(original_checkpoint)
    missing_checkpoint.pop("conversation_prompt_contract")
    torch.save(missing_checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="top-level conversation_prompt_contract"):
        run_scorecard(scorecard_path)

    conflicting_checkpoint = {
        **original_checkpoint,
        "conversation_prompt_contract": "legacy",
    }
    torch.save(conflicting_checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="expected 'openai_full_catalog_v1', got 'legacy'"):
        run_scorecard(scorecard_path)


def test_checkpoint_scorecard_rejects_byte_and_non_atomic_bpe_tokenizers(tmp_path: Path):
    byte_path, _training_path = _scorecard_fixture(
        tmp_path / "byte",
        tokenizer_kind="byte",
    )
    with pytest.raises(ValueError, match="requires a BPE tokenizer"):
        run_scorecard(byte_path)

    non_atomic_path, _training_path = _scorecard_fixture(
        tmp_path / "non-atomic",
        tokenizer_kind="bpe",
        atomic_bpe=False,
    )
    with pytest.raises(ValueError, match="requires an atomic EOS prompt boundary"):
        run_scorecard(non_atomic_path)


def test_checkpoint_scorecard_rejects_missing_and_conflicting_checkpoint_lineage(
    tmp_path: Path,
):
    scorecard_path, _training_path = _scorecard_fixture(tmp_path)
    config = yaml.safe_load(scorecard_path.read_text())
    checkpoint_path = Path(config["checkpoint"])
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    missing = dict(original)
    missing.pop("lineage")
    torch.save(missing, checkpoint_path)
    with pytest.raises(TypeError, match="checkpoint has no lineage"):
        run_scorecard(scorecard_path)

    conflicting = dict(original)
    conflicting["tokenizer"] = {**original["tokenizer"], "sha256": "f" * 64}
    torch.save(conflicting, checkpoint_path)
    with pytest.raises(ValueError, match="conflicting tokenizer identities"):
        run_scorecard(scorecard_path)

    wrong_architecture = dict(original)
    wrong_architecture["cfg"] = {**original["cfg"], "name": "different-model"}
    torch.save(wrong_architecture, checkpoint_path)
    with pytest.raises(ValueError, match="checkpoint architecture does not match"):
        run_scorecard(scorecard_path)


def test_scorecard_regular_reads_reject_symlink_oversize_and_open_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "target.yaml"
    target.write_text("kind: target\n", encoding="utf-8")
    symlink = tmp_path / "symlink.yaml"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="regular non-symlink"):
        scorecard_module._read_regular(
            symlink,
            label="test config",
            max_bytes=scorecard_module._MAX_CONFIG_BYTES,
        )

    oversized = tmp_path / "oversized.yaml"
    oversized.touch()
    os.truncate(oversized, scorecard_module._MAX_CONFIG_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds"):
        scorecard_module._read_regular(
            oversized,
            label="test config",
            max_bytes=scorecard_module._MAX_CONFIG_BYTES,
        )

    source = tmp_path / "source.yaml"
    replacement = tmp_path / "replacement.yaml"
    source.write_text("kind: original\n", encoding="utf-8")
    replacement.write_text("kind: replacement\n", encoding="utf-8")
    real_open = scorecard_module.os.open
    swapped = False

    def swap_path_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == source and not swapped:
            swapped = True
            os.replace(replacement, source)
        return descriptor

    monkeypatch.setattr(scorecard_module.os, "open", swap_path_after_open)
    with pytest.raises(RuntimeError, match="descriptor was being bound"):
        scorecard_module._read_regular(
            source,
            label="test config",
            max_bytes=scorecard_module._MAX_CONFIG_BYTES,
        )
    assert swapped is True
    assert source.read_text(encoding="utf-8") == "kind: replacement\n"


def test_checkpoint_loader_is_descriptor_bound_bounded_and_restricted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    marker = tmp_path / "pickle-executed.txt"
    unsafe_checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {"payload": _UnsafeCheckpointPayload(marker)},
        unsafe_checkpoint,
    )
    real_torch_load = scorecard_module.torch.load
    observed_weights_only: list[bool | None] = []

    def tracked_torch_load(*args, **kwargs):
        observed_weights_only.append(kwargs.get("weights_only"))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(scorecard_module.torch, "load", tracked_torch_load)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        scorecard_module._load_checkpoint(unsafe_checkpoint)
    assert observed_weights_only == [True]
    assert not marker.exists()

    oversized = tmp_path / "oversized.pt"
    oversized.touch()
    os.truncate(oversized, scorecard_module._MAX_CHECKPOINT_BYTES + 1)
    observed_weights_only.clear()
    with pytest.raises(ValueError, match="checkpoint exceeds"):
        scorecard_module._load_checkpoint(oversized)
    assert observed_weights_only == []

    safe = tmp_path / "safe.pt"
    raced_replacement = tmp_path / "raced-replacement.pt"
    torch.save({"stage": "safe"}, safe)
    torch.save(
        {"payload": _UnsafeCheckpointPayload(marker)},
        raced_replacement,
    )
    real_open = scorecard_module.os.open
    swapped = False

    def swap_checkpoint_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == safe and not swapped:
            swapped = True
            os.replace(raced_replacement, safe)
        return descriptor

    monkeypatch.setattr(scorecard_module.os, "open", swap_checkpoint_after_open)
    observed_weights_only.clear()
    with pytest.raises(RuntimeError, match="descriptor was being bound"):
        scorecard_module._load_checkpoint(safe)
    assert swapped is True
    assert observed_weights_only == []
    assert not marker.exists()


def test_scorecard_publish_rejects_symlink_nonregular_and_concurrent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b'{"result":"bound"}\n'
    output = tmp_path / "result.json"
    scorecard_module._publish(output, payload)
    scorecard_module._publish(output, payload)
    assert output.read_bytes() == payload
    with pytest.raises(RuntimeError, match="overwrite drifted"):
        scorecard_module._publish(output, b"different\n")

    target = tmp_path / "same-bytes.json"
    target.write_bytes(payload)
    symlink = tmp_path / "symlink-result.json"
    symlink.symlink_to(target)
    with pytest.raises(RuntimeError, match="not a regular file"):
        scorecard_module._publish(symlink, payload)

    directory = tmp_path / "directory-result.json"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="not a regular file"):
        scorecard_module._publish(directory, payload)

    raced = tmp_path / "raced-result.json"

    def race_with_symlink(_temporary, destination, **_kwargs):
        Path(destination).symlink_to(target)
        raise FileExistsError

    monkeypatch.setattr(scorecard_module.os, "link", race_with_symlink)
    with pytest.raises(RuntimeError, match="not a regular file"):
        scorecard_module._publish(raced, payload)
    assert raced.is_symlink()


def test_scorecard_publish_size_checks_and_path_binds_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b'{"result":"bound"}\n'
    oversized = tmp_path / "oversized-result.json"
    oversized.touch()
    os.truncate(oversized, scorecard_module._MAX_CONFIG_BYTES + 1)
    read_calls = 0

    def reject_unbounded_read(_descriptor, _size):
        nonlocal read_calls
        read_calls += 1
        raise AssertionError("oversized existing result must be rejected before reading")

    monkeypatch.setattr(scorecard_module.os, "read", reject_unbounded_read)
    with pytest.raises(RuntimeError, match="overwrite drifted"):
        scorecard_module._publish(oversized, payload)
    assert read_calls == 0

    monkeypatch.undo()
    existing = tmp_path / "existing-result.json"
    replacement = tmp_path / "replacement-result.json"
    existing.write_bytes(payload)
    replacement.write_bytes(b'{"result":"replacement"}\n')
    real_open = scorecard_module.os.open
    swapped = False

    def swap_existing_after_open(path, flags, *args, **kwargs):
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == existing and not swapped:
            swapped = True
            os.replace(replacement, existing)
        return descriptor

    monkeypatch.setattr(scorecard_module.os, "open", swap_existing_after_open)
    with pytest.raises(RuntimeError, match="descriptor was being bound"):
        scorecard_module._publish(existing, payload)
    assert swapped is True
    assert existing.read_bytes() == b'{"result":"replacement"}\n'


def test_scorecard_publish_detects_content_and_pathname_swaps_before_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b'{"result":"bound"}\n'
    tampered = b'{"result":"tampered"}\n'
    real_link = scorecard_module.os.link
    content_output = tmp_path / "content-swap.json"
    content_swapped = False

    def swap_content_then_link(source, destination, **kwargs):
        nonlocal content_swapped
        if Path(destination) == content_output and not content_swapped:
            content_swapped = True
            Path(source).write_bytes(tampered)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(scorecard_module.os, "link", swap_content_then_link)
    with pytest.raises(RuntimeError, match="failed post-link verification"):
        scorecard_module._publish(content_output, payload)
    assert content_swapped is True
    assert not content_output.exists()

    pathname_output = tmp_path / "pathname-swap.json"
    attacker = tmp_path / "attacker-source.json"
    attacker.write_bytes(tampered)
    swapped_source: Path | None = None

    def swap_pathname_then_link(source, destination, **kwargs):
        nonlocal swapped_source
        if Path(destination) == pathname_output and swapped_source is None:
            swapped_source = Path(source)
            os.replace(attacker, source)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(scorecard_module.os, "link", swap_pathname_then_link)
    with pytest.raises(RuntimeError, match="failed post-link verification"):
        scorecard_module._publish(pathname_output, payload)
    assert swapped_source is not None
    assert swapped_source.read_bytes() == tampered
    assert not pathname_output.exists()


def test_scorecard_publish_never_unlinks_raced_destination_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b'{"result":"bound"}\n'
    tampered = b'{"result":"tampered"}\n'
    replacement = b'{"result":"concurrent"}\n'
    output = tmp_path / "result.json"
    replacement_source = tmp_path / "replacement.json"
    replacement_source.write_bytes(replacement)
    real_link = scorecard_module.os.link
    real_rename = scorecard_module.os.rename
    link_tampered = False
    destination_replaced = False

    def tamper_then_link(source, destination, **kwargs):
        nonlocal link_tampered
        if Path(destination) == output and not link_tampered:
            link_tampered = True
            Path(source).write_bytes(tampered)
        return real_link(source, destination, **kwargs)

    def replace_before_rollback(source, destination):
        nonlocal destination_replaced
        if Path(source) == output and not destination_replaced:
            destination_replaced = True
            os.replace(replacement_source, output)
        return real_rename(source, destination)

    monkeypatch.setattr(scorecard_module.os, "link", tamper_then_link)
    monkeypatch.setattr(scorecard_module.os, "rename", replace_before_rollback)

    with pytest.raises(RuntimeError, match="failed post-link verification"):
        scorecard_module._publish(output, payload)

    assert link_tampered is True
    assert destination_replaced is True
    assert output.read_bytes() == replacement
