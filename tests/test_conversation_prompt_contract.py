from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from localagent.data.agent_synth import Sample
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    RESERVED_PROMPT_MARKERS,
    SYSTEM,
    TOOL_CATALOG_CLOSE,
    TOOL_CATALOG_OPEN,
    assistant_training_turns,
    render_agent_decode_prompt,
    render_function_catalog,
    render_tool_calls,
    schema_matches,
    validate_json_schema,
)
from localagent.data.render import (
    IGNORE,
    CatalogTokenCache,
    LazyCatalogTokenRow,
    conversation_row_token_counts,
    render_conversation,
    render_conversation_rows,
    render_conversation_rows_batch,
    shifted_token_counts,
    token_row_length,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import (
    ASSISTANT,
    BPE_EOS,
    TOOL,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    USER,
    ByteTokenizer,
    train_bpe,
)
from localagent.train.midtrain import (
    ConversationDataset,
    MixtureSource,
    ScheduledMixture,
    midtrain,
)
from localagent.train.sft import _evaluate_conversations, sft
from localagent.train.stage_budget import build_stage_budget_plan
from localagent.train.stage_data import single_turn_samples
from localagent.train.stage_sampling import encode_with_value_span, prepare_sft_data


@pytest.fixture(scope="module")
def bpe_tokenizer(tmp_path_factory):
    path = tmp_path_factory.mktemp("prompt-contract-tokenizer") / "tokenizer.json"
    documents = [
        (
            "catalog system user assistant tool response inspect satellite telemetry "
            "window samples query result acknowledged alpha beta gamma"
        )
    ]
    return train_bpe(documents, path, vocab_size=300, min_frequency=1)


def _parameters(*, extra_property: bool = False) -> dict:
    properties = {
        "satellite": {"type": "string"},
        "window": {
            "type": "object",
            "properties": {
                "samples": {"type": "array", "items": {"type": "number"}},
                "start": {"type": "string"},
            },
            "required": ["start", "samples"],
            "additionalProperties": False,
        },
    }
    if extra_property:
        properties["trace"] = {"type": "boolean"}
    return {
        "type": "object",
        "properties": properties,
        "required": ["satellite", "window"],
        "additionalProperties": False,
    }


def _tool(
    name: str = "lookup_satellite_telemetry_v9",
    *,
    description: str = "Fetch télémétrie for a satellite and sampling window.",
    extra_property: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=_parameters(extra_property=extra_property),
    )


def _conversation(
    *,
    system: str = "Never invent telemetry.",
    tool: ToolSpec | None = None,
) -> Conversation:
    selected = tool or _tool()
    arguments = {
        "satellite": "Asteria",
        "window": {"start": "now", "samples": [1, 2.5]},
    }
    return Conversation(
        tools=[selected],
        messages=[
            Message(role=Role.system, content=system),
            Message(role=Role.user, content="Inspect Asteria."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name=selected.name, arguments=arguments)],
            ),
            Message(role=Role.tool, tool_response='{"samples":[1,2.5]}'),
            Message(role=Role.user, content="Summarize."),
            Message(role=Role.assistant, content="Telemetry is stable."),
        ],
    )


def _subsequence_start(values: list[int], needle: list[int]) -> int:
    for index in range(len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def test_full_catalog_json_is_complete_canonical_and_order_preserving() -> None:
    first_schema = _parameters()
    second_schema = {
        "additionalProperties": False,
        "required": ["satellite", "window"],
        "properties": {
            "window": {
                "required": ["start", "samples"],
                "properties": {
                    "start": {"type": "string"},
                    "samples": {"items": {"type": "number"}, "type": "array"},
                },
                "additionalProperties": False,
                "type": "object",
            },
            "satellite": {"type": "string"},
        },
        "type": "object",
    }
    tools = [
        ToolSpec("zeta_lookup", "Zeta description.", first_schema),
        ToolSpec("alpha_lookup", "Alpha description.", second_schema),
    ]

    rendered = render_function_catalog(tools)
    repeated = render_function_catalog(tools)
    payload = json.loads(rendered.removeprefix(TOOL_CATALOG_OPEN).removesuffix(TOOL_CATALOG_CLOSE))

    assert rendered == repeated
    assert rendered == render_function_catalog(
        [
            ToolSpec("zeta_lookup", "Zeta description.", second_schema),
            ToolSpec("alpha_lookup", "Alpha description.", first_schema),
        ]
    )
    assert [entry["function"]["name"] for entry in payload["tools"]] == [
        "zeta_lookup",
        "alpha_lookup",
    ]
    assert payload["tools"][0]["function"] == {
        "name": "zeta_lookup",
        "description": "Zeta description.",
        "parameters": first_schema,
    }


def test_full_contract_is_sensitive_to_system_catalog_name_description_and_schema(
    bpe_tokenizer,
) -> None:
    baseline = render_conversation_rows(
        _conversation(),
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
    )[0]
    variants = [
        _conversation(system="Never invent or round telemetry."),
        _conversation(tool=_tool(name="fetch_orbit_samples")),
        _conversation(tool=_tool(description="Fetch exact orbital samples.")),
        _conversation(tool=_tool(extra_property=True)),
    ]

    baseline_ids, _ = baseline.materialize()
    for variant in variants:
        variant_ids, _ = render_conversation_rows(
            variant,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
        )[0].materialize()
        assert variant_ids != baseline_ids


def test_full_contract_rejects_raw_reserved_marker_collision(bpe_tokenizer) -> None:
    tools = [_tool()]
    injected_messages = [Message(role=Role.system, content=f"policy{USER}same")]
    separated_messages = [
        Message(role=Role.system, content="policy"),
        Message(role=Role.user, content="same"),
    ]
    manually_framed_injection = (
        render_function_catalog(tools) + BPE_EOS + SYSTEM + "policy" + USER + "same" + ASSISTANT
    )

    assert manually_framed_injection == render_agent_decode_prompt(separated_messages, tools)
    with pytest.raises(ValueError, match="contains reserved prompt marker"):
        render_agent_decode_prompt(injected_messages, tools)

    legacy_conversation = _conversation(system=f"policy{USER}same")
    assert render_conversation_rows(
        legacy_conversation,
        ByteTokenizer(),
        prompt_contract="legacy",
    ) == [render_conversation(legacy_conversation, ByteTokenizer())]


@pytest.mark.parametrize("marker", RESERVED_PROMPT_MARKERS)
def test_full_contract_rejects_every_reserved_marker_substring(
    bpe_tokenizer,
    marker: str,
) -> None:
    conversation = _conversation(system=f"policy-prefix{marker}suffix")
    with pytest.raises(ValueError, match="contains reserved prompt marker"):
        render_conversation_rows(
            conversation,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
        )


def test_full_contract_checks_every_rendered_string_surface(bpe_tokenizer) -> None:
    cases: dict[str, Conversation] = {}

    message_content = _conversation()
    message_content.messages[4].content = f"unsafe {BPE_EOS} user content"
    cases["message content"] = message_content

    tool_response = _conversation()
    tool_response.messages[3].tool_response = f"unsafe {ASSISTANT} response"
    cases["tool response"] = tool_response

    cases["tool name"] = _conversation(tool=_tool(name=f"lookup{TOOL}telemetry"))
    cases["tool description"] = _conversation(
        tool=_tool(description=f"unsafe {TOOL_CALL_OPEN} description")
    )

    call_name = _conversation()
    call_name.messages[2].tool_calls[0].name = f"lookup{TOOL_RESPONSE_CLOSE}telemetry"
    cases["tool call name"] = call_name

    argument_key = _conversation()
    argument_key.messages[2].tool_calls[0].arguments["window"][f"unsafe{SYSTEM}"] = "value"
    cases["argument key"] = argument_key

    argument_value = _conversation()
    argument_value.messages[2].tool_calls[0].arguments["window"]["samples"].append(
        [f"unsafe{TOOL_CATALOG_OPEN}value"]
    )
    cases["nested argument value"] = argument_value

    schema_key_tool = _tool()
    schema_key_tool.parameters[f"unsafe{TOOL}keyword"] = {"type": "string"}
    cases["schema key"] = _conversation(tool=schema_key_tool)

    property_name_tool = _tool()
    property_name_tool.parameters["properties"][f"unsafe{TOOL_CALL_CLOSE}property"] = {
        "type": "string"
    }
    cases["schema property name"] = _conversation(tool=property_name_tool)

    schema_description_tool = _tool()
    schema_description_tool.parameters["properties"]["satellite"]["description"] = (
        f"unsafe {TOOL_CATALOG_CLOSE} description"
    )
    cases["schema description"] = _conversation(tool=schema_description_tool)

    schema_enum_tool = _tool()
    schema_enum_tool.parameters["properties"]["satellite"]["enum"] = [
        {"nested": ["safe", f"unsafe{BPE_EOS}enum"]}
    ]
    cases["nested schema enum"] = _conversation(tool=schema_enum_tool)

    for conversation in cases.values():
        with pytest.raises(ValueError, match="contains reserved prompt marker"):
            render_conversation_rows(
                conversation,
                bpe_tokenizer,
                prompt_contract=OPENAI_FULL_CATALOG_V1,
            )


def test_full_contract_accepts_valid_nested_schema_and_argument_json(bpe_tokenizer) -> None:
    choice = {
        "mode": "safe",
        "labels": ["alpha", {"deep": "omega"}],
    }
    tool = ToolSpec(
        name="inspect_nested_payload",
        description="Inspect a nested payload without interpreting framing-like prose.",
        parameters={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "Nested request payload.",
                    "properties": {
                        "choice": {
                            "description": "A recursively structured exact choice.",
                            "enum": [
                                choice,
                                {"mode": "fast", "labels": ["beta"]},
                            ],
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["north", "south"],
                            },
                        },
                    },
                    "required": ["choice", "tags"],
                    "additionalProperties": False,
                }
            },
            "required": ["payload"],
            "additionalProperties": False,
        },
    )
    conversation = Conversation(
        tools=[tool],
        messages=[
            Message(role=Role.system, content="Keep nested values exact."),
            Message(role=Role.user, content="Inspect literal <|user> and <tool_call text."),
            Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(
                        name=tool.name,
                        arguments={
                            "payload": {
                                "choice": choice,
                                "tags": ["north", "south"],
                            }
                        },
                    )
                ],
            ),
            Message(
                role=Role.tool,
                tool_response='{"details":{"labels":["alpha"]},"status":"ok"}',
            ),
            Message(role=Role.assistant, content="Nested payload is valid."),
        ],
    )

    rows = render_conversation_rows(
        conversation,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=2048,
    )
    assert len(rows) == 2
    assert render_function_catalog(conversation.tools)
    assert render_tool_calls(conversation.messages[2].tool_calls)


def test_mask_boundaries_and_multiturn_eos_prompt_tokens_match_exactly(
    bpe_tokenizer,
) -> None:
    conversation = _conversation()
    rows = render_conversation_rows(
        conversation,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
    )
    turns = assistant_training_turns(conversation)

    assert len(rows) == len(turns) == 2
    for row, turn in zip(rows, turns, strict=True):
        ids, labels = row.materialize()
        prompt_text = render_agent_decode_prompt(
            conversation.messages[: turn.message_index],
            conversation.tools,
        )
        prompt_ids = bpe_tokenizer.encode(prompt_text)
        body_ids = bpe_tokenizer.encode(turn.body) + [bpe_tokenizer.eos_id]
        assert list(row.prompt_ids) == prompt_ids
        assert ids == prompt_ids + body_ids
        assert labels == [IGNORE] * len(prompt_ids) + body_ids

    second_prompt = list(rows[1].prompt_ids)
    exact_second_prompt = render_agent_decode_prompt(
        conversation.messages[:5],
        conversation.tools,
    )
    assert second_prompt == bpe_tokenizer.encode(exact_second_prompt)
    assert BPE_EOS in exact_second_prompt
    prior_body_with_eos = bpe_tokenizer.encode(
        render_tool_calls(conversation.messages[2].tool_calls) + BPE_EOS
    )
    assert _subsequence_start(second_prompt, prior_body_with_eos) >= 0
    assert second_prompt != bpe_tokenizer.encode(exact_second_prompt.replace(BPE_EOS, ""))

    without_system = Conversation(
        tools=[_tool()],
        messages=[
            Message(role=Role.user, content="Inspect without a system message."),
            Message(role=Role.assistant, content="No-system result."),
        ],
    )
    no_system_row = render_conversation_rows(
        without_system,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
    )[0]
    no_system_prompt = render_agent_decode_prompt(
        without_system.messages[:1],
        without_system.tools,
    )
    assert list(no_system_row.prompt_ids) == bpe_tokenizer.encode(no_system_prompt)


def test_full_catalog_batched_counts_match_materialized_rows_in_source_order(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    conversations = [
        _conversation(),
        _conversation(system="Keep exact orbital samples."),
    ]
    expected = [
        shifted_token_counts(row)
        for conversation in conversations
        for row in render_conversation_rows(
            conversation,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
        )
    ]
    original_encode_batch = bpe_tokenizer.encode_batch
    batch_sizes: list[int] = []

    def tracking_encode_batch(texts, add_eos=False):
        values = list(texts)
        batch_sizes.append(len(values))
        return original_encode_batch(values, add_eos=add_eos)

    monkeypatch.setattr(bpe_tokenizer, "encode_batch", tracking_encode_batch)
    cache = CatalogTokenCache(bpe_tokenizer)

    assert conversation_row_token_counts(
        conversations,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
        catalog_cache=cache,
        batch_size=2,
    ) == expected
    assert batch_sizes == [4, 4]
    assert cache.unique_catalogs == 1


def test_full_catalog_batched_rows_match_scalar_source_accounting_and_prefix_identity(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    unicode_empty = Conversation(
        tools=[
            _tool(
                name="inspect_unicode_window",
                description="Inspect a multilingual orbital window.",
            )
        ],
        messages=[
            Message(role=Role.user, content="Vérifie 궤도 🛰️."),
            Message(role=Role.assistant, content=""),
        ],
    )
    conversations = [
        _conversation(),
        unicode_empty,
        _conversation(system="Keep exact orbital samples."),
    ]
    expected = [
        row
        for conversation in conversations
        for row in render_conversation_rows(
            conversation,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=2048,
        )
    ]
    expected_texts = [
        text
        for conversation in conversations
        for turn in assistant_training_turns(conversation)
        for text in (turn.prompt_suffix, turn.body)
    ]
    original_encode_batch = bpe_tokenizer.encode_batch
    encoded_text_batches: list[list[str]] = []

    def tracking_encode_batch(texts, add_eos=False):
        values = list(texts)
        encoded_text_batches.append(values)
        return original_encode_batch(values, add_eos=add_eos)

    monkeypatch.setattr(bpe_tokenizer, "encode_batch", tracking_encode_batch)
    cache = CatalogTokenCache(bpe_tokenizer)
    actual = render_conversation_rows_batch(
        conversations,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=2048,
        catalog_cache=cache,
        batch_size=2,
    )

    assert [len(batch) for batch in encoded_text_batches] == [4, 4, 2]
    assert [text for batch in encoded_text_batches for text in batch] == expected_texts
    assert len(actual) == len(expected) == 5
    assert all(isinstance(row, LazyCatalogTokenRow) for row in actual)
    assert [row.materialize() for row in actual] == [row.materialize() for row in expected]
    assert [row.message_index for row in actual] == [2, 5, 1, 2, 5]
    assert actual[2].body_ids == (bpe_tokenizer.eos_id,)
    assert actual[0].prompt_ids.shared_prefix is actual[1].prompt_ids.shared_prefix
    assert actual[0].prompt_ids.shared_prefix is actual[3].prompt_ids.shared_prefix
    assert actual[3].prompt_ids.shared_prefix is actual[4].prompt_ids.shared_prefix
    assert actual[0].prompt_ids.shared_prefix is not actual[2].prompt_ids.shared_prefix
    assert cache.unique_catalogs == 2

    prepared = prepare_sft_data(
        [],
        bpe_tokenizer,
        conversations=conversations,
        sample_sources=[],
        conversation_sources=["first", "unicode", "third"],
        decay_samples=None,
        decay_sample_sources=None,
        lr_schedule="wsd",
        max_seq_len=2048,
        joint_tool_head=False,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        decay_conversations=[unicode_empty, _conversation()],
        decay_conversation_sources=["decay_unicode", "decay_default"],
    )
    assert [source for _, source in prepared.main_entries] == [
        "first",
        "first",
        "unicode",
        "third",
        "third",
    ]
    assert [row.materialize() for row, _ in prepared.main_entries] == [
        row.materialize() for row in expected
    ]
    assert [source for _, source in prepared.decay_entries] == [
        "decay_unicode",
        "decay_default",
        "decay_default",
    ]
    main_counts = [shifted_token_counts(row) for row, _ in prepared.main_entries]
    main_accounting = prepared.dataset_accounting["main"]
    assert main_accounting["input_tokens"] == sum(count[0] for count in main_counts)
    assert main_accounting["loss_tokens"] == sum(count[1] for count in main_counts)
    assert main_accounting["sources"]["first"]["rows"] == 2
    assert main_accounting["sources"]["unicode"]["rows"] == 1
    assert main_accounting["sources"]["third"]["rows"] == 2


def test_batched_renderer_rejects_misaligned_tokenizer_output(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    original_encode_batch = bpe_tokenizer.encode_batch

    def misaligned_encode_batch(texts, add_eos=False):
        encoded = original_encode_batch(texts, add_eos=add_eos)
        return encoded[:-1]

    monkeypatch.setattr(bpe_tokenizer, "encode_batch", misaligned_encode_batch)
    with pytest.raises(RuntimeError, match="batch output does not align"):
        render_conversation_rows_batch(
            [_conversation()],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
        )
    with pytest.raises(RuntimeError, match="batch output does not align"):
        conversation_row_token_counts(
            [_conversation()],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
        )


def test_batched_renderer_scalar_replay_preserves_earliest_failure(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    first = _conversation(system="EARLY scalar suffix failure.")
    second = _conversation(system="LATER scalar suffix failure.")
    original_encode = bpe_tokenizer.encode

    def scalar_encode(text, add_eos=False):
        if "EARLY" in text:
            raise ValueError("earliest scalar encoding error")
        if "LATER" in text:
            raise ValueError("later scalar encoding error")
        return original_encode(text, add_eos=add_eos)

    def failed_encode_batch(texts, add_eos=False):
        raise ValueError("unordered tokenizer batch error")

    monkeypatch.setattr(bpe_tokenizer, "encode", scalar_encode)
    monkeypatch.setattr(bpe_tokenizer, "encode_batch", failed_encode_batch)
    with pytest.raises(ValueError, match="earliest scalar encoding error"):
        render_conversation_rows_batch(
            [first, second],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=2048,
            batch_size=16,
        )


def test_batched_render_and_count_replay_earlier_body_before_later_suffix_failure(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    first = Conversation(
        tools=[],
        messages=[
            Message(role=Role.user, content="First prompt."),
            Message(role=Role.assistant, content="EARLY_BODY"),
        ],
    )
    later = Conversation(
        tools=[],
        messages=[
            Message(role=Role.user, content="LATER_SUFFIX"),
            Message(role=Role.assistant, content="Later body."),
        ],
    )
    original_encode = bpe_tokenizer.encode

    def ordered_scalar_encode(text, add_eos=False):
        if "EARLY_BODY" in text:
            raise ValueError("earlier scalar body error")
        if "LATER_SUFFIX" in text:
            raise ValueError("later scalar suffix error")
        return original_encode(text, add_eos=add_eos)

    def later_batch_failure(texts, add_eos=False):
        assert any("LATER_SUFFIX" in text for text in texts)
        raise ValueError("later unordered batch error")

    monkeypatch.setattr(bpe_tokenizer, "encode", ordered_scalar_encode)
    monkeypatch.setattr(bpe_tokenizer, "encode_batch", later_batch_failure)

    for consume in (
        lambda: render_conversation_rows_batch(
            [first, later],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
            batch_size=16,
        ),
        lambda: conversation_row_token_counts(
            [first, later],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
            batch_size=16,
        ),
    ):
        with pytest.raises(ValueError, match="earlier scalar body error"):
            consume()


def test_batched_render_and_count_replay_overlength_before_later_suffix_failure(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    first = _conversation()
    first_row = render_conversation_rows(
        first,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
    )[0]
    limit = token_row_length(first_row) - 1
    later = Conversation(
        tools=[],
        messages=[
            Message(role=Role.user, content="LATER_SUFFIX"),
            Message(role=Role.assistant, content="Later body."),
        ],
    )
    original_encode = bpe_tokenizer.encode

    def ordered_scalar_encode(text, add_eos=False):
        if "LATER_SUFFIX" in text:
            raise ValueError("later scalar suffix error")
        return original_encode(text, add_eos=add_eos)

    def later_batch_failure(texts, add_eos=False):
        assert any("LATER_SUFFIX" in text for text in texts)
        raise ValueError("later unordered batch error")

    monkeypatch.setattr(bpe_tokenizer, "encode", ordered_scalar_encode)
    monkeypatch.setattr(bpe_tokenizer, "encode_batch", later_batch_failure)

    for consume in (
        lambda: render_conversation_rows_batch(
            [first, later],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
            batch_size=16,
        ),
        lambda: conversation_row_token_counts(
            [first, later],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
            batch_size=16,
        ),
    ):
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            consume()


def test_batched_render_and_count_replay_generator_iteration_failure(
    bpe_tokenizer,
    monkeypatch,
) -> None:
    conversations = [_conversation(), _conversation(system="Second policy.")]
    expected_rows = [
        row
        for conversation in conversations
        for row in render_conversation_rows(
            conversation,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=1024,
        )
    ]
    expected_counts = [shifted_token_counts(row) for row in expected_rows]
    original_encode_batch = bpe_tokenizer.encode_batch

    def lazy_failed_encode_batch(texts, add_eos=False):
        encoded = original_encode_batch(texts, add_eos=add_eos)

        def output():
            yield encoded[0]
            raise ValueError("lazy tokenizer batch failure")

        return output()

    monkeypatch.setattr(bpe_tokenizer, "encode_batch", lazy_failed_encode_batch)
    actual_rows = render_conversation_rows_batch(
        conversations,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
        batch_size=16,
    )
    assert [row.materialize() for row in actual_rows] == [
        row.materialize() for row in expected_rows
    ]
    assert conversation_row_token_counts(
        conversations,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=1024,
        batch_size=16,
    ) == expected_counts


def test_batched_renderer_flushes_earlier_overlength_before_later_schema_error(
    bpe_tokenizer,
) -> None:
    first = _conversation()
    first_row = render_conversation_rows(
        first,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
    )[0]
    limit = token_row_length(first_row) - 1
    later_invalid = _conversation()
    later_invalid.messages[2].tool_calls[0].name = "missing_tool"

    with pytest.raises(ValueError) as scalar_error:
        render_conversation_rows(
            first,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
        )
    with pytest.raises(ValueError) as batch_error:
        render_conversation_rows_batch(
            [first, later_invalid],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
            batch_size=16,
        )
    assert str(batch_error.value) == str(scalar_error.value)
    assert "exceeds max_seq_len" in str(batch_error.value)


def test_full_contract_rejects_byte_tokenizer_and_overlength_without_truncating(
    bpe_tokenizer,
) -> None:
    conversation = _conversation()
    with pytest.raises(ValueError, match="requires a BPE tokenizer"):
        render_conversation_rows(
            conversation,
            ByteTokenizer(),
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=4096,
        )

    row = render_conversation_rows(
        conversation,
        bpe_tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
    )[0]
    limit = token_row_length(row) - 1
    with pytest.raises(
        ValueError,
        match="exceeds max_seq_len and cannot be truncated",
    ) as render_error:
        render_conversation_rows(
            conversation,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
        )
    with pytest.raises(
        ValueError,
        match="exceeds max_seq_len and cannot be truncated",
    ) as count_error:
        conversation_row_token_counts(
            [conversation],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
            max_seq_len=limit,
        )
    assert str(count_error.value) == str(render_error.value)
    assert "assistant_message_index=2" in str(count_error.value)
    with pytest.raises(ValueError, match="exceeds max_seq_len and cannot be truncated"):
        ConversationDataset(
            [conversation],
            bpe_tokenizer,
            limit,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )


def test_legacy_renderer_and_left_truncation_remain_exact() -> None:
    tokenizer = ByteTokenizer()
    conversation = _conversation()
    expected = render_conversation(conversation, tokenizer)
    default_rows = render_conversation_rows(conversation, tokenizer)
    explicit_rows = render_conversation_rows(
        conversation,
        tokenizer,
        prompt_contract="legacy",
    )

    assert default_rows == [expected]
    assert explicit_rows == [expected]
    assert render_conversation_rows(
        conversation,
        tokenizer,
        max_seq_len=17,
    ) == [(expected[0][-17:], expected[1][-17:])]
    assert conversation_row_token_counts(
        [conversation],
        tokenizer,
        max_seq_len=17,
    ) == [shifted_token_counts((expected[0][-17:], expected[1][-17:]))]
    conversations = [conversation, _conversation(system="A second trajectory.")]
    assert render_conversation_rows_batch(
        conversations,
        tokenizer,
        prompt_contract="legacy",
        max_seq_len=17,
        batch_size=1,
    ) == [
        row
        for item in conversations
        for row in render_conversation_rows(
            item,
            tokenizer,
            prompt_contract="legacy",
            max_seq_len=17,
        )
    ]

    changed = _conversation(
        system="A different ignored legacy system message.",
        tool=_tool(description="A different ignored legacy tool description."),
    )
    assert render_conversation(changed, tokenizer) == expected


def test_schema_validation_rejects_unsupported_and_nested_bool_number_enum_mismatch() -> None:
    with pytest.raises(ValueError, match="unsupported JSON Schema keywords"):
        validate_json_schema(
            {"oneOf": [{"type": "string"}, {"type": "number"}]},
            label="fixture",
        )

    nested_array_schema = validate_json_schema(
        {"enum": [[[True]], [[{"enabled": False}]]]},
        label="nested enum",
    )
    assert schema_matches([[True]], nested_array_schema)
    assert not schema_matches([[1]], nested_array_schema)
    assert schema_matches([[{"enabled": False}]], nested_array_schema)
    assert not schema_matches([[{"enabled": 0}]], nested_array_schema)
    assert validate_json_schema(
        {"type": "number", "description": "Distance in screen lengths."},
        label="described scalar",
    ) == {"type": "number", "description": "Distance in screen lengths."}
    with pytest.raises(ValueError, match="description must be text"):
        validate_json_schema(
            {"type": "number", "description": 3},
            label="invalid description",
        )


def test_reference_calls_fail_closed_on_unknown_or_schema_invalid_tool(
    bpe_tokenizer,
) -> None:
    unknown = _conversation()
    unknown.messages[2].tool_calls[0].name = "missing_tool"
    with pytest.raises(ValueError, match="references unknown tool"):
        render_conversation_rows(
            unknown,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
    with pytest.raises(ValueError, match="references unknown tool"):
        conversation_row_token_counts(
            [unknown],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
        )

    invalid = _conversation()
    invalid.messages[2].tool_calls[0].arguments["window"]["samples"] = [True]
    with pytest.raises(ValueError, match="arguments violate the schema"):
        render_conversation_rows(
            invalid,
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
    with pytest.raises(ValueError, match="arguments violate the schema"):
        conversation_row_token_counts(
            [invalid],
            bpe_tokenizer,
            prompt_contract=OPENAI_FULL_CATALOG_V1,
        )


def test_catalog_tokens_are_interned_across_conversations_and_sft_rows(
    bpe_tokenizer,
) -> None:
    conversations = [
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content=f"Inspect satellite {index}."),
                Message(role=Role.assistant, content=f"Result {index}."),
            ],
        )
        for index in range(100)
    ]
    dataset = ConversationDataset(
        conversations,
        bpe_tokenizer,
        1024,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )

    assert dataset.catalog_token_cache.unique_catalogs == 1
    assert all(isinstance(row, LazyCatalogTokenRow) for row in dataset.rows)
    shared_prefix = dataset.rows[0].prompt_ids.shared_prefix
    assert all(row.prompt_ids.shared_prefix is shared_prefix for row in dataset.rows)
    assert dataset.catalog_token_cache.retained_token_count == len(shared_prefix)

    prepared = prepare_sft_data(
        [],
        bpe_tokenizer,
        conversations=conversations,
        sample_sources=[],
        conversation_sources=["fixture"] * len(conversations),
        decay_samples=None,
        decay_sample_sources=None,
        lr_schedule="cosine",
        max_seq_len=1024,
        joint_tool_head=False,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    prepared_rows = [row for row, _ in prepared.main_entries]
    prepared_prefix = prepared_rows[0].prompt_ids.shared_prefix
    assert prepared.catalog_token_cache.unique_catalogs == 1
    assert all(row.prompt_ids.shared_prefix is prepared_prefix for row in prepared_rows)
    assert prepared.catalog_token_cache.retained_token_count == len(prepared_prefix)


def test_full_catalog_pointer_fast_path_matches_exact_contextual_encoding(
    bpe_tokenizer,
) -> None:
    conversation = Conversation(
        tools=[
            ToolSpec(
                name="search_docs",
                description="Search documents.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
        ],
        messages=[
            Message(role=Role.user, content="Find Asteria."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="search_docs", arguments={"query": "Asteria"})],
            ),
        ],
    )
    prepared = prepare_sft_data(
        [
            Sample(
                category="fixture",
                group="text",
                prompt="Say hello.",
                kind="text",
                target="Hello.",
            )
        ],
        bpe_tokenizer,
        conversations=[conversation],
        sample_sources=["fixture"],
        conversation_sources=["fixture"],
        decay_samples=None,
        decay_sample_sources=None,
        lr_schedule="cosine",
        max_seq_len=1024,
        joint_tool_head=True,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )

    turn = assistant_training_turns(conversation)[0]
    exact_prompt = render_function_catalog(conversation.tools) + BPE_EOS + turn.prompt_suffix
    exact_ids, exact_span = encode_with_value_span(
        bpe_tokenizer,
        exact_prompt,
        "Asteria",
        1024,
    )
    context_ids, _, _, gold_start, gold_end = prepared.multi_turn_items[0]

    assert list(context_ids) == exact_ids
    assert exact_span is not None
    assert (gold_start, gold_end) == exact_span


def _model_config(vocab_size: int) -> ModelConfig:
    config = ModelConfig(
        name="catalog-contract-test",
        vocab_size=vocab_size,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=1024,
        dropout=0.0,
    )
    config.assert_within_budget()
    return config


def test_sft_budget_runner_and_heldout_share_selected_contract(
    tmp_path: Path,
    bpe_tokenizer,
) -> None:
    train_rows = [
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content="Inspect train satellite."),
                Message(role=Role.assistant, content="Train result."),
            ],
        ),
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content="Inspect second train satellite."),
                Message(role=Role.assistant, content="Second train result."),
            ],
        ),
    ]
    eval_rows = [
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content="Inspect held-out satellite."),
                Message(role=Role.assistant, content="Held-out result."),
            ],
        )
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(
        "".join(conversation.to_json() + "\n" for conversation in train_rows),
        encoding="utf-8",
    )
    eval_path.write_text(
        "".join(conversation.to_json() + "\n" for conversation in eval_rows),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    bpe_tokenizer.save(tokenizer_path)
    model_config = _model_config(bpe_tokenizer.vocab_size)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(model_config.__dict__, sort_keys=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "sft",
                "model_config": str(model_path),
                "init_from": str(tmp_path / "unused-parent.pt"),
                "data": {
                    "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                    "conversations": [str(train_path)],
                    "eval_conversations": [str(eval_path)],
                    "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                    "seq_len": model_config.max_seq_len,
                    "shuffle": False,
                },
                "heads": {
                    "joint_tool_pointer": False,
                    "train_route_head": False,
                    "train_dense_selector": False,
                },
                "schedule": {"type": "cosine", "total_steps": 2},
                "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                "runtime": {"seed": 7},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = build_stage_budget_plan(config_path)
    samples = single_turn_samples(train_rows)
    _, _, _, metrics = sft(
        LocalAgentLM(model_config),
        samples,
        bpe_tokenizer,
        steps=2,
        batch_size=1,
        warmup=0,
        conversations=train_rows,
        sample_sources=[str(train_path)] * len(samples),
        conversation_sources=[str(train_path)] * len(train_rows),
        shuffle=False,
        max_seq_len=model_config.max_seq_len,
        seed=7,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        return_metrics=True,
        log=lambda *_: None,
    )
    heldout = _evaluate_conversations(
        LocalAgentLM(model_config),
        eval_rows,
        bpe_tokenizer,
        max_seq_len=model_config.max_seq_len,
        batch_size=1,
        device="cpu",
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )

    assert plan["data"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert plan["schedule"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert metrics["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert (
        plan["planned"]["horizon_totals"]["input_tokens"]
        == (metrics["token_accounting"]["input_tokens"])
    )
    assert (
        plan["planned"]["horizon_totals"]["loss_tokens"]
        == (metrics["token_accounting"]["loss_tokens"])
    )
    assert heldout["rows"] == 1


def test_midtrain_budget_runner_and_heldout_share_selected_contract(
    tmp_path: Path,
    bpe_tokenizer,
) -> None:
    train_rows = [
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content=f"Midtrain request {index}."),
                Message(role=Role.assistant, content=f"Midtrain answer {index}."),
            ],
        )
        for index in range(2)
    ]
    eval_rows = [
        Conversation(
            tools=[_tool()],
            messages=[
                Message(role=Role.user, content="Midtrain held-out request."),
                Message(role=Role.assistant, content="Midtrain held-out answer."),
            ],
        )
    ]
    train_path = tmp_path / "midtrain.jsonl"
    eval_path = tmp_path / "midtrain-eval.jsonl"
    train_path.write_text(
        "".join(conversation.to_json() + "\n" for conversation in train_rows),
        encoding="utf-8",
    )
    eval_path.write_text(
        "".join(conversation.to_json() + "\n" for conversation in eval_rows),
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    bpe_tokenizer.save(tokenizer_path)
    model_config = _model_config(bpe_tokenizer.vocab_size)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(model_config.__dict__, sort_keys=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "midtrain.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "midtrain",
                "model_config": str(model_path),
                "init_from": str(tmp_path / "unused-parent.pt"),
                "data": {
                    "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
                    "mixture": {"unit": "loss_tokens"},
                    "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                    "sources": [
                        {
                            "name": "agent",
                            "type": "conversations",
                            "path": str(train_path),
                            "weight": 1.0,
                        }
                    ],
                    "eval_sources": [
                        {
                            "name": "agent-eval",
                            "type": "conversations",
                            "path": str(eval_path),
                        }
                    ],
                },
                "schedule": {"type": "cosine", "total_steps": 2},
                "batch": {"micro_batch_size": 1, "grad_accum_steps": 1},
                "evaluation": {"batches_per_source": 1, "batch_size": 1, "seed": 17},
                "runtime": {"seed": 5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = build_stage_budget_plan(config_path)
    train_dataset = ConversationDataset(
        train_rows,
        bpe_tokenizer,
        model_config.max_seq_len,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    eval_dataset = ConversationDataset(
        eval_rows,
        bpe_tokenizer,
        model_config.max_seq_len,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    _, metrics = midtrain(
        LocalAgentLM(model_config),
        ScheduledMixture(
            [MixtureSource("agent", train_dataset, 1.0, 1.0)],
            unit="loss_tokens",
        ),
        steps=2,
        batch_size=1,
        warmup=0,
        lr_schedule="cosine",
        seed=5,
        eval_sources=[MixtureSource("agent-eval", eval_dataset, 1.0, 1.0)],
        eval_batches=1,
        eval_batch_size=1,
        eval_seed=17,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        return_metrics=True,
        log=lambda *_: None,
    )

    assert plan["data"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert plan["schedule"]["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert metrics["conversation_prompt_contract"] == OPENAI_FULL_CATALOG_V1
    assert metrics["heldout_eval"]["contract"]["conversation_prompt_contract"] == (
        OPENAI_FULL_CATALOG_V1
    )
    assert (
        plan["planned"]["horizon_totals"]["input_tokens"]
        == (metrics["token_accounting"]["input_tokens"])
    )
    assert (
        plan["planned"]["horizon_totals"]["loss_tokens"]
        == (metrics["token_accounting"]["loss_tokens"])
    )
