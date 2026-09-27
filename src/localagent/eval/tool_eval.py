"""AST-based tool-call evaluation for the internal BFCL-style scorecard.

This module does not implement, or claim compatibility with, the official Berkeley Function
Calling Leaderboard.  It applies BFCL-style ideas to LocalAgent's canonical ``Conversation``
schema: strict tool-call parsing, name/argument decomposition, parallel calls, abstention, and
teacher-forced-history multi-turn scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import (
    canonical_json_bytes,
    conversation_semantic_sha256,
)
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    RESERVED_PROMPT_MARKERS,
    assistant_training_examples,
    render_agent_decode_prompt,
    render_function_catalog,
    schema_matches,
    validate_json_schema,
    validate_tool_catalog,
)
from localagent.data.schema import Conversation, ToolCall, ToolSpec
from localagent.model.tokenizer import BPE_EOS, TOOL_CALL_CLOSE, TOOL_CALL_OPEN

__all__ = [
    "AssistantPrediction",
    "ParsedToolOutput",
    "arguments_schema_valid",
    "gold_output_token_statistics",
    "irrelevance_correct",
    "match_calls",
    "parse_tool_output",
    "prompt_token_statistics",
    "render_agent_decode_prompt",
    "render_function_catalog",
    "score_conversations",
    "score_dataset",
]

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_TOOL_MARKERS = (TOOL_CALL_OPEN, TOOL_CALL_CLOSE)
_NON_ENVELOPE_RESERVED_MARKERS = tuple(
    marker for marker in RESERVED_PROMPT_MARKERS if marker not in _TOOL_MARKERS
)
_FINISH_REASONS = frozenset({"caller_complete", "eos", "length"})


@dataclass(frozen=True)
class AssistantPrediction:
    """One assistant body plus its generation termination state.

    ``caller_complete`` is the compatibility contract for predictors that already provide a
    complete assistant body.  Autoregressive predictors must report ``eos`` or ``length`` so a
    syntactically complete prefix that merely hit its token cap cannot receive exact-match credit.
    """

    text: str
    finish_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("assistant prediction text must be text")
        if self.finish_reason not in _FINISH_REASONS:
            raise ValueError(
                "assistant prediction finish_reason must be one of "
                f"{sorted(_FINISH_REASONS)}, got {self.finish_reason!r}"
            )

    @property
    def complete(self) -> bool:
        """Whether the output is a declared complete assistant body."""

        return self.finish_reason != "length"

    @property
    def terminated_by_eos(self) -> bool:
        """Whether token generation observed EOS rather than a caller-supplied body."""

        return self.finish_reason == "eos"


@dataclass(frozen=True)
class ParsedToolOutput:
    """Strict parse result used by the scorecard.

    ``calls`` contains only individually valid call objects.  ``format_valid`` is false if the
    envelope has unmatched markers, non-whitespace outside call blocks, duplicate JSON keys,
    non-finite JSON values, extra object keys, an invalid call shape, or any reserved prompt
    framing marker is spilled into the assistant body.
    """

    calls: tuple[ToolCall, ...]
    format_valid: bool
    tool_syntax_present: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class _AssistantCase:
    case_id: str
    conversation_index: int
    message_index: int
    prompt: str
    expected_body: str
    expected_calls: tuple[ToolCall, ...]
    expected_text: str
    tools: tuple[ToolSpec, ...]
    category: str
    multi_turn: bool


Predictor = Callable[[str, Sequence[ToolSpec]], str | AssistantPrediction]


def match_calls(pred: list[ToolCall], ref: list[ToolCall]) -> bool:
    """Order-insensitive exact match on tool name and normalized arguments."""

    return sorted(call.normalized() for call in pred) == sorted(call.normalized() for call in ref)


def irrelevance_correct(pred: list[ToolCall]) -> bool:
    """For irrelevance samples, the model is correct iff it emitted no tool call."""

    return len(pred) == 0


def _strict_json_object(payload: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid tool-call JSON") from error
    if not isinstance(value, dict):
        raise TypeError("tool-call payload must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("tool-call payload must contain finite JSON values") from error
    return value


def _reserved_json_marker(value: Any) -> str | None:
    """Return the first reserved marker after JSON unescaping, including in object keys."""

    if isinstance(value, str):
        return next((marker for marker in RESERVED_PROMPT_MARKERS if marker in value), None)
    if isinstance(value, Mapping):
        for key, child in value.items():
            marker = _reserved_json_marker(key)
            if marker is not None:
                return marker
            marker = _reserved_json_marker(child)
            if marker is not None:
                return marker
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            marker = _reserved_json_marker(child)
            if marker is not None:
                return marker
    return None


def parse_tool_output(text: str) -> ParsedToolOutput:
    """Parse the exact LocalAgent tool-call envelope without repairing malformed output."""

    if not isinstance(text, str):
        raise TypeError("model output must be text")
    reserved_errors = [
        f"reserved_prompt_marker:{marker}"
        for marker in _NON_ENVELOPE_RESERVED_MARKERS
        if marker in text
    ]
    tool_syntax_present = any(marker in text for marker in _TOOL_MARKERS)
    if not tool_syntax_present:
        return ParsedToolOutput(
            calls=(),
            format_valid=not reserved_errors,
            tool_syntax_present=False,
            errors=tuple(reserved_errors),
        )

    calls: list[ToolCall] = []
    errors = reserved_errors
    cursor = 0
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        errors.append("unmatched_tool_call_marker")
    elif any(text.count(marker) != len(matches) for marker in _TOOL_MARKERS):
        errors.append("unmatched_or_nested_tool_call_marker")
    for match in matches:
        if text[cursor : match.start()].strip():
            errors.append("content_outside_tool_call")
        cursor = match.end()
        try:
            value = _strict_json_object(match.group(1))
            reserved_json_marker = _reserved_json_marker(value)
            if reserved_json_marker is not None:
                raise ValueError(
                    f"tool-call JSON contains reserved prompt marker {reserved_json_marker!r}"
                )
            if set(value) != {"name", "arguments"}:
                raise ValueError("tool-call object must contain exactly name and arguments")
            name = value["name"]
            arguments = value["arguments"]
            if not isinstance(name, str) or not name:
                raise ValueError("tool-call name must be non-empty text")
            if not isinstance(arguments, dict):
                raise TypeError("tool-call arguments must be a JSON object")
            calls.append(ToolCall(name=name, arguments=arguments))
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    if text[cursor:].strip():
        errors.append("content_outside_tool_call")
    if not calls:
        errors.append("no_valid_tool_call")
    return ParsedToolOutput(
        calls=tuple(calls),
        format_valid=not errors,
        tool_syntax_present=True,
        errors=tuple(errors),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rate(correct: int, total: int) -> dict[str, int | float | None]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def _validate_schema_definition(schema: Any, *, label: str) -> dict[str, Any]:
    """Compatibility wrapper for the shared recursive JSON Schema validator."""

    return validate_json_schema(schema, label=label)


def _schema_matches(value: Any, schema: Mapping[str, Any]) -> bool:
    """Compatibility wrapper for the shared schema matcher."""

    return schema_matches(value, schema)


def arguments_schema_valid(arguments: dict[str, Any], parameters: Mapping[str, Any]) -> bool:
    """Validate arguments using the shared prompt-contract schema semantics."""

    schema = validate_json_schema(parameters, label="tool parameters")
    if schema.get("type") != "object":
        raise ValueError("tool parameters must declare type='object'")
    return schema_matches(arguments, schema)


def _tool_registry(tools: Sequence[ToolSpec], *, label: str) -> dict[str, ToolSpec]:
    """Compatibility wrapper for the shared complete-catalog validator."""

    return validate_tool_catalog(tools, label=label)


def _calls_schema_valid(calls: Sequence[ToolCall], tools: Sequence[ToolSpec]) -> bool:
    registry = _tool_registry(tools, label="prediction registry")
    return bool(calls) and all(
        call.name in registry
        and isinstance(call.arguments, dict)
        and schema_matches(call.arguments, registry[call.name].parameters)
        for call in calls
    )


def _category(conversation: Conversation) -> str:
    for key in ("category", "kind", "group"):
        value = conversation.meta.get(key)
        if isinstance(value, str) and value:
            return value
    return "unlabeled"


def _assistant_cases(conversations: Sequence[Conversation]) -> tuple[_AssistantCase, ...]:
    cases: list[_AssistantCase] = []
    observed_ids: set[str] = set()
    for conversation_index, conversation in enumerate(conversations):
        examples = assistant_training_examples(conversation)
        multi_turn = len(examples) > 1
        for example in examples:
            message_index = example.message_index
            message = conversation.messages[message_index]
            prompt = example.prompt
            category = _category(conversation)
            descriptor = {
                "conversation_semantic_sha256": conversation_semantic_sha256(conversation),
                "message_index": message_index,
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "category": category,
                "multi_turn": multi_turn,
                "expected_calls": [
                    {"name": call.name, "arguments": call.arguments} for call in message.tool_calls
                ],
                "expected_text_sha256": _sha256(message.content.encode("utf-8")),
            }
            case_id = _sha256(canonical_json_bytes(descriptor))
            if case_id in observed_ids:
                raise ValueError(f"duplicate assistant evaluation case {case_id}")
            observed_ids.add(case_id)
            cases.append(
                _AssistantCase(
                    case_id=case_id,
                    conversation_index=conversation_index,
                    message_index=message_index,
                    prompt=prompt,
                    expected_body=example.body,
                    expected_calls=tuple(message.tool_calls),
                    expected_text=message.content,
                    tools=tuple(conversation.tools),
                    category=category,
                    multi_turn=multi_turn,
                )
            )
    if not cases:
        raise ValueError("tool-eval dataset contains no assistant decisions")
    return tuple(cases)


def _token_length_summary(lengths: Sequence[int]) -> dict[str, int | str]:
    if not lengths:
        raise ValueError("token-length summary requires at least one value")
    ordered = sorted(lengths)

    def nearest_rank(percent: int) -> int:
        index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
        return ordered[index]

    return {
        "minimum": ordered[0],
        "p50_nearest_rank": nearest_rank(50),
        "p95_nearest_rank": nearest_rank(95),
        "p99_nearest_rank": nearest_rank(99),
        "maximum": ordered[-1],
        "total": sum(ordered),
        "ordered_values_sha256": _sha256(canonical_json_bytes(ordered)),
    }


def prompt_token_statistics(
    conversations: Sequence[Conversation],
    tokenizer: Any,
    *,
    max_new_tokens: int,
    model_max_seq_len: int,
) -> dict[str, Any]:
    """Return deterministic full-catalog prompt lengths and the required context budget."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        isinstance(model_max_seq_len, bool)
        or not isinstance(model_max_seq_len, int)
        or model_max_seq_len < 1
    ):
        raise ValueError("model_max_seq_len must be a positive integer")
    cases = _assistant_cases(conversations)
    prompt_lengths = [len(tokenizer.encode(case.prompt)) for case in cases]
    catalog_cache: dict[str, int] = {}
    catalog_lengths: list[int] = []
    for conversation in conversations:
        catalog = render_function_catalog(conversation.tools) + BPE_EOS
        length = catalog_cache.get(catalog)
        if length is None:
            length = len(tokenizer.encode(catalog))
            catalog_cache[catalog] = length
        catalog_lengths.append(length)
    if not catalog_lengths:
        raise ValueError("scorecard dataset contains no conversations")
    required = max(prompt_lengths) + max_new_tokens
    return {
        "contract": OPENAI_FULL_CATALOG_V1,
        "token_counting": (
            "configured tokenizer encode(add_eos=False); nearest-rank percentiles; "
            "complete function catalog and its EOS boundary included in every prompt"
        ),
        "truncation": "forbidden",
        "generation_reserve_tokens": max_new_tokens,
        "conversations": len(conversations),
        "unique_catalogs": len(catalog_cache),
        "assistant_decisions": len(cases),
        "catalog_tokens": _token_length_summary(catalog_lengths),
        "prompt_tokens": _token_length_summary(prompt_lengths),
        "max_new_tokens": max_new_tokens,
        "required_context_tokens": required,
        "model_max_seq_len": model_max_seq_len,
        "context_headroom_tokens": model_max_seq_len - required,
        "fits_model_context": required <= model_max_seq_len,
    }


def gold_output_token_statistics(
    conversations: Sequence[Conversation],
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Tokenize every gold assistant body plus EOS and report a bounded budget audit."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    eos_id = getattr(tokenizer, "eos_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("tokenizer.eos_id must be a non-negative integer")

    cases = _assistant_cases(conversations)
    lengths: list[int] = []
    over_budget_case_ids: list[str] = []
    embedded_eos_case_ids: list[str] = []
    at_limit = 0
    for case in cases:
        body_ids = tokenizer.encode(case.expected_body)
        if (
            not isinstance(body_ids, Sequence)
            or isinstance(body_ids, (str, bytes, bytearray))
            or any(isinstance(token, bool) or not isinstance(token, int) for token in body_ids)
        ):
            raise TypeError("tokenizer.encode must return a sequence of integer token IDs")
        if eos_id in body_ids:
            embedded_eos_case_ids.append(case.case_id)
        body_plus_eos_tokens = len(body_ids) + 1
        lengths.append(body_plus_eos_tokens)
        if body_plus_eos_tokens > max_new_tokens:
            over_budget_case_ids.append(case.case_id)
        elif body_plus_eos_tokens == max_new_tokens:
            at_limit += 1

    return {
        "token_counting": (
            "configured tokenizer encode(add_eos=False) for each exact shared assistant body, "
            "followed by exactly one eos_id; nearest-rank percentiles"
        ),
        "truncation": "forbidden",
        "assistant_decisions": len(cases),
        "max_new_tokens": max_new_tokens,
        "gold_body_plus_eos_tokens": _token_length_summary(lengths),
        "outputs_at_limit": at_limit,
        "outputs_over_max_new_tokens": len(over_budget_case_ids),
        "outputs_with_embedded_eos": len(embedded_eos_case_ids),
        "over_budget_case_ids_sha256": _sha256(canonical_json_bytes(sorted(over_budget_case_ids))),
        "embedded_eos_case_ids_sha256": _sha256(
            canonical_json_bytes(sorted(embedded_eos_case_ids))
        ),
        "fits_generation_budget": not over_budget_case_ids and not embedded_eos_case_ids,
    }


def _counter_intersection_size(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left & right).values())


def _normalize_prediction(value: str | AssistantPrediction, *, case_id: str) -> AssistantPrediction:
    if isinstance(value, str):
        return AssistantPrediction(text=value, finish_reason="caller_complete")
    if isinstance(value, AssistantPrediction):
        return value
    raise TypeError(f"predictor returned an unsupported output for case {case_id}")


def score_conversations(
    conversations: Sequence[Conversation],
    predictor: Predictor,
) -> dict[str, Any]:
    """Score predictions over canonical conversations using gold-history decode prefixes.

    The tool-only multi-turn metrics use teacher-forced gold prior history.  An episode is correct
    only when every reference tool-call step in that multi-assistant conversation is an exact
    order-insensitive AST match.  No-tool turns are not part of that metric, and this is not a
    free-running environment evaluation.
    """

    cases = _assistant_cases(conversations)
    format_correct = 0
    schema_correct = 0
    schema_total = 0
    tool_format_correct = 0
    tool_schema_correct = 0
    action_correct = 0
    response_correct = 0
    tool_decisions = 0
    no_tool_decisions = 0
    name_case_correct = 0
    whole_call_correct = 0
    name_reference_calls = 0
    name_predicted_calls = 0
    name_intersection = 0
    argument_intersection = 0
    abstention_correct = 0
    text_exact_correct = 0
    parallel_correct = 0
    parallel_total = 0
    multi_argument_correct = 0
    multi_argument_total = 0
    completed_predictions = 0
    eos_terminated_predictions = 0
    finish_reason_counts: Counter[str] = Counter()
    teacher_forced_tool_steps: dict[int, list[bool]] = defaultdict(list)
    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    prediction_records: list[dict[str, Any]] = []

    case_descriptors = [
        {
            "case_id": case.case_id,
            "conversation_index": case.conversation_index,
            "message_index": case.message_index,
            "prompt_sha256": _sha256(case.prompt.encode("utf-8")),
            "category": case.category,
            "multi_turn": case.multi_turn,
            "reference_calls": [
                {"name": call.name, "arguments": call.arguments} for call in case.expected_calls
            ],
            "reference_text_sha256": _sha256(case.expected_text.encode("utf-8")),
        }
        for case in cases
    ]

    for case in cases:
        prediction = _normalize_prediction(
            predictor(case.prompt, case.tools),
            case_id=case.case_id,
        )
        generated = prediction.text
        parsed = parse_tool_output(generated)
        complete_format_valid = prediction.complete and parsed.format_valid
        strict_calls = list(parsed.calls) if complete_format_valid else []
        completed_predictions += int(prediction.complete)
        eos_terminated_predictions += int(prediction.terminated_by_eos)
        finish_reason_counts[prediction.finish_reason] += 1
        format_correct += int(complete_format_valid)

        tool_attempt = parsed.tool_syntax_present
        schema_valid = complete_format_valid and _calls_schema_valid(strict_calls, case.tools)
        if tool_attempt:
            schema_total += 1
            schema_correct += int(schema_valid)

        expected = list(case.expected_calls)
        is_tool = bool(expected)
        strict_nonempty_tool_format_valid = complete_format_valid and bool(strict_calls)
        whole_exact = complete_format_valid and match_calls(strict_calls, expected)
        name_exact = complete_format_valid and Counter(
            call.name for call in strict_calls
        ) == Counter(call.name for call in expected)
        abstained = complete_format_valid and not strict_calls and not parsed.tool_syntax_present
        text_exact = abstained and generated == case.expected_text

        if is_tool:
            tool_decisions += 1
            tool_format_correct += int(strict_nonempty_tool_format_valid)
            tool_schema_correct += int(schema_valid)
            name_case_correct += int(name_exact)
            whole_call_correct += int(whole_exact)
            action_correct += int(whole_exact)
            response_correct += int(whole_exact)

            expected_names = Counter(call.name for call in expected)
            predicted_names = Counter(call.name for call in strict_calls)
            expected_asts = Counter(call.normalized() for call in expected)
            predicted_asts = Counter(call.normalized() for call in strict_calls)
            name_reference_calls += len(expected)
            name_predicted_calls += len(strict_calls)
            name_intersection += _counter_intersection_size(expected_names, predicted_names)
            argument_intersection += _counter_intersection_size(expected_asts, predicted_asts)

            if len(expected) > 1:
                parallel_total += 1
                parallel_correct += int(whole_exact)
            if any(len(call.arguments) > 1 for call in expected):
                multi_argument_total += 1
                multi_argument_correct += int(whole_exact)
            if case.multi_turn:
                teacher_forced_tool_steps[case.conversation_index].append(whole_exact)
        else:
            no_tool_decisions += 1
            abstention_correct += int(abstained)
            text_exact_correct += int(text_exact)
            action_correct += int(abstained)
            response_correct += int(text_exact)

        category = category_counts[case.category]
        category[0] += int(whole_exact if is_tool else abstained)
        category[1] += 1
        category[2] += int(is_tool)
        category[3] += int(not is_tool)
        prediction_records.append(
            {
                "case_id": case.case_id,
                "prediction_sha256": _sha256(generated.encode("utf-8")),
                "finish_reason": prediction.finish_reason,
                "generation_complete": prediction.complete,
                "terminated_by_eos": prediction.terminated_by_eos,
                "parser_format_valid": parsed.format_valid,
                "format_valid": complete_format_valid,
                "schema_valid": schema_valid if tool_attempt else None,
                "name_exact": name_exact if is_tool else None,
                "whole_call_exact": whole_exact if is_tool else None,
                "abstained": abstained if not is_tool else None,
            }
        )

    name_precision = name_intersection / name_predicted_calls if name_predicted_calls else None
    name_recall = name_intersection / name_reference_calls if name_reference_calls else None
    name_f1 = (
        2 * name_precision * name_recall / (name_precision + name_recall)
        if name_precision is not None
        and name_recall is not None
        and name_precision + name_recall > 0
        else None
    )
    teacher_forced_tool_step_values = [
        value for values in teacher_forced_tool_steps.values() for value in values
    ]
    teacher_forced_tool_episode_values = [
        all(values) for values in teacher_forced_tool_steps.values() if values
    ]

    return {
        "contract": {
            "name": "LocalAgent BFCL-style internal agent scorecard",
            "official_bfcl": False,
            "external_native_benchmark": False,
            "decode": "greedy checkpoint generation supplied by the caller",
            "tool_ast": "order-insensitive exact name plus canonical JSON arguments",
            "format": (
                "strict LocalAgent <tool_call> envelope; no repair and no non-whitespace "
                "outside call blocks; reserved role/catalog/tool-response/EOS marker spill is "
                "invalid; format credit requires a complete generation"
            ),
            "termination": (
                "token generation must terminate on EOS; length-capped output is always "
                "inexact; bare-string predictors declare a caller-complete assistant body"
            ),
            "schema": (
                "fail-closed recursive subset: object/properties/required/additionalProperties, "
                "array/items, scalar types, enum, and non-validating string format annotations"
            ),
            "tool_format_validity_on_tool_decisions": (
                "reference tool-call decisions only; generation must be complete and the strict "
                "envelope must contain at least one parsed tool call"
            ),
            "schema_validity_on_tool_decisions": (
                "reference tool-call decisions only; requires strict non-empty tool format, "
                "registered tool names, and recursive argument-schema matches for every call"
            ),
            "tool_name": "case-exact name multiset plus micro precision/recall/F1",
            "arguments": (
                "exact normalized argument objects conditional on a matched tool name; "
                "extra/missing calls are penalized by whole-call exactness"
            ),
            "abstention": "strictly valid output with no tool-call syntax",
            "no_tool_text_exact": "literal Unicode string equality with no whitespace normalization",
            "prompt": (
                "deterministic complete OpenAI-style function catalog followed by role-preserving "
                "message history; EOS follows the catalog and every prior assistant body"
            ),
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "teacher_forced_tool_multi_turn": (
                "gold prior history; reference tool-call decisions only; no-tool decisions "
                "excluded; not a free-running environment rollout"
            ),
        },
        "case_set": {
            "sha256": _sha256(canonical_json_bytes(case_descriptors)),
            "conversations": len(conversations),
            "assistant_decisions": len(cases),
            "tool_decisions": tool_decisions,
            "no_tool_decisions": no_tool_decisions,
        },
        "metrics": {
            "action_exact": _rate(action_correct, len(cases)),
            "assistant_response_exact": _rate(response_correct, len(cases)),
            "generation_completion": _rate(completed_predictions, len(cases)),
            "format_validity": _rate(format_correct, len(cases)),
            "schema_validity_on_tool_attempts": _rate(schema_correct, schema_total),
            "tool_format_validity_on_tool_decisions": _rate(
                tool_format_correct,
                tool_decisions,
            ),
            "schema_validity_on_tool_decisions": _rate(
                tool_schema_correct,
                tool_decisions,
            ),
            "tool_name": {
                "case_exact": _rate(name_case_correct, tool_decisions),
                "matched_reference_calls": name_intersection,
                "reference_calls": name_reference_calls,
                "predicted_calls": name_predicted_calls,
                "precision": name_precision,
                "recall": name_recall,
                "f1": name_f1,
            },
            "arguments": {
                "exact_calls_given_matched_name": _rate(
                    argument_intersection,
                    name_intersection,
                ),
            },
            "whole_call_exact": _rate(whole_call_correct, tool_decisions),
            "abstention": _rate(abstention_correct, no_tool_decisions),
            "no_tool_text_exact": _rate(text_exact_correct, no_tool_decisions),
            "parallel_whole_call_exact": _rate(parallel_correct, parallel_total),
            "multi_argument_whole_call_exact": _rate(
                multi_argument_correct,
                multi_argument_total,
            ),
            "teacher_forced_tool_multi_turn": {
                "tool_step_exact": _rate(
                    sum(teacher_forced_tool_step_values),
                    len(teacher_forced_tool_step_values),
                ),
                "tool_episode_exact": _rate(
                    sum(teacher_forced_tool_episode_values),
                    len(teacher_forced_tool_episode_values),
                ),
            },
        },
        "by_category": {
            label: {
                "action_exact": _rate(counts[0], counts[1]),
                "tool_decisions": counts[2],
                "no_tool_decisions": counts[3],
            }
            for label, counts in sorted(category_counts.items())
        },
        "predictions": {
            "sha256": _sha256(canonical_json_bytes(prediction_records)),
            "records": len(prediction_records),
            "finish_reasons": dict(sorted(finish_reason_counts.items())),
            "complete": completed_predictions,
            "terminated_by_eos": eos_terminated_predictions,
            "raw_outputs_retained": False,
        },
    }


def score_dataset(
    jsonl_path: str | Path,
    model: Any,
    tokenizer: Any | None = None,
    *,
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Score a Conversation JSONL with a predictor or autoregressive LocalAgent model.

    This convenience function binds and reports the raw JSONL identity, but it cannot prove a
    sidecar or checkpoint lineage by itself.  Use ``localagent.eval.agent_scorecard`` for the
    fail-closed artifact runner.

    With ``tokenizer=None``, ``model`` must be a callable accepting ``(prompt, tools)`` and
    returning text or :class:`AssistantPrediction`.  A bare string declares a complete assistant
    body.  With a tokenizer, greedy ``inference.generate`` is used and EOS-versus-length
    termination is propagated into exact-match scoring.
    """

    path = Path(jsonl_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"tool-eval JSONL is missing or not a regular non-symlink file: {path}")
    payload = path.read_bytes()
    if not payload:
        raise ValueError("tool-eval JSONL must not be empty")
    conversations: list[Conversation] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValueError(f"tool-eval JSONL line {line_number} must end in exactly one LF")
        try:
            conversations.append(Conversation.from_json(line[:-1].decode("utf-8")))
        except (KeyError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"tool-eval JSONL line {line_number} is not a Conversation") from error

    if tokenizer is None:
        if not callable(model):
            raise TypeError("model must be a (prompt, tools) predictor when tokenizer is omitted")
        predictor: Predictor = model
        gold_output_budget = None
    else:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        gold_output_budget = gold_output_token_statistics(
            conversations,
            tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if not gold_output_budget["fits_generation_budget"]:
            raise ValueError(
                "tool-eval gold output budget exceeded: "
                + json.dumps(
                    gold_output_budget,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        from localagent.inference.generate import generate

        def predictor(prompt: str, tools: Sequence[ToolSpec]) -> AssistantPrediction:
            if not prompt.startswith(render_function_catalog(tools) + BPE_EOS):
                raise RuntimeError("tool-eval prompt is not bound to its function catalog")
            generated, stats = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            new_tokens = stats.new_tokens
            if (
                isinstance(new_tokens, bool)
                or not isinstance(new_tokens, int)
                or not 0 <= new_tokens <= max_new_tokens
            ):
                raise RuntimeError("generation returned an invalid new-token count")
            return AssistantPrediction(
                text=generated,
                finish_reason="eos" if new_tokens < max_new_tokens else "length",
            )

    result = score_conversations(conversations, predictor)
    result["dataset"] = {
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "manifest_verified": False,
    }
    if gold_output_budget is not None:
        result["gold_output_budget"] = gold_output_budget
    return result
