"""Deterministic, quota-aware ordering of canonical assistant decisions.

The ordering is intentionally a data-layer primitive.  It does not tokenize, sample, or mutate
conversations.  Every assistant message receives exactly one composite stratum, identified by the
signals that matter for agent-data coverage, and appears exactly once in the returned epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from localagent.data.prompt_contract import (
    RESERVED_PROMPT_MARKERS,
    FunctionCatalogCache,
    schema_matches,
)
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec

ORDERING_CONTRACT = "canonical-assistant-decision-quota-order-v1"
QUOTA_SAMPLING_MODE = "quota_stratified_no_replacement_v1"

DecisionKey = tuple[int, int]

_META_TEXT_FIELDS = ("category", "group", "kind")
_PRIMITIVE_ORDER = ("null", "boolean", "integer", "number", "string")
_FAILURE_MARKERS = ("failed", "error", "traceback")
_SUCCESS_MARKERS = ("all tests passed", "passed", "succeeded", "success")


def _assert_json_value(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} contains a non-string object key")
            _assert_json_value(item, label=f"{label}[{key!r}]")
        return
    raise TypeError(f"{label} contains a non-JSON value of type {type(value).__name__}")


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    _assert_json_value(value, label=label)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return encoded.encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{label} must be canonical finite JSON") from exc


def _sha256(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label)).hexdigest()


def _assert_no_reserved_markers(value: Any, *, label: str) -> None:
    if isinstance(value, str):
        for marker in RESERVED_PROMPT_MARKERS:
            if marker in value:
                raise ValueError(f"{label} contains reserved prompt marker {marker!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_reserved_markers(item, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_reserved_markers(key, label=f"{label} key")
            _assert_no_reserved_markers(item, label=f"{label}[{key!r}]")


@dataclass(frozen=True)
class DecisionStratum:
    """One disjoint assistant-decision stratum.

    ``assistant_ordinal`` is one-based within the conversation.  ``tool_names`` preserves call
    order, while ``argument_primitives`` is a sorted set of scalar JSON kinds observed recursively
    in the arguments.
    """

    category: str | None
    group: str | None
    meta_kind: str | None
    plan_len: int | None
    conversation_shape: str
    assistant_ordinal: int
    decision_kind: str
    tool_names: tuple[str, ...]
    argument_primitives: tuple[str, ...]
    has_enum_argument: bool
    has_multiple_arguments: bool
    recovery_relevant: bool
    schema_relevant: bool
    grounded_followup_relevant: bool

    def canonical_payload(self) -> dict[str, Any]:
        """Return the versioned JSON representation used for identity and ordering."""

        return {
            "argument_primitives": list(self.argument_primitives),
            "assistant_ordinal": self.assistant_ordinal,
            "category": self.category,
            "conversation_shape": self.conversation_shape,
            "decision_kind": self.decision_kind,
            "grounded_followup_relevant": self.grounded_followup_relevant,
            "group": self.group,
            "has_enum_argument": self.has_enum_argument,
            "has_multiple_arguments": self.has_multiple_arguments,
            "meta_kind": self.meta_kind,
            "plan_len": self.plan_len,
            "recovery_relevant": self.recovery_relevant,
            "schema_relevant": self.schema_relevant,
            "tool_names": list(self.tool_names),
        }

    @property
    def stratum_id(self) -> str:
        digest = _sha256(self.canonical_payload(), label="decision stratum")
        return f"decision-stratum-v1:{digest}"


@dataclass(frozen=True)
class DecisionStratumAudit:
    """Coverage evidence for one observed stratum.

    ``first_ordered_position`` is one-based so it can be compared directly with a prefix length.
    """

    stratum_id: str
    stratum: DecisionStratum
    total: int
    first_ordered_position: int

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-shaped per-stratum lineage record."""

        return {
            "first_ordered_position": self.first_ordered_position,
            "stratum": self.stratum.canonical_payload(),
            "stratum_id": self.stratum_id,
            "total": self.total,
        }


@dataclass(frozen=True)
class DecisionOrderAudit:
    """Auditable invariants and prefix coverage for a complete decision order."""

    contract: str
    source_conversation_count: int
    source_decision_count: int
    ordered_decision_count: int
    unique_decision_count: int
    observed_stratum_count: int
    frontload_decision_count: int
    order_sha256: str
    strata: tuple[DecisionStratumAudit, ...]
    ordered_stratum_ids: tuple[str, ...]

    @property
    def stratum_totals(self) -> dict[str, int]:
        """Return stable per-stratum totals keyed by canonical stratum identity."""

        return {entry.stratum_id: entry.total for entry in self.strata}

    def as_dict(self) -> dict[str, Any]:
        """Return the complete audit as a canonical JSON-shaped lineage object.

        Every tuple and dataclass is projected to JSON arrays and objects.  Calling this method
        repeatedly returns equal, independently owned values and never exposes mutable audit
        internals.
        """

        return {
            "contract": self.contract,
            "frontload_decision_count": self.frontload_decision_count,
            "observed_stratum_count": self.observed_stratum_count,
            "order_sha256": self.order_sha256,
            "ordered_decision_count": self.ordered_decision_count,
            "ordered_stratum_ids": list(self.ordered_stratum_ids),
            "source_conversation_count": self.source_conversation_count,
            "source_decision_count": self.source_decision_count,
            "strata": [entry.as_dict() for entry in self.strata],
            "unique_decision_count": self.unique_decision_count,
        }

    def prefix_counts(self, prefix_decisions: int) -> dict[str, int]:
        """Count every observed stratum in the first ``prefix_decisions`` decisions.

        Zero-count strata are retained, making this suitable for quota assertions.
        """

        if isinstance(prefix_decisions, bool) or not isinstance(prefix_decisions, int):
            raise TypeError("prefix_decisions must be an integer")
        if not 0 <= prefix_decisions <= self.ordered_decision_count:
            raise ValueError("prefix_decisions must be between zero and ordered_decision_count")
        counts = Counter(self.ordered_stratum_ids[:prefix_decisions])
        return {entry.stratum_id: counts[entry.stratum_id] for entry in self.strata}


@dataclass(frozen=True)
class DecisionOrdering:
    """A no-replacement epoch order plus its replayable audit."""

    keys: tuple[DecisionKey, ...]
    audit: DecisionOrderAudit


def quota_sampling_contract(
    ordering: DecisionOrdering,
    *,
    selected_decisions: int,
    require_all_strata: bool = True,
) -> dict[str, Any]:
    """Return compact lineage for a no-replacement prefix of ``ordering``.

    The complete order is bound by ``order_sha256`` while the selected prefix records exact
    per-stratum counts.  The repeated per-decision stratum-ID sequence is deliberately omitted
    from checkpoint metadata; it can be replayed from the frozen conversations.
    """

    if isinstance(selected_decisions, bool) or not isinstance(selected_decisions, int):
        raise TypeError("selected_decisions must be an integer")
    audit = ordering.audit
    if not 0 <= selected_decisions <= audit.ordered_decision_count:
        raise ValueError("selected_decisions must fit within the no-replacement decision epoch")
    if not isinstance(require_all_strata, bool):
        raise TypeError("require_all_strata must be boolean")
    if require_all_strata and selected_decisions < audit.frontload_decision_count:
        raise ValueError(
            "selected decision prefix is too short to cover every observed decision stratum"
        )
    prefix_counts = audit.prefix_counts(selected_decisions)
    compact_audit = audit.as_dict()
    compact_audit.pop("ordered_stratum_ids")
    return {
        "mode": QUOTA_SAMPLING_MODE,
        "no_replacement": True,
        "require_all_observed_strata": require_all_strata,
        "ordering": compact_audit,
        "selected_prefix": {
            "decisions": selected_decisions,
            "covered_strata": sum(count > 0 for count in prefix_counts.values()),
            "all_observed_strata_covered": all(count > 0 for count in prefix_counts.values()),
            "stratum_counts": prefix_counts,
        },
    }


@dataclass(frozen=True)
class _DecisionRecord:
    key: DecisionKey
    stratum: DecisionStratum
    decision_sha256: str


def _meta_fields(conversation: Conversation, *, label: str) -> dict[str, Any]:
    if not isinstance(conversation.meta, dict):
        raise TypeError(f"{label}.meta must be an object")
    _canonical_bytes(conversation.meta, label=f"{label}.meta")
    values: dict[str, Any] = {}
    for field in _META_TEXT_FIELDS:
        value = conversation.meta.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{label}.meta.{field} must be non-empty text when present")
        values[field] = value
    plan_len = conversation.meta.get("plan_len")
    if plan_len is not None and (
        isinstance(plan_len, bool) or not isinstance(plan_len, int) or plan_len < 0
    ):
        raise ValueError(f"{label}.meta.plan_len must be a non-negative integer when present")
    values["plan_len"] = plan_len
    return values


def _validate_tool_call(
    call: ToolCall,
    *,
    registry: Mapping[str, ToolSpec],
    label: str,
) -> ToolSpec:
    if not isinstance(call, ToolCall):
        raise TypeError(f"{label} must be a ToolCall")
    if not isinstance(call.name, str) or not call.name:
        raise ValueError(f"{label}.name must be non-empty text")
    if not isinstance(call.arguments, dict):
        raise TypeError(f"{label}.arguments must be an object")
    _canonical_bytes(call.arguments, label=f"{label}.arguments")
    _assert_no_reserved_markers(call.arguments, label=f"{label}.arguments")
    tool = registry.get(call.name)
    if tool is None:
        raise ValueError(f"{label} references unknown tool {call.name!r}")
    if not schema_matches(call.arguments, tool.parameters):
        raise ValueError(f"{label} arguments violate the schema for {call.name!r}")
    return tool


def _validate_messages(
    conversation: Conversation,
    *,
    conversation_index: int,
    registry: Mapping[str, ToolSpec],
    meta: Mapping[str, Any],
) -> tuple[int, ...]:
    label = f"conversation {conversation_index}"
    if not isinstance(conversation.messages, Sequence) or isinstance(
        conversation.messages, (str, bytes)
    ):
        raise TypeError(f"{label}.messages must be a sequence")
    assistant_indices: list[int] = []
    pending_tool_responses = 0
    actual_plan: list[str] = []
    for message_index, message in enumerate(conversation.messages):
        message_label = f"{label} message {message_index}"
        if not isinstance(message, Message):
            raise TypeError(f"{message_label} must be a Message")
        if not isinstance(message.role, Role):
            raise TypeError(f"{message_label}.role must be a canonical Role")
        if not isinstance(message.content, str):
            raise TypeError(f"{message_label}.content must be text")
        _assert_no_reserved_markers(message.content, label=f"{message_label}.content")
        if message.tool_response is not None and not isinstance(message.tool_response, str):
            raise TypeError(f"{message_label}.tool_response must be text or null")
        if message.tool_response is not None:
            _assert_no_reserved_markers(
                message.tool_response,
                label=f"{message_label}.tool_response",
            )
        if not isinstance(message.tool_calls, Sequence) or isinstance(
            message.tool_calls, (str, bytes)
        ):
            raise TypeError(f"{message_label}.tool_calls must be a sequence")

        if message.role == Role.assistant:
            if message.tool_response is not None:
                raise ValueError(f"{message_label} cannot contain a tool response")
            if message.content and message.tool_calls:
                raise ValueError(f"{message_label} mixes assistant text and tool calls")
            assistant_indices.append(message_index)
            for call_index, call in enumerate(message.tool_calls):
                _validate_tool_call(
                    call,
                    registry=registry,
                    label=f"{message_label} tool call {call_index}",
                )
                actual_plan.append(call.name)
            pending_tool_responses = len(message.tool_calls)
            continue

        if message.tool_calls:
            raise ValueError(f"{message_label} has tool calls outside an assistant decision")
        if message.role == Role.tool:
            if message.content:
                raise ValueError(f"{message_label} tool content must use tool_response")
            if pending_tool_responses < 1:
                raise ValueError(f"{message_label} has no prior assistant tool-call reference")
            pending_tool_responses -= 1
        elif message.tool_response is not None:
            raise ValueError(f"{message_label} has a tool response outside a tool message")
        else:
            pending_tool_responses = 0

    plan_len = meta["plan_len"]
    if plan_len is not None and plan_len != len(actual_plan):
        raise ValueError(
            f"{label}.meta.plan_len={plan_len} does not match {len(actual_plan)} tool calls"
        )
    raw_plan = conversation.meta.get("plan")
    if raw_plan is not None:
        if not isinstance(raw_plan, list) or any(
            not isinstance(name, str) or not name for name in raw_plan
        ):
            raise ValueError(f"{label}.meta.plan must be a list of non-empty tool names")
        if raw_plan != actual_plan:
            raise ValueError(f"{label}.meta.plan does not match assistant tool-call references")
    if meta["kind"] == "planner_episode" and (plan_len is None or raw_plan is None):
        raise ValueError(f"{label} planner_episode requires meta.plan and meta.plan_len")
    return tuple(assistant_indices)


def _primitive_kind(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def _collect_schema_signals(
    value: Any,
    schema: Mapping[str, Any],
    primitives: set[str],
) -> bool:
    """Collect scalar kinds and return whether an enum constrained the observed value."""

    primitive = _primitive_kind(value)
    if primitive is not None:
        primitives.add(primitive)
    has_enum = "enum" in schema
    if isinstance(value, list):
        item_schema = schema.get("items", {})
        for item in value:
            has_enum = _collect_schema_signals(item, item_schema, primitives) or has_enum
    elif isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", {})
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema is None:
                child_schema = additional if isinstance(additional, Mapping) else {}
            has_enum = _collect_schema_signals(item, child_schema, primitives) or has_enum
    return has_enum


def _argument_signals(
    calls: Sequence[ToolCall],
    registry: Mapping[str, ToolSpec],
) -> tuple[tuple[str, ...], bool, bool]:
    primitives: set[str] = set()
    has_enum = False
    has_multiple = False
    for call in calls:
        has_multiple = has_multiple or len(call.arguments) > 1
        properties = registry[call.name].parameters.get("properties", {})
        additional = registry[call.name].parameters.get("additionalProperties", {})
        for name, value in call.arguments.items():
            schema = properties.get(name)
            if schema is None:
                schema = additional if isinstance(additional, Mapping) else {}
            has_enum = _collect_schema_signals(value, schema, primitives) or has_enum
    ordered_primitives = tuple(kind for kind in _PRIMITIVE_ORDER if kind in primitives)
    return ordered_primitives, has_enum, has_multiple


def _string_leaves(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(item for child in value for item in _string_leaves(child))
    if isinstance(value, dict):
        return tuple(item for child in value.values() for item in _string_leaves(child))
    return ()


def _grounded_followup(
    conversation: Conversation,
    *,
    message_index: int,
) -> bool:
    message = conversation.messages[message_index]
    if not message.tool_calls:
        return False
    user_text = "\n".join(
        prior.content
        for prior in conversation.messages[:message_index]
        if prior.role == Role.user and prior.content
    )
    tool_responses = tuple(
        prior.tool_response
        for prior in conversation.messages[:message_index]
        if prior.role == Role.tool and prior.tool_response
    )
    if not tool_responses:
        return False
    for call in message.tool_calls:
        for value in _string_leaves(call.arguments):
            if value not in user_text and any(value in response for response in tool_responses):
                return True
    return False


def _recovery_relevant(conversation: Conversation) -> bool:
    metadata = " ".join(
        value.casefold()
        for name in ("category", "group", "kind", "stratum", "type")
        if isinstance((value := conversation.meta.get(name)), str)
    )
    if "recovery" in metadata:
        return True

    failure_seen = False
    remediation_seen = False
    retry_seen = False
    attempted_before_failure: set[str] = set()
    for message in conversation.messages:
        if message.role == Role.assistant and message.tool_calls:
            names = {call.name for call in message.tool_calls}
            if not failure_seen:
                attempted_before_failure.update(names)
            else:
                if names - attempted_before_failure:
                    remediation_seen = True
                if remediation_seen and names & attempted_before_failure:
                    retry_seen = True
        elif message.role == Role.tool and message.tool_response:
            response = message.tool_response.casefold()
            if not failure_seen and any(marker in response for marker in _FAILURE_MARKERS):
                failure_seen = True
            elif (
                failure_seen
                and remediation_seen
                and retry_seen
                and any(marker in response for marker in _SUCCESS_MARKERS)
            ):
                return True
    return False


def _explicit_abstention(meta: Mapping[str, Any]) -> bool:
    category = meta["category"]
    kind = meta["kind"]
    return (
        category in {"no_tool", "abstention", "irrelevance"}
        or kind in {"no_tool", "abstention"}
        or (kind == "planner_episode" and meta["plan_len"] == 0)
    )


def _message_payload(message: Message) -> dict[str, Any]:
    return {
        "content": message.content,
        "role": message.role.value,
        "tool_calls": [
            {"arguments": call.arguments, "name": call.name} for call in message.tool_calls
        ],
        "tool_response": message.tool_response,
    }


def _decision_record(
    conversation: Conversation,
    *,
    conversation_index: int,
    message_index: int,
    assistant_ordinal: int,
    assistant_count: int,
    registry: Mapping[str, ToolSpec],
    meta: Mapping[str, Any],
    recovery_relevant: bool,
) -> _DecisionRecord:
    message = conversation.messages[message_index]
    primitives, has_enum, has_multiple = _argument_signals(message.tool_calls, registry)
    abstention = _explicit_abstention(meta)
    if abstention and message.tool_calls:
        raise ValueError(
            f"conversation {conversation_index} message {message_index} calls a tool in an "
            "explicit abstention conversation"
        )
    if len(message.tool_calls) > 1:
        decision_kind = "parallel"
    elif message.tool_calls:
        decision_kind = "tool"
    elif abstention:
        decision_kind = "abstention"
    else:
        decision_kind = "text"
    trajectory = assistant_count > 1 or any(
        candidate.role == Role.tool for candidate in conversation.messages
    )
    grounded = _grounded_followup(conversation, message_index=message_index)
    metadata_schema_signal = any(
        "schema" in value.casefold()
        for name in ("category", "group", "kind", "stratum", "type")
        if isinstance((value := conversation.meta.get(name)), str)
    )
    schema_relevant = (
        metadata_schema_signal
        or has_enum
        or has_multiple
        or any(kind != "string" for kind in primitives)
    )
    stratum = DecisionStratum(
        category=meta["category"],
        group=meta["group"],
        meta_kind=meta["kind"],
        plan_len=meta["plan_len"],
        conversation_shape="trajectory" if trajectory else "simple",
        assistant_ordinal=assistant_ordinal,
        decision_kind=decision_kind,
        tool_names=tuple(call.name for call in message.tool_calls),
        argument_primitives=primitives,
        has_enum_argument=has_enum,
        has_multiple_arguments=has_multiple,
        recovery_relevant=recovery_relevant,
        schema_relevant=schema_relevant,
        grounded_followup_relevant=grounded,
    )
    key = (conversation_index, message_index)
    identity = {
        "contract": ORDERING_CONTRACT,
        "key": list(key),
        "message": _message_payload(message),
        "stratum": stratum.canonical_payload(),
    }
    return _DecisionRecord(
        key=key,
        stratum=stratum,
        decision_sha256=_sha256(identity, label=f"assistant decision {key}"),
    )


def order_assistant_decisions(
    conversations: Sequence[Conversation],
) -> DecisionOrdering:
    """Return a deterministic quota-first, no-replacement order over assistant messages.

    The first ``observed_stratum_count`` entries contain one canonical-hash-selected decision from
    every observed composite stratum, with lower-frequency strata first and canonical SHA-256 as
    the tie-break. Remaining decisions are hash-stable within each stratum and smoothly interleaved
    at centered proportional quantiles. The tuple key's second component is the absolute index in
    ``Conversation.messages`` (and is therefore directly dereferenceable).
    """

    if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
        raise TypeError("conversations must be a sequence")

    records: list[_DecisionRecord] = []
    catalog_cache = FunctionCatalogCache()
    for conversation_index, conversation in enumerate(conversations):
        label = f"conversation {conversation_index}"
        if not isinstance(conversation, Conversation):
            raise TypeError(f"{label} must be a Conversation")
        meta = _meta_fields(conversation, label=label)
        registry = catalog_cache.entry(conversation.tools).registry
        assistant_indices = _validate_messages(
            conversation,
            conversation_index=conversation_index,
            registry=registry,
            meta=meta,
        )
        recovery_relevant = _recovery_relevant(conversation)
        for assistant_ordinal, message_index in enumerate(assistant_indices, start=1):
            records.append(
                _decision_record(
                    conversation,
                    conversation_index=conversation_index,
                    message_index=message_index,
                    assistant_ordinal=assistant_ordinal,
                    assistant_count=len(assistant_indices),
                    registry=registry,
                    meta=meta,
                    recovery_relevant=recovery_relevant,
                )
            )

    source_keys = [record.key for record in records]
    if len(set(source_keys)) != len(source_keys):  # pragma: no cover - indices make this invariant
        raise RuntimeError("assistant decision references are not unique")

    groups: dict[DecisionStratum, list[_DecisionRecord]] = defaultdict(list)
    for record in records:
        groups[record.stratum].append(record)

    strata_by_id: dict[str, DecisionStratum] = {}
    for stratum in groups:
        previous = strata_by_id.setdefault(stratum.stratum_id, stratum)
        if previous != stratum:  # pragma: no cover - requires a SHA-256 collision
            raise RuntimeError("decision stratum SHA-256 collision")
    ordered_strata = sorted(
        groups,
        key=lambda stratum: (
            len(groups[stratum]),
            stratum.stratum_id,
            _canonical_bytes(stratum.canonical_payload(), label="decision stratum"),
        ),
    )

    frontloaded: list[_DecisionRecord] = []
    proportional: list[tuple[Fraction, str, _DecisionRecord]] = []
    for stratum in ordered_strata:
        members = sorted(
            groups[stratum],
            key=lambda record: (record.decision_sha256, record.key),
        )
        frontloaded.append(members[0])
        remainder = members[1:]
        remainder_count = len(remainder)
        for ordinal, record in enumerate(remainder, start=1):
            centered_quantile = Fraction(2 * ordinal - 1, 2 * remainder_count)
            tie_sha256 = hashlib.sha256(
                (
                    ORDERING_CONTRACT + "\0" + stratum.stratum_id + "\0" + record.decision_sha256
                ).encode("ascii")
            ).hexdigest()
            proportional.append((centered_quantile, tie_sha256, record))

    ordered_records = [
        *frontloaded,
        *(
            record
            for _quantile, _tie, record in sorted(
                proportional,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2].decision_sha256,
                    item[2].key,
                ),
            )
        ),
    ]
    ordered_keys = tuple(record.key for record in ordered_records)
    unique_count = len(set(ordered_keys))
    if len(ordered_records) != len(records) or unique_count != len(records):
        raise RuntimeError("quota ordering violated exact no-replacement coverage")

    ordered_stratum_ids = tuple(record.stratum.stratum_id for record in ordered_records)
    positions: dict[str, int] = {}
    for position, stratum_id in enumerate(ordered_stratum_ids, start=1):
        positions.setdefault(stratum_id, position)
    stratum_audits = tuple(
        DecisionStratumAudit(
            stratum_id=stratum.stratum_id,
            stratum=stratum,
            total=len(groups[stratum]),
            first_ordered_position=positions[stratum.stratum_id],
        )
        for stratum in sorted(groups, key=lambda item: item.stratum_id)
    )
    order_identity = {
        "contract": ORDERING_CONTRACT,
        "ordered_decisions": [
            {
                "decision_sha256": record.decision_sha256,
                "key": list(record.key),
                "stratum_id": record.stratum.stratum_id,
            }
            for record in ordered_records
        ],
    }
    audit = DecisionOrderAudit(
        contract=ORDERING_CONTRACT,
        source_conversation_count=len(conversations),
        source_decision_count=len(records),
        ordered_decision_count=len(ordered_records),
        unique_decision_count=unique_count,
        observed_stratum_count=len(groups),
        frontload_decision_count=len(groups),
        order_sha256=_sha256(order_identity, label="assistant decision order"),
        strata=stratum_audits,
        ordered_stratum_ids=ordered_stratum_ids,
    )
    if any(
        count < 1 for count in audit.prefix_counts(audit.frontload_decision_count).values()
    ):  # pragma: no cover - construction selects one member from each group
        raise RuntimeError("quota frontload omitted an observed decision stratum")
    return DecisionOrdering(keys=ordered_keys, audit=audit)
