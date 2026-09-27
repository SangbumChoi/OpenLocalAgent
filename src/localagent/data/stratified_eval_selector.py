"""Deterministic, content-addressed selection of bounded agent evaluation subsets.

The selector operates on canonical :class:`~localagent.data.schema.Conversation` rows.  It
first covers every observed evaluation stratum, then fills the remaining capacity in semantic
SHA-256 order.  Selected rows are returned in their original source order.

This module intentionally does not read or publish artifacts.  Callers can bind its canonical
audit to their own artifact lineage without coupling selection policy to a particular runner.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from localagent.data.conversation_artifact import (
    canonical_json_bytes,
    conversation_semantic_sha256,
)
from localagent.data.schema import Conversation, Role

ALGORITHM = "greedy_uncovered_strata_then_semantic_sha256_fill_v1"
AUDIT_SCHEMA_VERSION = 1

_CATEGORY = "single_turn_category"
_KIND = "multi_turn_kind"
_PLAN_LENGTH = "planner_plan_len"
_TOOL = "tool_name"
_BEHAVIOR = "behavior"
_FAMILY_ORDER = (_CATEGORY, _KIND, _PLAN_LENGTH, _TOOL, _BEHAVIOR)
_RECOVERY_MARKERS = ("error", "fail", "recover", "retry")
_ABSTENTION_MARKERS = {"abstain", "abstention", "irrelevance", "no_tool"}
_META_BEHAVIOR_KEYS = ("category", "kind", "group", "stratum", "type")


class InsufficientStratumCapacityError(ValueError):
    """Raised when ``max_rows`` is smaller than deterministic mandatory coverage."""

    def __init__(
        self,
        *,
        max_rows: int,
        required_rows: int,
        mandatory_strata: int,
    ) -> None:
        self.max_rows = max_rows
        self.required_rows = required_rows
        self.mandatory_strata = mandatory_strata
        super().__init__(
            "max_rows cannot cover all observed mandatory strata: "
            f"max_rows={max_rows}, required_rows={required_rows}, "
            f"mandatory_strata={mandatory_strata}"
        )


@dataclass(frozen=True, order=True)
class EvalStratum:
    """One namespaced capability stratum observed in an evaluation row."""

    family: str
    value: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILY_ORDER:
            raise ValueError(f"unsupported evaluation stratum family: {self.family!r}")
        if not self.value:
            raise ValueError("evaluation stratum value must be non-empty")


@dataclass(frozen=True)
class StratumCount:
    """Source and selected counts for one mandatory stratum."""

    stratum: EvalStratum
    source_rows: int
    selected_rows: int
    source_assistant_decisions: int
    selected_assistant_decisions: int

    def as_dict(self) -> dict[str, int]:
        return {
            "source_rows": self.source_rows,
            "selected_rows": self.selected_rows,
            "source_assistant_decisions": self.source_assistant_decisions,
            "selected_assistant_decisions": self.selected_assistant_decisions,
        }


@dataclass(frozen=True)
class StratifiedEvalAudit:
    """Canonical evidence for one bounded deterministic selection."""

    max_rows: int
    coverage_rows: int
    fill_rows: int
    source_rows: int
    selected_rows: int
    source_assistant_decisions: int
    selected_assistant_decisions: int
    source_unique_semantic_rows: int
    selected_unique_semantic_rows: int
    source_semantic_set_sha256: str
    selected_semantic_set_sha256: str
    selected_source_row_numbers: tuple[int, ...]
    stratum_counts: tuple[StratumCount, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the stable JSON-shaped audit, including a hash of its canonical core."""

        grouped: dict[str, dict[str, dict[str, int]]] = {family: {} for family in _FAMILY_ORDER}
        for count in self.stratum_counts:
            grouped[count.stratum.family][count.stratum.value] = count.as_dict()
        grouped = {family: values for family, values in grouped.items() if values}

        core: dict[str, Any] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "fingerprint_contract": {
                "semantic_row": (
                    "sha256(canonical compact sorted-key JSON of messages+tools; meta excluded)"
                ),
                "semantic_set": "sha256(sorted unique semantic row SHA-256 values joined by LF)",
                "coverage_tie_break": (
                    "maximum uncovered-strata gain, then semantic row SHA-256, "
                    "then sorted stratum identity, then source order"
                ),
                "fill_order": "semantic row SHA-256, then source order",
                "output_order": "ascending one-based source row number",
            },
            "capacity": {
                "max_rows": self.max_rows,
                "coverage_rows": self.coverage_rows,
                "fill_rows": self.fill_rows,
            },
            "source": {
                "rows": self.source_rows,
                "assistant_decisions": self.source_assistant_decisions,
                "unique_semantic_rows": self.source_unique_semantic_rows,
                "semantic_set_sha256": self.source_semantic_set_sha256,
            },
            "selected": {
                "rows": self.selected_rows,
                "assistant_decisions": self.selected_assistant_decisions,
                "unique_semantic_rows": self.selected_unique_semantic_rows,
                "semantic_set_sha256": self.selected_semantic_set_sha256,
                "source_row_numbers": list(self.selected_source_row_numbers),
            },
            "mandatory_strata": len(self.stratum_counts),
            "strata": grouped,
        }
        audit_sha256 = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
        return {**core, "audit_sha256": audit_sha256}

    def canonical_bytes(self) -> bytes:
        """Serialize the complete audit as canonical compact sorted-key JSON plus LF."""

        return canonical_json_bytes(self.as_dict())


@dataclass(frozen=True)
class StratifiedEvalSelection:
    """Selected conversations and the evidence describing their derivation."""

    conversations: tuple[Conversation, ...]
    source_row_numbers: tuple[int, ...]
    audit: StratifiedEvalAudit


@dataclass(frozen=True)
class _IndexedRow:
    source_index: int
    conversation: Conversation
    semantic_sha256: str
    assistant_decisions: int
    strata: frozenset[EvalStratum]

    @property
    def source_row_number(self) -> int:
        return self.source_index + 1


def _semantic_set_sha256(fingerprints: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(fingerprints))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _meta_marker(conversation: Conversation, markers: tuple[str, ...]) -> bool:
    for key in _META_BEHAVIOR_KEYS:
        value = conversation.meta.get(key)
        if isinstance(value, str):
            normalized = value.casefold()
            if any(marker in normalized for marker in markers):
                return True
    return False


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _has_structural_recovery(conversation: Conversation) -> bool:
    failure_seen = False
    for message in conversation.messages:
        if message.role == Role.tool and isinstance(message.tool_response, str):
            response = message.tool_response.casefold()
            failure_seen = failure_seen or any(marker in response for marker in _RECOVERY_MARKERS)
        elif failure_seen and message.role == Role.assistant and message.tool_calls:
            return True
    return False


def _schema_behavior_values(conversation: Conversation) -> set[str]:
    values: set[str] = set()
    tool_schemas = {tool.name: tool.parameters for tool in conversation.tools}
    for message in conversation.messages:
        if message.role != Role.assistant:
            continue
        for call in message.tool_calls:
            arguments = call.arguments
            if len(arguments) > 1:
                values.add("schema_multiple_arguments")
            parameters = tool_schemas.get(call.name, {})
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            required = parameters.get("required", ())
            required_names = set(required) if isinstance(required, list) else set()
            for name, value in arguments.items():
                values.add(f"schema_argument_type_{_json_type(value)}")
                if name not in required_names:
                    values.add("schema_optional_argument")
                property_schema = properties.get(name, {})
                if isinstance(property_schema, dict):
                    enum = property_schema.get("enum")
                    if isinstance(enum, list) and value in enum:
                        values.add("schema_enum_argument")
    if _meta_marker(conversation, ("schema",)):
        values.add("schema_episode")
    return values


def conversation_eval_strata(conversation: Conversation) -> frozenset[EvalStratum]:
    """Extract all mandatory evaluation strata from one canonical conversation.

    A single-turn row has exactly one assistant decision and must provide
    ``meta.category``.  A multi-turn row has two or more assistant decisions and must provide
    ``meta.kind``.  This decision-based definition also handles optional system messages.
    """

    if not isinstance(conversation, Conversation):
        raise TypeError("evaluation rows must be Conversation instances")
    if not isinstance(conversation.meta, dict):
        raise TypeError("Conversation.meta must be a dictionary")

    assistant_messages = [
        message for message in conversation.messages if message.role == Role.assistant
    ]
    if not assistant_messages:
        raise ValueError("evaluation Conversation must contain an assistant decision")

    strata: set[EvalStratum] = set()
    if len(assistant_messages) == 1:
        category = conversation.meta.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("single-turn evaluation Conversation requires non-empty meta.category")
        strata.add(EvalStratum(_CATEGORY, category))
    else:
        kind = conversation.meta.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError("multi-turn evaluation Conversation requires non-empty meta.kind")
        strata.add(EvalStratum(_KIND, kind))

    if "plan_len" in conversation.meta:
        plan_length = conversation.meta["plan_len"]
        if isinstance(plan_length, bool) or not isinstance(plan_length, int):
            raise ValueError("meta.plan_len must be a non-negative integer")
        if plan_length < 0:
            raise ValueError("meta.plan_len must be a non-negative integer")
        strata.add(EvalStratum(_PLAN_LENGTH, str(plan_length)))

    tool_names: set[str] = set()
    for message in assistant_messages:
        for call in message.tool_calls:
            if not isinstance(call.name, str) or not call.name:
                raise ValueError("assistant tool-call names must be non-empty strings")
            tool_names.add(call.name)
    strata.update(EvalStratum(_TOOL, name) for name in tool_names)

    category = conversation.meta.get("category")
    category_key = category.casefold() if isinstance(category, str) else ""
    abstention_tagged = category_key in _ABSTENTION_MARKERS or _meta_marker(
        conversation,
        ("abstain", "irrelevance"),
    )
    zero_step_plan = (
        conversation.meta.get("kind") == "planner_episode"
        and conversation.meta.get("plan_len") == 0
    )
    if abstention_tagged or zero_step_plan:
        strata.add(EvalStratum(_BEHAVIOR, "abstention"))
    if any(not message.tool_calls and bool(message.content) for message in assistant_messages):
        strata.add(EvalStratum(_BEHAVIOR, "text"))
    if any(len(message.tool_calls) > 1 for message in assistant_messages):
        strata.add(EvalStratum(_BEHAVIOR, "parallel"))
    if _meta_marker(conversation, _RECOVERY_MARKERS) or _has_structural_recovery(conversation):
        strata.add(EvalStratum(_BEHAVIOR, "recovery"))
    strata.update(
        EvalStratum(_BEHAVIOR, behavior) for behavior in _schema_behavior_values(conversation)
    )
    return frozenset(strata)


def _coverage_selection(rows: Sequence[_IndexedRow]) -> set[int]:
    uncovered = set().union(*(row.strata for row in rows))
    selected: set[int] = set()
    while uncovered:
        candidates = [
            row for row in rows if row.source_index not in selected and row.strata & uncovered
        ]
        if not candidates:
            raise RuntimeError("observed evaluation strata have no selectable source row")
        best = min(
            candidates,
            key=lambda row: (
                -len(row.strata & uncovered),
                row.semantic_sha256,
                tuple(sorted(row.strata)),
                row.source_index,
            ),
        )
        selected.add(best.source_index)
        uncovered.difference_update(best.strata)
    return selected


def select_stratified_eval_subset(
    conversations: Sequence[Conversation],
    *,
    max_rows: int,
) -> StratifiedEvalSelection:
    """Select at most ``max_rows`` while covering every observed mandatory stratum.

    The coverage phase is deterministic and content-addressed.  If its selected row union is
    larger than ``max_rows``, the function raises instead of silently dropping a stratum.  Once
    coverage is complete, remaining rows are added by semantic SHA-256.  Final output order
    always matches source order.
    """

    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    indexed_rows: list[_IndexedRow] = []
    for source_index, conversation in enumerate(conversations):
        strata = conversation_eval_strata(conversation)
        assistant_decisions = sum(
            message.role == Role.assistant for message in conversation.messages
        )
        indexed_rows.append(
            _IndexedRow(
                source_index=source_index,
                conversation=conversation,
                semantic_sha256=conversation_semantic_sha256(conversation),
                assistant_decisions=assistant_decisions,
                strata=strata,
            )
        )

    if not indexed_rows:
        raise ValueError("cannot select an evaluation subset from an empty source")

    coverage_indices = _coverage_selection(indexed_rows)
    all_strata = set().union(*(row.strata for row in indexed_rows))
    if len(coverage_indices) > max_rows:
        raise InsufficientStratumCapacityError(
            max_rows=max_rows,
            required_rows=len(coverage_indices),
            mandatory_strata=len(all_strata),
        )

    target_rows = min(max_rows, len(indexed_rows))
    selected_indices = set(coverage_indices)
    for row in sorted(
        indexed_rows,
        key=lambda row: (row.semantic_sha256, row.source_index),
    ):
        if len(selected_indices) == target_rows:
            break
        selected_indices.add(row.source_index)

    selected_rows = [row for row in indexed_rows if row.source_index in selected_indices]
    if len(selected_rows) != target_rows:
        raise RuntimeError("deterministic evaluation fill did not reach target capacity")
    if set().union(*(row.strata for row in selected_rows)) != all_strata:
        raise RuntimeError("selected evaluation rows do not cover every observed stratum")

    counts: list[StratumCount] = []
    for stratum in sorted(all_strata):
        source_members = [row for row in indexed_rows if stratum in row.strata]
        selected_members = [row for row in selected_rows if stratum in row.strata]
        counts.append(
            StratumCount(
                stratum=stratum,
                source_rows=len(source_members),
                selected_rows=len(selected_members),
                source_assistant_decisions=sum(row.assistant_decisions for row in source_members),
                selected_assistant_decisions=sum(
                    row.assistant_decisions for row in selected_members
                ),
            )
        )

    source_fingerprints = [row.semantic_sha256 for row in indexed_rows]
    selected_fingerprints = [row.semantic_sha256 for row in selected_rows]
    source_row_numbers = tuple(row.source_row_number for row in selected_rows)
    audit = StratifiedEvalAudit(
        max_rows=max_rows,
        coverage_rows=len(coverage_indices),
        fill_rows=len(selected_rows) - len(coverage_indices),
        source_rows=len(indexed_rows),
        selected_rows=len(selected_rows),
        source_assistant_decisions=sum(row.assistant_decisions for row in indexed_rows),
        selected_assistant_decisions=sum(row.assistant_decisions for row in selected_rows),
        source_unique_semantic_rows=len(set(source_fingerprints)),
        selected_unique_semantic_rows=len(set(selected_fingerprints)),
        source_semantic_set_sha256=_semantic_set_sha256(source_fingerprints),
        selected_semantic_set_sha256=_semantic_set_sha256(selected_fingerprints),
        selected_source_row_numbers=source_row_numbers,
        stratum_counts=tuple(counts),
    )
    return StratifiedEvalSelection(
        conversations=tuple(row.conversation for row in selected_rows),
        source_row_numbers=source_row_numbers,
        audit=audit,
    )
