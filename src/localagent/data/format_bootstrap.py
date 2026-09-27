"""Deterministic format-bootstrap curriculum derived from verified train Conversations.

The first paper-scale SFT run learned many local next-token decisions without learning one
complete strict tool-call sequence.  This module builds a small continuation-SFT artifact whose
source order is itself the curriculum:

1. short, single-call rows with zero or one argument;
2. single-call rows with multiple arguments;
3. parallel tool-call rows;
4. text and explicit-restraint rows.

Every emitted row is an unchanged member of a provenance-bound, rule-verified train artifact.
The selector does not generate slot values, flatten trajectories, or inspect evaluation targets.
It verifies the held-out artifact only to reject semantic/rendered-prompt overlap.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from localagent.data.conversation_artifact import (
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    ConversationOverlapAudit,
    FileIdentity,
    VerifiedConversationArtifact,
    assert_no_conversation_overlap,
    canonical_json_bytes,
    conversation_semantic_sha256,
    load_verified_conversation_artifact,
    self_hashed_manifest,
)
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    FunctionCatalogCache,
    assistant_training_turns,
)
from localagent.data.schema import Conversation, Role, ToolCall
from localagent.model.tokenizer import BPE_EOS, batched_token_lengths, load_tokenizer

FORMAT_BOOTSTRAP_CONFIG_KIND = "localagent_format_bootstrap_config"
FORMAT_BOOTSTRAP_RECEIPT_KIND = "localagent_format_bootstrap_receipt"
FORMAT_BOOTSTRAP_SCHEMA_VERSION = 1
FORMAT_BOOTSTRAP_ALGORITHM = "balanced_shortest_simple_turn_v1"
FORMAT_BOOTSTRAP_PHASES = (
    "format_core",
    "multi_argument",
    "parallel",
    "text",
)

_CONFIG_KEYS = frozenset(
    {
        "algorithm",
        "evaluation_holdout",
        "kind",
        "manifest",
        "out",
        "phases",
        "prompt_contract",
        "receipt",
        "schema_version",
        "source",
        "tokenizer",
    }
)
_ARTIFACT_BINDING_KEYS = frozenset(
    {"expected_identity", "generator_config", "manifest", "path"}
)
_TOKENIZER_KEYS = frozenset({"identity", "kind", "path"})

__all__ = [
    "FORMAT_BOOTSTRAP_ALGORITHM",
    "FORMAT_BOOTSTRAP_CONFIG_KIND",
    "FORMAT_BOOTSTRAP_PHASES",
    "FORMAT_BOOTSTRAP_RECEIPT_KIND",
    "FORMAT_BOOTSTRAP_SCHEMA_VERSION",
    "FormatBootstrapRecord",
    "FormatBootstrapSelection",
    "analyze_conversation_lengths",
    "build_format_bootstrap",
    "classify_format_bootstrap_phase",
    "select_format_bootstrap",
]


@dataclass(frozen=True)
class FormatBootstrapRecord:
    """One eligible source row and the deterministic signals used to order it."""

    conversation: Conversation
    source_row_number: int
    phase: str
    bucket: str
    semantic_sha256: str
    row_sha256: str
    prompt_suffix_tokens: int
    full_prompt_tokens: int
    target_tokens: int
    category: str
    tool_names: tuple[str, ...]
    argument_count: int

    @property
    def input_tokens(self) -> int:
        """Exact unpadded next-token input count for this one-decision row."""

        return self.full_prompt_tokens + self.target_tokens - 1


@dataclass(frozen=True)
class FormatBootstrapSelection:
    """Selected Conversations in curriculum order plus a replayable audit."""

    records: tuple[FormatBootstrapRecord, ...]
    audit: Mapping[str, Any]

    @property
    def conversations(self) -> tuple[Conversation, ...]:
        return tuple(record.conversation for record in self.records)


@dataclass(frozen=True)
class _PendingRecord:
    conversation: Conversation
    source_row_number: int
    phase: str
    bucket: str
    semantic_sha256: str
    row_sha256: str
    prompt_suffix: str
    body: str
    catalog_tokens: int
    category: str
    tool_names: tuple[str, ...]
    argument_count: int


def _canonical_object_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)[:-1]


def _sha256_object(value: Any) -> str:
    return hashlib.sha256(_canonical_object_bytes(value)).hexdigest()


def _fingerprint_set_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(values))).encode("ascii")).hexdigest()


def _ordered_fingerprint_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def _validate_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _phase_counts(value: Any) -> dict[str, int]:
    phases = _validate_exact_keys(value, frozenset(FORMAT_BOOTSTRAP_PHASES), label="phases")
    normalized: dict[str, int] = {}
    for phase in FORMAT_BOOTSTRAP_PHASES:
        count = phases[phase]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"phases.{phase} must be a non-negative integer")
        normalized[phase] = count
    if sum(normalized.values()) < 1:
        raise ValueError("format-bootstrap phases must select at least one row")
    return normalized


def _simple_assistant(conversation: Conversation):
    messages = conversation.messages
    if (
        len(messages) != 2
        or messages[0].role != Role.user
        or messages[1].role != Role.assistant
    ):
        return None
    return messages[1]


def classify_format_bootstrap_phase(conversation: Conversation) -> str | None:
    """Classify a simple one-decision Conversation into one curriculum phase.

    Multi-turn rows intentionally return ``None``.  The bootstrap is a strict-format recovery
    artifact, not a replacement for the full trajectory corpus.
    """

    assistant = _simple_assistant(conversation)
    if assistant is None:
        return None
    calls = assistant.tool_calls
    if not calls:
        return "text"
    if len(calls) > 1:
        return "parallel"
    return "format_core" if len(calls[0].arguments) <= 1 else "multi_argument"


def _record_bucket(
    phase: str,
    conversation: Conversation,
    calls: Sequence[ToolCall],
) -> str:
    if phase in {"format_core", "multi_argument"}:
        return calls[0].name
    if phase == "parallel":
        return json.dumps(
            [call.name for call in calls],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    category = conversation.meta.get("category", conversation.meta.get("kind", "text"))
    return str(category)


def _length_distribution(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {
            "count": 0,
            "max": 0,
            "mean_denominator": 0,
            "mean_numerator": 0,
            "min": 0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
        }
    ordered = sorted(values)

    def percentile(numerator: int, denominator: int) -> int:
        # Deterministic nearest-rank percentile, expressed without floating point.
        rank = max(1, (len(ordered) * numerator + denominator - 1) // denominator)
        return ordered[rank - 1]

    return {
        "count": len(ordered),
        "max": ordered[-1],
        "mean_denominator": len(ordered),
        "mean_numerator": sum(ordered),
        "min": ordered[0],
        "p50": percentile(50, 100),
        "p90": percentile(90, 100),
        "p95": percentile(95, 100),
    }


def _token_vectors_audit(
    *,
    full_prompt_tokens: Sequence[int],
    prompt_suffix_tokens: Sequence[int],
    target_tokens: Sequence[int],
) -> dict[str, Any]:
    if not (
        len(full_prompt_tokens) == len(prompt_suffix_tokens) == len(target_tokens)
    ):
        raise RuntimeError("token-audit vectors must have equal lengths")
    input_tokens = [
        prompt + target - 1
        for prompt, target in zip(full_prompt_tokens, target_tokens, strict=True)
    ]
    return {
        "full_prompt_tokens": _length_distribution(full_prompt_tokens),
        "input_tokens": _length_distribution(input_tokens),
        "prompt_suffix_tokens": _length_distribution(prompt_suffix_tokens),
        "target_tokens_including_eos": _length_distribution(target_tokens),
        "totals": {
            "input_tokens": sum(input_tokens),
            "loss_tokens": sum(target_tokens),
        },
    }


def _token_audit(records: Sequence[FormatBootstrapRecord]) -> dict[str, Any]:
    return _token_vectors_audit(
        full_prompt_tokens=[record.full_prompt_tokens for record in records],
        prompt_suffix_tokens=[record.prompt_suffix_tokens for record in records],
        target_tokens=[record.target_tokens for record in records],
    )


def _counter_dict(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def analyze_conversation_lengths(
    conversations: Sequence[Conversation],
    tokenizer,
    *,
    token_batch_size: int = 512,
) -> dict[str, Any]:
    """Measure exact full-catalog prompt and masked-target lengths for a Conversation corpus."""

    if isinstance(token_batch_size, bool) or not isinstance(token_batch_size, int):
        raise TypeError("token_batch_size must be an integer")
    if token_batch_size < 1:
        raise ValueError("token_batch_size must be positive")
    catalog_cache = FunctionCatalogCache()
    catalog_token_counts: dict[int, int] = {}
    suffixes: list[str] = []
    bodies: list[str] = []
    catalog_lengths: list[int] = []
    decision_kinds = Counter()
    categories = Counter()
    tool_names = Counter()
    simple_conversations = 0
    for conversation in conversations:
        category = str(
            conversation.meta.get("category", conversation.meta.get("kind", "unknown"))
        )
        categories[category] += 1
        simple_conversations += int(_simple_assistant(conversation) is not None)
        tools_id = id(conversation.tools)
        catalog_tokens = catalog_token_counts.get(tools_id)
        if catalog_tokens is None:
            catalog = catalog_cache.entry(conversation.tools).text + BPE_EOS
            catalog_tokens = len(tokenizer.encode(catalog))
            catalog_token_counts[tools_id] = catalog_tokens
        turns = assistant_training_turns(conversation, catalog_cache=catalog_cache)
        messages_by_index = {index: message for index, message in enumerate(conversation.messages)}
        for turn in turns:
            assistant = messages_by_index[turn.message_index]
            calls = assistant.tool_calls
            if not calls:
                decision_kinds["text"] += 1
            elif len(calls) > 1:
                decision_kinds["parallel"] += 1
            elif len(calls[0].arguments) > 1:
                decision_kinds["single_call_multi_argument"] += 1
            else:
                decision_kinds["single_call_zero_or_one_argument"] += 1
            for call in calls:
                tool_names[call.name] += 1
            suffixes.append(turn.prompt_suffix)
            bodies.append(turn.body)
            catalog_lengths.append(catalog_tokens)

    suffix_lengths = batched_token_lengths(
        tokenizer,
        suffixes,
        batch_size=token_batch_size,
    )
    body_lengths = batched_token_lengths(
        tokenizer,
        bodies,
        batch_size=token_batch_size,
    )
    full_prompt_lengths = [
        catalog + suffix
        for catalog, suffix in zip(catalog_lengths, suffix_lengths, strict=True)
    ]
    target_lengths = [body + 1 for body in body_lengths]
    return {
        "assistant_decisions": len(suffixes),
        "conversation_categories": dict(sorted(categories.items())),
        "conversation_shape": {
            "multi_turn": len(conversations) - simple_conversations,
            "simple": simple_conversations,
        },
        "decision_kinds": dict(sorted(decision_kinds.items())),
        "rows": len(conversations),
        "tokens": _token_vectors_audit(
            full_prompt_tokens=full_prompt_lengths,
            prompt_suffix_tokens=suffix_lengths,
            target_tokens=target_lengths,
        ),
        "tool_names": dict(sorted(tool_names.items())),
    }


def _balanced_shortest(
    records: Sequence[FormatBootstrapRecord],
    *,
    phase: str,
    count: int,
) -> list[FormatBootstrapRecord]:
    if count > len(records):
        raise ValueError(
            f"format-bootstrap phase {phase!r} requests {count} rows, "
            f"but only {len(records)} unique eligible rows exist"
        )
    by_bucket: dict[str, list[FormatBootstrapRecord]] = defaultdict(list)
    for record in records:
        by_bucket[record.bucket].append(record)
    for bucket_records in by_bucket.values():
        bucket_records.sort(
            key=lambda record: (
                record.target_tokens,
                record.prompt_suffix_tokens,
                record.full_prompt_tokens,
                record.semantic_sha256,
                record.source_row_number,
            )
        )
    buckets = sorted(
        by_bucket,
        key=lambda bucket: (
            hashlib.sha256(f"{phase}\x00{bucket}".encode("utf-8")).hexdigest(),
            bucket,
        ),
    )
    cursors = {bucket: 0 for bucket in buckets}
    selected: list[FormatBootstrapRecord] = []
    active = list(buckets)
    while len(selected) < count:
        next_active: list[str] = []
        for bucket in active:
            cursor = cursors[bucket]
            bucket_records = by_bucket[bucket]
            if cursor < len(bucket_records):
                if len(selected) < count:
                    selected.append(bucket_records[cursor])
                    cursors[bucket] = cursor + 1
                if cursors[bucket] < len(bucket_records):
                    next_active.append(bucket)
            if len(selected) == count:
                break
        if len(selected) == count:
            break
        if not next_active:
            raise RuntimeError("balanced format-bootstrap selector exhausted unexpectedly")
        active = next_active
    return selected


def select_format_bootstrap(
    conversations: Sequence[Conversation],
    tokenizer,
    phase_rows: Mapping[str, int],
    *,
    token_batch_size: int = 512,
) -> FormatBootstrapSelection:
    """Select a deterministic, balanced, short-first continuation-SFT curriculum.

    Lengths use the supplied tokenizer.  The target count includes EOS because that is the exact
    SFT loss mask, while ``full_prompt_tokens`` includes the complete function catalog.
    """

    phases = _phase_counts(phase_rows)
    if isinstance(token_batch_size, bool) or not isinstance(token_batch_size, int):
        raise TypeError("token_batch_size must be an integer")
    if token_batch_size < 1:
        raise ValueError("token_batch_size must be positive")

    catalog_cache = FunctionCatalogCache()
    catalog_token_counts: dict[int, int] = {}
    pending: list[_PendingRecord] = []
    excluded = Counter()
    for source_row_number, conversation in enumerate(conversations, start=1):
        phase = classify_format_bootstrap_phase(conversation)
        if phase is None:
            excluded["non_simple_conversations"] += 1
            continue
        turns = assistant_training_turns(conversation, catalog_cache=catalog_cache)
        if len(turns) != 1:
            raise RuntimeError("simple format-bootstrap row must have exactly one assistant turn")
        tools_id = id(conversation.tools)
        catalog_tokens = catalog_token_counts.get(tools_id)
        if catalog_tokens is None:
            catalog = catalog_cache.entry(conversation.tools).text + BPE_EOS
            catalog_tokens = len(tokenizer.encode(catalog))
            catalog_token_counts[tools_id] = catalog_tokens
        assistant = conversation.messages[1]
        calls = assistant.tool_calls
        tool_names = tuple(call.name for call in calls)
        category = str(
            conversation.meta.get("category", conversation.meta.get("kind", "unknown"))
        )
        pending.append(
            _PendingRecord(
                conversation=conversation,
                source_row_number=source_row_number,
                phase=phase,
                bucket=_record_bucket(phase, conversation, calls),
                semantic_sha256=conversation_semantic_sha256(conversation),
                row_sha256=hashlib.sha256(
                    (conversation.to_json() + "\n").encode("utf-8")
                ).hexdigest(),
                prompt_suffix=turns[0].prompt_suffix,
                body=turns[0].body,
                catalog_tokens=catalog_tokens,
                category=category,
                tool_names=tool_names,
                argument_count=sum(len(call.arguments) for call in calls),
            )
        )

    suffix_lengths = batched_token_lengths(
        tokenizer,
        (record.prompt_suffix for record in pending),
        batch_size=token_batch_size,
    )
    body_lengths = batched_token_lengths(
        tokenizer,
        (record.body for record in pending),
        batch_size=token_batch_size,
    )
    if len(suffix_lengths) != len(pending) or len(body_lengths) != len(pending):
        raise RuntimeError("format-bootstrap token lengths do not align with source rows")

    unique_by_semantic: dict[str, FormatBootstrapRecord] = {}
    semantic_duplicates = 0
    for item, suffix_tokens, body_tokens in zip(
        pending,
        suffix_lengths,
        body_lengths,
        strict=True,
    ):
        record = FormatBootstrapRecord(
            conversation=item.conversation,
            source_row_number=item.source_row_number,
            phase=item.phase,
            bucket=item.bucket,
            semantic_sha256=item.semantic_sha256,
            row_sha256=item.row_sha256,
            prompt_suffix_tokens=suffix_tokens,
            full_prompt_tokens=item.catalog_tokens + suffix_tokens,
            target_tokens=body_tokens + 1,
            category=item.category,
            tool_names=item.tool_names,
            argument_count=item.argument_count,
        )
        previous = unique_by_semantic.get(record.semantic_sha256)
        if previous is None:
            unique_by_semantic[record.semantic_sha256] = record
            continue
        semantic_duplicates += 1
        if (
            record.target_tokens,
            record.prompt_suffix_tokens,
            record.source_row_number,
        ) < (
            previous.target_tokens,
            previous.prompt_suffix_tokens,
            previous.source_row_number,
        ):
            unique_by_semantic[record.semantic_sha256] = record

    eligible_by_phase: dict[str, list[FormatBootstrapRecord]] = {
        phase: [] for phase in FORMAT_BOOTSTRAP_PHASES
    }
    for record in unique_by_semantic.values():
        eligible_by_phase[record.phase].append(record)

    selected: list[FormatBootstrapRecord] = []
    phase_audits: dict[str, Any] = {}
    for phase in FORMAT_BOOTSTRAP_PHASES:
        eligible = eligible_by_phase[phase]
        phase_selected = _balanced_shortest(
            eligible,
            phase=phase,
            count=phases[phase],
        )
        start = len(selected) + 1
        selected.extend(phase_selected)
        end = len(selected)
        phase_audits[phase] = {
            "buckets": {
                "eligible": _counter_dict([record.bucket for record in eligible]),
                "selected": _counter_dict([record.bucket for record in phase_selected]),
            },
            "categories": {
                "eligible": _counter_dict([record.category for record in eligible]),
                "selected": _counter_dict([record.category for record in phase_selected]),
            },
            "eligible_rows": len(eligible),
            "output_positions_one_based": {
                "end": end,
                "start": start if phase_selected else 0,
            },
            "selected_rows": len(phase_selected),
            "selected_source_row_numbers": [
                record.source_row_number for record in phase_selected
            ],
            "row_set_sha256": _fingerprint_set_sha256(
                [record.row_sha256 for record in phase_selected]
            ),
            "semantic_set_sha256": _fingerprint_set_sha256(
                [record.semantic_sha256 for record in phase_selected]
            ),
            "tokens": {
                "eligible": _token_audit(eligible),
                "selected": _token_audit(phase_selected),
            },
            "tool_names": {
                "eligible": _counter_dict(
                    [name for record in eligible for name in record.tool_names]
                ),
                "selected": _counter_dict(
                    [name for record in phase_selected for name in record.tool_names]
                ),
            },
        }

    selected_semantic = [record.semantic_sha256 for record in selected]
    selected_rows = [record.row_sha256 for record in selected]
    if len(set(selected_semantic)) != len(selected_semantic):
        raise RuntimeError("format-bootstrap selection contains duplicate semantic rows")
    source_semantic = [record.semantic_sha256 for record in unique_by_semantic.values()]
    audit_core = {
        "algorithm": FORMAT_BOOTSTRAP_ALGORITHM,
        "consumption_contract": {
            "configured_data_sampling": "absent",
            "resolved_sft_mode": "source_order_wrapping_v1",
            "shuffle": False,
            "warning": (
                "quota_stratified_no_replacement_v1 is intentionally incompatible because it "
                "reorders assistant decisions across curriculum phases"
            ),
        },
        "fingerprint_contract": {
            "balanced_buckets": (
                "phase-specific bucket round-robin; bucket order is SHA-256(phase NUL bucket)"
            ),
            "canonical_row": (
                "sha256(Conversation.to_json UTF-8 followed by LF; metadata included)"
            ),
            "row_tie_break": (
                "target tokens including EOS, prompt-suffix tokens, full-prompt tokens, "
                "semantic SHA-256, one-based source row"
            ),
            "semantic_row": (
                "sha256(canonical compact sorted-key JSON of messages+tools; meta excluded)"
            ),
        },
        "output": {
            "assistant_decisions": len(selected),
            "ordered_row_sha256": _ordered_fingerprint_sha256(selected_rows),
            "ordered_semantic_sha256": _ordered_fingerprint_sha256(selected_semantic),
            "rows": len(selected),
            "row_set_sha256": _fingerprint_set_sha256(selected_rows),
            "semantic_set_sha256": _fingerprint_set_sha256(selected_semantic),
            "tokens": _token_audit(selected),
            "unique_semantic_rows": len(set(selected_semantic)),
        },
        "phase_order": list(FORMAT_BOOTSTRAP_PHASES),
        "phases": phase_audits,
        "schema_version": FORMAT_BOOTSTRAP_SCHEMA_VERSION,
        "source": {
            "assistant_decisions": len(pending),
            "eligible_rows": len(pending),
            "excluded": dict(sorted(excluded.items())),
            "phase_eligible_rows": {
                phase: len(eligible_by_phase[phase]) for phase in FORMAT_BOOTSTRAP_PHASES
            },
            "rows": len(conversations),
            "semantic_duplicates_excluded": semantic_duplicates,
            "semantic_set_sha256": _fingerprint_set_sha256(source_semantic),
            "unique_eligible_semantic_rows": len(unique_by_semantic),
        },
    }
    audit = {
        **audit_core,
        "audit_sha256": _sha256_object(audit_core),
    }
    return FormatBootstrapSelection(records=tuple(selected), audit=audit)


def _repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/localagent").is_dir():
            return candidate
    return Path.cwd()


def _resolve_path(value: Any, *, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_bound_artifact(
    value: Any,
    *,
    root: Path,
    expected_split: str,
    label: str,
) -> VerifiedConversationArtifact:
    binding = _validate_exact_keys(value, _ARTIFACT_BINDING_KEYS, label=label)
    expected_identity = binding["expected_identity"]
    if not isinstance(expected_identity, Mapping):
        raise TypeError(f"{label}.expected_identity must be a mapping")
    sidecar = expected_identity.get("sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError(f"{label}.expected_identity.sidecar must be a mapping")
    expected_sidecar = FileIdentity(
        bytes=sidecar.get("bytes"),
        sha256=sidecar.get("sha256"),
    )
    artifact = load_verified_conversation_artifact(
        _resolve_path(binding["path"], root=root, label=f"{label}.path"),
        config_path=_resolve_path(
            binding["generator_config"],
            root=root,
            label=f"{label}.generator_config",
        ),
        expected_split=expected_split,
        manifest_path=_resolve_path(
            binding["manifest"],
            root=root,
            label=f"{label}.manifest",
        ),
        expected_rule_verified=True,
        environment_policy="forbid",
        expected_manifest_identity=expected_sidecar,
    )
    actual_identity = artifact.lineage_identity()
    if actual_identity != dict(expected_identity):
        raise ValueError(f"{label} complete artifact identity mismatch")
    return artifact


def _load_bound_tokenizer(value: Any, *, root: Path):
    config = _validate_exact_keys(value, _TOKENIZER_KEYS, label="tokenizer")
    if config["kind"] != "bpe":
        raise ValueError("format-bootstrap tokenizer.kind must be 'bpe'")
    path = _resolve_path(config["path"], root=root, label="tokenizer.path")
    payload = path.read_bytes()
    actual_identity = FileIdentity.from_bytes(payload).as_dict()
    if config["identity"] != actual_identity:
        raise ValueError("format-bootstrap tokenizer identity mismatch")
    return load_tokenizer("bpe", path), actual_identity


def _json_primitive(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    return None


def _walk_argument_values(value: Any):
    primitive = _json_primitive(value)
    if primitive is not None:
        yield primitive
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_argument_values(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk_argument_values(child)


def _has_enum_argument(conversation: Conversation) -> bool:
    registry = {tool.name: tool for tool in conversation.tools}
    for message in conversation.messages:
        for call in message.tool_calls:
            properties = registry[call.name].parameters.get("properties", {})
            for name, value in call.arguments.items():
                schema = properties.get(name, {})
                if "enum" in schema and value in schema["enum"]:
                    return True
    return False


def _selected_manifest_counts(
    conversations: Sequence[Conversation],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    argument_value_counts = Counter()
    behavior_counts = {name: 0 for name in parent_manifest["behavior_counts"]}
    assistant_tool_calls = 0
    irrelevance = 0
    parallel = 0
    single_call = 0
    text = 0
    plan_lengths = Counter()
    for conversation in conversations:
        category = str(
            conversation.meta.get("category", conversation.meta.get("kind", "unknown"))
        )
        is_irrelevant = category == "no_tool"
        irrelevance += int(is_irrelevant)
        plan_len = conversation.meta.get("plan_len")
        if isinstance(plan_len, int) and not isinstance(plan_len, bool):
            plan_lengths[str(plan_len)] += 1
        calls = [
            call
            for message in conversation.messages
            if message.role == Role.assistant
            for call in message.tool_calls
        ]
        assistant_tool_calls += len(calls)
        has_parallel = any(
            len(message.tool_calls) > 1
            for message in conversation.messages
            if message.role == Role.assistant
        )
        parallel += int(has_parallel)
        if len(calls) == 1:
            single_call += 1
        if not calls and not is_irrelevant:
            text += 1

        primitive_types = set()
        for call in calls:
            for primitive in _walk_argument_values(call.arguments):
                argument_value_counts[primitive] += 1
                primitive_types.add(primitive)
        if "parallel_calls" in behavior_counts:
            behavior_counts["parallel_calls"] += int(has_parallel)
        if "multiple_arguments" in behavior_counts:
            behavior_counts["multiple_arguments"] += int(
                any(len(call.arguments) >= 2 for call in calls)
            )
        if "explicit_restraint" in behavior_counts:
            behavior_counts["explicit_restraint"] += int(
                is_irrelevant or plan_len == 0
            )
        if "enum_arguments" in behavior_counts:
            behavior_counts["enum_arguments"] += int(_has_enum_argument(conversation))
        for primitive in ("boolean", "integer", "number"):
            key = f"{primitive}_arguments"
            if key in behavior_counts:
                behavior_counts[key] += int(primitive in primitive_types)

    return {
        "argument_value_counts": dict(sorted(argument_value_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "irrelevance": irrelevance,
        "multi_turn": 0,
        "plan_length_counts": dict(sorted(plan_lengths.items())),
        "single_turn": len(conversations),
        "structural_counts": {
            "assistant_tool_calls": assistant_tool_calls,
            "irrelevance_conversations": irrelevance,
            "multi_turn_conversations": 0,
            "parallel_call_conversations": parallel,
            "single_call_conversations": single_call,
            "text_conversations": text,
        },
    }


def _build_manifest(
    *,
    source: VerifiedConversationArtifact,
    config_identity: FileIdentity,
    output_identity: FileIdentity,
    selection: FormatBootstrapSelection,
    overlap: ConversationOverlapAudit,
) -> tuple[dict[str, Any], bytes]:
    manifest = json.loads(canonical_json_bytes(source.manifest))
    counts = _selected_manifest_counts(selection.conversations, source.manifest)
    manifest.update(counts)
    manifest.update(
        {
            "complexity_contract": {
                "curriculum_order": list(FORMAT_BOOTSTRAP_PHASES),
                "environment_executed": False,
                "selector": FORMAT_BOOTSTRAP_ALGORITHM,
                "source_rows_are_unchanged": True,
            },
            "conversation_serialization": CONVERSATION_SERIALIZATION,
            "coverage_contract": {
                "format_bootstrap": selection.audit,
                "semantics": (
                    "exact configured phase counts; balanced within phase; unchanged source rows"
                ),
                "source_artifact": source.lineage_identity(),
            },
            "generator_config": config_identity.as_dict(),
            "kind": MANIFEST_KIND,
            "manifest_self_sha256": manifest.get("manifest_self_sha256"),
            "output_bytes": output_identity.bytes,
            "output_sha256": output_identity.sha256,
            "rows": len(selection.records),
            "rule_verification_scope": [
                *source.manifest["rule_verification_scope"],
                "provenance_bound_unchanged_train_subset",
                "deterministic_format_bootstrap_selector",
                "heldout_semantic_and_rendered_prompt_overlap_zero",
            ],
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "verification_claim": (
                "rule_audited_parent_subset_with_heldout_overlap_rejection"
            ),
        }
    )
    split_contract = dict(manifest["split_contract"])
    split_contract["format_bootstrap"] = {
        "eval_rendered_prompt_overlap": len(overlap.rendered_prompt_overlap_sha256),
        "eval_semantic_overlap": len(overlap.semantic_overlap_sha256),
        "source_rows_are_unchanged": True,
        "source_split": source.identity.split,
    }
    manifest["split_contract"] = split_contract
    manifest.pop("manifest_self_sha256", None)
    return self_hashed_manifest(manifest)


def _stage_temp_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _stage_conversations(
    path: Path,
    conversations: Sequence[Conversation],
) -> tuple[Path, FileIdentity]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for conversation in conversations:
                payload = (conversation.to_json() + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
                byte_count += len(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, FileIdentity(bytes=byte_count, sha256=digest.hexdigest())


def _receipt_bytes(value: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if "receipt_self_sha256" in value:
        raise ValueError("unsigned format-bootstrap receipt contains receipt_self_sha256")
    core = dict(value)
    receipt = {
        **core,
        "receipt_self_sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest(),
    }
    return receipt, canonical_json_bytes(receipt)


def build_format_bootstrap(config_path: str | Path) -> dict[str, Any]:
    """Build, seal, reload, and return one deterministic format-bootstrap receipt."""

    config_file = Path(config_path).resolve()
    config_payload = config_file.read_bytes()
    config = yaml.safe_load(config_payload.decode("utf-8", errors="strict"))
    config = _validate_exact_keys(config, _CONFIG_KEYS, label="format-bootstrap config")
    if config["kind"] != FORMAT_BOOTSTRAP_CONFIG_KIND:
        raise ValueError(
            f"format-bootstrap config.kind must be {FORMAT_BOOTSTRAP_CONFIG_KIND!r}"
        )
    if config["schema_version"] != FORMAT_BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError(
            "unsupported format-bootstrap config schema_version: "
            f"{config['schema_version']!r}"
        )
    if config["algorithm"] != FORMAT_BOOTSTRAP_ALGORITHM:
        raise ValueError(
            f"format-bootstrap algorithm must be {FORMAT_BOOTSTRAP_ALGORITHM!r}"
        )
    if config["prompt_contract"] != OPENAI_FULL_CATALOG_V1:
        raise ValueError(
            "format-bootstrap prompt_contract must be openai_full_catalog_v1"
        )

    root = _repository_root(config_file)
    source = _load_bound_artifact(
        config["source"],
        root=root,
        expected_split="train",
        label="source",
    )
    evaluation = _load_bound_artifact(
        config["evaluation_holdout"],
        root=root,
        expected_split="eval",
        label="evaluation_holdout",
    )
    tokenizer, tokenizer_identity = _load_bound_tokenizer(config["tokenizer"], root=root)
    output_path = _resolve_path(config["out"], root=root, label="out")
    manifest_path = _resolve_path(config["manifest"], root=root, label="manifest")
    receipt_path = _resolve_path(config["receipt"], root=root, label="receipt")
    destinations = {output_path.resolve(), manifest_path.resolve(), receipt_path.resolve()}
    if len(destinations) != 3:
        raise ValueError("format-bootstrap output, manifest, and receipt paths must be distinct")
    protected_inputs = {
        config_file,
        source.data_path.resolve(),
        source.manifest_path.resolve(),
        source.config_path.resolve(),
        evaluation.data_path.resolve(),
        evaluation.manifest_path.resolve(),
        evaluation.config_path.resolve(),
        _resolve_path(config["tokenizer"]["path"], root=root, label="tokenizer.path").resolve(),
    }
    if destinations & protected_inputs:
        raise ValueError("format-bootstrap destination path would overwrite a bound input")
    if not str(manifest_path).endswith(".manifest.v1.json"):
        raise ValueError("format-bootstrap manifest must use the versioned .manifest.v1.json suffix")

    phases = _phase_counts(config["phases"])
    selection = select_format_bootstrap(source.conversations, tokenizer, phases)
    corpus_analysis = {
        "evaluation_holdout": analyze_conversation_lengths(
            evaluation.conversations,
            tokenizer,
        ),
        "source": analyze_conversation_lengths(
            source.conversations,
            tokenizer,
        ),
    }
    overlap = assert_no_conversation_overlap(
        selection.conversations,
        evaluation.conversations,
        left_label="format-bootstrap train",
        right_label="heldout eval",
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )

    config_identity = FileIdentity.from_bytes(config_payload)
    output_temp, output_identity = _stage_conversations(
        output_path,
        selection.conversations,
    )
    manifest, manifest_payload = _build_manifest(
        source=source,
        config_identity=config_identity,
        output_identity=output_identity,
        selection=selection,
        overlap=overlap,
    )
    manifest_identity = FileIdentity.from_bytes(manifest_payload)
    manifest_temp = _stage_temp_bytes(manifest_path, manifest_payload)
    config_display_path = (
        str(config_file.relative_to(root)) if config_file.is_relative_to(root) else str(config_file)
    )
    receipt_core = {
        "algorithm": FORMAT_BOOTSTRAP_ALGORITHM,
        "config": {
            **config_identity.as_dict(),
            "path": config_display_path,
        },
        "corpus_analysis": corpus_analysis,
        "evaluation_holdout": evaluation.lineage_identity(),
        "kind": FORMAT_BOOTSTRAP_RECEIPT_KIND,
        "output": {
            "jsonl": {
                **output_identity.as_dict(),
                "path": str(config["out"]),
            },
            "manifest": {
                **manifest_identity.as_dict(),
                "manifest_self_sha256": manifest["manifest_self_sha256"],
                "path": str(config["manifest"]),
            },
            "rows": len(selection.records),
        },
        "overlap_audit": overlap.as_dict(),
        "prompt_contract": OPENAI_FULL_CATALOG_V1,
        "schema_version": FORMAT_BOOTSTRAP_SCHEMA_VERSION,
        "selection": selection.audit,
        "source": source.lineage_identity(),
        "tokenizer": {
            **tokenizer_identity,
            "kind": "bpe",
            "path": str(config["tokenizer"]["path"]),
        },
    }
    receipt, receipt_payload = _receipt_bytes(receipt_core)
    receipt_temp = _stage_temp_bytes(receipt_path, receipt_payload)

    try:
        os.replace(output_temp, output_path)
        os.replace(manifest_temp, manifest_path)
        os.replace(receipt_temp, receipt_path)
    finally:
        output_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        receipt_temp.unlink(missing_ok=True)

    rebound = load_verified_conversation_artifact(
        output_path,
        config_path=config_file,
        expected_split="train",
        manifest_path=manifest_path,
        expected_rule_verified=True,
        environment_policy="forbid",
        expected_manifest_identity=manifest_identity,
    )
    if rebound.identity.jsonl != output_identity:
        raise RuntimeError("published format-bootstrap JSONL identity changed after sealing")
    if tuple(conversation_semantic_sha256(row) for row in rebound.conversations) != tuple(
        record.semantic_sha256 for record in selection.records
    ):
        raise RuntimeError("published format-bootstrap row order disagrees with selection audit")
    if tuple(row.to_json() for row in rebound.conversations) != tuple(
        record.conversation.to_json() for record in selection.records
    ):
        raise RuntimeError("published format-bootstrap rows are not unchanged source rows")
    return receipt
