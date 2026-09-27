"""Deterministic broad-replay plus strict-format SFT decision ordering.

The first format-only recovery child demonstrated that a tiny policy can learn to terminate while
catastrophically forgetting the broader assistant distribution.  This module defines the bounded
counter-experiment: each fixed cycle mixes quota-stratified decisions from the full train artifact
with one decision from every sealed format-bootstrap phase.

Only assistant-decision keys are reordered.  Conversations remain unchanged, provenance-bound
``Conversation`` rows, and the returned order is a complete permutation suitable for the shared
SFT runner and no-model stage planner.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from localagent.data.conversation_artifact import conversation_semantic_sha256
from localagent.data.decision_quota_order import (
    DecisionKey,
    order_assistant_decisions,
    quota_sampling_contract,
)
from localagent.data.format_bootstrap import (
    FORMAT_BOOTSTRAP_PHASES,
    classify_format_bootstrap_phase,
)
from localagent.data.prompt_contract import assistant_training_turns

MIXED_REPLAY_SAMPLING_MODE = "general_format_mixed_no_replacement_v2"
MIXED_REPLAY_ORDERING_CONTRACT = "general-format-mixed-decision-order-v2"
GENERAL_COVERAGE_SPREAD_CONTRACT = "centered-quota-coverage-spread-v1"
PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE = "parent_quota_update_blocks_with_format_pulses_v3"
PARENT_ANCHORED_FORMAT_PULSE_ORDERING_CONTRACT = "parent-quota-update-blocks-with-format-pulses-v3"
PHASE_ROUND_ROBIN_CONTRACT = "phase_round_robin_v1"
CENTERED_UPDATE_QUANTILES_CONTRACT = "centered_update_quantiles_v1"

_CONFIG_KEYS = frozenset(
    {
        "cycle",
        "exclude_format_semantic_overlap",
        "format_source_index",
        "general_source_index",
        "mode",
    }
)
_CYCLE_LABELS = frozenset({"general", *FORMAT_BOOTSTRAP_PHASES})
_PARENT_ANCHORED_CONFIG_KEYS = frozenset(
    {
        "expected_parent_order_sha256",
        "expected_parent_prefix_sha256",
        "format_pulses",
        "format_source_index",
        "general_source_index",
        "mode",
        "parent_prefix_decisions",
        "update_decisions",
    }
)
_FORMAT_PULSE_CONFIG_KEYS = frozenset(
    {
        "count",
        "phase_order",
        "position_contract",
        "rows_per_phase",
        "within_pulse_order",
    }
)

__all__ = [
    "CENTERED_UPDATE_QUANTILES_CONTRACT",
    "GENERAL_COVERAGE_SPREAD_CONTRACT",
    "MIXED_REPLAY_ORDERING_CONTRACT",
    "MIXED_REPLAY_SAMPLING_MODE",
    "PARENT_ANCHORED_FORMAT_PULSE_ORDERING_CONTRACT",
    "PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE",
    "PHASE_ROUND_ROBIN_CONTRACT",
    "mixed_replay_sampling_window",
    "parent_anchored_format_pulse_sampling_window",
]


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_key_sha256(keys: Sequence[DecisionKey]) -> str:
    return _sha256_json(
        [[conversation_index, message_index] for conversation_index, message_index in keys]
    )


def _validate_source_index(value: Any, *, label: str, source_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"data.sampling.{label} must be an integer")
    if not 0 <= value < source_count:
        raise ValueError(f"data.sampling.{label} must select one configured conversation source")
    return value


def _natural_decision_keys(conversations: Sequence[Any]) -> tuple[DecisionKey, ...]:
    return tuple(
        (conversation_index, turn.message_index)
        for conversation_index, conversation in enumerate(conversations)
        for turn in assistant_training_turns(conversation)
    )


def _spread_quota_coverage(
    selected_keys: Sequence[DecisionKey],
    *,
    coverage_decisions: int,
    general_per_cycle: int,
) -> tuple[tuple[DecisionKey, ...], dict[str, Any]]:
    """Spread the quota frontload across the selected horizon without changing its set.

    ``order_assistant_decisions`` intentionally puts one representative from every observed
    stratum at the start of its order.  That is useful for fixed-prefix coverage, but a
    continuation pilot would otherwise apply most of its high-learning-rate updates only to that
    rare-first block.  Centered quantiles distribute those same representatives across the whole
    selected horizon; all remaining selected keys retain their proportional-order sequence.
    """

    selected = tuple(selected_keys)
    if not selected:
        raise ValueError("mixed replay general selection must not be empty")
    if (
        isinstance(coverage_decisions, bool)
        or not isinstance(coverage_decisions, int)
        or not 1 <= coverage_decisions <= len(selected)
    ):
        raise ValueError("coverage_decisions must fit within selected general decisions")
    if (
        isinstance(general_per_cycle, bool)
        or not isinstance(general_per_cycle, int)
        or general_per_cycle < 1
    ):
        raise ValueError("general_per_cycle must be a positive integer")
    if len(selected) % general_per_cycle:
        raise ValueError("selected general decisions must fill complete mixed-replay cycles")

    coverage_positions = tuple(
        ((2 * ordinal + 1) * len(selected)) // (2 * coverage_decisions)
        for ordinal in range(coverage_decisions)
    )
    if (
        len(set(coverage_positions)) != coverage_decisions
        or coverage_positions[0] < 0
        or coverage_positions[-1] >= len(selected)
    ):
        raise RuntimeError("centered quota coverage positions are not unique and in range")

    coverage = iter(selected[:coverage_decisions])
    proportional = iter(selected[coverage_decisions:])
    coverage_position_set = set(coverage_positions)
    spread = tuple(
        next(coverage) if position in coverage_position_set else next(proportional)
        for position in range(len(selected))
    )
    if len(set(spread)) != len(spread) or set(spread) != set(selected):
        raise RuntimeError("quota coverage spreading changed the selected general decision set")

    cycle_counts = tuple(
        sum(
            position in coverage_position_set
            for position in range(start, start + general_per_cycle)
        )
        for start in range(0, len(selected), general_per_cycle)
    )
    one_based_positions = [position + 1 for position in coverage_positions]
    return spread, {
        "contract": GENERAL_COVERAGE_SPREAD_CONTRACT,
        "coverage_decisions": coverage_decisions,
        "coverage_position_contract": (
            "floor((2*i+1)*selected_decisions/(2*coverage_decisions)), zero-based i"
        ),
        "coverage_position_index_base": 1,
        "coverage_positions_sha256_encoding": (
            "sha256(canonical compact JSON array of one-based integer positions)"
        ),
        "coverage_positions_sha256": _sha256_json(one_based_positions),
        "first_coverage_position": one_based_positions[0],
        "last_coverage_position": one_based_positions[-1],
        "max_coverage_decisions_per_cycle": max(cycle_counts),
        "min_coverage_decisions_per_cycle": min(cycle_counts),
        "selected_decisions": len(selected),
        "selected_order_sha256": _ordered_key_sha256(spread),
    }


def mixed_replay_sampling_window(
    source_conversations: Sequence[Sequence[Any]],
    *,
    selected_decisions: int,
    sampling_config: Mapping[str, Any],
) -> tuple[tuple[DecisionKey, ...], dict[str, Any]]:
    """Return a complete decision permutation with a sealed mixed-replay prefix.

    The ``general`` pool is quota ordered after excluding every Conversation semantic identity
    present in the format source.  Each format pool preserves the already balanced source order
    within one of the four format-bootstrap phases.  ``sampling_config.cycle`` names the exact
    per-cycle interleave; the fixed horizon must consume an integral number of cycles.
    """

    if not isinstance(sampling_config, Mapping):
        raise TypeError("data.sampling must be a mapping")
    missing = sorted(_CONFIG_KEYS - set(sampling_config))
    extra = sorted(set(sampling_config) - _CONFIG_KEYS)
    if missing or extra:
        raise ValueError(f"mixed replay sampling keys mismatch: missing={missing}, extra={extra}")
    if sampling_config.get("mode") != MIXED_REPLAY_SAMPLING_MODE:
        raise ValueError(f"data.sampling.mode must be {MIXED_REPLAY_SAMPLING_MODE!r}")
    if isinstance(selected_decisions, bool) or not isinstance(selected_decisions, int):
        raise TypeError("selected_decisions must be an integer")
    if selected_decisions < 1:
        raise ValueError("selected_decisions must be positive")

    source_count = len(source_conversations)
    general_source_index = _validate_source_index(
        sampling_config.get("general_source_index"),
        label="general_source_index",
        source_count=source_count,
    )
    format_source_index = _validate_source_index(
        sampling_config.get("format_source_index"),
        label="format_source_index",
        source_count=source_count,
    )
    if general_source_index == format_source_index:
        raise ValueError("mixed replay general and format sources must be distinct")

    exclude_overlap = sampling_config.get("exclude_format_semantic_overlap")
    if not isinstance(exclude_overlap, bool):
        raise TypeError("data.sampling.exclude_format_semantic_overlap must be boolean")
    if not exclude_overlap:
        raise ValueError("mixed replay requires exclude_format_semantic_overlap=true")

    raw_cycle = sampling_config.get("cycle")
    if not isinstance(raw_cycle, list) or not raw_cycle:
        raise TypeError("data.sampling.cycle must be a non-empty list")
    if not all(isinstance(label, str) for label in raw_cycle):
        raise TypeError("data.sampling.cycle entries must be strings")
    cycle = tuple(raw_cycle)
    unknown_labels = sorted(set(cycle) - _CYCLE_LABELS)
    missing_labels = sorted(_CYCLE_LABELS - set(cycle))
    if unknown_labels or missing_labels:
        raise ValueError(
            "mixed replay cycle labels mismatch: "
            f"missing={missing_labels}, unknown={unknown_labels}"
        )
    if selected_decisions % len(cycle):
        raise ValueError(
            "mixed replay horizon must consume an integral number of configured cycles"
        )
    cycle_repetitions = selected_decisions // len(cycle)
    cycle_counts = Counter(cycle)
    selected_counts = {
        label: cycle_counts[label] * cycle_repetitions
        for label in ("general", *FORMAT_BOOTSTRAP_PHASES)
    }

    general_conversations = tuple(source_conversations[general_source_index])
    format_conversations = tuple(source_conversations[format_source_index])
    format_semantics = {
        conversation_semantic_sha256(conversation) for conversation in format_conversations
    }
    general_candidate_indices = tuple(
        index
        for index, conversation in enumerate(general_conversations)
        if conversation_semantic_sha256(conversation) not in format_semantics
    )
    general_candidates = tuple(general_conversations[index] for index in general_candidate_indices)
    if not general_candidates:
        raise ValueError("mixed replay has no non-overlapping general conversations")

    general_ordering = order_assistant_decisions(general_candidates)
    required_general = selected_counts["general"]
    general_contract = quota_sampling_contract(
        general_ordering,
        selected_decisions=required_general,
        require_all_strata=True,
    )
    quota_selected_general_local = tuple(
        (
            general_candidate_indices[candidate_conversation_index],
            message_index,
        )
        for candidate_conversation_index, message_index in general_ordering.keys[:required_general]
    )
    selected_general_local, coverage_spread = _spread_quota_coverage(
        quota_selected_general_local,
        coverage_decisions=general_ordering.audit.frontload_decision_count,
        general_per_cycle=cycle_counts["general"],
    )

    format_phase_keys: dict[str, list[DecisionKey]] = {
        phase: [] for phase in FORMAT_BOOTSTRAP_PHASES
    }
    for conversation_index, conversation in enumerate(format_conversations):
        phase = classify_format_bootstrap_phase(conversation)
        if phase is None:
            raise ValueError(
                "mixed replay format source must contain only simple bootstrap conversations"
            )
        turns = assistant_training_turns(conversation)
        if len(turns) != 1:
            raise ValueError(
                "mixed replay format source rows must contain exactly one assistant decision"
            )
        format_phase_keys[phase].append((conversation_index, turns[0].message_index))

    selected_format_local: dict[str, tuple[DecisionKey, ...]] = {}
    for phase in FORMAT_BOOTSTRAP_PHASES:
        required = selected_counts[phase]
        available = format_phase_keys[phase]
        if required > len(available):
            raise ValueError(
                "mixed replay format phase is too small for the fixed horizon: "
                f"phase={phase}, required={required}, available={len(available)}"
            )
        selected_format_local[phase] = tuple(available[:required])

    source_offsets: list[int] = []
    offset = 0
    for conversations in source_conversations:
        source_offsets.append(offset)
        offset += len(conversations)

    def globalize(source_index: int, key: DecisionKey) -> DecisionKey:
        return source_offsets[source_index] + key[0], key[1]

    general_cursor = 0
    format_cursors = {phase: 0 for phase in FORMAT_BOOTSTRAP_PHASES}
    selected_prefix: list[DecisionKey] = []
    for _ in range(cycle_repetitions):
        for label in cycle:
            if label == "general":
                key = selected_general_local[general_cursor]
                general_cursor += 1
                selected_prefix.append(globalize(general_source_index, key))
            else:
                phase_keys = selected_format_local[label]
                phase_cursor = format_cursors[label]
                format_cursors[label] = phase_cursor + 1
                selected_prefix.append(globalize(format_source_index, phase_keys[phase_cursor]))

    selected_prefix_tuple = tuple(selected_prefix)
    if len(selected_prefix_tuple) != selected_decisions:
        raise RuntimeError("mixed replay selected prefix length drifted")
    if len(set(selected_prefix_tuple)) != len(selected_prefix_tuple):
        raise RuntimeError("mixed replay selected prefix contains duplicate decisions")

    natural_global_keys = tuple(
        globalize(source_index, key)
        for source_index, conversations in enumerate(source_conversations)
        for key in _natural_decision_keys(conversations)
    )
    if len(set(natural_global_keys)) != len(natural_global_keys):
        raise RuntimeError("configured source decision keys are not unique")
    selected_set = set(selected_prefix_tuple)
    if not selected_set <= set(natural_global_keys):
        raise RuntimeError("mixed replay selected a decision outside configured sources")
    complete_order = selected_prefix_tuple + tuple(
        key for key in natural_global_keys if key not in selected_set
    )
    if len(complete_order) != len(natural_global_keys) or set(complete_order) != set(
        natural_global_keys
    ):
        raise RuntimeError("mixed replay order is not a complete decision permutation")

    selected_general_semantics = {
        conversation_semantic_sha256(general_conversations[key[0]])
        for key in selected_general_local
    }
    selected_format_semantics = {
        conversation_semantic_sha256(format_conversations[key[0]])
        for phase in FORMAT_BOOTSTRAP_PHASES
        for key in selected_format_local[phase]
    }
    selected_semantic_overlap = selected_general_semantics & selected_format_semantics
    if selected_semantic_overlap:
        raise RuntimeError("mixed replay selected general/format semantic overlap")

    compact_general_contract = dict(general_contract)
    return complete_order, {
        "contract": MIXED_REPLAY_ORDERING_CONTRACT,
        "mode": MIXED_REPLAY_SAMPLING_MODE,
        "no_replacement": True,
        "selected_decisions": selected_decisions,
        "selected_unique_decisions": len(selected_set),
        "selected_order_sha256": _ordered_key_sha256(selected_prefix_tuple),
        "complete_order_sha256": _ordered_key_sha256(complete_order),
        "cycle": {
            "labels": list(cycle),
            "length": len(cycle),
            "repetitions": cycle_repetitions,
            "selected_counts": selected_counts,
        },
        "general": {
            "source_index": general_source_index,
            "source_conversations": len(general_conversations),
            "candidate_conversations": len(general_candidates),
            "excluded_format_semantic_conversations": (
                len(general_conversations) - len(general_candidates)
            ),
            "selected_decisions": required_general,
            "pre_spread_quota_selection": {
                "applies_to": "selected general decision set before coverage spreading",
                **compact_general_contract,
            },
            "coverage_spread": coverage_spread,
        },
        "format": {
            "source_index": format_source_index,
            "source_conversations": len(format_conversations),
            "available_by_phase": {
                phase: len(format_phase_keys[phase]) for phase in FORMAT_BOOTSTRAP_PHASES
            },
            "selected_by_phase": {
                phase: len(selected_format_local[phase]) for phase in FORMAT_BOOTSTRAP_PHASES
            },
        },
        "selected_general_format_semantic_overlap": 0,
    }


def _validate_exact_sampling_keys(
    value: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected, key=str)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return value


def _positive_sampling_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value


def _sampling_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA-256")
    return value


def _ordered_block_sha256(blocks: Sequence[Sequence[DecisionKey]]) -> str:
    return _sha256_json(
        [
            [[conversation_index, message_index] for conversation_index, message_index in block]
            for block in blocks
        ]
    )


def parent_anchored_format_pulse_sampling_window(
    source_conversations: Sequence[Sequence[Any]],
    *,
    selected_decisions: int,
    sampling_config: Mapping[str, Any],
) -> tuple[tuple[DecisionKey, ...], dict[str, Any]]:
    """Preserve a parent quota prefix in whole updates and insert centered format pulses.

    The parent prefix is the unchanged first ``parent_prefix_decisions`` keys from the complete
    quota order of the configured general source.  It is split into update-sized blocks before
    format-only pulse blocks are inserted, so no parent update is internally reordered.  The
    returned tuple is a complete no-replacement permutation over every configured source.
    """

    config = _validate_exact_sampling_keys(
        sampling_config,
        _PARENT_ANCHORED_CONFIG_KEYS,
        label="data.sampling",
    )
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise TypeError("data.sampling.mode must be a string")
    if mode != PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
        raise ValueError(
            f"data.sampling.mode must be {PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE!r}"
        )
    selected_decisions = _positive_sampling_int(
        selected_decisions,
        label="selected_decisions",
    )

    source_count = len(source_conversations)
    general_source_index = _validate_source_index(
        config.get("general_source_index"),
        label="general_source_index",
        source_count=source_count,
    )
    format_source_index = _validate_source_index(
        config.get("format_source_index"),
        label="format_source_index",
        source_count=source_count,
    )
    if general_source_index == format_source_index:
        raise ValueError("parent-anchored general and format sources must be distinct")

    parent_prefix_decisions = _positive_sampling_int(
        config.get("parent_prefix_decisions"),
        label="data.sampling.parent_prefix_decisions",
    )
    update_decisions = _positive_sampling_int(
        config.get("update_decisions"),
        label="data.sampling.update_decisions",
    )
    if parent_prefix_decisions % update_decisions:
        raise ValueError("data.sampling.parent_prefix_decisions must fill complete parent updates")
    expected_parent_order_sha256 = _sampling_sha256(
        config.get("expected_parent_order_sha256"),
        label="data.sampling.expected_parent_order_sha256",
    )
    expected_parent_prefix_sha256 = _sampling_sha256(
        config.get("expected_parent_prefix_sha256"),
        label="data.sampling.expected_parent_prefix_sha256",
    )

    pulses = _validate_exact_sampling_keys(
        config.get("format_pulses"),
        _FORMAT_PULSE_CONFIG_KEYS,
        label="data.sampling.format_pulses",
    )
    pulse_count = _positive_sampling_int(
        pulses.get("count"),
        label="data.sampling.format_pulses.count",
    )
    rows_per_phase = _positive_sampling_int(
        pulses.get("rows_per_phase"),
        label="data.sampling.format_pulses.rows_per_phase",
    )
    raw_phase_order = pulses.get("phase_order")
    if not isinstance(raw_phase_order, list):
        raise TypeError("data.sampling.format_pulses.phase_order must be a list")
    if raw_phase_order != list(FORMAT_BOOTSTRAP_PHASES):
        raise ValueError(
            "data.sampling.format_pulses.phase_order must be exactly "
            f"{list(FORMAT_BOOTSTRAP_PHASES)!r}"
        )
    phase_order = tuple(raw_phase_order)
    within_pulse_order = pulses.get("within_pulse_order")
    if not isinstance(within_pulse_order, str):
        raise TypeError("data.sampling.format_pulses.within_pulse_order must be a string")
    if within_pulse_order != PHASE_ROUND_ROBIN_CONTRACT:
        raise ValueError(
            f"data.sampling.format_pulses.within_pulse_order must be {PHASE_ROUND_ROBIN_CONTRACT!r}"
        )
    position_contract = pulses.get("position_contract")
    if not isinstance(position_contract, str):
        raise TypeError("data.sampling.format_pulses.position_contract must be a string")
    if position_contract != CENTERED_UPDATE_QUANTILES_CONTRACT:
        raise ValueError(
            "data.sampling.format_pulses.position_contract must be "
            f"{CENTERED_UPDATE_QUANTILES_CONTRACT!r}"
        )

    pulse_decisions = rows_per_phase * len(phase_order)
    if pulse_decisions != update_decisions:
        raise ValueError(
            "format-pulse rows must sum to one complete update: "
            f"rows_per_phase={rows_per_phase}, phases={len(phase_order)}, "
            f"update_decisions={update_decisions}"
        )
    parent_update_count = parent_prefix_decisions // update_decisions
    total_update_count = parent_update_count + pulse_count
    derived_selected_decisions = parent_prefix_decisions + pulse_count * update_decisions
    if selected_decisions != derived_selected_decisions:
        raise ValueError(
            "selected_decisions must equal parent prefix plus complete format pulses: "
            f"expected={derived_selected_decisions}, actual={selected_decisions}"
        )

    general_conversations = tuple(source_conversations[general_source_index])
    format_conversations = tuple(source_conversations[format_source_index])
    general_ordering = order_assistant_decisions(general_conversations)
    actual_parent_order_sha256 = general_ordering.audit.order_sha256
    if actual_parent_order_sha256 != expected_parent_order_sha256:
        raise ValueError(
            "parent quota order SHA-256 mismatch: "
            f"expected={expected_parent_order_sha256}, "
            f"actual={actual_parent_order_sha256}"
        )
    if parent_prefix_decisions > len(general_ordering.keys):
        raise ValueError(
            "parent quota source is too small for the configured prefix: "
            f"required={parent_prefix_decisions}, available={len(general_ordering.keys)}"
        )
    selected_parent_local = tuple(general_ordering.keys[:parent_prefix_decisions])
    actual_parent_prefix_sha256 = _ordered_key_sha256(selected_parent_local)
    if actual_parent_prefix_sha256 != expected_parent_prefix_sha256:
        raise ValueError(
            "parent quota prefix SHA-256 mismatch: "
            f"expected={expected_parent_prefix_sha256}, "
            f"actual={actual_parent_prefix_sha256}"
        )
    parent_blocks_local = tuple(
        selected_parent_local[start : start + update_decisions]
        for start in range(0, parent_prefix_decisions, update_decisions)
    )

    format_phase_keys: dict[str, list[DecisionKey]] = {phase: [] for phase in phase_order}
    for conversation_index, conversation in enumerate(format_conversations):
        phase = classify_format_bootstrap_phase(conversation)
        if phase is None:
            raise ValueError(
                "parent-anchored format source must contain only simple bootstrap conversations"
            )
        turns = assistant_training_turns(conversation)
        if len(turns) != 1:
            raise ValueError(
                "parent-anchored format source rows must contain exactly one assistant decision"
            )
        format_phase_keys[phase].append((conversation_index, turns[0].message_index))

    required_per_phase = pulse_count * rows_per_phase
    selected_format_local: dict[str, tuple[DecisionKey, ...]] = {}
    for phase in phase_order:
        available = format_phase_keys[phase]
        if required_per_phase > len(available):
            raise ValueError(
                "parent-anchored format phase is too small for the configured pulses: "
                f"phase={phase}, required={required_per_phase}, available={len(available)}"
            )
        selected_format_local[phase] = tuple(available[:required_per_phase])

    pulse_blocks_local = tuple(
        tuple(
            selected_format_local[phase][pulse_index * rows_per_phase + row_index]
            for row_index in range(rows_per_phase)
            for phase in phase_order
        )
        for pulse_index in range(pulse_count)
    )
    if any(len(block) != update_decisions for block in pulse_blocks_local):
        raise RuntimeError("format pulse did not produce one complete update")

    source_offsets: list[int] = []
    offset = 0
    for conversations in source_conversations:
        source_offsets.append(offset)
        offset += len(conversations)

    def globalize(source_index: int, key: DecisionKey) -> DecisionKey:
        return source_offsets[source_index] + key[0], key[1]

    parent_blocks_global = tuple(
        tuple(globalize(general_source_index, key) for key in block)
        for block in parent_blocks_local
    )
    selected_format_global = {
        phase: tuple(globalize(format_source_index, key) for key in selected_format_local[phase])
        for phase in phase_order
    }
    pulse_blocks_global = tuple(
        tuple(globalize(format_source_index, key) for key in block) for block in pulse_blocks_local
    )

    pulse_positions_zero_based = tuple(
        ((2 * pulse_index + 1) * total_update_count) // (2 * pulse_count)
        for pulse_index in range(pulse_count)
    )
    if (
        len(set(pulse_positions_zero_based)) != pulse_count
        or pulse_positions_zero_based[0] < 0
        or pulse_positions_zero_based[-1] >= total_update_count
    ):
        raise RuntimeError("centered format-pulse positions are not unique and in range")
    pulse_position_set = set(pulse_positions_zero_based)
    parent_block_cursor = 0
    pulse_block_cursor = 0
    interleaved_update_blocks: list[tuple[DecisionKey, ...]] = []
    for update_index in range(total_update_count):
        if update_index in pulse_position_set:
            interleaved_update_blocks.append(pulse_blocks_global[pulse_block_cursor])
            pulse_block_cursor += 1
        else:
            interleaved_update_blocks.append(parent_blocks_global[parent_block_cursor])
            parent_block_cursor += 1
    if parent_block_cursor != parent_update_count or pulse_block_cursor != pulse_count:
        raise RuntimeError("parent and pulse update block consumption drifted")
    interleaved_update_blocks_tuple = tuple(interleaved_update_blocks)
    selected_prefix_tuple = tuple(key for block in interleaved_update_blocks_tuple for key in block)
    if len(selected_prefix_tuple) != selected_decisions:
        raise RuntimeError("parent-anchored selected prefix length drifted")
    selected_set = set(selected_prefix_tuple)
    if len(selected_set) != len(selected_prefix_tuple):
        raise RuntimeError("parent-anchored selected prefix contains duplicate decisions")

    natural_global_keys = tuple(
        globalize(source_index, key)
        for source_index, conversations in enumerate(source_conversations)
        for key in _natural_decision_keys(conversations)
    )
    natural_set = set(natural_global_keys)
    if len(natural_set) != len(natural_global_keys):
        raise RuntimeError("configured source decision keys are not unique")
    if not selected_set <= natural_set:
        raise RuntimeError("parent-anchored replay selected a decision outside configured sources")
    complete_order = selected_prefix_tuple + tuple(
        key for key in natural_global_keys if key not in selected_set
    )
    if len(complete_order) != len(natural_global_keys) or set(complete_order) != natural_set:
        raise RuntimeError("parent-anchored replay order is not a complete decision permutation")

    selected_parent_semantics = {
        conversation_semantic_sha256(general_conversations[conversation_index])
        for conversation_index, _message_index in selected_parent_local
    }
    overlap_by_phase = {
        phase: sum(
            conversation_semantic_sha256(format_conversations[conversation_index])
            in selected_parent_semantics
            for conversation_index, _message_index in selected_format_local[phase]
        )
        for phase in phase_order
    }
    pulse_positions_one_based = [position + 1 for position in pulse_positions_zero_based]
    parent_prefix_counts = general_ordering.audit.prefix_counts(parent_prefix_decisions)

    return complete_order, {
        "contract": PARENT_ANCHORED_FORMAT_PULSE_ORDERING_CONTRACT,
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "no_replacement": True,
        "selected_decisions": selected_decisions,
        "selected_unique_decisions": len(selected_set),
        "selected_order_sha256": _ordered_key_sha256(selected_prefix_tuple),
        "complete_decisions": len(complete_order),
        "complete_order_sha256": _ordered_key_sha256(complete_order),
        "update_layout": {
            "update_decisions": update_decisions,
            "total_updates": total_update_count,
            "parent_updates": parent_update_count,
            "format_pulse_updates": pulse_count,
            "pulse_positions_index_base": 0,
            "pulse_positions_zero_based": list(pulse_positions_zero_based),
            "pulse_positions_one_based": pulse_positions_one_based,
            "pulse_position_formula": (
                "floor((2*i+1)*total_updates/(2*format_pulse_updates)), zero-based i"
            ),
            "pulse_positions_sha256_encoding": (
                "sha256(canonical compact JSON array of one-based integer positions)"
            ),
            "pulse_positions_sha256": _sha256_json(pulse_positions_one_based),
            "selected_row_label_contract": {
                "parent_label": "general",
                "rule": (
                    "update_index=floor(selected_row_index/update_decisions); "
                    "format pulse iff update_index is in pulse_positions_zero_based; "
                    "within a pulse, label=phase_order["
                    "(selected_row_index%update_decisions)%len(phase_order)]"
                ),
                "selected_row_index_base": 0,
                "update_kind_parent": "parent",
                "update_kind_pulse": "format_pulse",
            },
            "interleaved_update_blocks_sha256_encoding": (
                "sha256(canonical compact JSON nested decision-key blocks)"
            ),
            "interleaved_update_blocks_sha256": _ordered_block_sha256(
                interleaved_update_blocks_tuple
            ),
        },
        "parent": {
            "source_index": general_source_index,
            "source_conversations": len(general_conversations),
            "source_decisions": general_ordering.audit.source_decision_count,
            "quota_ordering_contract": general_ordering.audit.contract,
            "observed_strata": general_ordering.audit.observed_stratum_count,
            "frontload_decisions": general_ordering.audit.frontload_decision_count,
            "prefix_covered_strata": sum(count > 0 for count in parent_prefix_counts.values()),
            "expected_order_sha256": expected_parent_order_sha256,
            "order_sha256": actual_parent_order_sha256,
            "prefix_decisions": parent_prefix_decisions,
            "expected_prefix_sha256": expected_parent_prefix_sha256,
            "prefix_sha256": actual_parent_prefix_sha256,
            "global_prefix_sha256": _ordered_key_sha256(
                tuple(key for block in parent_blocks_global for key in block)
            ),
            "update_blocks": parent_update_count,
            "update_blocks_sha256_encoding": (
                "sha256(canonical compact JSON nested decision-key blocks)"
            ),
            "update_blocks_sha256": _ordered_block_sha256(parent_blocks_global),
        },
        "format_pulses": {
            "source_index": format_source_index,
            "source_conversations": len(format_conversations),
            "count": pulse_count,
            "rows_per_phase": rows_per_phase,
            "phase_order": list(phase_order),
            "within_pulse_order": within_pulse_order,
            "position_contract": position_contract,
            "selected_decisions": pulse_count * update_decisions,
            "available_by_phase": {phase: len(format_phase_keys[phase]) for phase in phase_order},
            "selected_by_phase": {
                phase: len(selected_format_local[phase]) for phase in phase_order
            },
            "selected_phase_local_order_sha256": {
                phase: _ordered_key_sha256(selected_format_local[phase]) for phase in phase_order
            },
            "selected_phase_global_order_sha256": {
                phase: _ordered_key_sha256(selected_format_global[phase]) for phase in phase_order
            },
            "pulse_blocks_sha256_encoding": (
                "sha256(canonical compact JSON nested decision-key blocks)"
            ),
            "pulse_blocks_local_sha256": _ordered_block_sha256(pulse_blocks_local),
            "pulse_blocks_sha256": _ordered_block_sha256(pulse_blocks_global),
        },
        "selected_parent_format_semantic_overlap": {
            "allowed": True,
            "counting_unit": (
                "selected format rows whose Conversation semantic SHA-256 occurs "
                "in the selected parent prefix"
            ),
            "by_phase": overlap_by_phase,
            "total": sum(overlap_by_phase.values()),
        },
    }
