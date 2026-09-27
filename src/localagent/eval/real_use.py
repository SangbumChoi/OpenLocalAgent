"""Fail-closed evaluation gates for public, realistic agent traces.

The core AST scorer in :mod:`localagent.eval.tool_eval` remains the authority for tool calls,
parallel calls, abstention, and teacher-forced multi-turn accuracy.  This module adds the pieces
needed before those numbers can be described as a *public real-use* evaluation:

* validate the public-snapshot ``Conversation.meta`` provenance contract;
* report dataset, category, behavior, capability, and multi-action coverage;
* compare exact observed metrics with caller-declared promotion thresholds; and
* optionally measure sparse-router utilization from telemetry actually exposed by the model.

It does not turn offline gold-history replay into an environment rollout.  The returned contract
states that limitation explicitly, and dense models report router metrics as unavailable rather
than receiving invented diversity numbers.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.prompt_contract import assistant_training_examples
from localagent.data.schema import Conversation, Role
from localagent.eval.tool_eval import (
    AssistantPrediction,
    Predictor,
    match_calls,
    parse_tool_output,
    score_conversations,
)

PUBLIC_GENERATOR = "public_agent_snapshot_v1"
PUBLIC_GROUP = "public_agent"
PUBLIC_KIND = "public_agent_trace"
PUBLIC_VERIFICATION_SCOPE = "schema_catalog_arguments_sequence_and_split_slots"
PUBLIC_BEHAVIORS = frozenset({"action", "abstention", "irrelevance"})

_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_FROZEN_DATASET_BYTES = 1024 * 1024 * 1024
_PROVENANCE_KEYS = frozenset(
    {
        "dataset",
        "subset",
        "revision",
        "record_id",
        "url",
        "license",
        "file_sha256",
        "source_line",
    }
)


def _finite_rate(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return int(value)


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256_text(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


@dataclass(frozen=True)
class RealUseRequirements:
    """Explicit coverage and quality thresholds for one promotion decision.

    Accuracy thresholds are required constructor arguments on purpose.  There is no universal
    production threshold that this evaluator can choose honestly on the caller's behalf.
    """

    min_action_exact: float
    min_tool_call_exact: float
    min_abstention_accuracy: float
    min_irrelevance_accuracy: float
    min_multi_turn_step_exact: float
    min_multi_turn_episode_exact: float
    min_category_action_exact: float
    min_conversations: int = 1
    min_datasets: int = 1
    min_categories: int = 1
    min_action_conversations: int = 1
    min_abstention_conversations: int = 1
    min_irrelevance_conversations: int = 1
    min_multi_turn_action_conversations: int = 1
    required_capabilities: tuple[str, ...] = ()
    require_sparse_router: bool = False
    min_router_utilization: float = 0.0
    min_router_normalized_entropy: float = 0.0
    min_router_category_divergence: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "min_action_exact",
            "min_tool_call_exact",
            "min_abstention_accuracy",
            "min_irrelevance_accuracy",
            "min_multi_turn_step_exact",
            "min_multi_turn_episode_exact",
            "min_category_action_exact",
            "min_router_utilization",
            "min_router_normalized_entropy",
            "min_router_category_divergence",
        ):
            _finite_rate(getattr(self, name), label=name)
        for name in (
            "min_conversations",
            "min_datasets",
            "min_categories",
            "min_action_conversations",
            "min_abstention_conversations",
            "min_irrelevance_conversations",
            "min_multi_turn_action_conversations",
        ):
            _positive_int(getattr(self, name), label=name, allow_zero=True)
        if not isinstance(self.require_sparse_router, bool):
            raise TypeError("require_sparse_router must be boolean")
        capabilities = tuple(self.required_capabilities)
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError("required_capabilities must be sorted and unique")
        for index, capability in enumerate(capabilities):
            _nonempty_text(capability, label=f"required_capabilities[{index}]")


def _public_metadata(conversation: Conversation, *, index: int) -> dict[str, Any]:
    meta = conversation.meta
    if not isinstance(meta, Mapping):
        raise TypeError(f"conversation[{index}].meta must be a mapping")

    expected_constants = {
        "group": PUBLIC_GROUP,
        "kind": PUBLIC_KIND,
        "split": "eval",
        "generator": PUBLIC_GENERATOR,
        "public_data": True,
        "rule_verified": True,
        "model_verified": False,
        "environment_executed": False,
        "verification_scope": PUBLIC_VERIFICATION_SCOPE,
    }
    for key, expected in expected_constants.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"conversation[{index}].meta.{key} must equal {expected!r}, "
                f"got {meta.get(key)!r}"
            )

    category = _nonempty_text(meta.get("category"), label=f"conversation[{index}].meta.category")
    behavior = meta.get("behavior")
    if behavior not in PUBLIC_BEHAVIORS:
        raise ValueError(
            f"conversation[{index}].meta.behavior must be one of "
            f"{sorted(PUBLIC_BEHAVIORS)}"
        )
    action_count = _positive_int(
        meta.get("action_count"),
        label=f"conversation[{index}].meta.action_count",
        allow_zero=True,
    )
    enrichment_level = _positive_int(
        meta.get("enrichment_level"),
        label=f"conversation[{index}].meta.enrichment_level",
        allow_zero=True,
    )
    parent_record_id = _nonempty_text(
        meta.get("parent_record_id"),
        label=f"conversation[{index}].meta.parent_record_id",
    )

    capabilities_value = meta.get("capabilities")
    if not isinstance(capabilities_value, Sequence) or isinstance(
        capabilities_value, (str, bytes, bytearray)
    ):
        raise TypeError(f"conversation[{index}].meta.capabilities must be a sequence")
    capabilities = tuple(capabilities_value)
    if capabilities != tuple(sorted(set(capabilities))):
        raise ValueError(
            f"conversation[{index}].meta.capabilities must be sorted and unique"
        )
    for capability_index, capability in enumerate(capabilities):
        _nonempty_text(
            capability,
            label=f"conversation[{index}].meta.capabilities[{capability_index}]",
        )

    provenance_value = meta.get("provenance")
    if not isinstance(provenance_value, Mapping):
        raise TypeError(f"conversation[{index}].meta.provenance must be a mapping")
    missing = sorted(_PROVENANCE_KEYS - set(provenance_value))
    extra = sorted(set(provenance_value) - _PROVENANCE_KEYS)
    if missing or extra:
        raise ValueError(
            f"conversation[{index}].meta.provenance keys mismatch: "
            f"missing={missing}, extra={extra}"
        )
    provenance = dict(provenance_value)
    for key in ("dataset", "subset", "revision", "record_id", "license"):
        _nonempty_text(
            provenance[key],
            label=f"conversation[{index}].meta.provenance.{key}",
        )
    url = _nonempty_text(
        provenance["url"],
        label=f"conversation[{index}].meta.provenance.url",
    )
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError(f"conversation[{index}] provenance URL must be public HTTPS")
    _sha256_text(
        provenance["file_sha256"],
        label=f"conversation[{index}].meta.provenance.file_sha256",
    )
    source_line = _positive_int(
        provenance["source_line"],
        label=f"conversation[{index}].meta.provenance.source_line",
    )

    actual_action_count = sum(
        len(message.tool_calls)
        for message in conversation.messages
        if message.role == Role.assistant
    )
    if action_count != actual_action_count:
        raise ValueError(
            f"conversation[{index}] action_count={action_count} does not match "
            f"{actual_action_count} assistant tool calls"
        )
    if behavior == "action" and action_count == 0:
        raise ValueError(f"conversation[{index}] action behavior has no action")
    if behavior in {"abstention", "irrelevance"} and action_count != 0:
        raise ValueError(f"conversation[{index}] {behavior} behavior contains an action")

    assistant_decisions = sum(
        message.role == Role.assistant for message in conversation.messages
    )
    tool_decisions = sum(
        message.role == Role.assistant and bool(message.tool_calls)
        for message in conversation.messages
    )
    parallel_decisions = sum(
        message.role == Role.assistant and len(message.tool_calls) > 1
        for message in conversation.messages
    )
    return {
        "category": category,
        "behavior": behavior,
        "action_count": action_count,
        "assistant_decisions": assistant_decisions,
        "tool_decisions": tool_decisions,
        "multi_turn_action": tool_decisions > 1,
        "parallel_decisions": parallel_decisions,
        "capabilities": capabilities,
        "enrichment_level": enrichment_level,
        "parent_record_id": parent_record_id,
        "dataset": provenance["dataset"],
        "subset": provenance["subset"],
        "revision": provenance["revision"],
        "record_id": provenance["record_id"],
        "license": provenance["license"],
        "file_sha256": provenance["file_sha256"],
        "source_line": source_line,
    }


def audit_public_real_use_cases(conversations: Sequence[Conversation]) -> dict[str, Any]:
    """Validate and summarize one held-out public real-use Conversation suite."""

    if not conversations:
        raise ValueError("real-use evaluation requires at least one conversation")
    records = [
        _public_metadata(conversation, index=index)
        for index, conversation in enumerate(conversations)
    ]
    datasets = Counter(record["dataset"] for record in records)
    categories = Counter(record["category"] for record in records)
    behaviors = Counter(record["behavior"] for record in records)
    capabilities = Counter(
        capability for record in records for capability in record["capabilities"]
    )
    licenses = Counter(record["license"] for record in records)
    source_revisions = sorted(
        {
            (record["dataset"], record["subset"], record["revision"], record["file_sha256"])
            for record in records
        }
    )
    descriptors = [
        {
            "dataset": record["dataset"],
            "subset": record["subset"],
            "revision": record["revision"],
            "record_id": record["record_id"],
            "source_line": record["source_line"],
            "category": record["category"],
            "behavior": record["behavior"],
            "action_count": record["action_count"],
            "capabilities": list(record["capabilities"]),
            "parent_record_id": record["parent_record_id"],
        }
        for record in records
    ]
    return {
        "contract": {
            "generator": PUBLIC_GENERATOR,
            "group": PUBLIC_GROUP,
            "kind": PUBLIC_KIND,
            "split": "eval",
            "public_data": True,
            "rule_verified": True,
            "model_verified": False,
            "environment_executed": False,
            "verification_scope": PUBLIC_VERIFICATION_SCOPE,
            "evaluation_semantics": (
                "offline strict AST scoring with gold prior history; not a live environment "
                "rollout or official upstream benchmark score"
            ),
        },
        "case_set_sha256": hashlib.sha256(canonical_json_bytes(descriptors)).hexdigest(),
        "conversations": len(records),
        "assistant_decisions": sum(record["assistant_decisions"] for record in records),
        "action_conversations": behaviors["action"],
        "abstention_conversations": behaviors["abstention"],
        "irrelevance_conversations": behaviors["irrelevance"],
        "multi_turn_action_conversations": sum(
            record["multi_turn_action"] for record in records
        ),
        "parallel_action_conversations": sum(
            record["parallel_decisions"] > 0 for record in records
        ),
        "tool_decisions": sum(record["tool_decisions"] for record in records),
        "actions": sum(record["action_count"] for record in records),
        "datasets": dict(sorted(datasets.items())),
        "categories": dict(sorted(categories.items())),
        "behaviors": dict(sorted(behaviors.items())),
        "capabilities": dict(sorted(capabilities.items())),
        "licenses": dict(sorted(licenses.items())),
        "source_revisions": [
            {
                "dataset": dataset,
                "subset": subset,
                "revision": revision,
                "file_sha256": file_sha256,
            }
            for dataset, subset, revision, file_sha256 in source_revisions
        ],
    }


def _entropy(load: Sequence[float]) -> tuple[float, float, float]:
    raw = -sum(value * math.log(value) for value in load if value > 0.0)
    normalized = raw / math.log(len(load)) if len(load) > 1 else 0.0
    return raw, normalized, math.exp(raw)


def _router_load(counts: Sequence[int]) -> list[float]:
    assignments = sum(counts)
    if assignments <= 0:
        raise ValueError("enabled router diagnostics must contain at least one assignment")
    return [count / assignments for count in counts]


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(l_value + r_value) / 2.0 for l_value, r_value in zip(left, right, strict=True)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint, strict=True)
            if value > 0.0
        )

    return (divergence(left) + divergence(right)) / (2.0 * math.log(2.0))


def summarize_router_diagnostics(
    records: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate JSON-safe per-prompt router telemetry without trusting derived load fields."""

    if not records:
        return {
            "available": False,
            "enabled": False,
            "reason": "no_router_diagnostics",
            "cases": 0,
        }

    enabled_values = []
    for index, (_category, diagnostics) in enumerate(records):
        enabled = diagnostics.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError(f"router diagnostics[{index}].enabled must be boolean")
        enabled_values.append(enabled)
    if any(enabled_values) and not all(enabled_values):
        raise ValueError("router diagnostics cannot mix dense and sparse observations")
    if not any(enabled_values):
        return {
            "available": True,
            "enabled": False,
            "reason": "dense_model",
            "cases": len(records),
        }

    expected_num_experts: int | None = None
    expected_top_k: int | None = None
    aggregate_counts: list[int] | None = None
    category_counts: dict[str, list[int]] = {}
    total_tokens = 0
    total_assignments = 0
    total_invocations = 0
    aux_losses: list[float] = []
    expected_total_parameters: int | None = None
    expected_active_parameters: int | None = None

    for index, (category_value, diagnostics) in enumerate(records):
        category = _nonempty_text(category_value, label=f"router records[{index}].category")
        num_experts = _positive_int(
            diagnostics.get("num_experts"),
            label=f"router diagnostics[{index}].num_experts",
        )
        top_k = _positive_int(
            diagnostics.get("top_k"),
            label=f"router diagnostics[{index}].top_k",
        )
        if num_experts < 2:
            raise ValueError("enabled sparse routing requires at least two experts")
        if top_k > num_experts:
            raise ValueError("router top_k cannot exceed num_experts")
        if expected_num_experts is None:
            expected_num_experts = num_experts
            expected_top_k = top_k
            aggregate_counts = [0] * num_experts
        elif num_experts != expected_num_experts or top_k != expected_top_k:
            raise ValueError("router shape changed across evaluation cases")

        raw_counts = diagnostics.get("expert_counts")
        if not isinstance(raw_counts, Sequence) or isinstance(
            raw_counts, (str, bytes, bytearray)
        ):
            raise TypeError(f"router diagnostics[{index}].expert_counts must be a sequence")
        if len(raw_counts) != num_experts:
            raise ValueError(f"router diagnostics[{index}] expert_counts length mismatch")
        counts = [
            _positive_int(
                count,
                label=f"router diagnostics[{index}].expert_counts[{expert}]",
                allow_zero=True,
            )
            for expert, count in enumerate(raw_counts)
        ]
        tokens = _positive_int(
            diagnostics.get("tokens"),
            label=f"router diagnostics[{index}].tokens",
        )
        assignments = _positive_int(
            diagnostics.get("assignments"),
            label=f"router diagnostics[{index}].assignments",
        )
        invocations = _positive_int(
            diagnostics.get("invocations"),
            label=f"router diagnostics[{index}].invocations",
        )
        if assignments != sum(counts):
            raise ValueError(f"router diagnostics[{index}] assignments != sum(expert_counts)")
        if assignments != tokens * top_k:
            raise ValueError(f"router diagnostics[{index}] assignments != tokens * top_k")
        total_parameters_value = diagnostics.get("total_parameters")
        active_parameters_value = diagnostics.get("active_parameters")
        if (total_parameters_value is None) != (active_parameters_value is None):
            raise ValueError(
                f"router diagnostics[{index}] must report total_parameters and "
                "active_parameters together"
            )
        if total_parameters_value is not None:
            total_parameters = _positive_int(
                total_parameters_value,
                label=f"router diagnostics[{index}].total_parameters",
            )
            active_parameters = _positive_int(
                active_parameters_value,
                label=f"router diagnostics[{index}].active_parameters",
            )
            if active_parameters > total_parameters:
                raise ValueError("router active_parameters cannot exceed total_parameters")
            if expected_total_parameters is None:
                expected_total_parameters = total_parameters
                expected_active_parameters = active_parameters
            elif (
                total_parameters != expected_total_parameters
                or active_parameters != expected_active_parameters
            ):
                raise ValueError("router parameter counts changed across evaluation cases")
        aux_loss_value = diagnostics.get("load_balance_loss")
        if isinstance(aux_loss_value, bool) or not isinstance(aux_loss_value, Real):
            raise TypeError(f"router diagnostics[{index}].load_balance_loss must be real")
        aux_loss = float(aux_loss_value)
        if not math.isfinite(aux_loss):
            raise ValueError(f"router diagnostics[{index}].load_balance_loss must be finite")

        if aggregate_counts is None:
            raise RuntimeError("router aggregation was not initialized")
        for expert, count in enumerate(counts):
            aggregate_counts[expert] += count
        bucket = category_counts.setdefault(category, [0] * num_experts)
        for expert, count in enumerate(counts):
            bucket[expert] += count
        total_tokens += tokens
        total_assignments += assignments
        total_invocations += invocations
        aux_losses.append(aux_loss)

    if expected_num_experts is None or expected_top_k is None or aggregate_counts is None:
        raise RuntimeError("enabled router aggregation produced no shape")
    load = _router_load(aggregate_counts)
    raw_entropy, normalized_entropy, effective_experts = _entropy(load)
    per_category = {}
    category_loads = {}
    for category, counts in sorted(category_counts.items()):
        category_load = _router_load(counts)
        category_loads[category] = category_load
        category_entropy, category_normalized, category_effective = _entropy(category_load)
        per_category[category] = {
            "assignments": sum(counts),
            "expert_counts": counts,
            "expert_load": category_load,
            "active_experts": sum(count > 0 for count in counts),
            "utilization": sum(count > 0 for count in counts) / expected_num_experts,
            "entropy": category_entropy,
            "normalized_entropy": category_normalized,
            "effective_experts": category_effective,
        }

    divergences = [
        _jensen_shannon(category_loads[left], category_loads[right])
        for left_index, left in enumerate(sorted(category_loads))
        for right in sorted(category_loads)[left_index + 1 :]
    ]
    return {
        "available": True,
        "enabled": True,
        "cases": len(records),
        "num_experts": expected_num_experts,
        "top_k": expected_top_k,
        "invocations": total_invocations,
        "tokens": total_tokens,
        "assignments": total_assignments,
        "expert_counts": aggregate_counts,
        "expert_load": load,
        "active_experts": sum(count > 0 for count in aggregate_counts),
        "utilization": sum(count > 0 for count in aggregate_counts) / expected_num_experts,
        "entropy": raw_entropy,
        "normalized_entropy": normalized_entropy,
        "effective_experts": effective_experts,
        "load_balance_loss_mean": sum(aux_losses) / len(aux_losses),
        "total_parameters": expected_total_parameters,
        "active_parameters": expected_active_parameters,
        "active_parameter_fraction": (
            expected_active_parameters / expected_total_parameters
            if expected_total_parameters is not None and expected_active_parameters is not None
            else None
        ),
        "category_distribution_divergence": {
            "pairs": len(divergences),
            "mean_normalized_jensen_shannon": (
                sum(divergences) / len(divergences) if divergences else 0.0
            ),
            "max_normalized_jensen_shannon": max(divergences, default=0.0),
        },
        "by_category": per_category,
    }


def collect_router_diagnostics(
    model: Any,
    conversations: Sequence[Conversation],
    tokenizer: Any,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run deterministic prompt-prefill forwards and summarize exposed router telemetry."""

    diagnostics_fn = getattr(model, "routing_diagnostics", None)
    if not callable(diagnostics_fn):
        return {
            "available": False,
            "enabled": False,
            "reason": "model_does_not_expose_routing_diagnostics",
            "cases": 0,
        }

    import torch

    examples: list[tuple[str, str]] = []
    for conversation in conversations:
        category = _nonempty_text(
            conversation.meta.get("category", "unlabeled"),
            label="conversation category",
        )
        examples.extend(
            (category, example.prompt) for example in assistant_training_examples(conversation)
        )
    if not examples:
        raise ValueError("router evaluation requires at least one assistant decision")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    records: list[tuple[str, Mapping[str, Any]]] = []
    try:
        with torch.no_grad():
            for index, (category, prompt) in enumerate(examples):
                token_ids = tokenizer.encode(prompt)
                if not isinstance(token_ids, Sequence) or isinstance(
                    token_ids, (str, bytes, bytearray)
                ):
                    raise TypeError("tokenizer.encode must return a sequence of token IDs")
                if not token_ids:
                    raise ValueError(f"router prompt[{index}] tokenized to an empty sequence")
                if any(isinstance(token, bool) or not isinstance(token, Integral) for token in token_ids):
                    raise TypeError("tokenizer.encode must return integer token IDs")
                max_seq_len = getattr(getattr(model, "cfg", None), "max_seq_len", None)
                if isinstance(max_seq_len, Integral) and len(token_ids) > int(max_seq_len):
                    raise ValueError(
                        f"router prompt[{index}] has {len(token_ids)} tokens, "
                        f"exceeding model max_seq_len={int(max_seq_len)}"
                    )
                indices = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
                model(indices)
                diagnostics = diagnostics_fn()
                if not isinstance(diagnostics, Mapping):
                    raise TypeError("model.routing_diagnostics() must return a mapping")
                records.append((category, diagnostics))
    finally:
        model.train(was_training)
    return summarize_router_diagnostics(records)


def _gate(name: str, observed: Any, required: Any, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
    }


def _metric_accuracy(metric: Mapping[str, Any], *, label: str) -> float | None:
    value = metric.get("accuracy")
    if value is None:
        return None
    return _finite_rate(value, label=label)


def _public_behavior_breakdown(
    conversations: Sequence[Conversation],
    predictions: Sequence[str | AssistantPrediction],
) -> dict[str, dict[str, int | float | None]]:
    """Rebucket the core scorer's exact action decision by public behavior label."""

    expected_cases = [
        (conversation.meta["behavior"], conversation.messages[example.message_index])
        for conversation in conversations
        for example in assistant_training_examples(conversation)
    ]
    if len(predictions) != len(expected_cases):
        raise RuntimeError("recorded prediction count differs from public assistant decisions")
    counts: dict[str, list[int]] = {
        behavior: [0, 0] for behavior in sorted(PUBLIC_BEHAVIORS)
    }
    for (behavior, message), prediction_value in zip(
        expected_cases, predictions, strict=True
    ):
        prediction = (
            AssistantPrediction(text=prediction_value, finish_reason="caller_complete")
            if isinstance(prediction_value, str)
            else prediction_value
        )
        if not isinstance(prediction, AssistantPrediction):
            raise TypeError("predictor returned an unsupported public real-use output")
        parsed = parse_tool_output(prediction.text)
        complete_format_valid = prediction.complete and parsed.format_valid
        strict_calls = list(parsed.calls) if complete_format_valid else []
        if message.tool_calls:
            correct = complete_format_valid and match_calls(
                strict_calls, list(message.tool_calls)
            )
        else:
            correct = (
                complete_format_valid
                and not strict_calls
                and not parsed.tool_syntax_present
            )
        counts[behavior][0] += int(correct)
        counts[behavior][1] += 1
    return {
        behavior: {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
        }
        for behavior, (correct, total) in sorted(counts.items())
    }


def score_public_real_use(
    conversations: Sequence[Conversation],
    predictor: Predictor,
    requirements: RealUseRequirements,
    *,
    router_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a held-out public suite and evaluate explicit, deterministic promotion gates."""

    coverage = audit_public_real_use_cases(conversations)
    recorded_predictions: list[str | AssistantPrediction] = []

    def recording_predictor(prompt: str, tools: Sequence[Any]) -> str | AssistantPrediction:
        prediction = predictor(prompt, tools)
        recorded_predictions.append(prediction)
        return prediction

    score = score_conversations(conversations, recording_predictor)
    by_behavior = _public_behavior_breakdown(conversations, recorded_predictions)
    metrics = score["metrics"]
    tool_multi = metrics["teacher_forced_tool_multi_turn"]
    category_accuracies = {
        category: _metric_accuracy(
            value["action_exact"],
            label=f"by_category.{category}.action_exact.accuracy",
        )
        for category, value in score["by_category"].items()
    }
    observed_capabilities = set(coverage["capabilities"])
    missing_capabilities = sorted(
        set(requirements.required_capabilities) - observed_capabilities
    )

    gates = [
        _gate(
            "minimum_conversations",
            coverage["conversations"],
            requirements.min_conversations,
            coverage["conversations"] >= requirements.min_conversations,
        ),
        _gate(
            "minimum_public_datasets",
            len(coverage["datasets"]),
            requirements.min_datasets,
            len(coverage["datasets"]) >= requirements.min_datasets,
        ),
        _gate(
            "minimum_categories",
            len(coverage["categories"]),
            requirements.min_categories,
            len(coverage["categories"]) >= requirements.min_categories,
        ),
        _gate(
            "minimum_action_conversations",
            coverage["action_conversations"],
            requirements.min_action_conversations,
            coverage["action_conversations"] >= requirements.min_action_conversations,
        ),
        _gate(
            "minimum_abstention_conversations",
            coverage["abstention_conversations"],
            requirements.min_abstention_conversations,
            coverage["abstention_conversations"] >= requirements.min_abstention_conversations,
        ),
        _gate(
            "minimum_irrelevance_conversations",
            coverage["irrelevance_conversations"],
            requirements.min_irrelevance_conversations,
            coverage["irrelevance_conversations"] >= requirements.min_irrelevance_conversations,
        ),
        _gate(
            "minimum_multi_turn_action_conversations",
            coverage["multi_turn_action_conversations"],
            requirements.min_multi_turn_action_conversations,
            coverage["multi_turn_action_conversations"]
            >= requirements.min_multi_turn_action_conversations,
        ),
        _gate(
            "required_capabilities",
            sorted(observed_capabilities),
            list(requirements.required_capabilities),
            not missing_capabilities,
        ),
    ]

    quality_requirements = (
        ("action_exact", metrics["action_exact"], requirements.min_action_exact),
        ("tool_call_exact", metrics["whole_call_exact"], requirements.min_tool_call_exact),
        (
            "abstention_accuracy",
            by_behavior["abstention"],
            requirements.min_abstention_accuracy,
        ),
        (
            "irrelevance_accuracy",
            by_behavior["irrelevance"],
            requirements.min_irrelevance_accuracy,
        ),
        (
            "teacher_forced_multi_turn_tool_step_exact",
            tool_multi["tool_step_exact"],
            requirements.min_multi_turn_step_exact,
        ),
        (
            "teacher_forced_multi_turn_tool_episode_exact",
            tool_multi["tool_episode_exact"],
            requirements.min_multi_turn_episode_exact,
        ),
    )
    for name, metric, threshold in quality_requirements:
        accuracy = _metric_accuracy(metric, label=f"{name}.accuracy")
        gates.append(
            _gate(
                name,
                accuracy,
                threshold,
                accuracy is not None and accuracy >= threshold,
            )
        )

    failing_categories = sorted(
        category
        for category, accuracy in category_accuracies.items()
        if accuracy is None or accuracy < requirements.min_category_action_exact
    )
    gates.append(
        _gate(
            "minimum_per_category_action_exact",
            category_accuracies,
            requirements.min_category_action_exact,
            not failing_categories,
        )
    )

    router = dict(router_report) if router_report is not None else {
        "available": False,
        "enabled": False,
        "reason": "router_report_not_supplied",
        "cases": 0,
    }
    if requirements.require_sparse_router:
        enabled = router.get("available") is True and router.get("enabled") is True
        gates.append(_gate("sparse_router_telemetry", enabled, True, enabled))
        utilization = router.get("utilization") if enabled else None
        entropy = router.get("normalized_entropy") if enabled else None
        divergence_value = router.get("category_distribution_divergence") if enabled else None
        divergence = (
            divergence_value.get("mean_normalized_jensen_shannon")
            if isinstance(divergence_value, Mapping)
            else None
        )
        gates.extend(
            [
                _gate(
                    "router_utilization",
                    utilization,
                    requirements.min_router_utilization,
                    isinstance(utilization, Real)
                    and not isinstance(utilization, bool)
                    and math.isfinite(float(utilization))
                    and float(utilization) >= requirements.min_router_utilization,
                ),
                _gate(
                    "router_normalized_entropy",
                    entropy,
                    requirements.min_router_normalized_entropy,
                    isinstance(entropy, Real)
                    and not isinstance(entropy, bool)
                    and math.isfinite(float(entropy))
                    and float(entropy) >= requirements.min_router_normalized_entropy,
                ),
                _gate(
                    "router_category_divergence",
                    divergence,
                    requirements.min_router_category_divergence,
                    isinstance(divergence, Real)
                    and not isinstance(divergence, bool)
                    and math.isfinite(float(divergence))
                    and float(divergence) >= requirements.min_router_category_divergence,
                ),
            ]
        )

    return {
        "contract": {
            "name": "LocalAgent public real-use promotion gate",
            "schema_version": 1,
            "scoring": "strict LocalAgent AST score_conversations",
            "multi_turn": (
                "teacher-forced gold prior history; exact tool-step and whole-tool-episode "
                "metrics; not a free-running environment success rate"
            ),
            "router": (
                "prompt-prefill aggregate telemetry exposed by the evaluated model; dense or "
                "unobserved routers have no utilization score"
            ),
        },
        "coverage": coverage,
        "score": score,
        "by_behavior": by_behavior,
        "router": router,
        "gates": {
            "all_passed": all(gate["passed"] for gate in gates),
            "passed": sum(gate["passed"] for gate in gates),
            "total": len(gates),
            "records": gates,
        },
    }


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _read_frozen_dataset(path: Path, *, expected_sha256: str) -> bytes:
    expected = _sha256_text(expected_sha256, label="expected_sha256")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"real-use dataset is missing or not a regular non-symlink file: {path}"
        ) from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"real-use dataset is not a regular file: {path}")
        if not 0 < initial.st_size <= _MAX_FROZEN_DATASET_BYTES:
            raise ValueError(
                "real-use dataset size must be in "
                f"[1, {_MAX_FROZEN_DATASET_BYTES}] bytes"
            )
        try:
            initial_path = path.lstat()
        except OSError as error:
            raise RuntimeError("real-use dataset pathname changed while opening") from error
        if not _same_file_state(initial, initial_path):
            raise RuntimeError("real-use dataset changed while binding its descriptor")
        chunks = []
        observed = 0
        digest = hashlib.sha256()
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_FROZEN_DATASET_BYTES + 1 - observed),
            )
            if not chunk:
                break
            observed += len(chunk)
            if observed > _MAX_FROZEN_DATASET_BYTES:
                raise ValueError(
                    f"real-use dataset exceeds {_MAX_FROZEN_DATASET_BYTES} bytes"
                )
            chunks.append(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        try:
            final_path = path.lstat()
        except OSError as error:
            raise RuntimeError("real-use dataset pathname changed while reading") from error
        if not _same_file_state(initial, final) or not _same_file_state(initial, final_path):
            raise RuntimeError("real-use dataset changed while being read")
        if digest.hexdigest() != expected:
            raise ValueError("real-use dataset SHA-256 does not match the frozen identity")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def score_public_real_use_dataset(
    jsonl_path: str | Path,
    expected_sha256: str,
    predictor: Predictor,
    requirements: RealUseRequirements,
    *,
    router_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify and score one immutable public real-use Conversation JSONL snapshot."""

    path = Path(jsonl_path)
    payload = _read_frozen_dataset(path, expected_sha256=expected_sha256)
    conversations = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValueError(
                f"real-use dataset line {line_number} must end in exactly one LF"
            )
        try:
            conversations.append(Conversation.from_json(line[:-1].decode("utf-8")))
        except (KeyError, TypeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"real-use dataset line {line_number} is not a Conversation"
            ) from error

    result = score_public_real_use(
        conversations,
        predictor,
        requirements,
        router_report=router_report,
    )
    result["dataset_artifact"] = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": expected_sha256,
        "frozen_identity_verified": True,
    }
    return result
