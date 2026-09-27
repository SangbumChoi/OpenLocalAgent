"""Deterministic, fail-closed RL-readiness summary from existing evaluation evidence.

This module does not generate samples or run training. It combines a sealed internal agent
scorecard with a sealed isolated RL preflight and promotes production RL only when:

* the evidence refers to the same SFT checkpoint and held-out eval corpus,
* expected-tool outputs demonstrate strict format, schema, name, and whole-call exactness,
* expected no-tool outputs demonstrate structural abstention,
* pinned teacher-forced SFT metrics remain non-inferior on the same held-out decisions,
* sampled rewards vary within at least one rollout group, and
* the isolated optimizer actually realizes the configured policy epochs.

Schema v2 uses tool-conditioned metrics and never treats aggregate ``action_exact`` as promotion
evidence. Schema-v1 configs and summaries remain supported for sealed-artifact verification. The
scorecard is explicitly an internal BFCL-style contract, not an official BFCL result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CONFIG_KIND = "localagent_rl_readiness_config"
SUMMARY_KIND = "localagent_rl_readiness_summary"
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, SCHEMA_VERSION})

_SCORECARD_KIND = "localagent_internal_agent_scorecard_result"
_PREFLIGHT_KIND = "localagent_one_update_training_preflight"
_PROMPT_CONTRACT = "openai_full_catalog_v1"
_INTERNAL_BENCHMARK_NAME = "LocalAgent BFCL-style internal agent scorecard"
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_EVIDENCE_BYTES = 512 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024 * 1024
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

_V1_CONFIG_KEYS = frozenset({"kind", "schema_version", "evidence", "thresholds"})
_V2_CONFIG_KEYS = frozenset(
    {"kind", "schema_version", "production", "evidence", "thresholds"}
)
_PRODUCTION_KEYS = frozenset({"rl_config", "parent_checkpoint", "execution"})
_PRODUCTION_RL_CONFIG_KEYS = frozenset({"path", "expected_canonical_sha256"})
_PRODUCTION_PARENT_KEYS = frozenset({"path", "expected_sha256"})
_PRODUCTION_EXECUTION_KEYS = frozenset(
    {
        "production_requested_device",
        "production_requested_dtype",
        "preflight_requested_device",
        "preflight_requested_dtype",
        "resolved_device",
        "resolved_dtype",
    }
)
_V1_EVIDENCE_KEYS = frozenset({"scorecard", "rl_preflight"})
_V2_SFT_METRICS_EVIDENCE_KEYS = frozenset({"scorecard", "rl_preflight", "sft_metrics"})
_V2_SWEEP_EVIDENCE_KEYS = frozenset({"scorecard", "rl_preflight", "sft_checkpoint_sweep"})
_EVIDENCE_SPEC_KEYS = frozenset({"path", "expected_self_sha256"})
_SFT_METRICS_EVIDENCE_SPEC_KEYS = frozenset({"path", "expected_sha256"})
_SFT_SWEEP_EVIDENCE_SPEC_KEYS = frozenset(
    {"path", "expected_self_sha256", "selected_checkpoint_sha256"}
)
_V1_THRESHOLD_KEYS = frozenset({"scorecard", "rl_preflight"})
_V2_THRESHOLD_KEYS = frozenset({"scorecard", "rl_preflight", "sft_metrics"})
_V1_SCORECARD_THRESHOLD_KEYS = frozenset(
    {
        "min_assistant_decisions",
        "min_generation_completion_rate",
        "max_generation_truncation_rate",
        "min_complete_format_rate",
        "min_schema_valid_attempt_rate",
        "min_action_exact_successes",
    }
)
_V2_SCORECARD_THRESHOLD_KEYS = frozenset(
    {
        "min_assistant_decisions",
        "min_generation_completion_rate",
        "max_generation_truncation_rate",
        "min_complete_format_successes",
        "min_complete_format_rate",
        "min_tool_format_successes",
        "min_tool_format_rate",
        "min_schema_valid_tool_successes",
        "min_schema_valid_tool_rate",
        "min_tool_name_case_exact_successes",
        "min_tool_name_case_exact_rate",
        "min_whole_call_exact_successes",
        "min_whole_call_exact_rate",
        "min_abstention_successes",
        "min_abstention_rate",
    }
)
_SFT_METRICS_THRESHOLD_KEYS = frozenset(
    {
        "max_mean_loss_increase",
        "max_assistant_token_accuracy_drop",
        "max_assistant_sequence_accuracy_drop",
    }
)
_SFT_SWEEP_KIND = "localagent_sft_checkpoint_sweep_result"
_SFT_SWEEP_SCHEMA_VERSION = 2
_SFT_SWEEP_TOP_LEVEL_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "inputs",
        "identity",
        "heldout",
        "thresholds",
        "execution",
        "selection_contract",
        "checkpoints",
        "summary",
        "result_sha256",
    }
)
_SFT_SWEEP_ARTIFACT_KEYS = frozenset({"path", "bytes", "sha256"})
_CONVERSATION_OVERLAP_KEYS = frozenset(
    {
        "fingerprint_contract",
        "left_rows",
        "right_rows",
        "left_rendered_prompts",
        "right_rendered_prompts",
        "left_semantic_set_sha256",
        "right_semantic_set_sha256",
        "left_rendered_prompt_set_sha256",
        "right_rendered_prompt_set_sha256",
        "semantic_overlap",
        "rendered_prompt_overlap",
        "semantic_overlap_sha256",
        "rendered_prompt_overlap_sha256",
    }
)
_PREFLIGHT_THRESHOLD_KEYS = frozenset(
    {
        "min_attempted_groups",
        "min_attempted_rollouts",
        "min_tool_syntax_rate",
        "min_complete_parser_valid_rate",
        "min_schema_valid_tool_rate",
        "min_exact_successes",
        "min_reward_unique_values",
        "min_informative_groups",
        "min_informative_group_rate",
        "min_realized_optimizer_updates",
        "max_truncation_rate",
    }
)
_POLICY_TRANSITION_KEYS = frozenset(
    {
        "contract",
        "model_named_parameter_names",
        "model_parameter_count",
        "compared_model_parameter_names",
        "compared_model_parameter_count",
        "changed_model_parameter_names",
        "changed_model_parameter_count",
        "first_changed_model_parameter",
        "initial_model_state_sha256",
        "final_model_state_sha256",
        "at_least_one_policy_tensor_changed",
        "production_schedule_total_steps",
        "execution_rollout_step_limit",
        "first_nonzero_learning_rate_step",
        "expected_learning_rates",
        "actual_learning_rates",
        "actual_learning_rates_match_expected",
        "nonzero_learning_rate_executed",
        "final_optimizer_learning_rates",
        "final_optimizer_learning_rate_matches_expected",
    }
)
_SCORECARD_TOP_LEVEL_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "benchmark",
        "provenance",
        "scorecard",
        "limitations",
        "result_self_sha256",
    }
)
_RATE_KEYS = frozenset({"correct", "total", "accuracy"})


def _is_schema_version(value: Any, expected: int) -> bool:
    """Return whether *value* is an exact non-boolean integer schema version."""

    return isinstance(value, int) and not isinstance(value, bool) and value == expected


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json_bytes(value: Any, *, trailing_lf: bool) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if trailing_lf:
        encoded += "\n"
    return encoded.encode("utf-8")


def _canonical_sha256(value: Any, *, trailing_lf: bool = False) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, trailing_lf=trailing_lf)).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


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


def _read_regular(path: Path, *, label: str, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if initial.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        try:
            bound_path = path.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} pathname changed while being read: {path}") from error
        if not _same_file_state(initial, bound_path):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")

        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        payload = b"".join(chunks)
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        if not _same_file_state(initial, final_descriptor) or not _same_file_state(
            initial,
            final_path,
        ):
            raise RuntimeError(f"{label} changed while it was being read: {path}")
        return payload, {
            "path": str(path),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if initial.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        try:
            bound_path = path.lstat()
        except OSError as error:
            raise RuntimeError(f"{label} pathname changed while being hashed: {path}") from error
        if not _same_file_state(initial, bound_path):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")

        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed))
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        if not _same_file_state(initial, final_descriptor) or not _same_file_state(
            initial,
            final_path,
        ):
            raise RuntimeError(f"{label} changed while it was being hashed: {path}")
        return {
            "path": str(path),
            "bytes": observed,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, *, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value!r}")

    try:
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _exact_mapping(
    value: Any,
    keys: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    mapping = _mapping(value, label=label)
    missing = sorted(keys - set(mapping))
    extra = sorted(set(mapping) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return mapping


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    result = _nonnegative_int(value, label=label)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _rate_threshold(value: Any, *, label: str, positive: bool) -> float:
    result = _finite_number(value, label=label)
    lower_bound = 0.0 < result if positive else 0.0 <= result
    if not lower_bound or result > 1.0:
        qualifier = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{label} must be in {qualifier}")
    return result


def _max_truncation_threshold(value: Any, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{label} must be in [0, 1)")
    return result


def _rate(value: Any, *, label: str, expected_total: int | None = None) -> dict[str, Any]:
    record = _exact_mapping(value, _RATE_KEYS, label=label)
    correct = _nonnegative_int(record["correct"], label=f"{label}.correct")
    total = _nonnegative_int(record["total"], label=f"{label}.total")
    if correct > total:
        raise ValueError(f"{label}.correct exceeds total")
    if expected_total is not None and total != expected_total:
        raise ValueError(f"{label}.total does not match the expected denominator")
    accuracy = record["accuracy"]
    if total == 0:
        if accuracy is not None:
            raise ValueError(f"{label}.accuracy must be null when total is zero")
        normalized_accuracy = None
    else:
        normalized_accuracy = _finite_number(accuracy, label=f"{label}.accuracy")
        expected_accuracy = correct / total
        if not math.isclose(
            normalized_accuracy,
            expected_accuracy,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"{label}.accuracy does not match correct / total")
    return {
        "count": correct,
        "total": total,
        "rate": normalized_accuracy,
    }


def _observed_rate(count: int, total: int) -> dict[str, int | float]:
    if total < 1 or not 0 <= count <= total:
        raise ValueError("cannot construct a rate from invalid counts")
    return {"count": count, "total": total, "rate": count / total}


def _load_config(
    path: str | Path,
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = Path(path)
    raw, artifact = _read_regular(source, label="RL-readiness config", max_bytes=_MAX_CONFIG_BYTES)
    try:
        config = yaml.load(raw.decode("utf-8", errors="strict"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("RL-readiness config is not valid UTF-8 YAML") from error
    config = _mapping(config, label="RL-readiness config")
    schema_version = config.get("schema_version")
    if (
        config.get("kind") != CONFIG_KIND
        or not any(
            _is_schema_version(schema_version, supported)
            for supported in _SUPPORTED_SCHEMA_VERSIONS
        )
    ):
        raise ValueError("unsupported RL-readiness config kind/schema_version")
    config = _exact_mapping(
        config,
        _V1_CONFIG_KEYS
        if schema_version == LEGACY_SCHEMA_VERSION
        else _V2_CONFIG_KEYS,
        label="RL-readiness config",
    )
    artifact["canonical_sha256"] = _canonical_sha256(config)
    if schema_version == LEGACY_SCHEMA_VERSION:
        evidence = _exact_mapping(
            config["evidence"],
            _V1_EVIDENCE_KEYS,
            label="config.evidence",
        )
    else:
        production = _exact_mapping(
            config["production"],
            _PRODUCTION_KEYS,
            label="config.production",
        )
        rl_config_specification = _exact_mapping(
            production["rl_config"],
            _PRODUCTION_RL_CONFIG_KEYS,
            label="config.production.rl_config",
        )
        parent_specification = _exact_mapping(
            production["parent_checkpoint"],
            _PRODUCTION_PARENT_KEYS,
            label="config.production.parent_checkpoint",
        )
        execution_specification = _exact_mapping(
            production["execution"],
            _PRODUCTION_EXECUTION_KEYS,
            label="config.production.execution",
        )
        for label, specification in (
            ("config.production.rl_config", rl_config_specification),
            ("config.production.parent_checkpoint", parent_specification),
        ):
            if not isinstance(specification["path"], str) or not specification["path"]:
                raise ValueError(f"{label}.path must be non-empty text")
        _valid_sha256(
            rl_config_specification["expected_canonical_sha256"],
            label="config.production.rl_config.expected_canonical_sha256",
        )
        _valid_sha256(
            parent_specification["expected_sha256"],
            label="config.production.parent_checkpoint.expected_sha256",
        )
        for key in sorted(_PRODUCTION_EXECUTION_KEYS):
            if (
                not isinstance(execution_specification[key], str)
                or not execution_specification[key]
            ):
                raise ValueError(
                    f"config.production.execution.{key} must be non-empty text"
                )
        evidence_mapping = _mapping(config["evidence"], label="config.evidence")
        evidence_key_set = frozenset(evidence_mapping)
        if evidence_key_set not in {
            _V2_SFT_METRICS_EVIDENCE_KEYS,
            _V2_SWEEP_EVIDENCE_KEYS,
        }:
            raise ValueError(
                "schema-v2 config.evidence must contain scorecard, rl_preflight, and "
                "exactly one of sft_metrics or sft_checkpoint_sweep"
            )
        evidence = evidence_mapping
    for key in ("scorecard", "rl_preflight"):
        value = evidence[key]
        specification = _exact_mapping(
            value,
            _EVIDENCE_SPEC_KEYS,
            label=f"config.evidence.{key}",
        )
        if not isinstance(specification["path"], str) or not specification["path"]:
            raise ValueError(f"config.evidence.{key}.path must be non-empty text")
        _valid_sha256(
            specification["expected_self_sha256"],
            label=f"config.evidence.{key}.expected_self_sha256",
        )
    if schema_version == SCHEMA_VERSION and "sft_metrics" in evidence:
        metrics_specification = _exact_mapping(
            evidence["sft_metrics"],
            _SFT_METRICS_EVIDENCE_SPEC_KEYS,
            label="config.evidence.sft_metrics",
        )
        if not isinstance(metrics_specification["path"], str) or not metrics_specification["path"]:
            raise ValueError("config.evidence.sft_metrics.path must be non-empty text")
        _valid_sha256(
            metrics_specification["expected_sha256"],
            label="config.evidence.sft_metrics.expected_sha256",
        )
    if schema_version == SCHEMA_VERSION and "sft_checkpoint_sweep" in evidence:
        sweep_specification = _exact_mapping(
            evidence["sft_checkpoint_sweep"],
            _SFT_SWEEP_EVIDENCE_SPEC_KEYS,
            label="config.evidence.sft_checkpoint_sweep",
        )
        if not isinstance(sweep_specification["path"], str) or not sweep_specification["path"]:
            raise ValueError("config.evidence.sft_checkpoint_sweep.path must be non-empty text")
        for key in ("expected_self_sha256", "selected_checkpoint_sha256"):
            _valid_sha256(
                sweep_specification[key],
                label=f"config.evidence.sft_checkpoint_sweep.{key}",
            )

    threshold_keys = (
        _V1_THRESHOLD_KEYS if schema_version == LEGACY_SCHEMA_VERSION else _V2_THRESHOLD_KEYS
    )
    thresholds = _exact_mapping(
        config["thresholds"],
        threshold_keys,
        label="config.thresholds",
    )
    scorecard_threshold_keys = (
        _V1_SCORECARD_THRESHOLD_KEYS
        if schema_version == LEGACY_SCHEMA_VERSION
        else _V2_SCORECARD_THRESHOLD_KEYS
    )
    scorecard_thresholds = _exact_mapping(
        thresholds["scorecard"],
        scorecard_threshold_keys,
        label="config.thresholds.scorecard",
    )
    preflight_thresholds = _exact_mapping(
        thresholds["rl_preflight"],
        _PREFLIGHT_THRESHOLD_KEYS,
        label="config.thresholds.rl_preflight",
    )
    _positive_int(
        scorecard_thresholds["min_assistant_decisions"],
        label="config.thresholds.scorecard.min_assistant_decisions",
    )
    scorecard_rate_keys = ["min_generation_completion_rate", "min_complete_format_rate"]
    if schema_version == LEGACY_SCHEMA_VERSION:
        scorecard_rate_keys.append("min_schema_valid_attempt_rate")
    else:
        scorecard_rate_keys.extend(
            [
                "min_tool_format_rate",
                "min_schema_valid_tool_rate",
                "min_tool_name_case_exact_rate",
                "min_whole_call_exact_rate",
                "min_abstention_rate",
            ]
        )
    for key in scorecard_rate_keys:
        _rate_threshold(
            scorecard_thresholds[key],
            label=f"config.thresholds.scorecard.{key}",
            positive=True,
        )
    _max_truncation_threshold(
        scorecard_thresholds["max_generation_truncation_rate"],
        label="config.thresholds.scorecard.max_generation_truncation_rate",
    )
    scorecard_count_keys = (
        ["min_action_exact_successes"]
        if schema_version == LEGACY_SCHEMA_VERSION
        else [
            "min_complete_format_successes",
            "min_tool_format_successes",
            "min_schema_valid_tool_successes",
            "min_tool_name_case_exact_successes",
            "min_whole_call_exact_successes",
            "min_abstention_successes",
        ]
    )
    for key in scorecard_count_keys:
        _positive_int(
            scorecard_thresholds[key],
            label=f"config.thresholds.scorecard.{key}",
        )

    for key in (
        "min_attempted_groups",
        "min_attempted_rollouts",
        "min_exact_successes",
        "min_informative_groups",
        "min_realized_optimizer_updates",
    ):
        _positive_int(
            preflight_thresholds[key],
            label=f"config.thresholds.rl_preflight.{key}",
        )
    unique_values = _positive_int(
        preflight_thresholds["min_reward_unique_values"],
        label="config.thresholds.rl_preflight.min_reward_unique_values",
    )
    if unique_values < 2:
        raise ValueError(
            "config.thresholds.rl_preflight.min_reward_unique_values must be at least 2"
        )
    for key in (
        "min_tool_syntax_rate",
        "min_complete_parser_valid_rate",
        "min_schema_valid_tool_rate",
        "min_informative_group_rate",
    ):
        _rate_threshold(
            preflight_thresholds[key],
            label=f"config.thresholds.rl_preflight.{key}",
            positive=True,
        )
    _max_truncation_threshold(
        preflight_thresholds["max_truncation_rate"],
        label="config.thresholds.rl_preflight.max_truncation_rate",
    )
    normalized_thresholds = {
        "scorecard": copy.deepcopy(scorecard_thresholds),
        "rl_preflight": copy.deepcopy(preflight_thresholds),
    }
    if schema_version == SCHEMA_VERSION:
        sft_thresholds = _exact_mapping(
            thresholds["sft_metrics"],
            _SFT_METRICS_THRESHOLD_KEYS,
            label="config.thresholds.sft_metrics",
        )
        for key in sorted(_SFT_METRICS_THRESHOLD_KEYS):
            threshold = _finite_number(
                sft_thresholds[key],
                label=f"config.thresholds.sft_metrics.{key}",
            )
            if threshold < 0.0:
                raise ValueError(f"config.thresholds.sft_metrics.{key} must be non-negative")
        normalized_thresholds["sft_metrics"] = copy.deepcopy(sft_thresholds)
    return schema_version, config, artifact, normalized_thresholds


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def _production_rl_schedule_prefix(config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the first nonzero-LR prefix using the production RL schedule semantics."""

    from localagent.train.loop import cosine_lr

    schedule = _mapping(config.get("schedule", {}), label="production RL schedule")
    optim = _mapping(config.get("optim", {}), label="production RL optim")
    total_steps = _positive_int(
        schedule.get("total_steps"),
        label="production RL schedule.total_steps",
    )
    warmup_steps = _nonnegative_int(
        schedule.get("warmup_steps", 5),
        label="production RL schedule.warmup_steps",
    )
    peak_lr = _finite_number(
        optim.get("lr", 2e-4),
        label="production RL optim.lr",
    )
    if peak_lr <= 0.0:
        raise ValueError("production RL optim.lr must be positive")
    learning_rates: list[float] = []
    first_nonzero_step = None
    for step in range(total_steps):
        learning_rate = float(
            cosine_lr(
                step,
                total_steps,
                peak_lr,
                warmup_steps,
                0.1,
            )
        )
        learning_rates.append(learning_rate)
        if learning_rate > 0.0:
            first_nonzero_step = step
            break
    if first_nonzero_step is None:
        raise ValueError("production RL schedule contains no nonzero learning rate")
    return {
        "production_schedule_total_steps": total_steps,
        "execution_rollout_step_limit": first_nonzero_step + 1,
        "first_nonzero_learning_rate_step": first_nonzero_step,
        "expected_learning_rates": learning_rates,
    }


def _load_production_binding(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the exact production RL config and immutable parent binding."""

    production = _mapping(config.get("production"), label="config.production")
    rl_config_specification = _mapping(
        production.get("rl_config"),
        label="config.production.rl_config",
    )
    parent_specification = _mapping(
        production.get("parent_checkpoint"),
        label="config.production.parent_checkpoint",
    )
    expected_execution = _mapping(
        production.get("execution"),
        label="config.production.execution",
    )

    production_config_path = Path(str(rl_config_specification["path"]))
    raw_config, config_artifact = _read_regular(
        production_config_path,
        label="production RL config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    try:
        production_config = yaml.load(
            raw_config.decode("utf-8", errors="strict"),
            Loader=_UniqueKeyLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("production RL config is not valid UTF-8 YAML") from error
    production_config = _mapping(
        production_config,
        label="production RL config",
    )
    config_canonical_sha256 = _canonical_sha256(production_config)
    if (
        config_canonical_sha256
        != rl_config_specification["expected_canonical_sha256"]
    ):
        raise ValueError(
            "production RL config canonical SHA-256 does not match the approved identity"
        )
    if production_config.get("stage", "rl") != "rl":
        raise ValueError("production RL config must declare stage='rl'")

    runtime = _mapping(
        production_config.get("runtime", {}),
        label="production RL runtime",
    )
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("production RL runtime.device must be non-empty text")
    if not isinstance(requested_dtype, str) or not requested_dtype:
        raise ValueError("production RL runtime.dtype must be non-empty text")
    if runtime.get("resume", False) is not False:
        raise ValueError("production RL readiness requires runtime.resume=false")
    if (
        expected_execution["production_requested_device"] != requested_device
        or expected_execution["production_requested_dtype"] != requested_dtype
    ):
        raise ValueError(
            "production RL requested device/dtype do not match the approved execution binding"
        )

    init_from = production_config.get("init_from")
    if not isinstance(init_from, str) or not init_from:
        raise ValueError("production RL init_from must be a non-empty path")
    parent_reference = str(parent_specification["path"])
    if not _same_resolved_path(init_from, parent_reference):
        raise ValueError(
            "production RL init_from does not match the approved parent checkpoint path"
        )
    parent_artifact = _hash_regular(
        Path(parent_reference),
        label="approved production RL parent checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if parent_artifact["sha256"] != parent_specification["expected_sha256"]:
        raise ValueError(
            "production RL parent checkpoint SHA-256 does not match the approved identity"
        )

    log = _mapping(production_config.get("log", {}), label="production RL log")
    out_dir = log.get("out_dir", "runs/rl")
    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("production RL log.out_dir must be a non-empty path")
    config_artifact.update(
        {
            "reference": str(production_config_path),
            "canonical_sha256": config_canonical_sha256,
        }
    )
    parent_artifact["reference"] = parent_reference
    binding = {
        "rl_config": config_artifact,
        "parent_checkpoint": parent_artifact,
        "execution": copy.deepcopy(expected_execution),
        "out_dir": out_dir,
    }
    return binding, production_config


def _load_json_artifact(
    specification: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = str(specification["path"])
    source = Path(reference)
    raw, artifact = _read_regular(source, label=label, max_bytes=_MAX_EVIDENCE_BYTES)
    payload = _strict_json(raw, label=label)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} root must be a JSON object")
    artifact["reference"] = reference
    return payload, artifact


def _assert_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    trailing_lf: bool,
    label: str,
) -> str:
    recorded = _valid_sha256(payload.get(field), label=f"{label}.{field}")
    if recorded != expected:
        raise ValueError(f"{label}.{field} does not match the configured evidence identity")
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop(field)
    observed = _canonical_sha256(unsigned, trailing_lf=trailing_lf)
    if recorded != observed:
        raise ValueError(f"{label} self-hash mismatch")
    return recorded


def _validate_scorecard(
    payload: Mapping[str, Any],
    *,
    readiness_schema_version: int,
) -> dict[str, Any]:
    result = _exact_mapping(payload, _SCORECARD_TOP_LEVEL_KEYS, label="scorecard result")
    if result["kind"] != _SCORECARD_KIND or not _is_schema_version(
        result["schema_version"],
        LEGACY_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported internal scorecard result")

    benchmark = _mapping(result["benchmark"], label="scorecard benchmark")
    if (
        benchmark.get("name") != _INTERNAL_BENCHMARK_NAME
        or benchmark.get("official_bfcl") is not False
        or benchmark.get("external_native_benchmark") is not False
    ):
        raise ValueError("scorecard must be the internal non-official BFCL-style contract")
    if benchmark.get("conversation_prompt_contract") != _PROMPT_CONTRACT:
        raise ValueError("scorecard prompt contract is unsupported")

    scorecard = _mapping(result["scorecard"], label="scorecard")
    contract = _mapping(scorecard.get("contract"), label="scorecard.contract")
    if (
        contract.get("official_bfcl") is not False
        or contract.get("external_native_benchmark") is not False
        or contract.get("conversation_prompt_contract") != _PROMPT_CONTRACT
    ):
        raise ValueError("scorecard scoring contract is not the expected internal contract")

    case_set = _mapping(scorecard.get("case_set"), label="scorecard.case_set")
    conversations = _positive_int(
        case_set.get("conversations"),
        label="scorecard.case_set.conversations",
    )
    assistant_decisions = _positive_int(
        case_set.get("assistant_decisions"),
        label="scorecard.case_set.assistant_decisions",
    )
    tool_decisions = _nonnegative_int(
        case_set.get("tool_decisions"),
        label="scorecard.case_set.tool_decisions",
    )
    no_tool_decisions = _nonnegative_int(
        case_set.get("no_tool_decisions"),
        label="scorecard.case_set.no_tool_decisions",
    )
    if tool_decisions + no_tool_decisions != assistant_decisions:
        raise ValueError("scorecard tool/no-tool counts do not sum to assistant decisions")
    case_set_sha256 = _valid_sha256(
        case_set.get("sha256"),
        label="scorecard.case_set.sha256",
    )

    metrics = _mapping(scorecard.get("metrics"), label="scorecard.metrics")
    completion = _rate(
        metrics.get("generation_completion"),
        label="scorecard.metrics.generation_completion",
        expected_total=assistant_decisions,
    )
    complete_format = _rate(
        metrics.get("format_validity"),
        label="scorecard.metrics.format_validity",
        expected_total=assistant_decisions,
    )
    schema_valid = _rate(
        metrics.get("schema_validity_on_tool_attempts"),
        label="scorecard.metrics.schema_validity_on_tool_attempts",
    )
    if schema_valid["total"] > assistant_decisions:
        raise ValueError("scorecard schema-attempt denominator exceeds assistant decisions")
    action_exact = _rate(
        metrics.get("action_exact"),
        label="scorecard.metrics.action_exact",
        expected_total=assistant_decisions,
    )
    if complete_format["count"] > completion["count"]:
        raise ValueError("scorecard complete format count exceeds completed generations")
    if schema_valid["count"] > complete_format["count"]:
        raise ValueError("scorecard schema-valid count exceeds complete format-valid count")
    if action_exact["count"] > complete_format["count"]:
        raise ValueError("scorecard exact actions exceed complete format-valid outputs")
    v2_metrics: dict[str, Any] = {}
    if readiness_schema_version == SCHEMA_VERSION:
        tool_format = _rate(
            metrics.get("tool_format_validity_on_tool_decisions"),
            label="scorecard.metrics.tool_format_validity_on_tool_decisions",
            expected_total=tool_decisions,
        )
        schema_valid_on_tool_decisions = _rate(
            metrics.get("schema_validity_on_tool_decisions"),
            label="scorecard.metrics.schema_validity_on_tool_decisions",
            expected_total=tool_decisions,
        )
        tool_name = _mapping(metrics.get("tool_name"), label="scorecard.metrics.tool_name")
        tool_name_case_exact = _rate(
            tool_name.get("case_exact"),
            label="scorecard.metrics.tool_name.case_exact",
            expected_total=tool_decisions,
        )
        whole_call_exact = _rate(
            metrics.get("whole_call_exact"),
            label="scorecard.metrics.whole_call_exact",
            expected_total=tool_decisions,
        )
        abstention = _rate(
            metrics.get("abstention"),
            label="scorecard.metrics.abstention",
            expected_total=no_tool_decisions,
        )
        if tool_format["count"] > complete_format["count"]:
            raise ValueError(
                "scorecard expected-tool format-valid count exceeds overall format-valid count"
            )
        if schema_valid_on_tool_decisions["count"] > tool_format["count"]:
            raise ValueError(
                "scorecard expected-tool schema-valid count exceeds strict tool-format count"
            )
        if tool_name_case_exact["count"] > tool_format["count"]:
            raise ValueError("scorecard exact tool-name count exceeds strict tool-format count")
        if whole_call_exact["count"] > schema_valid_on_tool_decisions["count"]:
            raise ValueError(
                "scorecard whole-call exact count exceeds expected-tool schema-valid count"
            )
        if whole_call_exact["count"] > tool_name_case_exact["count"]:
            raise ValueError("scorecard whole-call exact count exceeds exact tool-name count")
        if abstention["count"] > complete_format["count"] - tool_format["count"]:
            raise ValueError("scorecard abstentions exceed non-tool complete-format-valid outputs")
        v2_metrics = {
            "tool_format_validity_on_tool_decisions": tool_format,
            "schema_validity_on_tool_decisions": schema_valid_on_tool_decisions,
            "tool_name_case_exact": tool_name_case_exact,
            "whole_call_exact": whole_call_exact,
            "abstention": abstention,
        }

    predictions = _mapping(scorecard.get("predictions"), label="scorecard.predictions")
    records = _positive_int(predictions.get("records"), label="scorecard.predictions.records")
    complete = _nonnegative_int(
        predictions.get("complete"),
        label="scorecard.predictions.complete",
    )
    eos = _nonnegative_int(
        predictions.get("terminated_by_eos"),
        label="scorecard.predictions.terminated_by_eos",
    )
    if records != assistant_decisions or complete != completion["count"] or eos != complete:
        raise ValueError("scorecard prediction completion accounting is inconsistent")
    if predictions.get("raw_outputs_retained") is not False:
        raise ValueError("scorecard must declare that raw outputs were not retained")
    finish_reasons = _mapping(
        predictions.get("finish_reasons"),
        label="scorecard.predictions.finish_reasons",
    )
    if set(finish_reasons) - {"eos", "length"}:
        raise ValueError("scorecard contains unsupported finish reasons")
    finish_counts = {
        key: _nonnegative_int(value, label=f"scorecard finish reason {key}")
        for key, value in finish_reasons.items()
    }
    if (
        sum(finish_counts.values()) != assistant_decisions
        or finish_counts.get("eos", 0) != eos
        or finish_counts.get("length", 0) != assistant_decisions - complete
    ):
        raise ValueError("scorecard finish-reason accounting is inconsistent")

    provenance = _mapping(result["provenance"], label="scorecard.provenance")
    checkpoint = _mapping(provenance.get("checkpoint"), label="scorecard checkpoint provenance")
    if (
        checkpoint.get("stage") != "sft"
        or checkpoint.get("conversation_prompt_contract") != _PROMPT_CONTRACT
    ):
        raise ValueError("scorecard checkpoint is not an SFT full-catalog checkpoint")
    checkpoint_sha256 = _valid_sha256(
        checkpoint.get("sha256"),
        label="scorecard.provenance.checkpoint.sha256",
    )
    checkpoint_path = checkpoint.get("path")
    checkpoint_bytes = checkpoint.get("bytes")
    if readiness_schema_version == SCHEMA_VERSION:
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("scorecard checkpoint path must be non-empty text")
        checkpoint_bytes = _positive_int(
            checkpoint_bytes,
            label="scorecard.provenance.checkpoint.bytes",
        )
    cases = _mapping(provenance.get("cases"), label="scorecard case provenance")
    if (
        cases.get("split") != "eval"
        or cases.get("rule_verified") is not True
        or cases.get("environment_executed") is not False
    ):
        raise ValueError("scorecard cases are not verified held-out eval artifacts")
    if cases.get("case_set_sha256") != case_set_sha256:
        raise ValueError("scorecard case-set provenance mismatch")
    eval_jsonl = _mapping(cases.get("jsonl"), label="scorecard eval JSONL provenance")
    eval_jsonl_sha256 = _valid_sha256(
        eval_jsonl.get("sha256"),
        label="scorecard.provenance.cases.jsonl.sha256",
    )
    selection = _mapping(cases.get("selection"), label="scorecard case selection")
    if selection.get("algorithm") != "greedy_uncovered_strata_then_semantic_sha256_fill_v1":
        raise ValueError("scorecard does not use the frozen deterministic eval selector")
    selection_source = _mapping(selection.get("source"), label="scorecard selection source")
    eval_semantic_set_sha256 = _valid_sha256(
        selection_source.get("semantic_set_sha256"),
        label="scorecard selection source semantic_set_sha256",
    )
    selected_eval_semantic_set_sha256 = None
    if readiness_schema_version == SCHEMA_VERSION:
        selection_selected = _mapping(
            selection.get("selected"),
            label="scorecard selection selected subset",
        )
        selected_eval_semantic_set_sha256 = _valid_sha256(
            selection_selected.get("semantic_set_sha256"),
            label="scorecard selection selected semantic_set_sha256",
        )
        if (
            selection_selected.get("rows") != conversations
            or selection_selected.get("assistant_decisions") != assistant_decisions
        ):
            raise ValueError("scorecard selected-subset row/decision accounting is inconsistent")
    generation = _mapping(
        provenance.get("generation"),
        label="scorecard generation provenance",
    )
    if (
        generation.get("conversation_prompt_contract") != _PROMPT_CONTRACT
        or generation.get("truncation") != "forbidden"
        or generation.get("temperature") != 0.0
    ):
        raise ValueError("scorecard generation contract is not deterministic greedy decoding")

    truncations = assistant_decisions - complete
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": checkpoint_path,
        "checkpoint_bytes": checkpoint_bytes,
        "eval_jsonl_sha256": eval_jsonl_sha256,
        "eval_semantic_set_sha256": eval_semantic_set_sha256,
        "selected_eval_semantic_set_sha256": selected_eval_semantic_set_sha256,
        "case_set_sha256": case_set_sha256,
        "conversations": conversations,
        "assistant_decisions": assistant_decisions,
        "tool_decisions": tool_decisions,
        "no_tool_decisions": no_tool_decisions,
        "generation_completion": completion,
        "generation_truncation": _observed_rate(truncations, assistant_decisions),
        "complete_format_validity": complete_format,
        "schema_validity_on_tool_attempts": schema_valid,
        "action_exact": action_exact,
        **v2_metrics,
        "finish_reasons": dict(sorted(finish_counts.items())),
    }


def _artifact_paths_and_hashes(
    values: Any,
    *,
    split: str,
    label: str,
) -> tuple[set[str], set[str]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{label} must be a non-empty sequence")
    paths: set[str] = set()
    hashes: set[str] = set()
    for index, value in enumerate(values):
        artifact = _mapping(value, label=f"{label}[{index}]")
        if artifact.get("split") != split:
            raise ValueError(f"{label}[{index}] does not declare split={split!r}")
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"{label}[{index}].path must be non-empty text")
        jsonl = _mapping(artifact.get("jsonl"), label=f"{label}[{index}].jsonl")
        sha256 = _valid_sha256(
            jsonl.get("sha256"),
            label=f"{label}[{index}].jsonl.sha256",
        )
        paths.add(path)
        hashes.add(sha256)
    return paths, hashes


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate names")
    return list(value)


def _learning_rate_list(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    rates = [_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if any(rate < 0.0 for rate in rates):
        raise ValueError(f"{label} must contain non-negative learning rates")
    return rates


def _validate_policy_transition(
    value: Any,
    *,
    attempted_steps: int,
) -> dict[str, Any]:
    transition = _exact_mapping(
        value,
        _POLICY_TRANSITION_KEYS,
        label="RL policy transition",
    )
    if transition["contract"] != "exact_named_policy_parameter_comparison_v1":
        raise ValueError("RL policy transition contract is unsupported")
    model_names = _string_list(
        transition["model_named_parameter_names"],
        label="RL policy model parameter names",
    )
    compared_names = _string_list(
        transition["compared_model_parameter_names"],
        label="RL compared model parameter names",
    )
    changed_names = _string_list(
        transition["changed_model_parameter_names"],
        label="RL changed model parameter names",
    )
    model_count = _positive_int(
        transition["model_parameter_count"],
        label="RL model parameter count",
    )
    compared_count = _positive_int(
        transition["compared_model_parameter_count"],
        label="RL compared model parameter count",
    )
    changed_count = _nonnegative_int(
        transition["changed_model_parameter_count"],
        label="RL changed model parameter count",
    )
    if (
        model_count != len(model_names)
        or compared_count != len(compared_names)
        or changed_count != len(changed_names)
    ):
        raise ValueError("RL policy transition parameter counts are inconsistent")
    if compared_names != model_names:
        raise ValueError("RL policy transition did not compare every named model parameter")
    if any(name not in set(compared_names) for name in changed_names):
        raise ValueError("RL changed parameter names are outside the compared policy scope")

    changed = transition["at_least_one_policy_tensor_changed"]
    if not isinstance(changed, bool) or changed is not bool(changed_names):
        raise ValueError("RL policy transition boolean disagrees with changed parameter names")
    first_changed = transition["first_changed_model_parameter"]
    expected_first_changed = changed_names[0] if changed_names else None
    if first_changed != expected_first_changed:
        raise ValueError("RL policy transition first-changed parameter is inconsistent")
    initial_state_sha256 = _valid_sha256(
        transition["initial_model_state_sha256"],
        label="RL initial model state SHA-256",
    )
    final_state_sha256 = _valid_sha256(
        transition["final_model_state_sha256"],
        label="RL final model state SHA-256",
    )
    if (initial_state_sha256 != final_state_sha256) is not changed:
        raise ValueError("RL policy state digest and tensor-transition evidence disagree")

    production_steps = _positive_int(
        transition["production_schedule_total_steps"],
        label="RL production schedule total steps",
    )
    execution_limit = _positive_int(
        transition["execution_rollout_step_limit"],
        label="RL execution rollout step limit",
    )
    first_nonzero_step = _nonnegative_int(
        transition["first_nonzero_learning_rate_step"],
        label="RL first nonzero learning-rate step",
    )
    if (
        execution_limit != attempted_steps
        or execution_limit != first_nonzero_step + 1
        or production_steps < execution_limit
    ):
        raise ValueError("RL bounded production-schedule prefix is inconsistent")
    expected_rates = _learning_rate_list(
        transition["expected_learning_rates"],
        label="RL expected learning rates",
    )
    actual_rates = _learning_rate_list(
        transition["actual_learning_rates"],
        label="RL actual learning rates",
    )
    if len(expected_rates) != attempted_steps or len(actual_rates) != attempted_steps:
        raise ValueError("RL learning-rate sequence length does not match attempted steps")
    rates_match = expected_rates == actual_rates
    if transition["actual_learning_rates_match_expected"] is not rates_match:
        raise ValueError("RL learning-rate match boolean is inconsistent")
    nonzero_executed = any(rate > 0.0 for rate in actual_rates)
    if transition["nonzero_learning_rate_executed"] is not nonzero_executed:
        raise ValueError("RL nonzero learning-rate boolean is inconsistent")
    if (
        any(rate > 0.0 for rate in expected_rates[:first_nonzero_step])
        or expected_rates[first_nonzero_step] <= 0.0
    ):
        raise ValueError("RL expected learning rates do not identify the first nonzero step")

    final_optimizer_rates = _learning_rate_list(
        transition["final_optimizer_learning_rates"],
        label="RL final optimizer learning rates",
    )
    optimizer_rates_match = all(
        rate == expected_rates[-1] for rate in final_optimizer_rates
    )
    if (
        transition["final_optimizer_learning_rate_matches_expected"]
        is not optimizer_rates_match
    ):
        raise ValueError("RL final optimizer learning-rate boolean is inconsistent")
    return {
        "contract": transition["contract"],
        "model_parameter_count": model_count,
        "compared_model_parameter_count": compared_count,
        "changed_model_parameter_count": changed_count,
        "changed_model_parameter_names": changed_names,
        "initial_model_state_sha256": initial_state_sha256,
        "final_model_state_sha256": final_state_sha256,
        "at_least_one_policy_tensor_changed": changed,
        "production_schedule_total_steps": production_steps,
        "execution_rollout_step_limit": execution_limit,
        "first_nonzero_learning_rate_step": first_nonzero_step,
        "expected_learning_rates": expected_rates,
        "actual_learning_rates": actual_rates,
        "actual_learning_rates_match_expected": rates_match,
        "nonzero_learning_rate_executed": nonzero_executed,
        "final_optimizer_learning_rates": final_optimizer_rates,
        "final_optimizer_learning_rate_matches_expected": optimizer_rates_match,
    }


def _validate_preflight(
    payload: Mapping[str, Any],
    *,
    readiness_schema_version: int,
) -> dict[str, Any]:
    if (
        payload.get("kind") != _PREFLIGHT_KIND
        or not _is_schema_version(
            payload.get("schema_version"),
            LEGACY_SCHEMA_VERSION,
        )
    ):
        raise ValueError("unsupported one-update RL preflight receipt")
    status = payload.get("status")
    if status not in {"passed", "failed"}:
        raise ValueError("RL preflight status must be passed or failed")

    source = _mapping(payload.get("source"), label="RL preflight source")
    source_artifacts_untouched = source.get("source_artifacts_untouched")
    production_output_untouched = source.get("production_rl_output_untouched")
    if not isinstance(source_artifacts_untouched, bool) or not isinstance(
        production_output_untouched,
        bool,
    ):
        raise ValueError("RL preflight source-safety fields must be boolean")
    parent = _mapping(
        source.get("sft_parent_checkpoint"),
        label="RL preflight SFT parent",
    )
    parent_checkpoint_sha256 = _valid_sha256(
        parent.get("sha256"),
        label="RL preflight SFT parent SHA-256",
    )
    parent_checkpoint_path = parent.get("path")
    source_config_path = None
    source_config_canonical_sha256 = None
    if readiness_schema_version == SCHEMA_VERSION:
        if not isinstance(parent_checkpoint_path, str) or not parent_checkpoint_path:
            raise ValueError("RL preflight SFT parent path must be non-empty text")
        source_config = _mapping(
            source.get("config"),
            label="RL preflight source config",
        )
        source_config_path = source_config.get("path")
        if not isinstance(source_config_path, str) or not source_config_path:
            raise ValueError("RL preflight source config path must be non-empty text")
        source_config_canonical_sha256 = _valid_sha256(
            source_config.get("canonical_sha256"),
            label="RL preflight source config canonical SHA-256",
        )

    effective = _mapping(payload.get("effective"), label="RL preflight effective config")
    contract = _mapping(effective.get("contract"), label="RL preflight effective contract")
    if (
        contract.get("stage") != "rl"
        or contract.get("resume") is not False
        or contract.get("checkpoint_output") != "isolated_work_directory"
    ):
        raise ValueError("RL preflight is not an isolated RL execution")
    if readiness_schema_version == LEGACY_SCHEMA_VERSION:
        if contract.get("rollout_steps") != 1:
            raise ValueError("RL preflight is not an isolated one-rollout-step execution")
        contract_rollout_steps = 1
    else:
        contract_rollout_steps = _positive_int(
            contract.get("execution_rollout_step_limit"),
            label="RL preflight execution rollout step limit",
        )
        if contract.get("rollout_steps") != contract_rollout_steps:
            raise ValueError("RL preflight bounded rollout-step contract is inconsistent")
        production_schedule_steps = _positive_int(
            contract.get("production_schedule_total_steps"),
            label="RL preflight production schedule total steps",
        )
        if production_schedule_steps < contract_rollout_steps:
            raise ValueError("RL preflight bounded prefix exceeds the production schedule")
    group_size = _positive_int(contract.get("group_size"), label="RL preflight group_size")
    prompts_per_step = _positive_int(
        contract.get("prompts_per_step"),
        label="RL preflight prompts_per_step",
    )
    configured_policy_epochs = _positive_int(
        contract.get("configured_policy_epochs_preserved"),
        label="RL preflight configured policy epochs",
    )

    metrics = _mapping(payload.get("metrics"), label="RL preflight metrics")
    if metrics.get("stage") != "rl":
        raise ValueError("RL preflight metrics do not record stage='rl'")
    execution = None
    if readiness_schema_version == SCHEMA_VERSION:
        execution_record = _mapping(
            metrics.get("execution"),
            label="RL preflight execution",
        )
        execution = {}
        for key in (
            "requested_device",
            "resolved_device",
            "requested_dtype",
            "resolved_dtype",
        ):
            item = execution_record.get(key)
            if not isinstance(item, str) or not item:
                raise ValueError(f"RL preflight execution.{key} must be non-empty text")
            execution[key] = item
    accounting = _mapping(
        metrics.get("rl_accounting"),
        label="RL preflight accounting",
    )
    attempted_steps = _positive_int(
        accounting.get("attempted_rollout_steps"),
        label="RL attempted rollout steps",
    )
    attempted_groups = _positive_int(
        accounting.get("attempted_groups"),
        label="RL attempted groups",
    )
    attempted_rollouts = _positive_int(
        accounting.get("attempted_rollouts"),
        label="RL attempted rollouts",
    )
    if attempted_steps != contract_rollout_steps:
        raise ValueError("RL preflight accounting does not match the bounded rollout prefix")
    if (
        attempted_groups != prompts_per_step * attempted_steps
        or attempted_rollouts != attempted_groups * group_size
    ):
        raise ValueError("RL preflight group/rollout accounting is inconsistent")
    informative_groups = _nonnegative_int(
        accounting.get("informative_groups"),
        label="RL informative groups",
    )
    if informative_groups > attempted_groups:
        raise ValueError("RL informative groups exceed attempted groups")
    zero_signal_steps = _nonnegative_int(
        accounting.get("zero_signal_steps"),
        label="RL zero-signal steps",
    )
    if zero_signal_steps > attempted_steps:
        raise ValueError("RL zero-signal steps exceed attempted steps")
    if (informative_groups == 0) != (zero_signal_steps == attempted_steps):
        raise ValueError("RL informative-group and zero-signal accounting disagree")
    realized_updates = _nonnegative_int(
        accounting.get("realized_optimizer_updates"),
        label="RL realized optimizer updates",
    )
    policy_epochs = _positive_int(
        accounting.get("policy_epochs_per_informative_batch"),
        label="RL policy epochs",
    )
    if policy_epochs != configured_policy_epochs:
        raise ValueError("RL configured and recorded policy epochs disagree")
    if realized_updates > attempted_steps * policy_epochs:
        raise ValueError("RL realized optimizer updates exceed the bounded-prefix policy budget")
    if informative_groups == 0 and realized_updates != 0:
        raise ValueError("RL optimizer updates were reported without an informative group")
    if contract.get("realized_optimizer_updates") != realized_updates:
        raise ValueError("RL effective contract and accounting disagree on optimizer updates")

    measurement = _mapping(payload.get("measurement"), label="RL preflight measurement")
    policy_transition = None
    if readiness_schema_version == SCHEMA_VERSION:
        policy_transition = _validate_policy_transition(
            measurement.get("policy_transition"),
            attempted_steps=attempted_steps,
        )
        contract_expected_rates = _learning_rate_list(
            contract.get("expected_learning_rates"),
            label="RL preflight contract expected learning rates",
        )
        accounting_actual_rates = _learning_rate_list(
            accounting.get("learning_rate_history"),
            label="RL preflight accounting learning-rate history",
        )
        if (
            contract.get("production_schedule_total_steps")
            != policy_transition["production_schedule_total_steps"]
            or contract.get("execution_rollout_step_limit")
            != policy_transition["execution_rollout_step_limit"]
            or contract.get("first_nonzero_learning_rate_step")
            != policy_transition["first_nonzero_learning_rate_step"]
            or contract_expected_rates
            != policy_transition["expected_learning_rates"]
            or accounting_actual_rates != policy_transition["actual_learning_rates"]
        ):
            raise ValueError(
                "RL policy transition disagrees with contract/accounting learning rates"
            )
        fixed_horizon = _mapping(
            accounting.get("fixed_horizon_progress"),
            label="RL preflight fixed-horizon progress",
        )
        if (
            fixed_horizon.get("planned_rollout_steps")
            != policy_transition["production_schedule_total_steps"]
            or fixed_horizon.get("completed_rollout_steps") != attempted_steps
            or fixed_horizon.get("execution_rollout_step_limit")
            != policy_transition["execution_rollout_step_limit"]
            or fixed_horizon.get("bounded_prefix")
            is not (
                attempted_steps
                < policy_transition["production_schedule_total_steps"]
            )
        ):
            raise ValueError("RL preflight fixed-horizon progress is inconsistent")
    observation = _mapping(
        measurement.get("rollout_observability"),
        label="RL rollout observability",
    )
    parsing = _mapping(observation.get("parsing"), label="RL parsing observation")
    parser_format_valid = _nonnegative_int(
        parsing.get("parser_format_valid_rollouts"),
        label="RL parser-format-valid rollouts",
    )
    complete_parser_valid = _nonnegative_int(
        parsing.get("complete_parser_format_valid_rollouts"),
        label="RL complete parser-valid rollouts",
    )
    tool_syntax = _nonnegative_int(
        parsing.get("parser_tool_syntax_rollouts"),
        label="RL tool-syntax rollouts",
    )
    tool_reward_rollouts = _nonnegative_int(
        parsing.get("tool_reward_rollouts"),
        label="RL tool-reward rollouts",
    )
    text_reward_rollouts = _nonnegative_int(
        parsing.get("text_reward_rollouts"),
        label="RL text-reward rollouts",
    )
    schema_valid_tool = _nonnegative_int(
        parsing.get("strict_tool_format_valid_rollouts"),
        label="RL strict schema-valid tool rollouts",
    )
    if any(
        value > attempted_rollouts
        for value in (parser_format_valid, complete_parser_valid, tool_syntax)
    ):
        raise ValueError("RL parsing count exceeds attempted rollouts")
    if complete_parser_valid > parser_format_valid:
        raise ValueError("RL complete parser-valid count exceeds parser-valid count")
    if tool_reward_rollouts + text_reward_rollouts != attempted_rollouts:
        raise ValueError("RL tool/text reward rows do not cover every rollout")
    if schema_valid_tool > tool_reward_rollouts or schema_valid_tool > parser_format_valid:
        raise ValueError("RL schema-valid tool count is inconsistent")

    reward = _mapping(observation.get("reward"), label="RL reward observation")
    distribution = reward.get("distribution")
    if not isinstance(distribution, list) or not distribution:
        raise ValueError("RL reward distribution must be a non-empty list")
    distribution_summary: list[dict[str, Any]] = []
    observed_reward_hex: set[str] = set()
    distribution_count = 0
    for index, value in enumerate(distribution):
        row = _mapping(value, label=f"RL reward distribution[{index}]")
        reward_value = _finite_number(
            row.get("reward"),
            label=f"RL reward distribution[{index}].reward",
        )
        reward_hex = row.get("reward_hex")
        if not isinstance(reward_hex, str) or reward_hex != reward_value.hex():
            raise ValueError(f"RL reward distribution[{index}].reward_hex is inconsistent")
        if reward_hex in observed_reward_hex:
            raise ValueError("RL reward distribution contains duplicate values")
        observed_reward_hex.add(reward_hex)
        count = _positive_int(
            row.get("count"),
            label=f"RL reward distribution[{index}].count",
        )
        distribution_count += count
        distribution_summary.append(
            {"reward": reward_value, "reward_hex": reward_hex, "count": count}
        )
    if distribution_count != attempted_rollouts:
        raise ValueError("RL reward distribution does not cover every rollout")
    unique_reward_values = _positive_int(
        reward.get("unique_values"),
        label="RL unique reward values",
    )
    if unique_reward_values != len(distribution_summary):
        raise ValueError("RL unique reward count does not match the distribution")
    if unique_reward_values == 1 and informative_groups != 0:
        raise ValueError("RL globally constant rewards cannot produce informative groups")
    exact_successes = _nonnegative_int(
        reward.get("exact_success_rollouts"),
        label="RL exact-success rollouts",
    )
    if exact_successes > attempted_rollouts:
        raise ValueError("RL exact successes exceed attempted rollouts")

    truncation = _mapping(observation.get("truncation"), label="RL truncation observation")
    truncated = _nonnegative_int(
        truncation.get("truncated_rollouts"),
        label="RL truncated rollouts",
    )
    tokens = _mapping(observation.get("tokens"), label="RL token observation")
    generated_eos = _nonnegative_int(
        tokens.get("generated_eos_tokens"),
        label="RL generated EOS tokens",
    )
    if truncated > attempted_rollouts or generated_eos > attempted_rollouts:
        raise ValueError("RL termination counts exceed attempted rollouts")
    if generated_eos + truncated != attempted_rollouts:
        raise ValueError("RL rollouts are not fully accounted for as EOS or length-capped")
    if (
        accounting.get("truncated_rollouts") != truncated
        or accounting.get("generated_eos_tokens") != generated_eos
    ):
        raise ValueError("RL accounting and rollout termination observation disagree")

    data = _mapping(metrics.get("data"), label="RL preflight data evidence")
    if data.get("conversation_prompt_contract") != _PROMPT_CONTRACT:
        raise ValueError("RL preflight data prompt contract is unsupported")
    train_paths, train_hashes = _artifact_paths_and_hashes(
        data.get("train_artifacts"),
        split="train",
        label="RL train artifacts",
    )
    eval_paths, eval_hashes = _artifact_paths_and_hashes(
        data.get("eval_artifacts"),
        split="eval",
        label="RL eval artifacts",
    )
    if train_paths & eval_paths or train_hashes & eval_hashes:
        raise ValueError("RL train/eval leakage detected in declared artifact identities")
    split_audit = _mapping(data.get("split_audit"), label="RL split audit")
    selected_split_audit = _mapping(
        data.get("selected_eval_split_audit"),
        label="RL selected-eval split audit",
    )
    for label, audit in (
        ("RL split audit", split_audit),
        ("RL selected-eval split audit", selected_split_audit),
    ):
        if audit.get("row_overlap") != 0 or audit.get("prompt_overlap") != 0:
            raise ValueError(f"{label} reports train/eval leakage")
    eval_dataset_sha256 = _valid_sha256(
        split_audit.get("eval_dataset_sha256"),
        label="RL split audit eval_dataset_sha256",
    )
    selected_eval_dataset_sha256 = _valid_sha256(
        selected_split_audit.get("eval_dataset_sha256"),
        label="RL selected split audit eval_dataset_sha256",
    )

    heldout = _mapping(metrics.get("heldout_eval"), label="RL held-out evaluation")
    heldout_contract = _mapping(
        heldout.get("contract"),
        label="RL held-out evaluation contract",
    )
    if (
        heldout_contract.get("split") != "explicit_disjoint_eval_conversations"
        or heldout_contract.get("same_rows_pre_post") is not True
        or heldout_contract.get("current_gold_in_prompt") is not False
        or heldout_contract.get("conversation_prompt_contract") != _PROMPT_CONTRACT
    ):
        raise ValueError("RL held-out evaluation contract is not isolated and frozen")

    minimum_coverage = _mapping(
        data.get("preflight_minimum_coverage"),
        label="RL preflight minimum coverage",
    )
    selection_audit = _mapping(
        minimum_coverage.get("selection_audit"),
        label="RL preflight selection audit",
    )
    selection_source = _mapping(
        selection_audit.get("source"),
        label="RL preflight selection source",
    )
    selection_selected = _mapping(
        selection_audit.get("selected"),
        label="RL preflight selected subset",
    )
    eval_semantic_set_sha256 = _valid_sha256(
        selection_source.get("semantic_set_sha256"),
        label="RL preflight selection source semantic_set_sha256",
    )
    if eval_semantic_set_sha256 != eval_dataset_sha256:
        raise ValueError("RL selector source and split audit disagree on eval identity")
    selected_semantic_set_sha256 = _valid_sha256(
        selection_selected.get("semantic_set_sha256"),
        label="RL preflight selected semantic_set_sha256",
    )
    if selected_semantic_set_sha256 != selected_eval_dataset_sha256:
        raise ValueError("RL selector subset and selected split audit disagree on eval identity")

    validation_errors = payload.get("validation_errors")
    if not isinstance(validation_errors, list) or any(
        not isinstance(error, str) or not error for error in validation_errors
    ):
        raise ValueError("RL preflight validation_errors must be a list of non-empty strings")
    if status == "passed" and (validation_errors or payload.get("error") is not None):
        raise ValueError("passed RL preflight contains errors")

    result = {
        "status": status,
        "validation_errors": list(validation_errors),
        "source_artifacts_untouched": source_artifacts_untouched,
        "production_output_untouched": production_output_untouched,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "eval_jsonl_sha256": sorted(eval_hashes),
        "eval_semantic_set_sha256": eval_semantic_set_sha256,
        "selected_eval_semantic_set_sha256": selected_semantic_set_sha256,
        "attempted_steps": attempted_steps,
        "attempted_groups": attempted_groups,
        "attempted_rollouts": attempted_rollouts,
        "group_size": group_size,
        "tool_reward_rollouts": tool_reward_rollouts,
        "text_reward_rollouts": text_reward_rollouts,
        "tool_syntax": _observed_rate(tool_syntax, attempted_rollouts),
        "complete_parser_validity": _observed_rate(
            complete_parser_valid,
            attempted_rollouts,
        ),
        "schema_valid_tool_actions": _observed_rate(
            schema_valid_tool,
            tool_reward_rollouts,
        ),
        "exact_success": _observed_rate(exact_successes, attempted_rollouts),
        "truncation": _observed_rate(truncated, attempted_rollouts),
        "generated_eos": _observed_rate(generated_eos, attempted_rollouts),
        "reward_distribution": distribution_summary,
        "unique_reward_values": unique_reward_values,
        "informative_groups": informative_groups,
        "informative_group_rate": informative_groups / attempted_groups,
        "realized_optimizer_updates": realized_updates,
        "configured_policy_epochs": configured_policy_epochs,
    }
    if readiness_schema_version == SCHEMA_VERSION:
        assert (
            isinstance(parent_checkpoint_path, str)
            and isinstance(source_config_path, str)
            and isinstance(source_config_canonical_sha256, str)
            and execution is not None
            and policy_transition is not None
        )
        result.update(
            {
                "parent_checkpoint_path": parent_checkpoint_path,
                "source_config_path": source_config_path,
                "source_config_canonical_sha256": source_config_canonical_sha256,
                "execution": execution,
                "policy_transition": policy_transition,
            }
        )
    return result


def _validate_teacher_forced_measurement(
    value: Any,
    *,
    label: str,
    expected_rows: int,
    expected_loss_tokens: int | None = None,
) -> dict[str, Any]:
    measurement = _mapping(value, label=label)
    rows = _positive_int(measurement.get("rows"), label=f"{label}.rows")
    if rows != expected_rows:
        raise ValueError(f"{label}.rows does not match the scorecard decision count")
    loss_tokens = _positive_int(
        measurement.get("assistant_loss_tokens"),
        label=f"{label}.assistant_loss_tokens",
    )
    if expected_loss_tokens is not None and loss_tokens != expected_loss_tokens:
        raise ValueError(f"{label}.assistant_loss_tokens changed across evaluation")
    mean_loss = _finite_number(measurement.get("mean_loss"), label=f"{label}.mean_loss")
    if mean_loss < 0.0:
        raise ValueError(f"{label}.mean_loss must be non-negative")
    token_accuracy = _rate_threshold(
        measurement.get("assistant_token_accuracy"),
        label=f"{label}.assistant_token_accuracy",
        positive=False,
    )
    sequence_accuracy = _rate_threshold(
        measurement.get("assistant_sequence_accuracy"),
        label=f"{label}.assistant_sequence_accuracy",
        positive=False,
    )
    return {
        "rows": rows,
        "assistant_loss_tokens": loss_tokens,
        "mean_loss": mean_loss,
        "assistant_token_accuracy": token_accuracy,
        "assistant_sequence_accuracy": sequence_accuracy,
    }


def _validate_sft_metrics(
    payload: Mapping[str, Any],
    *,
    scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("stage") != "sft":
        raise ValueError("SFT metrics evidence does not record stage='sft'")
    if payload.get("conversation_prompt_contract") != _PROMPT_CONTRACT:
        raise ValueError("SFT metrics evidence uses an unsupported prompt contract")
    checkpoint_path = payload.get("checkpoint")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("SFT metrics checkpoint path must be non-empty text")
    checkpoint_artifact = _hash_regular(
        Path(checkpoint_path),
        label="SFT metrics checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if (
        checkpoint_artifact["sha256"] != scorecard["checkpoint_sha256"]
        or checkpoint_artifact["bytes"] != scorecard["checkpoint_bytes"]
        or checkpoint_artifact["path"] != scorecard["checkpoint_path"]
    ):
        raise ValueError("SFT metrics and scorecard do not reference the same checkpoint")

    heldout = _mapping(payload.get("heldout_eval"), label="SFT held-out evaluation")
    contract = _mapping(
        heldout.get("contract"),
        label="SFT held-out evaluation contract",
    )
    if (
        contract.get("kind") != "deterministic_teacher_forced_assistant_tokens"
        or contract.get("same_rows_pre_post") is not True
        or contract.get("conversation_prompt_contract") != _PROMPT_CONTRACT
        or contract.get("row_order") != "configured_jsonl_assistant_decision_order"
    ):
        raise ValueError("SFT teacher-forced evaluation contract is not frozen and deterministic")
    selection = _mapping(
        contract.get("selection"),
        label="SFT held-out evaluation selection",
    )
    if selection.get("algorithm") != "greedy_uncovered_strata_then_semantic_sha256_fill_v1":
        raise ValueError("SFT metrics do not use the frozen deterministic eval selector")
    selection_source = _mapping(
        selection.get("source"),
        label="SFT held-out evaluation selection source",
    )
    source_semantic_set_sha256 = _valid_sha256(
        selection_source.get("semantic_set_sha256"),
        label="SFT selection source semantic_set_sha256",
    )
    selected = _mapping(
        selection.get("selected"),
        label="SFT held-out evaluation selected subset",
    )
    selected_semantic_set_sha256 = _valid_sha256(
        selected.get("semantic_set_sha256"),
        label="SFT selection selected semantic_set_sha256",
    )
    if (
        source_semantic_set_sha256 != scorecard["eval_semantic_set_sha256"]
        or selected_semantic_set_sha256 != scorecard["selected_eval_semantic_set_sha256"]
        or selected.get("rows") != scorecard["conversations"]
        or selected.get("assistant_decisions") != scorecard["assistant_decisions"]
    ):
        raise ValueError("SFT metrics and scorecard do not share the same held-out subset")

    data = _mapping(payload.get("data"), label="SFT metrics data evidence")
    if data.get("heldout_content_overlap") != 0 or data.get("heldout_rendered_prompt_overlap") != 0:
        raise ValueError("SFT metrics report train/eval leakage")

    pre = _validate_teacher_forced_measurement(
        heldout.get("pre"),
        label="SFT held-out pre",
        expected_rows=scorecard["assistant_decisions"],
    )
    post = _validate_teacher_forced_measurement(
        heldout.get("post"),
        label="SFT held-out post",
        expected_rows=scorecard["assistant_decisions"],
        expected_loss_tokens=pre["assistant_loss_tokens"],
    )
    delta = _mapping(heldout.get("delta"), label="SFT held-out delta")
    expected_delta = {
        "mean_loss": post["mean_loss"] - pre["mean_loss"],
        "assistant_token_accuracy": (
            post["assistant_token_accuracy"] - pre["assistant_token_accuracy"]
        ),
        "assistant_sequence_accuracy": (
            post["assistant_sequence_accuracy"] - pre["assistant_sequence_accuracy"]
        ),
    }
    observed_delta: dict[str, float] = {}
    for key, expected in expected_delta.items():
        observed = _finite_number(delta.get(key), label=f"SFT held-out delta.{key}")
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"SFT held-out delta.{key} is inconsistent with pre/post")
        observed_delta[key] = observed

    return {
        "checkpoint": checkpoint_artifact,
        "source_eval_semantic_set_sha256": source_semantic_set_sha256,
        "selected_eval_semantic_set_sha256": selected_semantic_set_sha256,
        "pre": pre,
        "post": post,
        "delta": observed_delta,
        "mean_loss_increase": observed_delta["mean_loss"],
        "assistant_token_accuracy_drop": -observed_delta["assistant_token_accuracy"],
        "assistant_sequence_accuracy_drop": -observed_delta["assistant_sequence_accuracy"],
    }


def _validate_sweep_checkpoint_artifact(value: Any, *, label: str) -> dict[str, Any]:
    artifact = _exact_mapping(value, _SFT_SWEEP_ARTIFACT_KEYS, label=label)
    path = artifact["path"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path must be non-empty text")
    size = _positive_int(artifact["bytes"], label=f"{label}.bytes")
    sha256 = _valid_sha256(artifact["sha256"], label=f"{label}.sha256")
    return {"path": path, "bytes": size, "sha256": sha256}


def _validate_sweep_gate(
    value: Any,
    *,
    label: str,
    observed_key: str,
    expected_observed: float,
    maximum_key: str,
    expected_maximum: float,
) -> bool:
    gate = _mapping(value, label=label)
    observed = _finite_number(gate.get(observed_key), label=f"{label}.{observed_key}")
    maximum = _finite_number(gate.get(maximum_key), label=f"{label}.{maximum_key}")
    if maximum < 0.0:
        raise ValueError(f"{label}.{maximum_key} must be non-negative")
    if not math.isclose(observed, expected_observed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}.{observed_key} is inconsistent with baseline/post")
    if not math.isclose(maximum, expected_maximum, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{label}.{maximum_key} disagrees with sweep thresholds")
    passed = gate.get("passed")
    if not isinstance(passed, bool) or passed is not (observed <= maximum):
        raise ValueError(f"{label}.passed is inconsistent")
    return passed


def _validate_sft_checkpoint_sweep(
    payload: Mapping[str, Any],
    *,
    scorecard: Mapping[str, Any],
    selected_checkpoint_sha256: str,
) -> dict[str, Any]:
    result = _exact_mapping(
        payload,
        _SFT_SWEEP_TOP_LEVEL_KEYS,
        label="SFT checkpoint sweep result",
    )
    if result["kind"] != _SFT_SWEEP_KIND or not _is_schema_version(
        result["schema_version"],
        _SFT_SWEEP_SCHEMA_VERSION,
    ):
        raise ValueError("unsupported SFT checkpoint sweep result")
    if selected_checkpoint_sha256 != scorecard["checkpoint_sha256"]:
        raise ValueError(
            "configured sweep checkpoint and scorecard do not reference the same checkpoint"
        )

    inputs = _mapping(result["inputs"], label="SFT checkpoint sweep inputs")
    expected_parent_checkpoint_sha256 = _valid_sha256(
        inputs.get("expected_parent_checkpoint_sha256"),
        label="SFT checkpoint sweep inputs.expected_parent_checkpoint_sha256",
    )
    expected_eval = _exact_mapping(
        inputs.get("expected_eval"),
        frozenset({"conversations", "assistant_decisions", "assistant_loss_tokens"}),
        label="SFT checkpoint sweep inputs.expected_eval",
    )
    expected_conversations = _positive_int(
        expected_eval["conversations"],
        label="SFT checkpoint sweep inputs.expected_eval.conversations",
    )
    expected_assistant_decisions = _positive_int(
        expected_eval["assistant_decisions"],
        label="SFT checkpoint sweep inputs.expected_eval.assistant_decisions",
    )
    expected_assistant_loss_tokens = _positive_int(
        expected_eval["assistant_loss_tokens"],
        label="SFT checkpoint sweep inputs.expected_eval.assistant_loss_tokens",
    )
    expected_baseline = _exact_mapping(
        inputs.get("expected_baseline"),
        frozenset({"metrics", "absolute_tolerances"}),
        label="SFT checkpoint sweep inputs.expected_baseline",
    )
    expected_baseline_metrics = _mapping(
        expected_baseline["metrics"],
        label="SFT checkpoint sweep expected baseline metrics",
    )
    baseline_tolerances = _exact_mapping(
        expected_baseline["absolute_tolerances"],
        frozenset(
            {
                "mean_loss",
                "assistant_token_accuracy",
                "assistant_sequence_accuracy",
            }
        ),
        label="SFT checkpoint sweep expected baseline tolerances",
    )
    normalized_baseline_tolerances: dict[str, float] = {}
    for key, value in baseline_tolerances.items():
        tolerance = _finite_number(
            value,
            label=f"SFT checkpoint sweep expected baseline tolerances.{key}",
        )
        if tolerance < 0.0:
            raise ValueError(
                f"SFT checkpoint sweep expected baseline tolerances.{key} must be non-negative"
            )
        normalized_baseline_tolerances[key] = tolerance
    identity = _mapping(result["identity"], label="SFT checkpoint sweep identity")
    lineage = _mapping(
        identity.get("lineage"),
        label="SFT checkpoint sweep identity.lineage",
    )
    if lineage.get("parent_checkpoint_sha256") != expected_parent_checkpoint_sha256:
        raise ValueError("SFT checkpoint sweep lineage does not match expected parent checkpoint")

    heldout = _mapping(result["heldout"], label="SFT checkpoint sweep heldout")
    conversations = _positive_int(
        heldout.get("conversations"),
        label="SFT checkpoint sweep heldout.conversations",
    )
    assistant_decisions = _positive_int(
        heldout.get("assistant_decisions"),
        label="SFT checkpoint sweep heldout.assistant_decisions",
    )
    if (
        conversations != scorecard["conversations"]
        or assistant_decisions != scorecard["assistant_decisions"]
        or conversations != expected_conversations
        or assistant_decisions != expected_assistant_decisions
    ):
        raise ValueError("SFT checkpoint sweep and scorecard held-out cardinalities differ")
    heldout_assistant_loss_tokens = _positive_int(
        heldout.get("assistant_loss_tokens"),
        label="SFT checkpoint sweep heldout.assistant_loss_tokens",
    )
    if heldout_assistant_loss_tokens != expected_assistant_loss_tokens:
        raise ValueError("SFT checkpoint sweep assistant-loss-token expectation drifted")
    leakage = _exact_mapping(
        heldout.get("leakage_assurance"),
        frozenset(
            {
                "heldout_content_overlap",
                "heldout_rendered_prompt_overlap",
                "conversation_overlap_audit",
            }
        ),
        label="SFT checkpoint sweep heldout.leakage_assurance",
    )
    if leakage["heldout_content_overlap"] != 0 or leakage["heldout_rendered_prompt_overlap"] != 0:
        raise ValueError("SFT checkpoint sweep reports train/eval leakage")
    overlap_audit = _exact_mapping(
        leakage["conversation_overlap_audit"],
        _CONVERSATION_OVERLAP_KEYS,
        label="SFT checkpoint sweep conversation overlap audit",
    )
    if (
        overlap_audit.get("semantic_overlap") != 0
        or overlap_audit.get("rendered_prompt_overlap") != 0
        or overlap_audit.get("semantic_overlap_sha256") != []
        or overlap_audit.get("rendered_prompt_overlap_sha256") != []
    ):
        raise ValueError("SFT checkpoint sweep conversation overlap audit reports leakage")
    fingerprint_contract = _exact_mapping(
        overlap_audit["fingerprint_contract"],
        frozenset(
            {
                "conversation_prompt_contract",
                "semantic_row",
                "rendered_prompt",
                "set_aggregation",
            }
        ),
        label="SFT checkpoint sweep overlap fingerprint contract",
    )
    if fingerprint_contract["conversation_prompt_contract"] != _PROMPT_CONTRACT:
        raise ValueError("SFT checkpoint sweep overlap prompt contract is unsupported")
    for key in ("semantic_row", "rendered_prompt", "set_aggregation"):
        if not isinstance(fingerprint_contract[key], str) or not fingerprint_contract[key]:
            raise ValueError(
                f"SFT checkpoint sweep overlap fingerprint contract.{key} must be non-empty text"
            )
    for key in (
        "left_rows",
        "right_rows",
        "left_rendered_prompts",
        "right_rendered_prompts",
    ):
        _positive_int(
            overlap_audit[key],
            label=f"SFT checkpoint sweep conversation overlap audit.{key}",
        )
    for key in (
        "left_semantic_set_sha256",
        "right_semantic_set_sha256",
        "left_rendered_prompt_set_sha256",
        "right_rendered_prompt_set_sha256",
    ):
        _valid_sha256(
            overlap_audit.get(key),
            label=f"SFT checkpoint sweep conversation overlap audit.{key}",
        )
    if overlap_audit["right_semantic_set_sha256"] != scorecard["eval_semantic_set_sha256"]:
        raise ValueError("SFT checkpoint sweep overlap audit does not bind the scorecard eval set")

    sources = heldout.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("SFT checkpoint sweep heldout.sources must be non-empty")
    source_hashes: set[str] = set()
    for index, value in enumerate(sources):
        source = _mapping(value, label=f"SFT checkpoint sweep heldout.sources[{index}]")
        path = source.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"SFT checkpoint sweep heldout.sources[{index}].path must be non-empty text"
            )
        source_hashes.add(
            _valid_sha256(
                source.get("sha256"),
                label=f"SFT checkpoint sweep heldout.sources[{index}].sha256",
            )
        )
    if scorecard["eval_jsonl_sha256"] not in source_hashes:
        raise ValueError("SFT checkpoint sweep and scorecard do not share the eval JSONL")

    contract = _mapping(
        heldout.get("contract"),
        label="SFT checkpoint sweep heldout.contract",
    )
    if (
        contract.get("kind") != "deterministic_teacher_forced_assistant_tokens"
        or contract.get("same_rows_pre_post") is not True
        or contract.get("conversation_prompt_contract") != _PROMPT_CONTRACT
        or contract.get("row_order") != "configured_jsonl_assistant_decision_order"
    ):
        raise ValueError("SFT checkpoint sweep held-out contract is unsupported")
    selection = _mapping(
        contract.get("selection"),
        label="SFT checkpoint sweep held-out selection",
    )
    if selection.get("algorithm") != "greedy_uncovered_strata_then_semantic_sha256_fill_v1":
        raise ValueError("SFT checkpoint sweep does not use the frozen eval selector")
    selection_source = _mapping(
        selection.get("source"),
        label="SFT checkpoint sweep held-out selection source",
    )
    source_semantic_set_sha256 = _valid_sha256(
        selection_source.get("semantic_set_sha256"),
        label="SFT checkpoint sweep selection source semantic_set_sha256",
    )
    selection_selected = _mapping(
        selection.get("selected"),
        label="SFT checkpoint sweep held-out selected subset",
    )
    selected_semantic_set_sha256 = _valid_sha256(
        selection_selected.get("semantic_set_sha256"),
        label="SFT checkpoint sweep selected semantic_set_sha256",
    )
    if (
        source_semantic_set_sha256 != scorecard["eval_semantic_set_sha256"]
        or selected_semantic_set_sha256 != scorecard["selected_eval_semantic_set_sha256"]
        or selection_selected.get("rows") != conversations
        or selection_selected.get("assistant_decisions") != assistant_decisions
    ):
        raise ValueError("SFT checkpoint sweep and scorecard held-out selections differ")

    baseline = _validate_teacher_forced_measurement(
        heldout.get("baseline"),
        label="SFT checkpoint sweep baseline",
        expected_rows=assistant_decisions,
        expected_loss_tokens=heldout_assistant_loss_tokens,
    )
    pinned_baseline = _validate_teacher_forced_measurement(
        expected_baseline_metrics,
        label="SFT checkpoint sweep pinned baseline",
        expected_rows=assistant_decisions,
        expected_loss_tokens=heldout_assistant_loss_tokens,
    )
    for key, tolerance in normalized_baseline_tolerances.items():
        if not math.isclose(
            baseline[key],
            pinned_baseline[key],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(f"SFT checkpoint sweep baseline {key} exceeds its pinned tolerance")
    sweep_thresholds = _exact_mapping(
        result["thresholds"],
        _SFT_METRICS_THRESHOLD_KEYS,
        label="SFT checkpoint sweep thresholds",
    )
    normalized_thresholds: dict[str, float] = {}
    for key in sorted(_SFT_METRICS_THRESHOLD_KEYS):
        threshold = _finite_number(
            sweep_thresholds[key],
            label=f"SFT checkpoint sweep thresholds.{key}",
        )
        if threshold < 0.0:
            raise ValueError(f"SFT checkpoint sweep thresholds.{key} must be non-negative")
        normalized_thresholds[key] = threshold

    checkpoints = result["checkpoints"]
    if (
        not isinstance(checkpoints, Sequence)
        or isinstance(checkpoints, (str, bytes))
        or not checkpoints
    ):
        raise ValueError("SFT checkpoint sweep checkpoints must be non-empty")
    artifact_hashes: set[str] = set()
    artifact_paths: set[str] = set()
    eligible_count = 0
    selected_record: dict[str, Any] | None = None
    for index, value in enumerate(checkpoints):
        record = _mapping(value, label=f"SFT checkpoint sweep checkpoints[{index}]")
        artifact = _validate_sweep_checkpoint_artifact(
            record.get("artifact"),
            label=f"SFT checkpoint sweep checkpoints[{index}].artifact",
        )
        if artifact["sha256"] in artifact_hashes or artifact["path"] in artifact_paths:
            raise ValueError("SFT checkpoint sweep contains duplicate checkpoint identities")
        artifact_hashes.add(artifact["sha256"])
        artifact_paths.add(artifact["path"])
        retention_eligible = record.get("retention_eligible")
        if not isinstance(retention_eligible, bool):
            raise ValueError(
                f"SFT checkpoint sweep checkpoints[{index}].retention_eligible must be boolean"
            )
        eligible_count += int(retention_eligible)
        if artifact["sha256"] == selected_checkpoint_sha256:
            if selected_record is not None:
                raise ValueError("SFT checkpoint sweep selected checkpoint is ambiguous")
            selected_record = {**record, "artifact": artifact}
    if selected_record is None:
        raise ValueError("SFT checkpoint sweep does not contain the selected checkpoint")

    artifact = selected_record["artifact"]
    observed_artifact = _hash_regular(
        Path(artifact["path"]),
        label="selected SFT sweep checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if observed_artifact != artifact:
        raise ValueError("selected SFT sweep checkpoint path/bytes/SHA-256 changed")
    if (
        artifact["path"] != scorecard["checkpoint_path"]
        or artifact["bytes"] != scorecard["checkpoint_bytes"]
        or artifact["sha256"] != scorecard["checkpoint_sha256"]
    ):
        raise ValueError("selected SFT sweep checkpoint path/bytes/SHA-256 differ from scorecard")

    post = _validate_teacher_forced_measurement(
        selected_record.get("metrics"),
        label="selected SFT sweep checkpoint metrics",
        expected_rows=assistant_decisions,
        expected_loss_tokens=baseline["assistant_loss_tokens"],
    )
    expected_delta = {
        "mean_loss": post["mean_loss"] - baseline["mean_loss"],
        "assistant_token_accuracy": (
            post["assistant_token_accuracy"] - baseline["assistant_token_accuracy"]
        ),
        "assistant_sequence_accuracy": (
            post["assistant_sequence_accuracy"] - baseline["assistant_sequence_accuracy"]
        ),
    }
    delta = _mapping(
        selected_record.get("delta_from_baseline"),
        label="selected SFT sweep checkpoint delta",
    )
    for key, expected in expected_delta.items():
        observed = _finite_number(
            delta.get(key),
            label=f"selected SFT sweep checkpoint delta.{key}",
        )
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"selected SFT sweep checkpoint delta.{key} is inconsistent")

    gates = _mapping(
        selected_record.get("gates"),
        label="selected SFT sweep checkpoint gates",
    )
    expected_gate_keys = {
        "mean_loss_non_inferiority",
        "assistant_token_accuracy_non_inferiority",
        "assistant_sequence_accuracy_non_inferiority",
    }
    if set(gates) != expected_gate_keys:
        raise ValueError("selected SFT sweep checkpoint has unexpected eligibility gates")
    gate_passes = [
        _validate_sweep_gate(
            gates["mean_loss_non_inferiority"],
            label="selected SFT sweep mean-loss gate",
            observed_key="observed_increase",
            expected_observed=expected_delta["mean_loss"],
            maximum_key="maximum_increase",
            expected_maximum=normalized_thresholds["max_mean_loss_increase"],
        ),
        _validate_sweep_gate(
            gates["assistant_token_accuracy_non_inferiority"],
            label="selected SFT sweep token-accuracy gate",
            observed_key="observed_drop",
            expected_observed=-expected_delta["assistant_token_accuracy"],
            maximum_key="maximum_drop",
            expected_maximum=normalized_thresholds["max_assistant_token_accuracy_drop"],
        ),
        _validate_sweep_gate(
            gates["assistant_sequence_accuracy_non_inferiority"],
            label="selected SFT sweep sequence-accuracy gate",
            observed_key="observed_drop",
            expected_observed=-expected_delta["assistant_sequence_accuracy"],
            maximum_key="maximum_drop",
            expected_maximum=normalized_thresholds["max_assistant_sequence_accuracy_drop"],
        ),
    ]
    retention_eligible = selected_record["retention_eligible"]
    if retention_eligible is not all(gate_passes) or not retention_eligible:
        raise ValueError("selected SFT sweep checkpoint is not retention eligible")

    selection_contract = _mapping(
        result["selection_contract"],
        label="SFT checkpoint sweep selection contract",
    )
    if selection_contract.get(
        "eligible_filter"
    ) != "all_non_inferiority_gates_pass" or selection_contract.get("ranking") != [
        "assistant_sequence_accuracy_desc",
        "assistant_token_accuracy_desc",
        "mean_loss_asc",
        "completed_steps_asc",
        "checkpoint_sha256_desc",
    ]:
        raise ValueError("SFT checkpoint sweep selection contract is unsupported")
    summary = _mapping(result["summary"], label="SFT checkpoint sweep summary")
    if (
        summary.get("evaluated_checkpoints") != len(checkpoints)
        or summary.get("retention_eligible_checkpoints") != eligible_count
        or summary.get("failed_checkpoints") != len(checkpoints) - eligible_count
        or summary.get("status")
        != (
            "retention_eligible_checkpoint_found"
            if eligible_count
            else "no_retention_eligible_checkpoint"
        )
    ):
        raise ValueError("SFT checkpoint sweep summary accounting is inconsistent")
    best = _mapping(
        summary.get("best_retention_eligible_checkpoint"),
        label="SFT checkpoint sweep best retention-eligible checkpoint",
    )
    best_artifact = _validate_sweep_checkpoint_artifact(
        best.get("artifact"),
        label="SFT checkpoint sweep best checkpoint artifact",
    )
    if (
        best_artifact != artifact
        or best.get("checkpoint_step") != selected_record.get("checkpoint_step")
        or best.get("completed_steps") != selected_record.get("completed_steps")
        or best.get("metrics") != selected_record.get("metrics")
    ):
        raise ValueError("scorecard checkpoint is not the sweep-selected best checkpoint")

    return {
        "checkpoint": artifact,
        "source_eval_semantic_set_sha256": source_semantic_set_sha256,
        "selected_eval_semantic_set_sha256": selected_semantic_set_sha256,
        "pre": baseline,
        "post": post,
        "delta": expected_delta,
        "mean_loss_increase": expected_delta["mean_loss"],
        "assistant_token_accuracy_drop": -expected_delta["assistant_token_accuracy"],
        "assistant_sequence_accuracy_drop": -expected_delta["assistant_sequence_accuracy"],
        "retention_eligible": True,
        "sweep_thresholds": normalized_thresholds,
    }


def _gate(
    identifier: str,
    group: str,
    observed: Any,
    comparator: str,
    threshold: Any,
    *,
    description: str,
) -> dict[str, Any]:
    if observed is None and comparator in {">=", "<="}:
        passed = False
    elif comparator == ">=":
        passed = observed >= threshold
    elif comparator == "<=":
        passed = observed <= threshold
    elif comparator == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unsupported readiness comparator: {comparator}")
    return {
        "id": identifier,
        "group": group,
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
        "description": description,
    }


def _build_v1_gates(
    scorecard: Mapping[str, Any],
    preflight: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    score_thresholds = thresholds["scorecard"]
    preflight_thresholds = thresholds["rl_preflight"]
    score_completion = scorecard["generation_completion"]
    score_truncation = scorecard["generation_truncation"]
    score_format = scorecard["complete_format_validity"]
    score_schema = scorecard["schema_validity_on_tool_attempts"]
    score_exact = scorecard["action_exact"]
    sampled_syntax = preflight["tool_syntax"]
    sampled_parser = preflight["complete_parser_validity"]
    sampled_schema = preflight["schema_valid_tool_actions"]
    sampled_exact = preflight["exact_success"]
    sampled_truncation = preflight["truncation"]

    return [
        _gate(
            "preflight_status",
            "evidence",
            preflight["status"],
            "==",
            "passed",
            description="The isolated producer preflight must pass its own validations.",
        ),
        _gate(
            "source_artifacts_untouched",
            "evidence",
            preflight["source_artifacts_untouched"],
            "==",
            True,
            description="The isolated run must not mutate its source artifacts.",
        ),
        _gate(
            "production_output_untouched",
            "evidence",
            preflight["production_output_untouched"],
            "==",
            True,
            description="The preflight must not write into the production RL lineage.",
        ),
        _gate(
            "scorecard_sample_size",
            "greedy_policy_support",
            scorecard["assistant_decisions"],
            ">=",
            score_thresholds["min_assistant_decisions"],
            description="The deterministic held-out scorecard must have enough decisions.",
        ),
        _gate(
            "scorecard_generation_completion",
            "greedy_policy_support",
            score_completion["rate"],
            ">=",
            score_thresholds["min_generation_completion_rate"],
            description="Greedy outputs must usually terminate with EOS.",
        ),
        _gate(
            "scorecard_generation_truncation",
            "greedy_policy_support",
            score_truncation["rate"],
            "<=",
            score_thresholds["max_generation_truncation_rate"],
            description="Length-capped greedy outputs must not dominate evaluation.",
        ),
        _gate(
            "scorecard_complete_format",
            "greedy_policy_support",
            score_format["rate"],
            ">=",
            score_thresholds["min_complete_format_rate"],
            description="Completed greedy outputs must satisfy the strict parser envelope.",
        ),
        _gate(
            "scorecard_schema_validity",
            "greedy_policy_support",
            score_schema["rate"],
            ">=",
            score_thresholds["min_schema_valid_attempt_rate"],
            description="Tool attempts must resolve to known tools with schema-valid arguments.",
        ),
        _gate(
            "scorecard_action_exact",
            "greedy_policy_support",
            score_exact["count"],
            ">=",
            score_thresholds["min_action_exact_successes"],
            description="Held-out greedy decoding must show non-zero exact action support.",
        ),
        _gate(
            "sampled_groups",
            "sampled_policy_support",
            preflight["attempted_groups"],
            ">=",
            preflight_thresholds["min_attempted_groups"],
            description="The preflight must exercise enough independent rollout groups.",
        ),
        _gate(
            "sampled_rollouts",
            "sampled_policy_support",
            preflight["attempted_rollouts"],
            ">=",
            preflight_thresholds["min_attempted_rollouts"],
            description="The preflight must exercise enough sampled continuations.",
        ),
        _gate(
            "sampled_tool_syntax",
            "sampled_policy_support",
            sampled_syntax["rate"],
            ">=",
            preflight_thresholds["min_tool_syntax_rate"],
            description="Tool markers show syntax attempts only; they do not imply validity.",
        ),
        _gate(
            "sampled_complete_parser_validity",
            "sampled_policy_support",
            sampled_parser["rate"],
            ">=",
            preflight_thresholds["min_complete_parser_valid_rate"],
            description=(
                "Parser-valid and non-truncated outputs may still be no-tool text or schema-invalid."
            ),
        ),
        _gate(
            "sampled_schema_valid_tool_actions",
            "learnable_reward_signal",
            sampled_schema["rate"],
            ">=",
            preflight_thresholds["min_schema_valid_tool_rate"],
            description=(
                "Strict tool envelopes must name registered tools and recursively satisfy schemas."
            ),
        ),
        _gate(
            "sampled_exact_success",
            "learnable_reward_signal",
            sampled_exact["count"],
            ">=",
            preflight_thresholds["min_exact_successes"],
            description="At least one sampled rollout must reach the exact canonical reward.",
        ),
        _gate(
            "sampled_truncation",
            "learnable_reward_signal",
            sampled_truncation["rate"],
            "<=",
            preflight_thresholds["max_truncation_rate"],
            description="Length caps must not supply the apparent reward contrast.",
        ),
        _gate(
            "reward_diversity",
            "learnable_reward_signal",
            preflight["unique_reward_values"],
            ">=",
            preflight_thresholds["min_reward_unique_values"],
            description="Globally constant rewards cannot produce a GRPO advantage.",
        ),
        _gate(
            "informative_groups",
            "learnable_reward_signal",
            preflight["informative_groups"],
            ">=",
            preflight_thresholds["min_informative_groups"],
            description="Reward diversity must occur within at least one rollout group.",
        ),
        _gate(
            "informative_group_rate",
            "learnable_reward_signal",
            preflight["informative_group_rate"],
            ">=",
            preflight_thresholds["min_informative_group_rate"],
            description="A minimum fraction of groups must produce non-zero advantages.",
        ),
        _gate(
            "realized_optimizer_updates",
            "learnable_reward_signal",
            preflight["realized_optimizer_updates"],
            ">=",
            preflight_thresholds["min_realized_optimizer_updates"],
            description="The isolated policy optimizer must actually step.",
        ),
        _gate(
            "configured_policy_epochs_realized",
            "learnable_reward_signal",
            preflight["realized_optimizer_updates"],
            ">=",
            preflight["configured_policy_epochs"],
            description="One informative preflight batch must exercise every configured epoch.",
        ),
    ]


def _build_v2_gates(
    scorecard: Mapping[str, Any],
    preflight: Mapping[str, Any],
    sft_metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    score_thresholds = thresholds["scorecard"]
    legacy_thresholds = {
        "scorecard": {
            "min_assistant_decisions": score_thresholds["min_assistant_decisions"],
            "min_generation_completion_rate": score_thresholds["min_generation_completion_rate"],
            "max_generation_truncation_rate": score_thresholds["max_generation_truncation_rate"],
            "min_complete_format_rate": score_thresholds["min_complete_format_rate"],
            "min_schema_valid_attempt_rate": 0.0,
            "min_action_exact_successes": 0,
        },
        "rl_preflight": thresholds["rl_preflight"],
    }
    common_gates = [
        gate
        for gate in _build_v1_gates(scorecard, preflight, legacy_thresholds)
        if gate["group"] != "greedy_policy_support"
    ]
    evidence_gates = [gate for gate in common_gates if gate["group"] == "evidence"]
    sampled_gates = [gate for gate in common_gates if gate["group"] != "evidence"]

    score_completion = scorecard["generation_completion"]
    score_truncation = scorecard["generation_truncation"]
    score_format = scorecard["complete_format_validity"]
    score_tool_format = scorecard["tool_format_validity_on_tool_decisions"]
    score_schema = scorecard["schema_validity_on_tool_decisions"]
    score_tool_name = scorecard["tool_name_case_exact"]
    score_whole_call = scorecard["whole_call_exact"]
    score_abstention = scorecard["abstention"]
    sft_thresholds = thresholds["sft_metrics"]

    greedy_gates = [
        _gate(
            "scorecard_sample_size",
            "greedy_policy_support",
            scorecard["assistant_decisions"],
            ">=",
            score_thresholds["min_assistant_decisions"],
            description="The deterministic held-out scorecard must have enough decisions.",
        ),
        _gate(
            "scorecard_generation_completion",
            "greedy_policy_support",
            score_completion["rate"],
            ">=",
            score_thresholds["min_generation_completion_rate"],
            description="Greedy outputs must usually terminate with EOS.",
        ),
        _gate(
            "scorecard_generation_truncation",
            "greedy_policy_support",
            score_truncation["rate"],
            "<=",
            score_thresholds["max_generation_truncation_rate"],
            description="Length-capped greedy outputs must not dominate evaluation.",
        ),
        _gate(
            "scorecard_complete_format_successes",
            "greedy_policy_support",
            score_format["count"],
            ">=",
            score_thresholds["min_complete_format_successes"],
            description="Overall held-out decoding must produce complete parser-valid outputs.",
        ),
        _gate(
            "scorecard_complete_format_rate",
            "greedy_policy_support",
            score_format["rate"],
            ">=",
            score_thresholds["min_complete_format_rate"],
            description="Overall held-out decoding must sustain complete parser validity.",
        ),
        _gate(
            "scorecard_tool_format_successes",
            "greedy_tool_policy_support",
            score_tool_format["count"],
            ">=",
            score_thresholds["min_tool_format_successes"],
            description=(
                "Expected-tool decisions must produce complete strict tool envelopes; "
                "no-tool abstentions cannot satisfy this gate."
            ),
        ),
        _gate(
            "scorecard_tool_format_rate",
            "greedy_tool_policy_support",
            score_tool_format["rate"],
            ">=",
            score_thresholds["min_tool_format_rate"],
            description="Strict tool-format validity is conditioned on expected-tool decisions.",
        ),
        _gate(
            "scorecard_schema_valid_tool_successes",
            "greedy_tool_policy_support",
            score_schema["count"],
            ">=",
            score_thresholds["min_schema_valid_tool_successes"],
            description="Expected-tool decisions must reach registered, schema-valid tool calls.",
        ),
        _gate(
            "scorecard_schema_valid_tool_rate",
            "greedy_tool_policy_support",
            score_schema["rate"],
            ">=",
            score_thresholds["min_schema_valid_tool_rate"],
            description="Schema validity is conditioned on all expected-tool decisions.",
        ),
        _gate(
            "scorecard_tool_name_case_exact_successes",
            "greedy_tool_policy_support",
            score_tool_name["count"],
            ">=",
            score_thresholds["min_tool_name_case_exact_successes"],
            description="Expected-tool decisions must exactly preserve registered tool names.",
        ),
        _gate(
            "scorecard_tool_name_case_exact_rate",
            "greedy_tool_policy_support",
            score_tool_name["rate"],
            ">=",
            score_thresholds["min_tool_name_case_exact_rate"],
            description="Exact case-sensitive tool-name accuracy uses the expected-tool denominator.",
        ),
        _gate(
            "scorecard_whole_call_exact_successes",
            "greedy_tool_policy_support",
            score_whole_call["count"],
            ">=",
            score_thresholds["min_whole_call_exact_successes"],
            description="At least the configured number of expected-tool actions must be exact.",
        ),
        _gate(
            "scorecard_whole_call_exact_rate",
            "greedy_tool_policy_support",
            score_whole_call["rate"],
            ">=",
            score_thresholds["min_whole_call_exact_rate"],
            description=(
                "Whole-call exactness is conditioned on expected-tool decisions and cannot "
                "be supplied by abstentions."
            ),
        ),
        _gate(
            "scorecard_abstention_successes",
            "greedy_abstention_support",
            score_abstention["count"],
            ">=",
            score_thresholds["min_abstention_successes"],
            description="Expected no-tool decisions must demonstrate structural abstention.",
        ),
        _gate(
            "scorecard_abstention_rate",
            "greedy_abstention_support",
            score_abstention["rate"],
            ">=",
            score_thresholds["min_abstention_rate"],
            description="Abstention accuracy is conditioned on expected no-tool decisions.",
        ),
    ]
    teacher_forced_gates = [
        _gate(
            "sft_mean_loss_non_inferiority",
            "teacher_forced_non_inferiority",
            sft_metrics["mean_loss_increase"],
            "<=",
            sft_thresholds["max_mean_loss_increase"],
            description="SFT must not regress held-out teacher-forced mean loss beyond tolerance.",
        ),
        _gate(
            "sft_token_accuracy_non_inferiority",
            "teacher_forced_non_inferiority",
            sft_metrics["assistant_token_accuracy_drop"],
            "<=",
            sft_thresholds["max_assistant_token_accuracy_drop"],
            description=(
                "SFT must not regress held-out assistant-token accuracy beyond tolerance."
            ),
        ),
        _gate(
            "sft_sequence_accuracy_non_inferiority",
            "teacher_forced_non_inferiority",
            sft_metrics["assistant_sequence_accuracy_drop"],
            "<=",
            sft_thresholds["max_assistant_sequence_accuracy_drop"],
            description=(
                "SFT must not regress held-out assistant-sequence accuracy beyond tolerance."
            ),
        ),
    ]
    transition = preflight["policy_transition"]
    update_gates = [
        _gate(
            "production_learning_rate_sequence",
            "learnable_reward_signal",
            transition["actual_learning_rates_match_expected"],
            "==",
            True,
            description=(
                "The isolated prefix must apply the exact learning-rate sequence from the "
                "full production schedule."
            ),
        ),
        _gate(
            "nonzero_learning_rate_executed",
            "learnable_reward_signal",
            transition["nonzero_learning_rate_executed"],
            "==",
            True,
            description="The isolated production-schedule prefix must reach a nonzero LR.",
        ),
        _gate(
            "final_optimizer_learning_rate",
            "learnable_reward_signal",
            transition["final_optimizer_learning_rate_matches_expected"],
            "==",
            True,
            description=(
                "The optimizer must retain the final learning rate from the approved prefix."
            ),
        ),
        _gate(
            "policy_tensor_transition",
            "learnable_reward_signal",
            transition["at_least_one_policy_tensor_changed"],
            "==",
            True,
            description=(
                "At least one exact named policy tensor must differ from the immutable parent."
            ),
        ),
    ]
    return (
        evidence_gates
        + greedy_gates
        + teacher_forced_gates
        + sampled_gates
        + update_gates
    )


def _summarize_rl_readiness(
    config_path: str | Path,
    *,
    allow_historical_v1: bool,
) -> dict[str, Any]:
    """Build a readiness summary, optionally allowing exact historical v1 reproduction."""

    schema_version, config, config_artifact, thresholds = _load_config(config_path)
    if schema_version == LEGACY_SCHEMA_VERSION and not allow_historical_v1:
        raise ValueError(
            "schema-v1 RL readiness is historical verification-only; "
            "production summarization requires schema_version 2"
        )
    if schema_version != LEGACY_SCHEMA_VERSION and allow_historical_v1:
        raise ValueError("historical RL-readiness reproduction requires schema_version 1")
    production_binding: dict[str, Any] | None = None
    production_config: dict[str, Any] | None = None
    if schema_version == SCHEMA_VERSION:
        production_binding, production_config = _load_production_binding(config)
    evidence = _mapping(config["evidence"], label="config.evidence")
    scorecard_spec = _mapping(evidence["scorecard"], label="config.evidence.scorecard")
    preflight_spec = _mapping(evidence["rl_preflight"], label="config.evidence.rl_preflight")

    scorecard_payload, scorecard_artifact = _load_json_artifact(
        scorecard_spec,
        label="internal scorecard result",
    )
    scorecard_self_sha256 = _assert_self_hash(
        scorecard_payload,
        field="result_self_sha256",
        expected=scorecard_spec["expected_self_sha256"],
        trailing_lf=True,
        label="internal scorecard result",
    )
    scorecard = _validate_scorecard(
        scorecard_payload,
        readiness_schema_version=schema_version,
    )

    preflight_payload, preflight_artifact = _load_json_artifact(
        preflight_spec,
        label="RL preflight receipt",
    )
    preflight_self_sha256 = _assert_self_hash(
        preflight_payload,
        field="receipt_self_sha256",
        expected=preflight_spec["expected_self_sha256"],
        trailing_lf=False,
        label="RL preflight receipt",
    )
    preflight = _validate_preflight(
        preflight_payload,
        readiness_schema_version=schema_version,
    )

    if scorecard["checkpoint_sha256"] != preflight["parent_checkpoint_sha256"]:
        raise ValueError("scorecard and RL preflight do not reference the same SFT checkpoint")
    if scorecard["eval_jsonl_sha256"] not in preflight["eval_jsonl_sha256"]:
        raise ValueError("scorecard and RL preflight do not reference the same eval JSONL")
    if scorecard["eval_semantic_set_sha256"] != preflight["eval_semantic_set_sha256"]:
        raise ValueError("scorecard and RL preflight do not share the same eval semantic set")
    if schema_version == SCHEMA_VERSION:
        assert production_binding is not None and production_config is not None
        production_config_artifact = production_binding["rl_config"]
        production_parent_artifact = production_binding["parent_checkpoint"]
        production_execution = production_binding["execution"]
        if (
            production_config_artifact["canonical_sha256"]
            != preflight["source_config_canonical_sha256"]
            or not _same_resolved_path(
                production_config_artifact["reference"],
                preflight["source_config_path"],
            )
        ):
            raise ValueError(
                "approved production RL config does not match the preflight source config"
            )
        if (
            production_parent_artifact["sha256"]
            != scorecard["checkpoint_sha256"]
            or production_parent_artifact["sha256"]
            != preflight["parent_checkpoint_sha256"]
        ):
            raise ValueError(
                "approved production parent checkpoint does not match evaluation evidence"
            )
        for evidence_path in (
            scorecard["checkpoint_path"],
            preflight["parent_checkpoint_path"],
        ):
            if not _same_resolved_path(
                production_parent_artifact["reference"],
                evidence_path,
            ):
                raise ValueError(
                    "approved production parent path does not match evaluation evidence"
                )
        preflight_execution = preflight["execution"]
        expected_preflight_execution = {
            "requested_device": production_execution["preflight_requested_device"],
            "resolved_device": production_execution["resolved_device"],
            "requested_dtype": production_execution["preflight_requested_dtype"],
            "resolved_dtype": production_execution["resolved_dtype"],
        }
        if preflight_execution != expected_preflight_execution:
            raise ValueError(
                "RL preflight execution identity does not match the approved binding"
            )
        production_schedule_prefix = _production_rl_schedule_prefix(
            production_config
        )
        observed_schedule_prefix = {
            key: preflight["policy_transition"][key]
            for key in (
                "production_schedule_total_steps",
                "execution_rollout_step_limit",
                "first_nonzero_learning_rate_step",
                "expected_learning_rates",
            )
        }
        if observed_schedule_prefix != production_schedule_prefix:
            raise ValueError(
                "RL preflight learning-rate prefix does not match the production config"
            )

    sft_metrics: dict[str, Any] | None = None
    sft_evidence_artifact: dict[str, Any] | None = None
    sft_evidence_kind: str | None = None
    sft_sweep_self_sha256: str | None = None
    if schema_version == SCHEMA_VERSION:
        if (
            scorecard["selected_eval_semantic_set_sha256"]
            != preflight["selected_eval_semantic_set_sha256"]
        ):
            raise ValueError(
                "scorecard and RL preflight do not share the same selected eval subset"
            )
        if "sft_metrics" in evidence:
            sft_metrics_spec = _mapping(
                evidence["sft_metrics"],
                label="config.evidence.sft_metrics",
            )
            sft_metrics_payload, sft_evidence_artifact = _load_json_artifact(
                sft_metrics_spec,
                label="SFT metrics evidence",
            )
            if sft_evidence_artifact["sha256"] != sft_metrics_spec["expected_sha256"]:
                raise ValueError(
                    "SFT metrics evidence SHA-256 does not match the configured identity"
                )
            sft_metrics = _validate_sft_metrics(
                sft_metrics_payload,
                scorecard=scorecard,
            )
            sft_evidence_kind = "training_metrics"
        else:
            sweep_spec = _mapping(
                evidence["sft_checkpoint_sweep"],
                label="config.evidence.sft_checkpoint_sweep",
            )
            sweep_payload, sft_evidence_artifact = _load_json_artifact(
                sweep_spec,
                label="SFT checkpoint sweep evidence",
            )
            sft_sweep_self_sha256 = _assert_self_hash(
                sweep_payload,
                field="result_sha256",
                expected=sweep_spec["expected_self_sha256"],
                trailing_lf=False,
                label="SFT checkpoint sweep result",
            )
            sft_metrics = _validate_sft_checkpoint_sweep(
                sweep_payload,
                scorecard=scorecard,
                selected_checkpoint_sha256=sweep_spec["selected_checkpoint_sha256"],
            )
            for key, observed in sft_metrics["sweep_thresholds"].items():
                configured = _finite_number(
                    thresholds["sft_metrics"][key],
                    label=f"config.thresholds.sft_metrics.{key}",
                )
                if not math.isclose(observed, configured, rel_tol=0.0, abs_tol=0.0):
                    raise ValueError(
                        "readiness and SFT checkpoint sweep non-inferiority "
                        f"thresholds disagree for {key}"
                    )
            sft_evidence_kind = "checkpoint_sweep"
        if (
            sft_metrics["selected_eval_semantic_set_sha256"]
            != preflight["selected_eval_semantic_set_sha256"]
        ):
            raise ValueError(
                "SFT metrics and RL preflight do not share the same selected eval subset"
            )
        gates = _build_v2_gates(scorecard, preflight, sft_metrics, thresholds)
    else:
        gates = _build_v1_gates(scorecard, preflight, thresholds)
    failed_gate_ids = [gate["id"] for gate in gates if not gate["passed"]]
    learnable_gates = [gate for gate in gates if gate["group"] == "learnable_reward_signal"]
    learnable_signal = all(gate["passed"] for gate in learnable_gates)
    promotion_allowed = not failed_gate_ids
    decision = {
        "status": "ready_for_production_rl" if promotion_allowed else "not_ready_for_rl",
        "promotion_allowed": promotion_allowed,
        "learnable_signal_observed": learnable_signal,
        "recommended_action": (
            "promote_to_production_rl"
            if promotion_allowed
            else "hold_rl_and_continue_format_action_supervision"
        ),
        "failed_gate_ids": failed_gate_ids,
    }

    contract: dict[str, Any] = {
        "decision_rule": "promotion_allowed iff every recorded gate passes",
        "fail_closed": True,
        "runs_generation": False,
        "runs_training": False,
        "scorecard_name": _INTERNAL_BENCHMARK_NAME,
        "official_bfcl": False,
        "external_native_benchmark": False,
        "conversation_prompt_contract": _PROMPT_CONTRACT,
        "schema_valid_tool_action": (
            "strict parser-valid tool envelope + registered tool name + recursive "
            "argument-schema match"
        ),
        "informative_group": (
            "one rollout group with non-constant rewards and therefore non-zero "
            "within-group advantages"
        ),
    }
    if schema_version == SCHEMA_VERSION:
        contract["promotion_exactness"] = (
            "expected-tool-conditioned whole_call_exact; action_exact is diagnostic only"
        )
        contract["teacher_forced_non_inferiority"] = (
            "same frozen held-out assistant decisions before and after SFT"
        )
        contract["teacher_forced_evidence_kind"] = sft_evidence_kind
        contract["production_authorization"] = (
            "schema-v2 self-hashed summary + current config/parent/execution revalidation"
        )

    evidence_summary: dict[str, Any] = {
        "config": config_artifact,
        "scorecard": {
            **scorecard_artifact,
            "result_self_sha256": scorecard_self_sha256,
            "checkpoint_sha256": scorecard["checkpoint_sha256"],
            "case_set_sha256": scorecard["case_set_sha256"],
        },
        "rl_preflight": {
            **preflight_artifact,
            "receipt_self_sha256": preflight_self_sha256,
            "parent_checkpoint_sha256": preflight["parent_checkpoint_sha256"],
            "producer_status": preflight["status"],
            "producer_validation_errors": preflight["validation_errors"],
        },
        "pairing": {
            "same_sft_checkpoint": True,
            "same_eval_jsonl": True,
            "same_eval_semantic_set": True,
            "eval_jsonl_sha256": scorecard["eval_jsonl_sha256"],
            "eval_semantic_set_sha256": scorecard["eval_semantic_set_sha256"],
            "heldout_split": "verified_disjoint_eval",
            "train_eval_row_overlap": 0,
            "train_eval_prompt_overlap": 0,
        },
    }
    if schema_version == SCHEMA_VERSION:
        assert (
            sft_metrics is not None
            and sft_evidence_artifact is not None
            and sft_evidence_kind is not None
        )
        if sft_evidence_kind == "training_metrics":
            evidence_summary["sft_metrics"] = {
                **sft_evidence_artifact,
                "checkpoint": sft_metrics["checkpoint"],
            }
        else:
            evidence_summary["sft_checkpoint_sweep"] = {
                **sft_evidence_artifact,
                "result_self_sha256": sft_sweep_self_sha256,
                "selected_checkpoint": sft_metrics["checkpoint"],
                "selected_checkpoint_retention_eligible": sft_metrics["retention_eligible"],
            }
        evidence_summary["pairing"].update(
            {
                "same_selected_eval_semantic_set": True,
                "selected_eval_semantic_set_sha256": scorecard["selected_eval_semantic_set_sha256"],
                "same_teacher_forced_checkpoint": True,
                "teacher_forced_evidence_kind": sft_evidence_kind,
                "same_production_rl_config_as_preflight": True,
                "same_immutable_parent_as_production": True,
                "same_preflight_execution_as_approved": True,
            }
        )

    greedy_heldout: dict[str, Any] = {
        "assistant_decisions": scorecard["assistant_decisions"],
        "tool_decisions": scorecard["tool_decisions"],
        "no_tool_decisions": scorecard["no_tool_decisions"],
        "generation_completion": scorecard["generation_completion"],
        "generation_truncation": scorecard["generation_truncation"],
        "complete_format_validity": scorecard["complete_format_validity"],
        "schema_validity_on_tool_attempts": scorecard["schema_validity_on_tool_attempts"],
        "action_exact": scorecard["action_exact"],
        "finish_reasons": scorecard["finish_reasons"],
        "tool_syntax_presence": {
            "available": False,
            "reason": (
                "the sealed aggregate scorecard retains parser/format counts but not an "
                "aggregate syntax-presence count"
            ),
        },
    }
    if schema_version == SCHEMA_VERSION:
        greedy_heldout.update(
            {
                "tool_format_validity_on_tool_decisions": scorecard[
                    "tool_format_validity_on_tool_decisions"
                ],
                "schema_validity_on_tool_decisions": scorecard["schema_validity_on_tool_decisions"],
                "tool_name_case_exact": scorecard["tool_name_case_exact"],
                "whole_call_exact": scorecard["whole_call_exact"],
                "abstention": scorecard["abstention"],
                "action_exact_promotion_evidence": False,
            }
        )

    funnel: dict[str, Any] = {
        "greedy_heldout": greedy_heldout,
        "sampled_preflight": {
            "attempted_steps": preflight["attempted_steps"],
            "attempted_groups": preflight["attempted_groups"],
            "attempted_rollouts": preflight["attempted_rollouts"],
            "group_size": preflight["group_size"],
            "tool_reward_rollouts": preflight["tool_reward_rollouts"],
            "text_reward_rollouts": preflight["text_reward_rollouts"],
            "tool_syntax_presence": preflight["tool_syntax"],
            "complete_parser_validity": preflight["complete_parser_validity"],
            "schema_valid_tool_actions": preflight["schema_valid_tool_actions"],
            "exact_success": preflight["exact_success"],
            "generation_truncation": preflight["truncation"],
            "generation_eos": preflight["generated_eos"],
        },
        "optimization_signal": {
            "reward_distribution": preflight["reward_distribution"],
            "unique_reward_values": preflight["unique_reward_values"],
            "informative_groups": preflight["informative_groups"],
            "attempted_groups": preflight["attempted_groups"],
            "informative_group_rate": preflight["informative_group_rate"],
            "realized_optimizer_updates": preflight["realized_optimizer_updates"],
            "configured_policy_epochs": preflight["configured_policy_epochs"],
            "global_reward_diversity_is_sufficient": False,
            "within_group_reward_diversity_required": True,
        },
    }
    if schema_version == SCHEMA_VERSION:
        assert sft_metrics is not None
        transition = preflight["policy_transition"]
        funnel["optimization_signal"]["policy_transition"] = transition
        funnel["teacher_forced_sft"] = {
            "evidence_kind": sft_evidence_kind,
            "pre": sft_metrics["pre"],
            "post": sft_metrics["post"],
            "delta": sft_metrics["delta"],
            "mean_loss_increase": sft_metrics["mean_loss_increase"],
            "assistant_token_accuracy_drop": sft_metrics["assistant_token_accuracy_drop"],
            "assistant_sequence_accuracy_drop": sft_metrics["assistant_sequence_accuracy_drop"],
        }

    limitations = [
        (
            "The scorecard is an internal Conversation-schema, BFCL-style benchmark and is "
            "not an official BFCL evaluation."
        ),
        (
            "Parser validity alone does not establish a tool action, registry membership, "
            "argument-schema validity, or exact correctness."
        ),
        (
            "Schema-valid tool rewards are offline canonical rewards; this summary does not "
            "establish environment task success."
        ),
        (
            "A one-step preflight tests whether an optimization signal is mechanically "
            "learnable; it does not predict final RL quality."
        ),
    ]
    if schema_version == SCHEMA_VERSION:
        limitations[3] = (
            "A bounded preflight through the first production-schedule nonzero learning rate "
            "tests whether an optimization signal changes policy tensors; it does not predict "
            "final RL quality."
        )
        limitations.append(
            "Teacher-forced non-inferiority complements but does not replace free-generation "
            "tool-action gates."
        )

    summary_without_hash: dict[str, Any] = {
        "kind": SUMMARY_KIND,
        "schema_version": schema_version,
        "contract": contract,
        "evidence": evidence_summary,
        "thresholds": thresholds,
        "funnel": funnel,
        "gates": gates,
        "decision": decision,
        "limitations": limitations,
    }
    if schema_version == SCHEMA_VERSION:
        assert production_binding is not None
        summary_without_hash["production"] = copy.deepcopy(production_binding)
    return {
        **summary_without_hash,
        "summary_self_sha256": _canonical_sha256(summary_without_hash),
    }


def summarize_rl_readiness(config_path: str | Path) -> dict[str, Any]:
    """Validate schema-v2 evidence and return a production RL promotion decision.

    Schema-v1 is deliberately refused here. Use
    :func:`reproduce_historical_rl_readiness_v1` only to verify a sealed historical result.
    """

    return _summarize_rl_readiness(config_path, allow_historical_v1=False)


def reproduce_historical_rl_readiness_v1(
    config_path: str | Path,
) -> dict[str, Any]:
    """Reproduce a schema-v1 summary exactly without authorizing production promotion."""

    return _summarize_rl_readiness(config_path, allow_historical_v1=True)


def assert_rl_readiness_summary(summary: Mapping[str, Any]) -> None:
    """Validate the summary identity and its fail-closed decision consistency."""

    if (
        summary.get("kind") != SUMMARY_KIND
        or not any(
            _is_schema_version(summary.get("schema_version"), supported)
            for supported in _SUPPORTED_SCHEMA_VERSIONS
        )
    ):
        raise ValueError("unsupported RL-readiness summary")
    recorded = _valid_sha256(
        summary.get("summary_self_sha256"),
        label="RL-readiness summary self-hash",
    )
    unsigned = copy.deepcopy(dict(summary))
    unsigned.pop("summary_self_sha256", None)
    if recorded != _canonical_sha256(unsigned):
        raise ValueError("RL-readiness summary self-hash mismatch")
    schema_version = summary["schema_version"]
    if schema_version == SCHEMA_VERSION:
        production = _exact_mapping(
            summary.get("production"),
            frozenset({"rl_config", "parent_checkpoint", "execution", "out_dir"}),
            label="RL-readiness production binding",
        )
        rl_config = _mapping(
            production["rl_config"],
            label="RL-readiness production RL config",
        )
        parent = _mapping(
            production["parent_checkpoint"],
            label="RL-readiness production parent checkpoint",
        )
        for key in ("path", "reference"):
            if not isinstance(rl_config.get(key), str) or not rl_config[key]:
                raise ValueError(
                    f"RL-readiness production RL config {key} must be non-empty text"
                )
            if not isinstance(parent.get(key), str) or not parent[key]:
                raise ValueError(
                    f"RL-readiness production parent checkpoint {key} must be non-empty text"
                )
        _positive_int(
            rl_config.get("bytes"),
            label="RL-readiness production RL config bytes",
        )
        _valid_sha256(
            rl_config.get("sha256"),
            label="RL-readiness production RL config SHA-256",
        )
        _valid_sha256(
            rl_config.get("canonical_sha256"),
            label="RL-readiness production RL config canonical SHA-256",
        )
        _positive_int(
            parent.get("bytes"),
            label="RL-readiness production parent checkpoint bytes",
        )
        _valid_sha256(
            parent.get("sha256"),
            label="RL-readiness production parent checkpoint SHA-256",
        )
        execution = _exact_mapping(
            production["execution"],
            _PRODUCTION_EXECUTION_KEYS,
            label="RL-readiness production execution",
        )
        for key in sorted(_PRODUCTION_EXECUTION_KEYS):
            if not isinstance(execution[key], str) or not execution[key]:
                raise ValueError(
                    f"RL-readiness production execution.{key} must be non-empty text"
                )
        if not isinstance(production["out_dir"], str) or not production["out_dir"]:
            raise ValueError("RL-readiness production out_dir must be a non-empty path")
    gates = summary.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("RL-readiness summary must contain gates")
    failed: list[str] = []
    learnable_passes: list[bool] = []
    observed_ids: set[str] = set()
    for index, value in enumerate(gates):
        gate = _mapping(value, label=f"RL-readiness gate[{index}]")
        identifier = gate.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in observed_ids:
            raise ValueError("RL-readiness summary contains an invalid or duplicate gate ID")
        observed_ids.add(identifier)
        passed = gate.get("passed")
        if not isinstance(passed, bool):
            raise ValueError(f"RL-readiness gate {identifier!r} has no boolean result")
        group = gate.get("group")
        if not isinstance(group, str) or not group:
            raise ValueError(f"RL-readiness gate {identifier!r} has no group")
        if not passed:
            failed.append(identifier)
        if group == "learnable_reward_signal":
            learnable_passes.append(passed)
    if not learnable_passes:
        raise ValueError("RL-readiness summary has no learnable-signal gates")
    if schema_version == SCHEMA_VERSION:
        required_update_gates = {
            "production_learning_rate_sequence",
            "nonzero_learning_rate_executed",
            "final_optimizer_learning_rate",
            "policy_tensor_transition",
        }
        if not required_update_gates <= observed_ids:
            raise ValueError("schema-v2 RL-readiness summary lacks policy-update gates")
    decision = _mapping(summary.get("decision"), label="RL-readiness decision")
    if decision.get("failed_gate_ids") != failed:
        raise ValueError("RL-readiness failed-gate list is inconsistent")
    expected_allowed = not failed
    if decision.get("promotion_allowed") is not expected_allowed:
        raise ValueError("RL-readiness promotion decision is inconsistent")
    if decision.get("learnable_signal_observed") is not all(learnable_passes):
        raise ValueError("RL-readiness learnable-signal decision is inconsistent")
    expected_status = "ready_for_production_rl" if expected_allowed else "not_ready_for_rl"
    if decision.get("status") != expected_status:
        raise ValueError("RL-readiness status is inconsistent")
    expected_action = (
        "promote_to_production_rl"
        if expected_allowed
        else "hold_rl_and_continue_format_action_supervision"
    )
    if decision.get("recommended_action") != expected_action:
        raise ValueError("RL-readiness recommended action is inconsistent")


def load_rl_readiness_summary(path: str | Path) -> dict[str, Any]:
    """Strictly load and self-verify one sealed RL-readiness summary."""

    raw, _ = _read_regular(
        Path(path),
        label="RL-readiness summary",
        max_bytes=_MAX_EVIDENCE_BYTES,
    )
    payload = _strict_json(raw, label="RL-readiness summary")
    summary = _mapping(payload, label="RL-readiness summary")
    assert_rl_readiness_summary(summary)
    return summary


def run_ready_rl(summary_path: str | Path) -> dict[str, Any]:
    """Revalidate a ready schema-v2 summary, then invoke the guarded RL runner once."""

    summary = load_rl_readiness_summary(summary_path)
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "schema-v1 RL readiness is historical analysis-only and cannot authorize RL"
        )
    decision = _mapping(summary.get("decision"), label="RL-readiness decision")
    if (
        decision.get("promotion_allowed") is not True
        or decision.get("status") != "ready_for_production_rl"
    ):
        raise ValueError("RL-readiness summary does not authorize production RL")

    evidence = _mapping(summary.get("evidence"), label="RL-readiness evidence")
    config_artifact = _mapping(
        evidence.get("config"),
        label="RL-readiness config evidence",
    )
    readiness_config_path = config_artifact.get("path")
    if not isinstance(readiness_config_path, str) or not readiness_config_path:
        raise ValueError("RL-readiness config evidence path must be non-empty text")
    current_summary = summarize_rl_readiness(readiness_config_path)
    assert_rl_readiness_summary(current_summary)
    if current_summary != summary:
        raise ValueError(
            "RL-readiness summary is stale relative to its config or bound evidence"
        )

    production = _mapping(
        summary["production"],
        label="RL-readiness production binding",
    )
    execution = _mapping(
        production["execution"],
        label="RL-readiness production execution",
    )
    requested_device = execution["production_requested_device"]
    requested_dtype = execution["production_requested_dtype"]
    from localagent.train.device import execution_metadata, resolve_device, resolve_dtype

    resolved_device = resolve_device(requested_device)
    resolved_dtype = resolve_dtype(resolved_device, requested_dtype)
    current_execution = execution_metadata(
        requested_device=requested_device,
        resolved_device=resolved_device,
        requested_dtype=requested_dtype,
        resolved_dtype=resolved_dtype,
    )
    guarded_execution = {
        key: current_execution[key]
        for key in (
            "requested_device",
            "resolved_device",
            "requested_dtype",
            "resolved_dtype",
        )
    }
    expected_execution = {
        "requested_device": requested_device,
        "resolved_device": execution["resolved_device"],
        "requested_dtype": requested_dtype,
        "resolved_dtype": execution["resolved_dtype"],
    }
    if guarded_execution != expected_execution:
        raise ValueError(
            "current production RL device/dtype resolution differs from readiness approval"
        )

    out_dir = Path(str(production["out_dir"]))
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(
            f"guarded RL requires an absent production output directory: {out_dir}"
        )

    production_config = _mapping(
        production["rl_config"],
        label="RL-readiness production RL config",
    )
    parent_checkpoint = _mapping(
        production["parent_checkpoint"],
        label="RL-readiness production parent checkpoint",
    )
    from localagent.train.rl import run as run_rl

    run_rl(
        str(production_config["reference"]),
        resume=False,
        _expected_config_canonical_sha256=str(
            production_config["canonical_sha256"]
        ),
        _expected_parent_checkpoint_sha256=str(parent_checkpoint["sha256"]),
        _expected_execution=expected_execution,
        _require_fresh_output_dir=True,
    )
    return {
        "stage": "rl",
        "status": "completed",
        "readiness_summary_sha256": summary["summary_self_sha256"],
        "production_config_canonical_sha256": production_config[
            "canonical_sha256"
        ],
        "parent_checkpoint_sha256": parent_checkpoint["sha256"],
        "execution": expected_execution,
        "out_dir": str(out_dir),
    }


def write_rl_readiness_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    """Write one canonical, deterministic JSON summary."""

    assert_rl_readiness_summary(summary)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(summary, trailing_lf=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary summary: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
