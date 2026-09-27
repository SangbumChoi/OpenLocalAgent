"""Isolated, one-update training preflights with measured resource receipts.

Each supported stage derives a one-step config in a new work directory, disables resume, and
redirects checkpoints away from the production run. Source artifacts and the production output
path are snapshotted before and after execution so a preflight cannot silently become part of the
real training lineage. SFT and RL additionally prove that the isolated checkpoint used the exact
parent, model config, tokenizer, and train/eval bytes declared by the source config.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import platform
import resource
import sys
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.model import ModelConfig
from localagent.train.stage_data import canonical_sha256, file_identity, tokenizer_identity

PREFLIGHT_KIND = "localagent_one_update_training_preflight"
PREFLIGHT_SCHEMA_VERSION = 1
RL_EVAL_COVERAGE_KIND = "localagent_rl_preflight_minimum_eval_coverage"
RL_EVAL_COVERAGE_SCHEMA_VERSION = 1
_SFT_ZERO_EXECUTED_LR_NON_LEARNING_LIMITATION = (
    "all bounded executed learning rates are zero, so this preflight proves frozen-state "
    "and optimizer-boundary integrity but cannot prove an unfrozen learning transition"
)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def build_one_update_pretrain_config(
    source: Mapping[str, Any],
    *,
    work_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Return an isolated one-update derivative without mutating ``source``."""

    effective = copy.deepcopy(dict(source))
    if effective.get("stage") != "pretrain":
        raise ValueError("one-update execution preflight currently supports stage 'pretrain'")

    schedule = _mapping(effective.get("schedule", {}), label="schedule")
    configured_steps = schedule.get("total_steps")
    if (
        isinstance(configured_steps, bool)
        or not isinstance(configured_steps, int)
        or configured_steps < 1
    ):
        raise ValueError("schedule.total_steps must be a positive integer")
    schedule["total_steps"] = 1
    effective["schedule"] = schedule

    batch = _mapping(effective.get("batch", {}), label="batch")
    for key in ("micro_batch_size", "grad_accum_steps"):
        value = batch.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"batch.{key} must be a positive integer")
    effective["batch"] = batch

    runtime = _mapping(effective.get("runtime", {}), label="runtime")
    runtime["resume"] = False
    if device is not None:
        if not isinstance(device, str) or not device:
            raise ValueError("device override must be a non-empty string")
        runtime["device"] = device
    effective["runtime"] = runtime

    isolated_out = Path(work_dir) / "run"
    log = _mapping(effective.get("log", {}), label="log")
    log["out_dir"] = str(isolated_out)
    log["ckpt_every"] = 1
    log["eval_every"] = 0
    log.pop("mirror_dir", None)
    effective["log"] = log
    return effective


def build_one_update_sft_config(
    source: Mapping[str, Any],
    *,
    work_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Return an isolated one-optimizer-update SFT child without mutating ``source``."""

    from localagent.train.replay_sampling import (
        MIXED_REPLAY_SAMPLING_MODE,
        PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    )
    from localagent.train.sft import resolve_sft_continuation

    effective = copy.deepcopy(dict(source))
    if effective.get("stage") != "sft":
        raise ValueError("one-update SFT preflight requires stage 'sft'")

    raw_parent = effective.get("init_from")
    if not isinstance(raw_parent, (str, Path)) or not str(raw_parent):
        raise ValueError("init_from must be a non-empty parent checkpoint path")
    # This preflight is a child of an already-completed SFT horizon, not a midtrain-to-SFT run.
    # Require the explicit mode while leaving its exact mapping and parent path untouched.
    if resolve_sft_continuation(effective) is None:
        raise ValueError(
            "one-update SFT preflight requires continuation.mode="
            "'fresh_optimizer_sft_child_v1'"
        )

    schedule = _mapping(effective.get("schedule", {}), label="schedule")
    configured_steps = schedule.get("total_steps")
    if (
        isinstance(configured_steps, bool)
        or not isinstance(configured_steps, int)
        or configured_steps < 1
    ):
        raise ValueError("schedule.total_steps must be a positive integer")
    data = _mapping(effective.get("data", {}), label="data")
    sampling = data.get("sampling")
    preserve_sampling_horizon = (
        isinstance(sampling, Mapping)
        and sampling.get("mode")
        in {
            MIXED_REPLAY_SAMPLING_MODE,
            PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        }
    )
    if not preserve_sampling_horizon:
        schedule["total_steps"] = 1
    effective["schedule"] = schedule

    batch = _mapping(effective.get("batch", {}), label="batch")
    for key in ("micro_batch_size", "grad_accum_steps"):
        value = batch.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"batch.{key} must be a positive integer")
    effective["batch"] = batch

    # Route-head and dense-selector training each create a separate optimizer after the LM update.
    # A one-update receipt cannot prove an exact global count when either path is enabled.
    heads = _mapping(effective.get("heads", {}), label="heads")
    for key in ("train_route_head", "train_dense_selector"):
        value = heads.get(key, True)
        if not isinstance(value, bool):
            raise TypeError(f"heads.{key} must be boolean")
        if value:
            raise ValueError(
                f"one-update SFT preflight requires heads.{key}=false "
                "to exclude auxiliary optimizer updates"
            )
    effective["heads"] = heads

    runtime = _mapping(effective.get("runtime", {}), label="runtime")
    runtime["resume"] = False
    if device is not None:
        if not isinstance(device, str) or not device:
            raise ValueError("device override must be a non-empty string")
        runtime["device"] = device
    effective["runtime"] = runtime

    isolated_out = Path(work_dir) / "run"
    log = _mapping(effective.get("log", {}), label="log")
    log["out_dir"] = str(isolated_out)
    log["ckpt_every"] = 1
    log.pop("mirror_dir", None)
    effective["log"] = log
    return effective


def build_one_update_rl_config(
    source: Mapping[str, Any],
    *,
    work_dir: str | Path,
    device: str | None = None,
    evaluation_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an isolated RL-prefix derivative without changing production numerics."""

    effective = copy.deepcopy(dict(source))
    if effective.get("stage") != "rl":
        raise ValueError("one-update RL preflight requires stage 'rl'")

    schedule = _mapping(effective.get("schedule", {}), label="schedule")
    configured_steps = schedule.get("total_steps")
    if (
        isinstance(configured_steps, bool)
        or not isinstance(configured_steps, int)
        or configured_steps < 1
    ):
        raise ValueError("schedule.total_steps must be a positive integer")
    effective["schedule"] = schedule

    rollout = _mapping(effective.get("rollout", {}), label="rollout")
    for key in ("prompts_per_step", "group_size", "max_new_tokens"):
        value = rollout.get(key)
        minimum = 2 if key == "group_size" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"rollout.{key} must be an integer >= {minimum}")
    effective["rollout"] = rollout

    policy = _mapping(effective.get("policy", {}), label="policy")
    policy_epochs = policy.get("epochs_per_rollout")
    if isinstance(policy_epochs, bool) or not isinstance(policy_epochs, int) or policy_epochs < 1:
        raise ValueError("policy.epochs_per_rollout must be a positive integer")
    effective["policy"] = policy

    runtime = _mapping(effective.get("runtime", {}), label="runtime")
    runtime["resume"] = False
    if device is not None:
        if not isinstance(device, str) or not device:
            raise ValueError("device override must be a non-empty string")
        runtime["device"] = device
    effective["runtime"] = runtime

    # Held-out generation is deterministic and consumes no training RNG. Keep it in the real
    # runner path, but cap it to the exact selector-derived mandatory-coverage set so the
    # preflight exercises evaluation without silently dropping an observed stratum.
    evaluation = _mapping(effective.get("evaluation", {}), label="evaluation")
    if "preflight_minimum_coverage" in evaluation:
        raise ValueError(
            "evaluation.preflight_minimum_coverage is derived-only and must not "
            "appear in a production config"
        )
    max_eval = evaluation.get("max_conversations")
    if max_eval is not None:
        if isinstance(max_eval, bool) or not isinstance(max_eval, int) or max_eval < 1:
            raise ValueError("evaluation.max_conversations must be a positive integer")
        if not isinstance(evaluation_coverage, Mapping):
            raise ValueError(
                "configured RL evaluation requires a verified minimum-coverage derivation"
            )
        coverage = copy.deepcopy(dict(evaluation_coverage))
        expected_keys = {
            "kind",
            "schema_version",
            "selector",
            "production_max_conversations",
            "minimum_coverage_rows",
            "mandatory_strata",
            "verified_eval_artifacts",
            "selection_audit",
            "derivation_sha256",
        }
        audit = coverage.get("selection_audit")
        capacity = audit.get("capacity") if isinstance(audit, Mapping) else None
        minimum_rows = coverage.get("minimum_coverage_rows")
        recorded_derivation = coverage.pop("derivation_sha256", None)
        audit_core = dict(audit) if isinstance(audit, Mapping) else {}
        recorded_audit_sha256 = audit_core.pop("audit_sha256", None)
        if (
            set(evaluation_coverage) != expected_keys
            or coverage.get("kind") != RL_EVAL_COVERAGE_KIND
            or coverage.get("schema_version") != RL_EVAL_COVERAGE_SCHEMA_VERSION
            or coverage.get("selector") != evaluation.get("selection")
            or coverage.get("production_max_conversations") != max_eval
            or not isinstance(coverage.get("verified_eval_artifacts"), list)
            or not coverage["verified_eval_artifacts"]
            or not isinstance(minimum_rows, int)
            or isinstance(minimum_rows, bool)
            or minimum_rows < 1
            or not isinstance(capacity, Mapping)
            or capacity.get("max_rows") != minimum_rows
            or capacity.get("coverage_rows") != minimum_rows
            or capacity.get("fill_rows") != 0
            or audit.get("algorithm") != coverage.get("selector")
            or audit.get("mandatory_strata") != coverage.get("mandatory_strata")
            or audit.get("selected", {}).get("rows") != minimum_rows
            or not _valid_sha256(recorded_derivation)
            or recorded_derivation != canonical_sha256(coverage)
            or not _valid_sha256(recorded_audit_sha256)
            or recorded_audit_sha256
            != hashlib.sha256(canonical_json_bytes(audit_core)).hexdigest()
        ):
            raise ValueError("RL evaluation minimum-coverage derivation is invalid")
        coverage["derivation_sha256"] = recorded_derivation
        evaluation["max_conversations"] = minimum_rows
        evaluation["preflight_minimum_coverage"] = coverage
    elif evaluation_coverage is not None:
        raise ValueError(
            "RL evaluation coverage derivation requires evaluation.max_conversations"
        )
    effective["evaluation"] = evaluation

    isolated_out = Path(work_dir) / "run"
    log = _mapping(effective.get("log", {}), label="log")
    log["out_dir"] = str(isolated_out)
    log["ckpt_every"] = 1
    log.pop("mirror_dir", None)
    effective["log"] = log
    return effective


def _optional_file_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"preflight checkpoint target must be a regular non-symlink file: {path}")
    return {"exists": True, "path": str(path), **file_identity(path)}


def _path_snapshot(path: Path) -> dict[str, Any]:
    """Snapshot a path without following symlinks, including complete directory contents."""

    if not path.exists() and not path.is_symlink():
        return {"exists": False, "path": str(path)}
    if path.is_symlink():
        return {
            "exists": True,
            "path": str(path),
            "kind": "symlink",
            "target": str(path.readlink()),
        }
    if path.is_file():
        return {
            "exists": True,
            "path": str(path),
            "kind": "file",
            **file_identity(path),
        }
    if not path.is_dir():
        return {"exists": True, "path": str(path), "kind": "unsupported"}

    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda value: value.name):
            relative = str(child.relative_to(path))
            if child.is_symlink():
                entries.append(
                    {"path": relative, "kind": "symlink", "target": str(child.readlink())}
                )
            elif child.is_dir():
                entries.append({"path": relative, "kind": "directory"})
                visit(child)
            elif child.is_file():
                entries.append({"path": relative, "kind": "file", **file_identity(child)})
            else:
                entries.append({"path": relative, "kind": "unsupported"})

    visit(path)
    return {
        "exists": True,
        "path": str(path),
        "kind": "directory",
        "entries": entries,
        "tree_sha256": canonical_sha256(entries),
    }


def _required_regular_file_snapshot(path: Path, *, label: str) -> dict[str, Any]:
    snapshot = _path_snapshot(path)
    if not snapshot["exists"]:
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if snapshot.get("kind") != "file":
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return snapshot


def _source_sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, list) or not value:
        raise TypeError(f"{label} must be a source or non-empty list of sources")
    return value


def derive_rl_eval_minimum_coverage(
    source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Derive the exact deterministic selector coverage from verified eval artifacts."""

    from localagent.data.stratified_eval_selector import (
        ALGORITHM,
        InsufficientStratumCapacityError,
        select_stratified_eval_subset,
    )
    from localagent.train.stage_data import load_conversation_source

    evaluation = _mapping(source.get("evaluation", {}), label="evaluation")
    if "preflight_minimum_coverage" in evaluation:
        raise ValueError(
            "evaluation.preflight_minimum_coverage is derived-only and must not "
            "appear in a production config"
        )
    configured_max = evaluation.get("max_conversations")
    if configured_max is None:
        return None
    if (
        isinstance(configured_max, bool)
        or not isinstance(configured_max, int)
        or configured_max < 1
    ):
        raise ValueError("evaluation.max_conversations must be a positive integer")
    if evaluation.get("selection") != ALGORITHM:
        raise ValueError(
            "RL preflight evaluation.selection must use the production stratified selector"
        )

    data = _mapping(source.get("data", {}), label="data")
    if data.get("strict_conversation_artifacts") is not True:
        raise ValueError(
            "RL preflight minimum coverage requires strict verified conversation artifacts"
        )
    specs = _source_sequence(
        data.get("eval_conversations"),
        label="data.eval_conversations",
    )
    loaded_sources = [
        load_conversation_source(
            spec,
            require_verified=True,
            expected_split="eval",
        )
        for spec in specs
    ]
    if not all(loaded.verified for loaded in loaded_sources):
        raise RuntimeError("RL preflight eval coverage source was not verified")
    conversations = [
        conversation
        for loaded in loaded_sources
        for conversation in loaded.conversations
    ]
    try:
        selection = select_stratified_eval_subset(conversations, max_rows=1)
        minimum_rows = selection.audit.coverage_rows
        mandatory_strata = len(selection.audit.stratum_counts)
    except InsufficientStratumCapacityError as exc:
        minimum_rows = exc.required_rows
        mandatory_strata = exc.mandatory_strata
        selection = select_stratified_eval_subset(
            conversations,
            max_rows=minimum_rows,
        )

    audit = selection.audit.as_dict()
    capacity = audit["capacity"]
    if (
        minimum_rows < 1
        or configured_max < minimum_rows
        or capacity != {
            "max_rows": minimum_rows,
            "coverage_rows": minimum_rows,
            "fill_rows": 0,
        }
        or audit.get("mandatory_strata") != mandatory_strata
        or audit.get("selected", {}).get("rows") != minimum_rows
    ):
        raise ValueError(
            "configured RL evaluation capacity cannot satisfy the deterministic "
            "mandatory-strata coverage contract"
        )

    payload: dict[str, Any] = {
        "kind": RL_EVAL_COVERAGE_KIND,
        "schema_version": RL_EVAL_COVERAGE_SCHEMA_VERSION,
        "selector": ALGORITHM,
        "production_max_conversations": configured_max,
        "minimum_coverage_rows": minimum_rows,
        "mandatory_strata": mandatory_strata,
        "verified_eval_artifacts": [
            {"path": str(loaded.path), **dict(loaded.identity)}
            for loaded in loaded_sources
        ],
        "selection_audit": audit,
    }
    payload["derivation_sha256"] = canonical_sha256(payload)
    return payload


def _declared_rl_data_snapshots(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = _mapping(source.get("data", {}), label="data")
    declared: list[dict[str, Any]] = []
    for role, key in (("train", "conversations"), ("eval", "eval_conversations")):
        for index, spec in enumerate(_source_sequence(data.get(key), label=f"data.{key}")):
            artifact: Any = None
            if isinstance(spec, Mapping):
                raw_path = spec.get("path")
                artifact = spec.get("artifact")
            else:
                raw_path = spec
            if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                raise ValueError(f"data.{key}[{index}] path must be non-empty")
            record: dict[str, Any] = {
                "role": role,
                "index": index,
                "jsonl": _required_regular_file_snapshot(
                    Path(raw_path),
                    label=f"RL {role} conversation JSONL",
                ),
            }
            if artifact is not None:
                artifact_mapping = _mapping(
                    artifact,
                    label=f"data.{key}[{index}].artifact",
                )
                for artifact_key, record_key in (
                    ("manifest", "manifest"),
                    ("generator_config", "generator_config"),
                ):
                    artifact_path = artifact_mapping.get(artifact_key)
                    if not isinstance(artifact_path, (str, Path)) or not str(artifact_path):
                        raise ValueError(
                            f"data.{key}[{index}].artifact.{artifact_key} must be non-empty"
                        )
                    record[record_key] = _required_regular_file_snapshot(
                        Path(artifact_path),
                        label=f"RL {role} conversation {artifact_key}",
                    )
            declared.append(record)
    return declared


def _optional_source_sequence(value: Any, *, label: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a source or list of sources")
    return value


def _declared_sft_data_snapshots(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = _mapping(source.get("data", {}), label="data")
    declared: list[dict[str, Any]] = []
    for role, key, required in (
        ("train", "conversations", True),
        ("decay", "decay_conversations", False),
        ("eval", "eval_conversations", False),
    ):
        specs = _optional_source_sequence(data.get(key), label=f"data.{key}")
        if required and not specs:
            raise ValueError(f"data.{key} must contain at least one source")
        for index, spec in enumerate(specs):
            artifact: Any = None
            if isinstance(spec, Mapping):
                raw_path = spec.get("path")
                artifact = spec.get("artifact")
            else:
                raw_path = spec
            if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                raise ValueError(f"data.{key}[{index}] path must be non-empty")
            record: dict[str, Any] = {
                "role": role,
                "index": index,
                "jsonl": _required_regular_file_snapshot(
                    Path(raw_path),
                    label=f"SFT {role} conversation JSONL",
                ),
            }
            if artifact is not None:
                artifact_mapping = _mapping(
                    artifact,
                    label=f"data.{key}[{index}].artifact",
                )
                for artifact_key, record_key in (
                    ("manifest", "manifest"),
                    ("generator_config", "generator_config"),
                ):
                    artifact_path = artifact_mapping.get(artifact_key)
                    if not isinstance(artifact_path, (str, Path)) or not str(artifact_path):
                        raise ValueError(
                            f"data.{key}[{index}].artifact.{artifact_key} must be non-empty"
                        )
                    record[record_key] = _required_regular_file_snapshot(
                        Path(artifact_path),
                        label=f"SFT {role} conversation {artifact_key}",
                    )
            declared.append(record)
    return declared


def _derive_sft_preflight_execution_contract(
    source: Mapping[str, Any],
    data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the bounded optimizer prefix from the sealed LM sampling contract."""

    from localagent.train.replay_sampling import (
        PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    )

    schedule = _mapping(source.get("schedule", {}), label="schedule")
    planned_updates = schedule.get("total_steps")
    if (
        isinstance(planned_updates, bool)
        or not isinstance(planned_updates, int)
        or planned_updates < 1
    ):
        raise ValueError("schedule.total_steps must be a positive integer")
    batch = _mapping(source.get("batch", {}), label="batch")
    effective_batch = int(batch["micro_batch_size"]) * int(batch["grad_accum_steps"])
    data = _mapping(source.get("data", {}), label="data")
    sampling = data.get("sampling")
    sampling_mode = sampling.get("mode") if isinstance(sampling, Mapping) else None
    first_pulse_update = None
    execution_limit = 1
    executed_through_first_pulse = False
    if sampling_mode == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
        sampling_contract = data_identity.get("decision_sampling")
        if not isinstance(sampling_contract, Mapping):
            raise ValueError(
                "parent-anchored SFT preflight requires a sealed decision-sampling contract"
            )
        update_layout = sampling_contract.get("update_layout")
        if not isinstance(update_layout, Mapping):
            raise ValueError(
                "parent-anchored SFT sampling contract has no update layout"
            )
        pulse_positions = update_layout.get("pulse_positions_zero_based")
        if (
            update_layout.get("pulse_positions_index_base") != 0
            or not isinstance(pulse_positions, (list, tuple))
            or not pulse_positions
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                for position in pulse_positions
            )
            or list(pulse_positions) != sorted(set(pulse_positions))
        ):
            raise ValueError(
                "parent-anchored SFT sampling contract pulse positions are invalid"
            )
        if (
            update_layout.get("total_updates") != planned_updates
            or pulse_positions[-1] >= planned_updates
        ):
            raise ValueError(
                "parent-anchored SFT sampling contract update horizon is invalid"
            )
        first_pulse_update = pulse_positions[0]
        execution_limit = first_pulse_update + 1
        executed_through_first_pulse = True

    return {
        "execution_optimizer_update_limit": execution_limit,
        "first_pulse_update_zero_based": first_pulse_update,
        "executed_through_first_pulse": executed_through_first_pulse,
        "executed_lm_decisions": execution_limit * effective_batch,
    }


def _derive_sft_data_identity_and_sampling(
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Rebuild SFT data identity and any exact mixed-replay prefix evidence."""

    from localagent.data.conversation_artifact import assert_no_conversation_overlap
    from localagent.data.decision_quota_order import QUOTA_SAMPLING_MODE, order_assistant_decisions
    from localagent.data.prompt_contract import (
        LEGACY_CONVERSATION_PROMPT_CONTRACT,
        resolve_conversation_prompt_contract,
    )
    from localagent.data.stratified_eval_selector import (
        ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
    )
    from localagent.data.stratified_eval_selector import select_stratified_eval_subset
    from localagent.train.sft import quota_sampling_window
    from localagent.train.replay_sampling import (
        MIXED_REPLAY_SAMPLING_MODE,
        PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        mixed_replay_sampling_window,
        parent_anchored_format_pulse_sampling_window,
    )
    from localagent.train.stage_data import load_conversation_source

    data = _mapping(source.get("data", {}), label="data")
    prompt_contract = resolve_conversation_prompt_contract(
        data.get("conversation_prompt_contract")
    )
    strict = data.get("strict_conversation_artifacts", False)
    if not isinstance(strict, bool):
        raise TypeError("data.strict_conversation_artifacts must be boolean")

    def load_sources(key: str, *, split: str, required: bool) -> list[Any]:
        specs = _optional_source_sequence(data.get(key), label=f"data.{key}")
        if required and not specs:
            raise ValueError(f"data.{key} must contain at least one source")
        return [
            load_conversation_source(
                spec,
                require_verified=strict,
                expected_split=split,
            )
            for spec in specs
        ]

    train_sources = load_sources("conversations", split="train", required=True)
    decay_sources = load_sources("decay_conversations", split="train", required=False)
    eval_sources = load_sources("eval_conversations", split="eval", required=False)
    train_conversations = [
        conversation for loaded in train_sources for conversation in loaded.conversations
    ]
    decay_conversations = [
        conversation for loaded in decay_sources for conversation in loaded.conversations
    ]
    eval_conversations = [
        conversation for loaded in eval_sources for conversation in loaded.conversations
    ]
    overlap = assert_no_conversation_overlap(
        [*train_conversations, *decay_conversations],
        eval_conversations,
        left_label="SFT main/decay training content",
        right_label="held-out",
        conversation_prompt_contract=prompt_contract,
    )

    evaluation = _mapping(source.get("evaluation", {}), label="evaluation")
    max_eval = evaluation.get("max_conversations")
    selection_mode = evaluation.get("selection")
    selection_audit = None
    if max_eval is None:
        if selection_mode is not None:
            raise ValueError("evaluation.selection requires evaluation.max_conversations")
    else:
        if selection_mode != STRATIFIED_EVAL_ALGORITHM:
            raise ValueError(
                "evaluation.selection must use the production stratified selector"
            )
        selection_audit = select_stratified_eval_subset(
            eval_conversations,
            max_rows=max_eval,
        ).audit.as_dict()

    identity: dict[str, Any] = {
        "conversations": [dict(loaded.identity) for loaded in train_sources],
        "eval_conversations": [dict(loaded.identity) for loaded in eval_sources],
        "decay_conversations": [dict(loaded.identity) for loaded in decay_sources],
        "conversation_overlap_audit": overlap.as_dict(),
        **({"eval_selection": selection_audit} if selection_audit is not None else {}),
    }
    if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT:
        identity["conversation_prompt_contract"] = prompt_contract

    sampling_evidence = None
    sampling = data.get("sampling")
    if sampling is not None:
        sampling = _mapping(sampling, label="data.sampling")
        sampling_mode = sampling.get("mode")
        if sampling_mode not in {
            QUOTA_SAMPLING_MODE,
            MIXED_REPLAY_SAMPLING_MODE,
            PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        }:
            raise ValueError(
                "data.sampling.mode must be one of "
                f"{QUOTA_SAMPLING_MODE!r}, {MIXED_REPLAY_SAMPLING_MODE!r}, "
                f"{PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE!r}"
            )
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            raise ValueError("decision sampling requires openai_full_catalog_v1")
        if bool(data.get("shuffle", True)):
            raise ValueError("decision sampling requires data.shuffle=false")
        if decay_sources:
            raise ValueError("decision sampling does not support decay_conversations")
        schedule = _mapping(source.get("schedule", {}), label="schedule")
        batch = _mapping(source.get("batch", {}), label="batch")
        selected_decisions = (
            int(schedule["total_steps"])
            * int(batch["micro_batch_size"])
            * int(batch["grad_accum_steps"])
        )
        if sampling_mode == QUOTA_SAMPLING_MODE:
            ordering = order_assistant_decisions(train_conversations)
            ordered_decision_keys, sampling_contract = quota_sampling_window(
                ordering,
                selected_decisions=selected_decisions,
                start_decision=sampling.get("start_decision", 0),
            )
        elif sampling_mode == MIXED_REPLAY_SAMPLING_MODE:
            ordered_decision_keys, sampling_contract = mixed_replay_sampling_window(
                [loaded.conversations for loaded in train_sources],
                selected_decisions=selected_decisions,
                sampling_config=sampling,
            )
            effective_batch = int(batch["micro_batch_size"]) * int(
                batch["grad_accum_steps"]
            )
            if sampling_contract["cycle"]["length"] != effective_batch:
                raise ValueError(
                    "mixed replay cycle must equal one complete optimizer update: "
                    f"cycle={sampling_contract['cycle']['length']}, "
                    f"effective_batch={effective_batch}"
                )
        else:
            ordered_decision_keys, sampling_contract = (
                parent_anchored_format_pulse_sampling_window(
                    [loaded.conversations for loaded in train_sources],
                    selected_decisions=selected_decisions,
                    sampling_config=sampling,
                )
            )
            effective_batch = int(batch["micro_batch_size"]) * int(
                batch["grad_accum_steps"]
            )
            update_decisions = sampling_contract["update_layout"][
                "update_decisions"
            ]
            if update_decisions != effective_batch:
                raise ValueError(
                    "parent-anchored replay update size must equal one complete "
                    "optimizer update: "
                    f"update_decisions={update_decisions}, "
                    f"effective_batch={effective_batch}"
                )
        identity["decision_sampling"] = sampling_contract
        execution_contract = _derive_sft_preflight_execution_contract(
            source,
            identity,
        )
        if sampling_mode != QUOTA_SAMPLING_MODE:
            exercised_decisions = execution_contract["executed_lm_decisions"]
            exercised_keys = tuple(ordered_decision_keys[:exercised_decisions])
            exercised_key_rows = [
                [conversation_index, message_index]
                for conversation_index, message_index in exercised_keys
            ]
            if len(exercised_key_rows) != exercised_decisions:
                raise RuntimeError(
                    "replay production order is shorter than the bounded preflight prefix"
                )
            sampling_evidence = {
                "kind": "localagent_sft_preflight_mixed_replay_prefix",
                "schema_version": 1,
                "production": {
                    "selected_decisions": selected_decisions,
                    "sampling_contract": sampling_contract,
                    "sampling_contract_sha256": canonical_sha256(sampling_contract),
                },
                "exercised_prefix": {
                    "decisions": exercised_decisions,
                    "decision_keys": exercised_key_rows,
                    "decision_keys_sha256": canonical_sha256(exercised_key_rows),
                    "equals_production_order_prefix": True,
                },
                "bounded_execution": execution_contract,
            }
    return identity, sampling_evidence


def _bind_sft_parent_checkpoint_identity(
    data_identity: Mapping[str, Any],
    sampling_evidence: Mapping[str, Any] | None,
    *,
    parent_checkpoint_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind parent-anchored sampling to the exact validated parent receipt."""

    identity = copy.deepcopy(dict(data_identity))
    decision_sampling = _mapping(
        identity.get("decision_sampling"),
        label="parent-anchored decision sampling",
    )
    binding = copy.deepcopy(dict(parent_checkpoint_binding))
    existing_binding = decision_sampling.get("parent_checkpoint_binding")
    if existing_binding is not None and existing_binding != binding:
        raise ValueError(
            "parent-anchored decision sampling contains a drifted parent binding"
        )

    if sampling_evidence is None:
        raise ValueError(
            "parent-anchored decision sampling requires production prefix evidence"
        )
    evidence = copy.deepcopy(dict(sampling_evidence))
    production = _mapping(
        evidence.get("production"),
        label="parent-anchored production sampling evidence",
    )
    evidence_contract = _mapping(
        production.get("sampling_contract"),
        label="parent-anchored production sampling contract",
    )
    evidence_binding = evidence_contract.get("parent_checkpoint_binding")
    if evidence_binding is not None and evidence_binding != binding:
        raise ValueError(
            "parent-anchored sampling evidence contains a drifted parent binding"
        )
    if {
        key: value
        for key, value in evidence_contract.items()
        if key != "parent_checkpoint_binding"
    } != {
        key: value
        for key, value in decision_sampling.items()
        if key != "parent_checkpoint_binding"
    }:
        raise ValueError(
            "parent-anchored data identity and production sampling evidence disagree"
        )

    decision_sampling["parent_checkpoint_binding"] = binding
    evidence_contract["parent_checkpoint_binding"] = binding
    identity["decision_sampling"] = decision_sampling
    production["sampling_contract"] = evidence_contract
    production["sampling_contract_sha256"] = canonical_sha256(evidence_contract)
    evidence["production"] = production
    return identity, evidence


def _derive_sft_data_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the exact data identity hashed by the production SFT runner."""

    identity, _sampling_evidence = _derive_sft_data_identity_and_sampling(source)
    return identity


def _snapshots_after(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    after: list[dict[str, Any]] = []
    for record in records:
        updated = {"role": record["role"], "index": record["index"]}
        for key in ("jsonl", "manifest", "generator_config"):
            if key in record:
                updated[key] = _path_snapshot(Path(record[key]["path"]))
        after.append(updated)
    return after


def _records_untouched(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> bool:
    return before == after


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _assert_rl_path_isolation(
    *,
    work_path: Path,
    receipt_path: Path,
    production_out: Path,
    source_files: list[Path],
) -> None:
    if _paths_overlap(work_path, receipt_path):
        raise ValueError("preflight work directory and receipt path must be disjoint")
    if _paths_overlap(work_path, production_out):
        raise ValueError("preflight work directory overlaps the production RL output path")
    if _paths_overlap(receipt_path, production_out):
        raise ValueError("preflight receipt path overlaps the production RL output path")
    for source_file in source_files:
        if _paths_overlap(work_path, source_file):
            raise ValueError(
                f"preflight work directory overlaps a source artifact: {source_file}"
            )
        if receipt_path.resolve(strict=False) == source_file.resolve(strict=False):
            raise ValueError(f"preflight receipt would replace a source artifact: {source_file}")


def _assert_sft_path_isolation(
    *,
    work_path: Path,
    receipt_path: Path,
    production_out: Path,
    source_files: list[Path],
) -> None:
    if _paths_overlap(work_path, receipt_path):
        raise ValueError("preflight work directory and receipt path must be disjoint")
    if _paths_overlap(work_path, production_out):
        raise ValueError("preflight work directory overlaps the production SFT output path")
    if _paths_overlap(receipt_path, production_out):
        raise ValueError("preflight receipt path overlaps the production SFT output path")
    for source_file in source_files:
        if _paths_overlap(work_path, source_file):
            raise ValueError(
                f"preflight work directory overlaps a source artifact: {source_file}"
            )
        if receipt_path.resolve(strict=False) == source_file.resolve(strict=False):
            raise ValueError(f"preflight receipt would replace a source artifact: {source_file}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _rl_lineage_data_identity(data: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "train_artifacts",
        "eval_artifacts",
        "split_audit",
        "selected_eval_split_audit",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("RL metrics data lineage is incomplete: " + ", ".join(missing))
    identity = {key: data[key] for key in required}
    for optional in (
        "prompt_sampling",
        "eval_selection",
        "preflight_minimum_coverage",
    ):
        if optional in data:
            identity[optional] = data[optional]
    if data.get("conversation_prompt_contract") is not None:
        for key in (
            "conversation_prompt_contract",
            "parent_conversation_prompt_contract",
            "context_preflight",
            "prompt_truncation",
            "schema_validation",
        ):
            if key not in data:
                raise ValueError(f"RL metrics data lineage is missing {key}")
            identity[key] = data[key]
    return identity


def _assert_recorded_data_artifacts(
    data: Mapping[str, Any],
    declared: list[dict[str, Any]],
) -> None:
    for role, artifacts_key, paths_key in (
        ("train", "train_artifacts", "paths"),
        ("eval", "eval_artifacts", "eval_paths"),
    ):
        expected = [record for record in declared if record["role"] == role]
        artifacts = data.get(artifacts_key)
        paths = data.get(paths_key)
        if not isinstance(artifacts, list) or len(artifacts) != len(expected):
            raise ValueError(f"RL metrics {artifacts_key} do not match declared sources")
        if not isinstance(paths, list) or paths != [row["jsonl"]["path"] for row in expected]:
            raise ValueError(f"RL metrics {paths_key} do not match declared sources")
        for configured, recorded in zip(expected, artifacts):
            if not isinstance(recorded, Mapping):
                raise TypeError(f"RL metrics {artifacts_key} entry must be a mapping")
            if recorded.get("path") != configured["jsonl"]["path"]:
                raise ValueError(f"RL metrics {artifacts_key} path mismatch")
            jsonl_identity = {
                key: configured["jsonl"][key] for key in ("bytes", "sha256")
            }
            if "manifest" not in configured:
                if any(recorded.get(key) != value for key, value in jsonl_identity.items()):
                    raise ValueError(f"RL metrics {artifacts_key} JSONL identity mismatch")
                continue
            if recorded.get("jsonl") != jsonl_identity:
                raise ValueError(f"RL metrics {artifacts_key} JSONL identity mismatch")
            sidecar = recorded.get("sidecar")
            generator = recorded.get("generator_config")
            manifest_identity = {
                key: configured["manifest"][key] for key in ("bytes", "sha256")
            }
            generator_identity = {
                key: configured["generator_config"][key] for key in ("bytes", "sha256")
            }
            if not isinstance(sidecar, Mapping) or any(
                sidecar.get(key) != value for key, value in manifest_identity.items()
            ):
                raise ValueError(f"RL metrics {artifacts_key} manifest identity mismatch")
            if generator != generator_identity:
                raise ValueError(
                    f"RL metrics {artifacts_key} generator config identity mismatch"
                )


def _lineage_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


class _ResourceSampler:
    """Poll process and accelerator allocation while the stage runner owns the main thread."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = _rss_bytes()
        self.peak_cuda_allocated_bytes = 0
        self.peak_cuda_reserved_bytes = 0
        self.peak_mps_allocated_bytes = 0
        self.peak_mps_driver_bytes = 0
        self.mps_recommended_max_memory_bytes = 0
        self.errors: list[str] = []

    def _sample(self) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
        try:
            if torch.cuda.is_available():
                self.peak_cuda_allocated_bytes = max(
                    self.peak_cuda_allocated_bytes,
                    int(torch.cuda.memory_allocated()),
                    int(torch.cuda.max_memory_allocated()),
                )
                self.peak_cuda_reserved_bytes = max(
                    self.peak_cuda_reserved_bytes,
                    int(torch.cuda.memory_reserved()),
                    int(torch.cuda.max_memory_reserved()),
                )
            if torch.backends.mps.is_available():
                self.mps_recommended_max_memory_bytes = max(
                    self.mps_recommended_max_memory_bytes,
                    int(torch.mps.recommended_max_memory()),
                )
                self.peak_mps_allocated_bytes = max(
                    self.peak_mps_allocated_bytes,
                    int(torch.mps.current_allocated_memory()),
                )
                self.peak_mps_driver_bytes = max(
                    self.peak_mps_driver_bytes,
                    int(torch.mps.driver_allocated_memory()),
                )
        except (RuntimeError, TypeError) as exc:  # pragma: no cover - backend/runtime dependent
            message = f"{type(exc).__name__}: {exc}"
            if message not in self.errors:
                self.errors.append(message)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._sample()
        self._thread = threading.Thread(
            target=self._loop,
            name="localagent-preflight-resource-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def as_dict(self, *, baseline_rss_bytes: int) -> dict[str, Any]:
        recommended = self.mps_recommended_max_memory_bytes
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "peak_process_rss_bytes": self.peak_rss_bytes,
            "peak_process_rss_delta_from_loaded_runner_bytes": max(
                0,
                self.peak_rss_bytes - baseline_rss_bytes,
            ),
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "peak_mps_allocated_bytes": self.peak_mps_allocated_bytes,
            "peak_mps_driver_allocated_bytes": self.peak_mps_driver_bytes,
            "mps_recommended_max_memory_bytes": recommended,
            "peak_mps_allocated_to_recommended_ratio": (
                self.peak_mps_allocated_bytes / recommended if recommended > 0 else None
            ),
            "peak_mps_driver_to_recommended_ratio": (
                self.peak_mps_driver_bytes / recommended if recommended > 0 else None
            ),
            "peak_mps_driver_within_recommended_working_set": (
                self.peak_mps_driver_bytes <= recommended if recommended > 0 else None
            ),
            "sampling_errors": self.errors,
        }


def seal_preflight_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the canonical semantic self-hash."""

    if "receipt_self_sha256" in receipt:
        raise ValueError("cannot seal a receipt that already has receipt_self_sha256")
    sealed = copy.deepcopy(dict(receipt))
    sealed["receipt_self_sha256"] = canonical_sha256(sealed)
    return sealed


def assert_preflight_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the receipt kind, version, and semantic self-hash."""

    schema_version = receipt.get("schema_version")
    if (
        receipt.get("kind") != PREFLIGHT_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PREFLIGHT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported one-update preflight receipt")
    recorded = receipt.get("receipt_self_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("preflight receipt has no valid self-hash")
    payload = dict(receipt)
    payload.pop("receipt_self_sha256", None)
    if recorded != canonical_sha256(payload):
        raise ValueError("preflight receipt self-hash mismatch")


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    assert_preflight_receipt(receipt)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing preflight receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"preflight receipt temporary path already exists: {temporary}")
    temporary.write_bytes(_receipt_bytes(receipt))
    temporary.replace(path)


def _synchronize_accelerator() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _sft_optimizer_step_values(checkpoint: Mapping[str, Any]) -> list[int]:
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise TypeError("isolated SFT checkpoint optimizer state is invalid")
    state = optimizer.get("state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("isolated SFT checkpoint optimizer has no parameter state")
    values: list[int] = []
    for parameter, raw_state in state.items():
        if not isinstance(raw_state, Mapping) or "step" not in raw_state:
            raise ValueError(
                f"isolated SFT optimizer parameter {parameter!r} has no step counter"
            )
        raw_step = raw_state["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ValueError("isolated SFT optimizer step counter must be scalar")
            raw_step = raw_step.detach().cpu().item()
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, (int, float))
            or not math.isfinite(float(raw_step))
            or float(raw_step) != int(raw_step)
        ):
            raise ValueError("isolated SFT optimizer step counter is invalid")
        values.append(int(raw_step))
    return values


def _sft_accounted_rows(token_accounting: Any) -> int | None:
    if not isinstance(token_accounting, Mapping):
        return None
    sources = token_accounting.get("sources")
    if not isinstance(sources, Mapping):
        return None
    rows = [value.get("rows") for value in sources.values() if isinstance(value, Mapping)]
    if len(rows) != len(sources) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in rows
    ):
        return None
    return sum(rows)


def _assert_recorded_sft_data_paths(
    data: Mapping[str, Any],
    declared: list[dict[str, Any]],
) -> None:
    for role, paths_key in (
        ("train", "paths"),
        ("decay", "decay_paths"),
        ("eval", "eval_paths"),
    ):
        expected = [
            record["jsonl"]["path"] for record in declared if record["role"] == role
        ]
        recorded = data.get(paths_key)
        if role == "train":
            if recorded != expected:
                raise ValueError("isolated SFT metrics paths do not match declared train sources")
        elif expected:
            if recorded != expected:
                raise ValueError(
                    f"isolated SFT metrics {paths_key} do not match declared {role} sources"
                )
        elif recorded not in (None, []):
            raise ValueError(f"isolated SFT metrics recorded unexpected {role} sources")


def _sft_metrics_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "stage",
            "checkpoint",
            "conversation_rows",
            "single_turn_rows",
            "probe_decision_rows",
            "loss_last",
            "loss_steps",
            "dataset_token_accounting",
            "token_accounting",
            "token_accounting_scope",
            "lm_sampling",
            "fixed_horizon_progress",
            "conversation_prompt_contract",
            "lineage",
            "data",
            "heldout_eval",
            "heldout_structured_eval",
            "execution",
            "continuation",
            "structured_heads",
        )
        if key in metrics
    }


def _expected_sft_lr_at_step(
    effective: Mapping[str, Any],
    *,
    step: int,
) -> float:
    from localagent.train.loop import cosine_lr, wsd_lr

    schedule = _mapping(effective.get("schedule", {}), label="schedule")
    optim = _mapping(effective.get("optim", {}), label="optim")
    peak = float(optim.get("lr", 1e-4))
    warmup = int(schedule.get("warmup_steps", 50))
    total_steps = int(schedule.get("total_steps", 1))
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < total_steps:
        raise ValueError("SFT learning-rate step must fall within the fixed horizon")
    if schedule.get("type", "cosine") == "wsd":
        return wsd_lr(
            step,
            total_steps,
            peak,
            warmup,
            float(schedule.get("decay_frac", 0.2)),
            min_ratio=0.0,
        )
    return cosine_lr(step, total_steps, peak, warmup, 0.1)


def _derive_rl_preflight_execution_contract(
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the shortest production-schedule prefix containing a nonzero RL LR."""

    from localagent.train.loop import cosine_lr

    schedule = _mapping(effective.get("schedule", {}), label="schedule")
    optim = _mapping(effective.get("optim", {}), label="optim")
    total_steps = schedule.get("total_steps")
    warmup_steps = schedule.get("warmup_steps", 5)
    peak_lr = optim.get("lr", 2e-4)
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < 1
    ):
        raise ValueError("schedule.total_steps must be a positive integer")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("schedule.warmup_steps must be a non-negative integer")
    if (
        isinstance(peak_lr, bool)
        or not isinstance(peak_lr, (int, float))
        or not math.isfinite(float(peak_lr))
        or float(peak_lr) <= 0.0
    ):
        raise ValueError("optim.lr must be a finite positive number")

    learning_rates: list[float] = []
    first_nonzero_step = None
    for step in range(total_steps):
        learning_rate = float(
            cosine_lr(
                step,
                total_steps,
                float(peak_lr),
                warmup_steps,
                0.1,
            )
        )
        learning_rates.append(learning_rate)
        if learning_rate > 0.0:
            first_nonzero_step = step
            break
    if first_nonzero_step is None:
        raise ValueError(
            "production RL schedule contains no nonzero learning-rate step"
        )
    return {
        "kind": "production_schedule_first_nonzero_lr_prefix_v1",
        "production_schedule_total_steps": total_steps,
        "production_schedule_warmup_steps": warmup_steps,
        "production_peak_learning_rate": float(peak_lr),
        "execution_rollout_step_limit": first_nonzero_step + 1,
        "first_nonzero_learning_rate_step": first_nonzero_step,
        "expected_learning_rates": learning_rates,
    }


def _expected_sft_step_zero_lr(effective: Mapping[str, Any]) -> float:
    return _expected_sft_lr_at_step(effective, step=0)


def _expected_sft_executed_learning_rates(
    effective: Mapping[str, Any],
    *,
    execution_update_limit: int,
) -> list[float]:
    if (
        isinstance(execution_update_limit, bool)
        or not isinstance(execution_update_limit, int)
        or execution_update_limit < 1
    ):
        raise ValueError("SFT execution update limit must be a positive integer")
    return [
        _expected_sft_lr_at_step(effective, step=step)
        for step in range(execution_update_limit)
    ]


def _sft_model_parameter_names(model_config: ModelConfig) -> list[str]:
    """Return canonical ``named_parameters`` order without allocating model storage."""

    from localagent.model import LocalAgentLM

    with torch.device("meta"):
        model = LocalAgentLM(model_config)
    return [name for name, _ in model.named_parameters()]


def _inspect_sft_model_parameter_scope(
    *,
    checkpoint: Mapping[str, Any],
    parent_checkpoint: Mapping[str, Any],
    effective: Mapping[str, Any],
    model_config: ModelConfig,
    execution_update_limit: int,
) -> tuple[dict[str, Any], list[str]]:
    """Compare the isolated LM state and derive exact optimizer-scope evidence."""

    errors: list[str] = []
    optim = _mapping(effective.get("optim", {}), label="optim")
    configured_freeze = optim.get("freeze_parameters")
    raw_frozen_names = (
        list(configured_freeze) if isinstance(configured_freeze, list) else []
    )
    invalid_frozen_indexes = [
        index for index, name in enumerate(raw_frozen_names) if not isinstance(name, str)
    ]
    frozen_names = [
        name for name in raw_frozen_names if isinstance(name, str)
    ]
    model_names = _sft_model_parameter_names(model_config)
    model_name_set = set(model_names)
    seen_frozen_names: set[str] = set()
    duplicate_frozen_names: list[str] = []
    for name in frozen_names:
        if name in seen_frozen_names and name not in duplicate_frozen_names:
            duplicate_frozen_names.append(name)
        seen_frozen_names.add(name)
    unknown_frozen_names = [
        name for name in frozen_names if name not in model_name_set
    ]
    if configured_freeze is not None and not isinstance(configured_freeze, list):
        errors.append("configured SFT frozen-parameter scope is not a list")
    if invalid_frozen_indexes:
        errors.append(
            "configured SFT frozen-parameter scope contains non-string entries at indexes: "
            + ", ".join(str(index) for index in invalid_frozen_indexes)
        )
    if duplicate_frozen_names:
        errors.append(
            "configured SFT frozen-parameter scope contains duplicates: "
            + ", ".join(duplicate_frozen_names)
        )
    if unknown_frozen_names:
        errors.append(
            "configured SFT frozen-parameter scope contains unknown names: "
            + ", ".join(unknown_frozen_names)
        )

    frozen_name_set = set(frozen_names)
    optimizer_model_names = [
        name for name in model_names if name not in frozen_name_set
    ]
    parent_state = parent_checkpoint.get(
        "state_dict",
        parent_checkpoint.get("model"),
    )
    child_state = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(parent_state, Mapping):
        errors.append("SFT continuation parent has no model state mapping")
        parent_state = {}
    if not isinstance(child_state, Mapping):
        errors.append("isolated SFT checkpoint has no model state mapping")
        child_state = {}

    missing_parent_names = [name for name in model_names if name not in parent_state]
    missing_child_names = [name for name in model_names if name not in child_state]
    if missing_parent_names:
        errors.append(
            "SFT continuation parent model state is missing named parameters: "
            + ", ".join(missing_parent_names)
        )
    if missing_child_names:
        errors.append(
            "isolated SFT model state is missing named parameters: "
            + ", ".join(missing_child_names)
        )

    incompatible_names: list[str] = []
    for name in model_names:
        parent_tensor = parent_state.get(name)
        child_tensor = child_state.get(name)
        if name in missing_parent_names or name in missing_child_names:
            continue
        if (
            not isinstance(parent_tensor, torch.Tensor)
            or not isinstance(child_tensor, torch.Tensor)
            or parent_tensor.shape != child_tensor.shape
            or parent_tensor.dtype != child_tensor.dtype
        ):
            incompatible_names.append(name)
    if incompatible_names:
        errors.append(
            "isolated SFT model tensors are incompatible with the parent: "
            + ", ".join(incompatible_names)
        )

    comparable_names = model_name_set.difference(
        missing_parent_names,
        missing_child_names,
        incompatible_names,
    )
    changed_frozen_names = [
        name
        for name in frozen_names
        if name in comparable_names
        and not torch.equal(parent_state[name], child_state[name])
    ]
    compared_frozen_names = [
        name for name in frozen_names if name in comparable_names
    ]
    if changed_frozen_names:
        errors.append(
            "isolated SFT changed configured frozen model tensors: "
            + ", ".join(changed_frozen_names)
        )
    if len(compared_frozen_names) != len(frozen_names):
        errors.append(
            "isolated SFT could not compare every configured frozen model tensor"
        )

    executed_learning_rates = _expected_sft_executed_learning_rates(
        effective,
        execution_update_limit=execution_update_limit,
    )
    transition_required = any(learning_rate != 0.0 for learning_rate in executed_learning_rates)
    first_changed_unfrozen_name = None
    if transition_required:
        for name in optimizer_model_names:
            if name in comparable_names and not torch.equal(
                parent_state[name],
                child_state[name],
            ):
                first_changed_unfrozen_name = name
                break
        if first_changed_unfrozen_name is None:
            errors.append(
                "isolated SFT bounded prefix includes a nonzero learning rate but changed "
                "no unfrozen model tensor"
            )
    non_learning_limitation = (
        _SFT_ZERO_EXECUTED_LR_NON_LEARNING_LIMITATION
        if not transition_required
        else None
    )
    evidence = {
        "model_named_parameter_names": model_names,
        "configured_frozen_model_parameter_names": frozen_names,
        "expected_optimizer_model_parameter_names": optimizer_model_names,
        "expected_trainable_model_tensor_count": len(optimizer_model_names),
        "training_contract_frozen_model_parameter_names": None,
        "training_contract_optimizer_model_parameter_names": None,
        "all_auxiliary_heads_disabled": None,
        "optimizer_param_group_parameter_count": None,
        "optimizer_parameter_count_matches_expected": None,
        "expected_completed_lm_cursor": None,
        "observed_completed_lm_cursor": None,
        "completed_lm_cursor_matches_expected": None,
        "compared_frozen_model_parameter_names": compared_frozen_names,
        "frozen_model_tensors_exactly_preserved": (
            len(compared_frozen_names) == len(frozen_names)
            and not changed_frozen_names
        ),
        "execution_optimizer_update_limit": execution_update_limit,
        "executed_learning_rates": executed_learning_rates,
        "step_zero_learning_rate": executed_learning_rates[0],
        "last_executed_learning_rate": executed_learning_rates[-1],
        "any_executed_learning_rate_nonzero": transition_required,
        "unfrozen_model_transition_required": transition_required,
        "first_changed_unfrozen_model_parameter": first_changed_unfrozen_name,
        "non_learning_limitation": non_learning_limitation,
    }
    return evidence, errors


def _validate_sft_preflight_outputs(
    *,
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    effective: Mapping[str, Any],
    model_config: ModelConfig,
    parent_checkpoint: Mapping[str, Any],
    parent_checkpoint_sha256: str,
    tokenizer_sha256: str,
    expected_data_identity: Mapping[str, Any],
    sampling_evidence: Mapping[str, Any] | None,
    execution_contract: Mapping[str, Any],
    declared_data: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    from localagent.train.sft import _load_validated_sft_resume_checkpoint

    try:
        _load_validated_sft_resume_checkpoint(checkpoint)
    except (TypeError, ValueError) as exc:
        errors.append(f"isolated SFT checkpoint resume integrity is invalid: {exc}")

    batch = _mapping(effective.get("batch", {}), label="batch")
    micro_batch_size = int(batch["micro_batch_size"])
    grad_accum_steps = int(batch["grad_accum_steps"])
    effective_batch_size = micro_batch_size * grad_accum_steps
    pad_to_input_tokens = batch.get("pad_to_input_tokens")
    optim = _mapping(effective.get("optim", {}), label="optim")
    expected_loss_normalization = optim.get(
        "loss_normalization",
        "microbatch_mean_v1",
    )
    expected_freeze_parameters = optim.get("freeze_parameters")
    continuation = effective.get("continuation")
    planned_optimizer_updates = int(effective["schedule"]["total_steps"])
    execution_update_limit = execution_contract.get(
        "execution_optimizer_update_limit"
    )
    if (
        isinstance(execution_update_limit, bool)
        or not isinstance(execution_update_limit, int)
        or execution_update_limit < 1
        or execution_update_limit > planned_optimizer_updates
    ):
        raise ValueError("SFT preflight execution update limit is invalid")
    executed_decisions = execution_update_limit * effective_batch_size
    expected_update_label = (
        "one" if execution_update_limit == 1 else str(execution_update_limit)
    )
    model_parameter_scope, scope_errors = _inspect_sft_model_parameter_scope(
        checkpoint=checkpoint,
        parent_checkpoint=parent_checkpoint,
        effective=effective,
        model_config=model_config,
        execution_update_limit=execution_update_limit,
    )
    errors.extend(scope_errors)
    expected_optimizer_model_names = model_parameter_scope[
        "expected_optimizer_model_parameter_names"
    ]

    check(metrics.get("stage") == "sft", "isolated metrics stage is not 'sft'")
    try:
        recorded_checkpoint = Path(str(metrics.get("checkpoint"))).resolve(strict=False)
    except (TypeError, ValueError):
        recorded_checkpoint = Path()
        errors.append("isolated metrics checkpoint path is invalid")
    check(
        recorded_checkpoint == checkpoint_path.resolve(strict=False),
        "isolated metrics checkpoint path is not the redirected checkpoint",
    )
    check(
        metrics.get("loss_steps") == execution_update_limit,
        f"isolated SFT run did not record {expected_update_label} loss step(s)",
    )
    check(checkpoint.get("stage") == "sft", "isolated checkpoint stage is not 'sft'")
    check(
        checkpoint.get("step") == execution_update_limit - 1,
        "isolated checkpoint did not stop at the bounded SFT update limit",
    )

    loss_history = checkpoint.get("loss_history")
    check(
        isinstance(loss_history, list)
        and len(loss_history) == execution_update_limit
        and all(
            not isinstance(loss, bool)
            and isinstance(loss, (int, float))
            and math.isfinite(float(loss))
            for loss in loss_history
        ),
        "isolated SFT checkpoint loss history does not match the bounded update count",
    )
    if (
        isinstance(loss_history, list)
        and len(loss_history) == execution_update_limit
    ):
        check(
            metrics.get("loss_last") == loss_history[-1],
            "isolated SFT metrics/checkpoint loss mismatch",
        )

    training_contract = checkpoint.get("training_contract")
    sampling_state = checkpoint.get("sampling_state")
    if not isinstance(training_contract, Mapping):
        errors.append("isolated SFT checkpoint training contract is missing")
    else:
        model_parameter_scope["training_contract_frozen_model_parameter_names"] = (
            training_contract.get("freeze_parameters")
        )
        model_parameter_scope[
            "training_contract_optimizer_model_parameter_names"
        ] = training_contract.get("optimizer_model_parameter_names")
        check(
            training_contract.get("steps") == planned_optimizer_updates,
            "isolated SFT training contract changed the planned fixed horizon",
        )
        check(
            training_contract.get("batch_size") == micro_batch_size,
            "isolated SFT training contract changed micro-batch size",
        )
        check(
            training_contract.get("accum_steps") == grad_accum_steps,
            "isolated SFT training contract changed gradient accumulation",
        )
        check(
            _valid_sha256(training_contract.get("initial_model_sha256")),
            "isolated SFT training contract has no initial-model state hash",
        )
        expected_sampling = expected_data_identity.get(
            "decision_sampling",
            {
                "mode": (
                    "iid_with_replacement_v1"
                    if bool(effective.get("data", {}).get("shuffle", True))
                    else "source_order_wrapping_v1"
                )
            },
        )
        check(
            training_contract.get("lm_sampling") == expected_sampling,
            "isolated SFT training contract changed LM sampling",
        )
        check(
            training_contract.get("pad_to_input_tokens") == pad_to_input_tokens,
            "isolated SFT training contract changed fixed LM padding",
        )
        check(
            training_contract.get("loss_normalization")
            == expected_loss_normalization,
            "isolated SFT training contract changed loss normalization",
        )
        check(
            training_contract.get("freeze_parameters")
            == expected_freeze_parameters,
            "isolated SFT training contract changed frozen parameters",
        )
        if expected_freeze_parameters is not None:
            check(
                training_contract.get("optimizer_model_parameter_names")
                == expected_optimizer_model_names,
                "isolated SFT training contract optimizer model parameter names "
                "do not equal model named_parameters minus the freeze list",
            )
        else:
            check(
                training_contract.get("optimizer_model_parameter_names") is None,
                "isolated SFT training contract recorded an unexpected frozen-model "
                "optimizer scope",
            )
        configured_optimizer_name = optim.get("name", "adamw")
        configured_weight_decay = optim.get("weight_decay", 0.0)
        configured_grad_clip = optim.get("grad_clip", 1.0)
        optimizer_contract = training_contract.get("optimizer")
        if not isinstance(optimizer_contract, Mapping):
            errors.append("isolated SFT training contract optimizer is missing")
        else:
            recorded_kind = optimizer_contract.get("kind")
            check(
                isinstance(configured_optimizer_name, str)
                and isinstance(recorded_kind, str)
                and configured_optimizer_name.casefold() == recorded_kind.casefold(),
                "isolated SFT training contract optimizer kind does not match optim.name",
            )
            for config_value, field in (
                (configured_weight_decay, "weight_decay"),
                (configured_grad_clip, "grad_clip"),
            ):
                recorded_value = optimizer_contract.get(field)
                check(
                    not isinstance(config_value, bool)
                    and isinstance(config_value, (int, float))
                    and math.isfinite(float(config_value))
                    and not isinstance(recorded_value, bool)
                    and isinstance(recorded_value, (int, float))
                    and math.isfinite(float(recorded_value))
                    and float(config_value) == float(recorded_value),
                    f"isolated SFT training contract optimizer {field} does not "
                    f"match optim.{field}",
                )
        check(
            metrics.get("lm_sampling") == expected_sampling,
            "isolated SFT metrics changed LM sampling",
        )
    fixed_horizon_progress = metrics.get("fixed_horizon_progress")
    check(
        isinstance(fixed_horizon_progress, Mapping)
        and fixed_horizon_progress.get("planned_optimizer_updates")
        == planned_optimizer_updates
        and fixed_horizon_progress.get("completed_optimizer_updates")
        == execution_update_limit
        and fixed_horizon_progress.get("partial")
        == (planned_optimizer_updates > execution_update_limit),
        "isolated SFT fixed-horizon progress is invalid",
    )
    check(
        checkpoint.get("fixed_horizon_progress") == fixed_horizon_progress,
        "isolated SFT metrics/checkpoint fixed-horizon progress mismatch",
    )

    if sampling_evidence is not None:
        production = sampling_evidence.get("production")
        exercised_prefix = sampling_evidence.get("exercised_prefix")
        production_contract = (
            production.get("sampling_contract")
            if isinstance(production, Mapping)
            else None
        )
        decision_keys = (
            exercised_prefix.get("decision_keys")
            if isinstance(exercised_prefix, Mapping)
            else None
        )
        check(
            sampling_evidence.get("kind")
            == "localagent_sft_preflight_mixed_replay_prefix"
            and sampling_evidence.get("schema_version") == 1,
            "isolated SFT mixed-replay sampling evidence kind/version is invalid",
        )
        check(
            isinstance(production, Mapping)
            and production.get("selected_decisions")
            == planned_optimizer_updates * effective_batch_size
            and isinstance(production_contract, Mapping)
            and production_contract == expected_data_identity.get("decision_sampling")
            and production.get("sampling_contract_sha256")
            == canonical_sha256(production_contract),
            "isolated SFT production mixed-replay contract is invalid",
        )
        check(
            isinstance(exercised_prefix, Mapping)
            and exercised_prefix.get("decisions") == executed_decisions
            and isinstance(decision_keys, list)
            and len(decision_keys) == executed_decisions
            and exercised_prefix.get("decision_keys_sha256")
            == canonical_sha256(decision_keys)
            and exercised_prefix.get("equals_production_order_prefix") is True,
            "isolated SFT exercised mixed-replay prefix is invalid",
        )
        recorded_bounded_execution = sampling_evidence.get("bounded_execution")
        if recorded_bounded_execution is not None:
            check(
                recorded_bounded_execution == execution_contract,
                "isolated SFT sampling evidence bounded-execution contract mismatch",
            )

    if not isinstance(sampling_state, Mapping):
        errors.append("isolated SFT checkpoint sampling state is missing")
    else:
        check(
            sampling_state.get("completed_steps") == execution_update_limit,
            "isolated SFT sampler did not complete the bounded update count",
        )
        check(
            sampling_state.get("completed_microbatches")
            == execution_update_limit * grad_accum_steps,
            "isolated SFT sampler did not complete the configured microbatches",
        )
        if execution_contract.get("executed_through_first_pulse") is True:
            observed_lm_cursor = sampling_state.get("lm_cursor")
            model_parameter_scope["expected_completed_lm_cursor"] = (
                executed_decisions
            )
            model_parameter_scope["observed_completed_lm_cursor"] = observed_lm_cursor
            model_parameter_scope["completed_lm_cursor_matches_expected"] = (
                observed_lm_cursor == executed_decisions
            )
            check(
                observed_lm_cursor == executed_decisions,
                "isolated parent-anchored SFT LM cursor does not prove consumption "
                "through the first pulse",
            )

    try:
        optimizer_steps = _sft_optimizer_step_values(checkpoint)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        optimizer_steps = []
    else:
        check(
            all(step == execution_update_limit for step in optimizer_steps),
            "isolated SFT Adam state does not prove exactly "
            f"{expected_update_label} optimizer update(s)",
        )
    optimizer = checkpoint.get("optimizer")
    if isinstance(optimizer, Mapping):
        param_groups = optimizer.get("param_groups")
        check(
            isinstance(param_groups, list) and bool(param_groups),
            "isolated SFT optimizer has no parameter groups",
        )
        if isinstance(param_groups, list) and param_groups:
            expected_lr = _expected_sft_lr_at_step(
                effective,
                step=execution_update_limit - 1,
            )
            check(
                all(
                    isinstance(group, Mapping)
                    and not isinstance(group.get("lr"), bool)
                    and isinstance(group.get("lr"), (int, float))
                    and math.isclose(
                        float(group["lr"]),
                        expected_lr,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    for group in param_groups
                ),
                "isolated SFT optimizer learning rate does not match preserved step-zero schedule",
            )
            heads = _mapping(effective.get("heads", {}), label="heads")
            all_auxiliary_heads_disabled = all(
                heads.get(key, True) is False
                for key in (
                    "joint_tool_pointer",
                    "train_route_head",
                    "train_dense_selector",
                )
            )
            model_parameter_scope["all_auxiliary_heads_disabled"] = (
                all_auxiliary_heads_disabled
            )
            if all_auxiliary_heads_disabled:
                parameter_count = 0
                valid_parameter_lists = True
                for group in param_groups:
                    parameters = group.get("params") if isinstance(group, Mapping) else None
                    if not isinstance(parameters, (list, tuple)):
                        valid_parameter_lists = False
                        continue
                    parameter_count += len(parameters)
                model_parameter_scope["optimizer_param_group_parameter_count"] = (
                    parameter_count if valid_parameter_lists else None
                )
                model_parameter_scope[
                    "optimizer_parameter_count_matches_expected"
                ] = (
                    valid_parameter_lists
                    and parameter_count
                    == model_parameter_scope["expected_trainable_model_tensor_count"]
                )
                check(
                    valid_parameter_lists,
                    "isolated SFT optimizer parameter groups have invalid parameter lists",
                )
                check(
                    valid_parameter_lists
                    and parameter_count
                    == model_parameter_scope["expected_trainable_model_tensor_count"],
                    "isolated LM-only SFT optimizer parameter count does not match "
                    "the expected trainable model tensor count",
                )

    lineage = metrics.get("lineage")
    checkpoint_lineage = checkpoint.get("lineage")
    expected_data_sha256 = canonical_sha256(expected_data_identity)
    if not isinstance(lineage, Mapping):
        errors.append("isolated SFT metrics have no lineage mapping")
    else:
        expected_lineage = {
            "stage": "sft",
            "config_sha256": _lineage_config_sha256(effective),
            "model_config_sha256": canonical_sha256(model_config.__dict__),
            "data_sha256": expected_data_sha256,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
        }
        for key, expected in expected_lineage.items():
            check(lineage.get(key) == expected, f"isolated SFT lineage {key} mismatch")
        check(
            dict(lineage) == checkpoint_lineage,
            "isolated SFT metrics/checkpoint lineage mismatch",
        )

    data = metrics.get("data")
    checkpoint_data = checkpoint.get("data")
    if not isinstance(data, Mapping):
        errors.append("isolated SFT metrics have no data mapping")
    else:
        try:
            _assert_recorded_sft_data_paths(data, declared_data)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        check(
            isinstance(checkpoint_data, Mapping)
            and canonical_sha256(data) == canonical_sha256(checkpoint_data),
            "isolated SFT metrics/checkpoint data metadata mismatch",
        )
        expected_selection = expected_data_identity.get("eval_selection")
        if expected_selection is not None:
            check(
                data.get("eval_selection") == expected_selection,
                "isolated SFT held-out selector audit mismatch",
            )

    tokenizer = checkpoint.get("tokenizer")
    check(
        isinstance(tokenizer, Mapping) and tokenizer.get("sha256") == tokenizer_sha256,
        "isolated SFT checkpoint tokenizer lineage mismatch",
    )
    check(
        checkpoint.get("continuation") == continuation
        and metrics.get("continuation") == continuation,
        "isolated SFT continuation contract mismatch",
    )

    execution = metrics.get("execution")
    check(
        isinstance(execution, Mapping) and checkpoint.get("execution") == execution,
        "isolated SFT metrics/checkpoint execution metadata mismatch",
    )
    requested_device = effective.get("runtime", {}).get("device", "auto")
    if isinstance(execution, Mapping):
        check(
            execution.get("requested_device") == requested_device,
            "isolated SFT runner did not use the effective requested device",
        )
        if requested_device == "mps":
            check(
                execution.get("resolved_device") == "mps",
                "isolated SFT runner did not resolve to MPS",
            )

    checkpoint_accounting = checkpoint.get("token_accounting")
    metrics_accounting = metrics.get("token_accounting")
    check(
        checkpoint_accounting == metrics_accounting,
        "isolated SFT metrics/checkpoint token accounting mismatch",
    )
    check(
        checkpoint.get("dataset_token_accounting")
        == metrics.get("dataset_token_accounting"),
        "isolated SFT metrics/checkpoint dataset accounting mismatch",
    )
    check(
        checkpoint.get("token_accounting_scope") == "language_model_microbatches"
        and metrics.get("token_accounting_scope") == "language_model_microbatches",
        "isolated SFT token-accounting scope mismatch",
    )
    check(
        _sft_accounted_rows(metrics_accounting) == executed_decisions,
        "isolated SFT accounting does not cover the exact bounded decision prefix",
    )

    dataset_accounting = metrics.get("dataset_token_accounting")
    main_dataset = (
        dataset_accounting.get("main")
        if isinstance(dataset_accounting, Mapping)
        else None
    )
    main_rows = _sft_accounted_rows(main_dataset)
    if (
        isinstance(training_contract, Mapping)
        and training_contract.get("lm_sampling") == {"mode": "source_order_wrapping_v1"}
    ):
        check(
            isinstance(main_rows, int) and main_rows >= executed_decisions,
            "isolated source-order SFT prefix wrapped within the bounded prefix",
        )
        if isinstance(main_rows, int) and main_rows > 0 and isinstance(sampling_state, Mapping):
            check(
                sampling_state.get("lm_cursor") == executed_decisions % main_rows,
                "isolated source-order SFT cursor does not match the consumed prefix",
            )
    elif (
        isinstance(training_contract, Mapping)
        and isinstance(training_contract.get("lm_sampling"), Mapping)
        and training_contract["lm_sampling"].get("no_replacement") is True
        and isinstance(sampling_state, Mapping)
    ):
        check(
            sampling_state.get("lm_cursor") == executed_decisions,
            "isolated no-replacement SFT cursor does not match the consumed prefix",
        )

    heldout = metrics.get("heldout_eval")
    check(
        checkpoint.get("heldout_eval") == heldout,
        "isolated SFT metrics/checkpoint held-out evaluation mismatch",
    )
    eval_declared = any(record["role"] == "eval" for record in declared_data)
    if eval_declared:
        check(
            isinstance(heldout, Mapping)
            and isinstance(heldout.get("pre"), Mapping)
            and isinstance(heldout.get("post"), Mapping)
            and isinstance(heldout.get("delta"), Mapping),
            "isolated SFT held-out evaluation did not complete pre and post passes",
        )
        baseline = checkpoint.get("heldout_baseline")
        if isinstance(heldout, Mapping):
            heldout_contract = heldout.get("contract")
            evaluation = _mapping(effective.get("evaluation", {}), label="evaluation")
            check(
                isinstance(heldout_contract, Mapping)
                and heldout_contract.get("pad_to_input_tokens")
                == evaluation.get("pad_to_input_tokens"),
                "isolated SFT held-out contract changed fixed evaluation padding",
            )
            check(
                isinstance(baseline, Mapping)
                and baseline.get("contract") == heldout.get("contract")
                and baseline.get("pre") == heldout.get("pre"),
                "isolated SFT held-out baseline is not integrity-bound to pre evaluation",
            )
    else:
        check(heldout is None, "isolated SFT recorded unexpected held-out evaluation")
    return errors, model_parameter_scope


def _rl_metrics_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    accounting = metrics.get("rl_accounting")
    prompt = metrics.get("prompt_accounting")
    data = metrics.get("data")
    heldout = metrics.get("heldout_eval")
    prompt_summary = None
    if isinstance(prompt, Mapping):
        prompt_summary = {
            key: prompt.get(key)
            for key in (
                "selected_prompts",
                "selected_prompt_tokens",
                "rollout_prompt_tokens",
                "generated_tokens",
                "generated_eos_tokens",
                "truncated_rollouts",
                "informative_steps",
                "informative_scoring_input_slots",
                "model_forward_token_slots",
            )
        }
    data_summary = None
    if isinstance(data, Mapping):
        data_summary = {
            key: data.get(key)
            for key in (
                "paths",
                "eval_paths",
                "train_artifacts",
                "eval_artifacts",
                "conversation_prompt_contract",
                "parent_conversation_prompt_contract",
                "prompt_truncation",
                "preflight_minimum_coverage",
            )
        }
        for audit_name in ("split_audit", "selected_eval_split_audit"):
            audit = data.get(audit_name)
            if isinstance(audit, Mapping):
                data_summary[audit_name] = {
                    key: audit.get(key)
                    for key in (
                        "train_dataset_sha256",
                        "eval_dataset_sha256",
                        "train_scored_rows_sha256",
                        "eval_scored_rows_sha256",
                        "train_scored_prompts_sha256",
                        "eval_scored_prompts_sha256",
                        "row_overlap",
                        "prompt_overlap",
                    )
                }
    heldout_summary = None
    if isinstance(heldout, Mapping):
        heldout_summary = {
            key: heldout.get(key) for key in ("contract", "pre", "post", "delta")
        }
    return {
        "stage": metrics.get("stage"),
        "checkpoint": metrics.get("checkpoint"),
        "reward_steps": metrics.get("reward_steps"),
        "mean_reward_last": metrics.get("mean_reward_last"),
        "rl_accounting": dict(accounting) if isinstance(accounting, Mapping) else accounting,
        "prompt_accounting": prompt_summary,
        "lineage": metrics.get("lineage"),
        "execution": metrics.get("execution"),
        "data": data_summary,
        "heldout_eval": heldout_summary,
    }


def _inspect_rl_policy_transition(
    *,
    checkpoint: Mapping[str, Any],
    parent_checkpoint: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_config: ModelConfig,
    execution_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Compare exact policy tensors and the applied bounded-prefix LR sequence."""

    from localagent.train.rl import _rl_resume_sha256

    errors: list[str] = []
    model_names = _sft_model_parameter_names(model_config)
    parent_state = parent_checkpoint.get(
        "state_dict",
        parent_checkpoint.get("model"),
    )
    child_state = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(parent_state, Mapping):
        errors.append("RL SFT parent has no model state mapping")
        parent_state = {}
    if not isinstance(child_state, Mapping):
        errors.append("isolated RL checkpoint has no model state mapping")
        child_state = {}

    missing_parent_names = [name for name in model_names if name not in parent_state]
    missing_child_names = [name for name in model_names if name not in child_state]
    if missing_parent_names:
        errors.append(
            "RL SFT parent model state is missing named parameters: "
            + ", ".join(missing_parent_names)
        )
    if missing_child_names:
        errors.append(
            "isolated RL model state is missing named parameters: "
            + ", ".join(missing_child_names)
        )

    incompatible_names: list[str] = []
    for name in model_names:
        if name in missing_parent_names or name in missing_child_names:
            continue
        parent_tensor = parent_state[name]
        child_tensor = child_state[name]
        if (
            not isinstance(parent_tensor, torch.Tensor)
            or not isinstance(child_tensor, torch.Tensor)
            or parent_tensor.shape != child_tensor.shape
            or parent_tensor.dtype != child_tensor.dtype
        ):
            incompatible_names.append(name)
    if incompatible_names:
        errors.append(
            "isolated RL model tensors are incompatible with the SFT parent: "
            + ", ".join(incompatible_names)
        )

    excluded = set(
        [
            *missing_parent_names,
            *missing_child_names,
            *incompatible_names,
        ]
    )
    comparable_names = [name for name in model_names if name not in excluded]
    changed_names = [
        name
        for name in comparable_names
        if _rl_resume_sha256(parent_state[name])
        != _rl_resume_sha256(child_state[name])
    ]
    if len(comparable_names) != len(model_names):
        errors.append("isolated RL preflight could not compare every policy tensor")
    if not changed_names:
        errors.append(
            "isolated RL nonzero-LR prefix changed no policy model tensor"
        )

    initial_state_sha256 = None
    final_state_sha256 = None
    if len(comparable_names) == len(model_names):
        initial_state_sha256 = _rl_resume_sha256(
            {name: parent_state[name] for name in model_names}
        )
        final_state_sha256 = _rl_resume_sha256(
            {name: child_state[name] for name in model_names}
        )
        if (initial_state_sha256 != final_state_sha256) is not bool(changed_names):
            errors.append("isolated RL policy state digest/transition mismatch")

    accounting = metrics.get("rl_accounting")
    actual_learning_rates = (
        accounting.get("learning_rate_history")
        if isinstance(accounting, Mapping)
        else None
    )
    expected_learning_rates = list(execution_contract["expected_learning_rates"])
    valid_actual_learning_rates = (
        isinstance(actual_learning_rates, list)
        and len(actual_learning_rates) == len(expected_learning_rates)
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in actual_learning_rates
        )
    )
    normalized_actual_learning_rates = (
        [float(value) for value in actual_learning_rates]
        if valid_actual_learning_rates
        else []
    )
    learning_rates_match = valid_actual_learning_rates and all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0)
        for actual, expected in zip(
            normalized_actual_learning_rates,
            expected_learning_rates,
            strict=True,
        )
    )
    if not valid_actual_learning_rates:
        errors.append("isolated RL actual learning-rate sequence is invalid")
    elif not learning_rates_match:
        errors.append(
            "isolated RL actual learning-rate sequence drifted from production"
        )
    nonzero_learning_rate_executed = any(
        value > 0.0 for value in normalized_actual_learning_rates
    )
    if not nonzero_learning_rate_executed:
        errors.append("isolated RL prefix executed no nonzero learning rate")

    optimizer = checkpoint.get("optimizer")
    param_groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
    final_optimizer_learning_rates = (
        [
            float(group["lr"])
            for group in param_groups
            if isinstance(group, Mapping)
            and not isinstance(group.get("lr"), bool)
            and isinstance(group.get("lr"), (int, float))
            and math.isfinite(float(group["lr"]))
        ]
        if isinstance(param_groups, list)
        else []
    )
    last_expected_lr = expected_learning_rates[-1]
    final_optimizer_lr_matches = (
        isinstance(param_groups, list)
        and bool(param_groups)
        and len(final_optimizer_learning_rates) == len(param_groups)
        and all(
            math.isclose(value, last_expected_lr, rel_tol=0.0, abs_tol=0.0)
            for value in final_optimizer_learning_rates
        )
    )
    if not final_optimizer_lr_matches:
        errors.append(
            "isolated RL optimizer does not retain the final production-prefix learning rate"
        )

    evidence = {
        "contract": "exact_named_policy_parameter_comparison_v1",
        "model_named_parameter_names": model_names,
        "model_parameter_count": len(model_names),
        "compared_model_parameter_names": comparable_names,
        "compared_model_parameter_count": len(comparable_names),
        "changed_model_parameter_names": changed_names,
        "changed_model_parameter_count": len(changed_names),
        "first_changed_model_parameter": changed_names[0] if changed_names else None,
        "initial_model_state_sha256": initial_state_sha256,
        "final_model_state_sha256": final_state_sha256,
        "at_least_one_policy_tensor_changed": bool(changed_names),
        "production_schedule_total_steps": execution_contract[
            "production_schedule_total_steps"
        ],
        "execution_rollout_step_limit": execution_contract[
            "execution_rollout_step_limit"
        ],
        "first_nonzero_learning_rate_step": execution_contract[
            "first_nonzero_learning_rate_step"
        ],
        "expected_learning_rates": expected_learning_rates,
        "actual_learning_rates": normalized_actual_learning_rates,
        "actual_learning_rates_match_expected": learning_rates_match,
        "nonzero_learning_rate_executed": nonzero_learning_rate_executed,
        "final_optimizer_learning_rates": final_optimizer_learning_rates,
        "final_optimizer_learning_rate_matches_expected": final_optimizer_lr_matches,
    }
    return evidence, errors


def _validate_rl_preflight_outputs(
    *,
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    effective: Mapping[str, Any],
    model_config: ModelConfig,
    parent_checkpoint_sha256: str,
    parent_checkpoint: Mapping[str, Any],
    tokenizer_sha256: str,
    declared_data: list[dict[str, Any]],
    execution_contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(metrics.get("stage") == "rl", "isolated metrics stage is not 'rl'")
    try:
        recorded_checkpoint = Path(str(metrics.get("checkpoint"))).resolve(strict=False)
    except (TypeError, ValueError):
        recorded_checkpoint = Path()
        errors.append("isolated metrics checkpoint path is invalid")
    check(
        recorded_checkpoint == checkpoint_path.resolve(strict=False),
        "isolated metrics checkpoint path is not the redirected checkpoint",
    )
    execution_rollout_steps = int(
        execution_contract["execution_rollout_step_limit"]
    )
    check(
        metrics.get("reward_steps") == execution_rollout_steps,
        "isolated RL run did not complete the required nonzero-LR prefix",
    )
    check(checkpoint.get("stage") == "rl", "isolated checkpoint stage is not 'rl'")
    check(
        checkpoint.get("step") == execution_rollout_steps - 1,
        "isolated checkpoint did not stop at the bounded-prefix boundary",
    )
    policy_transition, transition_errors = _inspect_rl_policy_transition(
        checkpoint=checkpoint,
        parent_checkpoint=parent_checkpoint,
        metrics=metrics,
        model_config=model_config,
        execution_contract=execution_contract,
    )
    errors.extend(transition_errors)

    lineage = metrics.get("lineage")
    checkpoint_lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        errors.append("isolated metrics have no lineage mapping")
    else:
        expected_lineage = {
            "stage": "rl",
            "config_sha256": _lineage_config_sha256(effective),
            "model_config_sha256": canonical_sha256(model_config.__dict__),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
        }
        for key, expected in expected_lineage.items():
            check(lineage.get(key) == expected, f"isolated RL lineage {key} mismatch")
        check(
            _valid_sha256(lineage.get("data_sha256")),
            "isolated RL lineage data_sha256 is invalid",
        )
        check(
            dict(lineage) == checkpoint_lineage,
            "isolated metrics/checkpoint lineage mismatch",
        )

    data = metrics.get("data")
    checkpoint_data = checkpoint.get("data")
    if not isinstance(data, Mapping):
        errors.append("isolated metrics have no data lineage mapping")
    else:
        try:
            _assert_recorded_data_artifacts(data, declared_data)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        try:
            expected_data_sha256 = canonical_sha256(_rl_lineage_data_identity(data))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if isinstance(lineage, Mapping):
                check(
                    lineage.get("data_sha256") == expected_data_sha256,
                    "isolated RL data lineage hash does not match recorded artifacts/audits",
                )
        check(
            isinstance(checkpoint_data, Mapping)
            and canonical_sha256(data) == canonical_sha256(checkpoint_data),
            "isolated metrics/checkpoint data lineage mismatch",
        )
        evaluation = effective.get("evaluation")
        coverage = (
            evaluation.get("preflight_minimum_coverage")
            if isinstance(evaluation, Mapping)
            else None
        )
        if coverage is not None:
            check(
                data.get("preflight_minimum_coverage") == coverage,
                "isolated RL metrics minimum-coverage contract mismatch",
            )
            check(
                data.get("eval_selection") == coverage.get("selection_audit"),
                "isolated RL eval selection drifted from minimum-coverage derivation",
            )

    tokenizer = checkpoint.get("tokenizer")
    check(
        isinstance(tokenizer, Mapping) and tokenizer.get("sha256") == tokenizer_sha256,
        "isolated checkpoint tokenizer lineage mismatch",
    )

    execution = metrics.get("execution")
    check(
        isinstance(execution, Mapping) and checkpoint.get("execution") == execution,
        "isolated RL metrics/checkpoint execution metadata mismatch",
    )
    runtime = _mapping(effective.get("runtime", {}), label="runtime")
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    if isinstance(execution, Mapping):
        from localagent.train.device import (
            execution_metadata,
            resolve_device,
            resolve_dtype,
        )

        resolved_device = resolve_device(requested_device)
        resolved_dtype = resolve_dtype(resolved_device, requested_dtype)
        expected_execution = execution_metadata(
            requested_device=requested_device,
            resolved_device=resolved_device,
            requested_dtype=requested_dtype,
            resolved_dtype=resolved_dtype,
        )
        for key in (
            "requested_device",
            "resolved_device",
            "requested_dtype",
            "resolved_dtype",
        ):
            check(
                execution.get(key) == expected_execution[key],
                f"isolated RL execution {key} mismatch",
            )

    accounting = metrics.get("rl_accounting")
    prompt_accounting = metrics.get("prompt_accounting")
    rollout = effective.get("rollout")
    policy = effective.get("policy")
    if (
        not isinstance(accounting, Mapping)
        or not isinstance(prompt_accounting, Mapping)
        or not isinstance(rollout, Mapping)
        or not isinstance(policy, Mapping)
    ):
        errors.append("isolated RL accounting is incomplete")
        return errors, policy_transition

    prompts_per_step = int(rollout["prompts_per_step"])
    group_size = int(rollout["group_size"])
    attempted_groups = execution_rollout_steps * prompts_per_step
    attempted_rollouts = attempted_groups * group_size
    policy_epochs = int(policy["epochs_per_rollout"])
    check(
        accounting.get("attempted_rollout_steps") == execution_rollout_steps,
        "isolated RL accounting did not attempt the exact bounded prefix",
    )
    check(
        accounting.get("attempted_groups") == attempted_groups,
        "isolated RL accounting group count mismatch",
    )
    check(
        accounting.get("attempted_rollouts") == attempted_rollouts,
        "isolated RL accounting rollout count mismatch",
    )
    check(
        accounting.get("policy_epochs_per_informative_batch") == policy_epochs,
        "isolated RL accounting changed policy epochs",
    )
    zero_signal_steps = accounting.get("zero_signal_steps")
    informative_steps = (
        execution_rollout_steps - zero_signal_steps
        if isinstance(zero_signal_steps, int)
        and not isinstance(zero_signal_steps, bool)
        and 0 <= zero_signal_steps <= execution_rollout_steps
        else None
    )
    check(
        informative_steps is not None,
        "isolated RL zero-signal step accounting is invalid",
    )
    check(
        informative_steps is not None
        and accounting.get("realized_optimizer_updates")
        == informative_steps * policy_epochs,
        "isolated RL prefix did not exercise every informative policy epoch",
    )
    check(
        isinstance(accounting.get("realized_optimizer_updates"), int)
        and accounting["realized_optimizer_updates"] >= policy_epochs,
        "isolated RL prefix did not realize a nonzero-LR policy update",
    )
    check(
        checkpoint.get("rl_accounting") == accounting,
        "isolated RL metrics/checkpoint accounting mismatch",
    )

    observability = accounting.get("rollout_observability")
    if not isinstance(observability, Mapping):
        errors.append("isolated RL accounting has no rollout observability")
        return errors, policy_transition
    reward_observation = observability.get("reward")
    parsing = observability.get("parsing")
    truncation = observability.get("truncation")
    tokens = observability.get("tokens")
    if not all(
        isinstance(value, Mapping)
        for value in (reward_observation, parsing, truncation, tokens)
    ):
        errors.append("isolated RL rollout observability is incomplete")
        return errors, policy_transition

    distribution = reward_observation.get("distribution")
    if not isinstance(distribution, list):
        errors.append("isolated RL reward distribution is invalid")
    else:
        counts = [
            item.get("count")
            for item in distribution
            if isinstance(item, Mapping)
            and isinstance(item.get("count"), int)
            and not isinstance(item.get("count"), bool)
            and item.get("count") >= 0
        ]
        check(
            len(counts) == len(distribution) and sum(counts) == attempted_rollouts,
            "isolated RL reward distribution does not cover every rollout",
        )
        check(
            reward_observation.get("unique_values") == len(distribution),
            "isolated RL reward diversity count mismatch",
        )
        check(
            len(distribution) >= 2,
            "isolated RL rollouts had no reward diversity and could not exercise an update",
        )

    parse_count = parsing.get("parser_format_valid_rollouts")
    complete_parse_count = parsing.get("complete_parser_format_valid_rollouts")
    check(
        isinstance(parse_count, int) and 0 <= parse_count <= attempted_rollouts,
        "isolated RL parser success count is invalid",
    )
    check(
        isinstance(complete_parse_count, int)
        and isinstance(parse_count, int)
        and 0 <= complete_parse_count <= parse_count,
        "isolated RL complete parser success count is invalid",
    )
    truncated = truncation.get("truncated_rollouts")
    check(
        isinstance(truncated, int) and 0 <= truncated <= attempted_rollouts,
        "isolated RL truncation count is invalid",
    )
    generated = tokens.get("generated_tokens")
    generated_eos = tokens.get("generated_eos_tokens")
    check(
        isinstance(generated, int)
        and attempted_rollouts <= generated
        <= attempted_rollouts * int(rollout["max_new_tokens"]),
        "isolated RL generated token count is invalid",
    )
    check(
        isinstance(generated_eos, int) and 0 <= generated_eos <= attempted_rollouts,
        "isolated RL generated EOS count is invalid",
    )
    check(
        prompt_accounting.get("generated_tokens") == generated
        and prompt_accounting.get("generated_eos_tokens") == generated_eos
        and prompt_accounting.get("truncated_rollouts") == truncated,
        "isolated RL prompt/rollout accounting mismatch",
    )
    check(
        checkpoint.get("prompt_accounting") == prompt_accounting,
        "isolated RL metrics/checkpoint prompt accounting mismatch",
    )
    return errors, policy_transition


def run_one_update_pretrain_preflight(
    config_path: str | Path,
    *,
    work_dir: str | Path,
    receipt_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run one isolated pretraining update and publish a self-hashed receipt."""

    import yaml

    from localagent.train.pretrain import run as run_pretrain

    source_path = Path(config_path)
    work_path = Path(work_dir)
    destination = Path(receipt_path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing preflight receipt: {destination}")
    if work_path.exists():
        raise FileExistsError(f"preflight work directory already exists: {work_path}")

    source_raw = source_path.read_bytes()
    source_identity_before = file_identity(source_path)
    if source_identity_before["sha256"] != hashlib.sha256(source_raw).hexdigest():
        raise RuntimeError("training config changed while the preflight was reading it")
    source = yaml.safe_load(source_raw)
    source = _mapping(source, label="training config")
    model_path = Path(source["model_config"])
    model_identity_before = file_identity(model_path)
    model_config = ModelConfig.from_yaml(model_path)
    model_config.assert_within_budget()
    source_log = _mapping(source.get("log", {}), label="log")
    production_checkpoint = Path(source_log.get("out_dir", "runs/pretrain")) / "latest.pt"
    production_before = _optional_file_snapshot(production_checkpoint)

    effective = build_one_update_pretrain_config(
        source,
        work_dir=work_path,
        device=device,
    )
    work_path.mkdir(parents=True, exist_ok=False)
    effective_path = work_path / "effective.yaml"
    effective_path.write_text(
        yaml.safe_dump(effective, sort_keys=False),
        encoding="utf-8",
    )

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    baseline_rss = _rss_bytes()
    sampler = _ResourceSampler()
    sampler.start()
    error: Exception | None = None
    try:
        run_pretrain(str(effective_path), resume=False)
        _synchronize_accelerator()
    except Exception as exc:  # noqa: BLE001 - a failed run must still publish its receipt
        error = exc
    finally:
        sampler.stop()
    wall_seconds = time.perf_counter() - started
    finished_at = datetime.now(UTC).isoformat()

    production_after = _optional_file_snapshot(production_checkpoint)
    production_untouched = production_before == production_after
    source_identity_after = file_identity(source_path)
    model_identity_after = file_identity(model_path)
    source_config_untouched = source_identity_before == source_identity_after
    model_config_untouched = model_identity_before == model_identity_after
    metrics_path = work_path / "run" / "metrics.json"
    checkpoint_path = work_path / "run" / "latest.pt"
    metrics = None
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    source_artifacts_untouched = source_config_untouched and model_config_untouched
    status = (
        "passed"
        if error is None and production_untouched and source_artifacts_untouched
        else "failed"
    )
    token_accounting = metrics.get("token_accounting") if isinstance(metrics, Mapping) else None
    input_tokens = (
        int(token_accounting["input_tokens"])
        if isinstance(token_accounting, Mapping)
        else None
    )
    receipt: dict[str, Any] = {
        "kind": PREFLIGHT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source": {
            "config": {
                "path": str(source_path),
                **source_identity_before,
                "canonical_sha256": canonical_sha256(source),
            },
            "model_config": {
                "path": str(model_path),
                **model_identity_before,
                "canonical_sha256": canonical_sha256(model_config.__dict__),
            },
            "config_after": {"path": str(source_path), **source_identity_after},
            "model_config_after": {"path": str(model_path), **model_identity_after},
            "source_artifacts_untouched": source_artifacts_untouched,
            "production_checkpoint_before": production_before,
            "production_checkpoint_after": production_after,
            "production_checkpoint_untouched": production_untouched,
        },
        "effective": {
            "config": {
                "path": str(effective_path),
                **file_identity(effective_path),
                "canonical_sha256": canonical_sha256(effective),
            },
            "config_payload": effective,
            "contract": {
                "stage": "pretrain",
                "optimizer_updates": 1,
                "resume": False,
                "evaluation": "disabled",
                "checkpoint_output": "isolated_work_directory",
                "production_schedule_total_steps": int(source["schedule"]["total_steps"]),
                "micro_batch_size": int(source["batch"]["micro_batch_size"]),
                "grad_accum_steps": int(source["batch"]["grad_accum_steps"]),
            },
        },
        "model": {
            "name": model_config.name,
            "exact_parameters": model_config.estimate_params(),
            "max_seq_len": model_config.max_seq_len,
            "vocab_size": model_config.vocab_size,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "measurement": {
            "scope": "whole isolated runner including load, one update, and checkpoint publication",
            "wall_seconds": wall_seconds,
            "input_tokens_per_wall_second": (
                input_tokens / wall_seconds
                if input_tokens is not None and wall_seconds > 0
                else None
            ),
            "memory": sampler.as_dict(baseline_rss_bytes=baseline_rss),
        },
        "artifacts": {
            "checkpoint": (
                {"path": str(checkpoint_path), **file_identity(checkpoint_path)}
                if checkpoint_path.exists()
                else None
            ),
            "metrics": (
                {"path": str(metrics_path), **file_identity(metrics_path)}
                if metrics_path.exists()
                else None
            ),
        },
        "metrics": metrics,
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else (
                {
                    "type": "ProductionCheckpointMutation",
                    "message": "production checkpoint changed during isolated preflight",
                }
                if not production_untouched
                else (
                    {
                        "type": "SourceArtifactMutation",
                        "message": "source config or model config changed during preflight",
                    }
                    if not source_artifacts_untouched
                    else None
                )
            )
        ),
    }
    sealed = seal_preflight_receipt(receipt)
    _write_receipt(destination, sealed)
    if error is not None:
        raise RuntimeError(
            f"one-update preflight failed; receipt written to {destination}"
        ) from error
    if not production_untouched:
        raise RuntimeError(
            f"one-update preflight touched the production checkpoint; receipt: {destination}"
        )
    if not source_artifacts_untouched:
        raise RuntimeError(
            f"one-update preflight source artifacts changed during execution; receipt: {destination}"
        )
    return sealed


def run_one_update_sft_preflight(
    config_path: str | Path,
    *,
    work_dir: str | Path,
    receipt_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the bounded isolated SFT prefix needed to exercise the configured contract."""

    import yaml

    from localagent.train.replay_sampling import (
        PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
    )
    from localagent.train.sft import (
        _validate_parent_anchored_sampling_parent,
        _validate_sft_continuation_parent,
        _validated_completed_sft_parent,
        resolve_sft_continuation,
        run as run_sft,
    )
    from localagent.train.stage_data import load_stage_parent_checkpoint

    source_path = Path(config_path)
    work_path = Path(work_dir)
    destination = Path(receipt_path)
    receipt_temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing preflight receipt: {destination}")
    if receipt_temporary.exists() or receipt_temporary.is_symlink():
        raise FileExistsError(
            f"preflight receipt temporary path already exists: {receipt_temporary}"
        )
    if work_path.exists() or work_path.is_symlink():
        raise FileExistsError(f"preflight work directory already exists: {work_path}")

    source_raw = source_path.read_bytes()
    source_config_before = _required_regular_file_snapshot(
        source_path,
        label="SFT training config",
    )
    if source_config_before["sha256"] != hashlib.sha256(source_raw).hexdigest():
        raise RuntimeError("SFT training config changed while the preflight was reading it")
    source = _mapping(yaml.safe_load(source_raw), label="training config")
    effective = build_one_update_sft_config(
        source,
        work_dir=work_path,
        device=device,
    )

    raw_model_path = source.get("model_config")
    raw_parent_path = source.get("init_from")
    if not isinstance(raw_model_path, (str, Path)) or not str(raw_model_path):
        raise ValueError("model_config must be a non-empty path")
    if not isinstance(raw_parent_path, (str, Path)) or not str(raw_parent_path):
        raise ValueError("init_from must be a non-empty completed SFT checkpoint path")
    model_path = Path(raw_model_path)
    parent_path = Path(raw_parent_path)
    model_before = _required_regular_file_snapshot(model_path, label="SFT model config")
    parent_before = _required_regular_file_snapshot(
        parent_path,
        label="SFT continuation parent checkpoint",
    )
    model_config = ModelConfig.from_yaml(model_path)
    model_config.assert_within_budget()

    data_cfg = _mapping(source.get("data", {}), label="data")
    raw_sampling_config = data_cfg.get("sampling")
    sampling_config = (
        _mapping(raw_sampling_config, label="data.sampling")
        if raw_sampling_config is not None
        else None
    )
    parent_anchored_sampling = (
        isinstance(sampling_config, Mapping)
        and sampling_config.get("mode")
        == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    )
    tokenizer_cfg = _mapping(data_cfg.get("tokenizer", {"kind": "byte"}), label="data.tokenizer")
    tokenizer_kind = tokenizer_cfg.get("kind", "byte")
    if not isinstance(tokenizer_kind, str) or not tokenizer_kind:
        raise ValueError("data.tokenizer.kind must be non-empty text")
    raw_tokenizer_path = tokenizer_cfg.get("path")
    tokenizer_before = None
    tokenizer_path = None
    if raw_tokenizer_path is not None:
        if not isinstance(raw_tokenizer_path, (str, Path)) or not str(raw_tokenizer_path):
            raise ValueError("data.tokenizer.path must be a non-empty path when configured")
        tokenizer_path = Path(raw_tokenizer_path)
        tokenizer_before = _required_regular_file_snapshot(
            tokenizer_path,
            label="SFT tokenizer",
        )
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=model_config.vocab_size,
        path=tokenizer_path,
    )

    parent_checkpoint, validated_parent_sha256 = load_stage_parent_checkpoint(
        parent_path,
        stage="rl",
        requested_model_config=model_config,
        expected_tokenizer_sha256=str(tokenizer_lineage["sha256"]),
    )
    continuation = resolve_sft_continuation(source)
    if continuation is None:
        raise ValueError(
            "one-update SFT preflight requires a fresh-optimizer continuation contract"
        )
    if parent_anchored_sampling and "parent" not in continuation:
        raise ValueError(
            "parent-anchored SFT preflight requires continuation.parent"
        )
    if "parent" in continuation:
        parent_pins = _validate_sft_continuation_parent(
            parent_checkpoint,
            checkpoint_sha256=validated_parent_sha256,
            continuation=continuation,
        )
        parent_checkpoint = dict(parent_checkpoint)
    else:
        parent_checkpoint = dict(_validated_completed_sft_parent(parent_checkpoint))
        unpinned_training_contract = _mapping(
            parent_checkpoint.get("training_contract"),
            label="SFT continuation parent training contract",
        )
        unpinned_lm_sampling = _mapping(
            unpinned_training_contract.get("lm_sampling"),
            label="SFT continuation parent LM sampling contract",
        )
        unpinned_sampling_state = _mapping(
            parent_checkpoint.get("sampling_state"),
            label="SFT continuation parent sampling state",
        )
        completed_lm_cursor = unpinned_sampling_state.get("lm_cursor")
        if (
            isinstance(completed_lm_cursor, bool)
            or not isinstance(completed_lm_cursor, int)
            or completed_lm_cursor < 0
        ):
            raise ValueError("SFT continuation parent LM cursor is invalid")
        parent_pins = {
            "checkpoint_sha256": validated_parent_sha256,
            "resume_integrity_sha256": parent_checkpoint[
                "resume_integrity_sha256"
            ],
            "training_contract_sha256": canonical_sha256(
                unpinned_training_contract
            ),
            "lm_sampling_sha256": canonical_sha256(unpinned_lm_sampling),
            "completed_steps": unpinned_sampling_state["completed_steps"],
            "completed_lm_cursor": completed_lm_cursor,
        }
        # Legacy continuation configs did not declare a seal. Validate the observed six-field
        # receipt through the same exact helper while retaining ``seal_configured=false`` below.
        parent_pins = _validate_sft_continuation_parent(
            parent_checkpoint,
            checkpoint_sha256=validated_parent_sha256,
            continuation={**continuation, "parent": parent_pins},
        )
    parent_anchor_binding = (
        _validate_parent_anchored_sampling_parent(
            parent_checkpoint,
            sampling_config,
        )
        if parent_anchored_sampling and sampling_config is not None
        else None
    )
    if validated_parent_sha256 != parent_before["sha256"]:
        raise RuntimeError("SFT continuation parent changed during preflight validation")
    parent_training_contract = _mapping(
        parent_checkpoint.get("training_contract"),
        label="SFT continuation parent training contract",
    )
    parent_sampling_state = _mapping(
        parent_checkpoint.get("sampling_state"),
        label="SFT continuation parent sampling state",
    )
    parent_completion = {
        **parent_pins,
        "seal_configured": "parent" in continuation,
        "step": parent_checkpoint["step"],
        "planned_steps": parent_training_contract["steps"],
        "micro_batch_size": parent_training_contract["batch_size"],
        "grad_accum_steps": parent_training_contract["accum_steps"],
        "completed_microbatches": parent_sampling_state["completed_microbatches"],
        **(
            {"parent_anchor_binding": parent_anchor_binding}
            if parent_anchor_binding is not None
            else {}
        ),
    }

    declared_data_before = _declared_sft_data_snapshots(source)
    expected_data_identity, sampling_evidence = _derive_sft_data_identity_and_sampling(
        effective
    )
    if parent_anchor_binding is not None:
        expected_data_identity, sampling_evidence = (
            _bind_sft_parent_checkpoint_identity(
                expected_data_identity,
                sampling_evidence,
                parent_checkpoint_binding=parent_anchor_binding,
            )
        )
    execution_contract = _derive_sft_preflight_execution_contract(
        effective,
        expected_data_identity,
    )
    execution_update_limit = execution_contract[
        "execution_optimizer_update_limit"
    ]
    expected_data_sha256 = canonical_sha256(expected_data_identity)

    log = _mapping(source.get("log", {}), label="log")
    raw_production_out = log.get("out_dir", "runs/sft")
    if not isinstance(raw_production_out, (str, Path)) or not str(raw_production_out):
        raise ValueError("log.out_dir must be a non-empty path")
    production_out = Path(raw_production_out)
    production_before = _path_snapshot(production_out)
    source_files = [source_path, model_path, parent_path]
    if tokenizer_path is not None:
        source_files.append(tokenizer_path)
    for record in declared_data_before:
        source_files.extend(
            Path(record[key]["path"])
            for key in ("jsonl", "manifest", "generator_config")
            if key in record
        )
    _assert_sft_path_isolation(
        work_path=work_path,
        receipt_path=destination,
        production_out=production_out,
        source_files=source_files,
    )

    work_path.mkdir(parents=True, exist_ok=False)
    effective_path = work_path / "effective.yaml"
    effective_path.write_text(
        yaml.safe_dump(effective, sort_keys=False),
        encoding="utf-8",
    )
    effective_config_before = _required_regular_file_snapshot(
        effective_path,
        label="isolated effective SFT config",
    )

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    baseline_rss = _rss_bytes()
    sampler = _ResourceSampler()
    sampler.start()
    run_error: Exception | None = None
    try:
        run_sft(
            str(effective_path),
            resume=False,
            _max_optimizer_updates=execution_update_limit,
        )
        _synchronize_accelerator()
    except Exception as exc:  # noqa: BLE001 - failed runs still receive sealed evidence
        run_error = exc
    finally:
        sampler.stop()
    wall_seconds = time.perf_counter() - started
    finished_at = datetime.now(UTC).isoformat()

    production_after = _path_snapshot(production_out)
    source_config_after = _path_snapshot(source_path)
    model_after = _path_snapshot(model_path)
    parent_after = _path_snapshot(parent_path)
    tokenizer_after = _path_snapshot(tokenizer_path) if tokenizer_path is not None else None
    declared_data_after = _snapshots_after(declared_data_before)
    effective_config_after = _path_snapshot(effective_path)
    production_untouched = production_before == production_after
    parent_untouched = parent_before == parent_after
    data_artifacts_untouched = _records_untouched(
        declared_data_before,
        declared_data_after,
    )
    source_artifacts_untouched = (
        source_config_before == source_config_after
        and model_before == model_after
        and parent_untouched
        and tokenizer_before == tokenizer_after
        and data_artifacts_untouched
    )
    effective_config_untouched = effective_config_before == effective_config_after

    isolated_out = work_path / "run"
    metrics_path = isolated_out / "metrics.json"
    checkpoint_path = isolated_out / "latest.pt"
    metrics: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    metrics_artifact: dict[str, Any] | None = None
    checkpoint_artifact: dict[str, Any] | None = None
    validation_errors: list[str] = []
    model_parameter_scope: dict[str, Any] | None = None
    if metrics_path.exists() and metrics_path.is_file() and not metrics_path.is_symlink():
        try:
            metrics_raw = metrics_path.read_bytes()
            metrics_artifact = {
                "path": str(metrics_path),
                "bytes": len(metrics_raw),
                "sha256": hashlib.sha256(metrics_raw).hexdigest(),
            }
            metrics = _mapping(
                json.loads(metrics_raw),
                label="isolated SFT metrics",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            validation_errors.append(f"could not read isolated SFT metrics: {exc}")
    else:
        validation_errors.append("isolated SFT metrics artifact is missing or unsafe")
    if checkpoint_path.exists() and checkpoint_path.is_file() and not checkpoint_path.is_symlink():
        try:
            checkpoint_raw = checkpoint_path.read_bytes()
            checkpoint_artifact = {
                "path": str(checkpoint_path),
                "bytes": len(checkpoint_raw),
                "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
            }
            loaded_checkpoint = torch.load(
                io.BytesIO(checkpoint_raw),
                map_location="cpu",
                weights_only=True,
            )
            checkpoint = _mapping(loaded_checkpoint, label="isolated SFT checkpoint")
            checkpoint_artifact["resume_integrity_sha256"] = checkpoint.get(
                "resume_integrity_sha256"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            validation_errors.append(f"could not read isolated SFT checkpoint: {exc}")
    else:
        validation_errors.append("isolated SFT checkpoint artifact is missing or unsafe")

    if not effective_config_untouched:
        validation_errors.append("isolated effective SFT config changed during execution")
    if metrics is not None and checkpoint is not None:
        output_errors, model_parameter_scope = _validate_sft_preflight_outputs(
            metrics=metrics,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            effective=effective,
            model_config=model_config,
            parent_checkpoint=parent_checkpoint,
            parent_checkpoint_sha256=str(parent_before["sha256"]),
            tokenizer_sha256=str(tokenizer_lineage["sha256"]),
            expected_data_identity=expected_data_identity,
            sampling_evidence=sampling_evidence,
            execution_contract=execution_contract,
            declared_data=declared_data_before,
        )
        validation_errors.extend(output_errors)

    optimizer_steps: list[int] = []
    optimizer_learning_rates: list[float] = []
    if checkpoint is not None:
        try:
            optimizer_steps = _sft_optimizer_step_values(checkpoint)
        except (TypeError, ValueError):
            pass
        optimizer = checkpoint.get("optimizer")
        param_groups = optimizer.get("param_groups") if isinstance(optimizer, Mapping) else None
        if isinstance(param_groups, list):
            optimizer_learning_rates = [
                float(group["lr"])
                for group in param_groups
                if isinstance(group, Mapping)
                and not isinstance(group.get("lr"), bool)
                and isinstance(group.get("lr"), (int, float))
                and math.isfinite(float(group["lr"]))
            ]
    realized_updates = (
        execution_update_limit
        if optimizer_steps
        and all(step == execution_update_limit for step in optimizer_steps)
        else None
    )
    status = (
        "passed"
        if (
            run_error is None
            and production_untouched
            and source_artifacts_untouched
            and effective_config_untouched
            and not validation_errors
        )
        else "failed"
    )
    metrics_summary = _sft_metrics_summary(metrics) if metrics is not None else None
    token_accounting = metrics.get("token_accounting") if metrics is not None else None
    input_tokens = (
        token_accounting.get("input_tokens")
        if isinstance(token_accounting, Mapping)
        else None
    )
    micro_batch_size = int(source["batch"]["micro_batch_size"])
    grad_accum_steps = int(source["batch"]["grad_accum_steps"])
    effective_batch_size = micro_batch_size * grad_accum_steps
    evaluation = _mapping(effective.get("evaluation", {}), label="evaluation")
    pad_to_input_tokens = source["batch"].get("pad_to_input_tokens")
    evaluation_pad_to_input_tokens = evaluation.get("pad_to_input_tokens")
    source_order_prefix = (
        effective.get("data", {}).get("sampling") is None
        and not bool(effective.get("data", {}).get("shuffle", True))
    )
    expected_executed_learning_rates = _expected_sft_executed_learning_rates(
        effective,
        execution_update_limit=execution_update_limit,
    )
    any_executed_learning_rate_nonzero = any(
        learning_rate != 0.0 for learning_rate in expected_executed_learning_rates
    )
    checkpoint_sampling_state = (
        checkpoint.get("sampling_state")
        if isinstance(checkpoint, Mapping)
        else None
    )
    observed_completed_lm_cursor = (
        checkpoint_sampling_state.get("lm_cursor")
        if isinstance(checkpoint_sampling_state, Mapping)
        else None
    )
    receipt: dict[str, Any] = {
        "kind": PREFLIGHT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source": {
            "config": {
                **source_config_before,
                "canonical_sha256": canonical_sha256(source),
            },
            "config_after": source_config_after,
            "model_config": {
                **model_before,
                "canonical_sha256": canonical_sha256(model_config.__dict__),
            },
            "model_config_after": model_after,
            "sft_parent_checkpoint": {
                **parent_before,
                "completion": parent_completion,
            },
            "sft_parent_checkpoint_after": parent_after,
            "sft_parent_checkpoint_untouched": parent_untouched,
            "tokenizer": (
                {
                    **tokenizer_before,
                    "tokenizer_kind": tokenizer_kind,
                    "lineage_sha256": tokenizer_lineage["sha256"],
                }
                if tokenizer_before is not None
                else {
                    "path": None,
                    "tokenizer_kind": tokenizer_kind,
                    "lineage_sha256": tokenizer_lineage["sha256"],
                }
            ),
            "tokenizer_after": tokenizer_after,
            "data_artifacts": declared_data_before,
            "data_artifacts_after": declared_data_after,
            "data_artifacts_untouched": data_artifacts_untouched,
            "sft_data_lineage": {
                "identity": expected_data_identity,
                "sha256": expected_data_sha256,
            },
            **(
                {"sft_sampling_lineage": sampling_evidence}
                if sampling_evidence is not None
                else {}
            ),
            "source_artifacts_untouched": source_artifacts_untouched,
            "production_sft_output_before": production_before,
            "production_sft_output_after": production_after,
            "production_sft_output_untouched": production_untouched,
        },
        "effective": {
            "config": {
                **effective_config_before,
                "canonical_sha256": canonical_sha256(effective),
            },
            "config_after": effective_config_after,
            "config_untouched": effective_config_untouched,
            "config_payload": effective,
            "contract": {
                "stage": "sft",
                "optimizer_updates": execution_update_limit,
                "realized_optimizer_updates": realized_updates,
                "optimizer_parameter_step_values": sorted(set(optimizer_steps)),
                "optimizer_learning_rates": optimizer_learning_rates,
                "expected_step_zero_learning_rate": expected_executed_learning_rates[0],
                "expected_executed_learning_rates": expected_executed_learning_rates,
                "expected_last_executed_learning_rate": (
                    expected_executed_learning_rates[-1]
                ),
                "any_executed_learning_rate_nonzero": (
                    any_executed_learning_rate_nonzero
                ),
                "resume": False,
                "checkpoint_every": 1,
                "checkpoint_output": "isolated_work_directory",
                "production_schedule_total_steps": int(source["schedule"]["total_steps"]),
                **execution_contract,
                "expected_completed_lm_cursor": (
                    execution_contract["executed_lm_decisions"]
                    if execution_contract["executed_through_first_pulse"]
                    else None
                ),
                "observed_completed_lm_cursor": observed_completed_lm_cursor,
                "micro_batch_size": micro_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": effective_batch_size,
                "pad_to_input_tokens": pad_to_input_tokens,
                "continuation": copy.deepcopy(source["continuation"]),
                "parent_checkpoint_sha256": parent_before["sha256"],
                "parent_pins": parent_pins,
                "parent_anchor_binding": parent_anchor_binding,
                "model_parameter_scope": model_parameter_scope,
                "heldout_evaluation": {
                    "scope": "production_selector_scope_preserved",
                    "max_conversations": evaluation.get("max_conversations"),
                    "selection": evaluation.get("selection"),
                    "pad_to_input_tokens": evaluation_pad_to_input_tokens,
                    "pre_and_post_required": bool(
                        _optional_source_sequence(
                            effective.get("data", {}).get("eval_conversations"),
                            label="data.eval_conversations",
                        )
                    ),
                },
                "lm_sampling": (
                    metrics.get("lm_sampling") if metrics is not None else None
                ),
                **(
                    {
                        "exercised_lm_prefix": sampling_evidence[
                            "exercised_prefix"
                        ],
                        "production_lm_sampling_sha256": sampling_evidence[
                            "production"
                        ]["sampling_contract_sha256"],
                    }
                    if sampling_evidence is not None
                    else {}
                ),
                "source_order_prefix_rows": (
                    execution_update_limit * effective_batch_size
                    if source_order_prefix
                    else None
                ),
            },
        },
        "model": {
            "name": model_config.name,
            "exact_parameters": model_config.estimate_params(),
            "max_seq_len": model_config.max_seq_len,
            "vocab_size": model_config.vocab_size,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "measurement": {
            "scope": (
                "whole production SFT runner including parent/model load, full configured "
                "held-out pre/post evaluation, "
                f"{execution_update_limit} bounded LM optimizer update(s), and checkpoint "
                "publication"
            ),
            "interpretation": (
                (
                    "optimizer-boundary, frozen-state, and unfrozen model-transition proof"
                    if any_executed_learning_rate_nonzero
                    else "optimizer-boundary and frozen-state proof, not learning evidence; "
                    "the bounded executed prefix has only zero learning rates"
                )
            ),
            "non_learning_limitation": (
                model_parameter_scope.get("non_learning_limitation")
                if isinstance(model_parameter_scope, Mapping)
                else (
                    _SFT_ZERO_EXECUTED_LR_NON_LEARNING_LIMITATION
                    if not any_executed_learning_rate_nonzero
                    else None
                )
            ),
            "sampling_limitation": (
                (
                    "source-order measurement covers only the first effective-batch content; "
                    "the explicit fixed LM width bounds later language-model tensor shapes"
                    if pad_to_input_tokens is not None
                    else "source-order measurement covers only the first effective-batch "
                    "prefix and does not prove worst-case memory for later curriculum phases"
                )
                if source_order_prefix
                else None
            ),
            "wall_seconds": wall_seconds,
            "optimizer_updates_per_wall_second": (
                realized_updates / wall_seconds
                if isinstance(realized_updates, int) and wall_seconds > 0
                else None
            ),
            "input_tokens_per_wall_second": (
                input_tokens / wall_seconds
                if isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and wall_seconds > 0
                else None
            ),
            "memory": sampler.as_dict(baseline_rss_bytes=baseline_rss),
        },
        "artifacts": {
            "checkpoint": checkpoint_artifact,
            "metrics": metrics_artifact,
        },
        "metrics": metrics_summary,
        "validation_errors": validation_errors,
        "error": (
            {"type": type(run_error).__name__, "message": str(run_error)}
            if run_error is not None
            else (
                {
                    "type": "ProductionSFTOutputMutation",
                    "message": "production SFT output path changed during isolated preflight",
                }
                if not production_untouched
                else (
                    {
                        "type": "SourceArtifactMutation",
                        "message": "SFT source artifact(s) changed during isolated preflight",
                    }
                    if not source_artifacts_untouched
                    else (
                        {
                            "type": "SFTPreflightValidationError",
                            "message": "; ".join(validation_errors),
                        }
                        if validation_errors
                        else None
                    )
                )
            )
        ),
    }
    sealed = seal_preflight_receipt(receipt)
    _write_receipt(destination, sealed)
    if status != "passed":
        raise RuntimeError(f"one-update SFT preflight failed; receipt written to {destination}")
    return sealed


def run_one_update_rl_preflight(
    config_path: str | Path,
    *,
    work_dir: str | Path,
    receipt_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run an isolated RL prefix through the first nonzero-LR update boundary."""

    import yaml

    from localagent.train.rl import run as run_rl

    source_path = Path(config_path)
    work_path = Path(work_dir)
    destination = Path(receipt_path)
    receipt_temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing preflight receipt: {destination}")
    if receipt_temporary.exists() or receipt_temporary.is_symlink():
        raise FileExistsError(
            f"preflight receipt temporary path already exists: {receipt_temporary}"
        )
    if work_path.exists() or work_path.is_symlink():
        raise FileExistsError(f"preflight work directory already exists: {work_path}")

    source_raw = source_path.read_bytes()
    source_config_before = _required_regular_file_snapshot(
        source_path,
        label="RL training config",
    )
    if source_config_before["sha256"] != hashlib.sha256(source_raw).hexdigest():
        raise RuntimeError("RL training config changed while the preflight was reading it")
    source = _mapping(yaml.safe_load(source_raw), label="training config")
    if source.get("stage") != "rl":
        raise ValueError("one-update RL preflight requires stage 'rl'")

    raw_model_path = source.get("model_config")
    raw_parent_path = source.get("init_from")
    if not isinstance(raw_model_path, (str, Path)) or not str(raw_model_path):
        raise ValueError("model_config must be a non-empty path")
    if not isinstance(raw_parent_path, (str, Path)) or not str(raw_parent_path):
        raise ValueError("init_from must be a non-empty SFT checkpoint path")
    model_path = Path(raw_model_path)
    parent_path = Path(raw_parent_path)
    model_before = _required_regular_file_snapshot(model_path, label="RL model config")
    parent_before = _required_regular_file_snapshot(
        parent_path,
        label="RL SFT parent checkpoint",
    )
    parent_raw = parent_path.read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != parent_before["sha256"]:
        raise RuntimeError("RL SFT parent changed while the preflight was reading it")
    loaded_parent = torch.load(
        io.BytesIO(parent_raw),
        map_location="cpu",
        weights_only=True,
    )
    parent_checkpoint = _mapping(loaded_parent, label="RL SFT parent checkpoint")
    model_config = ModelConfig.from_yaml(model_path)
    model_config.assert_within_budget()

    data_cfg = _mapping(source.get("data", {}), label="data")
    tokenizer_cfg = _mapping(data_cfg.get("tokenizer", {"kind": "byte"}), label="data.tokenizer")
    tokenizer_kind = tokenizer_cfg.get("kind", "byte")
    if not isinstance(tokenizer_kind, str) or not tokenizer_kind:
        raise ValueError("data.tokenizer.kind must be non-empty text")
    raw_tokenizer_path = tokenizer_cfg.get("path")
    tokenizer_before = None
    tokenizer_path = None
    if raw_tokenizer_path is not None:
        if not isinstance(raw_tokenizer_path, (str, Path)) or not str(raw_tokenizer_path):
            raise ValueError("data.tokenizer.path must be a non-empty path when configured")
        tokenizer_path = Path(raw_tokenizer_path)
        tokenizer_before = _required_regular_file_snapshot(
            tokenizer_path,
            label="RL tokenizer",
        )
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=model_config.vocab_size,
        path=tokenizer_path,
    )
    declared_data_before = _declared_rl_data_snapshots(source)
    evaluation_coverage = derive_rl_eval_minimum_coverage(source)

    log = _mapping(source.get("log", {}), label="log")
    production_out = Path(log.get("out_dir", "runs/rl"))
    production_before = _path_snapshot(production_out)
    source_files = [source_path, model_path, parent_path]
    if tokenizer_path is not None:
        source_files.append(tokenizer_path)
    for record in declared_data_before:
        source_files.extend(
            Path(record[key]["path"])
            for key in ("jsonl", "manifest", "generator_config")
            if key in record
        )
    _assert_rl_path_isolation(
        work_path=work_path,
        receipt_path=destination,
        production_out=production_out,
        source_files=source_files,
    )

    effective = build_one_update_rl_config(
        source,
        work_dir=work_path,
        device=device,
        evaluation_coverage=evaluation_coverage,
    )
    execution_contract = _derive_rl_preflight_execution_contract(effective)
    work_path.mkdir(parents=True, exist_ok=False)
    effective_path = work_path / "effective.yaml"
    effective_path.write_text(
        yaml.safe_dump(effective, sort_keys=False),
        encoding="utf-8",
    )

    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    baseline_rss = _rss_bytes()
    sampler = _ResourceSampler()
    sampler.start()
    run_error: Exception | None = None
    try:
        run_rl(
            str(effective_path),
            resume=False,
            _execution_rollout_step_limit=execution_contract[
                "execution_rollout_step_limit"
            ],
        )
        _synchronize_accelerator()
    except Exception as exc:  # noqa: BLE001 - failed runs still receive sealed evidence
        run_error = exc
    finally:
        sampler.stop()
    wall_seconds = time.perf_counter() - started
    finished_at = datetime.now(UTC).isoformat()

    production_after = _path_snapshot(production_out)
    source_config_after = _path_snapshot(source_path)
    model_after = _path_snapshot(model_path)
    parent_after = _path_snapshot(parent_path)
    tokenizer_after = _path_snapshot(tokenizer_path) if tokenizer_path is not None else None
    declared_data_after = _snapshots_after(declared_data_before)
    production_untouched = production_before == production_after
    source_artifacts_untouched = (
        source_config_before == source_config_after
        and model_before == model_after
        and parent_before == parent_after
        and tokenizer_before == tokenizer_after
        and _records_untouched(declared_data_before, declared_data_after)
    )

    isolated_out = work_path / "run"
    metrics_path = isolated_out / "metrics.json"
    checkpoint_path = isolated_out / "latest.pt"
    metrics: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    policy_transition: dict[str, Any] | None = None
    validation_errors: list[str] = []
    if metrics_path.exists() and metrics_path.is_file() and not metrics_path.is_symlink():
        try:
            loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = _mapping(loaded_metrics, label="isolated RL metrics")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            validation_errors.append(f"could not read isolated RL metrics: {exc}")
    else:
        validation_errors.append("isolated RL metrics artifact is missing or unsafe")
    if checkpoint_path.exists() and checkpoint_path.is_file() and not checkpoint_path.is_symlink():
        try:
            checkpoint_raw = checkpoint_path.read_bytes()
            loaded_checkpoint = torch.load(
                io.BytesIO(checkpoint_raw),
                map_location="cpu",
                weights_only=True,
            )
            checkpoint = _mapping(loaded_checkpoint, label="isolated RL checkpoint")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            validation_errors.append(f"could not read isolated RL checkpoint: {exc}")
    else:
        validation_errors.append("isolated RL checkpoint artifact is missing or unsafe")

    if metrics is not None and checkpoint is not None:
        output_errors, policy_transition = _validate_rl_preflight_outputs(
            metrics=metrics,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            effective=effective,
            model_config=model_config,
            parent_checkpoint_sha256=str(parent_before["sha256"]),
            parent_checkpoint=parent_checkpoint,
            tokenizer_sha256=str(tokenizer_lineage["sha256"]),
            declared_data=declared_data_before,
            execution_contract=execution_contract,
        )
        validation_errors.extend(output_errors)

    status = (
        "passed"
        if (
            run_error is None
            and production_untouched
            and source_artifacts_untouched
            and not validation_errors
        )
        else "failed"
    )
    metrics_summary = _rl_metrics_summary(metrics) if metrics is not None else None
    accounting = metrics.get("rl_accounting") if metrics is not None else None
    rollout_observability = (
        accounting.get("rollout_observability")
        if isinstance(accounting, Mapping)
        else None
    )
    realized_updates = (
        accounting.get("realized_optimizer_updates")
        if isinstance(accounting, Mapping)
        else None
    )
    receipt: dict[str, Any] = {
        "kind": PREFLIGHT_KIND,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source": {
            "config": {
                **source_config_before,
                "canonical_sha256": canonical_sha256(source),
            },
            "config_after": source_config_after,
            "model_config": {
                **model_before,
                "canonical_sha256": canonical_sha256(model_config.__dict__),
            },
            "model_config_after": model_after,
            "sft_parent_checkpoint": parent_before,
            "sft_parent_checkpoint_after": parent_after,
            "tokenizer": (
                {
                    **tokenizer_before,
                    "tokenizer_kind": tokenizer_kind,
                    "lineage_sha256": tokenizer_lineage["sha256"],
                }
                if tokenizer_before is not None
                else {
                    "path": None,
                    "tokenizer_kind": tokenizer_kind,
                    "lineage_sha256": tokenizer_lineage["sha256"],
                }
            ),
            "tokenizer_after": tokenizer_after,
            "data_artifacts": declared_data_before,
            "data_artifacts_after": declared_data_after,
            "evaluation_minimum_coverage_derivation": evaluation_coverage,
            "source_artifacts_untouched": source_artifacts_untouched,
            "production_rl_output_before": production_before,
            "production_rl_output_after": production_after,
            "production_rl_output_untouched": production_untouched,
        },
        "effective": {
            "config": {
                "path": str(effective_path),
                **file_identity(effective_path),
                "canonical_sha256": canonical_sha256(effective),
            },
            "config_payload": effective,
            "contract": {
                "stage": "rl",
                "rollout_steps": execution_contract[
                    "execution_rollout_step_limit"
                ],
                "execution_rollout_step_limit": execution_contract[
                    "execution_rollout_step_limit"
                ],
                "first_nonzero_learning_rate_step": execution_contract[
                    "first_nonzero_learning_rate_step"
                ],
                "expected_learning_rates": execution_contract[
                    "expected_learning_rates"
                ],
                "configured_policy_epochs_preserved": int(
                    source["policy"]["epochs_per_rollout"]
                ),
                "realized_optimizer_updates": realized_updates,
                "resume": False,
                "checkpoint_output": "isolated_work_directory",
                "production_schedule_total_steps": execution_contract[
                    "production_schedule_total_steps"
                ],
                "prompts_per_step": int(source["rollout"]["prompts_per_step"]),
                "group_size": int(source["rollout"]["group_size"]),
                "max_new_tokens": int(source["rollout"]["max_new_tokens"]),
                "heldout_max_conversations": effective["evaluation"].get(
                    "max_conversations"
                ),
                "heldout_minimum_coverage": (
                    {
                        "derivation_sha256": evaluation_coverage["derivation_sha256"],
                        "minimum_coverage_rows": evaluation_coverage[
                            "minimum_coverage_rows"
                        ],
                        "mandatory_strata": evaluation_coverage["mandatory_strata"],
                        "selection_audit_sha256": evaluation_coverage["selection_audit"][
                            "audit_sha256"
                        ],
                    }
                    if evaluation_coverage is not None
                    else None
                ),
            },
        },
        "model": {
            "name": model_config.name,
            "exact_parameters": model_config.estimate_params(),
            "max_seq_len": model_config.max_seq_len,
            "vocab_size": model_config.vocab_size,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "measurement": {
            "scope": (
                "whole isolated RL runner including source/data validation, capped held-out "
                "evaluation, the first nonzero-LR production prefix, and checkpoint publication"
            ),
            "wall_seconds": wall_seconds,
            "optimizer_updates_per_wall_second": (
                realized_updates / wall_seconds
                if isinstance(realized_updates, int) and wall_seconds > 0
                else None
            ),
            "memory": sampler.as_dict(baseline_rss_bytes=baseline_rss),
            "rollout_observability": rollout_observability,
            "policy_transition": policy_transition,
        },
        "artifacts": {
            "checkpoint": (
                {"path": str(checkpoint_path), **file_identity(checkpoint_path)}
                if checkpoint is not None
                else None
            ),
            "metrics": (
                {"path": str(metrics_path), **file_identity(metrics_path)}
                if metrics is not None
                else None
            ),
        },
        "metrics": metrics_summary,
        "validation_errors": validation_errors,
        "error": (
            {"type": type(run_error).__name__, "message": str(run_error)}
            if run_error is not None
            else (
                {
                    "type": "ProductionRLOutputMutation",
                    "message": "production RL output path changed during isolated preflight",
                }
                if not production_untouched
                else (
                    {
                        "type": "SourceArtifactMutation",
                        "message": "RL source artifact(s) changed during isolated preflight",
                    }
                    if not source_artifacts_untouched
                    else (
                        {
                            "type": "RLPreflightValidationError",
                            "message": "; ".join(validation_errors),
                        }
                        if validation_errors
                        else None
                    )
                )
            )
        ),
    }
    sealed = seal_preflight_receipt(receipt)
    _write_receipt(destination, sealed)
    if status != "passed":
        raise RuntimeError(f"one-update RL preflight failed; receipt written to {destination}")
    return sealed


def run_one_update_training_preflight(
    config_path: str | Path,
    *,
    work_dir: str | Path,
    receipt_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Dispatch one isolated update to the stage-specific production runner."""

    import yaml

    source = _mapping(
        yaml.safe_load(Path(config_path).read_bytes()),
        label="training config",
    )
    stage = source.get("stage")
    if stage == "pretrain":
        return run_one_update_pretrain_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=receipt_path,
            device=device,
        )
    if stage == "sft":
        return run_one_update_sft_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=receipt_path,
            device=device,
        )
    if stage == "rl":
        return run_one_update_rl_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=receipt_path,
            device=device,
        )
    raise ValueError(f"one-update preflight does not support stage {stage!r}")
