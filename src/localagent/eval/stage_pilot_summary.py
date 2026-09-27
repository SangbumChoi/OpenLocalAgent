"""Deterministic validation and summarization for a bounded midtrain -> SFT -> RL pilot.

The summary is deliberately mechanical. It proves artifact pairing, lineage continuity, finite
reported values, realized accounting, frozen evaluation identity, and the structured-head
transition. It does not decide whether a metric delta is large enough to be useful.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from localagent.data.schema import Conversation, Role
from localagent.train.stage_data import canonical_sha256, sha256_file

STAGE_PILOT_SUMMARY_KIND = "localagent_stage_pilot_summary"
STAGE_PILOT_SUMMARY_SCHEMA_VERSION = 2
STAGE_ORDER = ("midtrain", "sft", "rl")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40,64}")
_SFT_STRUCTURED_HEADS = {
    "tool_pointer": ("tool_head", "ptr_head"),
    "route": ("route_head",),
    "dense_selector": ("dense_selector",),
}
_CHECKPOINT_STRUCTURED_HEADS = tuple(
    checkpoint_key
    for metric_keys in _SFT_STRUCTURED_HEADS.values()
    for checkpoint_key in metric_keys
)
_LIMITATIONS = (
    {
        "id": "single_seed",
        "statement": (
            "This bounded pilot contains one training seed and does not estimate variation "
            "across seeds."
        ),
    },
    {
        "id": "offline_canonical_reward",
        "statement": (
            "RL uses offline canonical exact-match rewards; it does not establish online task "
            "success."
        ),
    },
    {
        "id": "no_browsergym",
        "statement": "The three training stages do not execute BrowserGym tasks.",
    },
    {
        "id": "browser_action_result_separate",
        "statement": (
            "Any browser-action result is a separate evaluation artifact and is not inferred "
            "from this stage summary."
        ),
    },
    {
        "id": "artifact_identity_not_publication",
        "statement": (
            "Artifact paths and hashes identify files available to the local summarization run; "
            "they do not assert that checkpoints or datasets are published."
        ),
    },
)


@dataclass(frozen=True)
class StagePilotInput:
    """One metrics/checkpoint pair in the requested pilot stage order."""

    stage: str
    metrics_path: str | Path
    checkpoint_path: str | Path
    config_path: str | Path


@dataclass
class _LoadedStage:
    specification: StagePilotInput
    metrics_path: Path
    checkpoint_path: Path
    config_path: Path
    metrics_artifact: dict[str, int | str]
    checkpoint_artifact: dict[str, int | str]
    config_artifact: dict[str, int | str]
    metrics: dict[str, Any]
    checkpoint: Mapping[str, Any]
    config: dict[str, Any]
    lineage: dict[str, Any]
    execution: dict[str, Any]
    heldout_eval: dict[str, Any]
    model_config_sha256: str
    tokenizer_sha256: str

    @property
    def stage(self) -> str:
        return self.specification.stage


@dataclass(frozen=True)
class _EvalArtifact:
    path: Path
    identity: dict[str, Any]

    @property
    def key(self) -> tuple[int, str]:
        return int(self.identity["bytes"]), str(self.identity["sha256"])


def _artifact(path: str | Path, *, label: str) -> tuple[Path, dict[str, int | str]]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {source}")
    if not source.is_file():
        raise ValueError(f"{label} is missing or is not a file: {source}")
    return source, {
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _valid_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _integer(value: object, *, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _assert_finite_json(value: object, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, label=f"{label}[{index}]")


def _load_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, int | str], dict]:
    source, artifact = _artifact(path, label=label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    _assert_finite_json(payload, label=label)
    return source, artifact, payload


def _load_checkpoint(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, dict[str, int | str], Mapping[str, Any]]:
    source, artifact = _artifact(path, label=label)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} must contain a mapping: {source}")
    return source, artifact, checkpoint


def _load_config(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, dict[str, int | str], dict[str, Any]]:
    source, artifact = _artifact(path, label=label)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is invalid YAML: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {source}")
    _assert_finite_json(payload, label=label)
    return source, artifact, payload


def _artifact_record(
    reference: str | Path,
    identity: Mapping[str, int | str],
    *,
    availability: str = "local_file_verified",
) -> dict[str, Any]:
    return {
        "path": str(reference),
        "bytes": int(identity["bytes"]),
        "sha256": str(identity["sha256"]),
        "availability": availability,
    }


def _validate_lineage(
    lineage: Mapping[str, Any],
    *,
    stage: str,
    label: str,
) -> dict[str, Any]:
    if lineage.get("version") != 1:
        raise ValueError(f"{label}.version must be 1")
    if lineage.get("stage") != stage:
        raise ValueError(f"{label}.stage must be {stage!r}")
    for key in (
        "config_sha256",
        "model_config_sha256",
        "data_sha256",
        "tokenizer_sha256",
    ):
        _valid_sha256(lineage.get(key), label=f"{label}.{key}")
    if "parent_checkpoint_sha256" in lineage:
        _valid_sha256(
            lineage["parent_checkpoint_sha256"],
            label=f"{label}.parent_checkpoint_sha256",
        )

    git = _mapping(lineage.get("git"), label=f"{label}.git")
    commit = git.get("commit")
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"{label}.git.commit must be a lowercase Git object ID")
    if type(git.get("dirty")) is not bool:
        raise ValueError(f"{label}.git.dirty must be a boolean")
    _valid_sha256(
        git.get("worktree_sha256"),
        label=f"{label}.git.worktree_sha256",
    )
    if "repository_sha256" in git:
        _valid_sha256(
            git["repository_sha256"],
            label=f"{label}.git.repository_sha256",
        )
    _assert_finite_json(lineage, label=label)
    return copy.deepcopy(dict(lineage))


def _path_ends_with(path: Path, suffix: Path) -> bool:
    suffix_parts = tuple(part for part in suffix.parts if part not in ("", "."))
    return bool(suffix_parts) and path.parts[-len(suffix_parts) :] == suffix_parts


def _checkpoint_reference(
    metrics: Mapping[str, Any],
    *,
    metrics_path: Path,
    checkpoint_path: Path,
    checkpoint_artifact: Mapping[str, int | str],
    label: str,
) -> None:
    declared = metrics.get("checkpoint")
    recorded_identity = metrics.get("checkpoint_identity")
    if isinstance(declared, Mapping):
        recorded_identity = declared
        declared = declared.get("path")
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"{label}.checkpoint must name the paired checkpoint file")

    actual = checkpoint_path.resolve()
    reference = Path(declared)
    if reference.is_absolute():
        matches = reference.resolve() == actual
    else:
        candidates = {
            (Path.cwd() / reference).resolve(),
            (metrics_path.parent / reference).resolve(),
        }
        matches = actual in candidates or _path_ends_with(actual, reference)
    if not matches:
        raise ValueError(
            f"{label}.checkpoint does not reference the supplied checkpoint: {declared}"
        )

    if recorded_identity is not None:
        identity = _mapping(recorded_identity, label=f"{label}.checkpoint identity")
        if identity.get("sha256") != checkpoint_artifact["sha256"]:
            raise ValueError(f"{label} checkpoint SHA-256 does not match the supplied file")
        if identity.get("bytes") != checkpoint_artifact["bytes"]:
            raise ValueError(f"{label} checkpoint byte size does not match the supplied file")


def _checkpoint_config(checkpoint: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    config = checkpoint.get("cfg")
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "__dict__"):
        return vars(config)
    raise ValueError(f"{label} checkpoint has no model config mapping")


def _validate_heldout(
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    heldout = _mapping(metrics.get("heldout_eval"), label=f"{label}.heldout_eval")
    if checkpoint.get("heldout_eval") != heldout:
        raise ValueError(f"{label} metrics/checkpoint held-out evaluation mismatch")
    contract = _mapping(heldout.get("contract"), label=f"{label}.heldout_eval.contract")
    for section in ("pre", "post", "delta"):
        _mapping(heldout.get(section), label=f"{label}.heldout_eval.{section}")
    if not (
        contract.get("same_draws_pre_post") is True
        or contract.get("same_rows_pre_post") is True
    ):
        raise ValueError(f"{label} held-out evaluation does not freeze pre/post examples")
    _assert_finite_json(heldout, label=f"{label}.heldout_eval")
    return copy.deepcopy(dict(heldout))


def _load_stage(specification: StagePilotInput) -> _LoadedStage:
    label = specification.stage
    config_path, config_artifact, config = _load_config(
        specification.config_path,
        label=f"{label} config",
    )
    metrics_path, metrics_artifact, metrics = _load_json(
        specification.metrics_path,
        label=f"{label} metrics",
    )
    checkpoint_path, checkpoint_artifact, checkpoint = _load_checkpoint(
        specification.checkpoint_path,
        label=f"{label} checkpoint",
    )
    _checkpoint_reference(
        metrics,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        checkpoint_artifact=checkpoint_artifact,
        label=f"{label} metrics",
    )
    for source_name, payload in (("metrics", metrics), ("checkpoint", checkpoint)):
        if payload.get("stage") != label:
            raise ValueError(
                f"{label} {source_name} records stage {payload.get('stage')!r}"
            )

    metrics_lineage = _mapping(metrics.get("lineage"), label=f"{label} metrics lineage")
    checkpoint_lineage = _mapping(
        checkpoint.get("lineage"),
        label=f"{label} checkpoint lineage",
    )
    if metrics_lineage != checkpoint_lineage:
        raise ValueError(f"{label} metrics/checkpoint lineage mismatch")
    lineage = _validate_lineage(
        metrics_lineage,
        stage=label,
        label=f"{label} lineage",
    )
    if config.get("stage") != label:
        raise ValueError(f"{label} config records stage {config.get('stage')!r}")
    normalized_config = copy.deepcopy(config)
    runtime = normalized_config.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    observed_config_sha256 = canonical_sha256(normalized_config)
    if observed_config_sha256 != lineage["config_sha256"]:
        raise ValueError(f"{label} canonical config does not match lineage")
    model_config_sha256 = _valid_sha256(
        lineage.get("model_config_sha256"),
        label=f"{label} model_config_sha256",
    )
    observed_model_config_sha256 = canonical_sha256(
        _checkpoint_config(checkpoint, label=label)
    )
    if observed_model_config_sha256 != model_config_sha256:
        raise ValueError(f"{label} checkpoint model config does not match lineage")
    tokenizer_sha256 = _valid_sha256(
        lineage.get("tokenizer_sha256"),
        label=f"{label} tokenizer_sha256",
    )
    tokenizer = _mapping(
        checkpoint.get("tokenizer"),
        label=f"{label} checkpoint tokenizer",
    )
    if tokenizer.get("sha256") != tokenizer_sha256:
        raise ValueError(f"{label} checkpoint tokenizer does not match lineage")

    execution = _mapping(metrics.get("execution"), label=f"{label} metrics execution")
    if not execution:
        raise ValueError(f"{label} metrics execution record is empty")
    if checkpoint.get("execution") != execution:
        raise ValueError(f"{label} metrics/checkpoint execution mismatch")
    _assert_finite_json(execution, label=f"{label}.execution")

    heldout_eval = _validate_heldout(metrics, checkpoint, label=label)
    return _LoadedStage(
        specification=specification,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        metrics_artifact=metrics_artifact,
        checkpoint_artifact=checkpoint_artifact,
        config_artifact=config_artifact,
        metrics=metrics,
        checkpoint=checkpoint,
        config=config,
        lineage=lineage,
        execution=copy.deepcopy(dict(execution)),
        heldout_eval=heldout_eval,
        model_config_sha256=model_config_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )


def _validate_history(stage: _LoadedStage) -> dict[str, Any]:
    if stage.stage in {"midtrain", "sft"}:
        history_key = "loss_history"
        steps_key = "loss_steps"
        last_key = "loss_last"
        metric_name = "loss"
    else:
        history_key = "reward_history"
        steps_key = "reward_steps"
        last_key = "mean_reward_last"
        metric_name = "mean_reward"

    history = stage.checkpoint.get(history_key)
    if not isinstance(history, (list, tuple)) or not history:
        raise ValueError(f"{stage.stage} checkpoint {history_key} must be non-empty")
    values = [
        _finite_number(value, label=f"{stage.stage} {history_key}[{index}]")
        for index, value in enumerate(history)
    ]
    steps = _integer(
        stage.metrics.get(steps_key),
        label=f"{stage.stage} metrics {steps_key}",
        minimum=1,
    )
    if steps != len(values):
        raise ValueError(f"{stage.stage} metrics {steps_key} does not match checkpoint history")
    last = _finite_number(
        stage.metrics.get(last_key),
        label=f"{stage.stage} metrics {last_key}",
    )
    if last != values[-1]:
        raise ValueError(f"{stage.stage} metrics {last_key} does not match checkpoint history")
    if stage.stage == "midtrain" and stage.metrics.get("steps_completed") != steps:
        raise ValueError("midtrain steps_completed does not match its loss history")
    return {
        "metric": metric_name,
        "steps": steps,
        "last": last,
    }


def _validate_token_accounting(stage: _LoadedStage) -> dict[str, Any]:
    accounting = _mapping(
        stage.metrics.get("token_accounting"),
        label=f"{stage.stage} metrics token_accounting",
    )
    if stage.checkpoint.get("token_accounting") != accounting:
        raise ValueError(f"{stage.stage} metrics/checkpoint token accounting mismatch")
    totals = {}
    for key in ("input_tokens", "loss_tokens"):
        totals[key] = _integer(
            accounting.get(key),
            label=f"{stage.stage} token_accounting.{key}",
            minimum=1,
        )
    sources = _mapping(
        accounting.get("sources"),
        label=f"{stage.stage} token_accounting.sources",
    )
    if not sources:
        raise ValueError(f"{stage.stage} token accounting has no sources")
    observed_totals = {"input_tokens": 0, "loss_tokens": 0}
    for source_name, source_value in sources.items():
        source = _mapping(
            source_value,
            label=f"{stage.stage} token_accounting.sources.{source_name}",
        )
        for key in observed_totals:
            observed_totals[key] += _integer(
                source.get(key),
                label=f"{stage.stage} token_accounting.sources.{source_name}.{key}",
                minimum=0,
            )
    if observed_totals != totals:
        raise ValueError(f"{stage.stage} token-accounting source totals do not add up")
    return copy.deepcopy(dict(accounting))


def _validate_rl_accounting(stage: _LoadedStage) -> tuple[dict[str, Any], str]:
    accounting = _mapping(
        stage.metrics.get("rl_accounting"),
        label="rl metrics rl_accounting",
    )
    if stage.checkpoint.get("rl_accounting") != accounting:
        raise ValueError("rl metrics/checkpoint accounting mismatch")
    attempted_rollouts = _integer(
        accounting.get("attempted_rollouts"),
        label="rl attempted_rollouts",
        minimum=1,
    )
    realized_updates = _integer(
        accounting.get("realized_optimizer_updates"),
        label="rl realized_optimizer_updates",
        minimum=0,
    )
    for key in ("attempted_rollout_steps", "attempted_groups"):
        if key in accounting:
            _integer(accounting[key], label=f"rl {key}", minimum=1)
    classification = "zero_signal" if realized_updates == 0 else "optimizer_updates_realized"
    result = copy.deepcopy(dict(accounting))
    result["attempted_rollouts"] = attempted_rollouts
    result["realized_optimizer_updates"] = realized_updates
    return result, classification


def _validate_rl_contract(stage: _LoadedStage) -> dict[str, Any]:
    reward_contract = _mapping(
        stage.metrics.get("reward_contract"),
        label="rl metrics reward_contract",
    )
    if stage.checkpoint.get("reward_contract") != reward_contract:
        raise ValueError("rl metrics/checkpoint reward contract mismatch")
    if reward_contract.get("environment") != "canonical_toolcalls":
        raise ValueError("rl reward environment is not offline canonical_toolcalls")
    if reward_contract.get("learned_judge") is not False:
        raise ValueError("rl reward contract must explicitly disable learned judges")
    policy_contract = _mapping(
        stage.metrics.get("policy_contract"),
        label="rl metrics policy_contract",
    )
    if stage.checkpoint.get("policy_contract") != policy_contract:
        raise ValueError("rl metrics/checkpoint policy contract mismatch")
    return {
        "reward": copy.deepcopy(dict(reward_contract)),
        "policy": copy.deepcopy(dict(policy_contract)),
    }


def _validate_structured_head_transition(
    sft: _LoadedStage,
    rl: _LoadedStage,
) -> dict[str, Any]:
    sft_metrics = _mapping(
        sft.metrics.get("structured_heads"),
        label="sft metrics structured_heads",
    )
    for summary_name, checkpoint_keys in _SFT_STRUCTURED_HEADS.items():
        if sft_metrics.get(summary_name) is not True:
            raise ValueError(f"sft structured head {summary_name} is not available")
        for checkpoint_key in checkpoint_keys:
            state = sft.checkpoint.get(checkpoint_key)
            if not isinstance(state, Mapping) or not state:
                raise ValueError(f"sft checkpoint has no {checkpoint_key} state")

    if rl.metrics.get("structured_heads_available") is not False:
        raise ValueError("rl metrics must explicitly mark structured heads unavailable")
    if rl.checkpoint.get("structured_heads_available") is not False:
        raise ValueError("rl checkpoint must explicitly mark structured heads unavailable")
    metrics_invalidated = rl.metrics.get("invalidated_structured_heads")
    checkpoint_invalidated = rl.checkpoint.get("invalidated_structured_heads")
    if metrics_invalidated != checkpoint_invalidated:
        raise ValueError("rl metrics/checkpoint structured-head invalidation mismatch")
    if not isinstance(metrics_invalidated, list) or len(set(metrics_invalidated)) != len(
        metrics_invalidated
    ):
        raise ValueError("rl invalidated_structured_heads must be a unique list")
    if set(metrics_invalidated) != set(_CHECKPOINT_STRUCTURED_HEADS):
        raise ValueError("rl does not explicitly invalidate every available SFT structured head")
    retained = [
        key
        for key in _CHECKPOINT_STRUCTURED_HEADS
        if rl.checkpoint.get(key) is not None
    ]
    if retained:
        raise ValueError("rl checkpoint retains invalidated structured heads: " + ", ".join(retained))
    return {
        "sft_available": copy.deepcopy(dict(sft_metrics)),
        "rl_structured_heads_available": False,
        "rl_invalidated": list(metrics_invalidated),
    }


def _stage_data(stage: _LoadedStage) -> Mapping[str, Any]:
    checkpoint_data = _mapping(
        stage.checkpoint.get("data"),
        label=f"{stage.stage} checkpoint data",
    )
    if "data" in stage.metrics and stage.metrics["data"] != checkpoint_data:
        raise ValueError(f"{stage.stage} metrics/checkpoint data metadata mismatch")
    return checkpoint_data


def _resolve_recorded_file(reference: str, *, stage: _LoadedStage, label: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        candidates = [path]
    else:
        bases = [Path.cwd(), *stage.metrics_path.resolve().parents]
        candidates = [base / path for base in bases]
    matches = []
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if not matches:
        raise ValueError(f"{label} is missing or is not a file: {reference}")
    if len(matches) > 1:
        raise ValueError(f"{label} resolves to multiple files: {reference}")
    if any(candidate.is_symlink() for candidate in candidates if candidate.exists()):
        raise ValueError(f"{label} must not be a symbolic link: {reference}")
    return matches[0]


def _resolve_recorded_directory(
    reference: str,
    *,
    stage: _LoadedStage,
    label: str,
) -> Path:
    path = Path(reference)
    if path.is_absolute():
        candidates = [path]
    else:
        bases = [Path.cwd(), stage.config_path.resolve().parent, *stage.metrics_path.resolve().parents]
        candidates = [base / path for base in bases]
    matches = []
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if not matches:
        raise ValueError(f"{label} is missing or is not a directory: {reference}")
    if len(matches) > 1:
        raise ValueError(f"{label} resolves to multiple directories: {reference}")
    if any(candidate.is_symlink() for candidate in candidates if candidate.exists()):
        raise ValueError(f"{label} must not be a symbolic link: {reference}")
    return matches[0]


def _configured_paths(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    paths = []
    for index, path in enumerate(value):
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError(f"{label}[{index}] must be a path")
        paths.append(str(path))
    return paths


def _conversation_data_artifact(
    reference: str,
    *,
    stage: _LoadedStage,
    role: str,
    name: str | None = None,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    path = _resolve_recorded_file(
        reference,
        stage=stage,
        label=f"{stage.stage} {role} conversation artifact",
    )
    identity = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    record: dict[str, Any] = {
        **_artifact_record(reference, identity),
        "role": role,
        "type": "conversations",
    }
    if name is not None:
        record["name"] = name
    return record, identity


def _shard_data_artifact(
    reference: str,
    *,
    stage: _LoadedStage,
    role: str,
    name: str,
    split: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    root = _resolve_recorded_directory(
        reference,
        stage=stage,
        label=f"{stage.stage} {role} shard root",
    )
    manifest_path = root / "manifest.json"
    _, identity, manifest = _load_json(
        manifest_path,
        label=f"{stage.stage} {role} shard manifest",
    )
    if split not in _mapping(manifest.get("splits"), label="shard manifest.splits"):
        raise ValueError(f"{stage.stage} shard manifest has no {split!r} split")
    manifest_sha256 = canonical_sha256(manifest)
    record = {
        **_artifact_record(str(Path(reference) / "manifest.json"), identity),
        "role": role,
        "name": name,
        "type": "packed_shard_manifest",
        "source_path": reference,
        "split": split,
        "manifest_sha256": manifest_sha256,
    }
    return record, {"manifest_sha256": manifest_sha256, "split": split}


def _assert_recorded_paths(
    recorded: Mapping[str, Any],
    key: str,
    configured: Sequence[str],
    *,
    label: str,
) -> None:
    raw = recorded.get(key, [])
    if not isinstance(raw, list) or raw != list(configured):
        raise ValueError(f"{label}.{key} does not match the canonical config")


def _midtrain_data_provenance(
    stage: _LoadedStage,
    data_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recorded = _stage_data(stage)
    artifacts: list[dict[str, Any]] = []
    source_identities = []
    eval_source_identities = []

    configured_sources = data_config.get("sources")
    if not isinstance(configured_sources, list) or not configured_sources:
        raise ValueError("midtrain config data.sources must be a non-empty list")
    recorded_sources = recorded.get("sources")
    if not isinstance(recorded_sources, list) or len(recorded_sources) != len(
        configured_sources
    ):
        raise ValueError("midtrain checkpoint data.sources does not match the canonical config")
    for index, (raw_config, raw_recorded) in enumerate(
        zip(configured_sources, recorded_sources, strict=True)
    ):
        source = _mapping(raw_config, label=f"midtrain config data.sources[{index}]")
        metadata = _mapping(raw_recorded, label=f"midtrain data.sources[{index}]")
        name = source.get("name")
        source_type = source.get("type")
        reference = source.get("path")
        if not all(isinstance(value, str) and value for value in (name, source_type, reference)):
            raise ValueError(f"midtrain config data.sources[{index}] is incomplete")
        expected_metadata = {
            "name": name,
            "type": source_type,
            "path": reference,
            "split": source.get("split"),
        }
        if dict(metadata) != expected_metadata:
            raise ValueError(
                f"midtrain checkpoint data.sources[{index}] does not match the canonical config"
            )
        if source_type == "shards":
            split = str(source.get("split", "train"))
            artifact, identity = _shard_data_artifact(
                reference,
                stage=stage,
                role="train",
                name=name,
                split=split,
            )
        elif source_type == "conversations":
            artifact, identity = _conversation_data_artifact(
                reference,
                stage=stage,
                role="train",
                name=name,
            )
        else:
            raise ValueError(f"unknown midtrain source type {source_type!r}")
        artifacts.append(artifact)
        start_weight = _finite_number(
            source.get("weight"),
            label=f"midtrain config data.sources[{index}].weight",
        )
        end_weight = _finite_number(
            source.get("end_weight", start_weight),
            label=f"midtrain config data.sources[{index}].end_weight",
        )
        source_identities.append(
            {
                "name": name,
                "type": source_type,
                "split": source.get("split"),
                "start_weight": start_weight,
                "end_weight": end_weight,
                "artifact": identity,
            }
        )

    configured_eval = data_config.get("eval_sources")
    if not isinstance(configured_eval, list) or not configured_eval:
        raise ValueError("midtrain config data.eval_sources must be a non-empty list")
    recorded_eval = recorded.get("eval_sources")
    if not isinstance(recorded_eval, list) or len(recorded_eval) != len(configured_eval):
        raise ValueError(
            "midtrain checkpoint data.eval_sources does not match the canonical config"
        )
    for index, (raw_config, raw_recorded) in enumerate(
        zip(configured_eval, recorded_eval, strict=True)
    ):
        source = _mapping(raw_config, label=f"midtrain config data.eval_sources[{index}]")
        metadata = _mapping(raw_recorded, label=f"midtrain data.eval_sources[{index}]")
        name = source.get("name")
        source_type = source.get("type")
        reference = source.get("path")
        if not all(isinstance(value, str) and value for value in (name, source_type, reference)):
            raise ValueError(f"midtrain config data.eval_sources[{index}] is incomplete")
        expected_metadata = {
            "name": name,
            "type": source_type,
            "path": reference,
            "split": source.get("split"),
        }
        if dict(metadata) != expected_metadata:
            raise ValueError(
                "midtrain checkpoint "
                f"data.eval_sources[{index}] does not match the canonical config"
            )
        if source_type == "shards":
            split = str(source.get("split", "val"))
            artifact, identity = _shard_data_artifact(
                reference,
                stage=stage,
                role="eval",
                name=name,
                split=split,
            )
        elif source_type == "conversations":
            artifact, identity = _conversation_data_artifact(
                reference,
                stage=stage,
                role="eval",
                name=name,
            )
        else:
            raise ValueError(f"unknown midtrain eval source type {source_type!r}")
        artifacts.append(artifact)
        eval_source_identities.append(
            {
                "name": name,
                "type": source_type,
                "split": source.get("split"),
                "artifact": identity,
            }
        )
    packed_holdout_audit = _mapping(
        recorded.get("packed_holdout_audit"),
        label="midtrain data.packed_holdout_audit",
    )
    return artifacts, {
        "sources": source_identities,
        "eval_sources": eval_source_identities,
        "packed_holdout_audit": copy.deepcopy(dict(packed_holdout_audit)),
    }


def _sft_data_provenance(
    stage: _LoadedStage,
    data_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recorded = _stage_data(stage)
    keys = (
        ("conversations", "paths", "train"),
        ("eval_conversations", "eval_paths", "eval"),
        ("decay_conversations", "decay_paths", "decay"),
    )
    artifacts = []
    identity: dict[str, Any] = {}
    for config_key, metadata_key, role in keys:
        paths = _configured_paths(
            data_config.get(config_key, []),
            label=f"sft config data.{config_key}",
        )
        _assert_recorded_paths(
            recorded,
            metadata_key,
            paths,
            label="sft checkpoint data",
        )
        identities = []
        for reference in paths:
            artifact, file_identity_value = _conversation_data_artifact(
                reference,
                stage=stage,
                role=role,
            )
            artifacts.append(artifact)
            identities.append(file_identity_value)
        identity[config_key] = identities
    return artifacts, identity


def _rl_data_provenance(
    stage: _LoadedStage,
    data_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recorded = _stage_data(stage)
    artifacts = []
    data_identity: dict[str, Any] = {}
    for config_key, metadata_key, artifact_key, role in (
        ("conversations", "paths", "train_artifacts", "train"),
        ("eval_conversations", "eval_paths", "eval_artifacts", "eval"),
    ):
        paths = _configured_paths(
            data_config.get(config_key, []),
            label=f"rl config data.{config_key}",
        )
        _assert_recorded_paths(
            recorded,
            metadata_key,
            paths,
            label="rl checkpoint data",
        )
        recorded_artifacts = recorded.get(artifact_key)
        if not isinstance(recorded_artifacts, list) or len(recorded_artifacts) != len(paths):
            raise ValueError(f"rl checkpoint data.{artifact_key} is incomplete")
        identities = []
        for index, (reference, raw_recorded) in enumerate(
            zip(paths, recorded_artifacts, strict=True)
        ):
            artifact, observed = _conversation_data_artifact(
                reference,
                stage=stage,
                role=role,
            )
            expected = _mapping(
                raw_recorded,
                label=f"rl checkpoint data.{artifact_key}[{index}]",
            )
            canonical_record = {"path": reference, **observed}
            if dict(expected) != canonical_record:
                raise ValueError(
                    f"rl checkpoint data.{artifact_key}[{index}] identity mismatch"
                )
            artifacts.append(artifact)
            identities.append(canonical_record)
        data_identity[artifact_key] = identities
    split_audit = _mapping(recorded.get("split_audit"), label="rl data.split_audit")
    data_identity["split_audit"] = copy.deepcopy(dict(split_audit))
    return artifacts, data_identity


def _stage_input_provenance(stage: _LoadedStage) -> dict[str, Any]:
    data_config = _mapping(
        stage.config.get("data"),
        label=f"{stage.stage} config data",
    )
    if stage.stage == "midtrain":
        data_artifacts, data_identity = _midtrain_data_provenance(stage, data_config)
    elif stage.stage == "sft":
        data_artifacts, data_identity = _sft_data_provenance(stage, data_config)
    else:
        data_artifacts, data_identity = _rl_data_provenance(stage, data_config)
    observed_data_sha256 = canonical_sha256(data_identity)
    if observed_data_sha256 != stage.lineage["data_sha256"]:
        raise ValueError(f"{stage.stage} canonical data identity does not match lineage")

    tokenizer_config = _mapping(
        data_config.get("tokenizer"),
        label=f"{stage.stage} config tokenizer",
    )
    tokenizer_checkpoint = _mapping(
        stage.checkpoint.get("tokenizer"),
        label=f"{stage.stage} checkpoint tokenizer",
    )
    tokenizer_kind = tokenizer_config.get("kind", "byte")
    if tokenizer_checkpoint.get("kind") != tokenizer_kind:
        raise ValueError(f"{stage.stage} tokenizer kind does not match the canonical config")
    tokenizer_path = tokenizer_config.get("path")
    if tokenizer_checkpoint.get("path") != tokenizer_path:
        raise ValueError(f"{stage.stage} tokenizer path does not match the canonical config")
    tokenizer_record: dict[str, Any] = {
        "kind": tokenizer_kind,
        "sha256": stage.tokenizer_sha256,
    }
    if tokenizer_path is None:
        tokenizer_record["availability"] = "implementation_identity_only"
    else:
        if not isinstance(tokenizer_path, str) or not tokenizer_path:
            raise ValueError(f"{stage.stage} tokenizer path is invalid")
        tokenizer_file = _resolve_recorded_file(
            tokenizer_path,
            stage=stage,
            label=f"{stage.stage} tokenizer artifact",
        )
        tokenizer_identity = {
            "bytes": tokenizer_file.stat().st_size,
            "sha256": sha256_file(tokenizer_file),
        }
        if tokenizer_identity["sha256"] != stage.tokenizer_sha256:
            raise ValueError(f"{stage.stage} tokenizer artifact does not match lineage")
        tokenizer_record.update(_artifact_record(tokenizer_path, tokenizer_identity))

    parent_reference = stage.config.get("init_from")
    parent_sha256 = stage.lineage.get("parent_checkpoint_sha256")
    if not isinstance(parent_reference, str) or not parent_reference:
        raise ValueError(f"{stage.stage} config init_from is missing")
    if parent_sha256 is None:
        raise ValueError(f"{stage.stage} lineage parent_checkpoint_sha256 is missing")
    try:
        parent_path = _resolve_recorded_file(
            parent_reference,
            stage=stage,
            label=f"{stage.stage} parent checkpoint",
        )
    except ValueError as error:
        if "is missing or is not a file" not in str(error):
            raise
        parent_record = {
            "path": parent_reference,
            "sha256": parent_sha256,
            "availability": "identity_only_file_not_available",
        }
    else:
        parent_identity = {
            "bytes": parent_path.stat().st_size,
            "sha256": sha256_file(parent_path),
        }
        if parent_identity["sha256"] != parent_sha256:
            raise ValueError(f"{stage.stage} parent checkpoint does not match lineage")
        parent_record = _artifact_record(parent_reference, parent_identity)

    return {
        "config": {
            **_artifact_record(stage.specification.config_path, stage.config_artifact),
            "canonical_sha256": stage.lineage["config_sha256"],
        },
        "tokenizer": tokenizer_record,
        "parent_checkpoint": parent_record,
        "data": {
            "canonical_data_sha256": stage.lineage["data_sha256"],
            "artifact_set_sha256": canonical_sha256(data_artifacts),
            "artifacts": data_artifacts,
        },
    }


def _recorded_eval_identities(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = data.get("eval_artifacts", [])
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise ValueError("data.eval_artifacts must be a list")
    identities = {}
    for index, value in enumerate(raw):
        artifact = _mapping(value, label=f"data.eval_artifacts[{index}]")
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"data.eval_artifacts[{index}].path is missing")
        if path in identities:
            raise ValueError(f"data.eval_artifacts repeats {path!r}")
        identities[path] = artifact
    return identities


def _eval_references(stage: _LoadedStage) -> list[tuple[str, Mapping[str, Any] | None]]:
    data = _stage_data(stage)
    recorded = _recorded_eval_identities(data)
    if stage.stage == "midtrain":
        eval_sources = data.get("eval_sources")
        if not isinstance(eval_sources, list):
            raise ValueError("midtrain data.eval_sources must be a list")
        names = []
        references = []
        for index, value in enumerate(eval_sources):
            source = _mapping(value, label=f"midtrain data.eval_sources[{index}]")
            name = source.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"midtrain data.eval_sources[{index}].name is missing")
            names.append(name)
            if source.get("type") != "conversations":
                continue
            path = source.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"midtrain data.eval_sources[{index}].path is missing")
            artifact = source.get("artifact")
            references.append(
                (
                    path,
                    _mapping(artifact, label=f"midtrain eval artifact {path}")
                    if artifact is not None
                    else recorded.get(path),
                )
            )
        contract_sources = stage.heldout_eval["contract"].get("sources")
        if contract_sources != names:
            raise ValueError("midtrain held-out contract does not match configured eval sources")
    else:
        paths = data.get("eval_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{stage.stage} data.eval_paths must be a non-empty list")
        references = []
        for index, path in enumerate(paths):
            if not isinstance(path, str) or not path:
                raise ValueError(f"{stage.stage} data.eval_paths[{index}] is missing")
            references.append((path, recorded.get(path)))
        if recorded and set(recorded) != set(paths):
            raise ValueError(f"{stage.stage} eval_artifacts do not align with eval_paths")
    if not references:
        raise ValueError(f"{stage.stage} has no conversation held-out artifact")
    return references


def _load_eval_artifacts(stage: _LoadedStage) -> list[_EvalArtifact]:
    artifacts = []
    seen_paths = set()
    for reference, recorded in _eval_references(stage):
        path = _resolve_recorded_file(
            reference,
            stage=stage,
            label=f"{stage.stage} evaluation artifact",
        )
        if path in seen_paths:
            raise ValueError(f"{stage.stage} repeats evaluation artifact {reference!r}")
        seen_paths.add(path)
        identity = {
            "path": reference,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "availability": "local_file_verified",
        }
        if recorded is not None:
            if recorded.get("bytes") != identity["bytes"]:
                raise ValueError(f"{stage.stage} recorded eval artifact byte size mismatch")
            if recorded.get("sha256") != identity["sha256"]:
                raise ValueError(f"{stage.stage} recorded eval artifact SHA-256 mismatch")
        artifacts.append(_EvalArtifact(path=path, identity=identity))
    return artifacts


def _conversation_fingerprints(
    artifacts: Sequence[_EvalArtifact],
) -> tuple[list[str], list[str]]:
    all_rows = []
    single_turn_rows = []
    for artifact in artifacts:
        try:
            with artifact.path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    conversation = Conversation.from_json(line)
                    payload = json.loads(conversation.to_json())
                    payload.pop("meta", None)
                    fingerprint = canonical_sha256(payload)
                    all_rows.append(fingerprint)
                    if (
                        len(conversation.messages) == 2
                        and conversation.messages[0].role == Role.user
                        and conversation.messages[1].role == Role.assistant
                    ):
                        single_turn_rows.append(fingerprint)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid canonical conversation evaluation artifact: {artifact.path}"
            ) from error
    if not all_rows:
        raise ValueError("conversation evaluation artifacts contain no rows")
    return all_rows, single_turn_rows


def _validate_frozen_eval_identity(
    stages: Mapping[str, _LoadedStage],
) -> dict[str, Any]:
    artifacts = {
        stage_name: _load_eval_artifacts(stages[stage_name])
        for stage_name in STAGE_ORDER
    }
    keys = {
        stage_name: {artifact.key for artifact in stage_artifacts}
        for stage_name, stage_artifacts in artifacts.items()
    }
    if keys["sft"] != keys["rl"]:
        raise ValueError("SFT and RL do not use the exact same frozen eval artifacts")
    if not keys["sft"].issubset(keys["midtrain"]):
        raise ValueError("midtrain does not include the frozen SFT/RL eval artifact")

    sft_all_rows, _ = _conversation_fingerprints(artifacts["sft"])
    expected_sft_dataset = canonical_sha256(sorted(sft_all_rows))
    sft_contract = stages["sft"].heldout_eval["contract"]
    if sft_contract.get("dataset_sha256") != expected_sft_dataset:
        raise ValueError("SFT held-out dataset fingerprint does not match its eval artifact")

    _, rl_single_turn_rows = _conversation_fingerprints(artifacts["rl"])
    if not rl_single_turn_rows:
        raise ValueError("RL frozen eval artifacts contain no single-turn scored rows")
    expected_rl_dataset = hashlib.sha256(
        "\n".join(sorted(rl_single_turn_rows)).encode("ascii")
    ).hexdigest()
    rl_contract = stages["rl"].heldout_eval["contract"]
    if rl_contract.get("dataset_sha256") != expected_rl_dataset:
        raise ValueError("RL held-out dataset fingerprint does not match its eval artifact")
    rl_data = _stage_data(stages["rl"])
    split_audit = _mapping(
        rl_data.get("split_audit"),
        label="rl data.split_audit",
    )
    if split_audit.get("eval_scored_rows_sha256") != expected_rl_dataset:
        raise ValueError("RL split audit does not match its frozen eval artifact")

    shared_identities = [
        {"bytes": byte_size, "sha256": digest}
        for byte_size, digest in sorted(keys["sft"])
    ]
    stage_identities = {
        stage_name: [
            copy.deepcopy(artifact.identity)
            for artifact in sorted(
                stage_artifacts,
                key=lambda item: (str(item.identity["sha256"]), int(item.identity["bytes"])),
            )
        ]
        for stage_name, stage_artifacts in artifacts.items()
    }
    return {
        "shared_sft_rl_artifact_set_sha256": canonical_sha256(shared_identities),
        "shared_sft_rl_artifacts": shared_identities,
        "stage_artifacts": stage_identities,
        "sft_dataset_sha256": expected_sft_dataset,
        "rl_scored_dataset_sha256": expected_rl_dataset,
    }


def _stage_record(
    stage: _LoadedStage,
    *,
    history: Mapping[str, Any],
    optimization: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage.stage,
        "artifacts": {
            "metrics": _artifact_record(
                stage.specification.metrics_path,
                stage.metrics_artifact,
            ),
            "checkpoint": _artifact_record(
                stage.specification.checkpoint_path,
                stage.checkpoint_artifact,
            ),
            "inputs": copy.deepcopy(dict(input_provenance)),
        },
        "lineage": copy.deepcopy(stage.lineage),
        "execution": copy.deepcopy(stage.execution),
        "history": copy.deepcopy(dict(history)),
        "optimization": copy.deepcopy(dict(optimization)),
        "heldout_eval": copy.deepcopy(stage.heldout_eval),
    }


def summarize_stage_pilot(inputs: Sequence[StagePilotInput]) -> dict[str, Any]:
    """Validate three stage artifacts and return one self-hashed canonical summary.

    Inputs are positional by design so an accidentally swapped SFT/RL result cannot be silently
    relabeled. A zero-update RL run is mechanically valid when all evidence is complete, but its
    optimization outcome is classified as ``zero_signal``.
    """

    specifications = list(inputs)
    observed_order = tuple(specification.stage for specification in specifications)
    if observed_order != STAGE_ORDER:
        raise ValueError(
            f"stage order must be {' -> '.join(STAGE_ORDER)}, got "
            f"{' -> '.join(observed_order) or '<empty>'}"
        )
    loaded_list = [_load_stage(specification) for specification in specifications]
    stages = {stage.stage: stage for stage in loaded_list}

    model_identities = {stage.model_config_sha256 for stage in loaded_list}
    if len(model_identities) != 1:
        raise ValueError("pilot stages do not share one model-config identity")
    tokenizer_identities = {stage.tokenizer_sha256 for stage in loaded_list}
    if len(tokenizer_identities) != 1:
        raise ValueError("pilot stages do not share one tokenizer identity")

    midtrain_sha256 = str(stages["midtrain"].checkpoint_artifact["sha256"])
    sft_sha256 = str(stages["sft"].checkpoint_artifact["sha256"])
    if stages["sft"].lineage.get("parent_checkpoint_sha256") != midtrain_sha256:
        raise ValueError("SFT parent_checkpoint_sha256 does not match the midtrain checkpoint")
    if stages["rl"].lineage.get("parent_checkpoint_sha256") != sft_sha256:
        raise ValueError("RL parent_checkpoint_sha256 does not match the SFT checkpoint")

    histories = {stage.stage: _validate_history(stage) for stage in loaded_list}
    midtrain_accounting = _validate_token_accounting(stages["midtrain"])
    sft_accounting = _validate_token_accounting(stages["sft"])
    rl_accounting, rl_classification = _validate_rl_accounting(stages["rl"])
    rl_contract = _validate_rl_contract(stages["rl"])
    structured_heads = _validate_structured_head_transition(stages["sft"], stages["rl"])
    frozen_eval = _validate_frozen_eval_identity(stages)
    input_provenance = {
        stage.stage: _stage_input_provenance(stage)
        for stage in loaded_list
    }

    stage_records = [
        _stage_record(
            stages["midtrain"],
            history=histories["midtrain"],
            optimization={"token_accounting": midtrain_accounting},
            input_provenance=input_provenance["midtrain"],
        ),
        _stage_record(
            stages["sft"],
            history=histories["sft"],
            optimization={
                "token_accounting": sft_accounting,
                "structured_heads": structured_heads["sft_available"],
            },
            input_provenance=input_provenance["sft"],
        ),
        _stage_record(
            stages["rl"],
            history=histories["rl"],
            optimization={
                "rl_accounting": rl_accounting,
                "classification": rl_classification,
                "structured_heads_available": False,
                "invalidated_structured_heads": structured_heads["rl_invalidated"],
                "contracts": rl_contract,
            },
            input_provenance=input_provenance["rl"],
        ),
    ]
    report = {
        "kind": STAGE_PILOT_SUMMARY_KIND,
        "schema_version": STAGE_PILOT_SUMMARY_SCHEMA_VERSION,
        "scope": {
            "design": "bounded_single_seed_stage_pilot",
            "stage_order": list(STAGE_ORDER),
            "claim": "mechanical_validation_only",
            "browser_action_result": "separate_not_included",
            "artifact_availability": (
                "local identities only; no checkpoint or dataset publication is asserted"
            ),
        },
        "identity": {
            "model_config_sha256": next(iter(model_identities)),
            "tokenizer_sha256": next(iter(tokenizer_identities)),
        },
        "validation": {
            "status": "mechanically_valid",
            "checks": {
                "stage_order": True,
                "metrics_checkpoint_file_identity": True,
                "canonical_config_identity": True,
                "canonical_data_identity": True,
                "complete_stage_lineage": True,
                "resolved_input_artifact_identity": True,
                "parent_checkpoint_chain": True,
                "same_model_identity": True,
                "same_tokenizer_identity": True,
                "frozen_eval_artifact_identity": True,
                "execution_records_present": True,
                "finite_losses_and_rewards": True,
                "nonzero_midtrain_sft_token_accounting": True,
                "rl_rollouts_attempted": True,
                "structured_head_transition": True,
            },
        },
        "rl_optimization_outcome": {
            "classification": rl_classification,
            "attempted_rollouts": rl_accounting["attempted_rollouts"],
            "realized_optimizer_updates": rl_accounting["realized_optimizer_updates"],
        },
        "frozen_eval": frozen_eval,
        "stages": stage_records,
        "limitations": copy.deepcopy(list(_LIMITATIONS)),
    }
    report["summary_sha256"] = canonical_sha256(report)
    return report


def write_stage_pilot_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write a finite, sorted JSON summary after checking its self-hash."""

    payload = copy.deepcopy(dict(summary))
    recorded_sha256 = payload.pop("summary_sha256", None)
    expected_sha256 = canonical_sha256(payload)
    if recorded_sha256 != expected_sha256:
        raise ValueError("stage pilot summary self-hash is missing or invalid")
    payload["summary_sha256"] = recorded_sha256
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination)
