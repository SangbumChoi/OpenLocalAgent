"""Fail-closed preparation and verification for sealed SFT promotion candidates.

The preparation half consumes and deterministically replays a schema-v2 checkpoint sweep,
independently reconstructs its retention selection, re-hashes every immutable archive, and emits
two candidate-specific agent-scorecard configs plus one self-hashed binding receipt.  The
verification half replays each supplied scorecard from the bound model, config, and cases before
producing a decision.  It never selects a replacement checkpoint and a development scorecard
alone can never authorize promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from localagent.data.conversation_artifact import (
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    audit_conversation_overlap,
    canonical_json_bytes,
    conversation_semantic_sha256,
    load_verified_conversation_artifact,
)
from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.eval import agent_scorecard as agent_scorecard_module
from localagent.eval.agent_scorecard import (
    CONFIG_KIND as SCORECARD_CONFIG_KIND,
)
from localagent.eval.agent_scorecard import (
    RESULT_KIND as SCORECARD_RESULT_KIND,
)
from localagent.eval.agent_scorecard import (
    SCHEMA_VERSION as SCORECARD_SCHEMA_VERSION,
)
from localagent.eval.agent_scorecard import run_scorecard
from localagent.eval.sft_checkpoint_sweep import (
    RESULT_KIND as SWEEP_RESULT_KIND,
)
from localagent.eval.sft_checkpoint_sweep import (
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
)
from localagent.eval.sft_checkpoint_sweep import run_sft_checkpoint_sweep
from localagent.eval.tool_eval import AssistantPrediction, score_conversations
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import BPETokenizer, ByteTokenizer
from localagent.train.stage_data import canonical_sha256, tokenizer_identity

BINDING_KIND = "localagent_sft_candidate_binding_receipt"
DECISION_KIND = "localagent_sft_candidate_promotion_decision"
SCHEMA_VERSION = 1

DEFAULT_BASE_SCORECARD_CONFIG = Path("configs/eval/webgpu-1m-sft-scorecard.yaml")
DEFAULT_CONFIRMATORY_CASES = Path("data/synth/agent_eval_confirmatory_v2.jsonl")
DEFAULT_CONFIRMATORY_MANIFEST = Path("data/synth/agent_eval_confirmatory_v2.jsonl.manifest.v1.json")
DEFAULT_CONFIRMATORY_GENERATOR_CONFIG = Path("configs/eval/paper-confirmatory-eval-split-v2.yaml")
DEFAULT_CONFIRMATORY_PROVENANCE = Path("data/provenance/paper/agent-eval-confirmatory-v2.json")

PROMOTION_THRESHOLDS = {
    "minimum_assistant_decisions": 512,
    "minimum_eos_completion_rate": 0.90,
    "maximum_truncation_rate": 0.10,
    "minimum_metric_successes": 1,
    "minimum_metric_rate": 0.05,
}

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_CASE_BYTES = 1024 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_RE = re.compile(r".+\.step-(\d{8})\.pt")

_SWEEP_TOP_LEVEL_KEYS = frozenset(
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
_SWEEP_RECORD_KEYS = frozenset(
    {
        "artifact",
        "checkpoint_step",
        "completed_steps",
        "planned_steps",
        "metrics",
        "delta_from_baseline",
        "gates",
        "retention_eligible",
    }
)
_SWEEP_METRIC_KEYS = frozenset(
    {
        "rows",
        "assistant_loss_tokens",
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
)
_BASELINE_TOLERANCE_KEYS = frozenset(
    {
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
)
_SWEEP_DELTA_KEYS = frozenset(
    {"mean_loss", "assistant_token_accuracy", "assistant_sequence_accuracy"}
)
_SWEEP_GATE_KEYS = frozenset(
    {
        "mean_loss_non_inferiority",
        "assistant_token_accuracy_non_inferiority",
        "assistant_sequence_accuracy_non_inferiority",
    }
)
_SWEEP_THRESHOLD_KEYS = frozenset(
    {
        "max_mean_loss_increase",
        "max_assistant_token_accuracy_drop",
        "max_assistant_sequence_accuracy_drop",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "evaluated_checkpoints",
        "retention_eligible_checkpoints",
        "failed_checkpoints",
        "status",
        "best_retention_eligible_checkpoint",
    }
)
_BEST_KEYS = frozenset({"artifact", "checkpoint_step", "completed_steps", "metrics"})
_FILE_RECORD_KEYS = frozenset({"path", "bytes", "sha256"})
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
_PROMOTION_METRIC_GATE_NAMES = frozenset(
    {
        "overall_complete_format",
        "expected_tool_strict_format",
        "expected_tool_schema_validity",
        "expected_tool_case_exact_name",
        "expected_tool_whole_call_exact",
        "expected_no_tool_structural_abstention",
    }
)


@dataclass(frozen=True)
class ValidatedSweep:
    """Selection-critical facts reconstructed from a sealed sweep result."""

    result: dict[str, Any]
    result_artifact: dict[str, Any]
    candidate_record: dict[str, Any]
    candidate_artifact: dict[str, Any]
    training_config_artifact: dict[str, Any]
    training_config_sha256: str
    sweep_config_artifact: dict[str, Any]
    sweep_config_sha256: str
    model_config_artifact: dict[str, Any]
    model_config_sha256: str
    tokenizer_record: dict[str, Any]
    heldout_source_bindings: tuple[dict[str, Any], ...]
    checkpoint_artifacts: tuple[dict[str, Any], ...]
    lineage: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float equality coercions."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_number(value: Any, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _unit_interval(value: Any, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _exact_mapping(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _require_keys(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    result = _mapping(value, label=label)
    missing = sorted(keys - set(result))
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict finite UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON values") from error
    return value


def _yaml_mapping(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a YAML mapping")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from error
    return dict(value)


def _literal_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty literal relative path")
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX path separators")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or str(parsed) != value
        or any(part in {".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"{label} must be a normalized literal relative path")
    return value


def _path_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return value


def _resolve_input(repository_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _relative_input(repository_root: Path, value: Any, *, label: str) -> tuple[str, Path]:
    literal = _literal_relative_path(value, label=label)
    _reject_symlink_ancestors(
        repository_root.absolute(),
        label=f"{label} repository root",
    )
    _reject_symlink_components(repository_root, literal, label=label)
    return literal, repository_root / literal


@contextmanager
def _at_repository_root(repository_root: Path):
    """Temporarily run path-sensitive sealed evaluators from their repository root."""

    previous = Path.cwd()
    os.chdir(repository_root)
    try:
        yield
    finally:
        os.chdir(previous)


def _reject_symlink_components(repository_root: Path, value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute():
        return
    current = repository_root
    try:
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"{label} repository root must not be a symlink: {current}")
    except FileNotFoundError:
        pass
    for part in path.parts:
        current = current / part
        try:
            observed = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} path contains a symlink component: {current}")


def _stable_stat_equal(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
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
        pathname = path.lstat()
        if not _stable_stat_equal(initial, pathname):
            raise RuntimeError(f"{label} changed while its pathname was bound: {path}")
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
        final_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final_path = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} disappeared while being read: {path}") from error
    if (
        not _stable_stat_equal(initial, final_descriptor)
        or not _stable_stat_equal(initial, final_path)
        or observed != initial.st_size
    ):
        raise RuntimeError(f"{label} changed while being read: {path}")
    return b"".join(chunks)


def _hash_regular(path: Path, *, label: str, max_bytes: int) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
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
        pathname = path.lstat()
        if not _stable_stat_equal(initial, pathname):
            raise RuntimeError(f"{label} changed while its pathname was bound: {path}")
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
            digest.update(chunk)
        final_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        final_path = path.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} disappeared while being hashed: {path}") from error
    if (
        not _stable_stat_equal(initial, final_descriptor)
        or not _stable_stat_equal(initial, final_path)
        or observed != initial.st_size
    ):
        raise RuntimeError(f"{label} changed while being hashed: {path}")
    return {"bytes": observed, "sha256": digest.hexdigest()}


def _file_record(path_text: str, payload: bytes) -> dict[str, Any]:
    return {"path": path_text, "bytes": len(payload), "sha256": _sha256(payload)}


def _validated_file_record(
    value: Any,
    *,
    label: str,
    literal_relative: bool = False,
) -> dict[str, Any]:
    record = _exact_mapping(value, _FILE_RECORD_KEYS, label=label)
    if literal_relative:
        path = _literal_relative_path(record["path"], label=f"{label}.path")
    else:
        path = _path_text(record["path"], label=f"{label}.path")
    size = _strict_int(record["bytes"], label=f"{label}.bytes", minimum=1)
    sha256 = _require_sha256(record["sha256"], label=f"{label}.sha256")
    return {"path": path, "bytes": size, "sha256": sha256}


def _assert_current_file(
    record: Mapping[str, Any],
    *,
    repository_root: Path,
    label: str,
    max_bytes: int,
) -> None:
    _reject_symlink_components(
        repository_root,
        str(record["path"]),
        label=label,
    )
    observed = _hash_regular(
        _resolve_input(repository_root, str(record["path"])),
        label=label,
        max_bytes=max_bytes,
    )
    expected = {"bytes": record["bytes"], "sha256": record["sha256"]}
    if observed != expected:
        raise ValueError(
            f"{label} byte identity mismatch: expected={expected}, observed={observed}"
        )


def _load_canonical_json_file(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_JSON_BYTES,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, label=label, max_bytes=max_bytes)
    value = _strict_json(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value, payload


def _normalized_training_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _validated_sweep_tokenizer_record(
    value: Any,
    *,
    training_config: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    data = _mapping(
        training_config.get("data"),
        label="selected SFT training config.data",
    )
    configured = _mapping(
        data.get("tokenizer", {"kind": "byte"}),
        label="selected SFT training config.data.tokenizer",
    )
    if set(configured) - {"kind", "path"}:
        raise ValueError("selected SFT training tokenizer contains unsupported fields")
    configured_kind = configured.get("kind", "byte")
    if configured_kind not in {"byte", "bpe"}:
        raise ValueError("selected SFT training tokenizer kind is unsupported")

    raw = _mapping(value, label="SFT checkpoint sweep identity.tokenizer")
    expected_keys = {"kind", "vocab_size", "sha256"}
    if "artifact" in raw:
        expected_keys.add("artifact")
    tokenizer = _exact_mapping(
        raw,
        frozenset(expected_keys),
        label="SFT checkpoint sweep identity.tokenizer",
    )
    if tokenizer["kind"] != configured_kind:
        raise ValueError("SFT checkpoint sweep tokenizer kind disagrees with its training config")
    _strict_int(
        tokenizer["vocab_size"],
        label="SFT checkpoint sweep identity.tokenizer.vocab_size",
        minimum=1,
    )
    tokenizer_sha256 = _require_sha256(
        tokenizer["sha256"],
        label="SFT checkpoint sweep identity.tokenizer.sha256",
    )
    if configured_kind == "byte":
        if configured.get("path") is not None or "artifact" in tokenizer:
            raise ValueError("byte tokenizer must not use an artifact path")
        expected_sha256 = tokenizer_identity(
            "byte",
            vocab_size=tokenizer["vocab_size"],
        )["sha256"]
        if tokenizer_sha256 != expected_sha256:
            raise ValueError("SFT checkpoint sweep byte tokenizer identity is invalid")
        return copy.deepcopy(tokenizer)

    configured_path, _configured_source = _relative_input(
        repository_root,
        configured.get("path"),
        label="selected SFT training config.data.tokenizer.path",
    )
    artifact = _validated_file_record(
        tokenizer.get("artifact"),
        label="SFT checkpoint sweep identity.tokenizer.artifact",
        literal_relative=True,
    )
    if artifact["path"] != configured_path:
        raise ValueError(
            "SFT checkpoint sweep and training config reference different tokenizer paths"
        )
    if artifact["sha256"] != tokenizer_sha256:
        raise ValueError("SFT checkpoint sweep tokenizer hash disagrees with its artifact")
    _assert_current_file(
        artifact,
        repository_root=repository_root,
        label="bound SFT checkpoint sweep tokenizer artifact",
        max_bytes=_MAX_CONFIG_BYTES * 16,
    )
    return {**copy.deepcopy(tokenizer), "artifact": artifact}


def _configured_eval_source_specs(training_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = _mapping(
        training_config.get("data"),
        label="selected SFT training config.data",
    )
    raw = data.get("eval_conversations")
    values = raw if isinstance(raw, list) else [raw]
    if not values or any(value is None for value in values):
        raise ValueError("selected SFT training config.data.eval_conversations must be non-empty")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        source = _mapping(
            value,
            label=f"selected SFT training eval source {index}",
        )
        artifact = _exact_mapping(
            source.get("artifact"),
            frozenset(
                {
                    "manifest",
                    "generator_config",
                    "expected_split",
                    "expected_rule_verified",
                    "environment_policy",
                }
            ),
            label=f"selected SFT training eval source {index}.artifact",
        )
        if (
            artifact["expected_split"] != "eval"
            or artifact["expected_rule_verified"] is not True
            or artifact["environment_policy"] != "forbid"
        ):
            raise ValueError(
                f"selected SFT training eval source {index} has an unsafe artifact contract"
            )
        result.append({"path": source.get("path"), "artifact": artifact})
    return result


def _validated_heldout_source_bindings(
    value: Any,
    *,
    training_config: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("SFT checkpoint sweep heldout.sources must be a non-empty list")
    configured = _configured_eval_source_specs(training_config)
    if len(value) != len(configured):
        raise ValueError("SFT checkpoint sweep heldout sources disagree with its training config")

    bindings: list[dict[str, Any]] = []
    for index, (reported_value, source_config) in enumerate(zip(value, configured, strict=True)):
        label = f"SFT checkpoint sweep heldout.sources[{index}]"
        reported = _exact_mapping(
            reported_value,
            frozenset(
                {
                    "path",
                    "kind",
                    "schema_version",
                    "split",
                    "jsonl",
                    "sidecar",
                    "generator_config",
                }
            ),
            label=label,
        )
        jsonl_path, _jsonl_source = _relative_input(
            repository_root,
            source_config["path"],
            label=f"selected SFT training eval source {index}.path",
        )
        reported_path = _literal_relative_path(
            reported["path"],
            label=f"{label}.path",
        )
        if reported_path != jsonl_path:
            raise ValueError(f"{label} path disagrees with its training config")
        if (
            reported["kind"] != MANIFEST_KIND
            or type(reported["schema_version"]) is not int
            or reported["schema_version"] != MANIFEST_SCHEMA_VERSION
            or reported["split"] != "eval"
        ):
            raise ValueError(f"{label} artifact identity is invalid")

        manifest_path, _manifest_source = _relative_input(
            repository_root,
            source_config["artifact"]["manifest"],
            label=f"selected SFT training eval source {index}.artifact.manifest",
        )
        generator_path, _generator_source = _relative_input(
            repository_root,
            source_config["artifact"]["generator_config"],
            label=(f"selected SFT training eval source {index}.artifact.generator_config"),
        )
        jsonl_identity = _exact_mapping(
            reported["jsonl"],
            frozenset({"bytes", "sha256"}),
            label=f"{label}.jsonl",
        )
        jsonl = _validated_file_record(
            {"path": reported_path, **jsonl_identity},
            label=f"{label}.jsonl",
            literal_relative=True,
        )
        sidecar_raw = _exact_mapping(
            reported["sidecar"],
            frozenset({"bytes", "sha256", "manifest_self_sha256"}),
            label=f"{label}.sidecar",
        )
        manifest = _validated_file_record(
            {
                "path": manifest_path,
                "bytes": sidecar_raw["bytes"],
                "sha256": sidecar_raw["sha256"],
            },
            label=f"{label}.sidecar",
            literal_relative=True,
        )
        manifest_self_sha256 = _require_sha256(
            sidecar_raw["manifest_self_sha256"],
            label=f"{label}.sidecar.manifest_self_sha256",
        )
        generator_identity = _exact_mapping(
            reported["generator_config"],
            frozenset({"bytes", "sha256"}),
            label=f"{label}.generator_config",
        )
        generator = _validated_file_record(
            {
                "path": generator_path,
                **generator_identity,
            },
            label=f"{label}.generator_config",
            literal_relative=True,
        )
        files = {
            "jsonl": jsonl,
            "manifest": {
                **manifest,
                "manifest_self_sha256": manifest_self_sha256,
            },
            "generator_config": generator,
        }
        for name, record, maximum in (
            ("jsonl", jsonl, _MAX_CASE_BYTES),
            ("manifest", manifest, _MAX_CONFIG_BYTES),
            ("generator_config", generator, _MAX_CONFIG_BYTES),
        ):
            _assert_current_file(
                record,
                repository_root=repository_root,
                label=f"bound {label} {name}",
                max_bytes=maximum,
            )
        bindings.append(
            {
                "reported_identity": copy.deepcopy(reported),
                "files": files,
            }
        )
    return tuple(bindings)


def _validate_sweep_metric(value: Any, *, label: str) -> dict[str, Any]:
    metric = _exact_mapping(value, _SWEEP_METRIC_KEYS, label=label)
    rows = _strict_int(metric["rows"], label=f"{label}.rows", minimum=1)
    loss_tokens = _strict_int(
        metric["assistant_loss_tokens"],
        label=f"{label}.assistant_loss_tokens",
        minimum=1,
    )
    return {
        "rows": rows,
        "assistant_loss_tokens": loss_tokens,
        "mean_loss": _nonnegative_number(metric["mean_loss"], label=f"{label}.mean_loss"),
        "assistant_token_accuracy": _unit_interval(
            metric["assistant_token_accuracy"],
            label=f"{label}.assistant_token_accuracy",
        ),
        "assistant_sequence_accuracy": _unit_interval(
            metric["assistant_sequence_accuracy"],
            label=f"{label}.assistant_sequence_accuracy",
        ),
    }


def _validate_sweep_record(
    value: Any,
    *,
    index: int,
    baseline: Mapping[str, Any],
    thresholds: Mapping[str, float],
    expected_rows: int,
    expected_loss_tokens: int,
) -> dict[str, Any]:
    label = f"SFT checkpoint sweep checkpoints[{index}]"
    record = _exact_mapping(value, _SWEEP_RECORD_KEYS, label=label)
    artifact = _validated_file_record(
        record["artifact"],
        label=f"{label}.artifact",
        literal_relative=True,
    )
    archive_match = _ARCHIVE_RE.fullmatch(Path(artifact["path"]).name)
    if archive_match is None:
        raise ValueError(f"{label}.artifact.path is not an immutable step archive")
    checkpoint_step = _strict_int(
        record["checkpoint_step"],
        label=f"{label}.checkpoint_step",
    )
    completed_steps = _strict_int(
        record["completed_steps"],
        label=f"{label}.completed_steps",
        minimum=1,
    )
    planned_steps = _strict_int(
        record["planned_steps"],
        label=f"{label}.planned_steps",
        minimum=1,
    )
    if checkpoint_step != completed_steps - 1 or completed_steps > planned_steps:
        raise ValueError(f"{label} step accounting is inconsistent")
    if int(archive_match.group(1)) != completed_steps:
        raise ValueError(f"{label} archive filename disagrees with completed_steps")
    metrics = _validate_sweep_metric(record["metrics"], label=f"{label}.metrics")
    if metrics["rows"] != expected_rows or metrics["assistant_loss_tokens"] != expected_loss_tokens:
        raise ValueError(f"{label}.metrics held-out counts disagree with sweep inputs")

    expected_delta = {
        "mean_loss": metrics["mean_loss"] - float(baseline["mean_loss"]),
        "assistant_token_accuracy": (
            metrics["assistant_token_accuracy"] - float(baseline["assistant_token_accuracy"])
        ),
        "assistant_sequence_accuracy": (
            metrics["assistant_sequence_accuracy"] - float(baseline["assistant_sequence_accuracy"])
        ),
    }
    delta_raw = _exact_mapping(
        record["delta_from_baseline"],
        _SWEEP_DELTA_KEYS,
        label=f"{label}.delta_from_baseline",
    )
    delta = {
        key: _finite_number(delta_raw[key], label=f"{label}.delta_from_baseline.{key}")
        for key in sorted(_SWEEP_DELTA_KEYS)
    }
    if delta != expected_delta:
        raise ValueError(f"{label}.delta_from_baseline is inconsistent with metrics")

    expected_gates = {
        "mean_loss_non_inferiority": {
            "observed_increase": expected_delta["mean_loss"],
            "maximum_increase": thresholds["max_mean_loss_increase"],
            "passed": expected_delta["mean_loss"] <= thresholds["max_mean_loss_increase"],
        },
        "assistant_token_accuracy_non_inferiority": {
            "observed_drop": -expected_delta["assistant_token_accuracy"],
            "maximum_drop": thresholds["max_assistant_token_accuracy_drop"],
            "passed": (
                -expected_delta["assistant_token_accuracy"]
                <= thresholds["max_assistant_token_accuracy_drop"]
            ),
        },
        "assistant_sequence_accuracy_non_inferiority": {
            "observed_drop": -expected_delta["assistant_sequence_accuracy"],
            "maximum_drop": thresholds["max_assistant_sequence_accuracy_drop"],
            "passed": (
                -expected_delta["assistant_sequence_accuracy"]
                <= thresholds["max_assistant_sequence_accuracy_drop"]
            ),
        },
    }
    gates = _exact_mapping(record["gates"], _SWEEP_GATE_KEYS, label=f"{label}.gates")
    for gate_name, expected_gate in expected_gates.items():
        observed_gate = _exact_mapping(
            gates[gate_name],
            frozenset(expected_gate),
            label=f"{label}.gates.{gate_name}",
        )
        for key in expected_gate:
            if key == "passed":
                if not isinstance(observed_gate[key], bool):
                    raise TypeError(f"{label}.gates.{gate_name}.passed must be boolean")
            else:
                _finite_number(
                    observed_gate[key],
                    label=f"{label}.gates.{gate_name}.{key}",
                )
        if not _json_equal(observed_gate, expected_gate):
            raise ValueError(f"{label}.gates.{gate_name} is inconsistent")
    eligible = record["retention_eligible"]
    if not isinstance(eligible, bool):
        raise TypeError(f"{label}.retention_eligible must be boolean")
    expected_eligible = all(gate["passed"] for gate in expected_gates.values())
    if eligible is not expected_eligible:
        raise ValueError(f"{label}.retention_eligible is inconsistent with gates")
    return {
        "artifact": artifact,
        "checkpoint_step": checkpoint_step,
        "completed_steps": completed_steps,
        "planned_steps": planned_steps,
        "metrics": metrics,
        "delta_from_baseline": delta,
        "gates": expected_gates,
        "retention_eligible": eligible,
    }


def _best_retention_record(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    eligible = [record for record in records if record["retention_eligible"]]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda record: (
            float(record["metrics"]["assistant_sequence_accuracy"]),
            float(record["metrics"]["assistant_token_accuracy"]),
            -float(record["metrics"]["mean_loss"]),
            -int(record["completed_steps"]),
            str(record["artifact"]["sha256"]),
        ),
    )
    return {
        "artifact": copy.deepcopy(dict(best["artifact"])),
        "checkpoint_step": best["checkpoint_step"],
        "completed_steps": best["completed_steps"],
        "metrics": copy.deepcopy(dict(best["metrics"])),
    }


def load_validated_sweep_result(
    path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> ValidatedSweep:
    """Load and independently validate one canonical schema-v2 sealed sweep result."""

    root = Path(repository_root)
    source_text, source_path = _relative_input(
        root,
        str(path),
        label="SFT checkpoint sweep result path",
    )
    result, payload = _load_canonical_json_file(
        source_path,
        label="SFT checkpoint sweep result",
    )
    result = _exact_mapping(
        result,
        _SWEEP_TOP_LEVEL_KEYS,
        label="SFT checkpoint sweep result",
    )
    schema_version = result["schema_version"]
    if (
        result["kind"] != SWEEP_RESULT_KIND
        or type(schema_version) is not int
        or schema_version != SWEEP_SCHEMA_VERSION
    ):
        raise ValueError("SFT checkpoint sweep result kind/schema is invalid")
    recorded_self_hash = _require_sha256(
        result["result_sha256"],
        label="SFT checkpoint sweep result.result_sha256",
    )
    without_hash = copy.deepcopy(result)
    without_hash.pop("result_sha256")
    if recorded_self_hash != canonical_sha256(without_hash):
        raise ValueError("SFT checkpoint sweep result self-hash mismatch")

    inputs = _require_keys(
        result["inputs"],
        frozenset(
            {
                "sweep_config",
                "sweep_config_sha256",
                "training_config",
                "training_config_sha256",
                "expected_eval",
                "expected_baseline",
            }
        ),
        label="SFT checkpoint sweep result.inputs",
    )
    expected_eval = _exact_mapping(
        inputs["expected_eval"],
        frozenset({"conversations", "assistant_decisions", "assistant_loss_tokens"}),
        label="SFT checkpoint sweep result.inputs.expected_eval",
    )
    for key in expected_eval:
        expected_eval[key] = _strict_int(
            expected_eval[key],
            label=f"SFT checkpoint sweep result.inputs.expected_eval.{key}",
            minimum=1,
        )
    expected_baseline = _exact_mapping(
        inputs["expected_baseline"],
        frozenset({"metrics", "absolute_tolerances"}),
        label="SFT checkpoint sweep result.inputs.expected_baseline",
    )
    configured_baseline = _validate_sweep_metric(
        expected_baseline["metrics"],
        label="SFT checkpoint sweep result.inputs.expected_baseline.metrics",
    )
    baseline_tolerances_raw = _exact_mapping(
        expected_baseline["absolute_tolerances"],
        _BASELINE_TOLERANCE_KEYS,
        label="SFT checkpoint sweep result.inputs.expected_baseline.absolute_tolerances",
    )
    baseline_tolerances = {
        key: _nonnegative_number(
            baseline_tolerances_raw[key],
            label=(
                "SFT checkpoint sweep result.inputs.expected_baseline."
                f"absolute_tolerances.{key}"
            ),
        )
        for key in sorted(_BASELINE_TOLERANCE_KEYS)
    }
    if (
        configured_baseline["rows"] != expected_eval["assistant_decisions"]
        or configured_baseline["assistant_loss_tokens"] != expected_eval["assistant_loss_tokens"]
    ):
        raise ValueError("SFT checkpoint sweep baseline counts disagree with expected_eval")
    heldout = _require_keys(
        result["heldout"],
        frozenset(
            {
                "sources",
                "baseline",
                "conversations",
                "assistant_decisions",
                "assistant_loss_tokens",
            }
        ),
        label="SFT checkpoint sweep result.heldout",
    )
    heldout_baseline = _validate_sweep_metric(
        heldout["baseline"],
        label="SFT checkpoint sweep result.heldout.baseline",
    )
    if (
        heldout_baseline["rows"] != configured_baseline["rows"]
        or heldout_baseline["assistant_loss_tokens"]
        != configured_baseline["assistant_loss_tokens"]
    ):
        raise ValueError(
            "SFT checkpoint sweep heldout baseline counts disagree with configured baseline"
        )
    for key, tolerance in baseline_tolerances.items():
        difference = abs(
            float(heldout_baseline[key]) - float(configured_baseline[key])
        )
        if difference > tolerance:
            raise ValueError(
                "SFT checkpoint sweep heldout baseline disagrees with configured baseline: "
                f"{key} absolute_difference={difference}, tolerance={tolerance}"
            )
    for key in ("conversations", "assistant_decisions", "assistant_loss_tokens"):
        observed = _strict_int(
            heldout[key],
            label=f"SFT checkpoint sweep result.heldout.{key}",
            minimum=1,
        )
        if observed != expected_eval[key]:
            raise ValueError(f"SFT checkpoint sweep heldout.{key} disagrees with expected_eval")

    thresholds_raw = _exact_mapping(
        result["thresholds"],
        _SWEEP_THRESHOLD_KEYS,
        label="SFT checkpoint sweep result.thresholds",
    )
    thresholds = {
        key: _nonnegative_number(
            thresholds_raw[key],
            label=f"SFT checkpoint sweep result.thresholds.{key}",
        )
        for key in sorted(_SWEEP_THRESHOLD_KEYS)
    }
    records_raw = result["checkpoints"]
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("SFT checkpoint sweep result.checkpoints must be a non-empty list")
    records = [
        _validate_sweep_record(
            value,
            index=index,
            baseline=heldout_baseline,
            thresholds=thresholds,
            expected_rows=expected_eval["assistant_decisions"],
            expected_loss_tokens=expected_eval["assistant_loss_tokens"],
        )
        for index, value in enumerate(records_raw)
    ]
    identities = [(record["completed_steps"], record["artifact"]["path"]) for record in records]
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ValueError("SFT checkpoint sweep records are not uniquely sorted by completed step")
    if len({record["artifact"]["path"] for record in records}) != len(records):
        raise ValueError("SFT checkpoint sweep contains duplicate archive paths")
    if len({record["artifact"]["sha256"] for record in records}) != len(records):
        raise ValueError("SFT checkpoint sweep contains duplicate archive byte identities")
    for index, record in enumerate(records):
        _assert_current_file(
            record["artifact"],
            repository_root=root,
            label=f"SFT checkpoint sweep archive {index}",
            max_bytes=_MAX_CHECKPOINT_BYTES,
        )

    selection_contract = _exact_mapping(
        result["selection_contract"],
        frozenset({"eligible_filter", "ranking"}),
        label="SFT checkpoint sweep result.selection_contract",
    )
    if selection_contract != {
        "eligible_filter": "all_non_inferiority_gates_pass",
        "ranking": [
            "assistant_sequence_accuracy_desc",
            "assistant_token_accuracy_desc",
            "mean_loss_asc",
            "completed_steps_asc",
            "checkpoint_sha256_desc",
        ],
    }:
        raise ValueError("SFT checkpoint sweep selection contract is unsupported")

    best = _best_retention_record(records)
    eligible_count = sum(bool(record["retention_eligible"]) for record in records)
    expected_summary = {
        "evaluated_checkpoints": len(records),
        "retention_eligible_checkpoints": eligible_count,
        "failed_checkpoints": len(records) - eligible_count,
        "status": (
            "retention_eligible_checkpoint_found"
            if best is not None
            else "no_retention_eligible_checkpoint"
        ),
        "best_retention_eligible_checkpoint": best,
    }
    summary = _exact_mapping(
        result["summary"],
        _SUMMARY_KEYS,
        label="SFT checkpoint sweep result.summary",
    )
    for key in (
        "evaluated_checkpoints",
        "retention_eligible_checkpoints",
        "failed_checkpoints",
    ):
        _strict_int(summary[key], label=f"SFT checkpoint sweep result.summary.{key}")
    if not _json_equal(summary, expected_summary):
        raise ValueError("SFT checkpoint sweep summary does not match independent selection")
    if best is None:
        raise ValueError("SFT checkpoint sweep selected no retention-eligible archive")
    best = _exact_mapping(best, _BEST_KEYS, label="selected retention archive")
    matches = [
        record
        for record in records
        if {
            "artifact": record["artifact"],
            "checkpoint_step": record["checkpoint_step"],
            "completed_steps": record["completed_steps"],
            "metrics": record["metrics"],
        }
        == best
    ]
    if len(matches) != 1 or not matches[0]["retention_eligible"]:
        raise ValueError(
            "SFT checkpoint sweep must identify exactly one summary-selected eligible archive"
        )
    selected = copy.deepcopy(matches[0])

    training_record = _validated_file_record(
        inputs["training_config"],
        label="SFT checkpoint sweep result.inputs.training_config",
        literal_relative=True,
    )
    training_hash = _require_sha256(
        inputs["training_config_sha256"],
        label="SFT checkpoint sweep result.inputs.training_config_sha256",
    )
    _assert_current_file(
        training_record,
        repository_root=root,
        label="selected SFT training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_payload = _read_regular(
        _resolve_input(root, training_record["path"]),
        label="selected SFT training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_config = _yaml_mapping(training_payload, label="selected SFT training config")
    if _normalized_training_config_sha256(training_config) != training_hash:
        raise ValueError("selected SFT training config canonical hash mismatch")

    sweep_config_record = _validated_file_record(
        inputs["sweep_config"],
        label="SFT checkpoint sweep result.inputs.sweep_config",
        literal_relative=True,
    )
    sweep_config_hash = _require_sha256(
        inputs["sweep_config_sha256"],
        label="SFT checkpoint sweep result.inputs.sweep_config_sha256",
    )
    _assert_current_file(
        sweep_config_record,
        repository_root=root,
        label="bound SFT checkpoint sweep config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    sweep_config_payload = _read_regular(
        _resolve_input(root, sweep_config_record["path"]),
        label="bound SFT checkpoint sweep config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    sweep_config = _yaml_mapping(
        sweep_config_payload,
        label="bound SFT checkpoint sweep config",
    )
    if canonical_sha256(sweep_config) != sweep_config_hash:
        raise ValueError("bound SFT checkpoint sweep config canonical hash mismatch")
    identity = _require_keys(
        result["identity"],
        frozenset(
            {
                "model_config",
                "model_config_sha256",
                "tokenizer",
                "lineage",
            }
        ),
        label="SFT checkpoint sweep result.identity",
    )
    model_record = _validated_file_record(
        identity["model_config"],
        label="SFT checkpoint sweep result.identity.model_config",
        literal_relative=True,
    )
    model_hash = _require_sha256(
        identity["model_config_sha256"],
        label="SFT checkpoint sweep result.identity.model_config_sha256",
    )
    configured_model_path, _configured_model_source = _relative_input(
        root,
        training_config.get("model_config"),
        label="selected SFT training config.model_config",
    )
    if model_record["path"] != configured_model_path:
        raise ValueError("SFT checkpoint sweep and training config reference different model paths")
    _assert_current_file(
        model_record,
        repository_root=root,
        label="bound SFT checkpoint sweep model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    model_payload = _read_regular(
        _resolve_input(root, model_record["path"]),
        label="bound SFT checkpoint sweep model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    model_mapping = _yaml_mapping(
        model_payload,
        label="bound SFT checkpoint sweep model config",
    )
    extra_model_fields = sorted(set(model_mapping) - set(ModelConfig.__dataclass_fields__))
    if extra_model_fields:
        raise ValueError(
            f"bound SFT checkpoint sweep model config has unsupported fields: {extra_model_fields}"
        )
    model_config = ModelConfig(**model_mapping)
    model_config.assert_within_budget()
    if canonical_sha256(model_config.__dict__) != model_hash:
        raise ValueError("bound SFT checkpoint sweep model config canonical hash mismatch")
    tokenizer_record = _validated_sweep_tokenizer_record(
        identity["tokenizer"],
        training_config=training_config,
        repository_root=root,
    )
    if tokenizer_record["vocab_size"] != model_config.vocab_size:
        raise ValueError(
            "SFT checkpoint sweep tokenizer vocabulary disagrees with its model config"
        )
    heldout_source_bindings = _validated_heldout_source_bindings(
        heldout["sources"],
        training_config=training_config,
        repository_root=root,
    )
    lineage = _mapping(identity["lineage"], label="SFT checkpoint sweep result.identity.lineage")
    if (
        _strict_int(
            lineage.get("version"),
            label="SFT checkpoint sweep result.identity.lineage.version",
            minimum=1,
        )
        != 1
        or lineage.get("stage") != "sft"
    ):
        raise ValueError("SFT sweep lineage version/stage is invalid")
    for key in ("config_sha256", "model_config_sha256", "data_sha256", "tokenizer_sha256"):
        _require_sha256(
            lineage.get(key),
            label=f"SFT checkpoint sweep result.identity.lineage.{key}",
        )
    if lineage.get("config_sha256") != training_hash:
        raise ValueError("SFT sweep lineage does not bind the selected training config")
    if lineage.get("model_config_sha256") != model_hash:
        raise ValueError("SFT sweep lineage does not bind the reported model config")
    if lineage.get("tokenizer_sha256") != tokenizer_record["sha256"]:
        raise ValueError("SFT sweep lineage does not bind the reported tokenizer")
    if "parent_checkpoint_sha256" in lineage:
        _require_sha256(
            lineage["parent_checkpoint_sha256"],
            label=("SFT checkpoint sweep result.identity.lineage.parent_checkpoint_sha256"),
        )
    git = _require_keys(
        lineage.get("git"),
        frozenset({"commit", "repository_sha256", "worktree_sha256", "dirty"}),
        label="SFT checkpoint sweep result.identity.lineage.git",
    )
    commit = git["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("SFT checkpoint sweep lineage Git commit is invalid")
    for key in ("repository_sha256", "worktree_sha256"):
        _require_sha256(
            git[key],
            label=f"SFT checkpoint sweep result.identity.lineage.git.{key}",
        )
    if not isinstance(git["dirty"], bool):
        raise TypeError("SFT checkpoint sweep lineage git.dirty must be boolean")
    try:
        json.dumps(lineage, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("SFT sweep lineage contains invalid JSON") from error

    return ValidatedSweep(
        result=copy.deepcopy(result),
        result_artifact=_file_record(source_text, payload),
        candidate_record=selected,
        candidate_artifact=copy.deepcopy(selected["artifact"]),
        training_config_artifact=training_record,
        training_config_sha256=training_hash,
        sweep_config_artifact=sweep_config_record,
        sweep_config_sha256=sweep_config_hash,
        model_config_artifact=model_record,
        model_config_sha256=model_hash,
        tokenizer_record=tokenizer_record,
        heldout_source_bindings=heldout_source_bindings,
        checkpoint_artifacts=tuple(copy.deepcopy(record["artifact"]) for record in records),
        lineage=copy.deepcopy(lineage),
    )


def _assert_sweep_reported_inputs_current(
    sweep: ValidatedSweep,
    *,
    repository_root: Path,
    phase: str,
) -> None:
    checks: list[tuple[Mapping[str, Any], str, int]] = [
        (
            sweep.sweep_config_artifact,
            "SFT checkpoint sweep config",
            _MAX_CONFIG_BYTES,
        ),
        (
            sweep.training_config_artifact,
            "SFT training config",
            _MAX_CONFIG_BYTES,
        ),
        (
            sweep.model_config_artifact,
            "SFT model config",
            _MAX_CONFIG_BYTES,
        ),
    ]
    tokenizer_artifact = sweep.tokenizer_record.get("artifact")
    if tokenizer_artifact is not None:
        checks.append(
            (
                tokenizer_artifact,
                "SFT tokenizer artifact",
                _MAX_CONFIG_BYTES * 16,
            )
        )
    for source_index, source in enumerate(sweep.heldout_source_bindings):
        for name, maximum in (
            ("jsonl", _MAX_CASE_BYTES),
            ("manifest", _MAX_CONFIG_BYTES),
            ("generator_config", _MAX_CONFIG_BYTES),
        ):
            checks.append(
                (
                    source["files"][name],
                    f"SFT heldout source {source_index} {name}",
                    maximum,
                )
            )
    for index, artifact in enumerate(sweep.checkpoint_artifacts):
        checks.append(
            (
                artifact,
                f"SFT checkpoint archive {index}",
                _MAX_CHECKPOINT_BYTES,
            )
        )
    for record, label, maximum in checks:
        _assert_current_file(
            record,
            repository_root=repository_root,
            label=f"bound {label} {phase}",
            max_bytes=maximum,
        )


def _replay_sweep_and_match(
    sweep: ValidatedSweep,
    *,
    repository_root: Path,
) -> None:
    """Rerun the sealed sweep and require a byte-for-byte-equivalent canonical result."""

    _assert_sweep_reported_inputs_current(
        sweep,
        repository_root=repository_root,
        phase="before replay",
    )
    with _at_repository_root(repository_root):
        replayed = run_sft_checkpoint_sweep(sweep.sweep_config_artifact["path"])
    if not isinstance(replayed, Mapping):
        raise TypeError("replayed SFT checkpoint sweep did not return a mapping")
    try:
        json.dumps(replayed, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("replayed SFT checkpoint sweep returned invalid JSON") from error
    if not _json_equal(replayed, sweep.result):
        raise ValueError("replayed SFT checkpoint sweep does not match the supplied sealed result")
    _assert_sweep_reported_inputs_current(
        sweep,
        repository_root=repository_root,
        phase="after replay",
    )


def _selection_contract(
    conversations: Sequence[Any],
    value: Any,
    *,
    label: str,
) -> tuple[Sequence[Any], dict[str, Any] | None]:
    if value is None:
        return conversations, None
    keys = frozenset(
        {
            "algorithm",
            "max_rows",
            "expected_source_rows",
            "expected_source_assistant_decisions",
            "expected_source_semantic_set_sha256",
            "expected_selected_rows",
            "expected_selected_assistant_decisions",
            "expected_selected_semantic_set_sha256",
            "expected_audit_sha256",
        }
    )
    contract = _exact_mapping(value, keys, label=label)
    if contract["algorithm"] != STRATIFIED_EVAL_ALGORITHM:
        raise ValueError(f"{label}.algorithm is unsupported")
    max_rows = _strict_int(contract["max_rows"], label=f"{label}.max_rows", minimum=1)
    for key in (
        "expected_source_rows",
        "expected_source_assistant_decisions",
        "expected_selected_rows",
        "expected_selected_assistant_decisions",
    ):
        _strict_int(contract[key], label=f"{label}.{key}", minimum=1)
    for key in (
        "expected_source_semantic_set_sha256",
        "expected_selected_semantic_set_sha256",
        "expected_audit_sha256",
    ):
        _require_sha256(contract[key], label=f"{label}.{key}")
    selection = select_stratified_eval_subset(conversations, max_rows=max_rows)
    audit = selection.audit.as_dict()
    observed = {
        "algorithm": audit["algorithm"],
        "max_rows": audit["capacity"]["max_rows"],
        "expected_source_rows": audit["source"]["rows"],
        "expected_source_assistant_decisions": audit["source"]["assistant_decisions"],
        "expected_source_semantic_set_sha256": audit["source"]["semantic_set_sha256"],
        "expected_selected_rows": audit["selected"]["rows"],
        "expected_selected_assistant_decisions": audit["selected"]["assistant_decisions"],
        "expected_selected_semantic_set_sha256": audit["selected"]["semantic_set_sha256"],
        "expected_audit_sha256": audit["audit_sha256"],
    }
    if observed != contract:
        raise ValueError(
            f"{label} does not match the independently reconstructed deterministic selection"
        )
    return selection.conversations, audit


def _semantic_set_sha256(conversations: Sequence[Any]) -> str:
    identities = sorted({conversation_semantic_sha256(item) for item in conversations})
    return _sha256(("\n".join(identities)).encode("ascii"))


def _case_set(conversations: Sequence[Any]) -> dict[str, Any]:
    result = score_conversations(
        conversations,
        lambda _prompt, _tools: AssistantPrediction(
            text="",
            finish_reason="caller_complete",
        ),
    )
    case_set = _exact_mapping(
        result["case_set"],
        frozenset(
            {
                "sha256",
                "conversations",
                "assistant_decisions",
                "tool_decisions",
                "no_tool_decisions",
            }
        ),
        label="scorecard case set",
    )
    _require_sha256(case_set["sha256"], label="scorecard case set.sha256")
    for key in (
        "conversations",
        "assistant_decisions",
        "tool_decisions",
        "no_tool_decisions",
    ):
        _strict_int(
            case_set[key],
            label=f"scorecard case set.{key}",
            minimum=0 if key in {"tool_decisions", "no_tool_decisions"} else 1,
        )
    if (
        case_set["tool_decisions"] + case_set["no_tool_decisions"]
        != case_set["assistant_decisions"]
    ):
        raise RuntimeError("scorecard case-set decision accounting is inconsistent")
    return case_set


def _case_binding(
    cases_config: Any,
    *,
    repository_root: Path,
    label: str,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    cases = _mapping(cases_config, label=f"{label}.config")
    required = {
        "path",
        "manifest",
        "generator_config",
        "expected_split",
        "expected_rule_verified",
        "environment_policy",
    }
    extra = sorted(set(cases) - required - {"selection"})
    missing = sorted(required - set(cases))
    if missing or extra:
        raise ValueError(f"{label}.config keys mismatch: missing={missing}, extra={extra}")
    case_paths: dict[str, Path] = {}
    for key in ("path", "manifest", "generator_config"):
        literal, resolved = _relative_input(
            repository_root,
            cases[key],
            label=f"{label}.config.{key}",
        )
        cases[key] = literal
        case_paths[key] = resolved
    if cases["expected_split"] != "eval":
        raise ValueError(f"{label} must use the eval split")
    if cases["expected_rule_verified"] is not True:
        raise ValueError(f"{label} must require rule-verified cases")
    if cases["environment_policy"] != "forbid":
        raise ValueError(f"{label} must forbid environment-executed cases")
    if "selection" in cases and cases["selection"] is None:
        raise TypeError(f"{label}.config.selection cannot be null")

    artifact = load_verified_conversation_artifact(
        case_paths["path"],
        manifest_path=case_paths["manifest"],
        config_path=case_paths["generator_config"],
        expected_split="eval",
        expected_rule_verified=True,
        environment_policy="forbid",
        max_jsonl_bytes=_MAX_CASE_BYTES,
    )
    selected, selection_audit = _selection_contract(
        artifact.conversations,
        cases.get("selection"),
        label=f"{label}.config.selection",
    )
    case_set = _case_set(selected)
    semantic_set_sha256 = _semantic_set_sha256(selected)
    if selection_audit is not None:
        expected_semantic = selection_audit["selected"]["semantic_set_sha256"]
        if semantic_set_sha256 != expected_semantic:
            raise RuntimeError(f"{label} selected semantic identity is inconsistent")
    elif case_set["conversations"] != len(artifact.conversations):
        raise RuntimeError(f"{label} unexpectedly changed an unselected case set")

    expected_provenance = {
        **artifact.lineage_identity(),
        "case_set_sha256": case_set["sha256"],
        "rule_verified": True,
        "environment_executed": False,
        **({"selection": selection_audit} if selection_audit is not None else {}),
    }
    binding = {
        "config_contract": copy.deepcopy(cases),
        "files": {
            "jsonl": {
                "path": cases["path"],
                **artifact.identity.jsonl.as_dict(),
            },
            "manifest": {
                "path": cases["manifest"],
                **artifact.identity.sidecar.as_dict(),
            },
            "generator_config": {
                "path": cases["generator_config"],
                **artifact.identity.generator_config.as_dict(),
            },
        },
        "artifact_identity": artifact.lineage_identity(),
        "selection": {
            "configured": copy.deepcopy(cases.get("selection")),
            "audit": copy.deepcopy(selection_audit),
        },
        "selected_semantic_set_sha256": semantic_set_sha256,
        "case_set": case_set,
        "expected_result_provenance": expected_provenance,
        "manifest_self_sha256": artifact.identity.manifest_self_sha256,
    }
    return binding, tuple(selected)


def _direct_dev_confirm_overlap(
    development_conversations: Sequence[Any],
    confirmatory_conversations: Sequence[Any],
) -> dict[str, Any]:
    """Recompute both overlap dimensions from parsed rows under the production prompt contract."""

    audit = audit_conversation_overlap(
        development_conversations,
        confirmatory_conversations,
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    )
    if not audit.clean:
        raise ValueError(
            "development and confirmatory case artifacts overlap: "
            f"semantic={len(audit.semantic_overlap_sha256)}, "
            f"rendered_prompts={len(audit.rendered_prompt_overlap_sha256)}"
        )
    value = audit.as_dict()
    if value["semantic_overlap"] != 0 or value["rendered_prompt_overlap"] != 0:
        raise RuntimeError("clean development/confirmatory overlap audit is inconsistent")
    return value


def _validate_overlap_audit(
    value: Any,
    *,
    label: str,
    development: Mapping[str, Any],
    confirmatory: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _require_keys(
        value,
        frozenset(
            {
                "left_rows",
                "right_rows",
                "left_semantic_set_sha256",
                "right_semantic_set_sha256",
                "semantic_overlap",
                "semantic_overlap_sha256",
                "rendered_prompt_overlap",
                "rendered_prompt_overlap_sha256",
            }
        ),
        label=label,
    )
    for key in ("left_rows", "right_rows", "semantic_overlap", "rendered_prompt_overlap"):
        _strict_int(audit[key], label=f"{label}.{key}")
    for key in ("left_semantic_set_sha256", "right_semantic_set_sha256"):
        _require_sha256(audit[key], label=f"{label}.{key}")
    for key in ("semantic_overlap_sha256", "rendered_prompt_overlap_sha256"):
        if not isinstance(audit[key], list) or audit[key]:
            raise ValueError(f"{label}.{key} must be an empty list")
    expected = {
        "left_rows": development["case_set"]["conversations"],
        "right_rows": confirmatory["case_set"]["conversations"],
        "left_semantic_set_sha256": development["selected_semantic_set_sha256"],
        "right_semantic_set_sha256": confirmatory["selected_semantic_set_sha256"],
        "semantic_overlap": 0,
        "rendered_prompt_overlap": 0,
    }
    for key, expected_value in expected.items():
        if audit[key] != expected_value:
            raise ValueError(f"{label}.{key} does not match the bound case sets")
    return copy.deepcopy(audit)


def _validate_confirmatory_provenance(
    path_text: str,
    *,
    repository_root: Path,
    confirmatory_cases: Mapping[str, Any],
    development_cases: Mapping[str, Any],
) -> dict[str, Any]:
    path_text, path = _relative_input(
        repository_root,
        path_text,
        label="confirmatory split provenance receipt path",
    )
    receipt, payload = _load_canonical_json_file(
        path,
        label="confirmatory split provenance receipt",
    )
    schema_version = receipt.get("schema_version")
    if (
        receipt.get("kind") != "localagent_confirmatory_eval_split_receipt"
        or type(schema_version) is not int
        or schema_version != 2
    ):
        raise ValueError("confirmatory split provenance receipt kind/schema is invalid")
    self_sha256 = _require_sha256(
        receipt.get("receipt_self_sha256"),
        label="confirmatory split provenance receipt.receipt_self_sha256",
    )
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_self_sha256")
    if _sha256(canonical_json_bytes(unsigned)) != self_sha256:
        raise ValueError("confirmatory split provenance receipt self-hash mismatch")

    config = _validated_file_record(
        receipt.get("config"),
        label="confirmatory split provenance receipt.config",
        literal_relative=True,
    )
    if not _json_equal(config, confirmatory_cases["files"]["generator_config"]):
        raise ValueError("confirmatory provenance generator-config identity mismatch")
    output = _require_keys(
        receipt.get("output"),
        frozenset({"jsonl", "manifest", "rows"}),
        label="confirmatory split provenance receipt.output",
    )
    jsonl = _validated_file_record(
        output["jsonl"],
        label="confirmatory split provenance receipt.output.jsonl",
        literal_relative=True,
    )
    manifest_raw = _require_keys(
        output["manifest"],
        frozenset({"path", "bytes", "sha256", "manifest_self_sha256"}),
        label="confirmatory split provenance receipt.output.manifest",
    )
    manifest = _validated_file_record(
        {key: manifest_raw[key] for key in _FILE_RECORD_KEYS},
        label="confirmatory split provenance receipt.output.manifest",
        literal_relative=True,
    )
    manifest_self = _require_sha256(
        manifest_raw["manifest_self_sha256"],
        label="confirmatory split provenance receipt.output.manifest.manifest_self_sha256",
    )
    rows = _strict_int(
        output["rows"],
        label="confirmatory split provenance receipt.output.rows",
        minimum=1,
    )
    if not _json_equal(jsonl, confirmatory_cases["files"]["jsonl"]):
        raise ValueError("confirmatory provenance JSONL identity mismatch")
    if not _json_equal(manifest, confirmatory_cases["files"]["manifest"]):
        raise ValueError("confirmatory provenance manifest identity mismatch")
    if manifest_self != confirmatory_cases["manifest_self_sha256"]:
        raise ValueError("confirmatory provenance manifest self-hash mismatch")
    if rows != confirmatory_cases["case_set"]["conversations"]:
        raise ValueError("confirmatory provenance row count mismatch")

    filtered = _require_keys(
        receipt.get("filtered_selection"),
        frozenset({"algorithm", "audit_sha256", "source", "selected"}),
        label="confirmatory split provenance receipt.filtered_selection",
    )
    if filtered["algorithm"] != STRATIFIED_EVAL_ALGORITHM:
        raise ValueError("confirmatory filtered-selection algorithm is unsupported")
    filtered_audit = _require_sha256(
        filtered["audit_sha256"],
        label="confirmatory filtered selection audit_sha256",
    )
    source = _require_keys(
        filtered["source"],
        frozenset({"rows", "assistant_decisions", "semantic_set_sha256"}),
        label="confirmatory filtered selection source",
    )
    selected = _require_keys(
        filtered["selected"],
        frozenset({"rows", "assistant_decisions", "semantic_set_sha256"}),
        label="confirmatory filtered selection selected",
    )
    for container_label, container in (("source", source), ("selected", selected)):
        _strict_int(
            container["rows"],
            label=f"confirmatory filtered selection {container_label}.rows",
            minimum=1,
        )
        _strict_int(
            container["assistant_decisions"],
            label=(f"confirmatory filtered selection {container_label}.assistant_decisions"),
            minimum=1,
        )
        _require_sha256(
            container["semantic_set_sha256"],
            label=(f"confirmatory filtered selection {container_label}.semantic_set_sha256"),
        )
    expected_selected = {
        "rows": confirmatory_cases["case_set"]["conversations"],
        "assistant_decisions": confirmatory_cases["case_set"]["assistant_decisions"],
        "semantic_set_sha256": confirmatory_cases["selected_semantic_set_sha256"],
    }
    if {key: selected[key] for key in expected_selected} != expected_selected:
        raise ValueError("confirmatory filtered-selection identity mismatch")

    development_selection = _require_keys(
        receipt.get("development_selection"),
        frozenset({"algorithm", "audit_sha256", "selected"}),
        label="confirmatory split provenance receipt.development_selection",
    )
    configured_development = development_cases["selection"]["audit"]
    if configured_development is None:
        raise ValueError("development scorecard must retain its frozen selection contract")
    if (
        development_selection["algorithm"] != configured_development["algorithm"]
        or development_selection["audit_sha256"] != configured_development["audit_sha256"]
    ):
        raise ValueError("confirmatory provenance development-selection audit mismatch")
    provenance_development_selected = _require_keys(
        development_selection["selected"],
        frozenset({"rows", "assistant_decisions", "semantic_set_sha256"}),
        label="confirmatory provenance development selection selected",
    )
    for key in ("rows", "assistant_decisions"):
        _strict_int(
            provenance_development_selected[key],
            label=f"confirmatory provenance development selection selected.{key}",
            minimum=1,
        )
    _require_sha256(
        provenance_development_selected["semantic_set_sha256"],
        label=("confirmatory provenance development selection selected.semantic_set_sha256"),
    )
    expected_development_selected = {
        "rows": development_cases["case_set"]["conversations"],
        "assistant_decisions": development_cases["case_set"]["assistant_decisions"],
        "semantic_set_sha256": development_cases["selected_semantic_set_sha256"],
    }
    if {
        key: provenance_development_selected[key] for key in expected_development_selected
    } != expected_development_selected:
        raise ValueError("confirmatory provenance development case-set identity mismatch")

    reference = _require_keys(
        receipt.get("reference_contract"),
        frozenset(
            {
                "confirm_rows",
                "confirm_assistant_decisions",
                "confirm_semantic_set_sha256",
                "primary_selected_semantic_set_sha256",
                "inner_filtered_selection_audit_sha256",
                "prompt_contract",
            }
        ),
        label="confirmatory split provenance receipt.reference_contract",
    )
    _strict_int(
        reference["confirm_rows"],
        label="confirmatory reference contract.confirm_rows",
        minimum=1,
    )
    _strict_int(
        reference["confirm_assistant_decisions"],
        label="confirmatory reference contract.confirm_assistant_decisions",
        minimum=1,
    )
    for key in (
        "confirm_semantic_set_sha256",
        "primary_selected_semantic_set_sha256",
        "inner_filtered_selection_audit_sha256",
    ):
        _require_sha256(
            reference[key],
            label=f"confirmatory reference contract.{key}",
        )
    if (
        reference["confirm_rows"] != expected_selected["rows"]
        or reference["confirm_assistant_decisions"] != expected_selected["assistant_decisions"]
        or reference["confirm_semantic_set_sha256"] != expected_selected["semantic_set_sha256"]
        or reference["primary_selected_semantic_set_sha256"]
        != expected_development_selected["semantic_set_sha256"]
        or reference["inner_filtered_selection_audit_sha256"] != filtered_audit
        or reference["prompt_contract"] != OPENAI_FULL_CATALOG_V1
    ):
        raise ValueError("confirmatory reference contract does not match the bound case sets")
    reference_sha256 = _require_sha256(
        receipt.get("reference_contract_sha256"),
        label="confirmatory split provenance receipt.reference_contract_sha256",
    )
    if reference_sha256 != _sha256(canonical_json_bytes(reference)):
        raise ValueError("confirmatory reference contract SHA-256 mismatch")
    overlap = _require_keys(
        receipt.get("overlap_evidence"),
        frozenset({"development"}),
        label="confirmatory split provenance receipt.overlap_evidence",
    )
    development_overlap = _validate_overlap_audit(
        overlap["development"],
        label="confirmatory development overlap evidence",
        development=development_cases,
        confirmatory=confirmatory_cases,
    )
    manifest_path = _resolve_input(
        repository_root,
        confirmatory_cases["files"]["manifest"]["path"],
    )
    manifest_payload = _read_regular(
        manifest_path,
        label="confirmatory case manifest",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    manifest_value = _strict_json(manifest_payload, label="confirmatory case manifest")
    if manifest_payload != canonical_json_bytes(manifest_value):
        raise ValueError("confirmatory case manifest must use canonical JSON bytes")
    coverage = _require_keys(
        manifest_value.get("coverage_contract"),
        frozenset({"confirmatory_eval_split"}),
        label="confirmatory case manifest.coverage_contract",
    )
    manifest_contract = _require_keys(
        coverage["confirmatory_eval_split"],
        frozenset(
            {
                "filtered_selection",
                "reference_contract",
                "reference_contract_sha256",
            }
        ),
        label="confirmatory case manifest confirmatory selection contract",
    )
    manifest_filtered = _require_keys(
        manifest_contract["filtered_selection"],
        frozenset({"algorithm", "audit_sha256", "source", "selected"}),
        label="confirmatory case manifest filtered selection",
    )
    manifest_source = _require_keys(
        manifest_filtered["source"],
        frozenset({"rows", "assistant_decisions", "semantic_set_sha256"}),
        label="confirmatory case manifest filtered source",
    )
    manifest_selected = _require_keys(
        manifest_filtered["selected"],
        frozenset({"rows", "assistant_decisions", "semantic_set_sha256"}),
        label="confirmatory case manifest filtered selected",
    )
    manifest_filtered_identity = {
        "algorithm": manifest_filtered["algorithm"],
        "audit_sha256": manifest_filtered["audit_sha256"],
        "source": {
            key: manifest_source[key]
            for key in ("rows", "assistant_decisions", "semantic_set_sha256")
        },
        "selected": {
            key: manifest_selected[key]
            for key in ("rows", "assistant_decisions", "semantic_set_sha256")
        },
    }
    expected_filtered_identity = {
        "algorithm": filtered["algorithm"],
        "audit_sha256": filtered_audit,
        "source": {
            key: source[key] for key in ("rows", "assistant_decisions", "semantic_set_sha256")
        },
        "selected": copy.deepcopy(expected_selected),
    }
    if not _json_equal(manifest_filtered_identity, expected_filtered_identity):
        raise ValueError(
            "confirmatory manifest and provenance filtered-selection identities disagree"
        )
    if (
        not _json_equal(manifest_contract["reference_contract"], reference)
        or manifest_contract["reference_contract_sha256"] != reference_sha256
    ):
        raise ValueError("confirmatory manifest and provenance reference contracts disagree")
    split_contract = _require_keys(
        manifest_value.get("split_contract"),
        frozenset({"confirmatory_eval_split"}),
        label="confirmatory case manifest.split_contract",
    )
    manifest_split = _require_keys(
        split_contract["confirmatory_eval_split"],
        frozenset({"development_overlap"}),
        label="confirmatory case manifest split contract",
    )
    manifest_development_overlap = _validate_overlap_audit(
        manifest_split["development_overlap"],
        label="confirmatory case manifest development overlap",
        development=development_cases,
        confirmatory=confirmatory_cases,
    )
    overlap_identity_keys = (
        "left_rows",
        "right_rows",
        "left_semantic_set_sha256",
        "right_semantic_set_sha256",
        "semantic_overlap",
        "semantic_overlap_sha256",
        "rendered_prompt_overlap",
        "rendered_prompt_overlap_sha256",
    )
    if {key: manifest_development_overlap[key] for key in overlap_identity_keys} != {
        key: development_overlap[key] for key in overlap_identity_keys
    }:
        raise ValueError(
            "confirmatory manifest and provenance development-overlap identities disagree"
        )

    return {
        "artifact": _file_record(path_text, payload),
        "receipt_self_sha256": self_sha256,
        "config": config,
        "output": {
            "jsonl": jsonl,
            "manifest": {**manifest, "manifest_self_sha256": manifest_self},
            "rows": rows,
        },
        "development_selection": {
            "algorithm": development_selection["algorithm"],
            "audit_sha256": development_selection["audit_sha256"],
            "selected": copy.deepcopy(provenance_development_selected),
        },
        "filtered_selection": {
            "algorithm": filtered["algorithm"],
            "audit_sha256": filtered_audit,
            "source": {
                key: source[key] for key in ("rows", "assistant_decisions", "semantic_set_sha256")
            },
            "selected": copy.deepcopy(expected_selected),
        },
        "reference_contract": copy.deepcopy(reference),
        "reference_contract_sha256": reference_sha256,
        "development_overlap": development_overlap,
        "manifest_contract": {
            "filtered_selection": manifest_filtered_identity,
            "reference_contract": copy.deepcopy(reference),
            "reference_contract_sha256": reference_sha256,
            "development_overlap": {
                key: manifest_development_overlap[key] for key in overlap_identity_keys
            },
        },
    }


def _scorecard_config(value: Any, *, label: str) -> dict[str, Any]:
    keys = frozenset(
        {
            "kind",
            "schema_version",
            "checkpoint",
            "training_config",
            "model_config",
            "tokenizer",
            "cases",
            "generation",
        }
    )
    config = _exact_mapping(value, keys, label=label)
    schema_version = config["schema_version"]
    if (
        config["kind"] != SCORECARD_CONFIG_KIND
        or type(schema_version) is not int
        or schema_version != SCORECARD_SCHEMA_VERSION
    ):
        raise ValueError(f"{label} kind/schema is invalid")
    return config


def _model_and_tokenizer_binding(
    config: Mapping[str, Any],
    *,
    training_config: Mapping[str, Any],
    training_record: Mapping[str, Any],
    training_sha256: str,
    repository_root: Path,
) -> dict[str, Any]:
    model_path_text, model_path = _relative_input(
        repository_root,
        config["model_config"],
        label="scorecard model_config",
    )
    referenced_model, referenced_model_path = _relative_input(
        repository_root,
        training_config.get("model_config"),
        label="selected training config.model_config",
    )
    if referenced_model != model_path_text:
        raise ValueError("scorecard and selected training config reference different model paths")
    model_payload = _read_regular(
        model_path,
        label="scorecard model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    model_mapping = _yaml_mapping(model_payload, label="scorecard model config")
    extra = sorted(set(model_mapping) - set(ModelConfig.__dataclass_fields__))
    if extra:
        raise ValueError(f"scorecard model config has unsupported fields: {extra}")
    model_config = ModelConfig(**model_mapping)
    model_config.assert_within_budget()
    model_canonical_sha256 = canonical_sha256(model_config.__dict__)
    referenced_model_payload = _read_regular(
        referenced_model_path,
        label="selected training config referenced model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    if referenced_model_payload != model_payload:
        raise ValueError("scorecard and selected training config reference different model configs")

    tokenizer_config = _mapping(config["tokenizer"], label="scorecard tokenizer")
    if set(tokenizer_config) - {"kind", "path"}:
        raise ValueError("scorecard tokenizer contains unsupported fields")
    kind = tokenizer_config.get("kind")
    data = _mapping(training_config.get("data"), label="selected training config.data")
    if data.get("conversation_prompt_contract") != OPENAI_FULL_CATALOG_V1:
        raise ValueError("selected training config uses the wrong conversation prompt contract")
    training_tokenizer = _mapping(
        data.get("tokenizer", {"kind": "byte"}),
        label="selected training config.data.tokenizer",
    )
    if set(training_tokenizer) - {"kind", "path"}:
        raise ValueError("selected training config tokenizer contains unsupported fields")
    if training_tokenizer.get("kind", "byte") != kind:
        raise ValueError("scorecard and selected training config tokenizer kinds differ")

    runtime_package: dict[str, str] | None = None
    tokenizer_payload: bytes | None = None
    if kind == "byte":
        if tokenizer_config.get("path") is not None or training_tokenizer.get("path") is not None:
            raise ValueError("byte tokenizer must not declare a path in either config")
        tokenizer = ByteTokenizer()
        tokenizer_record = tokenizer_identity("byte", vocab_size=tokenizer.vocab_size)
        tokenizer_artifact = None
    elif kind == "bpe":
        tokenizer_path_text, tokenizer_path = _relative_input(
            repository_root,
            tokenizer_config.get("path"),
            label="scorecard tokenizer.path",
        )
        referenced_tokenizer, referenced_tokenizer_path = _relative_input(
            repository_root,
            training_tokenizer.get("path"),
            label="selected training config.data.tokenizer.path",
        )
        if referenced_tokenizer != tokenizer_path_text:
            raise ValueError(
                "scorecard and selected training config reference different tokenizer paths"
            )
        tokenizer_payload = _read_regular(
            tokenizer_path,
            label="scorecard BPE tokenizer",
            max_bytes=_MAX_CONFIG_BYTES * 16,
        )
        try:
            import tokenizers as tokenizers_package
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError("BPE scorecard preparation requires tokenizers") from error
        try:
            tokenizer = BPETokenizer(
                Tokenizer.from_str(tokenizer_payload.decode("utf-8", errors="strict"))
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("scorecard BPE tokenizer is not a valid UTF-8 artifact") from error
        tokenizer_record = {
            "kind": "bpe",
            "vocab_size": tokenizer.vocab_size,
            "artifact": {
                "bytes": len(tokenizer_payload),
                "sha256": _sha256(tokenizer_payload),
            },
            "sha256": _sha256(tokenizer_payload),
        }
        package_version = getattr(tokenizers_package, "__version__", None)
        if not isinstance(package_version, str) or not package_version:
            raise RuntimeError("tokenizers package has no usable version")
        runtime_package = {"name": "tokenizers", "version": package_version}
        tokenizer_artifact = _file_record(tokenizer_path_text, tokenizer_payload)
        referenced_tokenizer_payload = _read_regular(
            referenced_tokenizer_path,
            label="selected training config referenced tokenizer",
            max_bytes=_MAX_CONFIG_BYTES * 16,
        )
        if referenced_tokenizer_payload != tokenizer_payload:
            raise ValueError(
                "scorecard and selected training config reference different tokenizers"
            )
    else:
        raise ValueError("scorecard tokenizer.kind must be byte or bpe")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("scorecard tokenizer vocabulary does not match the model config")

    model = LocalAgentLM(model_config)
    model_parameters = model.num_params()
    del model
    return {
        "training_config": {
            **copy.deepcopy(dict(training_record)),
            "canonical_sha256": training_sha256,
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
        },
        "model_config": {
            **_file_record(model_path_text, model_payload),
            "canonical_sha256": model_canonical_sha256,
            "name": model_config.name,
            "parameters": model_parameters,
        },
        "tokenizer": {
            "kind": kind,
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer_record["sha256"],
            "runtime_package": runtime_package,
            "artifact": tokenizer_artifact,
        },
    }


def _assert_scorecard_static_matches_sweep(
    static: Mapping[str, Any],
    sweep: ValidatedSweep,
) -> None:
    model = _mapping(static["model_config"], label="bound scorecard model config")
    model_artifact = {key: model[key] for key in ("path", "bytes", "sha256")}
    if (
        not _json_equal(model_artifact, sweep.model_config_artifact)
        or model["canonical_sha256"] != sweep.model_config_sha256
    ):
        raise ValueError("scorecard and SFT checkpoint sweep reference different model configs")
    tokenizer = _mapping(static["tokenizer"], label="bound scorecard tokenizer")
    tokenizer_identity_record = {key: tokenizer[key] for key in ("kind", "vocab_size", "sha256")}
    if tokenizer["artifact"] is not None:
        tokenizer_identity_record["artifact"] = copy.deepcopy(tokenizer["artifact"])
    if not _json_equal(tokenizer_identity_record, sweep.tokenizer_record):
        raise ValueError("scorecard and SFT checkpoint sweep reference different tokenizers")


def _evaluator_module_binding() -> dict[str, Any]:
    sources = {
        "agent_scorecard": Path(agent_scorecard_module.__file__).resolve(),
        "prompt_contract": Path(agent_scorecard_module.prompt_contract_module.__file__).resolve(),
        "stratified_eval_selector": Path(
            agent_scorecard_module.stratified_eval_selector_module.__file__
        ).resolve(),
        "tool_eval": Path(agent_scorecard_module.__file__).with_name("tool_eval.py").resolve(),
    }
    return {
        name: _file_record(
            str(path),
            _read_regular(path, label=f"evaluator module {name}", max_bytes=_MAX_CONFIG_BYTES),
        )
        for name, path in sorted(sources.items())
    }


def _serialize_scorecard_config(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config),
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")


def _expected_scorecard_provenance(
    *,
    config: Mapping[str, Any],
    config_record: Mapping[str, Any],
    sweep: ValidatedSweep,
    cases: Mapping[str, Any],
    static: Mapping[str, Any],
    evaluator_modules: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evaluator_modules": copy.deepcopy(dict(evaluator_modules)),
        "scorecard_config": {
            **copy.deepcopy(dict(config_record)),
            "canonical_sha256": canonical_sha256(config),
        },
        "checkpoint": {
            **copy.deepcopy(sweep.candidate_artifact),
            "stage": "sft",
            "step": sweep.candidate_record["checkpoint_step"],
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
        },
        "checkpoint_lineage": copy.deepcopy(sweep.lineage),
        "training_config": copy.deepcopy(static["training_config"]),
        "model_config": copy.deepcopy(static["model_config"]),
        "tokenizer": copy.deepcopy(static["tokenizer"]),
        "training_corpus": {
            "checkpoint_lineage_data_sha256": sweep.lineage.get("data_sha256"),
            "independently_reconstructed_by_scorecard": False,
        },
        "cases": copy.deepcopy(cases["expected_result_provenance"]),
    }


def _reject_symlink_ancestors(path: Path, *, label: str = "output path") -> None:
    current = path
    while True:
        try:
            observed = current.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"{label} has a symlink ancestor: {current}")
            if current == path and not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"{label} parent is not a directory: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _prepare_output_path(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        observed = None
    if observed is not None:
        kind = "symlink" if stat.S_ISLNK(observed.st_mode) else "existing path"
        raise FileExistsError(f"refusing to replace {kind}: {path}")
    _reject_symlink_ancestors(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(path.parent)


def _publish_bundle(entries: Sequence[tuple[Path, bytes]]) -> None:
    if not entries:
        raise ValueError("output bundle must not be empty")
    normalized = [path.absolute() for path, _payload in entries]
    if len(set(normalized)) != len(normalized):
        raise ValueError("output bundle contains duplicate destinations")
    for path, _payload in entries:
        _prepare_output_path(path)

    prepared: list[tuple[Path, int, str, os.stat_result]] = []
    published: list[tuple[Path, int, os.stat_result]] = []
    try:
        for path, payload in entries:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            try:
                directory_state = os.fstat(directory_descriptor)
                visible_parent = path.parent.lstat()
            except BaseException:
                os.close(directory_descriptor)
                raise
            if (
                not stat.S_ISDIR(directory_state.st_mode)
                or directory_state.st_dev != visible_parent.st_dev
                or directory_state.st_ino != visible_parent.st_ino
            ):
                os.close(directory_descriptor)
                raise RuntimeError(f"output parent changed while being opened: {path.parent}")
            temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_descriptor = os.open(
                    temporary_name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_descriptor,
                )
            except BaseException:
                os.close(directory_descriptor)
                raise
            try:
                try:
                    offset = 0
                    while offset < len(payload):
                        offset += os.write(temporary_descriptor, payload[offset:])
                    os.fsync(temporary_descriptor)
                    os.lseek(temporary_descriptor, 0, os.SEEK_SET)
                    chunks: list[bytes] = []
                    while chunk := os.read(temporary_descriptor, 1024 * 1024):
                        chunks.append(chunk)
                    if b"".join(chunks) != payload:
                        raise RuntimeError(
                            f"temporary output verification failed: {path.parent / temporary_name}"
                        )
                    temporary_state = os.fstat(temporary_descriptor)
                finally:
                    os.close(temporary_descriptor)
            except BaseException:
                try:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
                finally:
                    os.close(directory_descriptor)
                raise
            prepared.append((path, directory_descriptor, temporary_name, temporary_state))

        for destination, directory_descriptor, temporary_name, temporary_state in prepared:
            try:
                os.link(
                    temporary_name,
                    destination.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing concurrently created output: {destination}"
                ) from error
            destination_state = os.stat(
                destination.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            published.append((destination, directory_descriptor, destination_state))
            if (
                temporary_state.st_dev != destination_state.st_dev
                or temporary_state.st_ino != destination_state.st_ino
                or not stat.S_ISREG(destination_state.st_mode)
            ):
                raise RuntimeError(
                    f"published output is not the prepared regular file: {destination}"
                )
            _reject_symlink_ancestors(destination.parent)
            visible_destination = destination.lstat()
            if (
                visible_destination.st_dev != destination_state.st_dev
                or visible_destination.st_ino != destination_state.st_ino
            ):
                raise RuntimeError(f"output path changed while being published: {destination}")
            os.fsync(directory_descriptor)
    except BaseException:
        for destination, directory_descriptor, expected in reversed(published):
            try:
                observed = os.stat(
                    destination.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if observed.st_dev == expected.st_dev and observed.st_ino == expected.st_ino:
                os.unlink(destination.name, dir_fd=directory_descriptor)
        raise
    finally:
        for _path, directory_descriptor, temporary_name, _state in prepared:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
            finally:
                os.close(directory_descriptor)


def _self_hashed_receipt(core: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if field in core:
        raise ValueError(f"unsigned receipt must not contain {field}")
    value = copy.deepcopy(dict(core))
    value[field] = _sha256(canonical_json_bytes(value))
    return value


def prepare_sft_candidate(
    sweep_result_path: str | Path,
    development_config_output: str | Path,
    confirmatory_config_output: str | Path,
    binding_output: str | Path,
    *,
    base_scorecard_config: str | Path = DEFAULT_BASE_SCORECARD_CONFIG,
    confirmatory_cases: str | Path = DEFAULT_CONFIRMATORY_CASES,
    confirmatory_manifest: str | Path = DEFAULT_CONFIRMATORY_MANIFEST,
    confirmatory_generator_config: str | Path = DEFAULT_CONFIRMATORY_GENERATOR_CONFIG,
    confirmatory_provenance: str | Path = DEFAULT_CONFIRMATORY_PROVENANCE,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Replay the sweep, then prepare two scorecard configs and a sealed candidate binding."""

    root = Path(repository_root)
    raw_outputs = {
        "development": development_config_output,
        "confirmatory": confirmatory_config_output,
        "binding": binding_output,
    }
    output_texts: dict[str, str] = {}
    output_paths: dict[str, Path] = {}
    for name, value in raw_outputs.items():
        literal, resolved = _relative_input(
            root,
            str(value),
            label=f"{name} output path",
        )
        output_texts[name] = literal
        output_paths[name] = resolved
    for path in output_paths.values():
        _prepare_output_path(path)

    sweep = load_validated_sweep_result(sweep_result_path, repository_root=root)
    _replay_sweep_and_match(sweep, repository_root=root)
    base_text, base_path = _relative_input(
        root,
        str(base_scorecard_config),
        label="base development scorecard config path",
    )
    base_payload = _read_regular(
        base_path,
        label="base development scorecard config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    base_config = _scorecard_config(
        _yaml_mapping(base_payload, label="base development scorecard config"),
        label="base development scorecard config",
    )
    development_config = copy.deepcopy(base_config)
    development_config["checkpoint"] = sweep.candidate_artifact["path"]
    development_config["training_config"] = sweep.training_config_artifact["path"]
    development_config["generation"] = {"device": "mps", "max_new_tokens": 96}

    confirmatory_config = copy.deepcopy(development_config)
    confirmatory_case_paths = {}
    for key, value in (
        ("path", confirmatory_cases),
        ("manifest", confirmatory_manifest),
        ("generator_config", confirmatory_generator_config),
    ):
        literal, _resolved = _relative_input(
            root,
            str(value),
            label=f"confirmatory cases {key}",
        )
        confirmatory_case_paths[key] = literal
    confirmatory_config["cases"] = {
        **confirmatory_case_paths,
        "expected_split": "eval",
        "expected_rule_verified": True,
        "environment_policy": "forbid",
    }
    if "selection" in confirmatory_config["cases"]:
        raise RuntimeError("confirmatory scorecard must not re-select the frozen case artifact")

    training_payload = _read_regular(
        _resolve_input(root, sweep.training_config_artifact["path"]),
        label="selected SFT training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_config = _yaml_mapping(training_payload, label="selected SFT training config")
    static = _model_and_tokenizer_binding(
        development_config,
        training_config=training_config,
        training_record=sweep.training_config_artifact,
        training_sha256=sweep.training_config_sha256,
        repository_root=root,
    )
    _assert_scorecard_static_matches_sweep(static, sweep)
    development_cases_binding, development_conversations = _case_binding(
        development_config["cases"],
        repository_root=root,
        label="development cases",
    )
    confirmatory_cases_binding, confirmatory_conversations = _case_binding(
        confirmatory_config["cases"],
        repository_root=root,
        label="confirmatory cases",
    )
    direct_overlap = _direct_dev_confirm_overlap(
        development_conversations,
        confirmatory_conversations,
    )
    confirmatory_provenance_text = _literal_relative_path(
        str(confirmatory_provenance),
        label="confirmatory split provenance receipt path",
    )
    confirmatory_receipt = _validate_confirmatory_provenance(
        confirmatory_provenance_text,
        repository_root=root,
        confirmatory_cases=confirmatory_cases_binding,
        development_cases=development_cases_binding,
    )

    development_payload = _serialize_scorecard_config(development_config)
    confirmatory_payload = _serialize_scorecard_config(confirmatory_config)
    development_config_record = _file_record(
        output_texts["development"],
        development_payload,
    )
    confirmatory_config_record = _file_record(
        output_texts["confirmatory"],
        confirmatory_payload,
    )
    evaluator_modules = _evaluator_module_binding()
    development_expected = _expected_scorecard_provenance(
        config=development_config,
        config_record=development_config_record,
        sweep=sweep,
        cases=development_cases_binding,
        static=static,
        evaluator_modules=evaluator_modules,
    )
    confirmatory_expected = _expected_scorecard_provenance(
        config=confirmatory_config,
        config_record=confirmatory_config_record,
        sweep=sweep,
        cases=confirmatory_cases_binding,
        static=static,
        evaluator_modules=evaluator_modules,
    )

    binding_core = {
        "kind": BINDING_KIND,
        "schema_version": SCHEMA_VERSION,
        "preparation_contract": {
            "device": "mps",
            "max_new_tokens": 96,
            "confirmatory_is_preselected": True,
            "confirmatory_reselection_forbidden": True,
            "base_scorecard_config": _file_record(base_text, base_payload),
        },
        "sweep": {
            "result": sweep.result_artifact,
            "result_self_sha256": sweep.result["result_sha256"],
            "config": {
                **sweep.sweep_config_artifact,
                "canonical_sha256": sweep.sweep_config_sha256,
            },
            "checkpoint_artifacts": list(sweep.checkpoint_artifacts),
            "training_config": {
                **sweep.training_config_artifact,
                "canonical_sha256": sweep.training_config_sha256,
            },
            "model_config": {
                **sweep.model_config_artifact,
                "canonical_sha256": sweep.model_config_sha256,
            },
            "tokenizer": copy.deepcopy(sweep.tokenizer_record),
            "heldout_sources": list(sweep.heldout_source_bindings),
        },
        "candidate": {
            "artifact": sweep.candidate_artifact,
            "checkpoint_step": sweep.candidate_record["checkpoint_step"],
            "completed_steps": sweep.candidate_record["completed_steps"],
            "planned_steps": sweep.candidate_record["planned_steps"],
            "retention_eligible": True,
            "selection": "summary.best_retention_eligible_checkpoint",
            "checkpoint_lineage": sweep.lineage,
        },
        "scorecards": {
            "development": {
                "config": {
                    **development_config_record,
                    "canonical_sha256": canonical_sha256(development_config),
                },
                "cases": development_cases_binding,
                "expected_provenance": development_expected,
            },
            "confirmatory": {
                "config": {
                    **confirmatory_config_record,
                    "canonical_sha256": canonical_sha256(confirmatory_config),
                },
                "cases": confirmatory_cases_binding,
                "direct_development_overlap": direct_overlap,
                "frozen_selection_provenance": confirmatory_receipt,
                "expected_provenance": confirmatory_expected,
            },
        },
    }
    binding = _self_hashed_receipt(binding_core, field="binding_self_sha256")
    binding_payload = canonical_json_bytes(binding)
    _publish_bundle(
        [
            (output_paths["development"], development_payload),
            (output_paths["confirmatory"], confirmatory_payload),
            (output_paths["binding"], binding_payload),
        ]
    )
    return binding


def _validate_binding_receipt(
    path: str | Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], bytes, ValidatedSweep]:
    _path_text_value, source = _relative_input(
        repository_root,
        str(path),
        label="SFT candidate binding receipt path",
    )
    binding, payload = _load_canonical_json_file(
        source,
        label="SFT candidate binding receipt",
    )
    binding = _exact_mapping(
        binding,
        frozenset(
            {
                "kind",
                "schema_version",
                "preparation_contract",
                "sweep",
                "candidate",
                "scorecards",
                "binding_self_sha256",
            }
        ),
        label="SFT candidate binding receipt",
    )
    schema_version = binding["schema_version"]
    if (
        binding["kind"] != BINDING_KIND
        or type(schema_version) is not int
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError("SFT candidate binding receipt kind/schema is invalid")
    self_sha256 = _require_sha256(
        binding["binding_self_sha256"],
        label="SFT candidate binding receipt.binding_self_sha256",
    )
    unsigned = copy.deepcopy(binding)
    unsigned.pop("binding_self_sha256")
    if _sha256(canonical_json_bytes(unsigned)) != self_sha256:
        raise ValueError("SFT candidate binding receipt self-hash mismatch")

    preparation = _exact_mapping(
        binding["preparation_contract"],
        frozenset(
            {
                "device",
                "max_new_tokens",
                "confirmatory_is_preselected",
                "confirmatory_reselection_forbidden",
                "base_scorecard_config",
            }
        ),
        label="SFT candidate binding receipt.preparation_contract",
    )
    if (
        preparation["device"] != "mps"
        or _strict_int(
            preparation["max_new_tokens"],
            label="SFT candidate binding receipt.preparation_contract.max_new_tokens",
            minimum=1,
        )
        != 96
        or preparation["confirmatory_is_preselected"] is not True
        or preparation["confirmatory_reselection_forbidden"] is not True
    ):
        raise ValueError("SFT candidate binding preparation contract is invalid")
    base_record = _validated_file_record(
        preparation["base_scorecard_config"],
        label="SFT candidate binding base scorecard config",
        literal_relative=True,
    )
    _assert_current_file(
        base_record,
        repository_root=repository_root,
        label="bound base development scorecard config",
        max_bytes=_MAX_CONFIG_BYTES,
    )

    sweep_binding = _exact_mapping(
        binding["sweep"],
        frozenset(
            {
                "result",
                "result_self_sha256",
                "config",
                "checkpoint_artifacts",
                "training_config",
                "model_config",
                "tokenizer",
                "heldout_sources",
            }
        ),
        label="SFT candidate binding receipt.sweep",
    )
    sweep_result_record = _validated_file_record(
        sweep_binding["result"],
        label="SFT candidate binding sweep result",
        literal_relative=True,
    )
    _assert_current_file(
        sweep_result_record,
        repository_root=repository_root,
        label="bound SFT checkpoint sweep result",
        max_bytes=_MAX_JSON_BYTES,
    )
    sweep = load_validated_sweep_result(
        sweep_result_record["path"],
        repository_root=repository_root,
    )
    _replay_sweep_and_match(sweep, repository_root=repository_root)
    if not _json_equal(sweep.result_artifact, sweep_result_record):
        raise ValueError("SFT candidate binding sweep-result byte record is inconsistent")
    if sweep_binding["result_self_sha256"] != sweep.result["result_sha256"]:
        raise ValueError("SFT candidate binding sweep self-hash mismatch")
    bound_sweep_config = _exact_mapping(
        sweep_binding["config"],
        frozenset({"path", "bytes", "sha256", "canonical_sha256"}),
        label="SFT candidate binding sweep config",
    )
    validated_sweep_config = _validated_file_record(
        {key: bound_sweep_config[key] for key in _FILE_RECORD_KEYS},
        label="SFT candidate binding sweep config",
        literal_relative=True,
    )
    sweep_config_canonical = _require_sha256(
        bound_sweep_config["canonical_sha256"],
        label="SFT candidate binding sweep config.canonical_sha256",
    )
    if (
        not _json_equal(validated_sweep_config, sweep.sweep_config_artifact)
        or sweep_config_canonical != sweep.sweep_config_sha256
    ):
        raise ValueError("SFT candidate binding sweep config has drifted")
    checkpoint_artifacts_raw = sweep_binding["checkpoint_artifacts"]
    if not isinstance(checkpoint_artifacts_raw, list) or not checkpoint_artifacts_raw:
        raise ValueError("SFT candidate binding checkpoint_artifacts must be a non-empty list")
    checkpoint_artifacts = [
        _validated_file_record(
            record,
            label=f"SFT candidate binding checkpoint_artifacts[{index}]",
            literal_relative=True,
        )
        for index, record in enumerate(checkpoint_artifacts_raw)
    ]
    if not _json_equal(checkpoint_artifacts, list(sweep.checkpoint_artifacts)):
        raise ValueError("SFT candidate binding checkpoint archive inventory has drifted")
    bound_training = _exact_mapping(
        sweep_binding["training_config"],
        frozenset({"path", "bytes", "sha256", "canonical_sha256"}),
        label="SFT candidate binding training config",
    )
    validated_training = _validated_file_record(
        {key: bound_training[key] for key in _FILE_RECORD_KEYS},
        label="SFT candidate binding training config",
        literal_relative=True,
    )
    training_canonical = _require_sha256(
        bound_training["canonical_sha256"],
        label="SFT candidate binding training config.canonical_sha256",
    )
    if (
        not _json_equal(validated_training, sweep.training_config_artifact)
        or training_canonical != sweep.training_config_sha256
    ):
        raise ValueError("SFT candidate binding selected training config has drifted")
    bound_model = _exact_mapping(
        sweep_binding["model_config"],
        frozenset({"path", "bytes", "sha256", "canonical_sha256"}),
        label="SFT candidate binding model config",
    )
    validated_model = _validated_file_record(
        {key: bound_model[key] for key in _FILE_RECORD_KEYS},
        label="SFT candidate binding model config",
        literal_relative=True,
    )
    model_canonical = _require_sha256(
        bound_model["canonical_sha256"],
        label="SFT candidate binding model config.canonical_sha256",
    )
    if (
        not _json_equal(validated_model, sweep.model_config_artifact)
        or model_canonical != sweep.model_config_sha256
    ):
        raise ValueError("SFT candidate binding selected model config has drifted")
    if not _json_equal(sweep_binding["tokenizer"], sweep.tokenizer_record):
        raise ValueError("SFT candidate binding selected tokenizer has drifted")
    heldout_sources = sweep_binding["heldout_sources"]
    if not isinstance(heldout_sources, list) or not _json_equal(
        heldout_sources,
        list(sweep.heldout_source_bindings),
    ):
        raise ValueError("SFT candidate binding heldout source inventory has drifted")

    candidate = _exact_mapping(
        binding["candidate"],
        frozenset(
            {
                "artifact",
                "checkpoint_step",
                "completed_steps",
                "planned_steps",
                "retention_eligible",
                "selection",
                "checkpoint_lineage",
            }
        ),
        label="SFT candidate binding receipt.candidate",
    )
    candidate_artifact = _validated_file_record(
        candidate["artifact"],
        label="SFT candidate binding candidate.artifact",
        literal_relative=True,
    )
    expected_candidate = {
        "artifact": sweep.candidate_artifact,
        "checkpoint_step": sweep.candidate_record["checkpoint_step"],
        "completed_steps": sweep.candidate_record["completed_steps"],
        "planned_steps": sweep.candidate_record["planned_steps"],
        "retention_eligible": True,
        "selection": "summary.best_retention_eligible_checkpoint",
        "checkpoint_lineage": sweep.lineage,
    }
    for key in ("checkpoint_step", "completed_steps", "planned_steps"):
        _strict_int(
            candidate[key],
            label=f"SFT candidate binding candidate.{key}",
            minimum=0 if key == "checkpoint_step" else 1,
        )
    if not isinstance(candidate["retention_eligible"], bool):
        raise TypeError("SFT candidate binding candidate.retention_eligible must be boolean")
    if not _json_equal(candidate_artifact, sweep.candidate_artifact) or not _json_equal(
        candidate,
        expected_candidate,
    ):
        raise ValueError("SFT candidate binding does not name the sweep summary-selected archive")

    scorecards = _exact_mapping(
        binding["scorecards"],
        frozenset({"development", "confirmatory"}),
        label="SFT candidate binding receipt.scorecards",
    )
    development_binding = _exact_mapping(
        scorecards["development"],
        frozenset({"config", "cases", "expected_provenance"}),
        label="SFT candidate binding development scorecard",
    )
    confirmatory_binding = _exact_mapping(
        scorecards["confirmatory"],
        frozenset(
            {
                "config",
                "cases",
                "direct_development_overlap",
                "frozen_selection_provenance",
                "expected_provenance",
            }
        ),
        label="SFT candidate binding confirmatory scorecard",
    )

    configs: dict[str, dict[str, Any]] = {}
    config_records: dict[str, dict[str, Any]] = {}
    for name, scorecard_binding in (
        ("development", development_binding),
        ("confirmatory", confirmatory_binding),
    ):
        raw_record = _exact_mapping(
            scorecard_binding["config"],
            frozenset({"path", "bytes", "sha256", "canonical_sha256"}),
            label=f"SFT candidate binding {name} config",
        )
        record = _validated_file_record(
            {key: raw_record[key] for key in _FILE_RECORD_KEYS},
            label=f"SFT candidate binding {name} config",
            literal_relative=True,
        )
        canonical = _require_sha256(
            raw_record["canonical_sha256"],
            label=f"SFT candidate binding {name} config.canonical_sha256",
        )
        _assert_current_file(
            record,
            repository_root=repository_root,
            label=f"bound {name} scorecard config",
            max_bytes=_MAX_CONFIG_BYTES,
        )
        config_payload = _read_regular(
            _resolve_input(repository_root, record["path"]),
            label=f"bound {name} scorecard config",
            max_bytes=_MAX_CONFIG_BYTES,
        )
        config = _scorecard_config(
            _yaml_mapping(config_payload, label=f"bound {name} scorecard config"),
            label=f"bound {name} scorecard config",
        )
        if canonical_sha256(config) != canonical:
            raise ValueError(f"bound {name} scorecard config canonical hash mismatch")
        configs[name] = config
        config_records[name] = {**record, "canonical_sha256": canonical}

    development_config = configs["development"]
    confirmatory_config = configs["confirmatory"]
    for name, config in configs.items():
        if (
            config["checkpoint"] != sweep.candidate_artifact["path"]
            or config["training_config"] != sweep.training_config_artifact["path"]
            or not _json_equal(
                config["generation"],
                {"device": "mps", "max_new_tokens": 96},
            )
        ):
            raise ValueError(f"{name} scorecard config is not bound to the selected candidate")
    for key in (
        "kind",
        "schema_version",
        "checkpoint",
        "training_config",
        "model_config",
        "tokenizer",
    ):
        if development_config[key] != confirmatory_config[key]:
            raise ValueError(f"development and confirmatory scorecard configs disagree on {key}")
    if "selection" in _mapping(
        confirmatory_config["cases"],
        label="bound confirmatory scorecard cases",
    ):
        raise ValueError("confirmatory scorecard must not re-select the frozen artifact")

    training_payload = _read_regular(
        _resolve_input(repository_root, sweep.training_config_artifact["path"]),
        label="bound selected SFT training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_config = _yaml_mapping(
        training_payload,
        label="bound selected SFT training config",
    )
    static = _model_and_tokenizer_binding(
        development_config,
        training_config=training_config,
        training_record=sweep.training_config_artifact,
        training_sha256=sweep.training_config_sha256,
        repository_root=repository_root,
    )
    _assert_scorecard_static_matches_sweep(static, sweep)
    development_cases, development_conversations = _case_binding(
        development_config["cases"],
        repository_root=repository_root,
        label="development cases",
    )
    confirmatory_cases, confirmatory_conversations = _case_binding(
        confirmatory_config["cases"],
        repository_root=repository_root,
        label="confirmatory cases",
    )
    if not _json_equal(development_cases, development_binding["cases"]):
        raise ValueError("bound development case-set identity has drifted")
    if not _json_equal(confirmatory_cases, confirmatory_binding["cases"]):
        raise ValueError("bound confirmatory case-set identity has drifted")
    direct_overlap = _direct_dev_confirm_overlap(
        development_conversations,
        confirmatory_conversations,
    )
    if not _json_equal(
        direct_overlap,
        confirmatory_binding["direct_development_overlap"],
    ):
        raise ValueError("bound direct development/confirmatory overlap audit has drifted")
    frozen_receipt_record = _require_keys(
        confirmatory_binding["frozen_selection_provenance"],
        frozenset({"artifact"}),
        label="bound confirmatory frozen-selection provenance",
    )
    frozen_receipt_artifact = _validated_file_record(
        frozen_receipt_record["artifact"],
        label="bound confirmatory frozen-selection provenance artifact",
        literal_relative=True,
    )
    revalidated_confirmatory_receipt = _validate_confirmatory_provenance(
        frozen_receipt_artifact["path"],
        repository_root=repository_root,
        confirmatory_cases=confirmatory_cases,
        development_cases=development_cases,
    )
    if not _json_equal(
        revalidated_confirmatory_receipt,
        confirmatory_binding["frozen_selection_provenance"],
    ):
        raise ValueError("bound confirmatory frozen-selection provenance has drifted")

    evaluator_modules = _evaluator_module_binding()
    expected_development = _expected_scorecard_provenance(
        config=development_config,
        config_record={key: config_records["development"][key] for key in _FILE_RECORD_KEYS},
        sweep=sweep,
        cases=development_cases,
        static=static,
        evaluator_modules=evaluator_modules,
    )
    expected_confirmatory = _expected_scorecard_provenance(
        config=confirmatory_config,
        config_record={key: config_records["confirmatory"][key] for key in _FILE_RECORD_KEYS},
        sweep=sweep,
        cases=confirmatory_cases,
        static=static,
        evaluator_modules=evaluator_modules,
    )
    if not _json_equal(
        development_binding["expected_provenance"],
        expected_development,
    ):
        raise ValueError("bound development scorecard provenance contract has drifted")
    if not _json_equal(
        confirmatory_binding["expected_provenance"],
        expected_confirmatory,
    ):
        raise ValueError("bound confirmatory scorecard provenance contract has drifted")
    return binding, payload, sweep


def load_validated_candidate_binding(
    path: str | Path,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Revalidate a canonical binding receipt and every currently bound input artifact."""

    binding, _payload, _sweep = _validate_binding_receipt(
        path,
        repository_root=Path(repository_root),
    )
    return binding


def _validate_rate(
    value: Any,
    *,
    label: str,
    expected_total: int | None = None,
) -> dict[str, Any]:
    rate = _exact_mapping(value, _RATE_KEYS, label=label)
    correct = _strict_int(rate["correct"], label=f"{label}.correct")
    total = _strict_int(rate["total"], label=f"{label}.total")
    accuracy = _finite_number(rate["accuracy"], label=f"{label}.accuracy")
    if correct > total:
        raise ValueError(f"{label}.correct exceeds total")
    if expected_total is not None and total != expected_total:
        raise ValueError(f"{label}.total does not match its case-set denominator")
    expected_accuracy = correct / total if total else None
    if expected_accuracy is None or accuracy != expected_accuracy:
        raise ValueError(f"{label}.accuracy is inconsistent with counts")
    return {"correct": correct, "total": total, "accuracy": accuracy}


def _validate_evaluator_provenance(
    value: Any,
    *,
    expected_modules: Mapping[str, Any],
    label: str,
) -> None:
    evaluator = _require_keys(
        value,
        frozenset({"source_tree", "modules"}),
        label=label,
    )
    modules = _mapping(evaluator["modules"], label=f"{label}.modules")
    if not _json_equal(modules, expected_modules):
        raise ValueError(f"{label}.modules do not match the bound evaluator sources")
    source_tree = _require_keys(
        evaluator["source_tree"],
        frozenset({"commit", "repository_sha256", "worktree_sha256", "dirty"}),
        label=f"{label}.source_tree",
    )
    commit = source_tree["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{label}.source_tree.commit must be a lowercase Git SHA-1")
    for key in ("repository_sha256", "worktree_sha256"):
        _require_sha256(source_tree[key], label=f"{label}.source_tree.{key}")
    if not isinstance(source_tree["dirty"], bool):
        raise TypeError(f"{label}.source_tree.dirty must be boolean")


def _validate_generation_provenance(
    value: Any,
    *,
    decisions: int,
    label: str,
) -> None:
    generation = _require_keys(
        value,
        frozenset(
            {
                "requested_device",
                "resolved_device",
                "requested_dtype",
                "resolved_dtype",
                "temperature",
                "max_new_tokens",
                "serial_generation_calls",
                "serial_prefill_calls",
                "generation_batch_size",
                "maximum_non_eos_new_tokens",
                "conversation_prompt_contract",
                "truncation",
                "generation_reserve_tokens",
                "prompt_budget",
                "gold_output_budget",
            }
        ),
        label=label,
    )
    if (
        generation["requested_device"] != "mps"
        or generation["resolved_device"] != "mps"
        or generation["requested_dtype"] != "fp32"
        or generation["resolved_dtype"] != "fp32"
        or _finite_number(generation["temperature"], label=f"{label}.temperature") != 0.0
        or generation["conversation_prompt_contract"] != OPENAI_FULL_CATALOG_V1
        or generation["truncation"] != "forbidden"
    ):
        raise ValueError(f"{label} violates the frozen MPS greedy-generation contract")
    integer_expectations = {
        "max_new_tokens": 96,
        "serial_generation_calls": decisions,
        "serial_prefill_calls": decisions,
        "generation_batch_size": 1,
        "maximum_non_eos_new_tokens": decisions * 96,
        "generation_reserve_tokens": 96,
    }
    for key, expected in integer_expectations.items():
        observed = _strict_int(
            generation[key],
            label=f"{label}.{key}",
            minimum=1,
        )
        if observed != expected:
            raise ValueError(f"{label}.{key} does not match the frozen generation contract")
    prompt_budget = _require_keys(
        generation["prompt_budget"],
        frozenset({"assistant_decisions", "truncation", "generation_reserve_tokens"}),
        label=f"{label}.prompt_budget",
    )
    if (
        _strict_int(
            prompt_budget["assistant_decisions"],
            label=f"{label}.prompt_budget.assistant_decisions",
            minimum=1,
        )
        != decisions
        or prompt_budget["truncation"] != "forbidden"
        or _strict_int(
            prompt_budget["generation_reserve_tokens"],
            label=f"{label}.prompt_budget.generation_reserve_tokens",
            minimum=1,
        )
        != 96
    ):
        raise ValueError(f"{label}.prompt_budget is inconsistent")
    gold_budget = _require_keys(
        generation["gold_output_budget"],
        frozenset({"assistant_decisions", "max_new_tokens", "fits_generation_budget"}),
        label=f"{label}.gold_output_budget",
    )
    if (
        _strict_int(
            gold_budget["assistant_decisions"],
            label=f"{label}.gold_output_budget.assistant_decisions",
            minimum=1,
        )
        != decisions
        or _strict_int(
            gold_budget["max_new_tokens"],
            label=f"{label}.gold_output_budget.max_new_tokens",
            minimum=1,
        )
        != 96
        or gold_budget["fits_generation_budget"] is not True
    ):
        raise ValueError(f"{label}.gold_output_budget is inconsistent")


def _replay_scorecard_and_match(
    supplied: Mapping[str, Any],
    *,
    config_record: Mapping[str, Any],
    scorecard_name: str,
    repository_root: Path,
) -> None:
    """Run the bound evaluator and require exact agreement with supplied evidence."""

    _assert_current_file(
        config_record,
        repository_root=repository_root,
        label=f"bound {scorecard_name} scorecard config before replay",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    with _at_repository_root(repository_root):
        replayed = run_scorecard(config_record["path"])
    if not isinstance(replayed, Mapping):
        raise TypeError(f"replayed {scorecard_name} scorecard did not return a mapping")
    try:
        json.dumps(replayed, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"replayed {scorecard_name} scorecard returned invalid JSON") from error
    if not _json_equal(replayed, supplied):
        raise ValueError(f"replayed {scorecard_name} scorecard does not match supplied evidence")
    _assert_current_file(
        config_record,
        repository_root=repository_root,
        label=f"bound {scorecard_name} scorecard config after replay",
        max_bytes=_MAX_CONFIG_BYTES,
    )


def _validate_scorecard_result(
    path: str | Path,
    *,
    binding: Mapping[str, Any],
    scorecard_name: str,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_text, source = _relative_input(
        repository_root,
        str(path),
        label=f"{scorecard_name} scorecard result path",
    )
    result, payload = _load_canonical_json_file(
        source,
        label=f"{scorecard_name} scorecard result",
    )
    result = _exact_mapping(
        result,
        _SCORECARD_TOP_LEVEL_KEYS,
        label=f"{scorecard_name} scorecard result",
    )
    schema_version = result["schema_version"]
    if (
        result["kind"] != SCORECARD_RESULT_KIND
        or type(schema_version) is not int
        or schema_version != SCORECARD_SCHEMA_VERSION
    ):
        raise ValueError(f"{scorecard_name} scorecard result kind/schema is invalid")
    self_sha256 = _require_sha256(
        result["result_self_sha256"],
        label=f"{scorecard_name} scorecard result.result_self_sha256",
    )
    unsigned = copy.deepcopy(result)
    unsigned.pop("result_self_sha256")
    if _sha256(canonical_json_bytes(unsigned)) != self_sha256:
        raise ValueError(f"{scorecard_name} scorecard result self-hash mismatch")
    scorecard_binding = binding["scorecards"][scorecard_name]
    _replay_scorecard_and_match(
        result,
        config_record=scorecard_binding["config"],
        scorecard_name=scorecard_name,
        repository_root=repository_root,
    )
    benchmark = _require_keys(
        result["benchmark"],
        frozenset(
            {
                "official_bfcl",
                "external_native_benchmark",
                "conversation_prompt_contract",
            }
        ),
        label=f"{scorecard_name} scorecard benchmark",
    )
    if (
        benchmark["official_bfcl"] is not False
        or benchmark["external_native_benchmark"] is not False
        or benchmark["conversation_prompt_contract"] != OPENAI_FULL_CATALOG_V1
    ):
        raise ValueError(f"{scorecard_name} scorecard benchmark contract is invalid")

    expected = scorecard_binding["expected_provenance"]
    provenance = _require_keys(
        result["provenance"],
        frozenset(
            {
                "evaluator",
                "scorecard_config",
                "checkpoint",
                "checkpoint_lineage",
                "training_config",
                "model_config",
                "tokenizer",
                "training_corpus",
                "cases",
                "generation",
            }
        ),
        label=f"{scorecard_name} scorecard provenance",
    )
    _validate_evaluator_provenance(
        provenance["evaluator"],
        expected_modules=expected["evaluator_modules"],
        label=f"{scorecard_name} scorecard provenance.evaluator",
    )
    for key in (
        "scorecard_config",
        "checkpoint",
        "checkpoint_lineage",
        "training_config",
        "model_config",
        "tokenizer",
        "training_corpus",
        "cases",
    ):
        if not _json_equal(provenance[key], expected[key]):
            raise ValueError(
                f"{scorecard_name} scorecard provenance.{key} does not match its binding"
            )

    score = _exact_mapping(
        result["scorecard"],
        frozenset({"contract", "case_set", "metrics", "by_category", "predictions"}),
        label=f"{scorecard_name} scorecard payload",
    )
    case_set = _exact_mapping(
        score["case_set"],
        frozenset(
            {
                "sha256",
                "conversations",
                "assistant_decisions",
                "tool_decisions",
                "no_tool_decisions",
            }
        ),
        label=f"{scorecard_name} scorecard case_set",
    )
    _require_sha256(
        case_set["sha256"],
        label=f"{scorecard_name} scorecard case_set.sha256",
    )
    for key in (
        "conversations",
        "assistant_decisions",
        "tool_decisions",
        "no_tool_decisions",
    ):
        _strict_int(
            case_set[key],
            label=f"{scorecard_name} scorecard case_set.{key}",
            minimum=0 if key in {"tool_decisions", "no_tool_decisions"} else 1,
        )
    if (
        case_set["tool_decisions"] + case_set["no_tool_decisions"]
        != case_set["assistant_decisions"]
    ):
        raise ValueError(f"{scorecard_name} scorecard case-set counts are inconsistent")
    if not _json_equal(case_set, scorecard_binding["cases"]["case_set"]):
        raise ValueError(f"{scorecard_name} scorecard case-set identity mismatch")
    decisions = _strict_int(
        case_set["assistant_decisions"],
        label=f"{scorecard_name} scorecard assistant decisions",
        minimum=1,
    )
    _validate_generation_provenance(
        provenance["generation"],
        decisions=decisions,
        label=f"{scorecard_name} scorecard provenance.generation",
    )

    predictions = _exact_mapping(
        score["predictions"],
        frozenset(
            {
                "sha256",
                "records",
                "finish_reasons",
                "complete",
                "terminated_by_eos",
                "raw_outputs_retained",
            }
        ),
        label=f"{scorecard_name} scorecard predictions",
    )
    _require_sha256(
        predictions["sha256"],
        label=f"{scorecard_name} scorecard predictions.sha256",
    )
    records = _strict_int(
        predictions["records"],
        label=f"{scorecard_name} scorecard predictions.records",
        minimum=1,
    )
    complete = _strict_int(
        predictions["complete"],
        label=f"{scorecard_name} scorecard predictions.complete",
    )
    eos = _strict_int(
        predictions["terminated_by_eos"],
        label=f"{scorecard_name} scorecard predictions.terminated_by_eos",
    )
    if records != decisions or predictions["raw_outputs_retained"] is not False:
        raise ValueError(f"{scorecard_name} scorecard prediction accounting is invalid")
    finish_reasons = _mapping(
        predictions["finish_reasons"],
        label=f"{scorecard_name} scorecard predictions.finish_reasons",
    )
    if not finish_reasons or set(finish_reasons) - {"eos", "length"}:
        raise ValueError(f"{scorecard_name} scorecard has unsupported finish reasons")
    finish_counts = {
        key: _strict_int(
            value,
            label=f"{scorecard_name} scorecard predictions.finish_reasons.{key}",
        )
        for key, value in finish_reasons.items()
    }
    eos_count = finish_counts.get("eos", 0)
    length_count = finish_counts.get("length", 0)
    if (
        sum(finish_counts.values()) != decisions
        or eos != eos_count
        or complete != eos_count
        or eos_count + length_count != decisions
    ):
        raise ValueError(f"{scorecard_name} scorecard finish-reason accounting is invalid")

    metrics = _mapping(score["metrics"], label=f"{scorecard_name} scorecard metrics")
    generation_completion = _validate_rate(
        metrics.get("generation_completion"),
        label=f"{scorecard_name} scorecard metrics.generation_completion",
        expected_total=decisions,
    )
    if generation_completion["correct"] != eos_count:
        raise ValueError(
            f"{scorecard_name} scorecard EOS count disagrees with generation completion"
        )
    metric_specs = {
        "overall_complete_format": (
            metrics.get("format_validity"),
            decisions,
        ),
        "expected_tool_strict_format": (
            metrics.get("tool_format_validity_on_tool_decisions"),
            case_set["tool_decisions"],
        ),
        "expected_tool_schema_validity": (
            metrics.get("schema_validity_on_tool_decisions"),
            case_set["tool_decisions"],
        ),
        "expected_tool_case_exact_name": (
            _require_keys(
                metrics.get("tool_name"),
                frozenset({"case_exact"}),
                label=f"{scorecard_name} scorecard metrics.tool_name",
            )["case_exact"],
            case_set["tool_decisions"],
        ),
        "expected_tool_whole_call_exact": (
            metrics.get("whole_call_exact"),
            case_set["tool_decisions"],
        ),
        "expected_no_tool_structural_abstention": (
            metrics.get("abstention"),
            case_set["no_tool_decisions"],
        ),
    }
    rates = {
        name: _validate_rate(
            value,
            label=f"{scorecard_name} scorecard gate metric {name}",
            expected_total=_strict_int(
                total,
                label=f"{scorecard_name} scorecard gate metric {name} denominator",
            ),
        )
        for name, (value, total) in metric_specs.items()
    }

    decision_pass = decisions >= PROMOTION_THRESHOLDS["minimum_assistant_decisions"]
    eos_rate = eos_count / decisions
    truncation_rate = length_count / decisions
    metric_gates = {
        name: {
            "successes": rate["correct"],
            "total": rate["total"],
            "rate": rate["accuracy"],
            "passed": (
                rate["correct"] >= PROMOTION_THRESHOLDS["minimum_metric_successes"]
                and rate["accuracy"] >= PROMOTION_THRESHOLDS["minimum_metric_rate"]
            ),
        }
        for name, rate in rates.items()
    }
    gate = {
        "assistant_decisions": {
            "observed": decisions,
            "minimum": PROMOTION_THRESHOLDS["minimum_assistant_decisions"],
            "passed": decision_pass,
        },
        "eos_completion": {
            "observed_rate": eos_rate,
            "minimum_rate": PROMOTION_THRESHOLDS["minimum_eos_completion_rate"],
            "successes": eos_count,
            "total": decisions,
            "passed": eos_rate >= PROMOTION_THRESHOLDS["minimum_eos_completion_rate"],
        },
        "truncation": {
            "observed_rate": truncation_rate,
            "maximum_rate": PROMOTION_THRESHOLDS["maximum_truncation_rate"],
            "truncated": length_count,
            "total": decisions,
            "passed": truncation_rate <= PROMOTION_THRESHOLDS["maximum_truncation_rate"],
        },
        "metrics": metric_gates,
    }
    gate["passed"] = (
        gate["assistant_decisions"]["passed"]
        and gate["eos_completion"]["passed"]
        and gate["truncation"]["passed"]
        and all(value["passed"] for value in metric_gates.values())
    )
    artifact = _file_record(path_text, payload)
    return result, {
        "artifact": artifact,
        "result_self_sha256": self_sha256,
        "candidate": copy.deepcopy(provenance["checkpoint"]),
        "case_set": copy.deepcopy(case_set),
        "predictions_sha256": predictions["sha256"],
        "gate": gate,
    }


def _assert_binding_files_still_current(
    binding: Mapping[str, Any],
    *,
    repository_root: Path,
    binding_record: Mapping[str, Any],
    evaluation_records: Sequence[Mapping[str, Any]],
) -> None:
    checks: list[tuple[Mapping[str, Any], str, int]] = [
        (
            binding_record,
            "bound SFT candidate binding receipt",
            _MAX_JSON_BYTES,
        ),
        (
            binding["preparation_contract"]["base_scorecard_config"],
            "bound base development scorecard config",
            _MAX_CONFIG_BYTES,
        ),
        (
            binding["sweep"]["result"],
            "bound SFT checkpoint sweep result",
            _MAX_JSON_BYTES,
        ),
        (
            binding["sweep"]["config"],
            "bound SFT checkpoint sweep config",
            _MAX_CONFIG_BYTES,
        ),
        (
            binding["sweep"]["training_config"],
            "bound selected SFT training config",
            _MAX_CONFIG_BYTES,
        ),
        (
            binding["sweep"]["model_config"],
            "bound sweep-reported SFT model config",
            _MAX_CONFIG_BYTES,
        ),
        (
            binding["candidate"]["artifact"],
            "bound summary-selected SFT archive",
            _MAX_CHECKPOINT_BYTES,
        ),
    ]
    for index, artifact in enumerate(binding["sweep"]["checkpoint_artifacts"]):
        checks.append(
            (
                artifact,
                f"bound SFT checkpoint archive {index}",
                _MAX_CHECKPOINT_BYTES,
            )
        )
    sweep_tokenizer_artifact = binding["sweep"]["tokenizer"].get("artifact")
    if sweep_tokenizer_artifact is not None:
        checks.append(
            (
                sweep_tokenizer_artifact,
                "bound sweep-reported SFT tokenizer artifact",
                _MAX_CONFIG_BYTES * 16,
            )
        )
    for source_index, source in enumerate(binding["sweep"]["heldout_sources"]):
        for name, maximum in (
            ("jsonl", _MAX_CASE_BYTES),
            ("manifest", _MAX_CONFIG_BYTES),
            ("generator_config", _MAX_CONFIG_BYTES),
        ):
            checks.append(
                (
                    source["files"][name],
                    f"bound sweep-reported heldout source {source_index} {name}",
                    maximum,
                )
            )
    static = binding["scorecards"]["development"]["expected_provenance"]
    checks.append(
        (
            static["model_config"],
            "bound scorecard model config",
            _MAX_CONFIG_BYTES,
        )
    )
    tokenizer_artifact = static["tokenizer"]["artifact"]
    if tokenizer_artifact is not None:
        checks.append(
            (
                tokenizer_artifact,
                "bound scorecard tokenizer artifact",
                _MAX_CONFIG_BYTES * 16,
            )
        )
    for name in ("development", "confirmatory"):
        scorecard = binding["scorecards"][name]
        checks.append(
            (
                scorecard["config"],
                f"bound {name} scorecard config",
                _MAX_CONFIG_BYTES,
            )
        )
        for file_name, maximum in (
            ("jsonl", _MAX_CASE_BYTES),
            ("manifest", _MAX_CONFIG_BYTES),
            ("generator_config", _MAX_CONFIG_BYTES),
        ):
            checks.append(
                (
                    scorecard["cases"]["files"][file_name],
                    f"bound {name} cases {file_name}",
                    maximum,
                )
            )
    checks.append(
        (
            binding["scorecards"]["confirmatory"]["frozen_selection_provenance"]["artifact"],
            "bound confirmatory split provenance receipt",
            _MAX_JSON_BYTES,
        )
    )
    for name, record in binding["scorecards"]["development"]["expected_provenance"][
        "evaluator_modules"
    ].items():
        checks.append((record, f"bound evaluator module {name}", _MAX_CONFIG_BYTES))
    for index, record in enumerate(evaluation_records):
        checks.append(
            (
                record,
                f"supplied scorecard result {index}",
                _MAX_JSON_BYTES,
            )
        )
    for record, label, maximum in checks:
        _assert_current_file(
            record,
            repository_root=repository_root,
            label=label,
            max_bytes=maximum,
        )


def verify_sft_candidate_promotion(
    binding_path: str | Path,
    development_scorecard_path: str | Path,
    confirmatory_scorecard_path: str | Path | None = None,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Verify scorecards against one binding and return a sealed promotion decision."""

    root = Path(repository_root)
    binding_path_text = _literal_relative_path(
        str(binding_path),
        label="SFT candidate binding receipt path",
    )
    binding, binding_payload, sweep = _validate_binding_receipt(
        binding_path_text,
        repository_root=root,
    )
    _development_result, development = _validate_scorecard_result(
        development_scorecard_path,
        binding=binding,
        scorecard_name="development",
        repository_root=root,
    )
    confirmatory: dict[str, Any] | None = None
    if confirmatory_scorecard_path is not None:
        _confirmatory_result, confirmatory = _validate_scorecard_result(
            confirmatory_scorecard_path,
            binding=binding,
            scorecard_name="confirmatory",
            repository_root=root,
        )
        if not _json_equal(confirmatory["candidate"], development["candidate"]):
            raise ValueError("confirmatory scorecard is not for the same summary-selected archive")
    evaluation_records = [development["artifact"]]
    if confirmatory is not None:
        evaluation_records.append(confirmatory["artifact"])
    binding_record = _file_record(binding_path_text, binding_payload)
    _assert_binding_files_still_current(
        binding,
        repository_root=root,
        binding_record=binding_record,
        evaluation_records=evaluation_records,
    )
    development_passed = bool(development["gate"]["passed"])
    confirmatory_supplied = confirmatory is not None
    confirmatory_passed = bool(confirmatory and confirmatory["gate"]["passed"])
    promotion_allowed = development_passed and confirmatory_supplied and confirmatory_passed
    if promotion_allowed:
        status = "promotion_authorized"
    elif not development_passed:
        status = "development_gate_failed"
    elif not confirmatory_supplied:
        status = "confirmatory_scorecard_required"
    else:
        status = "confirmatory_gate_failed"

    core = {
        "kind": DECISION_KIND,
        "schema_version": SCHEMA_VERSION,
        "binding": {
            **binding_record,
            "binding_self_sha256": binding["binding_self_sha256"],
        },
        "sweep_result_self_sha256": sweep.result["result_sha256"],
        "candidate": copy.deepcopy(binding["candidate"]),
        "thresholds": copy.deepcopy(PROMOTION_THRESHOLDS),
        "evaluations": {
            "development": development,
            "confirmatory": confirmatory,
        },
        "decision": {
            "development_passed": development_passed,
            "confirmatory_supplied": confirmatory_supplied,
            "confirmatory_passed": confirmatory_passed,
            "promotion_allowed": promotion_allowed,
            "status": status,
            "fallback_checkpoint_allowed": False,
        },
    }
    return _self_hashed_receipt(core, field="decision_self_sha256")


_DECISION_TOP_LEVEL_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "binding",
        "sweep_result_self_sha256",
        "candidate",
        "thresholds",
        "evaluations",
        "decision",
        "decision_self_sha256",
    }
)


def assert_sft_candidate_promotion_decision_integrity(
    value: Mapping[str, Any],
) -> None:
    """Validate only the decision envelope and self-hash, not its evidence or gate algebra."""

    decision = _exact_mapping(
        value,
        _DECISION_TOP_LEVEL_KEYS,
        label="SFT candidate promotion decision",
    )
    schema_version = decision["schema_version"]
    if (
        decision["kind"] != DECISION_KIND
        or type(schema_version) is not int
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError("SFT candidate promotion decision kind/schema is invalid")
    try:
        json.dumps(decision, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("SFT candidate promotion decision contains invalid JSON") from error
    self_sha256 = _require_sha256(
        decision["decision_self_sha256"],
        label="SFT candidate promotion decision.decision_self_sha256",
    )
    unsigned = copy.deepcopy(decision)
    unsigned.pop("decision_self_sha256")
    if _sha256(canonical_json_bytes(unsigned)) != self_sha256:
        raise ValueError("SFT candidate promotion decision self-hash mismatch")


def _validate_decision_candidate(value: Any) -> dict[str, Any]:
    candidate = _exact_mapping(
        value,
        frozenset(
            {
                "artifact",
                "checkpoint_step",
                "completed_steps",
                "planned_steps",
                "retention_eligible",
                "selection",
                "checkpoint_lineage",
            }
        ),
        label="SFT candidate promotion decision.candidate",
    )
    artifact = _validated_file_record(
        candidate["artifact"],
        label="SFT candidate promotion decision.candidate.artifact",
        literal_relative=True,
    )
    checkpoint_step = _strict_int(
        candidate["checkpoint_step"],
        label="SFT candidate promotion decision.candidate.checkpoint_step",
    )
    completed_steps = _strict_int(
        candidate["completed_steps"],
        label="SFT candidate promotion decision.candidate.completed_steps",
        minimum=1,
    )
    planned_steps = _strict_int(
        candidate["planned_steps"],
        label="SFT candidate promotion decision.candidate.planned_steps",
        minimum=1,
    )
    if checkpoint_step != completed_steps - 1 or completed_steps > planned_steps:
        raise ValueError("SFT candidate promotion decision candidate step accounting is invalid")
    if candidate["retention_eligible"] is not True:
        raise ValueError("SFT candidate promotion decision candidate must be retention eligible")
    if candidate["selection"] != "summary.best_retention_eligible_checkpoint":
        raise ValueError("SFT candidate promotion decision candidate selection is invalid")
    lineage = _mapping(
        candidate["checkpoint_lineage"],
        label="SFT candidate promotion decision.candidate.checkpoint_lineage",
    )
    if not lineage:
        raise ValueError("SFT candidate promotion decision candidate lineage must not be empty")
    return {**candidate, "artifact": artifact}


def _validate_decision_case_set(value: Any, *, label: str) -> dict[str, Any]:
    case_set = _exact_mapping(
        value,
        frozenset(
            {
                "sha256",
                "conversations",
                "assistant_decisions",
                "tool_decisions",
                "no_tool_decisions",
            }
        ),
        label=label,
    )
    _require_sha256(case_set["sha256"], label=f"{label}.sha256")
    for key in ("conversations", "assistant_decisions"):
        _strict_int(case_set[key], label=f"{label}.{key}", minimum=1)
    for key in ("tool_decisions", "no_tool_decisions"):
        _strict_int(case_set[key], label=f"{label}.{key}")
    if (
        case_set["tool_decisions"] + case_set["no_tool_decisions"]
        != case_set["assistant_decisions"]
    ):
        raise ValueError(f"{label} decision accounting is invalid")
    return case_set


def _validate_promotion_gate(
    value: Any,
    *,
    case_set: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    gate = _exact_mapping(
        value,
        frozenset(
            {
                "assistant_decisions",
                "eos_completion",
                "truncation",
                "metrics",
                "passed",
            }
        ),
        label=label,
    )
    decisions_gate = _exact_mapping(
        gate["assistant_decisions"],
        frozenset({"observed", "minimum", "passed"}),
        label=f"{label}.assistant_decisions",
    )
    observed = _strict_int(
        decisions_gate["observed"],
        label=f"{label}.assistant_decisions.observed",
        minimum=1,
    )
    minimum = _strict_int(
        decisions_gate["minimum"],
        label=f"{label}.assistant_decisions.minimum",
        minimum=1,
    )
    if (
        observed != case_set["assistant_decisions"]
        or minimum != PROMOTION_THRESHOLDS["minimum_assistant_decisions"]
    ):
        raise ValueError(f"{label}.assistant_decisions disagrees with evidence or thresholds")
    expected_decisions_passed = observed >= minimum
    if decisions_gate["passed"] is not expected_decisions_passed:
        raise ValueError(f"{label}.assistant_decisions.passed is inconsistent")

    eos_gate = _exact_mapping(
        gate["eos_completion"],
        frozenset(
            {
                "observed_rate",
                "minimum_rate",
                "successes",
                "total",
                "passed",
            }
        ),
        label=f"{label}.eos_completion",
    )
    eos_successes = _strict_int(
        eos_gate["successes"],
        label=f"{label}.eos_completion.successes",
    )
    eos_total = _strict_int(
        eos_gate["total"],
        label=f"{label}.eos_completion.total",
        minimum=1,
    )
    if eos_successes > eos_total or eos_total != observed:
        raise ValueError(f"{label}.eos_completion counts are inconsistent")
    eos_rate = _unit_interval(
        eos_gate["observed_rate"],
        label=f"{label}.eos_completion.observed_rate",
    )
    minimum_eos = _unit_interval(
        eos_gate["minimum_rate"],
        label=f"{label}.eos_completion.minimum_rate",
    )
    if not _json_equal(eos_rate, eos_successes / eos_total) or not _json_equal(
        minimum_eos,
        PROMOTION_THRESHOLDS["minimum_eos_completion_rate"],
    ):
        raise ValueError(f"{label}.eos_completion rate algebra is inconsistent")
    expected_eos_passed = eos_rate >= minimum_eos
    if eos_gate["passed"] is not expected_eos_passed:
        raise ValueError(f"{label}.eos_completion.passed is inconsistent")

    truncation_gate = _exact_mapping(
        gate["truncation"],
        frozenset(
            {
                "observed_rate",
                "maximum_rate",
                "truncated",
                "total",
                "passed",
            }
        ),
        label=f"{label}.truncation",
    )
    truncated = _strict_int(
        truncation_gate["truncated"],
        label=f"{label}.truncation.truncated",
    )
    truncation_total = _strict_int(
        truncation_gate["total"],
        label=f"{label}.truncation.total",
        minimum=1,
    )
    if truncated > truncation_total or truncation_total != observed:
        raise ValueError(f"{label}.truncation counts are inconsistent")
    truncation_rate = _unit_interval(
        truncation_gate["observed_rate"],
        label=f"{label}.truncation.observed_rate",
    )
    maximum_truncation = _unit_interval(
        truncation_gate["maximum_rate"],
        label=f"{label}.truncation.maximum_rate",
    )
    if (
        not _json_equal(truncation_rate, truncated / truncation_total)
        or not _json_equal(
            maximum_truncation,
            PROMOTION_THRESHOLDS["maximum_truncation_rate"],
        )
        or eos_successes + truncated != observed
    ):
        raise ValueError(f"{label}.truncation rate algebra is inconsistent")
    expected_truncation_passed = truncation_rate <= maximum_truncation
    if truncation_gate["passed"] is not expected_truncation_passed:
        raise ValueError(f"{label}.truncation.passed is inconsistent")

    metrics = _exact_mapping(
        gate["metrics"],
        _PROMOTION_METRIC_GATE_NAMES,
        label=f"{label}.metrics",
    )
    metric_denominators = {
        "overall_complete_format": observed,
        "expected_tool_strict_format": case_set["tool_decisions"],
        "expected_tool_schema_validity": case_set["tool_decisions"],
        "expected_tool_case_exact_name": case_set["tool_decisions"],
        "expected_tool_whole_call_exact": case_set["tool_decisions"],
        "expected_no_tool_structural_abstention": case_set["no_tool_decisions"],
    }
    metric_passes: list[bool] = []
    for name in sorted(_PROMOTION_METRIC_GATE_NAMES):
        metric = _exact_mapping(
            metrics[name],
            frozenset({"successes", "total", "rate", "passed"}),
            label=f"{label}.metrics.{name}",
        )
        successes = _strict_int(
            metric["successes"],
            label=f"{label}.metrics.{name}.successes",
        )
        total = _strict_int(
            metric["total"],
            label=f"{label}.metrics.{name}.total",
            minimum=1,
        )
        if successes > total or total != metric_denominators[name]:
            raise ValueError(f"{label}.metrics.{name} counts are inconsistent")
        rate = _unit_interval(metric["rate"], label=f"{label}.metrics.{name}.rate")
        if not _json_equal(rate, successes / total):
            raise ValueError(f"{label}.metrics.{name}.rate is inconsistent")
        expected_passed = (
            successes >= PROMOTION_THRESHOLDS["minimum_metric_successes"]
            and rate >= PROMOTION_THRESHOLDS["minimum_metric_rate"]
        )
        if metric["passed"] is not expected_passed:
            raise ValueError(f"{label}.metrics.{name}.passed is inconsistent")
        metric_passes.append(expected_passed)

    expected_passed = (
        expected_decisions_passed
        and expected_eos_passed
        and expected_truncation_passed
        and all(metric_passes)
    )
    if gate["passed"] is not expected_passed:
        raise ValueError(f"{label}.passed is inconsistent with its component gates")
    return gate


def _validate_decision_evaluation(
    value: Any,
    *,
    name: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"SFT candidate promotion decision.evaluations.{name}"
    evaluation = _exact_mapping(
        value,
        frozenset(
            {
                "artifact",
                "result_self_sha256",
                "candidate",
                "case_set",
                "predictions_sha256",
                "gate",
            }
        ),
        label=label,
    )
    _validated_file_record(
        evaluation["artifact"],
        label=f"{label}.artifact",
        literal_relative=True,
    )
    _require_sha256(
        evaluation["result_self_sha256"],
        label=f"{label}.result_self_sha256",
    )
    result_candidate = _exact_mapping(
        evaluation["candidate"],
        frozenset(
            {
                "path",
                "bytes",
                "sha256",
                "stage",
                "step",
                "conversation_prompt_contract",
            }
        ),
        label=f"{label}.candidate",
    )
    result_candidate_artifact = _validated_file_record(
        {key: result_candidate[key] for key in _FILE_RECORD_KEYS},
        label=f"{label}.candidate",
        literal_relative=True,
    )
    expected_result_candidate = {
        **candidate["artifact"],
        "stage": "sft",
        "step": candidate["checkpoint_step"],
        "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
    }
    if not _json_equal(result_candidate_artifact, candidate["artifact"]) or not _json_equal(
        result_candidate, expected_result_candidate
    ):
        raise ValueError(f"{label}.candidate does not match the bound candidate")
    case_set = _validate_decision_case_set(
        evaluation["case_set"],
        label=f"{label}.case_set",
    )
    _require_sha256(
        evaluation["predictions_sha256"],
        label=f"{label}.predictions_sha256",
    )
    _validate_promotion_gate(
        evaluation["gate"],
        case_set=case_set,
        label=f"{label}.gate",
    )
    return evaluation


def assert_sft_candidate_promotion_decision(value: Mapping[str, Any]) -> None:
    """Fully validate a decision receipt's evidence shape, gate algebra, and outcome."""

    assert_sft_candidate_promotion_decision_integrity(value)
    decision = dict(value)
    if not _json_equal(decision["thresholds"], PROMOTION_THRESHOLDS):
        raise ValueError("SFT candidate promotion decision thresholds are invalid")
    _require_sha256(
        decision["sweep_result_self_sha256"],
        label="SFT candidate promotion decision.sweep_result_self_sha256",
    )
    candidate = _validate_decision_candidate(decision["candidate"])
    binding_identity = _exact_mapping(
        decision["binding"],
        frozenset({"path", "bytes", "sha256", "binding_self_sha256"}),
        label="SFT candidate promotion decision.binding",
    )
    _validated_file_record(
        {key: binding_identity[key] for key in _FILE_RECORD_KEYS},
        label="SFT candidate promotion decision.binding",
        literal_relative=True,
    )
    _require_sha256(
        binding_identity["binding_self_sha256"],
        label="SFT candidate promotion decision.binding.binding_self_sha256",
    )

    evaluations = _exact_mapping(
        decision["evaluations"],
        frozenset({"development", "confirmatory"}),
        label="SFT candidate promotion decision.evaluations",
    )
    development = _validate_decision_evaluation(
        evaluations["development"],
        name="development",
        candidate=candidate,
    )
    confirmatory_value = evaluations["confirmatory"]
    confirmatory = (
        None
        if confirmatory_value is None
        else _validate_decision_evaluation(
            confirmatory_value,
            name="confirmatory",
            candidate=candidate,
        )
    )

    outcome = _exact_mapping(
        decision["decision"],
        frozenset(
            {
                "development_passed",
                "confirmatory_supplied",
                "confirmatory_passed",
                "promotion_allowed",
                "status",
                "fallback_checkpoint_allowed",
            }
        ),
        label="SFT candidate promotion decision.decision",
    )
    for key in (
        "development_passed",
        "confirmatory_supplied",
        "confirmatory_passed",
        "promotion_allowed",
        "fallback_checkpoint_allowed",
    ):
        if not isinstance(outcome[key], bool):
            raise TypeError(f"SFT candidate promotion decision.decision.{key} must be boolean")
    development_passed = bool(development["gate"]["passed"])
    confirmatory_supplied = confirmatory is not None
    confirmatory_passed = bool(confirmatory and confirmatory["gate"]["passed"])
    promotion_allowed = development_passed and confirmatory_supplied and confirmatory_passed
    if promotion_allowed:
        status = "promotion_authorized"
    elif not development_passed:
        status = "development_gate_failed"
    elif not confirmatory_supplied:
        status = "confirmatory_scorecard_required"
    else:
        status = "confirmatory_gate_failed"
    expected_outcome = {
        "development_passed": development_passed,
        "confirmatory_supplied": confirmatory_supplied,
        "confirmatory_passed": confirmatory_passed,
        "promotion_allowed": promotion_allowed,
        "status": status,
        "fallback_checkpoint_allowed": False,
    }
    if not _json_equal(outcome, expected_outcome):
        raise ValueError("SFT candidate promotion decision outcome is inconsistent")


def verify_decision_against_artifacts(
    decision: Mapping[str, Any],
    binding_path: str | Path,
    development_scorecard_path: str | Path,
    confirmatory_scorecard_path: str | Path | None = None,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Replay bound artifacts and require that they reproduce exactly this decision."""

    assert_sft_candidate_promotion_decision(decision)
    reproduced = verify_sft_candidate_promotion(
        binding_path,
        development_scorecard_path,
        confirmatory_scorecard_path,
        repository_root=repository_root,
    )
    if not _json_equal(reproduced, decision):
        raise ValueError("promotion decision does not match replayed bound artifacts")
    return reproduced


def write_sft_candidate_promotion_decision(
    decision: Mapping[str, Any],
    path: str | Path,
    *,
    binding_path: str | Path,
    development_scorecard_path: str | Path,
    confirmatory_scorecard_path: str | Path | None = None,
    repository_root: str | Path = ".",
) -> None:
    """Replay evidence, then atomically publish one canonical decision."""

    verify_decision_against_artifacts(
        decision,
        binding_path,
        development_scorecard_path,
        confirmatory_scorecard_path,
        repository_root=repository_root,
    )
    root = Path(repository_root)
    _path_text_value, destination = _relative_input(
        root,
        str(path),
        label="SFT candidate promotion decision output path",
    )
    _publish_bundle([(destination, canonical_json_bytes(dict(decision)))])


prepare_candidate_scorecards = prepare_sft_candidate
verify_candidate_promotion = verify_sft_candidate_promotion


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point for candidate preparation and promotion verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="seal candidate-specific development and confirmatory scorecard configs",
    )
    prepare_parser.add_argument("--sweep-result", required=True)
    prepare_parser.add_argument("--development-config-out", required=True)
    prepare_parser.add_argument("--confirmatory-config-out", required=True)
    prepare_parser.add_argument("--binding-out", required=True)
    prepare_parser.add_argument(
        "--base-scorecard-config",
        default=str(DEFAULT_BASE_SCORECARD_CONFIG),
    )
    prepare_parser.add_argument(
        "--confirmatory-cases",
        default=str(DEFAULT_CONFIRMATORY_CASES),
    )
    prepare_parser.add_argument(
        "--confirmatory-manifest",
        default=str(DEFAULT_CONFIRMATORY_MANIFEST),
    )
    prepare_parser.add_argument(
        "--confirmatory-generator-config",
        default=str(DEFAULT_CONFIRMATORY_GENERATOR_CONFIG),
    )
    prepare_parser.add_argument(
        "--confirmatory-provenance",
        default=str(DEFAULT_CONFIRMATORY_PROVENANCE),
    )
    prepare_parser.add_argument("--repository-root", default=".")

    verify_parser = subparsers.add_parser(
        "verify",
        help="verify bound scorecards and publish a fail-closed promotion decision",
    )
    verify_parser.add_argument("--binding", required=True)
    verify_parser.add_argument("--development-scorecard", required=True)
    verify_parser.add_argument("--confirmatory-scorecard")
    verify_parser.add_argument("--decision-out", required=True)
    verify_parser.add_argument("--repository-root", default=".")

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            binding = prepare_sft_candidate(
                arguments.sweep_result,
                arguments.development_config_out,
                arguments.confirmatory_config_out,
                arguments.binding_out,
                base_scorecard_config=arguments.base_scorecard_config,
                confirmatory_cases=arguments.confirmatory_cases,
                confirmatory_manifest=arguments.confirmatory_manifest,
                confirmatory_generator_config=arguments.confirmatory_generator_config,
                confirmatory_provenance=arguments.confirmatory_provenance,
                repository_root=arguments.repository_root,
            )
            print(
                json.dumps(
                    {
                        "binding_self_sha256": binding["binding_self_sha256"],
                        "candidate": binding["candidate"]["artifact"],
                        "development_config": binding["scorecards"]["development"]["config"],
                        "confirmatory_config": binding["scorecards"]["confirmatory"]["config"],
                    },
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        decision = verify_sft_candidate_promotion(
            arguments.binding,
            arguments.development_scorecard,
            arguments.confirmatory_scorecard,
            repository_root=arguments.repository_root,
        )
        write_sft_candidate_promotion_decision(
            decision,
            arguments.decision_out,
            binding_path=arguments.binding,
            development_scorecard_path=arguments.development_scorecard,
            confirmatory_scorecard_path=arguments.confirmatory_scorecard,
            repository_root=arguments.repository_root,
        )
        print(
            json.dumps(
                {
                    **decision["decision"],
                    "decision_self_sha256": decision["decision_self_sha256"],
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        if not decision["decision"]["promotion_allowed"]:
            parser.exit(
                1,
                "promotion denied: "
                f"{decision['decision']['status']} (decision receipt was written)\n",
            )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
