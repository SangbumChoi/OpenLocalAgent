"""Fail-closed artifact receipt for the parent-anchored 1M SFT recovery run.

This verifier is deliberately read-only.  It accepts only the completed fixed horizon configured
for the parent-anchored format-pulse recovery lane, validates every immutable resume archive, and
emits an integrity/accounting receipt.  It does not make a quality or retention claim.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from localagent.data.conversation_artifact import (
    assert_no_conversation_overlap,
    conversation_semantic_sha256,
)
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    resolve_conversation_prompt_contract,
)
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.loop import cosine_lr
from localagent.train.replay_sampling import (
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
)
from localagent.train.sft import (
    _SFT_RESUME_SEALED_FIELDS,
    _load_validated_sft_resume_checkpoint,
    _prepared_sft_sha256,
    _resume_sha256,
    _sealed_resume_sha256,
    _tokenizer_contract,
    _validate_parent_anchored_sampling_parent,
    _validate_sft_continuation_parent,
    _validated_resume_heldout_baseline,
)
from localagent.train.stage_sampling import prepare_sft_data
from localagent.train.stage_budget import (
    PLAN_KIND,
    PLAN_SCHEMA_VERSION,
    assert_stage_budget_self_hash,
    canonical_plan_bytes,
    verify_stage_budget_plan,
)
from localagent.train.stage_data import (
    canonical_sha256,
    load_conversation_source,
    probe_decisions,
    single_turn_samples,
    tokenizer_identity,
)
from localagent.train.update_preflight import (
    PREFLIGHT_KIND,
    PREFLIGHT_SCHEMA_VERSION,
    _bind_sft_parent_checkpoint_identity,
    _derive_sft_data_identity_and_sampling,
    _derive_sft_preflight_execution_contract,
    _expected_sft_executed_learning_rates,
    assert_preflight_receipt,
)

RECEIPT_KIND = "localagent_sft_production_receipt"
RECEIPT_SCHEMA_VERSION = 1

PRODUCTION_TOTAL_STEPS = 372
PRODUCTION_CHECKPOINT_EVERY = 12
PRODUCTION_ARCHIVE_COUNT = 31
PRODUCTION_GRAD_ACCUM_STEPS = 8
PRODUCTION_MICRO_BATCH_SIZE = 2
PRODUCTION_DECISIONS_PER_UPDATE = 16
PRODUCTION_FROZEN_PARAMETERS = (
    "loop_embed",
    "embed.weight",
    "in_proj.weight",
    "out_proj.weight",
)

# These are lane identities, not merely examples.  The verifier additionally requires callers to
# supply each root out-of-band so that a copied receipt cannot bootstrap its own trust anchors.
PRODUCTION_CONFIG_FILE_SHA256 = (
    "d06b097d5b969b310cb9e8624b0b1cf217313d12d3c2a94249c0b4bc1332b948"
)
PRODUCTION_CONFIG_CANONICAL_SHA256 = (
    "5d5bad982ed212fd472aa8c3a59edaa900857253acc59fe3a27e14b9e6e0cd06"
)
PRODUCTION_MODEL_CONFIG_FILE_SHA256 = (
    "1679904c225f193a3e0d527a8021b7a460757efba54f3609dc3770497f914b40"
)
PRODUCTION_MODEL_CONFIG_CANONICAL_SHA256 = (
    "be82a2aa915e03eb44c538f8a5e91d3cc6b69f8c45f5206e6ef89ab99cc0ef4d"
)
PRODUCTION_TOKENIZER_SHA256 = (
    "a6de45b9f5e5d7b570c3b12191cc9299fe728e857844b56486760c32f5d45436"
)
PRODUCTION_PARENT_CHECKPOINT_SHA256 = (
    "1913123ea0982f675f0add7c5b23154faf6adda99424a0a2009130104c32021f"
)
PRODUCTION_DATA_SHA256 = (
    "d70e7ea88a40bdcfa38e5a1f5c4c634e2d1a1a29a3c9f89fd3ea690b1875d21a"
)
PRODUCTION_BUDGET_PLAN_FILE_SHA256 = (
    "89e2e1c7867b0fcdd1dfb38fc41b61d98a8cf37d3b1cf83fedbe070ae4469461"
)
PRODUCTION_PREFLIGHT_FILE_SHA256 = (
    "8fe0d5f9ec0e00c4247ca7ea20891b15acf09ac2d9554c5210e0155b3153fa53"
)
PRODUCTION_PREFLIGHT_SELF_SHA256_PREFIX = "cfd209d2"
PRODUCTION_MODEL_NAME = "webgpu-1m-bpe-router"
PRODUCTION_MODEL_PARAMETERS = 980_480

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_RE = re.compile(r"latest\.step-(\d{8})\.pt")
_EXECUTION_KEYS = frozenset(
    {
        "cuda_available",
        "mps_available",
        "mps_built",
        "platform",
        "python_version",
        "requested_device",
        "requested_dtype",
        "resolved_device",
        "resolved_dtype",
        "torch_interop_threads",
        "torch_intraop_threads",
        "torch_version",
    }
)
_EXPECTED_ROOT_KEYS = frozenset(
    {
        "budget_plan_file_sha256",
        "budget_plan_self_sha256",
        "config_canonical_sha256",
        "config_file_sha256",
        "data_sha256",
        "model_config_canonical_sha256",
        "model_config_file_sha256",
        "parent_checkpoint_sha256",
        "preflight_file_sha256",
        "preflight_self_sha256",
        "tokenizer_sha256",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "artifacts",
        "contract",
        "external_roots",
        "kind",
        "receipt_self_sha256",
        "schema_version",
        "scope",
        "validation",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "archives",
        "budget_plan",
        "config",
        "latest_checkpoint",
        "live_evidence_inventory",
        "metrics",
        "model_config",
        "parent_checkpoint",
        "preflight",
        "preflight_evidence",
        "run_directory",
        "source_inputs",
    }
)
_VALIDATION_KEYS = frozenset(
    {
        "all_resume_seals_valid",
        "archive_hashes_unique",
        "artifacts_rehashed_after_validation",
        "complete_live_evidence_inventory_rehashed",
        "complete_model_state_valid",
        "complete_optimizer_state_valid",
        "externally_rooted_artifacts",
        "exact_directory_set",
        "final_totals_exact",
        "frozen_tensors_equal_parent",
        "full_training_contract_recomputed",
        "heldout_pre_baseline_exact",
        "latest_final_archive_sealed_content_equal",
        "optimizer_scope_order_group_exact",
        "prefix_accounting_exact",
        "real_preflight_execution_proven",
        "stage_budget_replayed",
        "structured_heads_disabled",
        "unfrozen_transition_observed",
    }
)
_SCOPE = {
    "artifact_integrity_and_training_accounting_only": True,
    "checkpoint_sweep_authorized": False,
    "live_artifact_reverification_required": True,
    "quality_evaluated": False,
    "quality_claimed": False,
    "receipt_only_assertion_is_integrity_only": True,
    "retention_evaluated": False,
    "retention_claimed": False,
}


@dataclass(frozen=True)
class _FileState:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileState:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            links=int(value.st_nlink),
            size=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
        )


@dataclass(frozen=True)
class _ConfiguredSourceInput:
    path: Path
    label: str
    max_bytes: int
    role: str
    index: int
    kind: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_no_symlink_ancestors(
    path: Path,
    *,
    label: str,
    allow_missing_leaf: bool,
) -> None:
    """Reject link-based path substitution without resolving through any component."""

    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            observed = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            raise ValueError(f"{label} has a missing path component: {current}") from None
        except OSError as error:
            raise ValueError(f"{label} path component is inaccessible: {current}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} must not contain symlink components: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} ancestor is not a directory: {current}")


def _safe_absolute_path(
    value: str | Path,
    *,
    label: str,
    allow_missing_leaf: bool = False,
) -> Path:
    """Return a lexical absolute path after rejecting ``..`` and every symlink ancestor."""

    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{label} must be a non-empty path")
    raw = str(value)
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain '..' components")
    normalized = os.path.normpath(raw)
    if raw != normalized:
        raise ValueError(f"{label} must be lexically normalized")
    result = Path(normalized)
    _assert_no_symlink_ancestors(
        result,
        label=label,
        allow_missing_leaf=allow_missing_leaf,
    )
    return result


def _safe_configured_path(value: Any, *, label: str) -> Path:
    """Resolve one training-config path lexically from the invocation workspace."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError(f"{label} must not contain '..' components")
    path = raw if raw.is_absolute() else Path.cwd() / raw
    normalized = Path(os.path.normpath(str(path)))
    _assert_no_symlink_ancestors(
        normalized,
        label=label,
        allow_missing_leaf=False,
    )
    return normalized


def _open_directory_chain(path: Path, *, label: str) -> int:
    """Open an absolute directory one no-follow component at a time."""

    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        os.close(descriptor)
        raise ValueError(f"{label} changed or contains an unsafe component") from error
    return descriptor


def _validated_expected_roots(value: Mapping[str, Any]) -> dict[str, str]:
    roots = _mapping(value, label="external expected roots")
    _exact_keys(roots, _EXPECTED_ROOT_KEYS, label="external expected roots")
    normalized = {
        key: _sha256(raw, label=f"external expected roots.{key}")
        for key, raw in roots.items()
    }
    pinned = {
        "budget_plan_file_sha256": PRODUCTION_BUDGET_PLAN_FILE_SHA256,
        "config_canonical_sha256": PRODUCTION_CONFIG_CANONICAL_SHA256,
        "config_file_sha256": PRODUCTION_CONFIG_FILE_SHA256,
        "data_sha256": PRODUCTION_DATA_SHA256,
        "model_config_canonical_sha256": PRODUCTION_MODEL_CONFIG_CANONICAL_SHA256,
        "model_config_file_sha256": PRODUCTION_MODEL_CONFIG_FILE_SHA256,
        "parent_checkpoint_sha256": PRODUCTION_PARENT_CHECKPOINT_SHA256,
        "preflight_file_sha256": PRODUCTION_PREFLIGHT_FILE_SHA256,
        "tokenizer_sha256": PRODUCTION_TOKENIZER_SHA256,
    }
    for key, expected in pinned.items():
        if normalized[key] != expected:
            raise ValueError(f"external expected root {key} is not the frozen 1M lane identity")
    if not normalized["preflight_self_sha256"].startswith(
        PRODUCTION_PREFLIGHT_SELF_SHA256_PREFIX
    ):
        raise ValueError("external preflight self-hash is not the frozen 1M lane receipt")
    return normalized


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _strict_int(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    expected: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    if expected is not None and value != expected:
        raise ValueError(f"{label} must be exactly {expected}, got {value}")
    return value


def _finite_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    return _mapping(value, label=label)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_yaml(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(
            payload.decode("utf-8", errors="strict"),
            Loader=_UniqueKeySafeLoader,
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from error
    value = _mapping(value, label=label)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only finite JSON-compatible values") from error
    return value


def _regular_state(path: Path, *, label: str, max_bytes: int) -> _FileState:
    try:
        observed = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible: {path}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if observed.st_size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
    return _FileState.from_stat(observed)


def _read_descriptor(descriptor: int, *, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes")
    return b"".join(chunks)


def _open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _read_stable_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[bytes, dict[str, Any], _FileState]:
    before = _regular_state(path, label=label, max_bytes=max_bytes)
    try:
        descriptor = os.open(path, _open_flags())
    except OSError as error:
        raise ValueError(f"{label} could not be opened without following links: {path}") from error
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError(f"{label} changed while it was being opened: {path}")
        first = _read_descriptor(descriptor, max_bytes=max_bytes, label=label)
        second = _read_descriptor(descriptor, max_bytes=max_bytes, label=label)
        after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(path, label=label, max_bytes=max_bytes)
    if (
        first != second
        or len(first) != before.size
        or before != after_descriptor
        or before != after_path
    ):
        raise RuntimeError(f"{label} changed during stable double read: {path}")
    return (
        first,
        {
            "path": str(path),
            "bytes": len(first),
            "sha256": hashlib.sha256(first).hexdigest(),
        },
        before,
    )


def _hash_descriptor(descriptor: int, *, max_bytes: int, label: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes")
        digest.update(chunk)
    return digest.hexdigest(), size


def _hash_stable_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[dict[str, Any], _FileState]:
    """Hash a potentially large input twice without materializing it in memory."""

    before = _regular_state(path, label=label, max_bytes=max_bytes)
    try:
        descriptor = os.open(path, _open_flags())
    except OSError as error:
        raise ValueError(f"{label} could not be opened without following links: {path}") from error
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError(f"{label} changed while it was being opened: {path}")
        first_sha256, first_size = _hash_descriptor(
            descriptor,
            max_bytes=max_bytes,
            label=label,
        )
        second_sha256, second_size = _hash_descriptor(
            descriptor,
            max_bytes=max_bytes,
            label=label,
        )
        after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(path, label=label, max_bytes=max_bytes)
    if (
        first_sha256 != second_sha256
        or first_size != before.size
        or second_size != before.size
        or before != after_descriptor
        or before != after_path
    ):
        raise RuntimeError(f"{label} changed during stable double hash: {path}")
    return (
        {"path": str(path), "bytes": before.size, "sha256": first_sha256},
        before,
    )


def _load_stable_checkpoint(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], _FileState]:
    before = _regular_state(path, label=label, max_bytes=_MAX_CHECKPOINT_BYTES)
    try:
        descriptor = os.open(path, _open_flags())
    except OSError as error:
        raise ValueError(f"{label} could not be opened without following links: {path}") from error
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError(f"{label} changed while it was being opened: {path}")
        first_sha256, first_size = _hash_descriptor(
            descriptor,
            max_bytes=_MAX_CHECKPOINT_BYTES,
            label=label,
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            value = torch.load(handle, map_location="cpu", weights_only=True)
        second_sha256, second_size = _hash_descriptor(
            descriptor,
            max_bytes=_MAX_CHECKPOINT_BYTES,
            label=label,
        )
        after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(path, label=label, max_bytes=_MAX_CHECKPOINT_BYTES)
    if (
        first_sha256 != second_sha256
        or first_size != before.size
        or second_size != before.size
        or before != after_descriptor
        or before != after_path
    ):
        raise RuntimeError(f"{label} changed during stable load and re-hash: {path}")
    checkpoint = _mapping(value, label=label)
    return (
        checkpoint,
        {"path": str(path), "bytes": before.size, "sha256": first_sha256},
        before,
    )


def _rehash_stable(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    expected_state: _FileState,
    expected_sha256: str,
) -> None:
    before = _regular_state(path, label=label, max_bytes=max_bytes)
    if before != expected_state:
        raise RuntimeError(f"{label} changed before final re-hash: {path}")
    try:
        descriptor = os.open(path, _open_flags())
    except OSError as error:
        raise RuntimeError(f"{label} changed before final re-hash: {path}") from error
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        first_sha256, first_size = _hash_descriptor(
            descriptor,
            max_bytes=max_bytes,
            label=label,
        )
        second_sha256, second_size = _hash_descriptor(
            descriptor,
            max_bytes=max_bytes,
            label=label,
        )
        after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(path, label=label, max_bytes=max_bytes)
    if (
        opened != expected_state
        or after_descriptor != expected_state
        or after_path != expected_state
        or first_size != expected_state.size
        or second_size != expected_state.size
        or first_sha256 != expected_sha256
        or second_sha256 != expected_sha256
    ):
        raise RuntimeError(f"{label} changed during final stable re-hash: {path}")


def _directory_state(path: Path, *, label: str) -> _FileState:
    try:
        observed = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible: {path}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    return _FileState.from_stat(observed)


def _snapshot_exact_directory(
    directory: Path,
    *,
    expected_names: set[str],
) -> tuple[_FileState, list[str]]:
    before = _directory_state(directory, label="production SFT output directory")
    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError as error:
        raise ValueError(
            f"could not open production SFT directory without following links: {directory}"
        ) from error
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError("production SFT output directory changed while it was opened")
        names = sorted(os.listdir(descriptor))
        for name in names:
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
                raise ValueError(
                    f"production SFT artifact {name} must be a regular non-symlink file"
                )
    except OSError as error:
        raise ValueError(f"could not enumerate production SFT directory: {directory}") from error
    finally:
        os.close(descriptor)
    after = _directory_state(directory, label="production SFT output directory")
    if before != after:
        raise RuntimeError("production SFT output directory changed while it was enumerated")
    actual = set(names)
    if actual != expected_names:
        raise ValueError(
            "production SFT output directory fields differ: "
            f"missing={sorted(expected_names - actual)}, extra={sorted(actual - expected_names)}"
        )
    for name in names:
        _regular_state(
            directory / name,
            label=f"production SFT artifact {name}",
            max_bytes=_MAX_JSON_BYTES if name == "metrics.json" else _MAX_CHECKPOINT_BYTES,
        )
    return before, names


def _same_path(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    if not left_path.is_absolute():
        left_path = Path.cwd() / left_path
    if not right_path.is_absolute():
        right_path = Path.cwd() / right_path
    return os.path.normpath(str(left_path)) == os.path.normpath(str(right_path))


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_exact_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _require_exact(left: Any, right: Any, *, label: str) -> None:
    if not _exact_equal(left, right):
        raise ValueError(f"{label} mismatch")


def _canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
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


def _lineage_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _model_config(value: Mapping[str, Any]) -> ModelConfig:
    expected = set(ModelConfig.__dataclass_fields__)
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"model config keys mismatch: missing={missing}, extra={extra}")
    result = ModelConfig(**dict(value))
    result.assert_within_budget()
    if result.name != PRODUCTION_MODEL_NAME:
        raise ValueError("model config is not the frozen webgpu 1M SFT lane")
    if result.estimate_params() != PRODUCTION_MODEL_PARAMETERS:
        raise ValueError("model config exact parameter count differs from the frozen 1M lane")
    return result


def _model_parameter_names(model_config: ModelConfig) -> list[str]:
    with torch.device("meta"):
        model = LocalAgentLM(model_config)
    return [name for name, _ in model.named_parameters()]


def _model_parameter_specs(
    model_config: ModelConfig,
) -> tuple[list[str], dict[str, tuple[torch.Size, torch.dtype]]]:
    with torch.device("meta"):
        model = LocalAgentLM(model_config)
    entries = list(model.named_parameters())
    return (
        [name for name, _ in entries],
        {name: (parameter.shape, parameter.dtype) for name, parameter in entries},
    )


def _validate_model_state(
    value: Any,
    *,
    parameter_names: Sequence[str],
    parameter_specs: Mapping[str, tuple[torch.Size, torch.dtype]],
    label: str,
) -> dict[str, torch.Tensor]:
    state = _mapping(value, label=label)
    if list(state) != list(parameter_names):
        raise ValueError(f"{label} does not match exact model named-parameter order")
    validated: dict[str, torch.Tensor] = {}
    for name in parameter_names:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label}.{name} must be a tensor")
        expected_shape, expected_dtype = parameter_specs[name]
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{label}.{name} shape mismatch: {tuple(tensor.shape)} != "
                f"{tuple(expected_shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{label}.{name} dtype mismatch: {tensor.dtype} != {expected_dtype}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"{label}.{name} contains non-finite values")
        validated[name] = tensor
    return validated


def _source_specs(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a source or list of sources")
    return list(value)


def _configured_source_inputs(
    config: Mapping[str, Any],
) -> list[_ConfiguredSourceInput]:
    """Resolve every configured data/tokenizer file that contributes live evidence."""

    data = _mapping(config.get("data"), label="config.data")
    inputs: list[_ConfiguredSourceInput] = []
    for role, key in (
        ("train", "conversations"),
        ("decay", "decay_conversations"),
        ("eval", "eval_conversations"),
    ):
        for index, spec in enumerate(_source_specs(data.get(key, []), label=f"data.{key}")):
            if isinstance(spec, Mapping):
                jsonl_path = _safe_configured_path(
                    spec.get("path"), label=f"data.{key}[{index}].path"
                )
                artifact = spec.get("artifact")
            else:
                jsonl_path = _safe_configured_path(
                    spec,
                    label=f"data.{key}[{index}]",
                )
                artifact = None
            inputs.append(
                _ConfiguredSourceInput(
                    path=jsonl_path,
                    label=f"configured SFT {role} conversation JSONL {index}",
                    max_bytes=_MAX_SOURCE_BYTES,
                    role=role,
                    index=index,
                    kind="jsonl",
                )
            )
            if artifact is not None:
                artifact_mapping = _mapping(
                    artifact,
                    label=f"data.{key}[{index}].artifact",
                )
                for artifact_key, max_bytes in (
                    ("manifest", _MAX_JSON_BYTES),
                    ("generator_config", _MAX_CONFIG_BYTES),
                ):
                    artifact_path = _safe_configured_path(
                        artifact_mapping.get(artifact_key),
                        label=f"data.{key}[{index}].artifact.{artifact_key}",
                    )
                    inputs.append(
                        _ConfiguredSourceInput(
                            path=artifact_path,
                            label=(
                                f"configured SFT {role} conversation "
                                f"{artifact_key} {index}"
                            ),
                            max_bytes=max_bytes,
                            role=role,
                            index=index,
                            kind=artifact_key,
                        )
                    )
    tokenizer = _mapping(data.get("tokenizer"), label="config.data.tokenizer")
    tokenizer_path = tokenizer.get("path")
    if tokenizer_path is not None:
        inputs.append(
            _ConfiguredSourceInput(
                path=_safe_configured_path(
                    tokenizer_path,
                    label="config.data.tokenizer.path",
                ),
                label="configured SFT tokenizer",
                max_bytes=_MAX_SOURCE_BYTES,
                role="tokenizer",
                index=0,
                kind="tokenizer",
            )
        )
    return inputs


def _configured_runtime_requests(
    config: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, str]:
    runtime = _mapping(config.get("runtime"), label=f"{label}.runtime")
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    if requested_device not in {"auto", "mps"}:
        raise ValueError(f"{label} must request device auto or mps")
    if requested_dtype not in {"auto", "fp32", "float32"}:
        raise ValueError(f"{label} must request dtype auto or explicit float32")
    return str(requested_device), str(requested_dtype)


def _validate_mps_fp32_execution(
    value: Any,
    *,
    expected_requested_device: str,
    expected_requested_dtype: str,
    label: str,
) -> dict[str, Any]:
    execution = _mapping(value, label=label)
    _exact_keys(execution, _EXECUTION_KEYS, label=label)
    required = {
        "requested_device": expected_requested_device,
        "resolved_device": "mps",
        "requested_dtype": expected_requested_dtype,
        "resolved_dtype": "fp32",
    }
    for key, expected in required.items():
        if execution.get(key) != expected:
            raise ValueError(f"{label}.{key} must be exactly {expected!r}")
    for key, expected in (
        ("cuda_available", False),
        ("mps_built", True),
        ("mps_available", True),
    ):
        if execution.get(key) is not expected:
            raise ValueError(f"{label}.{key} must be exactly {expected!r}")
    for key in ("torch_version", "python_version", "platform"):
        if not isinstance(execution.get(key), str) or not execution[key]:
            raise ValueError(f"{label}.{key} must be a non-empty string")
    for key in ("torch_intraop_threads", "torch_interop_threads"):
        _strict_int(execution.get(key), label=f"{label}.{key}", minimum=1)
    return execution


def _validated_production_runtime_evidence(
    *,
    config: Mapping[str, Any],
    execution: Any,
    training_contract: Any,
) -> tuple[dict[str, Any], str]:
    """Bind requested auto/MPS config to recorded MPS/fp32 runner evidence."""

    requested_device, requested_dtype = _configured_runtime_requests(
        config,
        label="production config",
    )
    validated_execution = _validate_mps_fp32_execution(
        execution,
        expected_requested_device=requested_device,
        expected_requested_dtype=requested_dtype,
        label="production checkpoint execution",
    )
    contract = _mapping(
        training_contract,
        label="production checkpoint training contract evidence",
    )
    amp_dtype = contract.get("amp_dtype")
    if amp_dtype != "torch.float32":
        raise ValueError(
            "production checkpoint training contract amp_dtype must be exactly "
            "'torch.float32'"
        )
    return validated_execution, str(amp_dtype)


def _derive_expected_runner_materialization(
    *,
    config: Mapping[str, Any],
    model_config: ModelConfig,
    parent_state: Mapping[str, torch.Tensor],
    expected_lm_sampling: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    optimizer_names: Sequence[str],
    expected_execution: Mapping[str, Any],
    expected_amp_dtype: str,
) -> dict[str, Any]:
    """Re-render the exact SFT pools and rebuild every runner contract field."""

    data = _mapping(config.get("data"), label="config.data")
    strict = data.get("strict_conversation_artifacts", False)
    if not isinstance(strict, bool):
        raise TypeError("config.data.strict_conversation_artifacts must be boolean")
    prompt_contract = resolve_conversation_prompt_contract(
        data.get("conversation_prompt_contract")
    )

    def load_sources(key: str, *, split: str, required: bool) -> list[Any]:
        specs = _source_specs(data.get(key, []), label=f"config.data.{key}")
        if required and not specs:
            raise ValueError(f"config.data.{key} must contain at least one source")
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
    conversations = [
        conversation for source in train_sources for conversation in source.conversations
    ]
    conversation_sources = [
        str(source.path)
        for source in train_sources
        for _conversation in source.conversations
    ]
    decay_conversations = [
        conversation for source in decay_sources for conversation in source.conversations
    ]
    eval_conversations = [
        conversation for source in eval_sources for conversation in source.conversations
    ]
    full_eval_rows = len(eval_conversations)
    overlap = assert_no_conversation_overlap(
        [*conversations, *decay_conversations],
        eval_conversations,
        left_label="SFT main/decay training content",
        right_label="held-out",
        conversation_prompt_contract=prompt_contract,
    ).as_dict()
    if overlap != data_identity.get("conversation_overlap_audit"):
        raise ValueError("recomputed SFT conversation-overlap audit differs from data identity")

    evaluation = _mapping(config.get("evaluation"), label="config.evaluation")
    selection_audit = None
    max_eval = evaluation.get("max_conversations")
    if max_eval is not None:
        if evaluation.get("selection") != STRATIFIED_EVAL_ALGORITHM:
            raise ValueError("production evaluation selector contract changed")
        selection = select_stratified_eval_subset(
            eval_conversations,
            max_rows=max_eval,
        )
        eval_conversations = list(selection.conversations)
        selection_audit = selection.audit.as_dict()
        if selection_audit != data_identity.get("eval_selection"):
            raise ValueError("recomputed held-out selection differs from data identity")

    samples = []
    sample_sources: list[str] = []
    multi_turn_conversations = []
    multi_turn_sources: list[str] = []
    for conversation, source in zip(conversations, conversation_sources, strict=True):
        projected = single_turn_samples([conversation])
        if projected:
            samples.extend(projected)
            sample_sources.extend([source] * len(projected))
        else:
            multi_turn_conversations.append(conversation)
            multi_turn_sources.append(source)
    decision_samples = probe_decisions(conversations)

    tokenizer_config = _mapping(data.get("tokenizer"), label="config.data.tokenizer")
    tokenizer_kind = str(tokenizer_config.get("kind", "byte"))
    tokenizer_path = tokenizer_config.get("path")
    tokenizer = load_tokenizer(tokenizer_kind, tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the production model config")
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )

    seq_limit = min(
        _strict_int(data.get("seq_len"), label="config.data.seq_len", minimum=2),
        model_config.max_seq_len,
    )
    schedule = _mapping(config.get("schedule"), label="config.schedule")
    heads = _mapping(config.get("heads"), label="config.heads")
    joint_heads = heads.get("joint_tool_pointer")
    if joint_heads is not False:
        raise ValueError("production runner materialization requires disabled joint heads")
    decay_sample_rows = None
    decay_sample_sources = None
    decay_conversation_sources = None
    if decay_sources:
        if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT:
            decay_sample_rows = []
            decay_sample_sources = []
            for source in decay_sources:
                projected = single_turn_samples(source.conversations)
                decay_sample_rows.extend(projected)
                decay_sample_sources.extend(
                    [f"decay:{source.path}"] * len(projected)
                )
        else:
            decay_conversation_sources = [
                f"decay:{source.path}"
                for source in decay_sources
                for _conversation in source.conversations
            ]
    prepared = prepare_sft_data(
        samples,
        tokenizer,
        conversations=(
            multi_turn_conversations
            if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversations
        ),
        sample_sources=sample_sources,
        conversation_sources=(
            multi_turn_sources
            if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else conversation_sources
        ),
        decay_samples=decay_sample_rows,
        decay_sample_sources=decay_sample_sources,
        lr_schedule=str(schedule.get("type", "cosine")),
        max_seq_len=seq_limit,
        joint_tool_head=False,
        conversation_prompt_contract=prompt_contract,
        decay_conversations=(
            decay_conversations
            if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT and decay_sources
            else None
        ),
        decay_conversation_sources=decay_conversation_sources,
    )

    batch = _mapping(config.get("batch"), label="config.batch")
    pad_to_input_tokens = batch.get("pad_to_input_tokens")
    if isinstance(pad_to_input_tokens, bool) or not isinstance(pad_to_input_tokens, int):
        raise ValueError("config.batch.pad_to_input_tokens must be an integer")
    required_width = max(
        len(row[0]) - 1
        for pool in (prepared.main_entries, prepared.decay_entries)
        for row, _source in pool
    )
    if required_width > pad_to_input_tokens:
        raise ValueError("recomputed SFT row exceeds the fixed production LM width")

    requested_device, requested_dtype = _configured_runtime_requests(
        config,
        label="production config",
    )
    execution = _validate_mps_fp32_execution(
        expected_execution,
        expected_requested_device=requested_device,
        expected_requested_dtype=requested_dtype,
        label="expected production execution",
    )
    if expected_amp_dtype != "torch.float32":
        raise ValueError("expected production amp dtype must be exactly 'torch.float32'")
    runtime = _mapping(config.get("runtime"), label="config.runtime")

    optim = _mapping(config.get("optim"), label="config.optim")
    training_contract = {
        "version": 1,
        "steps": PRODUCTION_TOTAL_STEPS,
        "batch_size": PRODUCTION_MICRO_BATCH_SIZE,
        "accum_steps": PRODUCTION_GRAD_ACCUM_STEPS,
        "lr": float(optim["lr"]),
        "warmup": int(schedule["warmup_steps"]),
        "lr_schedule": str(schedule["type"]),
        "decay_frac": float(schedule.get("decay_frac", 0.2)),
        "shuffle": bool(data.get("shuffle", True)),
        "lm_sampling": copy.deepcopy(dict(expected_lm_sampling)),
        "joint_tool_head": False,
        "aux_weight": float(heads.get("tool_loss_weight", 1.0)),
        "ptr_weight": float(heads.get("pointer_loss_weight", 0.15)),
        "mt_weight": float(heads.get("multi_turn_head_weight", 1.0)),
        "multi_turn_batch_size": 0,
        "kd_type": "topk",
        "kd_k": 16,
        "kd_weight": 0.5,
        "kd_temperature": 2.0,
        "kd_enabled": False,
        "teacher_state_sha256": None,
        "teacher_cache_sha256": None,
        "max_seq_len": seq_limit,
        "pad_to_input_tokens": pad_to_input_tokens,
        "amp_dtype": expected_amp_dtype,
        "seed": int(runtime.get("seed", 0)),
        "conversation_prompt_contract": prompt_contract,
        "tokenizer": _tokenizer_contract(tokenizer),
        "prepared_data_sha256": _prepared_sft_sha256(prepared),
        "initial_model_sha256": _resume_sha256(parent_state),
        "initial_tool_head_sha256": None,
        "initial_ptr_head_sha256": None,
        "optimizer": {
            "kind": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": float(optim["weight_decay"]),
            "grad_clip": float(optim["grad_clip"]),
        },
        "loss_normalization": optim["loss_normalization"],
        "freeze_parameters": list(PRODUCTION_FROZEN_PARAMETERS),
        "optimizer_model_parameter_names": list(optimizer_names),
        "archive_checkpoints": True,
        "checkpoint_archive_every": PRODUCTION_CHECKPOINT_EVERY,
        "checkpoint_archive_format": "immutable_periodic_sft_v1",
    }
    data_metadata: dict[str, Any] = {
        "conversation_rows": len(conversations),
        "single_turn_rows": len(samples),
        "probe_decision_rows": len(decision_samples),
        "paths": [str(source.path) for source in train_sources],
        "conversation_overlap_audit": overlap,
        "conversation_prompt_contract": prompt_contract,
        "decision_sampling": copy.deepcopy(dict(expected_lm_sampling)),
    }
    if decay_sources:
        data_metadata.update(
            {
                "decay_conversation_rows": len(decay_conversations),
                "decay_paths": [str(source.path) for source in decay_sources],
            }
        )
    if eval_sources:
        data_metadata.update(
            {
                "eval_conversation_rows": len(eval_conversations),
                "eval_source_conversation_rows": full_eval_rows,
                "eval_paths": [str(source.path) for source in eval_sources],
                "heldout_content_overlap": 0,
                "heldout_rendered_prompt_overlap": 0,
                **(
                    {"eval_selection": selection_audit}
                    if selection_audit is not None
                    else {}
                ),
            }
        )
    evaluation_padding = evaluation.get("pad_to_input_tokens")
    heldout_contract = {
        "kind": "deterministic_teacher_forced_assistant_tokens",
        "row_order": "configured_jsonl_assistant_decision_order",
        "same_rows_pre_post": True,
        "max_seq_len": seq_limit,
        "pad_to_input_tokens": evaluation_padding,
        "dataset_sha256": canonical_sha256(
            sorted(conversation_semantic_sha256(row) for row in eval_conversations)
        ),
        **(
            {"selection": selection_audit}
            if selection_audit is not None
            else {}
        ),
        "conversation_prompt_contract": prompt_contract,
    }
    return {
        "training_contract": training_contract,
        "tokenizer_lineage": tokenizer_lineage,
        "tokenizer_metadata": {
            "kind": tokenizer_kind,
            "path": tokenizer_path,
            "sha256": tokenizer_lineage["sha256"],
        },
        "data_metadata": data_metadata,
        "execution": execution,
        "heldout_contract": heldout_contract,
    }


def _artifact_matches(
    value: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
    require_path: bool = True,
) -> None:
    record = _mapping(value, label=label)
    if require_path:
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not _same_path(raw_path, expected["path"]):
            raise ValueError(f"{label}.path mismatch")
    _strict_int(record.get("bytes"), label=f"{label}.bytes", minimum=0)
    _sha256(record.get("sha256"), label=f"{label}.sha256")
    if record["bytes"] != expected["bytes"] or record["sha256"] != expected["sha256"]:
        raise ValueError(f"{label} content identity mismatch")


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("stage") != "sft":
        raise ValueError("production config must declare stage='sft'")
    _configured_runtime_requests(config, label="production config")
    schedule = _mapping(config.get("schedule"), label="config.schedule")
    total_steps = _strict_int(
        schedule.get("total_steps"),
        label="config.schedule.total_steps",
        minimum=1,
        expected=PRODUCTION_TOTAL_STEPS,
    )
    if schedule.get("type") != "cosine":
        raise ValueError("production config schedule.type must be exactly 'cosine'")
    _strict_int(
        schedule.get("warmup_steps"),
        label="config.schedule.warmup_steps",
        minimum=0,
        expected=24,
    )

    batch = _mapping(config.get("batch"), label="config.batch")
    micro_batch_size = _strict_int(
        batch.get("micro_batch_size"),
        label="config.batch.micro_batch_size",
        minimum=1,
        expected=PRODUCTION_MICRO_BATCH_SIZE,
    )
    grad_accum_steps = _strict_int(
        batch.get("grad_accum_steps"),
        label="config.batch.grad_accum_steps",
        minimum=1,
        expected=PRODUCTION_GRAD_ACCUM_STEPS,
    )
    if micro_batch_size * grad_accum_steps != PRODUCTION_DECISIONS_PER_UPDATE:
        raise ValueError("configured effective batch must be exactly 16 decisions")

    log = _mapping(config.get("log"), label="config.log")
    checkpoint_every = _strict_int(
        log.get("ckpt_every"),
        label="config.log.ckpt_every",
        minimum=1,
        expected=PRODUCTION_CHECKPOINT_EVERY,
    )
    if log.get("archive_checkpoints") is not True:
        raise ValueError("config.log.archive_checkpoints must be exactly true")
    output = log.get("out_dir")
    if not isinstance(output, str) or not output:
        raise ValueError("config.log.out_dir must be a non-empty path")
    if total_steps % checkpoint_every or total_steps // checkpoint_every != PRODUCTION_ARCHIVE_COUNT:
        raise ValueError("configured SFT horizon must produce exactly 31 periodic archives")

    continuation = _mapping(config.get("continuation"), label="config.continuation")
    if continuation.get("mode") != "fresh_optimizer_sft_child_v1":
        raise ValueError("production config must use the fresh SFT child continuation mode")
    parent = _mapping(continuation.get("parent"), label="config.continuation.parent")
    for key in (
        "checkpoint_sha256",
        "resume_integrity_sha256",
        "training_contract_sha256",
        "lm_sampling_sha256",
    ):
        _sha256(parent.get(key), label=f"config.continuation.parent.{key}")
    _strict_int(
        parent.get("completed_steps"),
        label="config.continuation.parent.completed_steps",
        minimum=1,
    )
    _strict_int(
        parent.get("completed_lm_cursor"),
        label="config.continuation.parent.completed_lm_cursor",
        minimum=0,
    )

    data = _mapping(config.get("data"), label="config.data")
    sampling = _mapping(data.get("sampling"), label="config.data.sampling")
    if sampling.get("mode") != PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
        raise ValueError("production config must use parent-anchored format-pulse sampling")
    _strict_int(
        sampling.get("update_decisions"),
        label="config.data.sampling.update_decisions",
        minimum=1,
        expected=PRODUCTION_DECISIONS_PER_UPDATE,
    )
    parent_prefix = _strict_int(
        sampling.get("parent_prefix_decisions"),
        label="config.data.sampling.parent_prefix_decisions",
        minimum=0,
    )
    pulses = _mapping(
        sampling.get("format_pulses"),
        label="config.data.sampling.format_pulses",
    )
    pulse_count = _strict_int(
        pulses.get("count"),
        label="config.data.sampling.format_pulses.count",
        minimum=1,
        expected=24,
    )
    if parent_prefix + pulse_count * PRODUCTION_DECISIONS_PER_UPDATE != (
        total_steps * PRODUCTION_DECISIONS_PER_UPDATE
    ):
        raise ValueError("parent replay plus format pulses does not fill the fixed horizon")

    optim = _mapping(config.get("optim"), label="config.optim")
    frozen = optim.get("freeze_parameters")
    if not isinstance(frozen, list) or frozen != list(PRODUCTION_FROZEN_PARAMETERS):
        raise ValueError(
            "config.optim.freeze_parameters must contain the exact four production tensors"
        )
    if optim.get("name") != "adamw":
        raise ValueError("config.optim.name must be exactly 'adamw'")
    _finite_number(optim.get("lr"), label="config.optim.lr", minimum=0.0)
    _finite_number(
        optim.get("weight_decay"),
        label="config.optim.weight_decay",
        minimum=0.0,
    )
    _finite_number(
        optim.get("grad_clip"),
        label="config.optim.grad_clip",
        minimum=0.0,
    )
    if optim.get("loss_normalization") != "microbatch_mean_v1":
        raise ValueError("production loss normalization contract changed")

    heads = _mapping(config.get("heads"), label="config.heads")
    for key in ("joint_tool_pointer", "train_route_head", "train_dense_selector"):
        if heads.get(key) is not False:
            raise ValueError(f"config.heads.{key} must be exactly false")
    _strict_int(
        heads.get("multi_turn_batch_size", 0),
        label="config.heads.multi_turn_batch_size",
        minimum=0,
        expected=0,
    )
    if heads.get("example_centroids") is not False:
        raise ValueError("config.heads.example_centroids must be exactly false")

    init_from = config.get("init_from")
    model_config = config.get("model_config")
    if not isinstance(init_from, str) or not init_from:
        raise ValueError("config.init_from must be a non-empty parent path")
    if not isinstance(model_config, str) or not model_config:
        raise ValueError("config.model_config must be a non-empty path")
    return {
        "total_steps": total_steps,
        "checkpoint_every": checkpoint_every,
        "run_directory": _safe_configured_path(
            output,
            label="config.log.out_dir",
        ),
        "parent_path": _safe_configured_path(
            init_from,
            label="config.init_from",
        ),
        "model_config_path": _safe_configured_path(
            model_config,
            label="config.model_config",
        ),
        "continuation": continuation,
        "parent": parent,
        "sampling_config": sampling,
        "optim": optim,
        "heads": heads,
        "data": data,
        "batch": batch,
        "schedule": schedule,
    }


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    config_path: Path,
    config: Mapping[str, Any],
    config_artifact: Mapping[str, Any],
    model_config_path: Path,
    model_config: ModelConfig,
    model_config_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema_version = plan.get("schema_version")
    if (
        plan.get("kind") != PLAN_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PLAN_SCHEMA_VERSION
        or plan.get("stage") != "sft"
    ):
        raise ValueError("unsupported SFT stage-budget plan")
    request = _mapping(plan.get("request"), label="stage budget request")
    if not _same_path(request.get("config_path", ""), config_path):
        raise ValueError("stage budget request points at a different config")
    for key in ("configured_steps", "max_steps"):
        _strict_int(
            request.get(key),
            label=f"stage budget request.{key}",
            minimum=1,
            expected=PRODUCTION_TOTAL_STEPS,
        )
    if request.get("min_supervised_tokens") is not None:
        raise ValueError("production budget must not select a token-calibrated prefix")
    if request.get("max_supervised_tokens") is not None:
        raise ValueError("production budget must not cap the fixed horizon")

    identity = _mapping(plan.get("identity"), label="stage budget identity")
    plan_config = _mapping(identity.get("config"), label="stage budget config identity")
    _artifact_matches(plan_config, config_artifact, label="stage budget config identity")
    if plan_config.get("canonical_sha256") != canonical_sha256(config):
        raise ValueError("stage budget canonical config identity mismatch")
    plan_model = _mapping(
        identity.get("model_config"),
        label="stage budget model-config identity",
    )
    _artifact_matches(
        plan_model,
        model_config_artifact,
        label="stage budget model-config identity",
    )
    if not _same_path(plan_model.get("path", ""), model_config_path):
        raise ValueError("stage budget model-config path mismatch")
    if plan_model.get("canonical_sha256") != canonical_sha256(model_config.__dict__):
        raise ValueError("stage budget canonical model-config identity mismatch")
    tokenizer = _mapping(identity.get("tokenizer"), label="stage budget tokenizer identity")
    _sha256(tokenizer.get("sha256"), label="stage budget tokenizer identity.sha256")

    schedule = _mapping(plan.get("schedule"), label="stage budget schedule")
    for key, expected in (
        ("micro_batch_size", PRODUCTION_MICRO_BATCH_SIZE),
        ("grad_accum_steps", PRODUCTION_GRAD_ACCUM_STEPS),
    ):
        _strict_int(
            schedule.get(key),
            label=f"stage budget schedule.{key}",
            minimum=1,
            expected=expected,
        )
    if schedule.get("freeze_parameters") != list(PRODUCTION_FROZEN_PARAMETERS):
        raise ValueError("stage budget frozen-parameter scope mismatch")
    if schedule.get("loss_normalization") != "microbatch_mean_v1":
        raise ValueError("stage budget loss normalization mismatch")
    if schedule.get("joint_tool_pointer") is not False:
        raise ValueError("stage budget unexpectedly enables the joint structured head")
    _strict_int(
        schedule.get("multi_turn_batch_size"),
        label="stage budget schedule.multi_turn_batch_size",
        minimum=0,
        expected=0,
    )

    data = _mapping(plan.get("data"), label="stage budget data")
    sampling = _mapping(
        data.get("decision_sampling"),
        label="stage budget decision sampling",
    )
    if sampling.get("mode") != PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE:
        raise ValueError("stage budget sampling mode mismatch")
    update_layout = _mapping(
        sampling.get("update_layout"),
        label="stage budget sampling update layout",
    )
    _strict_int(
        update_layout.get("update_decisions"),
        label="stage budget update decisions",
        minimum=1,
        expected=PRODUCTION_DECISIONS_PER_UPDATE,
    )
    _strict_int(
        update_layout.get("total_updates"),
        label="stage budget total updates",
        minimum=1,
        expected=PRODUCTION_TOTAL_STEPS,
    )
    if schedule.get("lm_sampling") != sampling:
        raise ValueError("stage budget schedule/data LM sampling contracts differ")

    calibration = _mapping(plan.get("calibration"), label="stage budget calibration")
    if calibration.get("mode") != "full_horizon":
        raise ValueError("stage budget calibration must select the full fixed horizon")
    _strict_int(
        calibration.get("selected_steps"),
        label="stage budget selected steps",
        minimum=1,
        expected=PRODUCTION_TOTAL_STEPS,
    )
    planned = _mapping(plan.get("planned"), label="stage budget planned")
    updates = planned.get("updates")
    if not isinstance(updates, list) or len(updates) != PRODUCTION_TOTAL_STEPS:
        raise ValueError("stage budget must contain exactly 372 per-update records")
    horizon_accounting = _budget_prefix_accounting(updates, PRODUCTION_TOTAL_STEPS)
    _validate_planned_totals(
        planned.get("horizon_totals"),
        horizon_accounting,
        label="stage budget horizon totals",
    )
    _validate_planned_totals(
        planned.get("selected_totals"),
        horizon_accounting,
        label="stage budget selected totals",
    )
    heldout = _mapping(
        data.get("heldout_eval_token_accounting"),
        label="stage budget held-out accounting",
    )
    if heldout.get("accounting_kind") != "exact_shifted_masked_language_model_tokens":
        raise ValueError("stage budget held-out accounting kind mismatch")
    for key in ("rows", "input_tokens", "loss_tokens"):
        _strict_int(heldout.get(key), label=f"stage budget held-out {key}", minimum=1)
    return sampling, horizon_accounting


def _budget_prefix_accounting(
    updates: Sequence[Any],
    completed_steps: int,
) -> dict[str, Any]:
    _strict_int(
        completed_steps,
        label="budget prefix completed_steps",
        minimum=0,
    )
    if completed_steps > len(updates):
        raise ValueError("budget prefix exceeds the sealed update list")
    source_names: list[str] | None = None
    output: dict[str, Any] | None = None
    for index, raw_update in enumerate(updates):
        update = _mapping(raw_update, label=f"stage budget update[{index}]")
        _strict_int(
            update.get("step"),
            label=f"stage budget update[{index}].step",
            minimum=0,
            expected=index,
        )
        input_tokens = _strict_int(
            update.get("input_tokens"),
            label=f"stage budget update[{index}].input_tokens",
            minimum=0,
        )
        loss_tokens = _strict_int(
            update.get("loss_tokens"),
            label=f"stage budget update[{index}].loss_tokens",
            minimum=0,
        )
        sources = _mapping(
            update.get("sources"),
            label=f"stage budget update[{index}].sources",
        )
        if source_names is None:
            source_names = list(sources)
            if not source_names:
                raise ValueError("stage budget update sources must not be empty")
            output = {
                "input_tokens": 0,
                "loss_tokens": 0,
                "sources": {
                    source: {"input_tokens": 0, "loss_tokens": 0, "rows": 0}
                    for source in source_names
                },
            }
        elif list(sources) != source_names:
            raise ValueError("stage budget update source order/set drifted")
        source_input = 0
        source_loss = 0
        for source in source_names:
            metrics = _mapping(
                sources[source],
                label=f"stage budget update[{index}].sources[{source!r}]",
            )
            for key in ("draws", "rows", "input_tokens", "loss_tokens"):
                _strict_int(
                    metrics.get(key),
                    label=f"stage budget update[{index}].sources[{source!r}].{key}",
                    minimum=0,
                )
            if metrics["draws"] != metrics["rows"]:
                raise ValueError("stage budget draw/row accounting mismatch")
            source_input += metrics["input_tokens"]
            source_loss += metrics["loss_tokens"]
            if index < completed_steps:
                destination = output["sources"][source]
                for key in ("rows", "input_tokens", "loss_tokens"):
                    destination[key] += metrics[key]
        if source_input != input_tokens or source_loss != loss_tokens:
            raise ValueError("stage budget update totals do not equal its source totals")
        if index < completed_steps:
            output["input_tokens"] += input_tokens
            output["loss_tokens"] += loss_tokens
    if output is None:
        raise ValueError("stage budget update list must not be empty")
    return output


def _validate_planned_totals(
    value: Any,
    expected_accounting: Mapping[str, Any],
    *,
    label: str,
) -> None:
    totals = _mapping(value, label=label)
    _strict_int(
        totals.get("updates"),
        label=f"{label}.updates",
        minimum=1,
        expected=PRODUCTION_TOTAL_STEPS,
    )
    _strict_int(totals.get("input_tokens"), label=f"{label}.input_tokens", minimum=0)
    _strict_int(totals.get("loss_tokens"), label=f"{label}.loss_tokens", minimum=0)
    if (
        totals["input_tokens"] != expected_accounting["input_tokens"]
        or totals["loss_tokens"] != expected_accounting["loss_tokens"]
    ):
        raise ValueError(f"{label} token totals mismatch")
    sources = _mapping(totals.get("sources"), label=f"{label}.sources")
    if list(sources) != list(expected_accounting["sources"]):
        raise ValueError(f"{label} source order/set mismatch")
    for source, expected in expected_accounting["sources"].items():
        observed = _mapping(sources[source], label=f"{label}.sources[{source!r}]")
        for key in ("draws", "rows", "input_tokens", "loss_tokens"):
            _strict_int(observed.get(key), label=f"{label}.{source}.{key}", minimum=0)
        if observed["draws"] != observed["rows"]:
            raise ValueError(f"{label} draw/row accounting mismatch")
        for key in ("rows", "input_tokens", "loss_tokens"):
            if observed[key] != expected[key]:
                raise ValueError(f"{label} source accounting mismatch")


def _runner_data_identity(
    plan_data: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    def identities(key: str) -> list[dict[str, Any]]:
        values = plan_data.get(key, [])
        if not isinstance(values, list):
            raise TypeError(f"stage budget data.{key} must be a list")
        result = []
        for index, value in enumerate(values):
            record = _mapping(value, label=f"stage budget data.{key}[{index}]")
            result.append(
                _mapping(
                    record.get("artifact"),
                    label=f"stage budget data.{key}[{index}].artifact",
                )
            )
        return result

    identity: dict[str, Any] = {
        "conversations": identities("conversations"),
        "eval_conversations": identities("eval_conversations"),
        "decay_conversations": identities("decay_conversations"),
        "conversation_overlap_audit": copy.deepcopy(
            _mapping(
                plan_data.get("conversation_overlap_audit"),
                label="stage budget conversation overlap audit",
            )
        ),
    }
    if "eval_selection" in plan_data:
        identity["eval_selection"] = copy.deepcopy(plan_data["eval_selection"])
    if "conversation_prompt_contract" in plan_data:
        identity["conversation_prompt_contract"] = plan_data[
            "conversation_prompt_contract"
        ]
    identity["decision_sampling"] = copy.deepcopy(dict(sampling))
    return identity


def _validate_lineage(
    value: Any,
    *,
    config_sha256: str,
    model_config_sha256: str,
    data_sha256: str,
    tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    lineage = _mapping(value, label="SFT checkpoint lineage")
    expected_keys = {
        "version",
        "stage",
        "config_sha256",
        "model_config_sha256",
        "data_sha256",
        "tokenizer_sha256",
        "git",
        "parent_checkpoint_sha256",
    }
    if set(lineage) != expected_keys:
        raise ValueError("SFT checkpoint lineage fields differ from the exact stage contract")
    _strict_int(lineage.get("version"), label="lineage.version", minimum=1, expected=1)
    if lineage.get("stage") != "sft":
        raise ValueError("checkpoint lineage stage must be exactly 'sft'")
    expected = {
        "config_sha256": config_sha256,
        "model_config_sha256": model_config_sha256,
        "data_sha256": data_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    for key, digest in expected.items():
        _sha256(lineage.get(key), label=f"lineage.{key}")
        if lineage[key] != digest:
            raise ValueError(f"checkpoint lineage.{key} mismatch")
    git = _mapping(lineage.get("git"), label="checkpoint lineage.git")
    if set(git) != {"commit", "repository_sha256", "dirty", "worktree_sha256"}:
        raise ValueError("checkpoint Git lineage fields differ")
    if not isinstance(git.get("commit"), str) or re.fullmatch(
        r"[0-9a-f]{40}", git["commit"]
    ) is None:
        raise ValueError("checkpoint lineage.git.commit must be a lowercase Git SHA-1")
    _sha256(git.get("repository_sha256"), label="lineage.git.repository_sha256")
    _sha256(git.get("worktree_sha256"), label="lineage.git.worktree_sha256")
    if not isinstance(git.get("dirty"), bool):
        raise TypeError("checkpoint lineage.git.dirty must be boolean")
    return lineage


def _validate_metric_record(
    value: Any,
    *,
    label: str,
    expected_rows: int,
    expected_loss_tokens: int,
) -> dict[str, Any]:
    metrics = _mapping(value, label=label)
    expected_keys = {
        "rows",
        "assistant_loss_tokens",
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
    if set(metrics) != expected_keys:
        raise ValueError(f"{label} fields differ")
    _strict_int(
        metrics.get("rows"),
        label=f"{label}.rows",
        minimum=1,
        expected=expected_rows,
    )
    _strict_int(
        metrics.get("assistant_loss_tokens"),
        label=f"{label}.assistant_loss_tokens",
        minimum=1,
        expected=expected_loss_tokens,
    )
    _finite_number(metrics.get("mean_loss"), label=f"{label}.mean_loss", minimum=0.0)
    for key in ("assistant_token_accuracy", "assistant_sequence_accuracy"):
        _finite_number(
            metrics.get(key),
            label=f"{label}.{key}",
            minimum=0.0,
            maximum=1.0,
        )
    return metrics


def _validate_heldout_contract(
    value: Any,
    *,
    config: Mapping[str, Any],
    model_config: ModelConfig,
    plan_data: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(value, label="held-out contract")
    if contract.get("kind") != "deterministic_teacher_forced_assistant_tokens":
        raise ValueError("held-out contract kind mismatch")
    if contract.get("same_rows_pre_post") is not True:
        raise ValueError("held-out contract must require identical pre/post rows")
    if contract.get("conversation_prompt_contract") != config["data"].get(
        "conversation_prompt_contract"
    ):
        raise ValueError("held-out prompt contract mismatch")
    _sha256(contract.get("dataset_sha256"), label="held-out contract.dataset_sha256")
    expected_max_seq_len = min(
        _strict_int(
            config["data"].get("seq_len"),
            label="config.data.seq_len",
            minimum=2,
        ),
        _strict_int(
            model_config.max_seq_len,
            label="model config max_seq_len",
            minimum=2,
        ),
    )
    _strict_int(
        contract.get("max_seq_len"),
        label="held-out contract.max_seq_len",
        minimum=2,
        expected=expected_max_seq_len,
    )
    evaluation = _mapping(config.get("evaluation"), label="config.evaluation")
    expected_padding = evaluation.get("pad_to_input_tokens")
    if contract.get("pad_to_input_tokens") != expected_padding:
        raise ValueError("held-out fixed padding mismatch")
    if "eval_selection" in plan_data and contract.get("selection") != plan_data[
        "eval_selection"
    ]:
        raise ValueError("held-out selection audit mismatch")
    return contract


def _optimizer_step(value: Any, *, completed_steps: int, label: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != torch.Size([])
        or value.dtype != torch.float32
    ):
        raise ValueError(f"{label} must be one scalar float32 tensor")
    if not bool(torch.isfinite(value).item()) or value.item() != float(completed_steps):
        raise ValueError(f"{label} must equal completed steps")


def _adamw_group_contract(
    *,
    parameter_count: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    probe = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [probe],
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    group = dict(optimizer.state_dict()["param_groups"][0])
    group["params"] = list(range(parameter_count))
    return group


def _validate_optimizer(
    checkpoint: Mapping[str, Any],
    *,
    completed_steps: int,
    optimizer_names: Sequence[str],
    model_state: Mapping[str, torch.Tensor],
    expected_learning_rate: float,
    expected_weight_decay: float,
) -> dict[str, Any]:
    optimizer = _mapping(checkpoint.get("optimizer"), label="checkpoint optimizer")
    if set(optimizer) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state-dict fields differ")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("checkpoint optimizer must contain exactly one parameter group")
    group = _mapping(groups[0], label="checkpoint optimizer parameter group")
    expected_ids = list(range(len(optimizer_names)))
    expected_group = _adamw_group_contract(
        parameter_count=len(optimizer_names),
        learning_rate=expected_learning_rate,
        weight_decay=expected_weight_decay,
    )
    _require_exact(
        group,
        expected_group,
        label="checkpoint optimizer complete parameter-group contract",
    )
    state = _mapping(optimizer.get("state"), label="checkpoint optimizer state")
    if not _exact_equal(list(state), expected_ids):
        raise ValueError("checkpoint optimizer state parameter IDs differ")
    for parameter_id in expected_ids:
        parameter_name = optimizer_names[parameter_id]
        parameter = model_state[parameter_name]
        parameter_state = _mapping(
            state[parameter_id],
            label=f"checkpoint optimizer state[{parameter_id}]",
        )
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(
                f"checkpoint optimizer state[{parameter_id}] moment fields differ"
            )
        _optimizer_step(
            parameter_state.get("step"),
            completed_steps=completed_steps,
            label=f"checkpoint optimizer state[{parameter_id}].step",
        )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state.get(moment_name)
            if not isinstance(moment, torch.Tensor):
                raise TypeError(
                    f"checkpoint optimizer state[{parameter_id}].{moment_name} "
                    "must be a tensor"
                )
            if moment.shape != parameter.shape:
                raise ValueError(
                    f"checkpoint optimizer state[{parameter_id}].{moment_name} "
                    f"shape mismatch for {parameter_name}"
                )
            if moment.dtype != parameter.dtype:
                raise ValueError(
                    f"checkpoint optimizer state[{parameter_id}].{moment_name} "
                    f"dtype mismatch for {parameter_name}"
                )
            if not bool(torch.isfinite(moment).all().item()):
                raise ValueError(
                    f"checkpoint optimizer state[{parameter_id}].{moment_name} "
                    f"contains non-finite values"
                )
    return {
        "optimizer_group_count": 1,
        "optimizer_parameter_ids": expected_ids,
        "optimizer_state_parameter_ids": expected_ids,
        "optimizer_step_values": [completed_steps],
    }


def _checkpoint_record(
    checkpoint: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    completed_steps: int,
    expected_training_contract: Mapping[str, Any],
    expected_lm_sampling: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    expected_cfg: Mapping[str, Any],
    expected_dataset_accounting: Mapping[str, Any],
    expected_token_accounting: Mapping[str, Any],
    expected_heldout_contract: Mapping[str, Any],
    expected_heldout_pre: Mapping[str, Any],
    expected_prompt_contract: str,
    expected_tokenizer_metadata: Mapping[str, Any],
    expected_data_metadata: Mapping[str, Any],
    expected_execution: Mapping[str, Any],
    expected_frozen_hashes: Mapping[str, str],
    parent_state: Mapping[str, torch.Tensor],
    model_parameter_names: Sequence[str],
    model_parameter_specs: Mapping[str, tuple[torch.Size, torch.dtype]],
    optimizer_names: Sequence[str],
    archive: bool,
) -> dict[str, Any]:
    _load_validated_sft_resume_checkpoint(checkpoint)
    _strict_int(
        checkpoint.get("step"),
        label="checkpoint step",
        minimum=0,
        expected=completed_steps - 1,
    )
    sampling_state = _mapping(
        checkpoint.get("sampling_state"),
        label="checkpoint sampling state",
    )
    if set(sampling_state) != {
        "rng_state",
        "lm_cursor",
        "completed_steps",
        "completed_microbatches",
    }:
        raise ValueError("checkpoint sampling-state fields differ")
    _strict_int(
        sampling_state.get("completed_steps"),
        label="checkpoint completed_steps",
        minimum=1,
        expected=completed_steps,
    )
    _strict_int(
        sampling_state.get("completed_microbatches"),
        label="checkpoint completed_microbatches",
        minimum=1,
        expected=completed_steps * PRODUCTION_GRAD_ACCUM_STEPS,
    )
    _strict_int(
        sampling_state.get("lm_cursor"),
        label="checkpoint lm_cursor",
        minimum=1,
        expected=completed_steps * PRODUCTION_DECISIONS_PER_UPDATE,
    )
    history = checkpoint.get("loss_history")
    if not isinstance(history, list) or len(history) != completed_steps:
        raise ValueError("checkpoint loss_history length disagrees with completed steps")
    for index, loss in enumerate(history):
        _finite_number(loss, label=f"checkpoint loss_history[{index}]")

    _require_exact(
        checkpoint.get("training_contract"),
        expected_training_contract,
        label="checkpoint training contract",
    )
    _require_exact(
        expected_training_contract.get("lm_sampling"),
        expected_lm_sampling,
        label="checkpoint LM sampling",
    )
    _require_exact(checkpoint.get("lineage"), expected_lineage, label="checkpoint lineage")
    _require_exact(checkpoint.get("cfg"), expected_cfg, label="checkpoint model config")
    _require_exact(
        checkpoint.get("dataset_token_accounting"),
        expected_dataset_accounting,
        label="checkpoint dataset accounting",
    )
    _require_exact(
        checkpoint.get("token_accounting"),
        expected_token_accounting,
        label="checkpoint prefix token accounting",
    )
    if checkpoint.get("token_accounting_scope") != "language_model_microbatches":
        raise ValueError("checkpoint token-accounting scope mismatch")
    if checkpoint.get("conversation_prompt_contract") != expected_prompt_contract:
        raise ValueError("checkpoint conversation prompt contract mismatch")
    _require_exact(
        checkpoint.get("tokenizer"),
        expected_tokenizer_metadata,
        label="checkpoint tokenizer metadata",
    )
    _require_exact(
        checkpoint.get("data"),
        expected_data_metadata,
        label="checkpoint data metadata",
    )
    _require_exact(
        checkpoint.get("execution"),
        expected_execution,
        label="checkpoint execution metadata",
    )
    baseline = _validated_resume_heldout_baseline(
        checkpoint,
        expected_contract=expected_heldout_contract,
    )
    if not isinstance(baseline, Mapping):
        raise ValueError("checkpoint held-out baseline is missing")
    _require_exact(
        baseline.get("pre"),
        expected_heldout_pre,
        label="checkpoint held-out pre baseline",
    )
    if checkpoint.get("tool_head") is not None or checkpoint.get("ptr_head") is not None:
        raise ValueError("sealed SFT checkpoint unexpectedly contains structured heads")

    state = _validate_model_state(
        checkpoint.get("state_dict"),
        parameter_names=model_parameter_names,
        parameter_specs=model_parameter_specs,
        label="checkpoint model state",
    )
    frozen_hashes: dict[str, str] = {}
    for name in PRODUCTION_FROZEN_PARAMETERS:
        parent_tensor = parent_state.get(name)
        child_tensor = state.get(name)
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(
            child_tensor,
            torch.Tensor,
        ):
            raise TypeError(f"frozen parameter {name} must be a tensor")
        if not torch.equal(parent_tensor, child_tensor):
            raise ValueError(f"checkpoint changed frozen tensor {name}")
        frozen_hashes[name] = _resume_sha256(child_tensor)
    if frozen_hashes != dict(expected_frozen_hashes):
        raise ValueError("checkpoint frozen tensor identities differ from parent")
    first_changed = None
    for name in optimizer_names:
        parent_tensor = parent_state.get(name)
        child_tensor = state.get(name)
        if not isinstance(parent_tensor, torch.Tensor) or not isinstance(
            child_tensor,
            torch.Tensor,
        ):
            raise TypeError(f"unfrozen parameter {name} must be a tensor")
        if not torch.equal(parent_tensor, child_tensor):
            first_changed = name
            break
    if first_changed is None:
        raise ValueError("checkpoint proves no unfrozen model transition")
    optimizer_record = _validate_optimizer(
        checkpoint,
        completed_steps=completed_steps,
        optimizer_names=optimizer_names,
        model_state=state,
        expected_learning_rate=cosine_lr(
            completed_steps - 1,
            PRODUCTION_TOTAL_STEPS,
            float(expected_training_contract["lr"]),
            int(expected_training_contract["warmup"]),
            0.1,
        ),
        expected_weight_decay=float(
            _mapping(
                expected_training_contract["optimizer"],
                label="expected training optimizer",
            )["weight_decay"]
        ),
    )
    if archive:
        match = _ARCHIVE_RE.fullmatch(Path(str(artifact["path"])).name)
        if match is None or int(match.group(1)) != completed_steps:
            raise ValueError("archive filename disagrees with sealed completed steps")
    return {
        **dict(artifact),
        "checkpoint_step": completed_steps - 1,
        "completed_steps": completed_steps,
        "completed_microbatches": completed_steps * PRODUCTION_GRAD_ACCUM_STEPS,
        "lm_cursor": completed_steps * PRODUCTION_DECISIONS_PER_UPDATE,
        "loss_history_length": completed_steps,
        "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
        "training_contract_sha256": canonical_sha256(expected_training_contract),
        "lm_sampling_sha256": canonical_sha256(expected_lm_sampling),
        "token_accounting_sha256": canonical_sha256(expected_token_accounting),
        "frozen_tensor_sha256": frozen_hashes,
        "first_changed_unfrozen_model_parameter": first_changed,
        **optimizer_record,
    }


def _validate_training_contract(
    value: Any,
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(value, label="SFT training contract")
    _require_exact(
        contract,
        expected_contract,
        label="complete recomputed SFT training contract",
    )
    _sha256(
        contract.get("prepared_data_sha256"),
        label="training contract prepared_data_sha256",
    )
    return contract


def _validate_preflight_source_files(
    value: Any,
    *,
    source_inputs: Sequence[Mapping[str, Any]],
) -> None:
    configured: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    order: list[tuple[str, int]] = []
    for raw_input in source_inputs:
        source_input = _mapping(raw_input, label="configured source input")
        role = source_input.get("role")
        index = source_input.get("index")
        kind = source_input.get("kind")
        if role == "tokenizer":
            continue
        if (
            role not in {"train", "decay", "eval"}
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or kind not in {"jsonl", "manifest", "generator_config"}
        ):
            raise ValueError("configured source input inventory is invalid")
        key = (role, index)
        if key not in configured:
            configured[key] = {}
            order.append(key)
        if kind in configured[key]:
            raise ValueError("configured source input inventory contains duplicate roles")
        configured[key][str(kind)] = source_input

    records = value
    if not isinstance(records, list) or len(records) != len(order):
        raise ValueError("preflight data artifact inventory differs from configured sources")
    for raw_record, key in zip(records, order, strict=True):
        role, index = key
        expected = configured[key]
        record = _mapping(raw_record, label="preflight data artifact")
        expected_keys = {"role", "index", *expected}
        if set(record) != expected_keys:
            raise ValueError("preflight data artifact fields differ from configured sources")
        if record.get("role") != role:
            raise ValueError("preflight data artifact role differs from configured sources")
        _strict_int(
            record.get("index"),
            label="preflight data artifact index",
            minimum=0,
            expected=index,
        )
        for kind, source_input in expected.items():
            _artifact_matches(
                record.get(kind),
                source_input,
                label=f"preflight {role} source {index} {kind}",
            )


def _validate_preflight(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_artifact: Mapping[str, Any],
    model_config: ModelConfig,
    model_config_artifact: Mapping[str, Any],
    parent_artifact: Mapping[str, Any],
    parent_pins: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    parent_state: Mapping[str, torch.Tensor],
    data_identity: Mapping[str, Any],
    expected_lm_sampling: Mapping[str, Any],
    expected_sampling_evidence: Mapping[str, Any],
    expected_training_contract: Mapping[str, Any],
    expected_data_metadata: Mapping[str, Any],
    expected_tokenizer_sha256: str,
    source_inputs: Sequence[Mapping[str, Any]],
    model_parameter_names: Sequence[str],
    model_parameter_specs: Mapping[str, tuple[torch.Size, torch.dtype]],
    optimizer_names: Sequence[str],
) -> dict[str, Any]:
    expected_preflight_training_contract = copy.deepcopy(
        dict(expected_training_contract)
    )
    _strict_int(
        expected_preflight_training_contract.get("checkpoint_archive_every"),
        label="production training contract checkpoint_archive_every",
        minimum=1,
        expected=PRODUCTION_CHECKPOINT_EVERY,
    )
    expected_preflight_training_contract["checkpoint_archive_every"] = 1
    expected_preflight_training_contract_sha256 = canonical_sha256(
        expected_preflight_training_contract
    )

    _exact_keys(
        receipt,
        frozenset(
            {
                "artifacts",
                "effective",
                "environment",
                "error",
                "finished_at_utc",
                "kind",
                "measurement",
                "metrics",
                "model",
                "receipt_self_sha256",
                "schema_version",
                "source",
                "started_at_utc",
                "status",
                "validation_errors",
            }
        ),
        label="passed SFT preflight receipt",
    )
    schema_version = receipt.get("schema_version")
    if (
        receipt.get("kind") != PREFLIGHT_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PREFLIGHT_SCHEMA_VERSION
    ):
        raise ValueError("preflight receipt kind/schema mismatch")
    assert_preflight_receipt(receipt)
    if receipt.get("status") != "passed":
        raise ValueError("production verification requires a passed preflight")
    if receipt.get("validation_errors") != [] or receipt.get("error") is not None:
        raise ValueError("passed preflight contains validation errors")
    for key in ("started_at_utc", "finished_at_utc"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ValueError(f"passed preflight {key} is missing")

    source = _mapping(receipt.get("source"), label="preflight source")
    _exact_keys(
        source,
        frozenset(
            {
                "config",
                "config_after",
                "data_artifacts",
                "data_artifacts_after",
                "data_artifacts_untouched",
                "model_config",
                "model_config_after",
                "production_sft_output_after",
                "production_sft_output_before",
                "production_sft_output_untouched",
                "sft_data_lineage",
                "sft_parent_checkpoint",
                "sft_parent_checkpoint_after",
                "sft_parent_checkpoint_untouched",
                "sft_sampling_lineage",
                "source_artifacts_untouched",
                "tokenizer",
                "tokenizer_after",
            }
        ),
        label="preflight source",
    )
    _artifact_matches(
        source.get("config"),
        config_artifact,
        label="preflight source config",
    )
    if source["config"].get("canonical_sha256") != canonical_sha256(config):
        raise ValueError("preflight canonical config identity mismatch")
    _artifact_matches(
        source.get("model_config"),
        model_config_artifact,
        label="preflight source model config",
    )
    if source["model_config"].get("canonical_sha256") != canonical_sha256(
        model_config.__dict__
    ):
        raise ValueError("preflight canonical model-config identity mismatch")
    _artifact_matches(
        source.get("sft_parent_checkpoint"),
        parent_artifact,
        label="preflight parent checkpoint",
    )
    for key in (
        "sft_parent_checkpoint_untouched",
        "data_artifacts_untouched",
        "source_artifacts_untouched",
        "production_sft_output_untouched",
    ):
        if source.get(key) is not True:
            raise ValueError(f"preflight source.{key} must be exactly true")
    if source.get("config_after") != {
        key: value
        for key, value in _mapping(source["config"], label="preflight source config").items()
        if key != "canonical_sha256"
    }:
        raise ValueError("preflight source config changed during execution")
    if source.get("model_config_after") != {
        key: value
        for key, value in _mapping(
            source["model_config"],
            label="preflight source model config",
        ).items()
        if key != "canonical_sha256"
    }:
        raise ValueError("preflight source model config changed during execution")
    parent_before = _mapping(
        source["sft_parent_checkpoint"],
        label="preflight source parent checkpoint",
    )
    if source.get("sft_parent_checkpoint_after") != {
        key: value for key, value in parent_before.items() if key != "completion"
    }:
        raise ValueError("preflight parent checkpoint changed during execution")
    if source.get("data_artifacts") != source.get("data_artifacts_after"):
        raise ValueError("preflight data artifacts changed during execution")
    if not isinstance(source.get("data_artifacts"), list) or not source["data_artifacts"]:
        raise ValueError("preflight did not bind the configured SFT data artifacts")
    _validate_preflight_source_files(
        source["data_artifacts"],
        source_inputs=source_inputs,
    )
    if source.get("production_sft_output_before") != source.get(
        "production_sft_output_after"
    ):
        raise ValueError("preflight production output snapshots differ")
    tokenizer_source = _mapping(
        source.get("tokenizer"),
        label="preflight source tokenizer",
    )
    if (
        tokenizer_source.get("tokenizer_kind") != config["data"]["tokenizer"]["kind"]
        or tokenizer_source.get("lineage_sha256")
        != expected_tokenizer_sha256
    ):
        raise ValueError("preflight tokenizer contract mismatch")
    expected_tokenizer_path = config["data"]["tokenizer"].get("path")
    tokenizer_inputs = [
        _mapping(item, label="configured tokenizer input")
        for item in source_inputs
        if item.get("role") == "tokenizer"
    ]
    if expected_tokenizer_path is not None:
        if len(tokenizer_inputs) != 1:
            raise ValueError("configured tokenizer is absent from the source inventory")
        _artifact_matches(
            tokenizer_source,
            tokenizer_inputs[0],
            label="preflight tokenizer source",
        )
        if not _same_path(tokenizer_source.get("path", ""), expected_tokenizer_path):
            raise ValueError("preflight tokenizer path mismatch")
        if tokenizer_source.get("sha256") != expected_tokenizer_sha256:
            raise ValueError("preflight tokenizer file identity mismatch")
        if source.get("tokenizer_after") != {
            key: value
            for key, value in tokenizer_source.items()
            if key not in {"tokenizer_kind", "lineage_sha256"}
        }:
            raise ValueError("preflight tokenizer changed during execution")
    elif tokenizer_inputs:
        raise ValueError("source inventory records an unconfigured tokenizer file")

    lineage = _mapping(
        source.get("sft_data_lineage"),
        label="preflight SFT data lineage",
    )
    _require_exact(
        lineage.get("identity"),
        data_identity,
        label="preflight SFT data identity",
    )
    if lineage.get("sha256") != canonical_sha256(data_identity):
        raise ValueError("preflight SFT data identity hash mismatch")
    sampling_lineage = _mapping(
        source.get("sft_sampling_lineage"),
        label="preflight SFT sampling lineage",
    )
    _require_exact(
        sampling_lineage,
        expected_sampling_evidence,
        label="complete preflight SFT sampling evidence",
    )
    production = _mapping(
        sampling_lineage.get("production"),
        label="preflight production sampling",
    )
    _require_exact(
        production.get("sampling_contract"),
        expected_lm_sampling,
        label="preflight production sampling contract",
    )
    if production.get("sampling_contract_sha256") != canonical_sha256(
        expected_lm_sampling
    ):
        raise ValueError("preflight production sampling self-hash mismatch")

    effective = _mapping(receipt.get("effective"), label="preflight effective")
    _exact_keys(
        effective,
        frozenset(
            {
                "config",
                "config_after",
                "config_payload",
                "config_untouched",
                "contract",
            }
        ),
        label="preflight effective",
    )
    if effective.get("config_untouched") is not True:
        raise ValueError("preflight effective config was not stable")
    effective_config_identity = _mapping(
        effective.get("config"),
        label="preflight effective config identity",
    )
    effective_payload = _mapping(
        effective.get("config_payload"),
        label="preflight effective config payload",
    )
    if effective_config_identity.get("canonical_sha256") != canonical_sha256(
        effective_payload
    ):
        raise ValueError("preflight effective config canonical identity mismatch")
    if effective.get("config_after") != {
        key: value
        for key, value in effective_config_identity.items()
        if key != "canonical_sha256"
    }:
        raise ValueError("preflight effective config changed during execution")
    effective_config_path = _safe_absolute_path(
        effective_config_identity.get("path"),
        label="preflight effective config path",
    )
    (
        effective_config_bytes,
        observed_effective_identity,
        effective_config_state,
    ) = _read_stable_regular(
        effective_config_path,
        label="preflight effective config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    _artifact_matches(
        effective_config_identity,
        observed_effective_identity,
        label="preflight effective config artifact",
    )
    observed_effective_payload = _strict_yaml(
        effective_config_bytes,
        label="preflight effective config",
    )
    _require_exact(
        observed_effective_payload,
        effective_payload,
        label="preflight effective config bytes/payload",
    )

    contract = _mapping(effective.get("contract"), label="preflight effective contract")
    if contract.get("stage") != "sft":
        raise ValueError("preflight effective stage mismatch")
    for key, expected in (
        ("production_schedule_total_steps", PRODUCTION_TOTAL_STEPS),
        ("micro_batch_size", PRODUCTION_MICRO_BATCH_SIZE),
        ("grad_accum_steps", PRODUCTION_GRAD_ACCUM_STEPS),
        ("effective_batch_size", PRODUCTION_DECISIONS_PER_UPDATE),
    ):
        _strict_int(
            contract.get(key),
            label=f"preflight effective contract.{key}",
            minimum=1,
            expected=expected,
        )
    execution_contract = _derive_sft_preflight_execution_contract(
        config,
        data_identity,
    )
    _require_exact(
        execution_contract,
        {
            "execution_optimizer_update_limit": 8,
            "first_pulse_update_zero_based": 7,
            "executed_through_first_pulse": True,
            "executed_lm_decisions": 128,
        },
        label="production preflight execution prefix",
    )
    for key, expected in execution_contract.items():
        if not _exact_equal(contract.get(key), expected):
            raise ValueError(f"preflight effective contract.{key} mismatch")
    expected_learning_rates = _expected_sft_executed_learning_rates(
        config,
        execution_update_limit=8,
    )
    required_execution_values = {
        "optimizer_updates": 8,
        "realized_optimizer_updates": 8,
        "optimizer_parameter_step_values": [8],
        "optimizer_learning_rates": [expected_learning_rates[-1]],
        "expected_step_zero_learning_rate": expected_learning_rates[0],
        "expected_executed_learning_rates": expected_learning_rates,
        "expected_last_executed_learning_rate": expected_learning_rates[-1],
        "any_executed_learning_rate_nonzero": True,
        "resume": False,
        "checkpoint_every": 1,
        "checkpoint_output": "isolated_work_directory",
        "expected_completed_lm_cursor": 128,
        "observed_completed_lm_cursor": 128,
        "pad_to_input_tokens": config["batch"]["pad_to_input_tokens"],
    }
    for key, expected in required_execution_values.items():
        if not _exact_equal(contract.get(key), expected):
            raise ValueError(f"preflight effective contract.{key} mismatch")
    _require_exact(
        contract.get("continuation"),
        config["continuation"],
        label="preflight continuation",
    )
    if contract.get("parent_checkpoint_sha256") != parent_artifact["sha256"]:
        raise ValueError("preflight parent checkpoint identity mismatch")
    _require_exact(contract.get("parent_pins"), parent_pins, label="preflight parent pins")
    _require_exact(
        contract.get("parent_anchor_binding"),
        parent_binding,
        label="preflight parent-anchor binding",
    )
    _require_exact(
        contract.get("lm_sampling"),
        expected_lm_sampling,
        label="preflight observed LM sampling",
    )
    if contract.get("production_lm_sampling_sha256") != canonical_sha256(
        expected_lm_sampling
    ):
        raise ValueError("preflight production LM sampling identity mismatch")
    _require_exact(
        contract.get("exercised_lm_prefix"),
        expected_sampling_evidence["exercised_prefix"],
        label="preflight exercised production LM prefix",
    )
    scope = _mapping(
        contract.get("model_parameter_scope"),
        label="preflight model parameter scope",
    )
    required_scope = {
        "model_named_parameter_names": list(model_parameter_names),
        "configured_frozen_model_parameter_names": list(PRODUCTION_FROZEN_PARAMETERS),
        "expected_optimizer_model_parameter_names": list(optimizer_names),
        "training_contract_frozen_model_parameter_names": list(
            PRODUCTION_FROZEN_PARAMETERS
        ),
        "training_contract_optimizer_model_parameter_names": list(optimizer_names),
        "all_auxiliary_heads_disabled": True,
        "optimizer_parameter_count_matches_expected": True,
        "frozen_model_tensors_exactly_preserved": True,
        "execution_optimizer_update_limit": 8,
        "executed_learning_rates": expected_learning_rates,
        "step_zero_learning_rate": expected_learning_rates[0],
        "last_executed_learning_rate": expected_learning_rates[-1],
        "any_executed_learning_rate_nonzero": True,
        "expected_completed_lm_cursor": 128,
        "observed_completed_lm_cursor": 128,
        "completed_lm_cursor_matches_expected": True,
    }
    for key, expected in required_scope.items():
        if not _exact_equal(scope.get(key), expected):
            raise ValueError(f"preflight model parameter scope {key} mismatch")
    _strict_int(
        scope.get("optimizer_param_group_parameter_count"),
        label="preflight optimizer parameter count",
        minimum=1,
        expected=len(optimizer_names),
    )
    if scope.get("unfrozen_model_transition_required") is not True:
        raise ValueError("preflight did not require an unfrozen model transition")
    changed = scope.get("first_changed_unfrozen_model_parameter")
    if not isinstance(changed, str) or changed not in optimizer_names:
        raise ValueError("preflight did not prove an unfrozen model transition")
    if scope.get("non_learning_limitation") is not None:
        raise ValueError("preflight contains a non-learning limitation")

    artifacts = _mapping(receipt.get("artifacts"), label="preflight artifacts")
    _exact_keys(
        artifacts,
        frozenset({"checkpoint", "metrics"}),
        label="preflight artifacts",
    )
    checkpoint_identity = _mapping(
        artifacts.get("checkpoint"),
        label="preflight isolated checkpoint artifact",
    )
    metrics_identity = _mapping(
        artifacts.get("metrics"),
        label="preflight isolated metrics artifact",
    )
    checkpoint_path = _safe_absolute_path(
        checkpoint_identity.get("path"),
        label="preflight isolated checkpoint path",
    )
    metrics_path = _safe_absolute_path(
        metrics_identity.get("path"),
        label="preflight isolated metrics path",
    )
    if checkpoint_path.parent != metrics_path.parent:
        raise ValueError("preflight isolated checkpoint/metrics directories differ")
    if checkpoint_path.name != "latest.pt" or metrics_path.name != "metrics.json":
        raise ValueError("preflight isolated artifact names differ from the runner contract")

    expected_effective = copy.deepcopy(dict(config))
    runtime = _mapping(expected_effective.get("runtime"), label="preflight effective runtime")
    runtime["resume"] = False
    runtime["device"] = "mps"
    expected_effective["runtime"] = runtime
    log = _mapping(expected_effective.get("log"), label="preflight effective log")
    log["out_dir"] = str(checkpoint_path.parent)
    log["ckpt_every"] = 1
    log.pop("mirror_dir", None)
    expected_effective["log"] = log
    _require_exact(
        effective_payload,
        expected_effective,
        label="preflight complete effective config",
    )
    preflight_requested_device, preflight_requested_dtype = (
        _configured_runtime_requests(
            effective_payload,
            label="preflight effective config",
        )
    )
    if preflight_requested_device != "mps":
        raise ValueError("production preflight effective config must explicitly request MPS")

    (
        checkpoint,
        observed_checkpoint_identity,
        isolated_checkpoint_state,
    ) = _load_stable_checkpoint(
        checkpoint_path,
        label="preflight isolated SFT checkpoint",
    )
    _artifact_matches(
        checkpoint_identity,
        observed_checkpoint_identity,
        label="preflight isolated checkpoint artifact",
    )
    if checkpoint_identity.get("resume_integrity_sha256") != checkpoint.get(
        "resume_integrity_sha256"
    ):
        raise ValueError("preflight isolated checkpoint resume identity mismatch")
    _load_validated_sft_resume_checkpoint(checkpoint)
    _strict_int(
        checkpoint.get("step"),
        label="preflight isolated checkpoint step",
        minimum=0,
        expected=7,
    )
    isolated_training_contract = _mapping(
        checkpoint.get("training_contract"),
        label="preflight isolated complete training contract",
    )
    _require_exact(
        isolated_training_contract,
        expected_preflight_training_contract,
        label="preflight isolated complete training contract",
    )
    if (
        canonical_sha256(isolated_training_contract)
        != expected_preflight_training_contract_sha256
    ):
        raise ValueError("preflight isolated training contract canonical hash mismatch")
    _require_exact(
        checkpoint.get("data"),
        expected_data_metadata,
        label="preflight isolated data metadata",
    )
    preflight_state = _validate_model_state(
        checkpoint.get("state_dict"),
        parameter_names=model_parameter_names,
        parameter_specs=model_parameter_specs,
        label="preflight isolated model state",
    )
    for name in PRODUCTION_FROZEN_PARAMETERS:
        if not torch.equal(preflight_state[name], parent_state[name]):
            raise ValueError(f"preflight isolated checkpoint changed frozen tensor {name}")
    if not any(
        not torch.equal(preflight_state[name], parent_state[name])
        for name in optimizer_names
    ):
        raise ValueError("preflight isolated checkpoint proves no unfrozen transition")
    _validate_optimizer(
        checkpoint,
        completed_steps=8,
        optimizer_names=optimizer_names,
        model_state=preflight_state,
        expected_learning_rate=expected_learning_rates[-1],
        expected_weight_decay=float(config["optim"]["weight_decay"]),
    )

    (
        metrics_payload,
        observed_metrics_identity,
        isolated_metrics_state,
    ) = _read_stable_regular(
        metrics_path,
        label="preflight isolated SFT metrics",
        max_bytes=_MAX_JSON_BYTES,
    )
    _artifact_matches(
        metrics_identity,
        observed_metrics_identity,
        label="preflight isolated metrics artifact",
    )
    isolated_metrics = _strict_json(
        metrics_payload,
        label="preflight isolated SFT metrics",
    )
    if isolated_metrics.get("stage") != "sft" or isolated_metrics.get("loss_steps") != 8:
        raise ValueError("preflight isolated metrics do not prove eight SFT updates")
    if not _same_path(isolated_metrics.get("checkpoint", ""), checkpoint_path):
        raise ValueError("preflight isolated metrics checkpoint path mismatch")
    isolated_execution = _validate_mps_fp32_execution(
        isolated_metrics.get("execution"),
        expected_requested_device=preflight_requested_device,
        expected_requested_dtype=preflight_requested_dtype,
        label="preflight isolated metrics execution",
    )
    checkpoint_execution = _validate_mps_fp32_execution(
        checkpoint.get("execution"),
        expected_requested_device=preflight_requested_device,
        expected_requested_dtype=preflight_requested_dtype,
        label="preflight isolated checkpoint execution",
    )
    _require_exact(
        isolated_execution,
        checkpoint_execution,
        label="preflight isolated metrics/checkpoint execution",
    )
    _require_exact(
        isolated_metrics.get("lm_sampling"),
        expected_lm_sampling,
        label="preflight isolated LM sampling",
    )

    model = _mapping(receipt.get("model"), label="preflight model")
    _require_exact(
        model,
        {
            "name": PRODUCTION_MODEL_NAME,
            "exact_parameters": PRODUCTION_MODEL_PARAMETERS,
            "max_seq_len": model_config.max_seq_len,
            "vocab_size": model_config.vocab_size,
        },
        label="preflight frozen 1M model identity",
    )
    environment = _mapping(receipt.get("environment"), label="preflight environment")
    if environment.get("mps_built") is not True or environment.get("mps_available") is not True:
        raise ValueError("passed preflight did not execute on an available MPS backend")
    measurement = _mapping(receipt.get("measurement"), label="preflight measurement")
    if measurement.get("non_learning_limitation") is not None:
        raise ValueError("preflight measurement contains a non-learning limitation")
    preflight_summary = _mapping(
        receipt.get("metrics"),
        label="passed preflight isolated metrics summary",
    )
    _require_exact(
        preflight_summary.get("execution"),
        isolated_execution,
        label="preflight receipt/live execution evidence",
    )
    return {
        "artifacts": {
            "effective_config": {
                **observed_effective_identity,
                "canonical_sha256": canonical_sha256(observed_effective_payload),
            },
            "isolated_checkpoint": {
                **observed_checkpoint_identity,
                "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
            },
            "isolated_metrics": observed_metrics_identity,
        },
        "tracked": [
            (
                effective_config_path,
                "preflight effective SFT config",
                _MAX_CONFIG_BYTES,
                effective_config_state,
                str(observed_effective_identity["sha256"]),
            ),
            (
                checkpoint_path,
                "preflight isolated SFT checkpoint",
                _MAX_CHECKPOINT_BYTES,
                isolated_checkpoint_state,
                str(observed_checkpoint_identity["sha256"]),
            ),
            (
                metrics_path,
                "preflight isolated SFT metrics",
                _MAX_JSON_BYTES,
                isolated_metrics_state,
                str(observed_metrics_identity["sha256"]),
            ),
        ],
    }


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    latest_path: Path,
    latest_checkpoint: Mapping[str, Any],
    expected_token_accounting: Mapping[str, Any],
    expected_dataset_accounting: Mapping[str, Any],
    expected_lm_sampling: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    continuation: Mapping[str, Any],
    heldout_contract: Mapping[str, Any],
    heldout_pre: Mapping[str, Any],
    heldout_rows: int,
    heldout_loss_tokens: int,
) -> None:
    if metrics.get("stage") != "sft":
        raise ValueError("final metrics stage must be exactly 'sft'")
    if not isinstance(metrics.get("checkpoint"), str) or not _same_path(
        metrics["checkpoint"],
        latest_path,
    ):
        raise ValueError("final metrics checkpoint path mismatch")
    _strict_int(
        metrics.get("loss_steps"),
        label="final metrics loss_steps",
        minimum=1,
        expected=PRODUCTION_TOTAL_STEPS,
    )
    history = latest_checkpoint["loss_history"]
    if not _exact_equal(metrics.get("loss_last"), history[-1]):
        raise ValueError("final metrics loss_last disagrees with checkpoint history")
    _require_exact(
        metrics.get("token_accounting"),
        expected_token_accounting,
        label="final metrics token accounting",
    )
    _require_exact(
        metrics.get("dataset_token_accounting"),
        expected_dataset_accounting,
        label="final metrics dataset accounting",
    )
    if metrics.get("token_accounting_scope") != "language_model_microbatches":
        raise ValueError("final metrics token-accounting scope mismatch")
    _require_exact(
        metrics.get("lm_sampling"),
        expected_lm_sampling,
        label="final metrics LM sampling",
    )
    progress = {
        "planned_optimizer_updates": PRODUCTION_TOTAL_STEPS,
        "completed_optimizer_updates": PRODUCTION_TOTAL_STEPS,
        "partial": False,
    }
    _require_exact(
        metrics.get("fixed_horizon_progress"),
        progress,
        label="final metrics fixed-horizon progress",
    )
    _require_exact(
        latest_checkpoint.get("fixed_horizon_progress"),
        progress,
        label="latest checkpoint fixed-horizon progress",
    )
    _require_exact(metrics.get("lineage"), expected_lineage, label="final metrics lineage")
    _require_exact(
        metrics.get("data"),
        latest_checkpoint.get("data"),
        label="final metrics data metadata",
    )
    _require_exact(
        metrics.get("execution"),
        latest_checkpoint.get("execution"),
        label="final metrics execution metadata",
    )
    _require_exact(
        metrics.get("continuation"),
        continuation,
        label="final metrics continuation",
    )
    expected_heads = {"tool_pointer": False, "route": False, "dense_selector": False}
    _require_exact(
        metrics.get("structured_heads"),
        expected_heads,
        label="final metrics structured-head state",
    )
    for key in ("tool_head", "ptr_head", "route_head", "dense_selector"):
        if latest_checkpoint.get(key) is not None:
            raise ValueError(f"latest checkpoint unexpectedly contains {key}")
    if latest_checkpoint.get("heldout_structured_eval") is not None:
        raise ValueError("latest checkpoint unexpectedly contains structured held-out metrics")
    heldout = _mapping(metrics.get("heldout_eval"), label="final metrics held-out evaluation")
    _require_exact(
        latest_checkpoint.get("heldout_eval"),
        heldout,
        label="latest/metrics held-out evaluation",
    )
    _require_exact(heldout.get("contract"), heldout_contract, label="held-out contract")
    _require_exact(heldout.get("pre"), heldout_pre, label="held-out pre baseline")
    post = _validate_metric_record(
        heldout.get("post"),
        label="held-out post metrics",
        expected_rows=heldout_rows,
        expected_loss_tokens=heldout_loss_tokens,
    )
    delta = _mapping(heldout.get("delta"), label="held-out metric delta")
    expected_delta = {
        "mean_loss": post["mean_loss"] - heldout_pre["mean_loss"],
        "assistant_token_accuracy": (
            post["assistant_token_accuracy"]
            - heldout_pre["assistant_token_accuracy"]
        ),
        "assistant_sequence_accuracy": (
            post["assistant_sequence_accuracy"]
            - heldout_pre["assistant_sequence_accuracy"]
        ),
    }
    _require_exact(delta, expected_delta, label="held-out metric delta")


def _checkpoint_receipt_identity(value: Mapping[str, Any]) -> None:
    required = {
        "bytes",
        "checkpoint_step",
        "completed_microbatches",
        "completed_steps",
        "first_changed_unfrozen_model_parameter",
        "frozen_tensor_sha256",
        "lm_cursor",
        "lm_sampling_sha256",
        "loss_history_length",
        "optimizer_group_count",
        "optimizer_parameter_ids",
        "optimizer_state_parameter_ids",
        "optimizer_step_values",
        "path",
        "resume_integrity_sha256",
        "sha256",
        "token_accounting_sha256",
        "training_contract_sha256",
    }
    if set(value) != required:
        raise ValueError("checkpoint receipt identity fields differ")


def assert_sft_production_receipt(receipt: Mapping[str, Any]) -> None:
    """Independently validate schema-v1 receipt identities and arithmetic."""

    root = _mapping(receipt, label="SFT production receipt")
    _exact_keys(root, _RECEIPT_KEYS, label="SFT production receipt")
    schema_version = root.get("schema_version")
    if (
        root.get("kind") != RECEIPT_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported SFT production receipt kind/schema")
    recorded_self_hash = _sha256(
        root.get("receipt_self_sha256"),
        label="SFT production receipt self-hash",
    )
    unhashed = copy.deepcopy(root)
    unhashed.pop("receipt_self_sha256")
    if recorded_self_hash != canonical_sha256(unhashed):
        raise ValueError("SFT production receipt self-hash mismatch")
    _require_exact(root.get("scope"), _SCOPE, label="SFT production receipt scope")
    roots = _validated_expected_roots(
        _mapping(root.get("external_roots"), label="receipt external roots")
    )

    validation = _mapping(root.get("validation"), label="receipt validation")
    _exact_keys(validation, _VALIDATION_KEYS, label="receipt validation")
    for key, value in validation.items():
        if value is not True:
            raise ValueError(f"receipt validation.{key} must be exactly true")

    contract = _mapping(root.get("contract"), label="receipt contract")
    required_contract_keys = {
        "archive_count",
        "checkpoint_every",
        "decisions_per_update",
        "data_sha256",
        "expected_archive_completed_steps",
        "final_checkpoint_step",
        "final_completed_microbatches",
        "final_lm_cursor",
        "frozen_model_parameter_names",
        "frozen_tensor_sha256",
        "grad_accum_steps",
        "heldout_pre",
        "lm_sampling_sha256",
        "model_parameter_names",
        "optimizer_model_parameter_names",
        "parent",
        "prepared_data_sha256",
        "stage",
        "structured_heads",
        "total_steps",
        "tokenizer_sha256",
        "training_contract_sha256",
    }
    if set(contract) != required_contract_keys:
        raise ValueError("receipt contract fields differ")
    if contract.get("stage") != "sft":
        raise ValueError("receipt contract stage mismatch")
    for key, expected in (
        ("total_steps", PRODUCTION_TOTAL_STEPS),
        ("checkpoint_every", PRODUCTION_CHECKPOINT_EVERY),
        ("archive_count", PRODUCTION_ARCHIVE_COUNT),
        ("grad_accum_steps", PRODUCTION_GRAD_ACCUM_STEPS),
        ("decisions_per_update", PRODUCTION_DECISIONS_PER_UPDATE),
        ("final_checkpoint_step", PRODUCTION_TOTAL_STEPS - 1),
        (
            "final_completed_microbatches",
            PRODUCTION_TOTAL_STEPS * PRODUCTION_GRAD_ACCUM_STEPS,
        ),
        ("final_lm_cursor", PRODUCTION_TOTAL_STEPS * PRODUCTION_DECISIONS_PER_UPDATE),
    ):
        _strict_int(
            contract.get(key),
            label=f"receipt contract.{key}",
            minimum=0,
            expected=expected,
        )
    expected_steps = list(
        range(
            PRODUCTION_CHECKPOINT_EVERY,
            PRODUCTION_TOTAL_STEPS + 1,
            PRODUCTION_CHECKPOINT_EVERY,
        )
    )
    _require_exact(
        contract.get("expected_archive_completed_steps"),
        expected_steps,
        label="receipt expected archive steps",
    )
    frozen_names = contract.get("frozen_model_parameter_names")
    if frozen_names != list(PRODUCTION_FROZEN_PARAMETERS):
        raise ValueError("receipt frozen-parameter contract differs")
    model_names = contract.get("model_parameter_names")
    optimizer_names = contract.get("optimizer_model_parameter_names")
    if (
        not isinstance(model_names, list)
        or not model_names
        or not all(isinstance(name, str) and name for name in model_names)
        or len(set(model_names)) != len(model_names)
        or not isinstance(optimizer_names, list)
    ):
        raise ValueError("receipt model parameter order is invalid")
    expected_optimizer_names = [
        name for name in model_names if name not in PRODUCTION_FROZEN_PARAMETERS
    ]
    if optimizer_names != expected_optimizer_names:
        raise ValueError("receipt optimizer parameter order is not model order minus frozen names")
    frozen_hashes = _mapping(
        contract.get("frozen_tensor_sha256"),
        label="receipt frozen tensor identities",
    )
    if set(frozen_hashes) != set(PRODUCTION_FROZEN_PARAMETERS):
        raise ValueError("receipt frozen tensor identities differ")
    for name, digest in frozen_hashes.items():
        _sha256(digest, label=f"receipt frozen tensor {name}")
    training_contract_sha256 = _sha256(
        contract.get("training_contract_sha256"),
        label="receipt training contract identity",
    )
    lm_sampling_sha256 = _sha256(
        contract.get("lm_sampling_sha256"),
        label="receipt LM sampling identity",
    )
    if _sha256(contract.get("data_sha256"), label="receipt data identity") != roots[
        "data_sha256"
    ]:
        raise ValueError("receipt data identity differs from its external root")
    if _sha256(
        contract.get("tokenizer_sha256"),
        label="receipt tokenizer identity",
    ) != roots["tokenizer_sha256"]:
        raise ValueError("receipt tokenizer identity differs from its external root")
    _sha256(
        contract.get("prepared_data_sha256"),
        label="receipt prepared-data identity",
    )
    parent = _mapping(contract.get("parent"), label="receipt parent contract")
    for key in (
        "checkpoint_sha256",
        "resume_integrity_sha256",
        "training_contract_sha256",
        "lm_sampling_sha256",
    ):
        _sha256(parent.get(key), label=f"receipt parent.{key}")
    _strict_int(parent.get("completed_steps"), label="receipt parent completed_steps", minimum=1)
    _strict_int(
        parent.get("completed_lm_cursor"),
        label="receipt parent completed_lm_cursor",
        minimum=0,
    )
    heldout_pre = _mapping(contract.get("heldout_pre"), label="receipt held-out pre")
    if set(heldout_pre) != {
        "assistant_loss_tokens",
        "contract_sha256",
        "metrics_sha256",
        "rows",
    }:
        raise ValueError("receipt held-out pre fields differ")
    _strict_int(heldout_pre.get("rows"), label="receipt held-out rows", minimum=1)
    _strict_int(
        heldout_pre.get("assistant_loss_tokens"),
        label="receipt held-out loss tokens",
        minimum=1,
    )
    _sha256(heldout_pre.get("contract_sha256"), label="receipt held-out contract identity")
    _sha256(heldout_pre.get("metrics_sha256"), label="receipt held-out metrics identity")
    _require_exact(
        contract.get("structured_heads"),
        {
            "joint_tool_pointer": False,
            "train_dense_selector": False,
            "train_route_head": False,
        },
        label="receipt structured-head contract",
    )

    artifacts = _mapping(root.get("artifacts"), label="receipt artifacts")
    _exact_keys(artifacts, _ARTIFACT_KEYS, label="receipt artifacts")
    for name in (
        "budget_plan",
        "config",
        "metrics",
        "model_config",
        "parent_checkpoint",
        "preflight",
    ):
        identity = _mapping(artifacts.get(name), label=f"receipt artifact {name}")
        if not isinstance(identity.get("path"), str) or not identity["path"]:
            raise ValueError(f"receipt artifact {name}.path must be non-empty")
        _strict_int(identity.get("bytes"), label=f"receipt artifact {name}.bytes", minimum=0)
        _sha256(identity.get("sha256"), label=f"receipt artifact {name}.sha256")

    source_inputs = artifacts.get("source_inputs")
    if not isinstance(source_inputs, list) or not source_inputs:
        raise ValueError("receipt source-input inventory must be a non-empty list")
    seen_source_keys: set[tuple[str, int, str]] = set()
    has_train_jsonl = False
    tokenizer_inputs = 0
    for raw_input in source_inputs:
        source_input = _mapping(raw_input, label="receipt source input")
        if set(source_input) != {"bytes", "index", "kind", "path", "role", "sha256"}:
            raise ValueError("receipt source-input identity fields differ")
        role = source_input.get("role")
        kind = source_input.get("kind")
        if role not in {"train", "decay", "eval", "tokenizer"}:
            raise ValueError("receipt source-input role is invalid")
        allowed_kinds = (
            {"tokenizer"}
            if role == "tokenizer"
            else {"jsonl", "manifest", "generator_config"}
        )
        if kind not in allowed_kinds:
            raise ValueError("receipt source-input kind is invalid")
        index = _strict_int(
            source_input.get("index"),
            label="receipt source-input index",
            minimum=0,
        )
        source_key = (str(role), index, str(kind))
        if source_key in seen_source_keys:
            raise ValueError("receipt source-input roles contain a duplicate")
        seen_source_keys.add(source_key)
        if not isinstance(source_input.get("path"), str) or not source_input["path"]:
            raise ValueError("receipt source-input path must be non-empty")
        _strict_int(
            source_input.get("bytes"),
            label="receipt source-input bytes",
            minimum=0,
        )
        digest = _sha256(
            source_input.get("sha256"),
            label="receipt source-input SHA-256",
        )
        if role == "train" and kind == "jsonl":
            has_train_jsonl = True
        if role == "tokenizer":
            tokenizer_inputs += 1
            if index != 0 or digest != roots["tokenizer_sha256"]:
                raise ValueError("receipt tokenizer input differs from its external root")
    if not has_train_jsonl or tokenizer_inputs > 1:
        raise ValueError("receipt source-input inventory is incomplete")

    preflight_evidence = _mapping(
        artifacts.get("preflight_evidence"),
        label="receipt preflight evidence",
    )
    _exact_keys(
        preflight_evidence,
        frozenset({"effective_config", "isolated_checkpoint", "isolated_metrics"}),
        label="receipt preflight evidence",
    )
    for name, expected_fields in (
        (
            "effective_config",
            {"bytes", "canonical_sha256", "path", "sha256"},
        ),
        (
            "isolated_checkpoint",
            {"bytes", "path", "resume_integrity_sha256", "sha256"},
        ),
        ("isolated_metrics", {"bytes", "path", "sha256"}),
    ):
        evidence = _mapping(
            preflight_evidence.get(name),
            label=f"receipt preflight evidence {name}",
        )
        if set(evidence) != expected_fields:
            raise ValueError(f"receipt preflight evidence {name} fields differ")
        if not isinstance(evidence.get("path"), str) or not evidence["path"]:
            raise ValueError(f"receipt preflight evidence {name}.path must be non-empty")
        _strict_int(
            evidence.get("bytes"),
            label=f"receipt preflight evidence {name}.bytes",
            minimum=0,
        )
        _sha256(
            evidence.get("sha256"),
            label=f"receipt preflight evidence {name}.sha256",
        )
    _sha256(
        preflight_evidence["effective_config"].get("canonical_sha256"),
        label="receipt preflight effective config canonical identity",
    )
    _sha256(
        preflight_evidence["isolated_checkpoint"].get("resume_integrity_sha256"),
        label="receipt preflight isolated checkpoint resume identity",
    )

    live_inventory = _mapping(
        artifacts.get("live_evidence_inventory"),
        label="receipt live-evidence inventory",
    )
    _exact_keys(
        live_inventory,
        frozenset({"count", "entries", "sha256"}),
        label="receipt live-evidence inventory",
    )
    live_entries = live_inventory.get("entries")
    if not isinstance(live_entries, list) or not live_entries:
        raise ValueError("receipt live-evidence inventory entries must be non-empty")
    _strict_int(
        live_inventory.get("count"),
        label="receipt live-evidence inventory count",
        minimum=1,
        expected=len(live_entries),
    )
    if _sha256(
        live_inventory.get("sha256"),
        label="receipt live-evidence inventory SHA-256",
    ) != canonical_sha256(live_entries):
        raise ValueError("receipt live-evidence inventory self-hash mismatch")
    seen_live_labels: set[str] = set()
    for raw_entry in live_entries:
        entry = _mapping(raw_entry, label="receipt live-evidence entry")
        if set(entry) != {"bytes", "label", "path", "sha256"}:
            raise ValueError("receipt live-evidence entry fields differ")
        label = entry.get("label")
        if not isinstance(label, str) or not label or label in seen_live_labels:
            raise ValueError("receipt live-evidence labels must be non-empty and unique")
        seen_live_labels.add(label)
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("receipt live-evidence path must be non-empty")
        _strict_int(
            entry.get("bytes"),
            label="receipt live-evidence bytes",
            minimum=0,
        )
        _sha256(entry.get("sha256"), label="receipt live-evidence SHA-256")
    _sha256(
        artifacts["budget_plan"].get("plan_self_sha256"),
        label="receipt stage-budget self-hash",
    )
    _sha256(
        artifacts["config"].get("canonical_sha256"),
        label="receipt canonical config identity",
    )
    _sha256(
        artifacts["model_config"].get("canonical_sha256"),
        label="receipt canonical model-config identity",
    )
    _sha256(
        artifacts["preflight"].get("receipt_self_sha256"),
        label="receipt preflight self-hash",
    )
    rooted_artifacts = {
        "config_file_sha256": artifacts["config"]["sha256"],
        "config_canonical_sha256": artifacts["config"]["canonical_sha256"],
        "model_config_file_sha256": artifacts["model_config"]["sha256"],
        "model_config_canonical_sha256": artifacts["model_config"][
            "canonical_sha256"
        ],
        "budget_plan_file_sha256": artifacts["budget_plan"]["sha256"],
        "budget_plan_self_sha256": artifacts["budget_plan"]["plan_self_sha256"],
        "preflight_file_sha256": artifacts["preflight"]["sha256"],
        "preflight_self_sha256": artifacts["preflight"]["receipt_self_sha256"],
        "parent_checkpoint_sha256": artifacts["parent_checkpoint"]["sha256"],
        "data_sha256": contract["data_sha256"],
        "tokenizer_sha256": contract["tokenizer_sha256"],
    }
    _require_exact(rooted_artifacts, roots, label="receipt externally rooted artifacts")
    if artifacts["preflight"].get("status") != "passed":
        raise ValueError("receipt preflight artifact must record passed status")
    _sha256(
        artifacts["parent_checkpoint"].get("resume_integrity_sha256"),
        label="receipt parent resume integrity",
    )
    if artifacts["parent_checkpoint"]["sha256"] != parent["checkpoint_sha256"]:
        raise ValueError("receipt parent file identity differs from parent contract")
    if (
        artifacts["parent_checkpoint"]["resume_integrity_sha256"]
        != parent["resume_integrity_sha256"]
    ):
        raise ValueError("receipt parent resume identity differs from parent contract")
    run_directory = _mapping(
        artifacts.get("run_directory"),
        label="receipt run directory",
    )
    if set(run_directory) != {"entry_count", "expected_entries", "path"}:
        raise ValueError("receipt run-directory identity fields differ")
    if not isinstance(run_directory.get("path"), str) or not run_directory["path"]:
        raise ValueError("receipt run-directory path must be non-empty")
    expected_entries = run_directory.get("expected_entries")
    if not isinstance(expected_entries, list) or len(expected_entries) != (
        PRODUCTION_ARCHIVE_COUNT + 2
    ):
        raise ValueError("receipt run-directory entry list has wrong cardinality")
    _strict_int(
        run_directory.get("entry_count"),
        label="receipt run-directory entry_count",
        minimum=1,
        expected=len(expected_entries),
    )

    archives = artifacts.get("archives")
    if not isinstance(archives, list) or len(archives) != PRODUCTION_ARCHIVE_COUNT:
        raise ValueError("receipt archive cardinality mismatch")
    archive_hashes: list[str] = []
    archive_resume_hashes: list[str] = []
    archive_accounting_hashes: list[str] = []
    for expected_completed, raw_archive in zip(expected_steps, archives, strict=True):
        archive = _mapping(raw_archive, label="receipt archive")
        _checkpoint_receipt_identity(archive)
        for key, expected in (
            ("completed_steps", expected_completed),
            ("checkpoint_step", expected_completed - 1),
            (
                "completed_microbatches",
                expected_completed * PRODUCTION_GRAD_ACCUM_STEPS,
            ),
            ("lm_cursor", expected_completed * PRODUCTION_DECISIONS_PER_UPDATE),
            ("loss_history_length", expected_completed),
            ("optimizer_group_count", 1),
        ):
            _strict_int(
                archive.get(key),
                label=f"receipt archive {key}",
                minimum=0,
                expected=expected,
            )
        match = _ARCHIVE_RE.fullmatch(Path(str(archive.get("path"))).name)
        if match is None or int(match.group(1)) != expected_completed:
            raise ValueError("receipt archive path/completed-step mismatch")
        _strict_int(archive.get("bytes"), label="receipt archive bytes", minimum=1)
        archive_hashes.append(_sha256(archive.get("sha256"), label="receipt archive SHA-256"))
        archive_resume_hashes.append(
            _sha256(
                archive.get("resume_integrity_sha256"),
                label="receipt archive resume integrity",
            )
        )
        if archive.get("training_contract_sha256") != training_contract_sha256:
            raise ValueError("receipt archive training contract identity mismatch")
        if archive.get("lm_sampling_sha256") != lm_sampling_sha256:
            raise ValueError("receipt archive LM sampling identity mismatch")
        archive_accounting_hashes.append(
            _sha256(
                archive.get("token_accounting_sha256"),
                label="receipt archive accounting identity",
            )
        )
        _require_exact(
            archive.get("frozen_tensor_sha256"),
            frozen_hashes,
            label="receipt archive frozen tensors",
        )
        changed = archive.get("first_changed_unfrozen_model_parameter")
        if not isinstance(changed, str) or changed not in optimizer_names:
            raise ValueError("receipt archive unfrozen transition evidence is invalid")
        expected_ids = list(range(len(optimizer_names)))
        for key in ("optimizer_parameter_ids", "optimizer_state_parameter_ids"):
            _require_exact(
                archive.get(key),
                expected_ids,
                label=f"receipt archive {key}",
            )
        _require_exact(
            archive.get("optimizer_step_values"),
            [expected_completed],
            label="receipt archive optimizer steps",
        )
    if len(set(archive_hashes)) != len(archive_hashes):
        raise ValueError("receipt archive file hashes are not unique")
    if len(set(archive_resume_hashes)) != len(archive_resume_hashes):
        raise ValueError("receipt archive resume identities are not unique")
    if len(set(archive_accounting_hashes)) != len(archive_accounting_hashes):
        raise ValueError("receipt archive accounting identities are not unique")

    latest = _mapping(
        artifacts.get("latest_checkpoint"),
        label="receipt latest checkpoint",
    )
    _checkpoint_receipt_identity(latest)
    _strict_int(latest.get("bytes"), label="receipt latest bytes", minimum=1)
    _sha256(latest.get("sha256"), label="receipt latest SHA-256")
    _sha256(
        latest.get("resume_integrity_sha256"),
        label="receipt latest resume integrity",
    )
    _sha256(
        latest.get("token_accounting_sha256"),
        label="receipt latest accounting identity",
    )
    for key, expected in (
        ("completed_steps", PRODUCTION_TOTAL_STEPS),
        ("checkpoint_step", PRODUCTION_TOTAL_STEPS - 1),
        (
            "completed_microbatches",
            PRODUCTION_TOTAL_STEPS * PRODUCTION_GRAD_ACCUM_STEPS,
        ),
        ("lm_cursor", PRODUCTION_TOTAL_STEPS * PRODUCTION_DECISIONS_PER_UPDATE),
        ("loss_history_length", PRODUCTION_TOTAL_STEPS),
        ("optimizer_group_count", 1),
    ):
        _strict_int(
            latest.get(key),
            label=f"receipt latest {key}",
            minimum=0,
            expected=expected,
        )
    if latest.get("resume_integrity_sha256") != archives[-1][
        "resume_integrity_sha256"
    ]:
        raise ValueError("receipt latest/final archive sealed identities differ")
    if latest.get("training_contract_sha256") != training_contract_sha256:
        raise ValueError("receipt latest training contract identity mismatch")
    if latest.get("lm_sampling_sha256") != lm_sampling_sha256:
        raise ValueError("receipt latest LM sampling identity mismatch")
    if latest.get("token_accounting_sha256") != archives[-1][
        "token_accounting_sha256"
    ]:
        raise ValueError("receipt latest/final archive accounting identities differ")
    _require_exact(
        latest.get("frozen_tensor_sha256"),
        frozen_hashes,
        label="receipt latest frozen tensors",
    )
    changed = latest.get("first_changed_unfrozen_model_parameter")
    if not isinstance(changed, str) or changed not in optimizer_names:
        raise ValueError("receipt latest unfrozen transition evidence is invalid")
    expected_ids = list(range(len(optimizer_names)))
    _require_exact(
        latest.get("optimizer_parameter_ids"),
        expected_ids,
        label="receipt latest optimizer parameter IDs",
    )
    _require_exact(
        latest.get("optimizer_state_parameter_ids"),
        expected_ids,
        label="receipt latest optimizer state parameter IDs",
    )
    _require_exact(
        latest.get("optimizer_step_values"),
        [PRODUCTION_TOTAL_STEPS],
        label="receipt latest optimizer steps",
    )

    def live_entry(label: str, value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "label": label,
            "path": value["path"],
            "bytes": value["bytes"],
            "sha256": value["sha256"],
        }

    expected_live_entries = [
        live_entry("production SFT config", artifacts["config"]),
    ]
    for source_input in source_inputs:
        if source_input["role"] == "tokenizer":
            label = "configured SFT tokenizer"
        elif source_input["kind"] == "jsonl":
            label = (
                f"configured SFT {source_input['role']} conversation JSONL "
                f"{source_input['index']}"
            )
        else:
            label = (
                f"configured SFT {source_input['role']} conversation "
                f"{source_input['kind']} {source_input['index']}"
            )
        expected_live_entries.append(live_entry(label, source_input))
    expected_live_entries.extend(
        [
            live_entry("production SFT model config", artifacts["model_config"]),
            live_entry("SFT stage-budget plan", artifacts["budget_plan"]),
            live_entry("passed SFT preflight receipt", artifacts["preflight"]),
            live_entry(
                "SFT continuation parent checkpoint",
                artifacts["parent_checkpoint"],
            ),
            live_entry(
                "preflight effective SFT config",
                preflight_evidence["effective_config"],
            ),
            live_entry(
                "preflight isolated SFT checkpoint",
                preflight_evidence["isolated_checkpoint"],
            ),
            live_entry(
                "preflight isolated SFT metrics",
                preflight_evidence["isolated_metrics"],
            ),
            live_entry("final enriched SFT checkpoint", latest),
        ]
    )
    expected_live_entries.extend(
        live_entry(
            f"SFT archive at {completed_steps} completed steps",
            archive,
        )
        for completed_steps, archive in zip(expected_steps, archives, strict=True)
    )
    expected_live_entries.append(live_entry("final SFT metrics", artifacts["metrics"]))
    _require_exact(
        live_entries,
        expected_live_entries,
        label="receipt complete live-evidence inventory",
    )

    expected_names = {
        "latest.pt",
        "metrics.json",
        *(Path(str(archive["path"])).name for archive in archives),
    }
    if set(expected_entries) != expected_names or len(set(expected_entries)) != len(
        expected_entries
    ):
        raise ValueError("receipt run-directory entries differ from artifact identities")


def verify_sft_production_receipt_bytes(payload: bytes) -> dict[str, Any]:
    """Parse and independently verify one canonical receipt payload."""

    receipt = _strict_json(payload, label="SFT production receipt")
    if payload != _canonical_receipt_bytes(receipt):
        raise ValueError("SFT production receipt is not canonical JSON")
    assert_sft_production_receipt(receipt)
    return receipt


def verify_sft_production_run(
    config_path: str | Path,
    budget_plan_path: str | Path,
    preflight_path: str | Path,
    *,
    expected_roots: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a completed parent-anchored SFT run and return a signed schema-v1 receipt."""

    roots = _validated_expected_roots(expected_roots)
    config_path = _safe_absolute_path(
        config_path,
        label="production SFT config path",
    )
    budget_plan_path = _safe_absolute_path(
        budget_plan_path,
        label="SFT stage-budget plan path",
    )
    preflight_path = _safe_absolute_path(
        preflight_path,
        label="passed SFT preflight path",
    )
    tracked: list[tuple[Path, str, int, _FileState, str]] = []

    config_payload, config_artifact, config_state = _read_stable_regular(
        config_path,
        label="production SFT config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    if config_artifact["sha256"] != roots["config_file_sha256"]:
        raise ValueError("production SFT config file differs from the external root")
    tracked.append(
        (
            config_path,
            "production SFT config",
            _MAX_CONFIG_BYTES,
            config_state,
            str(config_artifact["sha256"]),
        )
    )
    config = _strict_yaml(config_payload, label="production SFT config")
    if canonical_sha256(config) != roots["config_canonical_sha256"]:
        raise ValueError("production SFT canonical config differs from the external root")
    configured_source_inputs = _configured_source_inputs(config)
    source_input_records: list[dict[str, Any]] = []
    for source_input in configured_source_inputs:
        source_identity, source_state = _hash_stable_regular(
            source_input.path,
            label=source_input.label,
            max_bytes=source_input.max_bytes,
        )
        record = {
            "role": source_input.role,
            "index": source_input.index,
            "kind": source_input.kind,
            **source_identity,
        }
        source_input_records.append(record)
        tracked.append(
            (
                source_input.path,
                source_input.label,
                source_input.max_bytes,
                source_state,
                str(source_identity["sha256"]),
            )
        )
        if (
            source_input.kind == "tokenizer"
            and source_identity["sha256"] != roots["tokenizer_sha256"]
        ):
            raise ValueError("configured tokenizer file differs from the external root")
    validated_config = _validate_config(config)

    model_config_path = validated_config["model_config_path"]
    model_payload, model_artifact, model_state = _read_stable_regular(
        model_config_path,
        label="production SFT model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    if model_artifact["sha256"] != roots["model_config_file_sha256"]:
        raise ValueError("production model-config file differs from the external root")
    tracked.append(
        (
            model_config_path,
            "production SFT model config",
            _MAX_CONFIG_BYTES,
            model_state,
            str(model_artifact["sha256"]),
        )
    )
    model_mapping = _strict_yaml(model_payload, label="production SFT model config")
    if canonical_sha256(model_mapping) != roots["model_config_canonical_sha256"]:
        raise ValueError("canonical production model config differs from the external root")
    model_config = _model_config(model_mapping)

    plan_payload, plan_artifact, plan_state = _read_stable_regular(
        budget_plan_path,
        label="SFT stage-budget plan",
        max_bytes=_MAX_JSON_BYTES,
    )
    if plan_artifact["sha256"] != roots["budget_plan_file_sha256"]:
        raise ValueError("stage-budget plan file differs from the external root")
    tracked.append(
        (
            budget_plan_path,
            "SFT stage-budget plan",
            _MAX_JSON_BYTES,
            plan_state,
            str(plan_artifact["sha256"]),
        )
    )
    plan = _strict_json(plan_payload, label="SFT stage-budget plan")
    if plan_payload != canonical_plan_bytes(plan):
        raise ValueError("SFT stage-budget plan is not canonical JSON")
    assert_stage_budget_self_hash(plan)
    if plan.get("plan_self_sha256") != roots["budget_plan_self_sha256"]:
        raise ValueError("stage-budget plan self-hash differs from the external root")
    replayed_plan = verify_stage_budget_plan(budget_plan_path)
    if canonical_plan_bytes(replayed_plan) != plan_payload:
        raise ValueError("stage-budget replay differs from the stable plan bytes")
    plan_sampling, final_accounting = _validate_plan(
        plan,
        config_path=config_path,
        config=config,
        config_artifact=config_artifact,
        model_config_path=model_config_path,
        model_config=model_config,
        model_config_artifact=model_artifact,
    )
    preflight_payload, preflight_artifact, preflight_state = _read_stable_regular(
        preflight_path,
        label="passed SFT preflight receipt",
        max_bytes=_MAX_JSON_BYTES,
    )
    if preflight_artifact["sha256"] != roots["preflight_file_sha256"]:
        raise ValueError("passed SFT preflight file differs from the external root")
    tracked.append(
        (
            preflight_path,
            "passed SFT preflight receipt",
            _MAX_JSON_BYTES,
            preflight_state,
            str(preflight_artifact["sha256"]),
        )
    )
    preflight = _strict_json(preflight_payload, label="passed SFT preflight receipt")
    if preflight_payload != _canonical_receipt_bytes(preflight):
        raise ValueError("passed SFT preflight receipt is not canonical JSON")
    assert_preflight_receipt(preflight)
    if preflight.get("receipt_self_sha256") != roots["preflight_self_sha256"]:
        raise ValueError("passed SFT preflight self-hash differs from the external root")
    if preflight.get("status") != "passed":
        raise ValueError("production verification requires a passed SFT preflight")

    run_directory = validated_config["run_directory"]
    archive_steps = list(
        range(
            PRODUCTION_CHECKPOINT_EVERY,
            PRODUCTION_TOTAL_STEPS + 1,
            PRODUCTION_CHECKPOINT_EVERY,
        )
    )
    archive_names = [f"latest.step-{step:08d}.pt" for step in archive_steps]
    expected_names = {"latest.pt", "metrics.json", *archive_names}
    directory_state, directory_names = _snapshot_exact_directory(
        run_directory,
        expected_names=expected_names,
    )

    parent_path = validated_config["parent_path"]
    if _same_path(parent_path, run_directory) or run_directory in parent_path.parents:
        raise ValueError("configured parent checkpoint must be outside the production run")
    parent_checkpoint, parent_artifact, parent_state_record = _load_stable_checkpoint(
        parent_path,
        label="SFT continuation parent checkpoint",
    )
    if parent_artifact["sha256"] != roots["parent_checkpoint_sha256"]:
        raise ValueError("SFT parent checkpoint differs from the external root")
    tracked.append(
        (
            parent_path,
            "SFT continuation parent checkpoint",
            _MAX_CHECKPOINT_BYTES,
            parent_state_record,
            str(parent_artifact["sha256"]),
        )
    )
    parent_pins = _validate_sft_continuation_parent(
        parent_checkpoint,
        checkpoint_sha256=str(parent_artifact["sha256"]),
        continuation=validated_config["continuation"],
    )
    parent_binding = _validate_parent_anchored_sampling_parent(
        parent_checkpoint,
        validated_config["sampling_config"],
    )
    model_parameter_names, model_parameter_specs = _model_parameter_specs(model_config)
    parent_state = _validate_model_state(
        parent_checkpoint.get("state_dict"),
        parameter_names=model_parameter_names,
        parameter_specs=model_parameter_specs,
        label="SFT continuation parent model state",
    )
    if not set(PRODUCTION_FROZEN_PARAMETERS).issubset(model_parameter_names):
        raise ValueError("production frozen tensors are absent from the model")
    optimizer_names = [
        name for name in model_parameter_names if name not in PRODUCTION_FROZEN_PARAMETERS
    ]
    frozen_hashes = {
        name: _resume_sha256(parent_state[name]) for name in PRODUCTION_FROZEN_PARAMETERS
    }

    source_data_identity, source_sampling_evidence = (
        _derive_sft_data_identity_and_sampling(config)
    )
    data_identity, sampling_evidence = _bind_sft_parent_checkpoint_identity(
        source_data_identity,
        source_sampling_evidence,
        parent_checkpoint_binding=parent_binding,
    )
    expected_lm_sampling = _mapping(
        data_identity.get("decision_sampling"),
        label="independently recomputed production LM sampling",
    )
    if "parent_checkpoint_binding" in plan_sampling:
        raise ValueError("stage budget must not pre-populate the runtime parent binding")
    expected_plan_sampling = copy.deepcopy(dict(expected_lm_sampling))
    expected_plan_sampling.pop("parent_checkpoint_binding")
    _require_exact(
        plan_sampling,
        expected_plan_sampling,
        label="stage-budget/recomputed production LM sampling",
    )
    plan_data = _mapping(plan.get("data"), label="stage budget data")
    plan_data_identity = _runner_data_identity(
        plan_data,
        sampling=expected_lm_sampling,
    )
    _require_exact(
        plan_data_identity,
        data_identity,
        label="stage-budget/recomputed SFT data identity",
    )
    tokenizer_identity = _mapping(
        _mapping(plan.get("identity"), label="stage budget identity").get("tokenizer"),
        label="stage budget tokenizer identity",
    )
    tokenizer_sha256 = _sha256(
        tokenizer_identity.get("sha256"),
        label="stage budget tokenizer SHA-256",
    )
    if tokenizer_sha256 != roots["tokenizer_sha256"]:
        raise ValueError("stage-budget tokenizer differs from the external root")
    config_sha256 = _lineage_config_sha256(config)
    model_config_sha256 = canonical_sha256(model_config.__dict__)
    data_sha256 = canonical_sha256(data_identity)
    if data_sha256 != roots["data_sha256"]:
        raise ValueError("recomputed SFT data identity differs from the external root")

    latest_path = run_directory / "latest.pt"
    latest_checkpoint, latest_artifact, latest_state = _load_stable_checkpoint(
        latest_path,
        label="final enriched SFT checkpoint",
    )
    production_execution, production_amp_dtype = (
        _validated_production_runtime_evidence(
            config=config,
            execution=latest_checkpoint.get("execution"),
            training_contract=latest_checkpoint.get("training_contract"),
        )
    )
    materialization = _derive_expected_runner_materialization(
        config=config,
        model_config=model_config,
        parent_state=parent_state,
        expected_lm_sampling=expected_lm_sampling,
        data_identity=data_identity,
        optimizer_names=optimizer_names,
        expected_execution=production_execution,
        expected_amp_dtype=production_amp_dtype,
    )
    materialized_tokenizer = _mapping(
        materialization["tokenizer_lineage"],
        label="recomputed tokenizer identity",
    )
    if materialized_tokenizer.get("sha256") != tokenizer_sha256:
        raise ValueError("recomputed tokenizer identity differs from the stage budget")
    expected_training_contract = _mapping(
        materialization["training_contract"],
        label="recomputed complete SFT training contract",
    )

    preflight_evidence = _validate_preflight(
        preflight,
        config=config,
        config_artifact=config_artifact,
        model_config=model_config,
        model_config_artifact=model_artifact,
        parent_artifact=parent_artifact,
        parent_pins=parent_pins,
        parent_binding=parent_binding,
        parent_state=parent_state,
        data_identity=data_identity,
        expected_lm_sampling=expected_lm_sampling,
        expected_sampling_evidence=sampling_evidence,
        expected_training_contract=expected_training_contract,
        expected_data_metadata=materialization["data_metadata"],
        expected_tokenizer_sha256=tokenizer_sha256,
        source_inputs=source_input_records,
        model_parameter_names=model_parameter_names,
        model_parameter_specs=model_parameter_specs,
        optimizer_names=optimizer_names,
    )
    tracked.extend(preflight_evidence["tracked"])

    tracked.append(
        (
            latest_path,
            "final enriched SFT checkpoint",
            _MAX_CHECKPOINT_BYTES,
            latest_state,
            str(latest_artifact["sha256"]),
        )
    )
    expected_lineage = _validate_lineage(
        latest_checkpoint.get("lineage"),
        config_sha256=config_sha256,
        model_config_sha256=model_config_sha256,
        data_sha256=data_sha256,
        tokenizer_sha256=tokenizer_sha256,
        parent_checkpoint_sha256=str(parent_artifact["sha256"]),
    )
    training_contract = _validate_training_contract(
        latest_checkpoint.get("training_contract"),
        expected_contract=expected_training_contract,
    )
    expected_cfg = model_config.__dict__
    dataset_accounting = plan_data.get("dataset_token_accounting")
    if not isinstance(dataset_accounting, Mapping):
        raise TypeError("stage budget dataset accounting must be a mapping")
    heldout_accounting = _mapping(
        plan_data.get("heldout_eval_token_accounting"),
        label="stage budget held-out accounting",
    )
    heldout_rows = _strict_int(
        heldout_accounting.get("rows"),
        label="stage budget held-out rows",
        minimum=1,
    )
    heldout_loss_tokens = _strict_int(
        heldout_accounting.get("loss_tokens"),
        label="stage budget held-out loss tokens",
        minimum=1,
    )
    latest_baseline = _mapping(
        latest_checkpoint.get("heldout_baseline"),
        label="latest checkpoint held-out baseline",
    )
    heldout_contract = _validate_heldout_contract(
        latest_baseline.get("contract"),
        config=config,
        model_config=model_config,
        plan_data=plan_data,
    )
    _require_exact(
        heldout_contract,
        materialization["heldout_contract"],
        label="recomputed held-out evaluation contract",
    )
    heldout_pre = _validate_metric_record(
        latest_baseline.get("pre"),
        label="latest checkpoint held-out pre baseline",
        expected_rows=heldout_rows,
        expected_loss_tokens=heldout_loss_tokens,
    )
    latest_record = _checkpoint_record(
        latest_checkpoint,
        latest_artifact,
        completed_steps=PRODUCTION_TOTAL_STEPS,
        expected_training_contract=training_contract,
        expected_lm_sampling=expected_lm_sampling,
        expected_lineage=expected_lineage,
        expected_cfg=expected_cfg,
        expected_dataset_accounting=dataset_accounting,
        expected_token_accounting=final_accounting,
        expected_heldout_contract=heldout_contract,
        expected_heldout_pre=heldout_pre,
        expected_prompt_contract=str(config["data"]["conversation_prompt_contract"]),
        expected_tokenizer_metadata=materialization["tokenizer_metadata"],
        expected_data_metadata=materialization["data_metadata"],
        expected_execution=materialization["execution"],
        expected_frozen_hashes=frozen_hashes,
        parent_state=parent_state,
        model_parameter_names=model_parameter_names,
        model_parameter_specs=model_parameter_specs,
        optimizer_names=optimizer_names,
        archive=False,
    )

    updates = plan["planned"]["updates"]
    archive_records: list[dict[str, Any]] = []
    archive_hashes: set[str] = set()
    final_archive_checkpoint: dict[str, Any] | None = None
    for completed_steps, archive_name in zip(archive_steps, archive_names, strict=True):
        archive_path = run_directory / archive_name
        checkpoint, artifact, state_record = _load_stable_checkpoint(
            archive_path,
            label=f"SFT archive at {completed_steps} completed steps",
        )
        tracked.append(
            (
                archive_path,
                f"SFT archive at {completed_steps} completed steps",
                _MAX_CHECKPOINT_BYTES,
                state_record,
                str(artifact["sha256"]),
            )
        )
        if artifact["sha256"] in archive_hashes:
            raise ValueError("SFT archive file hashes must be unique")
        archive_hashes.add(str(artifact["sha256"]))
        prefix_accounting = _budget_prefix_accounting(updates, completed_steps)
        record = _checkpoint_record(
            checkpoint,
            artifact,
            completed_steps=completed_steps,
            expected_training_contract=training_contract,
            expected_lm_sampling=expected_lm_sampling,
            expected_lineage=expected_lineage,
            expected_cfg=expected_cfg,
            expected_dataset_accounting=dataset_accounting,
            expected_token_accounting=prefix_accounting,
            expected_heldout_contract=heldout_contract,
            expected_heldout_pre=heldout_pre,
            expected_prompt_contract=str(config["data"]["conversation_prompt_contract"]),
            expected_tokenizer_metadata=materialization["tokenizer_metadata"],
            expected_data_metadata=materialization["data_metadata"],
            expected_execution=materialization["execution"],
            expected_frozen_hashes=frozen_hashes,
            parent_state=parent_state,
            model_parameter_names=model_parameter_names,
            model_parameter_specs=model_parameter_specs,
            optimizer_names=optimizer_names,
            archive=True,
        )
        archive_records.append(record)
        if completed_steps == PRODUCTION_TOTAL_STEPS:
            final_archive_checkpoint = checkpoint
    if final_archive_checkpoint is None:
        raise RuntimeError("final SFT archive was not loaded")
    for field in _SFT_RESUME_SEALED_FIELDS:
        if not _exact_equal(latest_checkpoint.get(field), final_archive_checkpoint.get(field)):
            raise ValueError(
                f"latest checkpoint sealed field {field!r} differs from the final archive"
            )
    if latest_checkpoint.get("resume_integrity_sha256") != final_archive_checkpoint.get(
        "resume_integrity_sha256"
    ):
        raise ValueError("latest/final archive resume seals differ")
    if _sealed_resume_sha256(latest_checkpoint) != _sealed_resume_sha256(
        final_archive_checkpoint
    ):
        raise ValueError("latest/final archive sealed resume content differs")

    metrics_path = run_directory / "metrics.json"
    metrics_payload, metrics_artifact, metrics_state = _read_stable_regular(
        metrics_path,
        label="final SFT metrics",
        max_bytes=_MAX_JSON_BYTES,
    )
    tracked.append(
        (
            metrics_path,
            "final SFT metrics",
            _MAX_JSON_BYTES,
            metrics_state,
            str(metrics_artifact["sha256"]),
        )
    )
    metrics = _strict_json(metrics_payload, label="final SFT metrics")
    _validate_metrics(
        metrics,
        latest_path=latest_path,
        latest_checkpoint=latest_checkpoint,
        expected_token_accounting=final_accounting,
        expected_dataset_accounting=dataset_accounting,
        expected_lm_sampling=expected_lm_sampling,
        expected_lineage=expected_lineage,
        continuation=validated_config["continuation"],
        heldout_contract=heldout_contract,
        heldout_pre=heldout_pre,
        heldout_rows=heldout_rows,
        heldout_loss_tokens=heldout_loss_tokens,
    )

    final_directory_state, final_directory_names = _snapshot_exact_directory(
        run_directory,
        expected_names=expected_names,
    )
    if directory_state != final_directory_state or directory_names != final_directory_names:
        raise RuntimeError("production SFT output directory changed during verification")
    for path, label, max_bytes, state_record, digest in tracked:
        _rehash_stable(
            path,
            label=label,
            max_bytes=max_bytes,
            expected_state=state_record,
            expected_sha256=digest,
        )
    last_directory_state, last_directory_names = _snapshot_exact_directory(
        run_directory,
        expected_names=expected_names,
    )
    if directory_state != last_directory_state or directory_names != last_directory_names:
        raise RuntimeError("production SFT output directory changed during artifact re-hash")

    plan_artifact_record = {
        **plan_artifact,
        "plan_self_sha256": plan["plan_self_sha256"],
    }
    preflight_artifact_record = {
        **preflight_artifact,
        "receipt_self_sha256": preflight["receipt_self_sha256"],
        "status": "passed",
    }
    parent_artifact_record = {
        **parent_artifact,
        "resume_integrity_sha256": parent_checkpoint["resume_integrity_sha256"],
    }
    live_evidence_entries = [
        {
            "label": label,
            "path": str(path),
            "bytes": state_record.size,
            "sha256": digest,
        }
        for path, label, _max_bytes, state_record, digest in tracked
    ]
    live_evidence_inventory = {
        "count": len(live_evidence_entries),
        "entries": live_evidence_entries,
        "sha256": canonical_sha256(live_evidence_entries),
    }
    receipt_without_hash = {
        "kind": RECEIPT_KIND,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "scope": copy.deepcopy(_SCOPE),
        "external_roots": copy.deepcopy(roots),
        "contract": {
            "stage": "sft",
            "total_steps": PRODUCTION_TOTAL_STEPS,
            "checkpoint_every": PRODUCTION_CHECKPOINT_EVERY,
            "archive_count": PRODUCTION_ARCHIVE_COUNT,
            "grad_accum_steps": PRODUCTION_GRAD_ACCUM_STEPS,
            "decisions_per_update": PRODUCTION_DECISIONS_PER_UPDATE,
            "expected_archive_completed_steps": archive_steps,
            "final_checkpoint_step": PRODUCTION_TOTAL_STEPS - 1,
            "final_completed_microbatches": (
                PRODUCTION_TOTAL_STEPS * PRODUCTION_GRAD_ACCUM_STEPS
            ),
            "final_lm_cursor": (
                PRODUCTION_TOTAL_STEPS * PRODUCTION_DECISIONS_PER_UPDATE
            ),
            "model_parameter_names": model_parameter_names,
            "frozen_model_parameter_names": list(PRODUCTION_FROZEN_PARAMETERS),
            "optimizer_model_parameter_names": optimizer_names,
            "frozen_tensor_sha256": frozen_hashes,
            "training_contract_sha256": canonical_sha256(training_contract),
            "lm_sampling_sha256": canonical_sha256(expected_lm_sampling),
            "data_sha256": data_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "prepared_data_sha256": training_contract["prepared_data_sha256"],
            "parent": copy.deepcopy(parent_pins),
            "heldout_pre": {
                "rows": heldout_rows,
                "assistant_loss_tokens": heldout_loss_tokens,
                "contract_sha256": canonical_sha256(heldout_contract),
                "metrics_sha256": canonical_sha256(heldout_pre),
            },
            "structured_heads": {
                "joint_tool_pointer": False,
                "train_route_head": False,
                "train_dense_selector": False,
            },
        },
        "artifacts": {
            "config": {
                **config_artifact,
                "canonical_sha256": canonical_sha256(config),
            },
            "model_config": {
                **model_artifact,
                "canonical_sha256": canonical_sha256(model_config.__dict__),
            },
            "budget_plan": plan_artifact_record,
            "preflight": preflight_artifact_record,
            "preflight_evidence": preflight_evidence["artifacts"],
            "parent_checkpoint": parent_artifact_record,
            "source_inputs": source_input_records,
            "live_evidence_inventory": live_evidence_inventory,
            "run_directory": {
                "path": str(run_directory),
                "entry_count": len(directory_names),
                "expected_entries": directory_names,
            },
            "archives": archive_records,
            "latest_checkpoint": latest_record,
            "metrics": metrics_artifact,
        },
        "validation": {key: True for key in sorted(_VALIDATION_KEYS)},
    }
    receipt = {
        **receipt_without_hash,
        "receipt_self_sha256": canonical_sha256(receipt_without_hash),
    }
    assert_sft_production_receipt(receipt)
    return receipt


def build_sft_production_receipt(
    config_path: str | Path,
    budget_plan_path: str | Path,
    preflight_path: str | Path,
    *,
    expected_roots: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility name for :func:`verify_sft_production_run`."""

    return verify_sft_production_run(
        config_path,
        budget_plan_path,
        preflight_path,
        expected_roots=expected_roots,
    )


def verify_sft_production_receipt_against_artifacts(
    receipt_path: str | Path,
    *,
    expected_receipt_file_sha256: str,
) -> dict[str, Any]:
    """Reopen a canonical receipt and independently replay every artifact binding.

    Receipt-only assertion is deliberately insufficient for checkpoint-sweep authorization.  This
    function requires a real pathname plus an out-of-band file hash, then rebuilds the receipt from
    the still-live artifacts and demands byte-semantic equality.
    """

    expected_file_sha256 = _sha256(
        expected_receipt_file_sha256,
        label="external SFT production receipt file SHA-256",
    )
    path = _safe_absolute_path(
        receipt_path,
        label="SFT production receipt path",
    )
    payload, artifact, state_record = _read_stable_regular(
        path,
        label="SFT production receipt",
        max_bytes=_MAX_JSON_BYTES,
    )
    if artifact["sha256"] != expected_file_sha256:
        raise ValueError("SFT production receipt file differs from its external root")
    receipt = verify_sft_production_receipt_bytes(payload)
    artifacts = _mapping(receipt.get("artifacts"), label="receipt artifacts")
    rebuilt = verify_sft_production_run(
        _mapping(artifacts["config"], label="receipt config artifact")["path"],
        _mapping(artifacts["budget_plan"], label="receipt budget artifact")["path"],
        _mapping(artifacts["preflight"], label="receipt preflight artifact")["path"],
        expected_roots=_mapping(
            receipt.get("external_roots"),
            label="receipt external roots",
        ),
    )
    _require_exact(
        rebuilt,
        receipt,
        label="receipt/live artifact reconstruction",
    )
    _rehash_stable(
        path,
        label="SFT production receipt",
        max_bytes=_MAX_JSON_BYTES,
        expected_state=state_record,
        expected_sha256=expected_file_sha256,
    )
    return receipt


def write_sft_production_receipt(
    path: str | Path,
    receipt: Mapping[str, Any],
) -> None:
    """Reverify live artifacts, then publish canonically without replacing any pathname."""

    assert_sft_production_receipt(receipt)
    artifacts = _mapping(receipt.get("artifacts"), label="receipt artifacts")
    rebuilt = verify_sft_production_run(
        _mapping(artifacts["config"], label="receipt config artifact")["path"],
        _mapping(artifacts["budget_plan"], label="receipt budget artifact")["path"],
        _mapping(artifacts["preflight"], label="receipt preflight artifact")["path"],
        expected_roots=_mapping(
            receipt.get("external_roots"),
            label="receipt external roots",
        ),
    )
    _require_exact(rebuilt, receipt, label="receipt/live artifact reconstruction before publish")
    destination = _safe_absolute_path(
        path,
        label="SFT production receipt output",
        allow_missing_leaf=True,
    )
    run_directory = _safe_absolute_path(
        _mapping(
            artifacts.get("run_directory"),
            label="receipt run directory",
        )["path"],
        label="receipt verified run directory",
    )
    if destination == run_directory or run_directory in destination.parents:
        raise ValueError("receipt output must be outside the verified production run directory")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing SFT production receipt: {destination}")
    payload = _canonical_receipt_bytes(receipt)
    directory_descriptor = _open_directory_chain(
        destination.parent,
        label="SFT production receipt output parent",
    )
    temporary_name = f".{destination.name}.{os.urandom(16).hex()}.tmp"
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(temporary_descriptor, payload[offset:])
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = None
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            if error.errno in {errno.EEXIST, errno.ELOOP}:
                raise FileExistsError(
                    f"refusing to replace existing SFT production receipt: {destination}"
                ) from error
            raise
        os.fsync(directory_descriptor)
        written, _, _ = _read_stable_regular(
            destination,
            label="published SFT production receipt",
            max_bytes=_MAX_JSON_BYTES,
        )
        if written != payload:
            raise RuntimeError("published SFT production receipt bytes differ")
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        os.close(directory_descriptor)


__all__ = [
    "PRODUCTION_ARCHIVE_COUNT",
    "PRODUCTION_CHECKPOINT_EVERY",
    "PRODUCTION_DECISIONS_PER_UPDATE",
    "PRODUCTION_FROZEN_PARAMETERS",
    "PRODUCTION_GRAD_ACCUM_STEPS",
    "PRODUCTION_MICRO_BATCH_SIZE",
    "PRODUCTION_TOTAL_STEPS",
    "PRODUCTION_BUDGET_PLAN_FILE_SHA256",
    "PRODUCTION_CONFIG_CANONICAL_SHA256",
    "PRODUCTION_CONFIG_FILE_SHA256",
    "PRODUCTION_DATA_SHA256",
    "PRODUCTION_MODEL_CONFIG_CANONICAL_SHA256",
    "PRODUCTION_MODEL_CONFIG_FILE_SHA256",
    "PRODUCTION_PARENT_CHECKPOINT_SHA256",
    "PRODUCTION_PREFLIGHT_FILE_SHA256",
    "PRODUCTION_TOKENIZER_SHA256",
    "RECEIPT_KIND",
    "RECEIPT_SCHEMA_VERSION",
    "assert_sft_production_receipt",
    "build_sft_production_receipt",
    "verify_sft_production_receipt_bytes",
    "verify_sft_production_receipt_against_artifacts",
    "verify_sft_production_run",
    "write_sft_production_receipt",
]
