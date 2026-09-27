"""Teacher-forced retention sweep over integrity-sealed SFT checkpoints.

The sweep deliberately reuses the deterministic SFT held-out evaluator.  Its YAML input only
chooses a training config, a closed checkpoint set, an expected parent, exact held-out
cardinalities, a pinned parent baseline, and non-inferiority tolerances. Evaluation data,
selection, prompt rendering, tokenizer, sequence length, batch size, padding width, device, and
dtype are inherited from the training config.
"""

from __future__ import annotations

import copy
import fnmatch
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
    canonical_json_bytes,
    conversation_semantic_sha256,
)
from localagent.data.prompt_contract import (
    LEGACY_CONVERSATION_PROMPT_CONTRACT,
    assert_prompt_contract_tokenizer,
    resolve_conversation_prompt_contract,
)
from localagent.data.schema import Conversation, Role
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import execution_metadata, resolve_device, resolve_dtype
from localagent.train.sft import (
    _evaluate_conversations,
    _load_validated_sft_resume_checkpoint,
    _validated_resume_heldout_baseline,
)
from localagent.train.stage_data import (
    canonical_sha256,
    checkpoint_tokenizer_sha256,
    load_conversation_source,
    tokenizer_identity,
)

CONFIG_KIND = "localagent_sft_checkpoint_sweep_config"
RESULT_KIND = "localagent_sft_checkpoint_sweep_result"
SCHEMA_VERSION = 2

_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
_ARCHIVE_NAME = re.compile(r".+\.step-(\d{8})\.pt")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONFIG_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "training_config",
        "checkpoints",
        "expected_parent_checkpoint_sha256",
        "expected_eval",
        "expected_baseline",
        "thresholds",
    }
)
_EXPECTED_EVAL_KEYS = frozenset(
    {"conversations", "assistant_decisions", "assistant_loss_tokens"}
)
_EXPECTED_BASELINE_KEYS = frozenset({"metrics", "absolute_tolerances"})
_BASELINE_TOLERANCE_KEYS = frozenset(
    {
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "max_mean_loss_increase",
        "max_assistant_token_accuracy_drop",
        "max_assistant_sequence_accuracy_drop",
    }
)
_METRIC_KEYS = frozenset(
    {
        "rows",
        "assistant_loss_tokens",
        "mean_loss",
        "assistant_token_accuracy",
        "assistant_sequence_accuracy",
    }
)


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
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            links=value.st_nlink,
            size=value.st_size,
            modified_ns=value.st_mtime_ns,
            changed_ns=value.st_ctime_ns,
        )


@dataclass(frozen=True)
class TeacherForcedSweepContext:
    """Frozen training/evaluation inputs shared by every checkpoint in one sweep."""

    training_config: Mapping[str, Any]
    training_config_artifact: Mapping[str, Any]
    training_config_sha256: str
    model_config: ModelConfig
    model_config_artifact: Mapping[str, Any]
    model_config_sha256: str
    tokenizer: Any
    tokenizer_record: Mapping[str, Any]
    tokenizer_sha256: str
    conversations: Sequence[Conversation]
    eval_sources: Sequence[Mapping[str, Any]]
    eval_contract: Mapping[str, Any]
    eval_selection: Mapping[str, Any] | None
    overlap_audit: Mapping[str, Any]
    conversation_count: int
    assistant_decisions: int
    assistant_loss_tokens: int
    prompt_contract: str
    max_seq_len: int
    batch_size: int
    pad_to_input_tokens: int | None
    requested_device: str
    requested_dtype: str


def _exact_mapping(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _nonempty_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value)


def _regular_state(path: Path, *, label: str, max_bytes: int | None = None) -> _FileState:
    try:
        observed = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible: {path}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if max_bytes is not None and observed.st_size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
    return _FileState.from_stat(observed)


def _read_regular(path: Path, *, label: str, max_bytes: int) -> tuple[bytes, _FileState]:
    before = _regular_state(path, label=label, max_bytes=max_bytes)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError(f"{label} changed while it was being opened: {path}")
        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - observed_bytes))
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(path, label=label, max_bytes=max_bytes)
    if before != after_descriptor or before != after_path:
        raise RuntimeError(f"{label} changed while it was being read: {path}")
    payload = b"".join(chunks)
    if len(payload) != before.size:
        raise RuntimeError(f"{label} returned an incomplete read: {path}")
    return payload, before


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
        raise ValueError(f"{label} must contain finite JSON values") from error
    return dict(value)


def _file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _normalized_training_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _model_config_from_mapping(value: Mapping[str, Any]) -> ModelConfig:
    keys = set(ModelConfig.__dataclass_fields__)
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(
            f"model config keys mismatch: missing={missing}, extra={extra}"
        )
    config = ModelConfig(**dict(value))
    config.assert_within_budget()
    return config


def _source_specs(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, Path, Mapping)):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty source or source list")
    return list(value)


def _optional_source_specs(value: Any, *, label: str) -> list[Any]:
    if value is None or value == []:
        return []
    return _source_specs(value, label=label)


def _assistant_decision_count(conversations: Sequence[Conversation]) -> int:
    return sum(
        message.role == Role.assistant
        for conversation in conversations
        for message in conversation.messages
    )


def load_sweep_context(
    training_config_path: str | Path,
    *,
    expected_conversations: int,
    expected_assistant_decisions: int,
    expected_assistant_loss_tokens: int,
) -> TeacherForcedSweepContext:
    """Load and freeze the verified held-out contract inherited from one SFT config."""

    expected_conversations = _positive_int(
        expected_conversations,
        label="expected_eval.conversations",
    )
    expected_assistant_decisions = _positive_int(
        expected_assistant_decisions,
        label="expected_eval.assistant_decisions",
    )
    expected_assistant_loss_tokens = _positive_int(
        expected_assistant_loss_tokens,
        label="expected_eval.assistant_loss_tokens",
    )
    training_path = Path(training_config_path)
    training_payload, _ = _read_regular(
        training_path,
        label="training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_config = _yaml_mapping(training_payload, label="training config")
    if training_config.get("stage", "sft") != "sft":
        raise ValueError("checkpoint sweep training config must declare stage='sft'")
    training_config_sha256 = _normalized_training_config_sha256(training_config)

    model_path = _nonempty_path(
        training_config.get("model_config"),
        label="training config model_config",
    )
    model_payload, _ = _read_regular(
        model_path,
        label="model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    model_config = _model_config_from_mapping(
        _yaml_mapping(model_payload, label="model config")
    )
    model_config_sha256 = canonical_sha256(model_config.__dict__)

    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("training config data must be a mapping")
    if data.get("strict_conversation_artifacts") is not True:
        raise ValueError(
            "checkpoint sweep requires data.strict_conversation_artifacts=true"
        )
    prompt_contract = resolve_conversation_prompt_contract(
        data.get("conversation_prompt_contract")
    )
    tokenizer_config = data.get("tokenizer", {"kind": "byte"})
    if not isinstance(tokenizer_config, Mapping):
        raise TypeError("training config data.tokenizer must be a mapping")
    extra_tokenizer_keys = sorted(set(tokenizer_config) - {"kind", "path"})
    if extra_tokenizer_keys:
        raise ValueError(
            f"training tokenizer has unsupported fields: {extra_tokenizer_keys}"
        )
    tokenizer_kind = tokenizer_config.get("kind", "byte")
    if not isinstance(tokenizer_kind, str) or not tokenizer_kind:
        raise ValueError("training tokenizer kind must be non-empty text")
    tokenizer_path = tokenizer_config.get("path")
    if tokenizer_path is not None and (
        not isinstance(tokenizer_path, str) or not tokenizer_path
    ):
        raise ValueError("training tokenizer path must be non-empty text")
    tokenizer = load_tokenizer(tokenizer_kind, tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("training tokenizer vocabulary does not match model config")
    assert_prompt_contract_tokenizer(tokenizer, prompt_contract)
    tokenizer_lineage = tokenizer_identity(
        tokenizer_kind,
        vocab_size=tokenizer.vocab_size,
        path=tokenizer_path,
    )
    tokenizer_record = {
        "kind": tokenizer_kind,
        "vocab_size": tokenizer.vocab_size,
        "sha256": tokenizer_lineage["sha256"],
        **(
            {
                "artifact": {
                    "path": str(tokenizer_path),
                    **dict(tokenizer_lineage["artifact"]),
                }
            }
            if "artifact" in tokenizer_lineage
            else {}
        ),
    }

    loaded_train_sources = [
        load_conversation_source(
            source,
            require_verified=True,
            expected_split="train",
        )
        for source in _source_specs(
            data.get("conversations"),
            label="training config data.conversations",
        )
    ]
    loaded_decay_sources = [
        load_conversation_source(
            source,
            require_verified=True,
            expected_split="train",
        )
        for source in _optional_source_specs(
            data.get("decay_conversations"),
            label="training config data.decay_conversations",
        )
    ]
    loaded_sources = [
        load_conversation_source(
            source,
            require_verified=True,
            expected_split="eval",
        )
        for source in _source_specs(
            data.get("eval_conversations"),
            label="training config data.eval_conversations",
        )
    ]
    full_eval_conversations: Sequence[Conversation] = tuple(
        conversation
        for source in loaded_sources
        for conversation in source.conversations
    )
    overlap_audit = assert_no_conversation_overlap(
        tuple(
            conversation
            for source in (*loaded_train_sources, *loaded_decay_sources)
            for conversation in source.conversations
        ),
        full_eval_conversations,
        left_label="SFT sweep reconstructed train/decay content",
        right_label="held-out",
        conversation_prompt_contract=prompt_contract,
    ).as_dict()
    conversations = full_eval_conversations
    evaluation = training_config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("training config evaluation must be a mapping")
    max_conversations = evaluation.get("max_conversations")
    selection_mode = evaluation.get("selection")
    selection_audit: Mapping[str, Any] | None = None
    if max_conversations is None:
        if selection_mode is not None:
            raise ValueError(
                "training evaluation.selection requires max_conversations"
            )
    else:
        max_conversations = _positive_int(
            max_conversations,
            label="training evaluation.max_conversations",
        )
        if selection_mode != STRATIFIED_EVAL_ALGORITHM:
            raise ValueError(
                "training evaluation.selection must be "
                f"{STRATIFIED_EVAL_ALGORITHM!r}"
            )
        selection = select_stratified_eval_subset(
            conversations,
            max_rows=max_conversations,
        )
        conversations = tuple(selection.conversations)
        selection_audit = selection.audit.as_dict()

    conversation_count = len(conversations)
    assistant_decisions = _assistant_decision_count(conversations)
    if conversation_count != expected_conversations:
        raise ValueError(
            "verified held-out conversation count mismatch: "
            f"expected={expected_conversations}, observed={conversation_count}"
        )
    if assistant_decisions != expected_assistant_decisions:
        raise ValueError(
            "verified held-out assistant-decision count mismatch: "
            f"expected={expected_assistant_decisions}, observed={assistant_decisions}"
        )
    if selection_audit is not None and (
        selection_audit["selected"]["rows"] != conversation_count
        or selection_audit["selected"]["assistant_decisions"] != assistant_decisions
    ):
        raise RuntimeError("held-out selection audit cardinalities are inconsistent")

    seq_len = data.get("seq_len", model_config.max_seq_len)
    seq_len = _positive_int(seq_len, label="training data.seq_len")
    max_seq_len = min(seq_len, model_config.max_seq_len)
    batch_size = _positive_int(
        evaluation.get("batch_size", 8),
        label="training evaluation.batch_size",
    )
    pad_to_input_tokens = evaluation.get("pad_to_input_tokens")
    if pad_to_input_tokens is not None:
        pad_to_input_tokens = _positive_int(
            pad_to_input_tokens,
            label="training evaluation.pad_to_input_tokens",
        )
        if pad_to_input_tokens > max_seq_len:
            raise ValueError(
                "training evaluation.pad_to_input_tokens exceeds sequence limit"
            )
    semantic_rows = [
        conversation_semantic_sha256(conversation) for conversation in conversations
    ]
    eval_contract = {
        "kind": "deterministic_teacher_forced_assistant_tokens",
        "row_order": (
            "configured_jsonl_order"
            if prompt_contract == LEGACY_CONVERSATION_PROMPT_CONTRACT
            else "configured_jsonl_assistant_decision_order"
        ),
        "same_rows_pre_post": True,
        "max_seq_len": max_seq_len,
        **(
            {"pad_to_input_tokens": pad_to_input_tokens}
            if pad_to_input_tokens is not None
            else {}
        ),
        "dataset_sha256": canonical_sha256(sorted(semantic_rows)),
        **({"selection": selection_audit} if selection_audit is not None else {}),
        **(
            {"conversation_prompt_contract": prompt_contract}
            if prompt_contract != LEGACY_CONVERSATION_PROMPT_CONTRACT
            else {}
        ),
    }
    runtime = training_config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise TypeError("training config runtime must be a mapping")
    requested_device = runtime.get("device", "auto")
    requested_dtype = runtime.get("dtype", "auto")
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("training runtime.device must be non-empty text")
    if not isinstance(requested_dtype, str) or not requested_dtype:
        raise ValueError("training runtime.dtype must be non-empty text")

    return TeacherForcedSweepContext(
        training_config=training_config,
        training_config_artifact=_file_record(training_path, training_payload),
        training_config_sha256=training_config_sha256,
        model_config=model_config,
        model_config_artifact=_file_record(model_path, model_payload),
        model_config_sha256=model_config_sha256,
        tokenizer=tokenizer,
        tokenizer_record=tokenizer_record,
        tokenizer_sha256=str(tokenizer_lineage["sha256"]),
        conversations=conversations,
        eval_sources=tuple(
            {"path": str(source.path), **copy.deepcopy(dict(source.identity))}
            for source in loaded_sources
        ),
        eval_contract=eval_contract,
        eval_selection=selection_audit,
        overlap_audit=overlap_audit,
        conversation_count=conversation_count,
        assistant_decisions=assistant_decisions,
        assistant_loss_tokens=expected_assistant_loss_tokens,
        prompt_contract=prompt_contract,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        pad_to_input_tokens=pad_to_input_tokens,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
    )


def _checkpoint_paths(config: Any) -> tuple[list[Path], dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("checkpoints must be a mapping")
    if set(config) == {"paths"}:
        raw_paths = config["paths"]
        if not isinstance(raw_paths, list) or not raw_paths:
            raise ValueError("checkpoints.paths must be a non-empty list")
        paths = [
            _nonempty_path(value, label=f"checkpoints.paths[{index}]")
            for index, value in enumerate(raw_paths)
        ]
        if len({str(path) for path in paths}) != len(paths):
            raise ValueError("checkpoints.paths contains duplicate path strings")
        discovery = {
            "mode": "explicit_paths",
            "configured_paths": [str(path) for path in paths],
        }
        _require_archive_paths(paths)
        return paths, discovery
    if set(config) == {"directory", "pattern"}:
        directory = _nonempty_path(config["directory"], label="checkpoints.directory")
        pattern = config["pattern"]
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("checkpoints.pattern must be non-empty text")
        if (
            pattern in {".", ".."}
            or "/" in pattern
            or "\\" in pattern
            or "**" in pattern
            or Path(pattern).is_absolute()
        ):
            raise ValueError(
                "checkpoints.pattern must be a basename-only, non-recursive pattern"
            )
        try:
            directory_state = directory.lstat()
        except OSError as error:
            raise ValueError(f"checkpoints.directory is missing: {directory}") from error
        if stat.S_ISLNK(directory_state.st_mode) or not stat.S_ISDIR(directory_state.st_mode):
            raise ValueError("checkpoints.directory must be a non-symlink directory")
        paths = sorted(
            (
                child
                for child in directory.iterdir()
                if fnmatch.fnmatchcase(child.name, pattern)
            ),
            key=lambda path: path.name,
        )
        if not paths:
            raise ValueError("checkpoint directory/pattern matched no files")
        _require_archive_paths(paths)
        return paths, {
            "mode": "safe_directory_pattern",
            "directory": str(directory),
            "pattern": pattern,
            "matched_paths": [str(path) for path in paths],
        }
    raise ValueError(
        "checkpoints must contain exactly paths or exactly directory and pattern"
    )


def _require_archive_paths(paths: Sequence[Path]) -> None:
    invalid = [str(path) for path in paths if _ARCHIVE_NAME.fullmatch(path.name) is None]
    if invalid:
        raise ValueError(
            "checkpoint sweep requires immutable archive-style filenames "
            "'<name>.step-<8 digits>.pt': "
            + json.dumps(invalid, separators=(",", ":"))
        )


def _hash_open_file(handle, *, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"checkpoint exceeds {max_bytes} bytes")
        digest.update(chunk)
    return digest.hexdigest(), size


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], _FileState]:
    before = _regular_state(
        path,
        label="checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = _FileState.from_stat(os.fstat(descriptor))
        if opened != before:
            raise RuntimeError(f"checkpoint changed while it was being opened: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first_sha256, first_size = _hash_open_file(
                handle,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            handle.seek(0)
            second_sha256, second_size = _hash_open_file(
                handle,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
            after_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    after_path = _regular_state(
        path,
        label="checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if (
        before != after_descriptor
        or before != after_path
        or first_sha256 != second_sha256
        or first_size != before.size
        or second_size != before.size
    ):
        raise RuntimeError(f"checkpoint changed while it was being loaded: {path}")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload must be a mapping")
    return (
        checkpoint,
        {"path": str(path), "bytes": before.size, "sha256": first_sha256},
        before,
    )


def _assert_checkpoint_unchanged(
    path: Path,
    *,
    expected_state: _FileState,
    expected_sha256: str,
) -> None:
    observed_state = _regular_state(
        path,
        label="checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if observed_state != expected_state:
        raise RuntimeError(f"checkpoint changed during evaluation: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            observed_sha256, observed_size = _hash_open_file(
                handle,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
        final_descriptor = _FileState.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    final_path = _regular_state(
        path,
        label="checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    )
    if (
        observed_sha256 != expected_sha256
        or observed_size != expected_state.size
        or final_descriptor != expected_state
        or final_path != expected_state
    ):
        raise RuntimeError(f"checkpoint changed during evaluation: {path}")


def _checkpoint_model_config(checkpoint: Mapping[str, Any]) -> ModelConfig:
    value = checkpoint.get("cfg")
    if not isinstance(value, Mapping):
        value = getattr(value, "__dict__", None)
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint cfg must be a mapping or dataclass")
    return _model_config_from_mapping(value)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_lineage(
    checkpoint: Mapping[str, Any],
    context: TeacherForcedSweepContext,
    *,
    expected_parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("checkpoint has no lineage mapping")
    if lineage.get("version") != 1 or lineage.get("stage") != "sft":
        raise ValueError("checkpoint lineage must declare SFT lineage version 1")
    expected = {
        "config_sha256": context.training_config_sha256,
        "model_config_sha256": context.model_config_sha256,
        "tokenizer_sha256": context.tokenizer_sha256,
    }
    for key, expected_value in expected.items():
        observed = _require_sha256(
            lineage.get(key),
            label=f"checkpoint lineage.{key}",
        )
        if observed != expected_value:
            raise ValueError(
                f"checkpoint lineage.{key} does not match the training config"
            )
    _require_sha256(
        lineage.get("data_sha256"),
        label="checkpoint lineage.data_sha256",
    )
    git = lineage.get("git")
    if not isinstance(git, Mapping):
        raise TypeError("checkpoint lineage.git must be a mapping")
    for key in ("repository_sha256", "worktree_sha256"):
        _require_sha256(git.get(key), label=f"checkpoint lineage.git.{key}")
    commit = git.get("commit")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ValueError("checkpoint lineage.git.commit must be a lowercase Git SHA-1")
    if not isinstance(git.get("dirty"), bool):
        raise TypeError("checkpoint lineage.git.dirty must be boolean")
    parent_checkpoint_sha256 = _require_sha256(
        lineage.get("parent_checkpoint_sha256"),
        label="checkpoint lineage.parent_checkpoint_sha256",
    )
    if parent_checkpoint_sha256 != expected_parent_checkpoint_sha256:
        raise ValueError(
            "checkpoint lineage.parent_checkpoint_sha256 does not match the configured parent"
        )
    try:
        json.dumps(lineage, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint lineage must contain finite JSON values") from error
    return copy.deepcopy(dict(lineage))


def _checkpoint_step(checkpoint: Mapping[str, Any], *, path: Path) -> tuple[int, int]:
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    completed_steps = step + 1
    training_contract = checkpoint.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise TypeError("checkpoint training_contract must be a mapping")
    planned_steps = _positive_int(
        training_contract.get("steps"),
        label="checkpoint training_contract.steps",
    )
    if completed_steps > planned_steps:
        raise ValueError("checkpoint step exceeds its fixed SFT horizon")
    sampling_state = checkpoint.get("sampling_state")
    if not isinstance(sampling_state, Mapping):
        raise TypeError("checkpoint sampling_state must be a mapping")
    if sampling_state.get("completed_steps") != completed_steps:
        raise ValueError("checkpoint completed-step accounting is inconsistent")
    accum_steps = _positive_int(
        training_contract.get("accum_steps"),
        label="checkpoint training_contract.accum_steps",
    )
    if sampling_state.get("completed_microbatches") != completed_steps * accum_steps:
        raise ValueError("checkpoint completed-microbatch accounting is inconsistent")
    name_match = _ARCHIVE_NAME.fullmatch(path.name)
    if name_match is None:
        raise ValueError("checkpoint path does not use the required immutable archive filename")
    if int(name_match.group(1)) != completed_steps:
        raise ValueError(
            "checkpoint archive filename step disagrees with its sealed payload"
        )
    return completed_steps, planned_steps


def _metric_mapping(value: Any, *, label: str) -> dict[str, Any]:
    metrics = _exact_mapping(value, _METRIC_KEYS, label=label)
    for key in ("rows", "assistant_loss_tokens"):
        _positive_int(metrics[key], label=f"{label}.{key}")
    mean_loss = _nonnegative_finite(metrics["mean_loss"], label=f"{label}.mean_loss")
    result = {**metrics, "mean_loss": mean_loss}
    for key in ("assistant_token_accuracy", "assistant_sequence_accuracy"):
        number = _nonnegative_finite(metrics[key], label=f"{label}.{key}")
        if number > 1.0:
            raise ValueError(f"{label}.{key} must be in [0, 1]")
        result[key] = number
    return result


def _expected_baseline(
    value: Any,
    *,
    expected_rows: int,
    expected_loss_tokens: int,
) -> dict[str, Any]:
    contract = _exact_mapping(value, _EXPECTED_BASELINE_KEYS, label="expected_baseline")
    metrics = _metric_mapping(contract["metrics"], label="expected_baseline.metrics")
    if metrics["rows"] != expected_rows:
        raise ValueError("expected_baseline.metrics.rows does not match expected_eval")
    if metrics["assistant_loss_tokens"] != expected_loss_tokens:
        raise ValueError(
            "expected_baseline.metrics.assistant_loss_tokens does not match expected_eval"
        )
    raw_tolerances = _exact_mapping(
        contract["absolute_tolerances"],
        _BASELINE_TOLERANCE_KEYS,
        label="expected_baseline.absolute_tolerances",
    )
    tolerances = {
        key: _nonnegative_finite(
            raw_tolerances[key],
            label=f"expected_baseline.absolute_tolerances.{key}",
        )
        for key in sorted(_BASELINE_TOLERANCE_KEYS)
    }
    return {"metrics": metrics, "absolute_tolerances": tolerances}


def _assert_expected_baseline(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    expected_metrics = expected["metrics"]
    if observed["rows"] != expected_metrics["rows"]:
        raise ValueError("checkpoint heldout baseline rows do not match the configured baseline")
    if observed["assistant_loss_tokens"] != expected_metrics["assistant_loss_tokens"]:
        raise ValueError(
            "checkpoint heldout baseline assistant loss tokens do not match "
            "the configured baseline"
        )
    for key, tolerance in expected["absolute_tolerances"].items():
        difference = abs(float(observed[key]) - float(expected_metrics[key]))
        if difference > float(tolerance):
            raise ValueError(
                f"checkpoint heldout baseline {key} differs from the configured baseline: "
                f"absolute_difference={difference}, tolerance={tolerance}"
            )


def _validate_checkpoint_overlap(
    checkpoint: Mapping[str, Any],
    *,
    expected_audit: Mapping[str, Any],
) -> dict[str, Any]:
    data = checkpoint.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("checkpoint data metadata must be a mapping")
    for key in ("heldout_content_overlap", "heldout_rendered_prompt_overlap"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"checkpoint data.{key} must be exactly zero")
    recorded_audit = data.get("conversation_overlap_audit")
    if not isinstance(recorded_audit, Mapping):
        raise TypeError("checkpoint data.conversation_overlap_audit must be a mapping")
    if dict(recorded_audit) != dict(expected_audit):
        raise ValueError(
            "checkpoint sealed conversation overlap audit does not match the reconstructed audit"
        )
    return {
        "heldout_content_overlap": 0,
        "heldout_rendered_prompt_overlap": 0,
        "conversation_overlap_audit": copy.deepcopy(dict(recorded_audit)),
    }


def _retention_result(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, Any], bool]:
    delta = {
        "mean_loss": float(metrics["mean_loss"]) - float(baseline["mean_loss"]),
        "assistant_token_accuracy": (
            float(metrics["assistant_token_accuracy"])
            - float(baseline["assistant_token_accuracy"])
        ),
        "assistant_sequence_accuracy": (
            float(metrics["assistant_sequence_accuracy"])
            - float(baseline["assistant_sequence_accuracy"])
        ),
    }
    gates = {
        "mean_loss_non_inferiority": {
            "observed_increase": delta["mean_loss"],
            "maximum_increase": thresholds["max_mean_loss_increase"],
            "passed": (
                delta["mean_loss"] <= thresholds["max_mean_loss_increase"]
            ),
        },
        "assistant_token_accuracy_non_inferiority": {
            "observed_drop": -delta["assistant_token_accuracy"],
            "maximum_drop": thresholds["max_assistant_token_accuracy_drop"],
            "passed": (
                -delta["assistant_token_accuracy"]
                <= thresholds["max_assistant_token_accuracy_drop"]
            ),
        },
        "assistant_sequence_accuracy_non_inferiority": {
            "observed_drop": -delta["assistant_sequence_accuracy"],
            "maximum_drop": thresholds[
                "max_assistant_sequence_accuracy_drop"
            ],
            "passed": (
                -delta["assistant_sequence_accuracy"]
                <= thresholds["max_assistant_sequence_accuracy_drop"]
            ),
        },
    }
    eligible = all(bool(gate["passed"]) for gate in gates.values())
    return delta, gates, eligible


def _best_checkpoint(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
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


def run_sft_checkpoint_sweep(config_path: str | Path) -> dict[str, Any]:
    """Evaluate one closed set of sealed checkpoints and return a self-hashed result."""

    config_source = Path(config_path)
    config_payload, _ = _read_regular(
        config_source,
        label="checkpoint sweep config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    config = _exact_mapping(
        _yaml_mapping(config_payload, label="checkpoint sweep config"),
        _CONFIG_KEYS,
        label="checkpoint sweep config",
    )
    schema_version = config.get("schema_version")
    if (
        config.get("kind") != CONFIG_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(
            f"checkpoint sweep config must be {CONFIG_KIND!r} schema_version {SCHEMA_VERSION}"
        )
    expected_eval = _exact_mapping(
        config["expected_eval"],
        _EXPECTED_EVAL_KEYS,
        label="expected_eval",
    )
    expected_eval = {
        key: _positive_int(value, label=f"expected_eval.{key}")
        for key, value in expected_eval.items()
    }
    expected_parent_checkpoint_sha256 = _require_sha256(
        config["expected_parent_checkpoint_sha256"],
        label="expected_parent_checkpoint_sha256",
    )
    expected_baseline = _expected_baseline(
        config["expected_baseline"],
        expected_rows=expected_eval["assistant_decisions"],
        expected_loss_tokens=expected_eval["assistant_loss_tokens"],
    )
    thresholds_raw = _exact_mapping(
        config["thresholds"],
        _THRESHOLD_KEYS,
        label="thresholds",
    )
    thresholds = {
        key: _nonnegative_finite(value, label=f"thresholds.{key}")
        for key, value in thresholds_raw.items()
    }
    training_config_path = _nonempty_path(
        config["training_config"],
        label="training_config",
    )
    context = load_sweep_context(
        training_config_path,
        expected_conversations=expected_eval["conversations"],
        expected_assistant_decisions=expected_eval["assistant_decisions"],
        expected_assistant_loss_tokens=expected_eval["assistant_loss_tokens"],
    )
    paths, discovery = _checkpoint_paths(config["checkpoints"])
    device = resolve_device(context.requested_device)
    dtype = resolve_dtype(device, context.requested_dtype)
    execution = execution_metadata(
        requested_device=context.requested_device,
        resolved_device=device,
        requested_dtype=context.requested_dtype,
        resolved_dtype=dtype,
    )

    records: list[dict[str, Any]] = []
    common_lineage: dict[str, Any] | None = None
    common_training_contract: dict[str, Any] | None = None
    common_baseline: dict[str, Any] | None = None
    common_overlap: dict[str, Any] | None = None
    observed_steps: set[int] = set()
    observed_file_identities: set[tuple[int, int]] = set()
    for path in paths:
        checkpoint, artifact, file_state = _load_checkpoint(path)
        file_identity = (file_state.device, file_state.inode)
        if file_identity in observed_file_identities:
            raise ValueError("checkpoint set aliases the same file more than once")
        observed_file_identities.add(file_identity)
        checkpoint = dict(_load_validated_sft_resume_checkpoint(checkpoint))
        if checkpoint.get("stage") != "sft":
            raise ValueError("checkpoint stage must be 'sft'")
        if checkpoint.get("conversation_prompt_contract") != context.prompt_contract:
            raise ValueError("checkpoint prompt contract does not match training config")
        checkpoint_config = _checkpoint_model_config(checkpoint)
        if checkpoint_config.__dict__ != context.model_config.__dict__:
            raise ValueError("checkpoint architecture does not match training config")
        if checkpoint_tokenizer_sha256(checkpoint) != context.tokenizer_sha256:
            raise ValueError("checkpoint tokenizer does not match training config")
        tokenizer_metadata = checkpoint.get("tokenizer")
        if not isinstance(tokenizer_metadata, Mapping):
            raise TypeError("checkpoint tokenizer metadata must be a mapping")
        if tokenizer_metadata.get("kind") != context.tokenizer_record["kind"]:
            raise ValueError("checkpoint tokenizer kind does not match training config")
        lineage = _validate_lineage(
            checkpoint,
            context,
            expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        )
        if common_lineage is None:
            common_lineage = lineage
        elif lineage != common_lineage:
            raise ValueError("checkpoint set has divergent training lineage")
        training_contract = checkpoint.get("training_contract")
        if not isinstance(training_contract, Mapping):
            raise TypeError("checkpoint training_contract must be a mapping")
        training_contract_copy = copy.deepcopy(dict(training_contract))
        if common_training_contract is None:
            common_training_contract = training_contract_copy
        elif training_contract_copy != common_training_contract:
            raise ValueError("checkpoint set has divergent SFT training contracts")
        overlap = _validate_checkpoint_overlap(
            checkpoint,
            expected_audit=context.overlap_audit,
        )
        if common_overlap is None:
            common_overlap = overlap
        elif overlap != common_overlap:
            raise ValueError("checkpoint set has divergent sealed overlap evidence")
        completed_steps, planned_steps = _checkpoint_step(checkpoint, path=path)
        if completed_steps in observed_steps:
            raise ValueError(
                f"checkpoint set contains duplicate completed step {completed_steps}"
            )
        observed_steps.add(completed_steps)

        heldout_baseline = _validated_resume_heldout_baseline(
            checkpoint,
            expected_contract=context.eval_contract,
        )
        if heldout_baseline is None:
            raise ValueError("checkpoint heldout baseline is missing")
        baseline_metrics = _metric_mapping(
            heldout_baseline["pre"],
            label="checkpoint heldout baseline",
        )
        if baseline_metrics["rows"] != context.assistant_decisions:
            raise ValueError(
                "checkpoint heldout baseline row count does not match assistant decisions"
            )
        if baseline_metrics["assistant_loss_tokens"] != context.assistant_loss_tokens:
            raise ValueError(
                "checkpoint heldout baseline assistant loss tokens do not match expected_eval"
            )
        _assert_expected_baseline(baseline_metrics, expected_baseline)
        normalized_baseline = {
            "contract": copy.deepcopy(dict(heldout_baseline["contract"])),
            "pre": baseline_metrics,
        }
        if common_baseline is None:
            common_baseline = normalized_baseline
        elif normalized_baseline != common_baseline:
            raise ValueError("checkpoint set has divergent heldout baseline")

        state = checkpoint.get("state_dict", checkpoint.get("model"))
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint has no state_dict/model mapping")
        model = LocalAgentLM(context.model_config)
        model.load_state_dict(state, strict=True)
        metrics = _metric_mapping(
            _evaluate_conversations(
                model,
                context.conversations,
                context.tokenizer,
                max_seq_len=context.max_seq_len,
                batch_size=context.batch_size,
                device=str(device),
                amp_dtype=dtype,
                conversation_prompt_contract=context.prompt_contract,
                pad_to_input_tokens=context.pad_to_input_tokens,
            ),
            label="teacher-forced checkpoint metrics",
        )
        if metrics["rows"] != context.assistant_decisions:
            raise RuntimeError(
                "teacher-forced evaluator did not score every assistant decision"
            )
        if metrics["assistant_loss_tokens"] != context.assistant_loss_tokens:
            raise RuntimeError(
                "teacher-forced evaluator assistant loss tokens do not match expected_eval"
            )
        delta, gates, retention_eligible = _retention_result(
            metrics,
            baseline_metrics,
            thresholds,
        )
        del model, state, checkpoint
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        _assert_checkpoint_unchanged(
            path,
            expected_state=file_state,
            expected_sha256=str(artifact["sha256"]),
        )
        records.append(
            {
                "artifact": artifact,
                "checkpoint_step": completed_steps - 1,
                "completed_steps": completed_steps,
                "planned_steps": planned_steps,
                "metrics": metrics,
                "delta_from_baseline": delta,
                "gates": gates,
                "retention_eligible": retention_eligible,
            }
        )

    if (
        common_lineage is None
        or common_training_contract is None
        or common_baseline is None
        or common_overlap is None
    ):
        raise RuntimeError("checkpoint sweep produced no validated checkpoints")
    records.sort(
        key=lambda record: (
            int(record["completed_steps"]),
            str(record["artifact"]["path"]),
        )
    )
    eligible_count = sum(record["retention_eligible"] for record in records)
    best = _best_checkpoint(records)
    report = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "sweep_config": _file_record(config_source, config_payload),
            "sweep_config_sha256": canonical_sha256(config),
            "training_config": copy.deepcopy(dict(context.training_config_artifact)),
            "training_config_sha256": context.training_config_sha256,
            "checkpoint_discovery": discovery,
            "expected_parent_checkpoint_sha256": expected_parent_checkpoint_sha256,
            "expected_eval": copy.deepcopy(expected_eval),
            "expected_baseline": copy.deepcopy(expected_baseline),
        },
        "identity": {
            "model_config": copy.deepcopy(dict(context.model_config_artifact)),
            "model_config_sha256": context.model_config_sha256,
            "tokenizer": copy.deepcopy(dict(context.tokenizer_record)),
            "lineage": common_lineage,
            "training_contract": common_training_contract,
        },
        "heldout": {
            "sources": copy.deepcopy(list(context.eval_sources)),
            "conversations": context.conversation_count,
            "assistant_decisions": context.assistant_decisions,
            "assistant_loss_tokens": context.assistant_loss_tokens,
            "contract": copy.deepcopy(dict(context.eval_contract)),
            "baseline": copy.deepcopy(dict(common_baseline["pre"])),
            "leakage_assurance": common_overlap,
        },
        "thresholds": thresholds,
        "execution": execution,
        "selection_contract": {
            "eligible_filter": "all_non_inferiority_gates_pass",
            "ranking": [
                "assistant_sequence_accuracy_desc",
                "assistant_token_accuracy_desc",
                "mean_loss_asc",
                "completed_steps_asc",
                "checkpoint_sha256_desc",
            ],
        },
        "checkpoints": records,
        "summary": {
            "evaluated_checkpoints": len(records),
            "retention_eligible_checkpoints": eligible_count,
            "failed_checkpoints": len(records) - eligible_count,
            "status": (
                "retention_eligible_checkpoint_found"
                if best is not None
                else "no_retention_eligible_checkpoint"
            ),
            "best_retention_eligible_checkpoint": best,
        },
    }
    report["result_sha256"] = canonical_sha256(report)
    return report


def assert_sft_checkpoint_sweep_result(result: Mapping[str, Any]) -> None:
    """Validate result kind, finite JSON content, and canonical self-hash."""

    schema_version = result.get("schema_version")
    if (
        result.get("kind") != RESULT_KIND
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError("SFT checkpoint sweep result kind/schema is invalid")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("SFT checkpoint sweep result must contain finite JSON") from error
    payload = copy.deepcopy(dict(result))
    recorded = payload.pop("result_sha256", None)
    if recorded != canonical_sha256(payload):
        raise ValueError("SFT checkpoint sweep result self-hash is missing or invalid")


def write_sft_checkpoint_sweep_result(
    result: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Atomically write canonical JSON without ever targeting an evaluated checkpoint."""

    assert_sft_checkpoint_sweep_result(result)
    destination = Path(path)
    protected = {
        Path(record["artifact"]["path"]).resolve()
        for record in result.get("checkpoints", [])
        if isinstance(record, Mapping)
        and isinstance(record.get("artifact"), Mapping)
        and isinstance(record["artifact"].get("path"), str)
    }
    if destination.resolve() in protected:
        raise ValueError("refusing to overwrite an evaluated checkpoint with sweep JSON")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"refusing to replace stale temporary sweep result: {temporary}"
        )
    try:
        temporary.write_bytes(canonical_json_bytes(dict(result)))
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
