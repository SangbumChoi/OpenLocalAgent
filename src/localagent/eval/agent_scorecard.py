"""Fail-closed trained-checkpoint runner for the internal BFCL-style scorecard.

Run with:

    python -m localagent.eval.agent_scorecard path/to/eval-scorecard.yaml

The result is explicitly an internal LocalAgent benchmark.  It neither downloads nor implements
the official BFCL benchmark and must not be reported as an official BFCL score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml

from localagent.data import prompt_contract as prompt_contract_module
from localagent.data import stratified_eval_selector as stratified_eval_selector_module
from localagent.data.conversation_artifact import (
    canonical_json_bytes,
    load_verified_conversation_artifact,
)
from localagent.data.prompt_contract import (
    OPENAI_FULL_CATALOG_V1,
    assert_prompt_contract_tokenizer,
)
from localagent.data.schema import Conversation, ToolSpec
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import (
    select_stratified_eval_subset,
)
from localagent.eval.tool_eval import (
    AssistantPrediction,
    gold_output_token_statistics,
    prompt_token_statistics,
    render_function_catalog,
    score_conversations,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import BPE_EOS, BPETokenizer, ByteTokenizer
from localagent.train.device import execution_metadata, resolve_device
from localagent.train.stage_data import (
    canonical_sha256,
    checkpoint_tokenizer_sha256,
    git_identity,
    tokenizer_identity,
)

CONFIG_KIND = "localagent_internal_agent_scorecard_config"
RESULT_KIND = "localagent_internal_agent_scorecard_result"
SCHEMA_VERSION = 1

_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_CASE_BYTES = 1024 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
_SHA256 = frozenset("0123456789abcdef")
_CONFIG_KEYS = frozenset(
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
_CASE_REQUIRED_KEYS = frozenset(
    {
        "path",
        "manifest",
        "generator_config",
        "expected_split",
        "expected_rule_verified",
        "environment_policy",
    }
)
_CASE_KEYS = _CASE_REQUIRED_KEYS | {"selection"}
_CASE_SELECTION_KEYS = frozenset(
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
_GENERATION_KEYS = frozenset({"device", "max_new_tokens"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_mapping(value: Any, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _mapping_with_optional(
    value: Any,
    required_keys: frozenset[str],
    optional_keys: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    missing = sorted(required_keys - set(value))
    extra = sorted(set(value) - required_keys - optional_keys)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return dict(value)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(value)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file_identity(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _same_bound_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare descriptor/path state strongly enough to detect ordinary read-time drift."""

    return (
        _same_file_snapshot(left, right)
        and left.st_mode == right.st_mode
        and left.st_nlink == right.st_nlink
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _pathname_state(path: Path, *, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise RuntimeError(
            f"{label} pathname changed while bound to an open descriptor: {path}"
        ) from error


@contextmanager
def _open_bound_regular(
    path: Path,
    *,
    label: str,
    max_bytes: int | None,
    missing_ok: bool = False,
) -> Iterator[tuple[int, os.stat_result] | None]:
    """Open first, then bind one non-symlink regular-file pathname to that descriptor."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        if missing_ok:
            yield None
            return
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error
    except OSError as error:
        raise ValueError(f"{label} is missing or not a regular non-symlink file: {path}") from error

    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError(f"{label} is not a regular file: {path}")
        if max_bytes is not None and initial.st_size > max_bytes:
            raise ValueError(f"{label} exceeds {max_bytes} bytes: {path}")
        bound_path = _pathname_state(path, label=label)
        if not _same_bound_file_state(initial, bound_path):
            raise RuntimeError(f"{label} changed while its descriptor was being bound: {path}")
        try:
            yield descriptor, initial
        finally:
            final_descriptor = os.fstat(descriptor)
            final_path = _pathname_state(path, label=label)
            if not _same_bound_file_state(initial, final_descriptor) or not _same_bound_file_state(
                initial, final_path
            ):
                raise RuntimeError(f"{label} changed while it was being read: {path}")
    finally:
        os.close(descriptor)


def _read_regular(path: Path, *, label: str, max_bytes: int) -> bytes:
    with _open_bound_regular(path, label=label, max_bytes=max_bytes) as opened:
        if opened is None:
            raise RuntimeError("required regular-file descriptor unexpectedly missing")
        descriptor, initial = opened
        payload, observed = _read_descriptor_payload(
            descriptor,
            label=label,
            max_bytes=max_bytes,
        )
        if not _same_bound_file_state(initial, observed):
            raise RuntimeError(f"{label} changed while it was being read: {path}")
    return payload


def _yaml_mapping(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = yaml.safe_load(decoded)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a YAML mapping")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from error
    return dict(value)


def _file_identity(path: Path, payload: bytes) -> dict[str, int | str]:
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256(payload)}


def _evaluator_snapshot() -> tuple[dict[str, Any], dict[str, tuple[Path, bytes]]]:
    sources = {
        "agent_scorecard": Path(__file__).resolve(),
        "prompt_contract": Path(prompt_contract_module.__file__).resolve(),
        "stratified_eval_selector": Path(
            stratified_eval_selector_module.__file__
        ).resolve(),
        "tool_eval": Path(__file__).with_name("tool_eval.py").resolve(),
    }
    payloads = {
        name: (
            path,
            _read_regular(path, label=f"evaluator module {name}", max_bytes=_MAX_CONFIG_BYTES),
        )
        for name, path in sources.items()
    }
    source_tree = git_identity(Path(__file__).resolve())
    if source_tree is None:
        raise RuntimeError("cannot bind the current evaluator source-tree identity")
    return {
        "source_tree": source_tree,
        "modules": {
            name: _file_identity(path, payload)
            for name, (path, payload) in sorted(payloads.items())
        },
    }, payloads


def _assert_evaluator_unchanged(
    expected: Mapping[str, Any],
    payloads: Mapping[str, tuple[Path, bytes]],
) -> None:
    for name, (path, payload) in payloads.items():
        observed = _read_regular(
            path,
            label=f"evaluator module {name}",
            max_bytes=_MAX_CONFIG_BYTES,
        )
        if observed != payload:
            raise RuntimeError(f"evaluator module {name} changed during evaluation")
    observed_tree = git_identity(Path(__file__).resolve())
    if observed_tree != expected["source_tree"]:
        raise RuntimeError("evaluator source tree changed during evaluation")


def _training_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(config))
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _model_config(value: Mapping[str, Any]) -> ModelConfig:
    extra = sorted(set(value) - set(ModelConfig.__dataclass_fields__))
    if extra:
        raise ValueError(f"model config contains unsupported fields: {extra}")
    config = ModelConfig(**dict(value))
    config.assert_within_budget()
    return config


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> ModelConfig:
    raw = checkpoint.get("cfg")
    if not isinstance(raw, Mapping):
        raw = getattr(raw, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint cfg must be a mapping or dataclass")
    missing = sorted(set(ModelConfig.__dataclass_fields__) - set(raw))
    if missing:
        raise ValueError(f"checkpoint cfg is missing architecture fields: {missing}")
    return _model_config({key: raw[key] for key in ModelConfig.__dataclass_fields__})


def _hash_open_file(handle, *, max_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    observed_bytes = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        observed_bytes += len(chunk)
        if observed_bytes > max_bytes:
            raise ValueError(f"checkpoint exceeds {max_bytes} bytes while hashing")
        digest.update(chunk)
    return digest.hexdigest(), observed_bytes


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, int | str]]:
    """Hash, restricted-deserialize, and re-hash one pathname-bound descriptor.

    ``weights_only=True`` prevents arbitrary pickle-global execution.  Matching pre/post hashes,
    descriptor state, and pathname state fail closed on ordinary concurrent mutation without
    retaining a second checkpoint-sized byte buffer in memory.
    """

    with _open_bound_regular(
        path,
        label="checkpoint",
        max_bytes=_MAX_CHECKPOINT_BYTES,
    ) as opened:
        if opened is None:
            raise RuntimeError("required checkpoint descriptor unexpectedly missing")
        descriptor, before = opened
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            first_sha256, first_bytes = _hash_open_file(
                handle,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
            handle.seek(0)
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
            handle.seek(0)
            second_sha256, second_bytes = _hash_open_file(
                handle,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
            after = os.fstat(handle.fileno())
        if (
            first_sha256 != second_sha256
            or first_bytes != before.st_size
            or second_bytes != before.st_size
            or not _same_bound_file_state(before, after)
        ):
            raise RuntimeError("checkpoint changed while it was being loaded")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload must be a mapping")
    return checkpoint, {
        "path": str(path),
        "bytes": before.st_size,
        "sha256": first_sha256,
    }


def _validate_lineage(
    checkpoint: Mapping[str, Any],
    *,
    training_config_sha256: str,
    model_config_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TypeError("checkpoint has no lineage mapping")
    if lineage.get("version") != 1:
        raise ValueError("checkpoint lineage version must be 1")
    stage = checkpoint.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("checkpoint stage must be non-empty text")
    if lineage.get("stage") != stage:
        raise ValueError("checkpoint stage and lineage stage disagree")
    observed_config = _require_sha256(
        lineage.get("config_sha256"),
        label="checkpoint lineage.config_sha256",
    )
    observed_model = _require_sha256(
        lineage.get("model_config_sha256"),
        label="checkpoint lineage.model_config_sha256",
    )
    training_data = _require_sha256(
        lineage.get("data_sha256"),
        label="checkpoint lineage.data_sha256",
    )
    observed_tokenizer = _require_sha256(
        lineage.get("tokenizer_sha256"),
        label="checkpoint lineage.tokenizer_sha256",
    )
    if observed_config != training_config_sha256:
        raise ValueError("training config does not match checkpoint lineage")
    if observed_model != model_config_sha256:
        raise ValueError("model config does not match checkpoint lineage")
    if observed_tokenizer != tokenizer_sha256:
        raise ValueError("tokenizer does not match checkpoint lineage")
    git = lineage.get("git")
    if not isinstance(git, Mapping):
        raise TypeError("checkpoint lineage has no source-tree identity")
    for key in ("commit", "repository_sha256", "worktree_sha256"):
        value = git.get(key)
        if key == "commit":
            if (
                not isinstance(value, str)
                or len(value) != 40
                or any(character not in _SHA256 for character in value)
            ):
                raise ValueError("checkpoint lineage git.commit must be a lowercase Git SHA-1")
        else:
            _require_sha256(value, label=f"checkpoint lineage git.{key}")
    if not isinstance(git.get("dirty"), bool):
        raise TypeError("checkpoint lineage git.dirty must be boolean")
    if "parent_checkpoint_sha256" in lineage:
        _require_sha256(
            lineage["parent_checkpoint_sha256"],
            label="checkpoint lineage.parent_checkpoint_sha256",
        )
    try:
        json.dumps(lineage, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint lineage must contain finite JSON values") from error
    return {**dict(lineage), "data_sha256": training_data}


def _tokenizer(
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], bytes | None]:
    allowed = {"kind", "path"}
    extra = sorted(set(config) - allowed)
    if extra:
        raise ValueError(f"tokenizer config contains unsupported fields: {extra}")
    kind = config.get("kind")
    if kind == "byte":
        if config.get("path") is not None:
            raise ValueError("byte tokenizer must not declare a path")
        tokenizer = ByteTokenizer()
        identity = tokenizer_identity("byte", vocab_size=tokenizer.vocab_size)
        return tokenizer, identity, None
    if kind == "bpe":
        tokenizer_path = _path(config.get("path"), label="tokenizer.path")
        payload = _read_regular(
            tokenizer_path,
            label="BPE tokenizer",
            max_bytes=_MAX_CONFIG_BYTES * 16,
        )
        try:
            import tokenizers as tokenizers_package
            from tokenizers import Tokenizer

            tokenizer = BPETokenizer(Tokenizer.from_str(payload.decode("utf-8", errors="strict")))
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("BPE tokenizer is not a valid UTF-8 tokenizer artifact") from error
        package_version = getattr(tokenizers_package, "__version__", None)
        if not isinstance(package_version, str) or not package_version:
            raise RuntimeError("imported tokenizers package has no usable version")
        identity = tokenizer_identity(
            "bpe",
            vocab_size=tokenizer.vocab_size,
            path=tokenizer_path,
        )
        identity["runtime_package"] = {
            "name": "tokenizers",
            "version": package_version,
        }
        if identity["sha256"] != _sha256(payload):
            raise RuntimeError("BPE tokenizer changed while its snapshot was loaded")
        return tokenizer, identity, payload
    raise ValueError("tokenizer.kind must be 'byte' or 'bpe'")


def _validate_training_references(
    training_config: Mapping[str, Any],
    *,
    model_payload: bytes,
    tokenizer_config: Mapping[str, Any],
    tokenizer_payload: bytes | None,
) -> str:
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("training config data must be a mapping")
    prompt_contract = data.get("conversation_prompt_contract")
    if prompt_contract != OPENAI_FULL_CATALOG_V1:
        raise ValueError(
            "training config data.conversation_prompt_contract must be "
            f"{OPENAI_FULL_CATALOG_V1!r}, got {prompt_contract!r}"
        )
    referenced_model = _path(
        training_config.get("model_config"),
        label="training config model_config",
    )
    referenced_model_payload = _read_regular(
        referenced_model,
        label="training config referenced model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    if referenced_model_payload != model_payload:
        raise ValueError("training config references a different model config artifact")
    declared_tokenizer = data.get("tokenizer", {"kind": "byte"})
    if not isinstance(declared_tokenizer, Mapping):
        raise TypeError("training config data.tokenizer must be a mapping")
    if declared_tokenizer.get("kind", "byte") != tokenizer_config.get("kind"):
        raise ValueError("training config and scorecard tokenizer kinds differ")
    if tokenizer_config.get("kind") == "bpe":
        referenced_tokenizer = _path(
            declared_tokenizer.get("path"),
            label="training config data.tokenizer.path",
        )
        referenced_payload = _read_regular(
            referenced_tokenizer,
            label="training config referenced tokenizer",
            max_bytes=_MAX_CONFIG_BYTES * 16,
        )
        if referenced_payload != tokenizer_payload:
            raise ValueError("training config references a different tokenizer artifact")
    elif declared_tokenizer.get("path") is not None:
        raise ValueError("training config byte tokenizer must not declare a path")
    return prompt_contract


def _select_scorecard_cases(
    conversations: Sequence[Conversation],
    selection_config: Any,
) -> tuple[Sequence[Conversation], dict[str, Any] | None]:
    """Apply and attest an optional deterministic bounded-case selection."""

    if selection_config is None:
        return conversations, None
    selection_contract = _exact_mapping(
        selection_config,
        _CASE_SELECTION_KEYS,
        label="scorecard cases.selection",
    )
    algorithm = selection_contract["algorithm"]
    if algorithm != STRATIFIED_EVAL_ALGORITHM:
        raise ValueError(
            f"cases.selection.algorithm must be {STRATIFIED_EVAL_ALGORITHM!r}"
        )
    max_rows = _positive_int(
        selection_contract["max_rows"],
        label="cases.selection.max_rows",
    )
    expected_counts = {
        key: _positive_int(
            selection_contract[key],
            label=f"cases.selection.{key}",
        )
        for key in (
            "expected_source_rows",
            "expected_source_assistant_decisions",
            "expected_selected_rows",
            "expected_selected_assistant_decisions",
        )
    }
    expected_hashes = {
        key: _require_sha256(
            selection_contract[key],
            label=f"cases.selection.{key}",
        )
        for key in (
            "expected_source_semantic_set_sha256",
            "expected_selected_semantic_set_sha256",
            "expected_audit_sha256",
        )
    }

    selection = select_stratified_eval_subset(
        conversations,
        max_rows=max_rows,
    )
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
    expected = {
        "algorithm": algorithm,
        "max_rows": max_rows,
        **expected_counts,
        **expected_hashes,
    }
    if observed != expected:
        raise ValueError(
            "scorecard case selection contract mismatch: "
            + json.dumps(
                {"expected": expected, "observed": observed},
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    return selection.conversations, audit


def run_scorecard(config_path: str | Path) -> dict[str, Any]:
    """Load one config, verify every artifact binding, and run the internal scorecard."""

    evaluator, evaluator_payloads = _evaluator_snapshot()
    config_source = Path(config_path)
    config_payload = _read_regular(
        config_source,
        label="scorecard config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    config = _exact_mapping(
        _yaml_mapping(config_payload, label="scorecard config"),
        _CONFIG_KEYS,
        label="scorecard config",
    )
    if config.get("kind") != CONFIG_KIND or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"scorecard config must be {CONFIG_KIND!r} schema_version {SCHEMA_VERSION}"
        )

    training_path = _path(config["training_config"], label="training_config")
    training_payload = _read_regular(
        training_path,
        label="training config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    training_config = _yaml_mapping(training_payload, label="training config")
    training_sha256 = _training_config_sha256(training_config)

    model_path = _path(config["model_config"], label="model_config")
    model_payload = _read_regular(
        model_path,
        label="model config",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    model_mapping = _yaml_mapping(model_payload, label="model config")
    model_config = _model_config(model_mapping)
    model_sha256 = canonical_sha256(model_config.__dict__)

    tokenizer_config = config["tokenizer"]
    if not isinstance(tokenizer_config, Mapping):
        raise TypeError("scorecard tokenizer must be a mapping")
    tokenizer, tokenizer_lineage, tokenizer_payload = _tokenizer(tokenizer_config)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match model config")
    conversation_prompt_contract = _validate_training_references(
        training_config,
        model_payload=model_payload,
        tokenizer_config=tokenizer_config,
        tokenizer_payload=tokenizer_payload,
    )
    assert_prompt_contract_tokenizer(tokenizer, conversation_prompt_contract)

    checkpoint_path = _path(config["checkpoint"], label="checkpoint")
    checkpoint, checkpoint_identity = _load_checkpoint(checkpoint_path)
    checkpoint_prompt_contract = checkpoint.get("conversation_prompt_contract")
    if checkpoint_prompt_contract != conversation_prompt_contract:
        raise ValueError(
            "checkpoint top-level conversation_prompt_contract does not match "
            f"the training config: expected {conversation_prompt_contract!r}, "
            f"got {checkpoint_prompt_contract!r}"
        )
    checkpoint_config = _checkpoint_config(checkpoint)
    if checkpoint_config.__dict__ != model_config.__dict__:
        raise ValueError("checkpoint architecture does not match model config")
    tokenizer_sha256 = str(tokenizer_lineage["sha256"])
    if checkpoint_tokenizer_sha256(checkpoint) != tokenizer_sha256:
        raise ValueError("checkpoint tokenizer metadata does not match configured tokenizer")
    checkpoint_tokenizer = checkpoint.get("tokenizer")
    if not isinstance(checkpoint_tokenizer, Mapping):
        raise TypeError("checkpoint tokenizer metadata must be a mapping")
    if checkpoint_tokenizer.get("kind") != tokenizer_config.get("kind"):
        raise ValueError("checkpoint tokenizer kind does not match configured tokenizer")
    lineage = _validate_lineage(
        checkpoint,
        training_config_sha256=training_sha256,
        model_config_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )

    state = checkpoint.get("state_dict", checkpoint.get("model"))
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint has no state_dict/model mapping")
    model = LocalAgentLM(model_config)
    model.load_state_dict(state, strict=True)

    cases_config = _mapping_with_optional(
        config["cases"],
        _CASE_REQUIRED_KEYS,
        frozenset({"selection"}),
        label="scorecard cases",
    )
    if cases_config["expected_split"] != "eval":
        raise ValueError("internal scorecard requires an explicit eval split")
    if not isinstance(cases_config["expected_rule_verified"], bool):
        raise TypeError("cases.expected_rule_verified must be boolean")
    environment_policy = cases_config["environment_policy"]
    if environment_policy not in {"forbid", "allow", "require"}:
        raise ValueError("cases.environment_policy must be forbid, allow, or require")
    if "selection" in cases_config and cases_config["selection"] is None:
        raise TypeError("cases.selection must be a mapping when configured")
    cases = load_verified_conversation_artifact(
        _path(cases_config["path"], label="cases.path"),
        manifest_path=_path(cases_config["manifest"], label="cases.manifest"),
        config_path=_path(
            cases_config["generator_config"],
            label="cases.generator_config",
        ),
        expected_split="eval",
        expected_rule_verified=cases_config["expected_rule_verified"],
        environment_policy=environment_policy,
        max_jsonl_bytes=_MAX_CASE_BYTES,
    )
    selected_cases, selection_audit = _select_scorecard_cases(
        cases.conversations,
        cases_config.get("selection"),
    )

    generation = _exact_mapping(
        config["generation"],
        _GENERATION_KEYS,
        label="scorecard generation",
    )
    requested_device = generation["device"]
    if not isinstance(requested_device, str) or not requested_device:
        raise ValueError("generation.device must be non-empty text")
    max_new_tokens = _positive_int(
        generation["max_new_tokens"],
        label="generation.max_new_tokens",
    )
    gold_output_budget = gold_output_token_statistics(
        selected_cases,
        tokenizer,
        max_new_tokens=max_new_tokens,
    )
    if not gold_output_budget["fits_generation_budget"]:
        raise ValueError(
            "scorecard gold output budget exceeded: "
            + json.dumps(
                gold_output_budget,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    prompt_budget = prompt_token_statistics(
        selected_cases,
        tokenizer,
        max_new_tokens=max_new_tokens,
        model_max_seq_len=model_config.max_seq_len,
    )
    generation_rows = _positive_int(
        prompt_budget["assistant_decisions"],
        label="scorecard prompt budget assistant_decisions",
    )
    if generation_rows != gold_output_budget["assistant_decisions"]:
        raise RuntimeError("scorecard prompt and gold output decision counts disagree")
    largest_prompt_tokens = prompt_budget["prompt_tokens"]["maximum"]
    if isinstance(largest_prompt_tokens, bool) or not isinstance(largest_prompt_tokens, int):
        raise TypeError("scorecard prompt-token preflight returned an invalid maximum")
    required_context_tokens = largest_prompt_tokens + max_new_tokens
    if (
        required_context_tokens != prompt_budget["required_context_tokens"]
        or required_context_tokens > model_config.max_seq_len
    ):
        raise ValueError(
            "scorecard context budget exceeded: "
            + json.dumps(prompt_budget, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
    device = resolve_device(requested_device)
    model.to(device)
    model.eval()
    runtime = execution_metadata(
        requested_device=requested_device,
        resolved_device=device,
        requested_dtype="fp32",
        resolved_dtype=torch.float32,
    )

    from localagent.inference.generate import generate

    def predictor(prompt: str, tools: Sequence[ToolSpec]) -> AssistantPrediction:
        catalog = render_function_catalog(tools) + BPE_EOS
        if not prompt.startswith(catalog):
            raise RuntimeError("scorecard prompt is not bound to the supplied function catalog")
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens + max_new_tokens > model_config.max_seq_len:
            raise ValueError(
                "scorecard decode budget exceeds model context: "
                f"prompt={prompt_tokens}, max_new={max_new_tokens}, "
                f"context={model_config.max_seq_len}"
            )
        generated, stats = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        new_tokens = stats.new_tokens
        if (
            isinstance(new_tokens, bool)
            or not isinstance(new_tokens, int)
            or not 0 <= new_tokens <= max_new_tokens
        ):
            raise RuntimeError("generation returned an invalid new-token count")
        return AssistantPrediction(
            text=generated,
            finish_reason="eos" if new_tokens < max_new_tokens else "length",
        )

    score = score_conversations(selected_cases, predictor)
    _assert_evaluator_unchanged(evaluator, evaluator_payloads)
    result_without_hash = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "LocalAgent BFCL-style internal agent scorecard",
            "official_bfcl": False,
            "external_native_benchmark": False,
            "conversation_prompt_contract": conversation_prompt_contract,
        },
        "provenance": {
            "evaluator": evaluator,
            "scorecard_config": {
                **_file_identity(config_source, config_payload),
                "canonical_sha256": canonical_sha256(config),
            },
            "checkpoint": {
                **checkpoint_identity,
                "stage": checkpoint["stage"],
                "step": checkpoint.get("step"),
                "conversation_prompt_contract": checkpoint_prompt_contract,
            },
            "checkpoint_lineage": lineage,
            "training_config": {
                **_file_identity(training_path, training_payload),
                "canonical_sha256": training_sha256,
                "conversation_prompt_contract": conversation_prompt_contract,
            },
            "model_config": {
                **_file_identity(model_path, model_payload),
                "canonical_sha256": model_sha256,
                "name": model_config.name,
                "parameters": model.num_params(),
            },
            "tokenizer": {
                "kind": tokenizer_config["kind"],
                "vocab_size": tokenizer.vocab_size,
                "sha256": tokenizer_sha256,
                "runtime_package": tokenizer_lineage.get("runtime_package"),
                "artifact": (
                    {
                        "path": str(tokenizer_config["path"]),
                        "bytes": len(tokenizer_payload),
                        "sha256": _sha256(tokenizer_payload),
                    }
                    if tokenizer_payload is not None
                    else None
                ),
            },
            "training_corpus": {
                "checkpoint_lineage_data_sha256": lineage["data_sha256"],
                "independently_reconstructed_by_scorecard": False,
            },
            "cases": {
                **cases.lineage_identity(),
                "case_set_sha256": score["case_set"]["sha256"],
                "rule_verified": cases.rule_verified,
                "environment_executed": cases.environment_executed,
                **(
                    {"selection": selection_audit}
                    if selection_audit is not None
                    else {}
                ),
            },
            "generation": {
                **runtime,
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "serial_generation_calls": generation_rows,
                "serial_prefill_calls": generation_rows,
                "generation_batch_size": 1,
                "maximum_non_eos_new_tokens": generation_rows * max_new_tokens,
                "kv_cache_scope": (
                    "one generation row; one prompt prefill followed by cached one-token "
                    "decode; reset before the next row"
                ),
                "conversation_prompt_contract": conversation_prompt_contract,
                "truncation": "forbidden",
                "generation_reserve_tokens": max_new_tokens,
                "prompt_budget": prompt_budget,
                "gold_output_budget": gold_output_budget,
                "termination": (
                    "EOS iff generated non-EOS tokens are fewer than max_new_tokens; "
                    "a length-capped output is always inexact"
                ),
            },
        },
        "scorecard": score,
        "limitations": [
            "Internal Conversation-schema benchmark; not an official BFCL score.",
            "No external native benchmark semantics or environment execution are implemented.",
            (
                "Overall action_exact mixes tool and no-tool decisions; tool readiness must use "
                "tool_format_validity_on_tool_decisions and "
                "schema_validity_on_tool_decisions, whose denominator is reference tool "
                "decisions only."
            ),
            (
                "The multi-turn metric uses gold prior history and reference tool-call decisions "
                "only; it excludes no-tool decisions and is not a free-running episode rollout."
            ),
            (
                "The training-corpus identity is checkpoint-bound but is not independently "
                "reconstructed by this evaluation runner."
            ),
        ],
    }
    return {
        **result_without_hash,
        "result_self_sha256": _sha256(canonical_json_bytes(result_without_hash)),
    }


def _read_descriptor_payload(
    descriptor: int,
    *,
    label: str,
    max_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    initial = os.fstat(descriptor)
    if not stat.S_ISREG(initial.st_mode):
        raise RuntimeError(f"{label} is not an open regular file")
    hard_limit = initial.st_size if max_bytes is None else max_bytes
    if initial.st_size > hard_limit:
        raise ValueError(f"{label} exceeds {hard_limit} bytes")
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    chunks: list[bytes] = []
    observed_bytes = 0
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, hard_limit + 1 - observed_bytes),
        ):
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > hard_limit:
                raise ValueError(f"{label} exceeds {hard_limit} bytes while reading")
            if observed_bytes == hard_limit:
                extra = os.read(descriptor, 1)
                if extra:
                    raise ValueError(f"{label} exceeds {hard_limit} bytes while reading")
                break
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)
    final = os.fstat(descriptor)
    if not _same_bound_file_state(initial, final) or observed_bytes != initial.st_size:
        raise RuntimeError(f"{label} changed while reading")
    return b"".join(chunks), final


def _descriptor_payload_matches(
    descriptor: int,
    payload: bytes,
    *,
    label: str,
) -> tuple[bool, os.stat_result]:
    """Size-check, then compare one descriptor to ``payload`` in bounded chunks."""

    initial = os.fstat(descriptor)
    if not stat.S_ISREG(initial.st_mode):
        raise RuntimeError(f"{label} is not an open regular file")
    if initial.st_size != len(payload):
        final = os.fstat(descriptor)
        if not _same_bound_file_state(initial, final):
            raise RuntimeError(f"{label} changed while comparing")
        return False, final

    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    offset = 0
    matches = True
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        while offset < len(payload):
            chunk = os.read(descriptor, min(1024 * 1024, len(payload) - offset))
            if not chunk:
                matches = False
                break
            if chunk != payload[offset : offset + len(chunk)]:
                matches = False
                break
            offset += len(chunk)
        if matches and os.read(descriptor, 1):
            matches = False
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)

    final = os.fstat(descriptor)
    if not _same_bound_file_state(initial, final):
        raise RuntimeError(f"{label} changed while comparing")
    return matches and offset == len(payload), final


def _regular_payload_match(
    path: Path,
    payload: bytes,
) -> tuple[bool, os.stat_result] | None:
    """Compare an existing destination without following links or reading excess bytes."""

    label = f"scorecard result destination {path}"
    try:
        with _open_bound_regular(
            path,
            label=label,
            max_bytes=None,
            missing_ok=True,
        ) as opened:
            if opened is None:
                return None
            descriptor, initial = opened
            matches, observed = _descriptor_payload_matches(
                descriptor,
                payload,
                label=label,
            )
            if not _same_bound_file_state(initial, observed):
                raise RuntimeError(f"{label} changed while comparing")
            return matches, observed
    except ValueError as error:
        raise RuntimeError(f"scorecard result destination is not a regular file: {path}") from error


def _remove_link_if_identity(path: Path, expected: os.stat_result) -> bool:
    """Remove only ``expected`` from ``path``, preserving a raced replacement.

    POSIX has no compare-and-unlink primitive.  Moving the current entry into a private
    quarantine directory atomically captures the inode that would be removed.  A replacement is
    hard-linked back without overwriting anything; only an entry proven to be ``expected`` is
    deleted.
    """

    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    if not _same_file_identity(current, expected):
        return False
    quarantine = Path(
        tempfile.mkdtemp(
            dir=path.parent,
            prefix=f".{path.name}.rollback.",
        )
    )
    captured = quarantine / "captured"
    try:
        try:
            os.rename(path, captured)
        except FileNotFoundError:
            return False
        observed = captured.lstat()
        if _same_file_identity(observed, expected):
            captured.unlink()
            return True
        try:
            os.link(captured, path, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError(
                "scorecard path changed during rollback; concurrent replacement was preserved "
                f"at {captured}"
            ) from error
        restored = path.lstat()
        if not _same_file_identity(restored, observed):
            raise RuntimeError(
                "scorecard path changed while restoring a concurrent replacement; "
                f"replacement was preserved at {captured}"
            )
        captured.unlink()
        return False
    finally:
        try:
            quarantine.rmdir()
        except OSError:
            pass


def _publish(path: Path, payload: bytes) -> None:
    existing = _regular_payload_match(path, payload)
    if existing is not None:
        matches, _identity = existing
        if not matches:
            raise RuntimeError(f"refusing to overwrite drifted scorecard result: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    temporary_identity: os.stat_result | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            temporary_identity = os.fstat(handle.fileno())
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_matches, written = _descriptor_payload_matches(
                handle.fileno(),
                payload,
                label="scorecard result temporary",
            )
            pathname = temporary.lstat()
            if not temporary_matches or not _same_file_snapshot(written, pathname):
                raise RuntimeError("scorecard result temporary changed before publication")
            temporary_identity = written
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                concurrent = _regular_payload_match(path, payload)
                if concurrent is None:
                    raise RuntimeError(f"concurrently created scorecard result disappeared: {path}")
                concurrent_matches, _concurrent_identity = concurrent
                if not concurrent_matches:
                    raise RuntimeError(
                        f"refusing to overwrite concurrently created scorecard result: {path}"
                    )
                return

            try:
                destination_after_link = path.lstat()
            except FileNotFoundError:
                destination_after_link = None
            try:
                source_after_link = temporary.lstat()
            except FileNotFoundError:
                source_after_link = None
            rollback_identity = (
                destination_after_link
                if destination_after_link is not None
                and (
                    _same_file_identity(destination_after_link, written)
                    or (
                        source_after_link is not None
                        and _same_file_identity(destination_after_link, source_after_link)
                    )
                )
                else None
            )
            try:
                descriptor_matches, descriptor_after_link = _descriptor_payload_matches(
                    handle.fileno(),
                    payload,
                    label="scorecard result temporary",
                )
                destination = _regular_payload_match(path, payload)
                if destination is None:
                    raise RuntimeError("scorecard result destination disappeared after linking")
                destination_matches, destination_identity = destination
                source_after_verification = temporary.lstat()
                if (
                    not descriptor_matches
                    or not _same_file_snapshot(written, descriptor_after_link)
                    or not _same_file_snapshot(descriptor_after_link, source_after_verification)
                    or not destination_matches
                    or not _same_file_snapshot(descriptor_after_link, destination_identity)
                ):
                    raise RuntimeError("linked scorecard result does not match the open temporary")
            except (OSError, RuntimeError) as error:
                try:
                    if rollback_identity is not None:
                        _remove_link_if_identity(path, rollback_identity)
                except RuntimeError as rollback_error:
                    raise RuntimeError(
                        "scorecard result failed post-link verification and could not be "
                        "safely rolled back"
                    ) from rollback_error
                raise RuntimeError("scorecard result failed post-link verification") from error
    finally:
        if temporary is not None and temporary_identity is not None:
            _remove_link_if_identity(temporary, temporary_identity)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="strict scorecard YAML config")
    parser.add_argument("--out", help="optional canonical JSON result path")
    arguments = parser.parse_args(argv)
    result = run_scorecard(arguments.config)
    payload = canonical_json_bytes(result)
    if arguments.out:
        _publish(Path(arguments.out), payload)
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
