"""Build a sealed confirmatory split from the frozen paper agent-eval artifact.

The confirmatory split is selected before, and independently of, candidate model metrics:

1. reconstruct the primary 512-row development selection;
2. remove every source row with a primary-selected semantic identity or rendered prefix;
3. run the existing stratified selector over the remaining source rows;
4. copy the selected canonical JSONL rows byte-for-byte in original source order.

The builder binds the exact source/config/tokenizer identities, both selector audits, the original
source-row mapping, teacher-forced token accounting, and zero-overlap evidence against the primary
split and every configured training source. Existing artifacts are accepted only when their bytes
exactly match a fresh deterministic derivation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from localagent.data.conversation_artifact import (
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    VerifiedConversationArtifact,
    assert_no_conversation_overlap,
    canonical_json_bytes,
    conversation_semantic_sha256,
    load_verified_conversation_artifact,
    rendered_assistant_prompts,
    rendered_prompt_sha256,
    self_hashed_manifest,
)
from localagent.data.prompt_contract import FunctionCatalogCache, OPENAI_FULL_CATALOG_V1
from localagent.data.render import (
    CatalogTokenCache,
    render_conversation_rows_batch,
    shifted_token_counts,
    token_row_length,
)
from localagent.data.schema import Conversation, Role
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import (
    StratifiedEvalSelection,
    select_stratified_eval_subset,
)
from localagent.model.tokenizer import load_tokenizer

CONFIG_KIND = "localagent_confirmatory_eval_split_config"
RECEIPT_KIND = "localagent_confirmatory_eval_split_receipt"
SCHEMA_VERSION = 2
ALGORITHM = (
    "exclude_primary_semantic_and_rendered_prompt_rows_then_"
    "greedy_uncovered_strata_then_semantic_sha256_fill_v1"
)
REJECTED_PREFLIGHT_ALGORITHM = (
    "exclude_primary_semantic_rows_then_greedy_uncovered_strata_then_semantic_sha256_fill_v1"
)

PRODUCTION_SOURCE_ROWS = 5_000
PRODUCTION_PRIMARY_ROWS = 512
PRODUCTION_PRIMARY_ASSISTANT_DECISIONS = 820
PRODUCTION_PRIMARY_SEMANTIC_SET_SHA256 = (
    "5eb08ef61dcdfab5889f66ecb04c17fbce6ce2726f868ab85f1afe6120505bf3"
)
PRODUCTION_PRIMARY_AUDIT_SHA256 = "342abcb7ad550d4b73c726c7bbf74a68cc1fef5bda1ca68707fd8b75a16bc641"
PRODUCTION_SOURCE_SEMANTIC_SET_SHA256 = (
    "02c7e08baaaa97b54f522ba3ee5f979993000de5a3c24507c7bc4a2479355999"
)
PRODUCTION_SOURCE_RENDERED_PROMPT_SET_SHA256 = (
    "be22b4d2c67112cb4015f0feafa9fc36c085dec37a39cef9f2e782500ec669b2"
)
PRODUCTION_PRIMARY_RENDERED_PROMPT_SET_SHA256 = (
    "4b4ac1e5fe5a4e9889beae61cfdec489f714c1047ce0c90bd40de9ebdafdb3a3"
)
PRODUCTION_SEMANTIC_EXCLUDED_ROWS = 512
PRODUCTION_RENDERED_PROMPT_EXCLUDED_ROWS = 40
PRODUCTION_REMAINING_ROWS = 4_448
PRODUCTION_CONFIRMATORY_ROWS = 512
PRODUCTION_CONFIRMATORY_ASSISTANT_DECISIONS = 819
PRODUCTION_CONFIRMATORY_SEMANTIC_SET_SHA256 = (
    "5126776338f59a908fd633383499983ec0f20f492126098f9a835b29461218c7"
)
PRODUCTION_CONFIRMATORY_RENDERED_PROMPT_SET_SHA256 = (
    "30da241bcd615a59b35e877c4c3858f88404e7b0d9a90785fe31ead072a31946"
)
PRODUCTION_FILTERED_AUDIT_SHA256 = (
    "86d831e067e2effaf29d75a2b14c4bf851abb837075202824b5d724e198c0926"
)
PRODUCTION_ORIGINAL_ROWS_SHA256 = "c1c67156489e732a5b07ef2d2f4a7dde027e47d7a16f072ebaf8ca7815a102d8"
PRODUCTION_REFERENCE_CONTRACT_SHA256 = (
    "1bb66f211f76c8ed5ce402dfdf7416ec5cff65248a98110222a75123d88dc962"
)
PRODUCTION_ASSISTANT_LOSS_TOKENS = 18_245
PRODUCTION_MAX_POST_SHIFT_INPUT_TOKENS = 3_597
PRODUCTION_OUTPUT_BYTES = 5_208_274
PRODUCTION_OUTPUT_SHA256 = "cd7da2af63d16e5d117a0325d8d034b05a42c95beb6b22477f45b424de8f5ed4"
PRODUCTION_MANIFEST_BYTES = 51_713
PRODUCTION_MANIFEST_SHA256 = "fcb08787429193ea7999641386ee8b60dcaf5f21ef55a0e952f130eafa1001fb"
PRODUCTION_MANIFEST_SELF_SHA256 = "31872a329fca97cc998adca8a3c2718c3704a947b3c6766ede08308dc04304a2"
PRODUCTION_PROVENANCE_BYTES = 48_703
PRODUCTION_PROVENANCE_SHA256 = "b7847f20970e9ac2adb3f9f8893ba9f35e37b3ed3b37b862a6f2a1697f744cb8"
PRODUCTION_PROVENANCE_SELF_SHA256 = (
    "407fff78cb68f32bcd792beec3d70be58e801c2caa54d5b4c5a98fa49186a4f7"
)
REJECTED_PREFLIGHT_CONFIRMATORY_ASSISTANT_DECISIONS = 837
REJECTED_PREFLIGHT_CONFIRMATORY_SEMANTIC_SET_SHA256 = (
    "91233e8c6f1ffaa9c028830e85635569677fff939d5beecfe69ed072394f06b6"
)
REJECTED_PREFLIGHT_FILTERED_AUDIT_SHA256 = (
    "19177ffb69090b00178e125672d4b5765327c38e48dba1da6c67f4629110d18b"
)
REJECTED_PREFLIGHT_ORIGINAL_ROWS_SHA256 = (
    "6ba932bc190b8ee5f674cb0a7ec309330eeb88eba217fca6a8f34fdab2138253"
)
REJECTED_PREFLIGHT_REFERENCE_CONTRACT_SHA256 = (
    "8e6163f051863724fe4cd24264d4fac9130d6848b960e405171f0a77fad9d9a5"
)
REJECTED_PREFLIGHT_RENDERED_PROMPT_OVERLAP = 7

_CONFIG_KEYS = frozenset(
    {
        "algorithm",
        "expected",
        "kind",
        "manifest",
        "max_rows",
        "out",
        "primary_source",
        "prompt_contract",
        "provenance",
        "rejected_preflight_reference",
        "schema_version",
        "token_accounting",
        "tokenizer",
        "train_sources",
    }
)
_ARTIFACT_BINDING_KEYS = frozenset({"expected_identity", "generator_config", "manifest", "path"})
_TRAIN_SOURCE_KEYS = frozenset({"artifact", "name"})
_TOKENIZER_KEYS = frozenset({"expected_identity", "kind", "path"})
_MAX_ROWS_KEYS = frozenset({"confirmatory", "primary"})
_TOKEN_ACCOUNTING_KEYS = frozenset({"max_seq_len"})
_EXPECTED_KEYS = frozenset(
    {
        "assistant_loss_tokens",
        "confirmatory_assistant_decisions",
        "confirmatory_max_post_shift_input_tokens",
        "confirmatory_rows",
        "confirmatory_rendered_prompt_set_sha256",
        "confirmatory_selected_semantic_set_sha256",
        "filtered_selection_audit_sha256",
        "original_source_row_numbers_sha256",
        "output_bytes",
        "output_sha256",
        "primary_selected_assistant_decisions",
        "primary_rendered_prompt_set_sha256",
        "primary_selected_rows",
        "primary_selected_semantic_set_sha256",
        "primary_selection_audit_sha256",
        "reference_contract_sha256",
        "remaining_rows",
        "rendered_prompt_excluded_rows",
        "semantic_excluded_rows",
        "source_rows",
        "source_rendered_prompt_set_sha256",
        "source_semantic_set_sha256",
    }
)
_REJECTED_REFERENCE_KEYS = frozenset(
    {
        "algorithm",
        "confirmatory_assistant_decisions",
        "confirmatory_selected_semantic_set_sha256",
        "filtered_selection_audit_sha256",
        "original_source_row_numbers_sha256",
        "reason",
        "reference_contract_sha256",
        "rendered_prompt_overlap",
        "status",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")

__all__ = [
    "ALGORITHM",
    "CONFIG_KIND",
    "PRODUCTION_ASSISTANT_LOSS_TOKENS",
    "PRODUCTION_CONFIRMATORY_ASSISTANT_DECISIONS",
    "PRODUCTION_CONFIRMATORY_ROWS",
    "PRODUCTION_CONFIRMATORY_RENDERED_PROMPT_SET_SHA256",
    "PRODUCTION_CONFIRMATORY_SEMANTIC_SET_SHA256",
    "PRODUCTION_FILTERED_AUDIT_SHA256",
    "PRODUCTION_MAX_POST_SHIFT_INPUT_TOKENS",
    "PRODUCTION_MANIFEST_BYTES",
    "PRODUCTION_MANIFEST_SELF_SHA256",
    "PRODUCTION_MANIFEST_SHA256",
    "PRODUCTION_ORIGINAL_ROWS_SHA256",
    "PRODUCTION_OUTPUT_BYTES",
    "PRODUCTION_OUTPUT_SHA256",
    "PRODUCTION_PRIMARY_AUDIT_SHA256",
    "PRODUCTION_PRIMARY_RENDERED_PROMPT_SET_SHA256",
    "PRODUCTION_PRIMARY_SEMANTIC_SET_SHA256",
    "PRODUCTION_PROVENANCE_BYTES",
    "PRODUCTION_PROVENANCE_SELF_SHA256",
    "PRODUCTION_PROVENANCE_SHA256",
    "PRODUCTION_REFERENCE_CONTRACT_SHA256",
    "REJECTED_PREFLIGHT_REFERENCE_CONTRACT_SHA256",
    "RECEIPT_KIND",
    "SCHEMA_VERSION",
    "ConfirmatoryEvalSelection",
    "assert_confirmatory_eval_receipt",
    "build_confirmatory_eval_split",
    "derive_confirmatory_eval_selection",
    "load_confirmatory_eval_split_config",
    "load_confirmatory_eval_receipt",
]


@dataclass(frozen=True)
class ConfirmatoryEvalSelection:
    """The primary exclusion, filtered selection, and original source-row mapping."""

    conversations: tuple[Conversation, ...]
    original_source_row_numbers: tuple[int, ...]
    primary: StratifiedEvalSelection
    filtered: StratifiedEvalSelection
    remaining_rows: int
    semantic_excluded_rows: int
    rendered_prompt_excluded_rows: int
    source_rendered_prompt_set_sha256: str
    primary_rendered_prompt_set_sha256: str
    confirmatory_rendered_prompt_set_sha256: str
    reference_contract: Mapping[str, Any]
    reference_contract_sha256: str
    original_source_row_numbers_sha256: str


@dataclass(frozen=True)
class _LoadedConfig:
    path: Path
    root: Path
    payload: bytes
    value: Mapping[str, Any]


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    value: dict[Any, Any] = {}
    for key_node, item_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in value
        except TypeError as exc:
            raise ValueError("confirmatory-eval config mapping key is not hashable") from exc
        if duplicate:
            raise ValueError(f"confirmatory-eval config contains duplicate YAML key {key!r}")
        value[key] = loader.construct_object(item_node, deep=deep)
    return value


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash the project's canonical JSON representation, including its trailing LF."""

    return _sha256(canonical_json_bytes(value))


def _fingerprint_set_sha256(values: Sequence[str]) -> str:
    return _sha256("\n".join(sorted(set(values))).encode("ascii"))


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


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256_string(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


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


def _validate_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    identity = _validate_exact_keys(value, frozenset({"bytes", "sha256"}), label=label)
    _nonnegative_int(identity["bytes"], label=f"{label}.bytes")
    _sha256_string(identity["sha256"], label=f"{label}.sha256")
    return identity


def _validate_artifact_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    binding = _validate_exact_keys(value, _ARTIFACT_BINDING_KEYS, label=label)
    for key in ("generator_config", "manifest", "path"):
        if not isinstance(binding[key], str) or not binding[key]:
            raise ValueError(f"{label}.{key} must be a non-empty path string")
    expected_identity = _validate_exact_keys(
        binding["expected_identity"],
        frozenset(
            {
                "generator_config",
                "jsonl",
                "kind",
                "schema_version",
                "sidecar",
                "split",
            }
        ),
        label=f"{label}.expected_identity",
    )
    if expected_identity["kind"] != MANIFEST_KIND:
        raise ValueError(f"{label}.expected_identity.kind must be {MANIFEST_KIND!r}")
    if expected_identity["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{label}.expected_identity.schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if expected_identity["split"] not in {"train", "eval"}:
        raise ValueError(f"{label}.expected_identity.split must be 'train' or 'eval'")
    _validate_identity(
        expected_identity["generator_config"],
        label=f"{label}.expected_identity.generator_config",
    )
    _validate_identity(expected_identity["jsonl"], label=f"{label}.expected_identity.jsonl")
    sidecar = _validate_exact_keys(
        expected_identity["sidecar"],
        frozenset({"bytes", "manifest_self_sha256", "sha256"}),
        label=f"{label}.expected_identity.sidecar",
    )
    _nonnegative_int(sidecar["bytes"], label=f"{label}.expected_identity.sidecar.bytes")
    _sha256_string(
        sidecar["manifest_self_sha256"],
        label=f"{label}.expected_identity.sidecar.manifest_self_sha256",
    )
    _sha256_string(
        sidecar["sha256"],
        label=f"{label}.expected_identity.sidecar.sha256",
    )
    return binding


def load_confirmatory_eval_split_config(config_path: str | Path) -> Mapping[str, Any]:
    """Strictly parse and validate one confirmatory-split YAML config."""

    path = Path(config_path).resolve()
    payload = path.read_bytes()
    try:
        decoded = payload.decode("utf-8", errors="strict")
        value = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("confirmatory-eval config is not valid UTF-8 YAML") from exc
    config = _validate_exact_keys(value, _CONFIG_KEYS, label="confirmatory-eval config")
    if config["kind"] != CONFIG_KIND:
        raise ValueError(f"confirmatory-eval config.kind must be {CONFIG_KIND!r}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"confirmatory-eval config.schema_version must be {SCHEMA_VERSION}")
    if config["algorithm"] != ALGORITHM:
        raise ValueError(f"confirmatory-eval config.algorithm must be {ALGORITHM!r}")
    if config["prompt_contract"] != OPENAI_FULL_CATALOG_V1:
        raise ValueError("confirmatory-eval prompt_contract must be openai_full_catalog_v1")

    _validate_artifact_binding(config["primary_source"], label="primary_source")
    train_sources = config["train_sources"]
    if not isinstance(train_sources, list) or not train_sources:
        raise ValueError("train_sources must be a non-empty list")
    names: set[str] = set()
    for index, value in enumerate(train_sources):
        entry = _validate_exact_keys(
            value,
            _TRAIN_SOURCE_KEYS,
            label=f"train_sources[{index}]",
        )
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"train_sources[{index}].name must be non-empty text")
        if name in names:
            raise ValueError(f"duplicate train source name: {name!r}")
        names.add(name)
        binding = _validate_artifact_binding(
            entry["artifact"],
            label=f"train_sources[{index}].artifact",
        )
        if binding["expected_identity"]["split"] != "train":
            raise ValueError(f"train_sources[{index}] must bind a train artifact")

    tokenizer = _validate_exact_keys(
        config["tokenizer"],
        _TOKENIZER_KEYS,
        label="tokenizer",
    )
    if tokenizer["kind"] != "bpe":
        raise ValueError("confirmatory-eval tokenizer.kind must be 'bpe'")
    if not isinstance(tokenizer["path"], str) or not tokenizer["path"]:
        raise ValueError("tokenizer.path must be a non-empty path string")
    _validate_identity(tokenizer["expected_identity"], label="tokenizer.expected_identity")

    max_rows = _validate_exact_keys(config["max_rows"], _MAX_ROWS_KEYS, label="max_rows")
    _positive_int(max_rows["primary"], label="max_rows.primary")
    _positive_int(max_rows["confirmatory"], label="max_rows.confirmatory")
    token_accounting = _validate_exact_keys(
        config["token_accounting"],
        _TOKEN_ACCOUNTING_KEYS,
        label="token_accounting",
    )
    _positive_int(
        token_accounting["max_seq_len"],
        label="token_accounting.max_seq_len",
    )

    expected = _validate_exact_keys(config["expected"], _EXPECTED_KEYS, label="expected")
    for key in (
        "assistant_loss_tokens",
        "confirmatory_assistant_decisions",
        "confirmatory_max_post_shift_input_tokens",
        "confirmatory_rows",
        "output_bytes",
        "primary_selected_assistant_decisions",
        "primary_selected_rows",
        "remaining_rows",
        "semantic_excluded_rows",
        "source_rows",
    ):
        _positive_int(expected[key], label=f"expected.{key}")
    _nonnegative_int(
        expected["rendered_prompt_excluded_rows"],
        label="expected.rendered_prompt_excluded_rows",
    )
    for key in (
        "confirmatory_rendered_prompt_set_sha256",
        "confirmatory_selected_semantic_set_sha256",
        "filtered_selection_audit_sha256",
        "original_source_row_numbers_sha256",
        "output_sha256",
        "primary_rendered_prompt_set_sha256",
        "primary_selected_semantic_set_sha256",
        "primary_selection_audit_sha256",
        "reference_contract_sha256",
        "source_rendered_prompt_set_sha256",
        "source_semantic_set_sha256",
    ):
        _sha256_string(expected[key], label=f"expected.{key}")
    rejected = _validate_exact_keys(
        config["rejected_preflight_reference"],
        _REJECTED_REFERENCE_KEYS,
        label="rejected_preflight_reference",
    )
    if rejected["status"] != "rejected":
        raise ValueError("rejected_preflight_reference.status must be 'rejected'")
    if rejected["algorithm"] != REJECTED_PREFLIGHT_ALGORITHM:
        raise ValueError(
            "rejected_preflight_reference.algorithm does not name the semantic-only preflight"
        )
    if not isinstance(rejected["reason"], str) or not rejected["reason"]:
        raise ValueError("rejected_preflight_reference.reason must be non-empty text")
    _positive_int(
        rejected["confirmatory_assistant_decisions"],
        label="rejected_preflight_reference.confirmatory_assistant_decisions",
    )
    _positive_int(
        rejected["rendered_prompt_overlap"],
        label="rejected_preflight_reference.rendered_prompt_overlap",
    )
    for key in (
        "confirmatory_selected_semantic_set_sha256",
        "filtered_selection_audit_sha256",
        "original_source_row_numbers_sha256",
        "reference_contract_sha256",
    ):
        _sha256_string(
            rejected[key],
            label=f"rejected_preflight_reference.{key}",
        )
    rejected_pins = {
        "algorithm": REJECTED_PREFLIGHT_ALGORITHM,
        "confirmatory_assistant_decisions": (REJECTED_PREFLIGHT_CONFIRMATORY_ASSISTANT_DECISIONS),
        "confirmatory_selected_semantic_set_sha256": (
            REJECTED_PREFLIGHT_CONFIRMATORY_SEMANTIC_SET_SHA256
        ),
        "filtered_selection_audit_sha256": (REJECTED_PREFLIGHT_FILTERED_AUDIT_SHA256),
        "original_source_row_numbers_sha256": (REJECTED_PREFLIGHT_ORIGINAL_ROWS_SHA256),
        "reference_contract_sha256": (REJECTED_PREFLIGHT_REFERENCE_CONTRACT_SHA256),
        "rendered_prompt_overlap": REJECTED_PREFLIGHT_RENDERED_PROMPT_OVERLAP,
        "status": "rejected",
    }
    for key, expected_value in rejected_pins.items():
        if rejected[key] != expected_value:
            raise ValueError(
                f"rejected_preflight_reference.{key} must preserve the observed rejected pin"
            )
    for key in ("manifest", "out", "provenance"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a non-empty path string")
    return config


def _loaded_config(config_path: str | Path) -> _LoadedConfig:
    path = Path(config_path).resolve()
    value = load_confirmatory_eval_split_config(path)
    return _LoadedConfig(
        path=path,
        root=_repository_root(path),
        payload=path.read_bytes(),
        value=value,
    )


def _load_bound_artifact(
    binding_value: Any,
    *,
    root: Path,
    expected_split: str,
    label: str,
) -> VerifiedConversationArtifact:
    binding = _validate_artifact_binding(binding_value, label=label)
    expected = binding["expected_identity"]
    sidecar = expected["sidecar"]
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
        expected_manifest_identity=FileIdentity(
            bytes=int(sidecar["bytes"]),
            sha256=str(sidecar["sha256"]),
        ),
    )
    if artifact.lineage_identity() != dict(expected):
        raise ValueError(f"{label} complete artifact identity mismatch")
    return artifact


def _load_bound_tokenizer(value: Any, *, root: Path):
    tokenizer = _validate_exact_keys(value, _TOKENIZER_KEYS, label="tokenizer")
    path = _resolve_path(tokenizer["path"], root=root, label="tokenizer.path")
    payload = path.read_bytes()
    identity = FileIdentity.from_bytes(payload).as_dict()
    if identity != dict(tokenizer["expected_identity"]):
        raise ValueError("tokenizer complete artifact identity mismatch")
    return load_tokenizer("bpe", path), identity


def derive_confirmatory_eval_selection(
    conversations: Sequence[Conversation],
    *,
    primary_max_rows: int,
    confirmatory_max_rows: int,
) -> ConfirmatoryEvalSelection:
    """Exclude primary semantic and rendered-prefix identities, then select confirmation."""

    primary_max_rows = _positive_int(primary_max_rows, label="primary_max_rows")
    confirmatory_max_rows = _positive_int(
        confirmatory_max_rows,
        label="confirmatory_max_rows",
    )
    source = tuple(conversations)
    primary = select_stratified_eval_subset(source, max_rows=primary_max_rows)
    primary_semantic = {
        conversation_semantic_sha256(conversation) for conversation in primary.conversations
    }
    catalog_cache = FunctionCatalogCache()
    primary_prompt_fingerprints = [
        rendered_prompt_sha256(prompt)
        for conversation in primary.conversations
        for prompt in rendered_assistant_prompts(
            conversation,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
            catalog_cache=catalog_cache,
        )
    ]
    primary_prompt_set = set(primary_prompt_fingerprints)
    source_prompt_fingerprints: list[str] = []
    remaining_rows: list[tuple[int, Conversation]] = []
    semantic_excluded_rows = 0
    rendered_prompt_excluded_rows = 0
    for source_row_number, conversation in enumerate(source, start=1):
        semantic_sha256 = conversation_semantic_sha256(conversation)
        prompt_fingerprints = tuple(
            rendered_prompt_sha256(prompt)
            for prompt in rendered_assistant_prompts(
                conversation,
                conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
                catalog_cache=catalog_cache,
            )
        )
        source_prompt_fingerprints.extend(prompt_fingerprints)
        if semantic_sha256 in primary_semantic:
            semantic_excluded_rows += 1
        elif primary_prompt_set.intersection(prompt_fingerprints):
            rendered_prompt_excluded_rows += 1
        else:
            remaining_rows.append((source_row_number, conversation))
    remaining = tuple(remaining_rows)
    filtered = select_stratified_eval_subset(
        tuple(conversation for _, conversation in remaining),
        max_rows=confirmatory_max_rows,
    )
    original_rows = tuple(
        remaining[filtered_row_number - 1][0] for filtered_row_number in filtered.source_row_numbers
    )
    selected = tuple(
        remaining[filtered_row_number - 1][1] for filtered_row_number in filtered.source_row_numbers
    )
    if selected != filtered.conversations:
        raise RuntimeError("filtered selection/original source-row mapping mismatch")
    if original_rows != tuple(sorted(original_rows)) or len(set(original_rows)) != len(
        original_rows
    ):
        raise RuntimeError("confirmatory original source rows are not strictly ordered")
    confirmatory_semantic = {
        conversation_semantic_sha256(conversation) for conversation in selected
    }
    if primary_semantic & confirmatory_semantic:
        raise RuntimeError("confirmatory selection retained a primary semantic identity")
    confirmatory_prompt_fingerprints = [
        rendered_prompt_sha256(prompt)
        for conversation in selected
        for prompt in rendered_assistant_prompts(
            conversation,
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
            catalog_cache=catalog_cache,
        )
    ]
    if primary_prompt_set & set(confirmatory_prompt_fingerprints):
        raise RuntimeError("confirmatory selection retained a primary rendered prefix")

    primary_audit = primary.audit.as_dict()
    filtered_audit = filtered.audit.as_dict()
    source_prompt_set_sha256 = _fingerprint_set_sha256(source_prompt_fingerprints)
    primary_prompt_set_sha256 = _fingerprint_set_sha256(primary_prompt_fingerprints)
    confirmatory_prompt_set_sha256 = _fingerprint_set_sha256(confirmatory_prompt_fingerprints)
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "prompt_contract": OPENAI_FULL_CATALOG_V1,
        "primary_selected_semantic_set_sha256": primary.audit.selected_semantic_set_sha256,
        "primary_rendered_prompt_set_sha256": primary_prompt_set_sha256,
        "source_semantic_set_sha256": primary.audit.source_semantic_set_sha256,
        "source_rendered_prompt_set_sha256": source_prompt_set_sha256,
        "semantic_excluded_rows": semantic_excluded_rows,
        "rendered_prompt_excluded_rows": rendered_prompt_excluded_rows,
        "remaining_rows": len(remaining),
        "confirm_rows": len(selected),
        "confirm_assistant_decisions": filtered.audit.selected_assistant_decisions,
        "confirm_semantic_set_sha256": filtered.audit.selected_semantic_set_sha256,
        "confirm_rendered_prompt_set_sha256": confirmatory_prompt_set_sha256,
        "confirm_original_source_row_numbers": list(original_rows),
        "inner_filtered_selection_audit_sha256": filtered_audit["audit_sha256"],
    }
    if primary_audit["algorithm"] != STRATIFIED_EVAL_ALGORITHM:
        raise RuntimeError("primary selector algorithm drifted")
    if filtered_audit["algorithm"] != STRATIFIED_EVAL_ALGORITHM:
        raise RuntimeError("filtered selector algorithm drifted")
    return ConfirmatoryEvalSelection(
        conversations=selected,
        original_source_row_numbers=original_rows,
        primary=primary,
        filtered=filtered,
        remaining_rows=len(remaining),
        semantic_excluded_rows=semantic_excluded_rows,
        rendered_prompt_excluded_rows=rendered_prompt_excluded_rows,
        source_rendered_prompt_set_sha256=source_prompt_set_sha256,
        primary_rendered_prompt_set_sha256=primary_prompt_set_sha256,
        confirmatory_rendered_prompt_set_sha256=confirmatory_prompt_set_sha256,
        reference_contract=contract,
        reference_contract_sha256=_canonical_sha256(contract),
        original_source_row_numbers_sha256=_canonical_sha256(list(original_rows)),
    )


def _token_accounting(
    conversations: Sequence[Conversation],
    tokenizer,
    *,
    max_seq_len: int,
) -> dict[str, Any]:
    rows = render_conversation_rows_batch(
        conversations,
        tokenizer,
        prompt_contract=OPENAI_FULL_CATALOG_V1,
        max_seq_len=max_seq_len,
        catalog_cache=CatalogTokenCache(tokenizer),
    )
    counts = [shifted_token_counts(row) for row in rows]
    if not counts or any(loss_tokens < 1 for _, loss_tokens in counts):
        raise ValueError("confirmatory split contains an untrainable assistant decision")
    input_widths = [token_row_length(row) - 1 for row in rows]
    return {
        "accounting_kind": "exact_shifted_masked_language_model_tokens",
        "assistant_decisions": len(rows),
        "assistant_loss_tokens": sum(loss for _, loss in counts),
        "input_tokens": sum(inputs for inputs, _ in counts),
        "max_post_shift_input_tokens": max(input_widths),
        "max_seq_len": max_seq_len,
        "min_post_shift_input_tokens": min(input_widths),
        "prompt_contract": OPENAI_FULL_CATALOG_V1,
    }


def _assert_expected(
    *,
    expected: Mapping[str, Any],
    source_rows: int,
    selection: ConfirmatoryEvalSelection,
    accounting: Mapping[str, Any],
    output_identity: FileIdentity,
) -> None:
    primary_audit = selection.primary.audit.as_dict()
    filtered_audit = selection.filtered.audit.as_dict()
    observed = {
        "assistant_loss_tokens": accounting["assistant_loss_tokens"],
        "confirmatory_assistant_decisions": selection.filtered.audit.selected_assistant_decisions,
        "confirmatory_max_post_shift_input_tokens": accounting["max_post_shift_input_tokens"],
        "confirmatory_rendered_prompt_set_sha256": (
            selection.confirmatory_rendered_prompt_set_sha256
        ),
        "confirmatory_rows": len(selection.conversations),
        "confirmatory_selected_semantic_set_sha256": (
            selection.filtered.audit.selected_semantic_set_sha256
        ),
        "filtered_selection_audit_sha256": filtered_audit["audit_sha256"],
        "original_source_row_numbers_sha256": (selection.original_source_row_numbers_sha256),
        "output_bytes": output_identity.bytes,
        "output_sha256": output_identity.sha256,
        "primary_rendered_prompt_set_sha256": (selection.primary_rendered_prompt_set_sha256),
        "primary_selected_assistant_decisions": (
            selection.primary.audit.selected_assistant_decisions
        ),
        "primary_selected_rows": len(selection.primary.conversations),
        "primary_selected_semantic_set_sha256": (
            selection.primary.audit.selected_semantic_set_sha256
        ),
        "primary_selection_audit_sha256": primary_audit["audit_sha256"],
        "reference_contract_sha256": selection.reference_contract_sha256,
        "remaining_rows": selection.remaining_rows,
        "rendered_prompt_excluded_rows": selection.rendered_prompt_excluded_rows,
        "semantic_excluded_rows": selection.semantic_excluded_rows,
        "source_rows": source_rows,
        "source_rendered_prompt_set_sha256": (selection.source_rendered_prompt_set_sha256),
        "source_semantic_set_sha256": selection.primary.audit.source_semantic_set_sha256,
    }
    for key, value in observed.items():
        if expected[key] != value:
            raise ValueError(
                f"confirmatory production expectation mismatch for {key}: "
                f"expected={expected[key]!r}, observed={value!r}"
            )


def _selected_source_payload(
    source: VerifiedConversationArtifact,
    selection: ConfirmatoryEvalSelection,
) -> bytes:
    payload = source.data_path.read_bytes()
    if FileIdentity.from_bytes(payload) != source.identity.jsonl:
        raise RuntimeError("verified source bytes changed before confirmatory materialization")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(source.conversations) or b"".join(lines) != payload:
        raise RuntimeError("verified source JSONL line mapping changed")
    selected_lines: list[bytes] = []
    for conversation, row_number in zip(
        selection.conversations,
        selection.original_source_row_numbers,
        strict=True,
    ):
        raw = lines[row_number - 1]
        if raw != (conversation.to_json() + "\n").encode("utf-8"):
            raise RuntimeError(
                f"verified source row {row_number} is not the selected Conversation identity"
            )
        selected_lines.append(raw)
    return b"".join(selected_lines)


def _argument_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _response_grounded_followup(conversation: Conversation) -> bool:
    initial_user = next(
        (
            message.content
            for message in conversation.messages
            if message.role == Role.user and message.content
        ),
        "",
    )
    prior_responses: list[str] = []
    for message in conversation.messages:
        if message.role == Role.tool and message.tool_response:
            prior_responses.append(message.tool_response)
        elif message.role == Role.assistant and prior_responses:
            for call in message.tool_calls:
                for value in call.arguments.values():
                    if (
                        isinstance(value, str)
                        and value
                        and value not in initial_user
                        and any(value in response for response in prior_responses)
                    ):
                        return True
    return False


def _verified_error_recovery(conversation: Conversation) -> bool:
    failure_seen = False
    remediation_seen = False
    retry_seen = False
    for message in conversation.messages:
        if message.role == Role.tool and message.tool_response:
            response = message.tool_response.casefold()
            if "failed" in response:
                failure_seen = True
            elif failure_seen and retry_seen and "all tests passed" in response:
                return True
        elif failure_seen and message.role == Role.assistant and message.tool_calls:
            names = {call.name for call in message.tool_calls}
            if any(name != "run_tests" for name in names):
                remediation_seen = True
            if remediation_seen and "run_tests" in names:
                retry_seen = True
    return False


def _selected_manifest_counts(
    conversations: Sequence[Conversation],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    structure = Counter({key: 0 for key in source_manifest["structural_counts"]})
    behaviors = Counter({key: 0 for key in source_manifest["behavior_counts"]})
    arguments: Counter[str] = Counter()
    plan_lengths: Counter[int] = Counter()
    single_turn = 0
    irrelevance = 0
    for conversation in conversations:
        assistant_calls = [
            len(message.tool_calls)
            for message in conversation.messages
            if message.role == Role.assistant
        ]
        total_calls = sum(assistant_calls)
        is_multi_turn = len(conversation.messages) > 2
        is_irrelevant = conversation.meta.get("category") == "no_tool"
        single_turn += int(not is_multi_turn)
        irrelevance += int(is_irrelevant)
        if is_multi_turn:
            structure["multi_turn_conversations"] += 1
        elif is_irrelevant:
            structure["irrelevance_conversations"] += 1
        elif total_calls == 0:
            structure["text_conversations"] += 1
        elif total_calls == 1:
            structure["single_call_conversations"] += 1
        else:
            structure["parallel_call_conversations"] += 1
        structure["assistant_tool_calls"] += total_calls

        if any(count > 1 for count in assistant_calls):
            behaviors["parallel_calls"] += 1
        if is_irrelevant or (
            conversation.meta.get("kind") == "planner_episode"
            and conversation.meta.get("plan_len") == 0
        ):
            behaviors["explicit_restraint"] += 1
        if _response_grounded_followup(conversation):
            behaviors["tool_response_grounded_followups"] += 1
        if _verified_error_recovery(conversation):
            behaviors["verified_error_recovery"] += 1

        registry = {tool.name: tool for tool in conversation.tools}
        present_types: set[str] = set()
        has_enum = False
        has_multiple = False
        for message in conversation.messages:
            for call in message.tool_calls:
                has_multiple = has_multiple or len(call.arguments) > 1
                properties = registry[call.name].parameters.get("properties", {})
                for name, value in call.arguments.items():
                    value_type = _argument_type(value)
                    arguments[value_type] += 1
                    present_types.add(value_type)
                    schema = properties.get(name, {})
                    has_enum = has_enum or value in schema.get("enum", ())
        for value_type in ("integer", "boolean", "number"):
            if value_type in present_types:
                behaviors[f"{value_type}_arguments"] += 1
        if has_enum:
            behaviors["enum_arguments"] += 1
        if has_multiple:
            behaviors["multiple_arguments"] += 1
        if conversation.meta.get("kind") == "planner_episode":
            plan_lengths[int(conversation.meta["plan_len"])] += 1

    return {
        "argument_value_counts": dict(sorted(arguments.items())),
        "behavior_counts": dict(sorted(behaviors.items())),
        "irrelevance": irrelevance,
        "multi_turn": len(conversations) - single_turn,
        "plan_length_counts": {
            str(length): count for length, count in sorted(plan_lengths.items())
        },
        "single_turn": single_turn,
        "structural_counts": dict(sorted(structure.items())),
    }


def _build_manifest(
    *,
    source: VerifiedConversationArtifact,
    config_identity: FileIdentity,
    output_identity: FileIdentity,
    selection: ConfirmatoryEvalSelection,
    accounting: Mapping[str, Any],
    development_overlap: Mapping[str, Any],
    rejected_preflight_reference: Mapping[str, Any],
    train_sources: Sequence[Mapping[str, Any]],
    train_overlaps: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    source_manifest = json.loads(json.dumps(source.manifest, ensure_ascii=False))
    counts = _selected_manifest_counts(selection.conversations, source_manifest)
    manifest = {
        **source_manifest,
        **counts,
        "complexity_contract": {
            "derived_subset": {
                "algorithm": ALGORITHM,
                "candidate_metrics_consulted": False,
                "original_source_rows_unchanged": True,
            },
            "source": source_manifest["complexity_contract"],
        },
        "conversation_serialization": CONVERSATION_SERIALIZATION,
        "coverage_contract": {
            "confirmatory_eval_split": {
                "filtered_selection": selection.filtered.audit.as_dict(),
                "original_source_row_numbers": list(selection.original_source_row_numbers),
                "original_source_row_numbers_sha256": (
                    selection.original_source_row_numbers_sha256
                ),
                "primary_selection": selection.primary.audit.as_dict(),
                "prompt_contract": OPENAI_FULL_CATALOG_V1,
                "reference_contract": dict(selection.reference_contract),
                "reference_contract_sha256": selection.reference_contract_sha256,
                "rejected_preflight_reference": dict(rejected_preflight_reference),
                "source_artifact": source.lineage_identity(),
                "token_accounting": dict(accounting),
                "train_source_artifacts": list(train_sources),
            },
            "semantics": (
                "exclude exact primary-selected semantic rows, rerun the frozen stratified "
                "selector on the remainder, and preserve original source-row identities"
            ),
        },
        "generator_config": config_identity.as_dict(),
        "kind": MANIFEST_KIND,
        "output_bytes": output_identity.bytes,
        "output_sha256": output_identity.sha256,
        "rows": len(selection.conversations),
        "rule_verification_scope": [
            *source_manifest["rule_verification_scope"],
            "provenance_bound_unchanged_eval_subset",
            "primary_semantic_exclusion",
            "development_and_train_rendered_prompt_overlap_zero",
        ],
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_contract": {
            "confirmatory_eval_split": {
                "development_overlap": dict(development_overlap),
                "train_source_overlaps": list(train_overlaps),
            },
            "source": source_manifest["split_contract"],
        },
        "verification_claim": (
            "rule_audited_parent_subset_with_primary_and_train_overlap_rejection"
        ),
    }
    manifest.pop("manifest_self_sha256", None)
    return self_hashed_manifest(manifest)


def _receipt(
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if "receipt_self_sha256" in core:
        raise ValueError("unsigned confirmatory receipt contains receipt_self_sha256")
    receipt = {
        **core,
        "receipt_self_sha256": _canonical_sha256(core),
    }
    return receipt, canonical_json_bytes(receipt)


def assert_confirmatory_eval_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate receipt kind, schema, and canonical semantic self-hash."""

    if receipt.get("kind") != RECEIPT_KIND or receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported confirmatory-eval receipt")
    recorded = _sha256_string(
        receipt.get("receipt_self_sha256"),
        label="receipt_self_sha256",
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_self_sha256", None)
    if recorded != _canonical_sha256(unsigned):
        raise ValueError("confirmatory-eval receipt self-hash mismatch")


def load_confirmatory_eval_receipt(path: str | Path) -> dict[str, Any]:
    """Load a receipt only when its encoding and self-hash are canonical."""

    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("confirmatory-eval receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("confirmatory-eval receipt must be a mapping")
    if payload != canonical_json_bytes(value):
        raise ValueError("confirmatory-eval receipt must use canonical JSON bytes")
    assert_confirmatory_eval_receipt(value)
    return value


def _assert_existing_exact(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace drifted confirmatory artifact: {path}")


def _publish_exact(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        _assert_existing_exact(path, payload)
        return
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
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise RuntimeError(
                    f"confirmatory artifact changed during publication: {path}"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RuntimeError(f"published confirmatory artifact failed verification: {path}")


def build_confirmatory_eval_split(config_path: str | Path) -> dict[str, Any]:
    """Derive, seal, publish, and re-verify the configured confirmatory split."""

    loaded = _loaded_config(config_path)
    config = loaded.value
    root = loaded.root
    source = _load_bound_artifact(
        config["primary_source"],
        root=root,
        expected_split="eval",
        label="primary_source",
    )
    max_rows = config["max_rows"]
    selection = derive_confirmatory_eval_selection(
        source.conversations,
        primary_max_rows=int(max_rows["primary"]),
        confirmatory_max_rows=int(max_rows["confirmatory"]),
    )
    development_overlap = assert_no_conversation_overlap(
        selection.primary.conversations,
        selection.conversations,
        left_label="primary development",
        right_label="confirmatory",
        conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
    ).as_dict()

    train_source_records: list[dict[str, Any]] = []
    train_overlap_records: list[dict[str, Any]] = []
    protected_inputs = {
        loaded.path,
        source.data_path.resolve(),
        source.manifest_path.resolve(),
        source.config_path.resolve(),
    }
    for index, entry in enumerate(config["train_sources"]):
        train = _load_bound_artifact(
            entry["artifact"],
            root=root,
            expected_split="train",
            label=f"train_sources[{index}].artifact",
        )
        audit = assert_no_conversation_overlap(
            train.conversations,
            selection.conversations,
            left_label=f"train source {entry['name']}",
            right_label="confirmatory",
            conversation_prompt_contract=OPENAI_FULL_CATALOG_V1,
        )
        train_source_records.append(
            {
                "identity": train.lineage_identity(),
                "name": entry["name"],
                "path": entry["artifact"]["path"],
            }
        )
        train_overlap_records.append(
            {
                "audit": audit.as_dict(),
                "name": entry["name"],
            }
        )
        protected_inputs.update(
            {
                train.data_path.resolve(),
                train.manifest_path.resolve(),
                train.config_path.resolve(),
            }
        )

    tokenizer, tokenizer_identity = _load_bound_tokenizer(
        config["tokenizer"],
        root=root,
    )
    tokenizer_path = _resolve_path(
        config["tokenizer"]["path"],
        root=root,
        label="tokenizer.path",
    ).resolve()
    protected_inputs.add(tokenizer_path)
    accounting = _token_accounting(
        selection.conversations,
        tokenizer,
        max_seq_len=int(config["token_accounting"]["max_seq_len"]),
    )
    output_payload = _selected_source_payload(source, selection)
    output_identity = FileIdentity.from_bytes(output_payload)
    _assert_expected(
        expected=config["expected"],
        source_rows=len(source.conversations),
        selection=selection,
        accounting=accounting,
        output_identity=output_identity,
    )

    output_path = _resolve_path(config["out"], root=root, label="out")
    manifest_path = _resolve_path(config["manifest"], root=root, label="manifest")
    provenance_path = _resolve_path(
        config["provenance"],
        root=root,
        label="provenance",
    )
    destinations = {
        output_path.resolve(),
        manifest_path.resolve(),
        provenance_path.resolve(),
    }
    if len(destinations) != 3:
        raise ValueError("confirmatory output, manifest, and provenance paths must be distinct")
    if destinations & protected_inputs:
        raise ValueError("confirmatory destination would overwrite a bound input")
    if not str(output_path).endswith(".jsonl"):
        raise ValueError("confirmatory output must use a .jsonl suffix")
    if not str(manifest_path).endswith(".jsonl.manifest.v1.json"):
        raise ValueError(
            "confirmatory manifest must use the versioned .jsonl.manifest.v1.json suffix"
        )
    if not str(provenance_path).endswith(".json"):
        raise ValueError("confirmatory provenance must use a .json suffix")

    config_identity = FileIdentity.from_bytes(loaded.payload)
    manifest, manifest_payload = _build_manifest(
        source=source,
        config_identity=config_identity,
        output_identity=output_identity,
        selection=selection,
        accounting=accounting,
        development_overlap=development_overlap,
        rejected_preflight_reference=config["rejected_preflight_reference"],
        train_sources=train_source_records,
        train_overlaps=train_overlap_records,
    )
    manifest_identity = FileIdentity.from_bytes(manifest_payload)
    config_display = (
        str(loaded.path.relative_to(root)) if loaded.path.is_relative_to(root) else str(loaded.path)
    )
    receipt_core: dict[str, Any] = {
        "algorithm": ALGORITHM,
        "config": {
            **config_identity.as_dict(),
            "path": config_display,
        },
        "development_selection": selection.primary.audit.as_dict(),
        "filtered_selection": selection.filtered.audit.as_dict(),
        "kind": RECEIPT_KIND,
        "original_source_mapping": {
            "source_row_numbers": list(selection.original_source_row_numbers),
            "source_row_numbers_sha256": (selection.original_source_row_numbers_sha256),
        },
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
            "rows": len(selection.conversations),
        },
        "overlap_evidence": {
            "development": development_overlap,
            "train_sources": train_overlap_records,
        },
        "prompt_contract": OPENAI_FULL_CATALOG_V1,
        "rejected_preflight_reference": dict(config["rejected_preflight_reference"]),
        "reference_contract": dict(selection.reference_contract),
        "reference_contract_sha256": selection.reference_contract_sha256,
        "schema_version": SCHEMA_VERSION,
        "source": source.lineage_identity(),
        "token_accounting": accounting,
        "tokenizer": {
            **tokenizer_identity,
            "kind": "bpe",
            "path": str(config["tokenizer"]["path"]),
        },
        "train_sources": train_source_records,
    }
    receipt, receipt_payload = _receipt(receipt_core)

    publications = (
        (output_path, output_payload),
        (manifest_path, manifest_payload),
        (provenance_path, receipt_payload),
    )
    for path, payload in publications:
        _assert_existing_exact(path, payload)
    for path, payload in publications:
        _publish_exact(path, payload)

    rebound = load_verified_conversation_artifact(
        output_path,
        config_path=loaded.path,
        expected_split="eval",
        manifest_path=manifest_path,
        expected_rule_verified=True,
        environment_policy="forbid",
        expected_manifest_identity=manifest_identity,
    )
    if rebound.identity.jsonl != output_identity:
        raise RuntimeError("published confirmatory JSONL identity changed after sealing")
    if tuple(conversation.to_json() for conversation in rebound.conversations) != tuple(
        conversation.to_json() for conversation in selection.conversations
    ):
        raise RuntimeError("published confirmatory rows changed after sealing")
    if load_confirmatory_eval_receipt(provenance_path) != receipt:
        raise RuntimeError("published confirmatory provenance changed after sealing")
    return receipt
