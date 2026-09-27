"""Deterministic, fail-closed audit records for packed pretraining corpora.

The packed-corpus manifest already carries shard hashes and split provenance.  This module turns
that mutable local layout into one content-addressed freeze record after independently verifying
the artifacts, tokenizer binding, split membership, decontamination inputs, and training-consumer
contracts.  The record deliberately has no timestamp or absolute path, so identical inputs produce
identical JSON on different machines.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from localagent.data.hf_corpus import (
    _verified_storage_admission,
    build_mixture_plan,
    normalize_evaluation_decontamination,
)
from localagent.data.pretrain_corpus import (
    DEFAULT_MAX_RAW_DOCUMENT_BYTES,
    MAX_SPLIT_ASSIGNMENT_LINE_BYTES,
    MANIFEST_VERSION,
    SPLIT_ASSIGNMENT_FORMAT,
    SPLIT_ASSIGNMENT_VERSION,
    PackedShardDataset,
    load_frozen_split_assignment_manifest,
)
from localagent.model.tokenizer import load_tokenizer

FREEZE_FORMAT = "localagent_packed_corpus_freeze"
FREEZE_SCHEMA_VERSION = 1
FREEZE_SPEC_KIND = "localagent_packed_corpus_freeze_spec"
FREEZE_SPEC_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_LENGTH_CHUNK_ROWS = 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, label: str) -> dict[str, int | str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing or is not a file: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _read_json_object(
    path: Path,
    *,
    label: str,
    max_bytes: int = DEFAULT_MAX_RAW_DOCUMENT_BYTES,
) -> dict[str, Any]:
    identity = _file_identity(path, label=label)
    if int(identity["bytes"]) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte audit limit: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _read_yaml_object(path: Path, *, label: str) -> dict[str, Any]:
    identity = _file_identity(path, label=label)
    if int(identity["bytes"]) > DEFAULT_MAX_RAW_DOCUMENT_BYTES:
        raise ValueError(
            f"{label} exceeds the {DEFAULT_MAX_RAW_DOCUMENT_BYTES}-byte audit limit: {path}"
        )
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{label} is not valid YAML: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping: {path}")
    return value


def _project_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _count_mapping_total(value: Any, *, label: str) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return sum(
        _nonnegative_count(count, label=f"{label}[{name!r}]") for name, count in value.items()
    )


def _assert_declared_artifact(
    declaration: Mapping[str, Any],
    path: Path,
    *,
    label: str,
) -> dict[str, int | str]:
    expected_bytes = declaration.get("bytes")
    expected_sha256 = declaration.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not _valid_sha256(expected_sha256)
    ):
        raise ValueError(f"{label} has an invalid declared content identity")
    actual = _file_identity(path, label=label)
    if actual["bytes"] != expected_bytes:
        raise ValueError(
            f"{label} byte-size mismatch: expected {expected_bytes}, got {actual['bytes']}"
        )
    if actual["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    return actual


def _joined_query_sha256(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...] = (),
) -> str:
    digest = hashlib.sha256()
    first = True
    for row in connection.execute(query, parameters):
        if not first:
            digest.update(b"\n")
        digest.update(str(row[0]).encode("utf-8"))
        first = False
    return digest.hexdigest()


def _audit_split_assignment(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream and independently verify the content-bound train/validation assignment."""

    frozen = load_frozen_split_assignment_manifest(manifest_path)
    assignment_digest = hashlib.sha256()
    first_assignment = True
    records = 0
    previous_key: tuple[str, str] | None = None
    previous_identity = ""
    previous_identity_split = ""

    with tempfile.TemporaryDirectory(prefix="localagent-corpus-freeze-") as temporary:
        database_path = Path(temporary) / "assignment.sqlite3"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                """
                CREATE TABLE assignments (
                    identity_sha256 TEXT NOT NULL,
                    document_sha256 TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    PRIMARY KEY (identity_sha256, document_sha256)
                ) WITHOUT ROWID
                """
            )
            with frozen.path.open("rb") as handle:
                header_line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
                if len(header_line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
                    raise ValueError("split-assignment header exceeds the bounded line limit")
                try:
                    header = json.loads(header_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("split-assignment header is invalid") from error
                if header != {
                    "format": SPLIT_ASSIGNMENT_FORMAT,
                    "schema_version": SPLIT_ASSIGNMENT_VERSION,
                }:
                    raise ValueError("split-assignment header is unsupported")

                line_number = 1
                while True:
                    line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
                    if not line:
                        break
                    line_number += 1
                    if len(line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
                        raise ValueError(
                            f"split-assignment row {line_number} exceeds the bounded line limit"
                        )
                    try:
                        row = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"split-assignment row {line_number} is invalid"
                        ) from error
                    if not isinstance(row, dict):
                        raise ValueError(f"split-assignment row {line_number} must be an object")
                    document_id = row.get("document_id")
                    document_sha256 = row.get("document_sha256")
                    identity = row.get("identity_sha256")
                    split = row.get("split")
                    if (
                        not isinstance(document_id, str)
                        or not _valid_sha256(document_sha256)
                        or not _valid_sha256(identity)
                        or identity != hashlib.sha256(document_id.encode("utf-8")).hexdigest()
                        or split not in {"train", "val"}
                    ):
                        raise ValueError(
                            f"split-assignment row {line_number} has an invalid binding"
                        )
                    key = (identity, document_sha256)
                    if previous_key is not None and key <= previous_key:
                        raise ValueError("split-assignment rows are not unique and sorted")
                    if identity == previous_identity and split != previous_identity_split:
                        raise ValueError(
                            "one document identity is assigned to both train and validation"
                        )
                    if identity != previous_identity:
                        previous_identity = identity
                        previous_identity_split = split
                    previous_key = key
                    value = f"{identity}:{document_sha256}:{split}".encode("ascii")
                    if not first_assignment:
                        assignment_digest.update(b"\n")
                    assignment_digest.update(value)
                    first_assignment = False
                    try:
                        connection.execute(
                            "INSERT INTO assignments VALUES (?, ?, ?, ?)",
                            (identity, document_sha256, document_id, split),
                        )
                    except sqlite3.IntegrityError as error:
                        raise ValueError("split-assignment contains a duplicate binding") from error
                    records += 1
            connection.commit()

            if records != frozen.records:
                raise ValueError(
                    "split-assignment record count mismatch: "
                    f"expected {frozen.records}, got {records}"
                )
            if assignment_digest.hexdigest() != frozen.assignment_sha256:
                raise ValueError("split-assignment semantic fingerprint mismatch")

            identity_overlap = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT identity_sha256
                        FROM assignments
                        GROUP BY identity_sha256
                        HAVING MIN(split) != MAX(split)
                    )
                    """
                ).fetchone()[0]
            )
            content_overlap = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT document_sha256
                        FROM assignments
                        GROUP BY document_sha256
                        HAVING MIN(split) != MAX(split)
                    )
                    """
                ).fetchone()[0]
            )
            duplicate_content = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM (
                        SELECT document_sha256
                        FROM assignments
                        GROUP BY document_sha256
                        HAVING COUNT(*) > 1
                    )
                    """
                ).fetchone()[0]
            )
            if identity_overlap or content_overlap:
                raise ValueError(
                    "packed train/validation split overlap: "
                    f"{identity_overlap} identity and {content_overlap} content fingerprint(s)"
                )
            if duplicate_content:
                raise ValueError(
                    "packed split assignment retains "
                    f"{duplicate_content} duplicate content fingerprint(s)"
                )

            split_audit: dict[str, dict[str, Any]] = {}
            total_identities = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT identity_sha256) FROM assignments"
                ).fetchone()[0]
            )
            for split in ("train", "val"):
                documents = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM assignments WHERE split = ?",
                        (split,),
                    ).fetchone()[0]
                )
                identities = int(
                    connection.execute(
                        """
                        SELECT COUNT(DISTINCT identity_sha256)
                        FROM assignments
                        WHERE split = ?
                        """,
                        (split,),
                    ).fetchone()[0]
                )
                document_ids_sha256 = _joined_query_sha256(
                    connection,
                    """
                    SELECT document_id
                    FROM assignments
                    WHERE split = ?
                    ORDER BY document_id
                    """,
                    (split,),
                )
                document_set_sha256 = _joined_query_sha256(
                    connection,
                    """
                    SELECT identity_sha256
                    FROM assignments
                    WHERE split = ?
                    ORDER BY identity_sha256
                    """,
                    (split,),
                )
                content_set_sha256 = _joined_query_sha256(
                    connection,
                    """
                    SELECT document_sha256
                    FROM assignments
                    WHERE split = ?
                    ORDER BY document_sha256
                    """,
                    (split,),
                )
                split_manifest = manifest.get("splits", {}).get(split)
                if not isinstance(split_manifest, Mapping):
                    raise ValueError(f"packed manifest has no {split!r} split metadata")
                expected_documents = split_manifest.get("documents")
                if expected_documents != documents:
                    raise ValueError(
                        f"{split} document count mismatch: "
                        f"expected {expected_documents}, got {documents}"
                    )
                if split_manifest.get("document_ids_sha256") != document_ids_sha256:
                    raise ValueError(f"{split} document-id fingerprint mismatch")
                if split_manifest.get("document_set_sha256") != document_set_sha256:
                    raise ValueError(f"{split} document-set fingerprint mismatch")
                split_audit[split] = {
                    "documents": documents,
                    "identities": identities,
                    "document_ids_sha256": document_ids_sha256,
                    "document_set_sha256": document_set_sha256,
                    "content_set_sha256": content_set_sha256,
                }

    declared_assignment = manifest.get("split_assignment")
    if not isinstance(declared_assignment, Mapping):
        raise ValueError("packed manifest has no split_assignment object")
    declared_identities = declared_assignment.get("identities")
    if declared_identities is not None and declared_identities != total_identities:
        raise ValueError("split-assignment identity count mismatch")
    if sum(split["documents"] for split in split_audit.values()) != records:
        raise ValueError("split document counts do not cover the assignment")
    return {
        "artifact": {"bytes": frozen.bytes, "sha256": frozen.sha256},
        "format": SPLIT_ASSIGNMENT_FORMAT,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
        "records": records,
        "identities": total_identities,
        "assignment_sha256": frozen.assignment_sha256,
        "seed": frozen.seed,
        "val_fraction": frozen.val_fraction,
        "splits": split_audit,
        "overlap": {
            "identity_fingerprints": identity_overlap,
            "content_fingerprints": content_overlap,
        },
        "duplicate_content_fingerprints": duplicate_content,
    }


def _audit_shards(
    corpus_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify shard files and recompute row/loss-token counts from bounded length chunks."""

    artifact_rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        dataset = PackedShardDataset(corpus_dir, split)
        rows = 0
        loss_tokens = 0
        split_manifest = manifest["splits"][split]
        entries = split_manifest.get("shards")
        if not isinstance(entries, list) or len(entries) != len(dataset.lengths):
            raise ValueError(f"{split} shard-entry count is invalid")
        for entry, lengths in zip(entries, dataset.lengths, strict=True):
            for start in range(0, len(lengths), _LENGTH_CHUNK_ROWS):
                values = np.asarray(
                    lengths[start : start + _LENGTH_CHUNK_ROWS],
                    dtype=np.int64,
                )
                if np.any(values < 2) or np.any(values > int(manifest["row_tokens"])):
                    raise ValueError(f"{split} shard contains an invalid packed-row length")
                loss_tokens += int((values - 1).sum())
            rows += len(lengths)
            artifact_rows.append(
                {
                    "split": split,
                    "rows": entry["rows"],
                    "tokens": {
                        "bytes": entry["bytes"],
                        "sha256": entry["sha256"],
                    },
                    "lengths": {
                        "bytes": entry["lengths_bytes"],
                        "sha256": entry["lengths_sha256"],
                    },
                }
            )
        if rows != split_manifest.get("rows"):
            raise ValueError(
                f"{split} packed-row count mismatch: "
                f"expected {split_manifest.get('rows')}, got {rows}"
            )
        if loss_tokens != split_manifest.get("tokens"):
            raise ValueError(
                f"{split} loss-token count mismatch: "
                f"expected {split_manifest.get('tokens')}, got {loss_tokens}"
            )
        source_token_counts = split_manifest.get("source_token_counts")
        if not isinstance(source_token_counts, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in source_token_counts.values()
        ):
            raise ValueError(f"{split} source-token counts are invalid")
        if sum(source_token_counts.values()) != loss_tokens:
            raise ValueError(f"{split} source-token counts do not sum to packed tokens")
        split_counts[split] = {"rows": rows, "tokens": loss_tokens}

    total_tokens = sum(value["tokens"] for value in split_counts.values())
    if manifest.get("total_tokens") != total_tokens:
        raise ValueError("packed total-token count does not match verified split counts")
    if manifest.get("train_tokens") != split_counts["train"]["tokens"]:
        raise ValueError("packed train-token count does not match verified train split")
    source_token_counts = manifest.get("source_token_counts")
    if (
        _count_mapping_total(
            source_token_counts,
            label="corpus source-token counts",
        )
        != total_tokens
    ):
        raise ValueError("corpus source-token counts do not sum to total tokens")
    total_documents = _nonnegative_count(
        manifest.get("total_documents"),
        label="corpus total_documents",
    )
    source_counts = manifest.get("source_counts")
    license_counts = manifest.get("license_counts")
    if (
        _count_mapping_total(
            source_counts,
            label="corpus source-document counts",
        )
        != total_documents
    ):
        raise ValueError("corpus source-document counts do not sum to total documents")
    if (
        _count_mapping_total(
            license_counts,
            label="corpus license counts",
        )
        != total_documents
    ):
        raise ValueError("corpus license counts do not sum to total documents")

    artifact_rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["tokens"]["sha256"]),
            str(row["lengths"]["sha256"]),
        )
    )
    return {
        "splits": split_counts,
        "artifacts": len(artifact_rows) * 2,
        "artifact_bytes": sum(
            int(row["tokens"]["bytes"]) + int(row["lengths"]["bytes"]) for row in artifact_rows
        ),
        "artifact_set_sha256": _canonical_sha256(artifact_rows),
    }


def _audit_tokenizer(
    tokenizer_path: Path | None,
    *,
    expected_kind: str,
    expected_vocab_size: int,
    expected_training_split: str | None,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    training = manifest.get("tokenizer_training")
    if not isinstance(training, Mapping):
        raise ValueError("packed manifest has no tokenizer_training lineage")

    if expected_kind == "byte":
        expected_fields = {"documents", "kind", "split", "trained"}
        unexpected_fields = sorted(set(training) - expected_fields)
        missing_fields = sorted(expected_fields - set(training))
        if unexpected_fields or missing_fields:
            details = []
            if unexpected_fields:
                details.append("unexpected " + ", ".join(repr(key) for key in unexpected_fields))
            if missing_fields:
                details.append("missing " + ", ".join(repr(key) for key in missing_fields))
            raise ValueError(
                "intrinsic byte tokenizer lineage must contain only kind/trained/split/documents "
                f"metadata ({'; '.join(details)})"
            )
        if tokenizer_path is not None:
            raise ValueError("intrinsic byte tokenizer must not have an artifact path")
        if training.get("kind") != "byte":
            raise ValueError("packed tokenizer kind does not match the freeze specification")
        if training.get("trained") is not False:
            raise ValueError("intrinsic byte tokenizer lineage must record trained=false")
        if training.get("split") is not None:
            raise ValueError("intrinsic byte tokenizer lineage must record split=null")
        documents = training.get("documents")
        if isinstance(documents, bool) or documents != 0:
            raise ValueError("intrinsic byte tokenizer lineage must record zero training documents")
        tokenizer = load_tokenizer("byte")
        if expected_vocab_size != tokenizer.vocab_size:
            raise ValueError("intrinsic byte tokenizer vocabulary must be exactly 256")
        if manifest.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("packed corpus vocabulary does not match the intrinsic byte tokenizer")
        return {
            "kind": "byte",
            "vocab_size": tokenizer.vocab_size,
            "trained": False,
            "training_split": None,
            "training_documents": 0,
        }

    if tokenizer_path is None:
        raise ValueError("artifact-backed tokenizer requires a path")
    artifact = training.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("packed tokenizer lineage has no content-addressed artifact")
    identity = _assert_declared_artifact(artifact, tokenizer_path, label="tokenizer artifact")
    if training.get("kind") != expected_kind:
        raise ValueError("packed tokenizer kind does not match the freeze specification")
    if training.get("vocab_size") != expected_vocab_size:
        raise ValueError("packed tokenizer vocabulary does not match the freeze specification")
    if training.get("split") != expected_training_split:
        raise ValueError("tokenizer was not trained on the freeze specification's required split")
    if training.get("trained") is not True:
        raise ValueError("paper corpus tokenizer lineage must record local split-only training")
    split = manifest.get("splits", {}).get(expected_training_split)
    if not isinstance(split, Mapping):
        raise ValueError("tokenizer training split is absent from the packed manifest")
    for key in ("documents", "document_ids_sha256", "document_set_sha256"):
        if training.get(key) != split.get(key):
            raise ValueError(f"tokenizer training lineage does not match split field {key!r}")
    excluded = manifest.get("total_documents", 0) - split.get("documents", 0)
    if training.get("excluded_documents") != excluded:
        raise ValueError("tokenizer excluded-document count is inconsistent")
    tokenizer = load_tokenizer(expected_kind, tokenizer_path)
    if tokenizer.vocab_size != expected_vocab_size:
        raise ValueError("tokenizer artifact vocabulary does not match the freeze specification")
    if int(manifest.get("vocab_size", -1)) != expected_vocab_size:
        raise ValueError("packed corpus vocabulary does not match the tokenizer")
    return {
        "kind": expected_kind,
        "vocab_size": expected_vocab_size,
        "artifact": identity,
        "trained": True,
        "training_split": expected_training_split,
        "training_documents": split["documents"],
        "training_document_ids_sha256": split["document_ids_sha256"],
        "training_document_set_sha256": split["document_set_sha256"],
        "excluded_documents": excluded,
    }


def _audit_evaluation_decontamination(
    *,
    root: Path,
    corpus_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    verified_source_manifests: set[tuple[int, str]],
) -> dict[str, Any]:
    policy = normalize_evaluation_decontamination(corpus_config)
    if policy is None:
        raise ValueError("freeze corpus config has no evaluation_decontamination policy")
    required = {
        str(suite["name"]): {key: suite[key] for key in ("bytes", "sha256") if key in suite}
        for suite in policy["required_suites"]
    }
    provenance = manifest.get("preparation", {}).get("provenance", {})
    decontamination = provenance.get("evaluation_denylist")
    if not isinstance(decontamination, Mapping):
        raise ValueError("packed manifest has no evaluation denylist provenance")
    declared_required_rows = decontamination.get("required_suites")
    if not isinstance(declared_required_rows, list):
        raise ValueError("packed manifest has no corpus-owned required-suite declaration")
    declared_required = {
        str(row["name"]): {key: row[key] for key in ("bytes", "sha256") if key in row}
        for row in declared_required_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    if declared_required != required:
        raise ValueError("packed required evaluation suites differ from the corpus config")

    inputs = decontamination.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("packed manifest has no evaluation denylist inputs")
    verified_inputs: dict[str, dict[str, Any]] = {}
    canonical_inputs: list[dict[str, Any]] = []
    for index, row in enumerate(inputs):
        if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
            raise ValueError(f"evaluation denylist input {index} is invalid")
        name = str(row["name"])
        if name in verified_inputs:
            raise ValueError(f"duplicate evaluation denylist input {name!r}")
        path = _project_path(root, row.get("path"), label=f"evaluation suite {name!r} path")
        identity = _assert_declared_artifact(row, path, label=f"evaluation suite {name!r}")
        verified_inputs[name] = {
            **identity,
            "source": row.get("source"),
        }
        canonical_inputs.append({"name": name, **identity})
    canonical_inputs.sort(key=lambda row: str(row["name"]))
    if decontamination.get("input_count") != len(canonical_inputs):
        raise ValueError("evaluation denylist input count mismatch")
    if decontamination.get("inputs_sha256") != _canonical_sha256(canonical_inputs):
        raise ValueError("evaluation denylist input-set fingerprint mismatch")
    missing = sorted(set(required) - set(verified_inputs))
    if missing:
        raise ValueError(
            "packed corpus is missing required evaluation suite(s): " + ", ".join(missing)
        )
    for name, constraints in required.items():
        for key, value in constraints.items():
            if verified_inputs[name][key] != value:
                raise ValueError(f"evaluation suite {name!r} violates config-owned {key}")

    policy_sources = decontamination.get("required_suite_policy_sources")
    if not isinstance(policy_sources, list) or not policy_sources:
        raise ValueError("required evaluation policy is not bound to a source manifest")
    for row in policy_sources:
        if not isinstance(row, Mapping):
            raise ValueError("required-suite policy source is invalid")
        identity = (row.get("bytes"), row.get("sha256"))
        if identity not in verified_source_manifests:
            raise ValueError("required-suite policy source is not a verified download manifest")

    list_manifests = decontamination.get("list_manifests", [])
    if not isinstance(list_manifests, list):
        raise ValueError("evaluation denylist list-manifest provenance is invalid")
    verified_lists = []
    for index, row in enumerate(list_manifests):
        if not isinstance(row, Mapping):
            raise ValueError(f"evaluation denylist list manifest {index} is invalid")
        path = _project_path(
            root,
            row.get("path"),
            label=f"evaluation denylist list manifest {index} path",
        )
        verified_lists.append(
            _assert_declared_artifact(
                row,
                path,
                label=f"evaluation denylist list manifest {index}",
            )
        )

    corpus_audit = manifest.get("corpus_audit", {}).get("evaluation_decontamination")
    if not isinstance(corpus_audit, Mapping) or corpus_audit.get("enabled") is not True:
        raise ValueError("packed corpus does not record enabled evaluation decontamination")
    normalized_entries = decontamination.get("normalized_entries")
    if corpus_audit.get("denylist_entries") != normalized_entries:
        raise ValueError("evaluation denylist normalized-entry count is inconsistent")
    return {
        "required_suites": [{"name": name, **required[name]} for name in sorted(required)],
        "inputs": [{"name": name, **verified_inputs[name]} for name in sorted(verified_inputs)],
        "input_count": len(verified_inputs),
        "inputs_sha256": _canonical_sha256(canonical_inputs),
        "list_manifests": sorted(
            verified_lists,
            key=lambda row: (str(row["sha256"]), int(row["bytes"])),
        ),
        "normalized_entries": normalized_entries,
        "audit": dict(corpus_audit),
    }


def _audit_source_provenance(
    *,
    root: Path,
    corpus_config_path: Path,
    corpus_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], set[tuple[int, str]]]:
    provenance = manifest.get("preparation", {}).get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("packed manifest has no preparation provenance")
    source_rows = provenance.get("source_manifests")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("packed manifest has no source download manifest provenance")
    config_identity = _file_identity(corpus_config_path, label="corpus mixture config")
    expected_sources = {
        str(source["name"])
        for source in corpus_config.get("sources", [])
        if isinstance(source, Mapping) and isinstance(source.get("name"), str)
    }
    if not expected_sources:
        raise ValueError("corpus mixture config has no named sources")
    expected_plan = build_mixture_plan(corpus_config_path)

    downloads: list[dict[str, Any]] = []
    verified_source_manifests: set[tuple[int, str]] = set()
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"source download manifest {index} is invalid")
        path = _project_path(root, row.get("path"), label=f"source manifest {index} path")
        identity = _assert_declared_artifact(row, path, label=f"source manifest {index}")
        download = _read_json_object(path, label=f"source manifest {index}")
        manifest_sha256 = download.get("manifest_sha256")
        unsigned_download = {
            key: value for key, value in download.items() if key != "manifest_sha256"
        }
        if not _valid_sha256(manifest_sha256):
            raise ValueError("source download manifest has an invalid self-hash")
        if manifest_sha256 != _canonical_sha256(unsigned_download):
            raise ValueError("source download manifest self-hash mismatch")
        if download.get("kind") != "localagent_hf_mixture_download_manifest":
            raise ValueError("source download manifest has an unsupported kind")
        download_version = download.get("version")
        if download_version not in {2, 3}:
            raise ValueError("source download manifest has an unsupported version")
        if download.get("config_bytes") != config_identity["bytes"]:
            raise ValueError("download manifest corpus-config byte-size mismatch")
        if download.get("config_sha256") != config_identity["sha256"]:
            raise ValueError("download manifest corpus-config SHA-256 mismatch")
        acquisition_plan = download.get("acquisition_plan")
        if not isinstance(acquisition_plan, Mapping):
            raise ValueError("download manifest has no acquisition plan")
        if dict(acquisition_plan) != expected_plan:
            raise ValueError("download manifest acquisition plan differs from corpus config")
        if download.get("plan_sha256") != expected_plan["plan_sha256"]:
            raise ValueError("download manifest acquisition-plan SHA-256 mismatch")
        if download_version == 3:
            declared_admission = download.get("storage_admission")
            expected_admission_keys = {
                "admission_sha256",
                "bytes",
                "path",
                "sha256",
            }
            if (
                not isinstance(declared_admission, Mapping)
                or set(declared_admission) != expected_admission_keys
            ):
                raise ValueError("version-3 download manifest has no valid storage admission")
            admission_name = declared_admission.get("path")
            if (
                not isinstance(admission_name, str)
                or not admission_name
                or Path(admission_name).name != admission_name
            ):
                raise ValueError("download manifest storage-admission path is invalid")
            admission_path = path.parent / admission_name
            admission_identity = _file_identity(
                admission_path,
                label=f"source manifest {index} storage admission",
            )
            admission = _verified_storage_admission(
                admission_path,
                expected_plan,
            )
            actual_admission = {
                "admission_sha256": admission["admission_sha256"],
                "bytes": admission_identity["bytes"],
                "path": admission_name,
                "sha256": admission_identity["sha256"],
            }
            if dict(declared_admission) != actual_admission:
                raise ValueError("download manifest storage-admission identity mismatch")

        runtime = download.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("download manifest has no acquisition runtime identity")
        runtime_sha256 = runtime.get("runtime_sha256")
        unsigned_runtime = {key: value for key, value in runtime.items() if key != "runtime_sha256"}
        if not _valid_sha256(runtime_sha256):
            raise ValueError("download manifest acquisition runtime has an invalid SHA-256")
        if runtime_sha256 != _canonical_sha256(unsigned_runtime):
            raise ValueError("download manifest acquisition runtime self-hash mismatch")

        expected_evidence = {
            str(evidence["id"]): evidence for evidence in expected_plan.get("license_evidence", [])
        }
        raw_evidence = download.get("license_evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("download manifest license evidence must be a list")
        verified_evidence: dict[str, dict[str, Any]] = {}
        for evidence_index, evidence in enumerate(raw_evidence):
            if not isinstance(evidence, Mapping):
                raise ValueError(f"download manifest license evidence {evidence_index} is invalid")
            evidence_id = evidence.get("id")
            if not isinstance(evidence_id, str) or evidence_id in verified_evidence:
                raise ValueError("download manifest license evidence IDs are invalid")
            expected = expected_evidence.get(evidence_id)
            if expected is None:
                raise ValueError(
                    f"download manifest has unplanned license evidence {evidence_id!r}"
                )
            for key in ("bytes", "sha256", "scope", "url"):
                if evidence.get(key) != expected[key]:
                    raise ValueError(
                        f"download manifest license evidence {evidence_id!r} differs on {key}"
                    )
            if (
                evidence.get("status") != "verified"
                or evidence.get("actual_bytes") != expected["bytes"]
                or evidence.get("actual_sha256") != expected["sha256"]
            ):
                raise ValueError(
                    f"download manifest license evidence {evidence_id!r} is not verified"
                )
            evidence_path = _project_path(
                root,
                evidence.get("path"),
                label=f"license evidence {evidence_id!r} path",
            )
            _assert_declared_artifact(
                expected,
                evidence_path,
                label=f"license evidence {evidence_id!r}",
            )
            verified_evidence[evidence_id] = dict(expected)
        if set(verified_evidence) != set(expected_evidence):
            missing = sorted(set(expected_evidence) - set(verified_evidence))
            raise ValueError(
                "download manifest is missing verified license evidence: " + ", ".join(missing)
            )

        embedded_policy = normalize_evaluation_decontamination(
            {"evaluation_decontamination": download.get("evaluation_decontamination")}
        )
        if embedded_policy != normalize_evaluation_decontamination(corpus_config):
            raise ValueError("download manifest evaluation policy differs from corpus config")
        download_sources = download.get("sources")
        if not isinstance(download_sources, Mapping) or set(download_sources) != expected_sources:
            raise ValueError("download manifest source set differs from corpus config")
        raw_path = _project_path(
            root,
            download.get("raw_jsonl"),
            label=f"source manifest {index} raw_jsonl",
        )
        raw_identity = _assert_declared_artifact(
            {
                "bytes": download.get("raw_jsonl_bytes"),
                "sha256": download.get("raw_jsonl_sha256"),
            },
            raw_path,
            label=f"source manifest {index} raw JSONL",
        )
        max_raw_bytes = expected_plan["storage"]["max_raw_jsonl_bytes"]
        if max_raw_bytes is not None and int(raw_identity["bytes"]) > int(max_raw_bytes):
            raise ValueError("download manifest raw JSONL exceeds its acquisition-plan bound")

        planned_sources = {str(source["name"]): source for source in expected_plan["sources"]}
        raw_source_artifacts = download.get("source_artifacts")
        if not isinstance(raw_source_artifacts, list) or len(raw_source_artifacts) != len(
            expected_plan["sources"]
        ):
            raise ValueError("download manifest source-artifact list is invalid")
        source_artifacts: list[dict[str, Any]] = []
        combined_digest = hashlib.sha256()
        combined_bytes = 0
        for source_index, planned in enumerate(expected_plan["sources"]):
            artifact = raw_source_artifacts[source_index]
            name = str(planned["name"])
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("name") != name
                or artifact.get("source_index") != source_index
            ):
                raise ValueError(
                    f"download source artifact {source_index} is out of order or invalid"
                )
            data_jsonl = artifact.get("data_jsonl")
            state_manifest = artifact.get("state_manifest")
            if not isinstance(data_jsonl, Mapping) or not isinstance(state_manifest, Mapping):
                raise ValueError(f"download source artifact {name!r} is incomplete")
            data_path = _project_path(
                root,
                data_jsonl.get("path"),
                label=f"download source {name!r} spool path",
            )
            data_identity = _assert_declared_artifact(
                data_jsonl,
                data_path,
                label=f"download source {name!r} spool",
            )
            with data_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                    combined_digest.update(chunk)
                    combined_bytes += len(chunk)

            state_path = _project_path(
                root,
                state_manifest.get("path"),
                label=f"download source {name!r} state path",
            )
            state_identity = _assert_declared_artifact(
                state_manifest,
                state_path,
                label=f"download source {name!r} state",
            )
            state = _read_json_object(
                state_path,
                label=f"download source {name!r} state",
            )
            state_sha256 = state.get("state_sha256")
            unsigned_state = {key: value for key, value in state.items() if key != "state_sha256"}
            if (
                not _valid_sha256(state_sha256)
                or state_sha256 != _canonical_sha256(unsigned_state)
                or state_manifest.get("state_sha256") != state_sha256
            ):
                raise ValueError(f"download source {name!r} state self-hash mismatch")
            expected_state = {
                "kind": "localagent_hf_mixture_source_state",
                "version": 1,
                "plan_sha256": expected_plan["plan_sha256"],
                "runtime_sha256": runtime_sha256,
                "source_index": source_index,
                "source_plan_sha256": _canonical_sha256(planned),
                "data_bytes": data_identity["bytes"],
                "data_sha256": data_identity["sha256"],
            }
            for key, expected_value in expected_state.items():
                if state.get(key) != expected_value:
                    raise ValueError(f"download source {name!r} state differs on {key}")
            source_stats = download_sources.get(name)
            if not isinstance(source_stats, Mapping):
                raise ValueError(f"download source {name!r} has invalid provenance")
            if state.get("stats") != source_stats:
                raise ValueError(f"download source {name!r} state statistics differ from manifest")
            if _count_mapping_total(
                state.get("license_counts"),
                label=f"download source {name!r} state license_counts",
            ) != source_stats.get("accepted_documents"):
                raise ValueError(f"download source {name!r} state license counts are inconsistent")
            source_artifacts.append(
                {
                    "name": name,
                    "source_index": source_index,
                    "data_jsonl": data_identity,
                    "state_manifest": {
                        **state_identity,
                        "state_sha256": state_sha256,
                    },
                }
            )
        if (
            combined_bytes != raw_identity["bytes"]
            or combined_digest.hexdigest() != raw_identity["sha256"]
        ):
            raise ValueError("download source spools do not concatenate to the committed raw JSONL")

        source_summaries = []
        accepted_chars_total = 0
        accepted_documents_total = 0
        for name in sorted(download_sources):
            source = download_sources[name]
            if not isinstance(source, Mapping):
                raise ValueError(f"download source {name!r} has invalid provenance")
            planned = planned_sources[name]
            for key in (
                "dataset",
                "license_evidence",
                "normalized_weight",
                "requested_chars",
                "revision",
                "subset",
            ):
                if source.get(key) != planned.get(key):
                    raise ValueError(
                        f"download source {name!r} differs from acquisition plan on {key}"
                    )
            accepted_chars = _nonnegative_count(
                source.get("accepted_chars"),
                label=f"download source {name!r} accepted_chars",
            )
            accepted_documents = _nonnegative_count(
                source.get("accepted_documents"),
                label=f"download source {name!r} accepted_documents",
            )
            if accepted_documents == 0:
                raise ValueError(f"download source {name!r} accepted no documents")
            exhausted = source.get("stream_exhausted_before_budget")
            if not isinstance(exhausted, bool):
                raise ValueError(f"download source {name!r} has an invalid exhaustion flag")
            if expected_plan["require_full_source_budgets"] and (
                exhausted or accepted_chars < int(planned["requested_chars"])
            ):
                raise ValueError(
                    f"download source {name!r} did not fill its required character budget"
                )
            skipped = source.get("skipped")
            _count_mapping_total(
                skipped,
                label=f"download source {name!r} skipped",
            )
            accepted_chars_total += accepted_chars
            accepted_documents_total += accepted_documents
            source_summaries.append(
                {
                    "name": name,
                    "dataset": source.get("dataset"),
                    "subset": source.get("subset"),
                    "revision": source.get("revision"),
                    "normalized_weight": source.get("normalized_weight"),
                    "license_evidence": source.get("license_evidence"),
                    "requested_chars": source.get("requested_chars"),
                    "accepted_chars": accepted_chars,
                    "accepted_documents": accepted_documents,
                    "skipped": dict(skipped),
                    "stream_exhausted_before_budget": exhausted,
                }
            )
        if download.get("requested_chars") != expected_plan["requested_chars"]:
            raise ValueError("download manifest requested-character total is inconsistent")
        if download.get("accepted_chars") != accepted_chars_total:
            raise ValueError("download manifest accepted-character total is inconsistent")
        if download.get("accepted_documents") != accepted_documents_total:
            raise ValueError("download manifest accepted-document total is inconsistent")
        if (
            _count_mapping_total(
                download.get("license_counts"),
                label="download manifest license_counts",
            )
            != accepted_documents_total
        ):
            raise ValueError("download manifest license counts are inconsistent")
        downloads.append(
            {
                "manifest": identity,
                "raw_jsonl": raw_identity,
                "kind": download["kind"],
                "version": download["version"],
                "manifest_sha256": manifest_sha256,
                "plan_sha256": expected_plan["plan_sha256"],
                "acquisition_plan": expected_plan,
                "runtime": dict(runtime),
                "license_evidence": [
                    verified_evidence[evidence_id] for evidence_id in sorted(verified_evidence)
                ],
                "seed": download.get("seed"),
                "requested_chars": download.get("requested_chars"),
                "accepted_chars": download.get("accepted_chars"),
                "accepted_documents": download.get("accepted_documents"),
                "license_counts": dict(download["license_counts"]),
                "source_artifacts": source_artifacts,
                "sources": source_summaries,
            }
        )
        verified_source_manifests.add((int(identity["bytes"]), str(identity["sha256"])))

    filtered = provenance.get("filtered_jsonl")
    if not isinstance(filtered, Mapping):
        raise ValueError("packed manifest has no filtered-corpus artifact")
    filtered_path = _project_path(root, filtered.get("path"), label="filtered corpus path")
    filtered_identity = _assert_declared_artifact(
        filtered,
        filtered_path,
        label="filtered corpus",
    )
    staging = manifest.get("preparation", {}).get("staging_database")
    if not isinstance(staging, Mapping):
        raise ValueError("packed manifest has no staging-database artifact")
    staging_path = _project_path(root, staging.get("path"), label="staging database path")
    staging_identity = _assert_declared_artifact(
        staging,
        staging_path,
        label="staging database",
    )
    downloads.sort(
        key=lambda row: (
            str(row["manifest"]["sha256"]),
            str(row["raw_jsonl"]["sha256"]),
        )
    )
    return (
        {
            "corpus_config": config_identity,
            "downloads": downloads,
            "filtered_jsonl": filtered_identity,
            "staging_database": {
                **staging_identity,
                "staging_version": staging.get("staging_version"),
            },
        },
        verified_source_manifests,
    )


def _audit_training_consumers(
    *,
    root: Path,
    paths: Iterable[Any],
    freeze_spec_path: Path,
    freeze_path: Path,
    corpus_dir: Path,
    tokenizer_path: Path | None,
    tokenizer_kind: str,
    vocab_size: int,
    seq_len: int,
    minimum_train_tokens: int,
) -> list[dict[str, Any]]:
    consumers = []
    for index, raw_path in enumerate(paths):
        path = _project_path(root, raw_path, label=f"training config {index}")
        config_identity = _file_identity(path, label=f"training config {index}")
        config = _read_yaml_object(path, label=f"training config {index}")
        if config.get("stage") != "pretrain":
            raise ValueError(f"training config {path} is not a pretraining stage")
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"training config {path} has no data mapping")
        configured_corpus = _project_path(
            root,
            data.get("shards_dir"),
            label=f"training config {path} shards_dir",
        )
        if configured_corpus.resolve() != corpus_dir.resolve():
            raise ValueError(f"training config {path} points at a different packed corpus")
        configured_tokenizer = data.get("tokenizer")
        if not isinstance(configured_tokenizer, Mapping):
            raise ValueError(f"training config {path} has no tokenizer mapping")
        if tokenizer_kind == "byte":
            if configured_tokenizer.get("kind") != "byte":
                raise ValueError(f"training config {path} tokenizer kind is inconsistent")
            unexpected_fields = sorted(set(configured_tokenizer) - {"kind"})
            if unexpected_fields:
                raise ValueError(
                    f"training config {path} intrinsic byte tokenizer must not declare "
                    + ", ".join(repr(key) for key in unexpected_fields)
                )
            if tokenizer_path is not None:
                raise ValueError("intrinsic byte tokenizer must not have an artifact path")
        else:
            configured_tokenizer_path = _project_path(
                root,
                configured_tokenizer.get("path"),
                label=f"training config {path} tokenizer path",
            )
            if tokenizer_path is None or (
                configured_tokenizer_path.resolve() != tokenizer_path.resolve()
            ):
                raise ValueError(f"training config {path} points at a different tokenizer")
            if configured_tokenizer.get("kind") != tokenizer_kind:
                raise ValueError(f"training config {path} tokenizer kind is inconsistent")
        if data.get("min_train_tokens") != minimum_train_tokens:
            raise ValueError(f"training config {path} has a different minimum token gate")
        configured_freeze = data.get("corpus_freeze")
        if not isinstance(configured_freeze, Mapping):
            raise ValueError(f"training config {path} does not require the corpus freeze")
        configured_freeze_spec = _project_path(
            root,
            configured_freeze.get("spec"),
            label=f"training config {path} corpus_freeze.spec",
        )
        configured_freeze_path = _project_path(
            root,
            configured_freeze.get("path"),
            label=f"training config {path} corpus_freeze.path",
        )
        if configured_freeze_spec.resolve() != freeze_spec_path.resolve():
            raise ValueError(f"training config {path} points at a different freeze specification")
        if configured_freeze_path.resolve() != freeze_path.resolve():
            raise ValueError(f"training config {path} points at a different corpus freeze")

        schedule = config.get("schedule")
        batch = config.get("batch")
        if not isinstance(schedule, Mapping) or not isinstance(batch, Mapping):
            raise ValueError(f"training config {path} has no schedule/batch contract")
        total_steps = _positive_int(
            schedule.get("total_steps"),
            label=f"training config {path} total_steps",
        )
        micro_batch_size = _positive_int(
            batch.get("micro_batch_size"),
            label=f"training config {path} micro_batch_size",
        )
        grad_accum_steps = _positive_int(
            batch.get("grad_accum_steps"),
            label=f"training config {path} grad_accum_steps",
        )
        scheduled_tokens = total_steps * micro_batch_size * grad_accum_steps * seq_len
        if scheduled_tokens < minimum_train_tokens:
            raise ValueError(
                f"training config {path} schedules {scheduled_tokens} tokens, "
                f"below the frozen minimum {minimum_train_tokens}"
            )

        model_path = _project_path(
            root,
            config.get("model_config"),
            label=f"training config {path} model_config",
        )
        model_identity = _file_identity(model_path, label=f"model config for {path}")
        model_config = _read_yaml_object(model_path, label=f"model config for {path}")
        if model_config.get("vocab_size") != vocab_size:
            raise ValueError(f"model config for {path} has a different vocabulary")
        if (
            isinstance(model_config.get("max_seq_len"), bool)
            or not isinstance(model_config.get("max_seq_len"), int)
            or int(model_config["max_seq_len"]) < seq_len
        ):
            raise ValueError(f"model config for {path} cannot consume packed sequence length")
        runtime = config.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError(f"training config {path} runtime is invalid")
        seed = runtime.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"training config {path} has no integer seed")
        consumers.append(
            {
                "label": path.stem,
                "artifact": config_identity,
                "model": {
                    "label": str(model_config.get("name", model_path.stem)),
                    "artifact": model_identity,
                },
                "seed": seed,
                "total_steps": total_steps,
                "micro_batch_size": micro_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "scheduled_tokens": scheduled_tokens,
            }
        )
    if not consumers:
        raise ValueError("freeze specification must name at least one training consumer")
    consumers.sort(
        key=lambda row: (
            int(row["seed"]),
            str(row["model"]["label"]),
            str(row["artifact"]["sha256"]),
        )
    )
    if len({str(row["artifact"]["sha256"]) for row in consumers}) != len(consumers):
        raise ValueError("freeze specification repeats a training config artifact")
    return consumers


def build_corpus_freeze(
    spec_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Audit a packed corpus and return its deterministic, self-hashed freeze record."""

    root = Path(project_root).resolve()
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = _read_yaml_object(spec_path, label="corpus freeze specification")
    if spec.get("kind") != FREEZE_SPEC_KIND:
        raise ValueError(f"freeze specification kind must be {FREEZE_SPEC_KIND!r}")
    if spec.get("schema_version") != FREEZE_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"freeze specification schema_version must be {FREEZE_SPEC_SCHEMA_VERSION}"
        )
    spec_identity = _file_identity(spec_path, label="corpus freeze specification")
    freeze_path = _project_path(root, spec.get("freeze_path"), label="freeze output path")
    corpus_config_path = _project_path(
        root,
        spec.get("corpus_config"),
        label="freeze corpus_config",
    )
    corpus_config = _read_yaml_object(corpus_config_path, label="corpus mixture config")
    corpus_dir = _project_path(root, spec.get("shards_dir"), label="freeze shards_dir")
    manifest_path = corpus_dir / "manifest.json"
    manifest = _read_json_object(manifest_path, label="packed corpus manifest")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("packed corpus manifest version is unsupported")
    manifest_identity = _file_identity(manifest_path, label="packed corpus manifest")

    tokenizer_spec = spec.get("tokenizer")
    if not isinstance(tokenizer_spec, Mapping):
        raise ValueError("freeze specification has no tokenizer mapping")
    tokenizer_kind = str(tokenizer_spec.get("kind", ""))
    tokenizer_path: Path | None
    if tokenizer_kind == "byte":
        vocab_size = _positive_int(
            tokenizer_spec.get("vocab_size"),
            label="freeze tokenizer vocab_size",
        )
        unexpected_fields = sorted(set(tokenizer_spec) - {"kind", "vocab_size"})
        if unexpected_fields:
            raise ValueError(
                "freeze intrinsic byte tokenizer must not declare "
                + ", ".join(repr(key) for key in unexpected_fields)
            )
        tokenizer_path = None
        if vocab_size != load_tokenizer("byte").vocab_size:
            raise ValueError("freeze intrinsic byte tokenizer vocabulary must be exactly 256")
    else:
        tokenizer_path = _project_path(
            root,
            tokenizer_spec.get("path"),
            label="freeze tokenizer path",
        )
        vocab_size = _positive_int(
            tokenizer_spec.get("vocab_size"),
            label="freeze tokenizer vocab_size",
        )
    expected = spec.get("expected")
    if not isinstance(expected, Mapping):
        raise ValueError("freeze specification has no expected contract mapping")
    seq_len = _positive_int(expected.get("seq_len"), label="freeze expected.seq_len")
    minimum_train_tokens = _positive_int(
        expected.get("min_train_tokens"),
        label="freeze expected.min_train_tokens",
    )
    tokenizer_training_split = expected.get("tokenizer_training_split")
    if tokenizer_kind == "byte":
        if "tokenizer_training_split" not in expected:
            raise ValueError("freeze expected contract must declare tokenizer_training_split")
        if tokenizer_training_split is not None:
            raise ValueError(
                "freeze byte tokenizer_training_split must be null because it is intrinsic"
            )
    elif tokenizer_training_split != "train":
        raise ValueError("freeze tokenizer_training_split must be 'train'")
    if manifest.get("seq_len") != seq_len:
        raise ValueError("packed sequence length does not match the freeze specification")
    if manifest.get("row_tokens") != seq_len + 1:
        raise ValueError("packed row width does not equal seq_len + 1")

    split_audit = _audit_split_assignment(manifest_path, manifest)
    shard_audit = _audit_shards(corpus_dir, manifest)
    train_tokens = int(shard_audit["splits"]["train"]["tokens"])
    if train_tokens < minimum_train_tokens:
        raise ValueError(
            "packed training corpus is below the frozen token gate: "
            f"available={train_tokens}, required={minimum_train_tokens}"
        )
    tokenizer_audit = _audit_tokenizer(
        tokenizer_path,
        expected_kind=tokenizer_kind,
        expected_vocab_size=vocab_size,
        expected_training_split=tokenizer_training_split,
        manifest=manifest,
    )
    provenance, source_manifest_identities = _audit_source_provenance(
        root=root,
        corpus_config_path=corpus_config_path,
        corpus_config=corpus_config,
        manifest=manifest,
    )
    decontamination = _audit_evaluation_decontamination(
        root=root,
        corpus_config=corpus_config,
        manifest=manifest,
        verified_source_manifests=source_manifest_identities,
    )
    quality_audit = manifest.get("corpus_audit", {}).get("quality_and_exact_deduplication")
    if (
        not isinstance(quality_audit, Mapping)
        or quality_audit.get("exact_content_deduplication") is not True
    ):
        raise ValueError("packed corpus does not record exact content deduplication")
    consumers = _audit_training_consumers(
        root=root,
        paths=spec.get("training_configs", []),
        freeze_spec_path=spec_path,
        freeze_path=freeze_path,
        corpus_dir=corpus_dir,
        tokenizer_path=tokenizer_path,
        tokenizer_kind=tokenizer_kind,
        vocab_size=vocab_size,
        seq_len=seq_len,
        minimum_train_tokens=minimum_train_tokens,
    )

    total_documents = _nonnegative_count(
        manifest.get("total_documents"),
        label="packed total_documents",
    )
    if total_documents != split_audit["records"]:
        raise ValueError("packed total-document count does not match split assignment")
    raw_documents = sum(
        _nonnegative_count(
            download.get("accepted_documents"),
            label="download accepted_documents",
        )
        for download in provenance["downloads"]
    )
    quality_input_documents = _nonnegative_count(
        quality_audit.get("input_documents"),
        label="quality audit input_documents",
    )
    if raw_documents != quality_input_documents:
        raise ValueError("download accepted-document count does not match corpus audit input count")
    payload: dict[str, Any] = {
        "format": FREEZE_FORMAT,
        "schema_version": FREEZE_SCHEMA_VERSION,
        "spec": spec_identity,
        "contract": {
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "minimum_train_tokens": minimum_train_tokens,
            "available_train_tokens": train_tokens,
            "surplus_train_tokens": train_tokens - minimum_train_tokens,
            "tokenizer_training_split": tokenizer_training_split,
        },
        "packed_corpus": {
            "manifest": {
                **manifest_identity,
                "canonical_sha256": _canonical_sha256(manifest),
            },
            "generation": manifest.get("generation"),
            "format": manifest.get("format"),
            "seq_len": manifest.get("seq_len"),
            "row_tokens": manifest.get("row_tokens"),
            "token_dtype": manifest.get("token_dtype"),
            "vocab_size": manifest.get("vocab_size"),
            "total_documents": total_documents,
            "total_tokens": manifest.get("total_tokens"),
            "train_tokens": manifest.get("train_tokens"),
            "source_counts": manifest.get("source_counts"),
            "source_token_counts": manifest.get("source_token_counts"),
            "license_counts": manifest.get("license_counts"),
            "shards": shard_audit,
        },
        "tokenizer": tokenizer_audit,
        "split_assignment": split_audit,
        "decontamination": decontamination,
        "quality_and_exact_deduplication": dict(quality_audit),
        "provenance": provenance,
        "training_consumers": consumers,
    }
    payload["freeze_sha256"] = _canonical_sha256(payload)
    return payload


def write_corpus_freeze(
    freeze: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Atomically write a canonical freeze record after checking its self-hash."""

    expected = freeze.get("freeze_sha256")
    unsigned = dict(freeze)
    unsigned.pop("freeze_sha256", None)
    if not _valid_sha256(expected) or _canonical_sha256(unsigned) != expected:
        raise ValueError("corpus freeze has an invalid self-hash")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def verify_corpus_freeze(
    freeze_path: str | Path,
    spec_path: str | Path,
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Rebuild and compare a freeze record, failing on any artifact or contract drift."""

    root = Path(project_root).resolve()
    recorded_path = Path(freeze_path)
    if not recorded_path.is_absolute():
        recorded_path = root / recorded_path
    expected = _read_json_object(recorded_path, label="corpus freeze")
    recorded_hash = expected.get("freeze_sha256")
    unsigned = dict(expected)
    unsigned.pop("freeze_sha256", None)
    if not _valid_sha256(recorded_hash) or _canonical_sha256(unsigned) != recorded_hash:
        raise ValueError("recorded corpus freeze has an invalid self-hash")
    actual = build_corpus_freeze(spec_path, project_root=project_root)
    if expected != actual:
        raise ValueError("recorded corpus freeze differs from the current audited artifacts")
    return actual
