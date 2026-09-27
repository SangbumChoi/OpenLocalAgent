"""Fail-closed derived-corpus fan-out from one verified parent corpus freeze.

Derived midtraining corpora are views of an already filtered, decontaminated, deduplicated parent
corpus.  This module deliberately does not call any filtering, near-deduplication, or evaluation
denylist builder.  It verifies the parent freeze, decodes the canonical filtered JSONL once, stages
each requested source family in SQLite, and reuses the parent's tokenizer and frozen split.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from localagent.data.corpus_freeze import verify_corpus_freeze
from localagent.data.pretrain_corpus import (
    DEFAULT_MAX_RAW_DOCUMENT_BYTES,
    MAX_SPLIT_ASSIGNMENT_LINE_BYTES,
    SPLIT_ASSIGNMENT_FORMAT,
    SPLIT_ASSIGNMENT_VERSION,
    STAGING_VERSION,
    CorpusDocument,
    DiskBackedCorpus,
    FrozenSplitAssignment,
    load_frozen_split_assignment_manifest,
    pack_disk_backed_shards,
)
from localagent.model.tokenizer import load_tokenizer

DERIVED_CORPUS_FORMAT = "localagent_parent_freeze_derived_corpus"
DERIVED_CORPUS_SCHEMA_VERSION = 1
ONE_PASS_FANOUT_CONTRACT = "canonical_parent_filtered_jsonl_single_decode_v1"
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DerivedCorpusGroup:
    """One output directory and its disjoint set of parent ``mixture_source`` values."""

    output_dir: Path
    mixture_sources: tuple[str, ...]
    logical_name: str = ""


@dataclass
class _GroupStage:
    definition: DerivedCorpusGroup
    temporary_output: Path
    database_path: Path
    connection: sqlite3.Connection
    expected_documents: int
    observed_documents: int = 0


@dataclass(frozen=True)
class _InodeIdentity:
    device: int
    inode: int


@dataclass
class _PinnedDirectory:
    path: Path
    name: str
    parent_fd: int
    fd: int
    identity: _InodeIdentity
    owned: bool


@dataclass(frozen=True)
class _OwnedFile:
    path: Path
    name: str
    parent_fd: int
    identity: _InodeIdentity
    bytes: int
    sha256: str


@dataclass
class _PublicationState:
    stage: _GroupStage
    parent_fd: int
    directories: list[_PinnedDirectory]
    files: list[_OwnedFile]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
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


@contextmanager
def _verified_regular_file_snapshot(
    path: Path,
    expected_identity: Any,
    *,
    label: str,
) -> Iterator[tuple[Path, dict[str, int | str]]]:
    """Copy one regular file from a pinned fd and verify the exact bytes copied.

    The caller consumes the private snapshot, never the original pathname. This couples artifact
    verification to the bytes actually parsed even if a non-cooperating process swaps the source
    path after it has been opened.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} is missing, unreadable, or a symbolic link: {path}") from error
    try:
        opened = os.fstat(source_fd)
        try:
            named = os.lstat(path)
        except OSError as error:
            raise ValueError(f"{label} path changed while it was opened: {path}") from error
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(f"{label} must resolve directly to one regular file: {path}")

        with tempfile.TemporaryDirectory(prefix="localagent-tokenizer-snapshot-") as directory:
            suffix = path.suffix if path.suffix else ".artifact"
            snapshot = Path(directory) / f"tokenizer{suffix}"
            digest = hashlib.sha256()
            copied_bytes = 0
            destination_fd = os.open(
                snapshot,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                while True:
                    chunk = os.read(source_fd, _HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    copied_bytes += len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_fd, view)
                        if written <= 0:
                            raise OSError("short tokenizer snapshot write")
                        view = view[written:]
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            actual_identity: dict[str, int | str] = {
                "bytes": copied_bytes,
                "sha256": digest.hexdigest(),
            }
            _assert_identity(actual_identity, expected_identity, label=label)
            if _file_identity(snapshot, label=f"private {label} snapshot") != actual_identity:
                raise ValueError(f"private {label} snapshot changed before loading")
            yield snapshot, actual_identity
    finally:
        os.close(source_fd)


def _declared_identity(value: Any, *, label: str) -> dict[str, int | str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity is missing")
    size = value.get("bytes")
    sha256 = value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0 or not _valid_sha256(sha256):
        raise ValueError(f"{label} identity is invalid")
    return {"bytes": size, "sha256": str(sha256)}


def _assert_identity(
    actual: Mapping[str, int | str],
    expected: Any,
    *,
    label: str,
) -> None:
    declaration = _declared_identity(expected, label=label)
    if dict(actual) != declaration:
        raise ValueError(f"{label} bytes/SHA-256 do not match the verified parent freeze")


def _resolve_input(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite path read-only using a percent-encoded file URI."""

    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _safe_output_path(root: Path, value: str | Path) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise ValueError(f"derived-corpus output has a symbolic-link ancestor: {cursor}")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    path = candidate.resolve(strict=False)
    if path == path.parent or not path.name:
        raise ValueError(f"unsafe derived-corpus output directory: {value}")
    if path == root or root.is_relative_to(path):
        raise ValueError(f"derived-corpus output may not contain the project root: {path}")
    if path.is_symlink():
        raise ValueError(f"derived-corpus output must not be a symbolic link: {path}")
    return path


def parse_group_definition(value: str) -> tuple[str, tuple[str, ...]]:
    """Parse ``OUTPUT_DIR=SOURCE[+SOURCE...]`` from the repeatable CLI option."""

    output, separator, raw_sources = value.partition("=")
    if not separator or not output.strip() or not raw_sources.strip():
        raise ValueError("group definitions must use OUTPUT_DIR=SOURCE[+SOURCE...] syntax")
    sources = tuple(part.strip() for part in raw_sources.split("+"))
    if any(
        not source
        or source != source.strip()
        or any(character.isspace() for character in source)
        or "=" in source
        for source in sources
    ):
        raise ValueError(f"invalid mixture_source in group definition {value!r}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"group definition repeats a mixture_source: {value!r}")
    return output.strip(), tuple(sorted(sources))


def normalize_group_definitions(
    groups: Mapping[str | Path, Sequence[str]],
    *,
    project_root: str | Path = ".",
) -> tuple[DerivedCorpusGroup, ...]:
    """Validate output/source ownership and return a deterministic group order.

    Unlike the compact CLI encoding, the direct mapping API permits ``+`` and ``=`` in source
    names because no delimiter parsing is involved.
    """

    root = Path(project_root).resolve()
    if not groups:
        raise ValueError("at least one derived-corpus group is required")
    normalized: list[DerivedCorpusGroup] = []
    source_owners: dict[str, Path] = {}
    output_paths: set[Path] = set()
    logical_names: set[str] = set()
    for raw_output, raw_sources in groups.items():
        output = _safe_output_path(root, raw_output)
        if output in output_paths:
            raise ValueError(f"derived-corpus output is repeated: {output}")
        output_paths.add(output)
        if isinstance(raw_sources, (str, bytes)) or not raw_sources:
            raise ValueError(f"derived-corpus group {output} has no mixture sources")
        sources = tuple(sorted(str(source) for source in raw_sources))
        if any(
            not source
            or source != source.strip()
            or any(character.isspace() for character in source)
            for source in sources
        ):
            raise ValueError(f"derived-corpus group {output} has an invalid mixture_source")
        if len(set(sources)) != len(sources):
            raise ValueError(f"derived-corpus group {output} repeats a mixture_source")
        logical_name = (
            output.relative_to(root).as_posix() if output.is_relative_to(root) else output.name
        )
        if logical_name in logical_names:
            raise ValueError(f"derived-corpus logical output name is repeated: {logical_name}")
        logical_names.add(logical_name)
        for source in sources:
            previous = source_owners.setdefault(source, output)
            if previous != output:
                raise ValueError(
                    f"mixture_source {source!r} belongs to multiple derived groups: "
                    f"{previous} and {output}"
                )
        normalized.append(DerivedCorpusGroup(output, sources, logical_name))

    ordered_outputs = sorted(output_paths, key=str)
    for index, left in enumerate(ordered_outputs):
        for right in ordered_outputs[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError(
                    f"derived-corpus outputs may not contain one another: {left} and {right}"
                )
    return tuple(sorted(normalized, key=lambda group: str(group.output_dir)))


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, int | str]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing, invalid, or a symbolic link: {path}")
    with path.open("rb") as handle:
        payload = handle.read(DEFAULT_MAX_RAW_DOCUMENT_BYTES + 1)
    if len(payload) > DEFAULT_MAX_RAW_DOCUMENT_BYTES:
        raise ValueError(f"{label} exceeds the audit size limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _parent_source_counts(
    manifest: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, int]:
    raw_counts = manifest.get("source_counts")
    frozen_counts = freeze.get("packed_corpus", {}).get("source_counts")
    if not isinstance(raw_counts, Mapping) or raw_counts != frozen_counts or not raw_counts:
        raise ValueError("parent source counts do not match the verified freeze")
    counts: dict[str, int] = {}
    for raw_name, count in raw_counts.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name.startswith("mixture:")
            or raw_name == "mixture:"
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError("parent source counts must be exact positive mixture_source counts")
        source = raw_name.removeprefix("mixture:")
        if source in counts:
            raise ValueError(f"parent source counts repeat mixture_source {source!r}")
        counts[source] = count
    total = manifest.get("total_documents")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("parent total_documents is invalid")
    if sum(counts.values()) != total:
        raise ValueError("parent source counts do not cover total_documents")
    return counts


def _load_parent_contract(
    *,
    freeze_path: Path,
    spec_path: Path,
    filtered_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[str, int],
    dict[str, Any],
    FrozenSplitAssignment,
    Path,
]:
    freeze = verify_corpus_freeze(
        freeze_path,
        spec_path,
        project_root=project_root,
    )
    if not isinstance(freeze, dict):
        raise ValueError("corpus freeze verifier returned an invalid record")

    recorded_freeze, freeze_identity = _read_json_object(
        freeze_path,
        label="parent corpus freeze",
    )
    if recorded_freeze != freeze:
        raise ValueError("parent corpus freeze changed after verification")
    spec_identity = _file_identity(spec_path, label="parent corpus freeze specification")
    _assert_identity(spec_identity, freeze.get("spec"), label="parent freeze specification")
    manifest, manifest_identity = _read_json_object(
        manifest_path,
        label="parent packed manifest",
    )
    packed = freeze.get("packed_corpus")
    if not isinstance(packed, Mapping):
        raise ValueError("verified parent freeze has no packed-corpus contract")
    frozen_manifest = packed.get("manifest")
    _assert_identity(manifest_identity, frozen_manifest, label="parent packed manifest")
    if not isinstance(frozen_manifest, Mapping) or frozen_manifest.get(
        "canonical_sha256"
    ) != _canonical_sha256(manifest):
        raise ValueError("parent packed manifest canonical identity does not match the freeze")

    contract = freeze.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("verified parent freeze has no contract")
    seq_len = contract.get("seq_len")
    vocab_size = contract.get("vocab_size")
    if (
        isinstance(seq_len, bool)
        or not isinstance(seq_len, int)
        or seq_len < 8
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise ValueError("verified parent freeze has an invalid sequence/tokenizer contract")
    if (
        manifest.get("seq_len") != seq_len
        or manifest.get("row_tokens") != seq_len + 1
        or manifest.get("vocab_size") != vocab_size
        or packed.get("seq_len") != seq_len
        or packed.get("vocab_size") != vocab_size
        or packed.get("generation") != manifest.get("generation")
        or packed.get("total_documents") != manifest.get("total_documents")
    ):
        raise ValueError("parent manifest sequence, vocabulary, generation, or count disagrees")

    frozen_assignment = load_frozen_split_assignment_manifest(manifest_path)
    split_contract = freeze.get("split_assignment")
    if not isinstance(split_contract, Mapping):
        raise ValueError("verified parent freeze has no split assignment")
    _assert_identity(
        {"bytes": frozen_assignment.bytes, "sha256": frozen_assignment.sha256},
        split_contract.get("artifact"),
        label="parent frozen split assignment",
    )
    split_val_fraction = split_contract.get("val_fraction")
    if (
        split_contract.get("assignment_sha256") != frozen_assignment.assignment_sha256
        or split_contract.get("records") != frozen_assignment.records
        or split_contract.get("seed") != frozen_assignment.seed
        or isinstance(split_val_fraction, bool)
        or not isinstance(split_val_fraction, (int, float))
        or float(split_val_fraction) != frozen_assignment.val_fraction
    ):
        raise ValueError("parent frozen split assignment disagrees with the freeze")

    tokenizer_contract = freeze.get("tokenizer")
    if not isinstance(tokenizer_contract, Mapping):
        raise ValueError("verified parent freeze has no tokenizer contract")
    kind = tokenizer_contract.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError("verified parent tokenizer kind is invalid")
    with _verified_regular_file_snapshot(
        tokenizer_path,
        tokenizer_contract.get("artifact"),
        label="parent tokenizer",
    ) as (tokenizer_snapshot, tokenizer_identity):
        tokenizer = load_tokenizer(kind, tokenizer_snapshot)
    if (
        int(tokenizer.vocab_size) != vocab_size
        or tokenizer_contract.get("vocab_size") != vocab_size
    ):
        raise ValueError("reused tokenizer vocabulary disagrees with the parent freeze")

    provenance = freeze.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("verified parent freeze has no source provenance")
    filtered_contract = _declared_identity(
        provenance.get("filtered_jsonl"),
        label="parent filtered JSONL",
    )
    if not filtered_path.is_file() or filtered_path.is_symlink():
        raise ValueError(f"parent filtered JSONL is missing or a symbolic link: {filtered_path}")
    preparation = manifest.get("preparation")
    if not isinstance(preparation, Mapping):
        raise ValueError("parent packed manifest has no preparation contract")
    staging_declaration = preparation.get("staging_database")
    if not isinstance(staging_declaration, Mapping):
        raise ValueError("parent packed manifest has no staging-database artifact")
    staging_value = staging_declaration.get("path")
    if not isinstance(staging_value, str) or not staging_value:
        raise ValueError("parent staging-database path is invalid")
    staging_path = _resolve_input(project_root, staging_value)
    staging_identity = _file_identity(
        staging_path,
        label="parent staging database",
    )
    _assert_identity(
        staging_identity,
        staging_declaration,
        label="parent staging database",
    )
    frozen_staging = provenance.get("staging_database")
    _assert_identity(
        staging_identity,
        frozen_staging,
        label="freeze-bound parent staging database",
    )
    if (
        staging_declaration.get("staging_version") != STAGING_VERSION
        or not isinstance(frozen_staging, Mapping)
        or frozen_staging.get("staging_version") != STAGING_VERSION
    ):
        raise ValueError("parent staging-database version disagrees with the freeze")
    DiskBackedCorpus(staging_path)

    source_counts = _parent_source_counts(manifest, freeze)
    parent_identity = {
        "freeze": {
            "artifact": freeze_identity,
            "canonical_sha256": _canonical_sha256(freeze),
            "freeze_sha256": freeze.get("freeze_sha256"),
            "format": freeze.get("format"),
            "schema_version": freeze.get("schema_version"),
        },
        "freeze_specification": spec_identity,
        "filtered_jsonl": filtered_contract,
        "packed_manifest": {
            **manifest_identity,
            "canonical_sha256": frozen_manifest["canonical_sha256"],
            "generation": manifest.get("generation"),
        },
        "split_assignment": {
            "artifact": {
                "bytes": frozen_assignment.bytes,
                "sha256": frozen_assignment.sha256,
            },
            "assignment_sha256": frozen_assignment.assignment_sha256,
            "records": frozen_assignment.records,
            "seed": frozen_assignment.seed,
            "val_fraction": frozen_assignment.val_fraction,
        },
        "tokenizer": {
            "kind": kind,
            "vocab_size": vocab_size,
            "artifact": tokenizer_identity,
        },
        "staging_database": {
            "artifact": staging_identity,
            "staging_version": STAGING_VERSION,
        },
    }
    return (
        freeze,
        manifest,
        tokenizer,
        source_counts,
        parent_identity,
        frozen_assignment,
        staging_path,
    )


def _create_assignment_index(
    path: Path,
    assignment: FrozenSplitAssignment,
) -> sqlite3.Connection:
    if assignment.path.is_symlink():
        raise ValueError("parent split assignment changed to a symbolic link")
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE assignments (
            identity TEXT NOT NULL,
            document_sha256 TEXT NOT NULL,
            document_id TEXT NOT NULL,
            split TEXT NOT NULL,
            seen INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (identity, document_sha256)
        ) WITHOUT ROWID
        """
    )
    digest = hashlib.sha256()
    artifact_digest = hashlib.sha256()
    artifact_bytes = 0
    first = True
    records = 0
    previous_key: tuple[str, str] | None = None
    previous_identity = ""
    previous_identity_split = ""
    with assignment.path.open("rb") as handle:
        header_line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
        if len(header_line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
            raise ValueError("parent split assignment header is too large")
        artifact_digest.update(header_line)
        artifact_bytes += len(header_line)
        try:
            header = json.loads(header_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("parent split assignment has an invalid header") from error
        if header != {
            "format": SPLIT_ASSIGNMENT_FORMAT,
            "schema_version": SPLIT_ASSIGNMENT_VERSION,
        }:
            raise ValueError("parent split assignment header is unsupported")
        line_no = 1
        while True:
            raw_line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
            if not raw_line:
                break
            line_no += 1
            if len(raw_line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
                raise ValueError(f"parent split assignment row {line_no} is too large")
            artifact_digest.update(raw_line)
            artifact_bytes += len(raw_line)
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"parent split assignment row {line_no} is invalid") from error
            if not isinstance(row, dict) or set(row) != {
                "document_id",
                "document_sha256",
                "identity_sha256",
                "split",
            }:
                raise ValueError(f"parent split assignment row {line_no} is not canonical")
            document_id = row["document_id"]
            text_sha256 = row["document_sha256"]
            identity = row["identity_sha256"]
            split = row["split"]
            if (
                not isinstance(document_id, str)
                or not document_id
                or not _valid_sha256(text_sha256)
                or not _valid_sha256(identity)
                or identity != hashlib.sha256(document_id.encode("utf-8")).hexdigest()
                or split not in {"train", "val"}
            ):
                raise ValueError(f"parent split assignment row {line_no} has an invalid binding")
            key = (identity, text_sha256)
            if previous_key is not None and key <= previous_key:
                raise ValueError("parent split assignment rows are not unique and sorted")
            previous_key = key
            if identity != previous_identity:
                previous_identity = identity
                previous_identity_split = split
            elif previous_identity_split != split:
                raise ValueError("one parent document identity belongs to multiple splits")
            connection.execute(
                """
                INSERT INTO assignments (
                    identity, document_sha256, document_id, split
                ) VALUES (?, ?, ?, ?)
                """,
                (identity, text_sha256, document_id, split),
            )
            value = f"{identity}:{text_sha256}:{split}"
            if not first:
                digest.update(b"\n")
            digest.update(value.encode("ascii"))
            first = False
            records += 1
    if (
        artifact_bytes != assignment.bytes
        or artifact_digest.hexdigest() != assignment.sha256
        or records != assignment.records
        or digest.hexdigest() != assignment.assignment_sha256
    ):
        raise ValueError("parent split assignment content disagrees with its manifest")
    connection.commit()
    return connection


def _create_group_stage(
    definition: DerivedCorpusGroup,
    *,
    expected_documents: int,
) -> _GroupStage:
    output = definition.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.derived.",
            dir=output.parent,
        )
    )
    database = temporary / "derived-staging.sqlite3"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA temp_store=FILE")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE documents (
                digest TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                license TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                raw_text_sha TEXT NOT NULL,
                identity TEXT NOT NULL,
                decontaminated INTEGER NOT NULL,
                near_keep INTEGER NOT NULL,
                near_rank INTEGER NOT NULL,
                split TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
    except BaseException:
        try:
            if connection is not None:
                connection.close()
        finally:
            shutil.rmtree(temporary)
        raise
    assert connection is not None
    return _GroupStage(
        definition=definition,
        temporary_output=temporary,
        database_path=database,
        connection=connection,
        expected_documents=expected_documents,
    )


def _decode_canonical_document(raw_line: bytes, *, line_no: int) -> CorpusDocument:
    try:
        row = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"parent filtered JSONL row {line_no} is invalid UTF-8 JSON") from error
    if not isinstance(row, dict) or set(row) != {
        "doc_id",
        "license",
        "meta",
        "source",
        "text",
    }:
        raise ValueError(f"parent filtered JSONL row {line_no} is not a CorpusDocument")
    if (
        not isinstance(row["text"], str)
        or not isinstance(row["source"], str)
        or not isinstance(row["doc_id"], str)
        or not row["doc_id"]
        or not isinstance(row["license"], str)
        or not isinstance(row["meta"], dict)
    ):
        raise ValueError(f"parent filtered JSONL row {line_no} has invalid field types")
    document = CorpusDocument(
        text=row["text"],
        source=row["source"],
        doc_id=row["doc_id"],
        license=row["license"],
        meta=dict(row["meta"]),
    )
    try:
        canonical = _canonical_json_bytes(asdict(document)) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"parent filtered JSONL row {line_no} is not canonical") from error
    if raw_line != canonical:
        raise ValueError(f"parent filtered JSONL row {line_no} is not canonical")
    return document


def _iter_canonical_documents(
    path: Path,
    *,
    expected_identity: Mapping[str, int | str],
) -> Iterator[tuple[int, CorpusDocument]]:
    """Decode the exact parent filtered stream once while verifying its content identity."""

    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as handle:
        line_no = 0
        while True:
            raw_line = handle.readline(DEFAULT_MAX_RAW_DOCUMENT_BYTES + 1)
            if not raw_line:
                break
            line_no += 1
            if len(raw_line) > DEFAULT_MAX_RAW_DOCUMENT_BYTES:
                raise ValueError(
                    f"parent filtered JSONL row {line_no} exceeds the record size limit"
                )
            digest.update(raw_line)
            total_bytes += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"parent filtered JSONL row {line_no} is blank")
            yield line_no, _decode_canonical_document(raw_line, line_no=line_no)
    actual = {"bytes": total_bytes, "sha256": digest.hexdigest()}
    if path.is_symlink():
        raise ValueError("parent filtered JSONL changed to a symbolic link")
    if actual != dict(expected_identity):
        raise ValueError("parent filtered JSONL bytes/SHA-256 changed after freeze verification")


def _insert_group_document(
    stage: _GroupStage,
    document: CorpusDocument,
    *,
    identity: str,
    text_sha256: str,
    split: str,
    order: int,
) -> None:
    try:
        meta_json = _canonical_json_bytes(document.meta).decode("utf-8")
        stage.connection.execute(
            """
            INSERT INTO documents (
                digest, text, source, doc_id, license, meta_json, raw_text_sha,
                identity, decontaminated, near_keep, near_rank, split
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
            """,
            (
                f"{identity}:{text_sha256}",
                document.text,
                document.source,
                document.doc_id,
                document.license,
                meta_json,
                text_sha256,
                identity,
                order,
                split,
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"parent filtered JSONL repeats document binding {document.doc_id!r}"
        ) from error
    stage.observed_documents += 1


def _assert_parent_staging_row(
    row: sqlite3.Row | None,
    document: CorpusDocument,
    *,
    identity: str,
    text_sha256: str,
    split: str,
    order: int,
) -> None:
    if row is None:
        raise ValueError("parent filtered JSONL has more rows than its frozen staging database")
    expected_meta = _canonical_json_bytes(document.meta).decode("utf-8")
    if (
        str(row["text"]) != document.text
        or str(row["source"]) != document.source
        or str(row["doc_id"]) != document.doc_id
        or str(row["license"]) != document.license
        or str(row["meta_json"]) != expected_meta
        or str(row["raw_text_sha"]) != text_sha256
        or str(row["identity"]) != identity
        or str(row["split"]) != split
    ):
        raise ValueError(
            f"parent filtered row {order} was relabeled or differs from frozen staging provenance"
        )


def _staging_assignment_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    first = True
    for row in connection.execute(
        """
        SELECT identity, raw_text_sha, split
        FROM documents
        ORDER BY identity, raw_text_sha
        """
    ):
        if not first:
            digest.update(b"\n")
        digest.update(f"{row[0]}:{row[1]}:{row[2]}".encode("ascii"))
        first = False
    return digest.hexdigest()


def _stage_metadata(
    stage: _GroupStage,
    *,
    freeze: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    parent_identity: Mapping[str, Any],
    source_expected_counts: Mapping[str, int],
    all_group_mapping: Mapping[str, Sequence[str]],
    require_complete_parent: bool,
) -> dict[str, Any]:
    split_counts = {
        str(row["split"]): int(row["documents"])
        for row in stage.connection.execute(
            "SELECT split, COUNT(*) AS documents FROM documents GROUP BY split"
        )
    }
    split_counts = {split: split_counts.get(split, 0) for split in ("train", "val")}
    assignment_sha256 = _staging_assignment_sha256(stage.connection)
    group_counts = {
        source: int(source_expected_counts[source]) for source in stage.definition.mixture_sources
    }
    expected_group_documents = {
        output: sum(int(source_expected_counts[source]) for source in sources)
        for output, sources in all_group_mapping.items()
    }
    assigned_sources = {source for sources in all_group_mapping.values() for source in sources}
    no_rescreen = {
        "performed": False,
        "rationale": (
            "Rows are an identity-checked partition of the retained parent corpus. Re-running "
            "evaluation denylist screening or near-deduplication would create a different corpus "
            "and could change the frozen document split."
        ),
    }
    provenance = {
        "format": DERIVED_CORPUS_FORMAT,
        "schema_version": DERIVED_CORPUS_SCHEMA_VERSION,
        "parent": dict(parent_identity),
        "group": {
            "logical_output_directory": stage.definition.logical_name,
            "mixture_sources": list(stage.definition.mixture_sources),
            "expected_source_documents": group_counts,
            "expected_documents": stage.expected_documents,
            "observed_documents": stage.observed_documents,
            "splits": split_counts,
            "assignment_sha256": assignment_sha256,
        },
        "fanout": {
            "contract": ONE_PASS_FANOUT_CONTRACT,
            "parent_filtered_decodes": 1,
            "dispatch_key": "meta.mixture_source",
            "group_mapping": {
                output: list(sources) for output, sources in sorted(all_group_mapping.items())
            },
            "expected_parent_source_documents": dict(sorted(source_expected_counts.items())),
            "expected_group_documents": dict(sorted(expected_group_documents.items())),
            "unassigned_parent_sources": sorted(set(source_expected_counts) - assigned_sources),
            "group_disjoint": True,
            "parent_order_preserved": True,
            "complete_parent_required_by_invocation": require_complete_parent,
        },
        "publication": {
            "coordination": "cooperative_exclusive_lock_files",
            "lock_contract": (
                "Cooperating publishers never replace or remove another publisher's held "
                "lock pathname."
            ),
            "commit": "no_replace_manifests_last_replayable",
            "crash_atomic_multi_group": False,
            "failure_state": (
                "Failures never delete public destination names and may leave complete groups or "
                "exact manifestless subsets for deterministic replay."
            ),
            "interrupted_state": (
                "SIGKILL or power loss can additionally leave stale cooperative locks."
            ),
            "recovery": (
                "After confirming no publisher is alive, manually remove stale lock files and "
                "rerun the same deterministic command to repair exact manifestless subsets."
            ),
        },
        "inherited_decontamination_audit": freeze.get("decontamination"),
        "no_rescreen": no_rescreen,
    }
    parent_audit = parent_manifest.get("corpus_audit")
    if not isinstance(parent_audit, Mapping):
        raise ValueError("parent packed manifest has no corpus audit to inherit")
    corpus_audit = {
        "quality_and_exact_deduplication": {
            "mode": "inherited_verified_parent_freeze",
            "rerun": False,
            "parent_audit": freeze.get("quality_and_exact_deduplication"),
        },
        "evaluation_decontamination": {
            "mode": "inherited_verified_parent_freeze",
            "rerun": False,
            "parent_audit": freeze.get("decontamination"),
            "no_rescreen": no_rescreen,
        },
        "near_deduplication": {
            "mode": "inherited_verified_parent_manifest",
            "rerun": False,
            "parent_audit": parent_audit.get("near_deduplication"),
        },
        "split_assignment": {
            "mode": "inherited_verified_parent_split",
            "assignment_sha256": assignment_sha256,
            "splits": split_counts,
        },
        "derived_fanout": provenance["fanout"],
    }
    staging_config = {
        "seed": parent_manifest["seed"],
        "val_fraction": parent_manifest["val_fraction"],
        "split_assignment_mode": "inherited_verified_parent_split",
        "filtering_mode": "none_parent_retained_rows_only",
        "near_dedup": False,
        "evaluation_rescreen": False,
    }
    stage.connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        (
            ("staging_version", json.dumps(STAGING_VERSION)),
            (
                "corpus_audit",
                _canonical_json_bytes(corpus_audit).decode("utf-8"),
            ),
            (
                "staging_config",
                _canonical_json_bytes(staging_config).decode("utf-8"),
            ),
        ),
    )
    stage.connection.commit()
    return provenance


def _rewrite_generation(
    output: Path,
    manifest: dict[str, Any],
    *,
    deterministic_generation: str,
) -> None:
    old_generation = manifest.get("generation")
    if not isinstance(old_generation, str) or len(old_generation) != 32:
        raise ValueError("derived packer returned an invalid generation")
    old_directory = output / "generations" / old_generation
    new_directory = output / "generations" / deterministic_generation
    if old_directory != new_directory:
        if new_directory.exists():
            raise ValueError("deterministic derived generation already exists")
        old_directory.replace(new_directory)
    old_prefix = f"generations/{old_generation}/"
    new_prefix = f"generations/{deterministic_generation}/"
    assignment = manifest.get("split_assignment")
    if not isinstance(assignment, dict) or not isinstance(assignment.get("path"), str):
        raise ValueError("derived manifest has no split assignment")
    if not assignment["path"].startswith(old_prefix):
        raise ValueError("derived split assignment is outside its generation")
    assignment["path"] = new_prefix + assignment["path"][len(old_prefix) :]
    for split in ("train", "val"):
        split_manifest = manifest.get("splits", {}).get(split)
        if not isinstance(split_manifest, dict):
            raise ValueError(f"derived manifest has no {split!r} split")
        shards = split_manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"derived manifest {split!r} shards are invalid")
        for shard in shards:
            if not isinstance(shard, dict):
                raise ValueError("derived manifest has an invalid shard entry")
            for key in ("tokens", "lengths"):
                value = shard.get(key)
                if not isinstance(value, str) or not value.startswith(old_prefix):
                    raise ValueError("derived shard is outside its generation")
                shard[key] = new_prefix + value[len(old_prefix) :]
    manifest["generation"] = deterministic_generation


def _write_final_manifest(
    stage: _GroupStage,
    manifest: dict[str, Any],
    *,
    provenance: dict[str, Any],
    tokenizer_identity: Mapping[str, Any],
    rows_per_shard: int,
) -> dict[str, Any]:
    assignment_sha256 = str(provenance["group"]["assignment_sha256"])
    generation_payload = {
        "format": DERIVED_CORPUS_FORMAT,
        "schema_version": DERIVED_CORPUS_SCHEMA_VERSION,
        "parent_freeze_sha256": provenance["parent"]["freeze"]["freeze_sha256"],
        "parent_manifest_sha256": provenance["parent"]["packed_manifest"]["sha256"],
        "parent_split_sha256": provenance["parent"]["split_assignment"]["artifact"]["sha256"],
        "tokenizer_sha256": tokenizer_identity["artifact"]["sha256"],
        "seq_len": manifest["seq_len"],
        "rows_per_shard": rows_per_shard,
        "group_sources": provenance["group"]["mixture_sources"],
        "group_counts": provenance["group"]["expected_source_documents"],
        "assignment_sha256": assignment_sha256,
        "fanout_mapping": provenance["fanout"]["group_mapping"],
        "complete_parent_required": provenance["fanout"]["complete_parent_required_by_invocation"],
    }
    deterministic_generation = _canonical_sha256(generation_payload)[:32]
    _rewrite_generation(
        stage.temporary_output,
        manifest,
        deterministic_generation=deterministic_generation,
    )
    preparation = manifest.get("preparation")
    if not isinstance(preparation, dict):
        raise ValueError("derived manifest has no preparation provenance")
    staging_artifact = _file_identity(
        stage.database_path,
        label="derived staging database",
    )
    preparation["staging_database"] = {
        "path": stage.database_path.name,
        **staging_artifact,
        "staging_version": STAGING_VERSION,
    }
    preparation["mode"] = "parent_freeze_derived_sqlite_view"
    preparation["provenance"] = provenance
    manifest["derived_corpus"] = provenance
    tokenizer_artifact = tokenizer_identity["artifact"]
    manifest["tokenizer"] = {
        "kind": tokenizer_identity["kind"],
        "vocab_size": tokenizer_identity["vocab_size"],
        "bytes": tokenizer_artifact["bytes"],
        "sha256": tokenizer_artifact["sha256"],
        "reuse": "exact_parent_freeze_tokenizer",
    }
    manifest_path = stage.temporary_output / "manifest.json"
    temporary = stage.temporary_output / ".manifest.final.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(manifest_path)
    return manifest


def _verify_child_assignment(
    stage: _GroupStage,
    manifest: Mapping[str, Any],
) -> None:
    frozen = load_frozen_split_assignment_manifest(stage.temporary_output / "manifest.json")
    with closing(sqlite3.connect(stage.database_path)) as connection:
        expected_assignment = _staging_assignment_sha256(connection)
    if (
        frozen.records != stage.expected_documents
        or frozen.assignment_sha256 != expected_assignment
        or manifest.get("split_assignment_sha256") != expected_assignment
    ):
        raise ValueError("derived child split membership does not match the parent partition")
    with closing(sqlite3.connect(stage.database_path)) as connection:
        for split in ("train", "val"):
            expected = int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE split = ?",
                    (split,),
                ).fetchone()[0]
            )
            actual = manifest.get("splits", {}).get(split, {}).get("documents")
            if actual != expected:
                raise ValueError(
                    f"derived child {split!r} membership count does not match its staging view"
                )
    declared_counts = (
        manifest.get("derived_corpus", {}).get("group", {}).get("expected_source_documents")
    )
    if not isinstance(declared_counts, Mapping):
        raise ValueError("derived child has no declared source counts")
    expected_sources = {
        f"mixture:{source}": int(declared_counts[source])
        for source in stage.definition.mixture_sources
    }
    if manifest.get("source_counts") != expected_sources:
        raise ValueError("derived child source counts do not match its declared group")


def _tree_identity(root: Path) -> list[dict[str, int | str]]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"derived output is not a regular directory: {root}")
    rows: list[dict[str, int | str]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = str(path.relative_to(root))
        if stat.S_ISDIR(metadata.st_mode):
            rows.append({"path": relative, "kind": "directory"})
        elif stat.S_ISREG(metadata.st_mode):
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    **_file_identity(path, label="derived output artifact"),
                }
            )
        else:
            raise ValueError(f"derived output contains a symlink or special node: {path}")
    return rows


def _inode_identity(metadata: os.stat_result) -> _InodeIdentity:
    return _InodeIdentity(device=int(metadata.st_dev), inode=int(metadata.st_ino))


def _open_directory_at(parent_fd: int, name: str, path: Path, *, owned: bool) -> _PinnedDirectory:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(
            f"derived publication directory is invalid or was swapped: {path}"
        ) from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"derived publication path is not a directory: {path}")
    return _PinnedDirectory(
        path=path,
        name=name,
        parent_fd=parent_fd,
        fd=descriptor,
        identity=_inode_identity(metadata),
        owned=owned,
    )


def _file_identity_at(
    parent_fd: int,
    name: str,
    *,
    label: str,
) -> tuple[_InodeIdentity, dict[str, int | str]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{label} is missing, invalid, or a symbolic link") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        return _inode_identity(metadata), {"bytes": size, "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)


def _begin_publication(stage: _GroupStage, *, preexisting: bool) -> _PublicationState:
    output = stage.definition.output_dir
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(output.parent, flags)
    created_identity: _InodeIdentity | None = None
    try:
        if not preexisting:
            try:
                os.mkdir(output.name, dir_fd=parent_fd)
            except FileExistsError as error:
                raise ValueError(f"derived output appeared during publication: {output}") from error
            created = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(created.st_mode):
                raise ValueError(f"derived output reservation is not a directory: {output}")
            created_identity = _inode_identity(created)
        root = _open_directory_at(
            parent_fd,
            output.name,
            output,
            owned=not preexisting,
        )
        if created_identity is not None and root.identity != created_identity:
            os.close(root.fd)
            raise ValueError(f"derived output reservation was swapped: {output}")
        return _PublicationState(
            stage=stage,
            parent_fd=parent_fd,
            directories=[root],
            files=[],
        )
    except BaseException:
        # Never unlink/rmdir a public pathname after a separate identity check: a
        # non-cooperating writer could replace the name between those operations.
        os.close(parent_fd)
        raise


def _ensure_publication_directory(
    state: _PublicationState,
    parent: _PinnedDirectory,
    source: Path,
) -> _PinnedDirectory:
    target = parent.path / source.name
    created_identity: _InodeIdentity | None = None
    owned = False
    try:
        os.mkdir(source.name, dir_fd=parent.fd)
        metadata = os.stat(source.name, dir_fd=parent.fd, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"derived publication directory is invalid: {target}")
        created_identity = _inode_identity(metadata)
        owned = True
    except FileExistsError:
        pass
    try:
        pinned = _open_directory_at(parent.fd, source.name, target, owned=owned)
    except BaseException:
        # Keep a safely replayable empty directory rather than risk deleting a
        # concurrently substituted public pathname.
        raise
    if created_identity is not None and pinned.identity != created_identity:
        os.close(pinned.fd)
        raise ValueError(f"derived publication directory was swapped: {target}")
    state.directories.append(pinned)
    return pinned


def _ensure_publication_file(
    state: _PublicationState,
    parent: _PinnedDirectory,
    source: Path,
) -> None:
    expected = _file_identity(source, label="private derived publication artifact")
    source_metadata = source.lstat()
    if not stat.S_ISREG(source_metadata.st_mode):
        raise ValueError(f"private derived publication artifact is not a regular file: {source}")
    source_inode = _inode_identity(source_metadata)
    created = False
    try:
        os.link(
            source,
            source.name,
            dst_dir_fd=parent.fd,
            follow_symlinks=False,
        )
        created = True
        state.files.append(
            _OwnedFile(
                path=parent.path / source.name,
                name=source.name,
                parent_fd=parent.fd,
                identity=source_inode,
                bytes=int(expected["bytes"]),
                sha256=str(expected["sha256"]),
            )
        )
    except FileExistsError:
        pass
    actual_inode, actual = _file_identity_at(
        parent.fd,
        source.name,
        label=f"derived publication artifact {parent.path / source.name}",
    )
    if actual != expected or (created and actual_inode != source_inode):
        raise ValueError(
            f"derived publication artifact differs or was swapped: {parent.path / source.name}"
        )


def _populate_publication_directory(
    state: _PublicationState,
    source: Path,
    target: _PinnedDirectory,
    *,
    skip_manifest: bool,
) -> None:
    for artifact in sorted(source.iterdir()):
        if skip_manifest and artifact.name == "manifest.json":
            continue
        metadata = artifact.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            child = _ensure_publication_directory(state, target, artifact)
            _populate_publication_directory(
                state,
                artifact,
                child,
                skip_manifest=False,
            )
        elif stat.S_ISREG(metadata.st_mode):
            _ensure_publication_file(state, target, artifact)
        else:
            raise ValueError(
                f"private derived output contains a symlink or special node: {artifact}"
            )


def _close_publication(state: _PublicationState) -> None:
    for directory in reversed(state.directories):
        try:
            os.close(directory.fd)
        except OSError:
            pass
    try:
        os.close(state.parent_fd)
    except OSError:
        pass


def _abandon_publication(state: _PublicationState) -> None:
    """Close pinned descriptors without deleting any public destination pathname.

    Conditional unlink/rmdir is not portable: verifying an inode and deleting its name are
    separate operations, so a non-cooperating writer could substitute foreign content in between.
    No-replace publication leaves only complete groups or exact manifestless subsets, both of
    which deterministic replay can recognize and finish.
    """

    _close_publication(state)


def _retire_owned_lock(lock_path: Path, identity: _InodeIdentity) -> None:
    """Remove an owned lock without unlinking its public pathname.

    The public entry is first moved into a fresh private quarantine directory. If a
    non-cooperating writer substituted another inode after the ownership check, those foreign
    bytes are preserved in quarantine and restored at the public name with a no-replace hard link
    when the node type permits it.
    """

    try:
        metadata = lock_path.lstat()
    except OSError:
        return
    if not stat.S_ISREG(metadata.st_mode) or _inode_identity(metadata) != identity:
        return

    try:
        quarantine = Path(
            tempfile.mkdtemp(
                prefix=f".{lock_path.name}.cleanup.",
                dir=lock_path.parent,
            )
        )
    except OSError:
        return
    candidate = quarantine / "candidate"
    try:
        os.rename(lock_path, candidate)
    except OSError:
        try:
            quarantine.rmdir()
        except OSError:
            pass
        return

    try:
        moved = candidate.lstat()
    except OSError:
        return
    if stat.S_ISREG(moved.st_mode) and _inode_identity(moved) == identity:
        try:
            candidate.unlink()
            quarantine.rmdir()
        except OSError:
            # The public lock name is already gone. Preserve any raced quarantine entry.
            pass
        return

    # A foreign replacement won the check-to-move race. Never delete it. Restoring via link is
    # no-replace; if another entry has appeared, quarantine remains for manual recovery.
    try:
        os.link(candidate, lock_path, follow_symlinks=False)
    except OSError:
        pass


@contextmanager
def _publication_locks(stages: Sequence[_GroupStage]) -> Iterator[None]:
    """Acquire ephemeral locks under a strictly cooperative lock-file contract.

    Cooperating publishers never replace or remove another publisher's held lock pathname. That
    rule is what makes checked cleanup of an owned lock possible after setup failures; public
    corpus destinations themselves make no such cooperation assumption and are never deleted.
    """

    acquired: list[tuple[Path, int, _InodeIdentity]] = []
    try:
        for stage in sorted(stages, key=lambda item: str(item.definition.output_dir)):
            output = stage.definition.output_dir
            lock_path = output.parent / f".{output.name}.derived-corpus.lock"
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as error:
                raise ValueError(
                    f"derived output is locked by another publisher: {output}"
                ) from error
            metadata = os.fstat(descriptor)
            acquired.append((lock_path, descriptor, _inode_identity(metadata)))
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
        yield
    finally:
        for lock_path, descriptor, identity in reversed(acquired):
            try:
                os.close(descriptor)
            except OSError:
                pass
            _retire_owned_lock(lock_path, identity)


def _publish_groups(stages: Sequence[_GroupStage]) -> set[Path]:
    """Use cooperative locks and no-replace hard links, publishing each manifest last.

    Failures never delete a public destination name; they can leave complete groups or exact
    manifestless subsets. SIGKILL or power loss is not a crash-atomic multi-group commit and may
    additionally leave a stale cooperative lock. After manually confirming/removing a stale lock,
    deterministic replay completes exact subsets without overwriting them.
    """

    with _publication_locks(stages):
        expected_trees = {
            stage.definition.output_dir: _tree_identity(stage.temporary_output) for stage in stages
        }
        expected_maps = {
            output: {str(row["path"]): row for row in rows}
            for output, rows in expected_trees.items()
        }
        preexisting: set[Path] = set()
        complete: set[Path] = set()
        repairable: set[Path] = set()
        for stage in stages:
            output = stage.definition.output_dir
            if output.exists():
                preexisting.add(output)
                actual_rows = _tree_identity(output)
                if actual_rows == expected_trees[output]:
                    complete.add(output)
                    continue
                actual_map = {str(row["path"]): row for row in actual_rows}
                if "manifest.json" not in actual_map and all(
                    expected_maps[output].get(path) == row for path, row in actual_map.items()
                ):
                    repairable.add(output)
                else:
                    raise ValueError(
                        f"existing derived output differs from the verified replay: {output}"
                    )

        states: list[_PublicationState] = []
        try:
            for stage in stages:
                output = stage.definition.output_dir
                if output in complete:
                    continue
                state = _begin_publication(stage, preexisting=output in repairable)
                states.append(state)
                _populate_publication_directory(
                    state,
                    stage.temporary_output,
                    state.directories[0],
                    skip_manifest=True,
                )

            for stage in stages:
                output = stage.definition.output_dir
                if output in complete and _tree_identity(output) != expected_trees[output]:
                    raise ValueError(
                        f"existing derived output changed during publication: {output}"
                    )

            for state in states:
                _ensure_publication_file(
                    state,
                    state.directories[0],
                    state.stage.temporary_output / "manifest.json",
                )

            for stage in stages:
                output = stage.definition.output_dir
                if _tree_identity(output) != expected_trees[output]:
                    raise ValueError(f"published derived output identity mismatch: {output}")
        except BaseException:
            for state in reversed(states):
                _abandon_publication(state)
            raise
        for state in states:
            _close_publication(state)
        return preexisting


def prepare_derived_corpora(
    *,
    freeze_path: str | Path,
    spec_path: str | Path,
    parent_filtered_jsonl: str | Path,
    parent_manifest: str | Path,
    tokenizer_path: str | Path,
    groups: Mapping[str | Path, Sequence[str]],
    project_root: str | Path = ".",
    rows_per_shard: int = 2048,
    require_complete_parent: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build disjoint child shard corpora from one verified parent retained-document stream.

    All children are packed in private sibling directories first. Existing byte-identical outputs
    are accepted as deterministic replays; an existing differing output aborts the whole publish.
    Publication uses cooperative lock files, no-replace artifact links, and manifests last.
    Failures never delete public destination names and can leave complete groups or exact
    manifestless subsets. SIGKILL or power loss is not a crash-atomic multi-group commit and can
    additionally leave stale locks. After confirming no publisher is alive, remove stale locks
    manually and rerun the same deterministic command to repair an exact partial group.
    """

    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    root = Path(project_root).resolve()
    freeze_file = _resolve_input(root, freeze_path)
    spec_file = _resolve_input(root, spec_path)
    filtered_file = _resolve_input(root, parent_filtered_jsonl)
    manifest_file = _resolve_input(root, parent_manifest)
    tokenizer_file = _resolve_input(root, tokenizer_path)
    definitions = normalize_group_definitions(groups, project_root=root)
    (
        freeze,
        manifest,
        tokenizer,
        source_expected,
        parent_identity,
        frozen_assignment,
        parent_staging_path,
    ) = _load_parent_contract(
        freeze_path=freeze_file,
        spec_path=spec_file,
        filtered_path=filtered_file,
        manifest_path=manifest_file,
        tokenizer_path=tokenizer_file,
        project_root=root,
    )

    owners = {
        source: definition for definition in definitions for source in definition.mixture_sources
    }
    unknown_configured = sorted(set(owners) - set(source_expected))
    if unknown_configured:
        raise ValueError(
            "derived groups name mixture sources absent from the parent freeze: "
            + ", ".join(unknown_configured)
        )
    missing_sources = sorted(set(source_expected) - set(owners))
    if require_complete_parent and missing_sources:
        raise ValueError(
            "derived groups do not assign every parent mixture_source: "
            + ", ".join(missing_sources)
        )

    expected_by_output = {
        definition.output_dir: sum(source_expected[source] for source in definition.mixture_sources)
        for definition in definitions
    }
    if any(count <= 0 for count in expected_by_output.values()):
        raise ValueError("every derived group must have a positive parent document count")
    all_group_mapping = {
        definition.logical_name: definition.mixture_sources for definition in definitions
    }
    stages: list[_GroupStage] = []
    manifests: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="localagent-derived-fanout-") as control_dir:
        assignment_connection = _create_assignment_index(
            Path(control_dir) / "parent-assignment.sqlite3",
            frozen_assignment,
        )
        parent_staging_connection: sqlite3.Connection | None = None
        try:
            parent_staging_connection = _open_sqlite_read_only(parent_staging_path)
            parent_staging_connection.row_factory = sqlite3.Row
            parent_staging_rows = parent_staging_connection.execute(
                """
                SELECT text, source, doc_id, license, meta_json, raw_text_sha,
                       identity, split
                FROM documents
                WHERE decontaminated = 1 AND near_keep = 1
                ORDER BY near_rank
                """
            )
            for definition in definitions:
                stages.append(
                    _create_group_stage(
                        definition,
                        expected_documents=expected_by_output[definition.output_dir],
                    )
                )
            stage_by_output = {stage.definition.output_dir: stage for stage in stages}
            observed_sources: Counter[str] = Counter()
            seen_documents = 0
            expected_filtered_identity = parent_identity["filtered_jsonl"]
            for order, document in _iter_canonical_documents(
                filtered_file,
                expected_identity=expected_filtered_identity,
            ):
                seen_documents += 1
                mixture_source = document.meta.get("mixture_source")
                if not isinstance(mixture_source, str) or not mixture_source:
                    raise ValueError(
                        f"parent filtered row {order} has no protected meta.mixture_source"
                    )
                if mixture_source not in source_expected:
                    raise ValueError(
                        f"parent filtered row {order} has unknown mixture_source {mixture_source!r}"
                    )
                observed_sources[mixture_source] += 1
                identity = hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()
                text_sha256 = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
                assignment_row = assignment_connection.execute(
                    """
                    SELECT document_id, split, seen
                    FROM assignments
                    WHERE identity = ? AND document_sha256 = ?
                    """,
                    (identity, text_sha256),
                ).fetchone()
                if assignment_row is None or assignment_row["document_id"] != document.doc_id:
                    raise ValueError(
                        f"parent filtered row {order} is not bound to the frozen split"
                    )
                if int(assignment_row["seen"]) != 0:
                    raise ValueError(
                        f"parent filtered row {order} repeats a frozen document binding"
                    )
                _assert_parent_staging_row(
                    parent_staging_rows.fetchone(),
                    document,
                    identity=identity,
                    text_sha256=text_sha256,
                    split=str(assignment_row["split"]),
                    order=order,
                )
                assignment_connection.execute(
                    """
                    UPDATE assignments
                    SET seen = 1
                    WHERE identity = ? AND document_sha256 = ?
                    """,
                    (identity, text_sha256),
                )
                owner = owners.get(mixture_source)
                if owner is not None:
                    _insert_group_document(
                        stage_by_output[owner.output_dir],
                        document,
                        identity=identity,
                        text_sha256=text_sha256,
                        split=str(assignment_row["split"]),
                        order=order,
                    )
            if parent_staging_rows.fetchone() is not None:
                raise ValueError(
                    "parent filtered JSONL is truncated relative to frozen staging provenance"
                )
            _assert_identity(
                _file_identity(
                    parent_staging_path,
                    label="parent staging database",
                ),
                parent_identity["staging_database"]["artifact"],
                label="parent staging database",
            )
            _assert_identity(
                _file_identity(
                    manifest_file,
                    label="parent packed manifest",
                ),
                parent_identity["packed_manifest"],
                label="parent packed manifest",
            )
            assignment_connection.commit()
            total_documents = manifest.get("total_documents")
            if seen_documents != total_documents or seen_documents != frozen_assignment.records:
                raise ValueError(
                    "parent filtered JSONL document count does not match the freeze/manifest"
                )
            unseen = int(
                assignment_connection.execute(
                    "SELECT COUNT(*) FROM assignments WHERE seen = 0"
                ).fetchone()[0]
            )
            if unseen:
                raise ValueError(
                    "parent filtered JSONL does not cover every frozen split-assignment row"
                )
            if dict(observed_sources) != source_expected:
                raise ValueError(
                    "parent filtered mixture_source counts do not match the freeze/manifest"
                )

            provenance_by_output: dict[Path, dict[str, Any]] = {}
            for stage in stages:
                if stage.observed_documents != stage.expected_documents:
                    raise ValueError(
                        f"derived group {stage.definition.output_dir} document count mismatch: "
                        f"expected {stage.expected_documents}, got {stage.observed_documents}"
                    )
                provenance_by_output[stage.definition.output_dir] = _stage_metadata(
                    stage,
                    freeze=freeze,
                    parent_manifest=manifest,
                    parent_identity=parent_identity,
                    source_expected_counts=source_expected,
                    all_group_mapping=all_group_mapping,
                    require_complete_parent=require_complete_parent,
                )
                stage.connection.close()

            tokenizer_contract = parent_identity["tokenizer"]
            tokenizer_lineage = {
                "kind": tokenizer_contract["kind"],
                "vocab_size": tokenizer_contract["vocab_size"],
                "artifact": tokenizer_contract["artifact"],
                "trained": False,
                "split": None,
                "reuse": "exact_parent_freeze_tokenizer",
            }
            for stage in stages:
                corpus = DiskBackedCorpus(stage.database_path)
                child_manifest = pack_disk_backed_shards(
                    corpus,
                    tokenizer,
                    int(manifest["seq_len"]),
                    str(stage.temporary_output),
                    rows_per_shard=rows_per_shard,
                    tokenizer_training=tokenizer_lineage,
                    preparation_provenance=provenance_by_output[stage.definition.output_dir],
                )
                child_manifest = _write_final_manifest(
                    stage,
                    child_manifest,
                    provenance=provenance_by_output[stage.definition.output_dir],
                    tokenizer_identity=tokenizer_contract,
                    rows_per_shard=rows_per_shard,
                )
                _verify_child_assignment(stage, child_manifest)
                manifests[str(stage.definition.output_dir)] = child_manifest

            existing = _publish_groups(stages)
            for stage in stages:
                if stage.definition.output_dir in existing:
                    shutil.rmtree(stage.temporary_output)
        finally:
            assignment_connection.close()
            if parent_staging_connection is not None:
                parent_staging_connection.close()
            for stage in stages:
                try:
                    stage.connection.close()
                except sqlite3.ProgrammingError:
                    pass
                if stage.temporary_output.exists():
                    shutil.rmtree(stage.temporary_output)

    return manifests
