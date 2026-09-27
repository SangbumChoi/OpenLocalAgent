"""Private, prompt-only Mind2Web decontamination export.

This adapter consumes caller-verified local snapshots.  It never downloads data and deliberately
does not emit actions, candidates, labels, or scores.  The resulting canonical JSONL is suitable
only as an evaluation denylist input; it is not a Mind2Web scorer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
import stat
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO

from localagent.data.mind2web_dom_ranker import (
    MIND2WEB_DOM_RANKING_AUDIT_KIND,
    MIND2WEB_DOM_RANKING_AUDIT_SCHEMA_VERSION,
    MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION,
    Mind2WebDomRankerConfig,
    RankedDom,
    implementation_identity,
    load_mind2web_dom_ranker_config,
    rank_mind2web_dom,
    runtime_identity,
)

MIND2WEB_PROMPT_ADAPTER = "mind2web-private-prompt-rows-v1"
MIND2WEB_PROMPT_ADAPTER_VERSION = MIND2WEB_PROMPT_ADAPTER
MIND2WEB_AUDIT_KIND = "localagent_mind2web_prompt_adapter_audit"
MIND2WEB_AUDIT_SCHEMA_VERSION = 2
MIND2WEB_RANKED_AUDIT_SCHEMA_VERSION = 3
MIND2WEB_ARCHIVE_ATTESTATION_KIND = "localagent_mind2web_protected_archive_attestation"
MIND2WEB_ARCHIVE_ATTESTATION_SCHEMA_VERSION = 1
MIND2WEB_ADAPTER_IMPLEMENTATION_MODULE = "localagent.data.mind2web_prompts"
MIND2WEB_ADAPTER_IMPLEMENTATION_PATH = "src/localagent/data/mind2web_prompts.py"
PRODUCTION_MIND2WEB_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
PRODUCTION_MIND2WEB_SPLIT = "cross_domain+cross_task+cross_website"
PRODUCTION_MIND2WEB_ARCHIVE_BYTES = 567_745_122
PRODUCTION_MIND2WEB_ARCHIVE_SHA256 = (
    "8f5fbe72afab942fe97cdf7fb397e179885d89b5c16862288e9a14bc6d41ca89"
)
PRODUCTION_MIND2WEB_ARCHIVE_PASSWORD = b"mind2web"
PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED = True
PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES = 6_107_912_752
PRODUCTION_MIND2WEB_MEMBERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "cross_domain": tuple(
            f"test_domain/test_domain_{index}.json" for index in range(10)
        ),
        "cross_task": tuple(f"test_task/test_task_{index}.json" for index in range(3)),
        "cross_website": tuple(
            f"test_website/test_website_{index}.json" for index in range(2)
        ),
    }
)
PRODUCTION_MIND2WEB_TASK_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "cross_domain": 912,
        "cross_task": 252,
        "cross_website": 177,
    }
)

DEFAULT_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_SOURCE_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024 * 1024
DEFAULT_MAX_SOURCES = 32
DEFAULT_MAX_COMPRESSION_RATIO = 32
# The largest production member is 770,875,063 bytes, so a one-Gi-character record ceiling admits
# every possible attested member while remaining far below the independent eight-GiB source cap.
# Large records are decoded with exponentially growing read attempts below; fixed 64-KiB retries
# make ``json.raw_decode`` quadratic on real DOM-heavy tasks.
DEFAULT_MAX_RECORD_CHARS = 1024 * 1024 * 1024
# Match the downstream generic suite freezer's per-prompt and per-source ceilings. Full held-out
# HTML that exceeds these limits needs a separately frozen, gold-independent ranker; it must not
# be target-informed or silently truncated by this adapter.
DEFAULT_MAX_PROMPT_BYTES = 512 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ROWS = 100_000

_READ_CHUNK_BYTES = 1024 * 1024
_JSON_CHUNK_CHARS = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_TASK_KEYS = frozenset(
    {
        "website",
        "domain",
        "subdomain",
        "annotation_id",
        "confirmed_task",
        "action_reprs",
        "actions",
    }
)
_ACTION_KEYS = frozenset(
    {
        "action_uid",
        "raw_html",
        "cleaned_html",
        "operation",
        "pos_candidates",
        "neg_candidates",
    }
)
_OPERATION_KEYS = frozenset({"op", "original_op", "value"})
_NORMALIZED_OPERATIONS = frozenset({"CLICK", "TYPE", "SELECT"})


@dataclass(frozen=True)
class Mind2WebSource:
    """One immutable local Mind2Web shard identity supplied by the caller."""

    path: Path
    bytes: int
    sha256: str
    archive_member: str | None = None


@dataclass(frozen=True)
class Mind2WebArchive:
    """One immutable local copy of the protected upstream ``test.zip``."""

    path: Path
    bytes: int
    sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_implementation_identity() -> dict[str, int | str]:
    path = Path(__file__)
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "module": MIND2WEB_ADAPTER_IMPLEMENTATION_MODULE,
        "path": MIND2WEB_ADAPTER_IMPLEMENTATION_PATH,
        "sha256": _sha256(payload),
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("revision must be a lowercase 40-character immutable commit SHA")
    return revision


def _validated_source_path(
    source: Mind2WebSource | Mind2WebArchive,
    *,
    max_source_bytes: int,
    label: str = "source",
) -> Path:
    path = Path(source.path)
    if not _SHA256_RE.fullmatch(source.sha256):
        raise ValueError(f"{label} {path} has an invalid expected SHA-256")
    if (
        isinstance(source.bytes, bool)
        or not isinstance(source.bytes, int)
        or source.bytes < 0
    ):
        raise ValueError(f"{label} {path} has an invalid expected byte size")
    if source.bytes > max_source_bytes:
        raise ValueError(f"{label} exceeds the {max_source_bytes}-byte cap: {path}")
    return path


def _snapshot_verified_source(
    source: Mind2WebSource | Mind2WebArchive,
    snapshot_path: Path,
    *,
    max_source_bytes: int,
    label: str = "source",
) -> tuple[Path, dict[str, int | str]]:
    """Copy and hash one source once, then return the private verified parse snapshot."""

    path = _validated_source_path(
        source,
        max_source_bytes=max_source_bytes,
        label=label,
    )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file: {path}") from error

    observed_bytes = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb") as source_handle, snapshot_path.open("xb") as snapshot:
            source_stat = os.fstat(source_handle.fileno())
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"{label} must be a regular non-symlink file: {path}")
            if source_stat.st_size > max_source_bytes:
                raise ValueError(f"{label} exceeds the {max_source_bytes}-byte cap: {path}")
            if source_stat.st_size != source.bytes:
                raise ValueError(
                    f"{label} byte-size mismatch for {path}: "
                    f"expected {source.bytes}, got {source_stat.st_size}"
                )
            for chunk in iter(lambda: source_handle.read(_READ_CHUNK_BYTES), b""):
                observed_bytes += len(chunk)
                if observed_bytes > max_source_bytes:
                    raise ValueError(
                        f"{label} exceeds the {max_source_bytes}-byte cap: {path}"
                    )
                snapshot.write(chunk)
                digest.update(chunk)
            snapshot.flush()
            os.fsync(snapshot.fileno())
    except Exception:
        snapshot_path.unlink(missing_ok=True)
        raise

    observed_sha256 = digest.hexdigest()
    if observed_bytes != source.bytes:
        snapshot_path.unlink(missing_ok=True)
        raise ValueError(
            f"{label} byte-size mismatch for {path}: "
            f"expected {source.bytes}, got {observed_bytes}"
        )
    if observed_sha256 != source.sha256:
        snapshot_path.unlink(missing_ok=True)
        raise ValueError(f"{label} SHA-256 mismatch for {path}")
    snapshot_path.chmod(0o400)
    return (
        snapshot_path,
        {"bytes": observed_bytes, "name": path.name, "sha256": observed_sha256},
    )


def _production_member_splits() -> dict[str, str]:
    if set(PRODUCTION_MIND2WEB_MEMBERS) != set(PRODUCTION_MIND2WEB_TASK_COUNTS):
        raise AssertionError("production Mind2Web split policies disagree")
    member_splits: dict[str, str] = {}
    for split_name, members in PRODUCTION_MIND2WEB_MEMBERS.items():
        if not members:
            raise AssertionError(f"production Mind2Web split {split_name!r} has no members")
        for member in members:
            if member in member_splits:
                raise AssertionError(f"duplicate production Mind2Web member policy: {member}")
            member_splits[member] = split_name
    return member_splits


def _validate_archive_member_name(name: str) -> None:
    member = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or member.as_posix() != name
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ValueError(f"unsafe Mind2Web archive member path: {name!r}")
    if len(name.encode("utf-8")) > 256:
        raise ValueError(f"Mind2Web archive member path is too long: {name!r}")


def _attest_production_archive(
    archive_path: Path,
    *,
    source_identities: Mapping[str, Mapping[str, int | str]],
    max_source_bytes: int,
    max_total_source_bytes: int,
    max_sources: int,
    max_compression_ratio: int,
) -> list[dict[str, int | str]]:
    """Bind every extracted source identity to plaintext streamed from the pinned ZIP."""

    expected_member_splits = _production_member_splits()
    expected_members = frozenset(expected_member_splits)
    if len(expected_members) > max_sources:
        raise ValueError(f"production Mind2Web archive exceeds max_sources={max_sources}")
    if frozenset(source_identities) != expected_members:
        missing = sorted(expected_members - frozenset(source_identities))
        extra = sorted(frozenset(source_identities) - expected_members)
        raise ValueError(
            "production Mind2Web extracted sources do not match the official member set: "
            f"missing={missing}, extra={extra}"
        )

    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) > max_sources:
                raise ValueError(
                    f"Mind2Web archive exceeds max_sources={max_sources}"
                )
            names = [info.filename for info in infos]
            for name in names:
                _validate_archive_member_name(name)
            if len(names) != len(set(names)):
                raise ValueError("Mind2Web archive contains duplicate member names")
            observed_members = frozenset(names)
            if observed_members != expected_members:
                missing = sorted(expected_members - observed_members)
                extra = sorted(observed_members - expected_members)
                raise ValueError(
                    "production Mind2Web archive member set mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            info_by_name = {info.filename: info for info in infos}
            declared_uncompressed_bytes = 0
            for member in sorted(expected_members):
                info = info_by_name[member]
                if info.is_dir():
                    raise ValueError(f"Mind2Web archive member must be a file: {member}")
                unix_mode = info.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in (0, stat.S_IFREG):
                    raise ValueError(
                        f"Mind2Web archive member must be regular: {member}"
                    )
                if info.flag_bits & 0x20:
                    raise ValueError(
                        f"Mind2Web archive member uses patched data: {member}"
                    )
                if info.flag_bits & 0x40:
                    raise ValueError(
                        f"Mind2Web archive member uses unsupported strong encryption: {member}"
                    )
                encrypted = bool(info.flag_bits & 0x1)
                if encrypted is not PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED:
                    raise ValueError(
                        f"Mind2Web archive member encryption mismatch: {member}"
                    )
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise ValueError(
                        f"Mind2Web archive member must use DEFLATE: {member}"
                    )
                if info.file_size <= 0 or info.file_size > max_source_bytes:
                    raise ValueError(
                        f"Mind2Web archive member exceeds source bounds: {member}"
                    )
                if info.compress_size <= 0:
                    raise ValueError(
                        f"Mind2Web archive member has invalid compressed size: {member}"
                    )
                if info.file_size > info.compress_size * max_compression_ratio:
                    raise ValueError(
                        f"Mind2Web archive member exceeds compression-ratio cap: {member}"
                    )
                source_identity = source_identities[member]
                if source_identity["bytes"] != info.file_size:
                    raise ValueError(
                        "Mind2Web archive plaintext byte size does not match "
                        f"extracted source: {member}"
                    )
                declared_uncompressed_bytes += info.file_size
                if declared_uncompressed_bytes > max_total_source_bytes:
                    raise ValueError(
                        "Mind2Web archive exceeds max_total_source_bytes="
                        f"{max_total_source_bytes}"
                    )
            if declared_uncompressed_bytes != PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "production Mind2Web archive uncompressed byte total mismatch: "
                    f"expected {PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES}, "
                    f"got {declared_uncompressed_bytes}"
                )

            member_audits: list[dict[str, int | str]] = []
            observed_total = 0
            password = (
                PRODUCTION_MIND2WEB_ARCHIVE_PASSWORD
                if PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED
                else None
            )
            for member in sorted(expected_members):
                info = info_by_name[member]
                digest = hashlib.sha256()
                observed_bytes = 0
                with archive.open(info, mode="r", pwd=password) as handle:
                    for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
                        observed_bytes += len(chunk)
                        observed_total += len(chunk)
                        if observed_bytes > max_source_bytes:
                            raise ValueError(
                                f"Mind2Web archive member exceeds source cap: {member}"
                            )
                        if observed_total > max_total_source_bytes:
                            raise ValueError(
                                "Mind2Web archive exceeds max_total_source_bytes="
                                f"{max_total_source_bytes}"
                            )
                        digest.update(chunk)
                observed_sha256 = digest.hexdigest()
                source_identity = source_identities[member]
                if observed_bytes != info.file_size:
                    raise ValueError(
                        f"Mind2Web archive member byte count changed while reading: {member}"
                    )
                if (
                    observed_bytes != source_identity["bytes"]
                    or not hmac.compare_digest(
                        observed_sha256,
                        str(source_identity["sha256"]),
                    )
                ):
                    raise ValueError(
                        "Mind2Web archive plaintext does not match extracted source: "
                        f"{member}"
                    )
                member_audits.append(
                    {
                        "bytes": observed_bytes,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "member": member,
                        "sha256": observed_sha256,
                        "split": expected_member_splits[member],
                    }
                )
    except (NotImplementedError, RuntimeError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot verify protected Mind2Web archive: {error}") from error

    if observed_total != PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            "production Mind2Web archive streamed byte total mismatch: "
            f"expected {PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES}, got {observed_total}"
        )
    return member_audits


class _Utf8ChunkReader:
    """Bounded incremental UTF-8 reader over a binary source."""

    def __init__(self, handle: BinaryIO, *, max_bytes: int) -> None:
        import codecs

        self._handle = handle
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._max_bytes = max_bytes
        self._bytes = 0
        self._finished = False

    def read(self, chunk_bytes: int = _JSON_CHUNK_CHARS) -> str:
        if self._finished:
            return ""
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("UTF-8 reader chunk_bytes must be a positive integer")
        payload = self._handle.read(chunk_bytes)
        self._bytes += len(payload)
        if self._bytes > self._max_bytes:
            raise ValueError(f"decoded JSON exceeds the {self._max_bytes}-byte cap")
        if payload:
            try:
                return self._decoder.decode(payload, final=False)
            except UnicodeDecodeError as error:
                raise ValueError("source is not valid UTF-8") from error
        self._finished = True
        try:
            return self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise ValueError("source is not valid UTF-8") from error

    @property
    def finished(self) -> bool:
        return self._finished


def _iter_json_array(
    path: Path,
    *,
    max_bytes: int,
    max_record_chars: int,
) -> Iterator[Any]:
    """Stream one top-level JSON array without importing a large parser dependency."""

    decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_constant,
    )
    with path.open("rb") as binary:
        reader = _Utf8ChunkReader(binary, max_bytes=max_bytes)
        buffer = ""
        position = 0
        started = False
        expect_value = True
        after_comma = False

        def fill(chunk_bytes: int = _JSON_CHUNK_CHARS) -> bool:
            nonlocal buffer, position
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = reader.read(chunk_bytes)
            buffer += chunk
            return bool(chunk)

        while True:
            while position >= len(buffer) and not reader.finished:
                fill()
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    raise ValueError(f"{path} is empty")
                if buffer[position] != "[":
                    raise ValueError(f"{path} must contain one top-level JSON array")
                position += 1
                started = True
                continue

            while position >= len(buffer) and not reader.finished:
                fill()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
            if position < len(buffer) and buffer[position] == "]":
                if after_comma:
                    raise ValueError(f"{path} JSON array has a trailing comma")
                position += 1
                while not reader.finished:
                    fill()
                if buffer[position:].strip():
                    raise ValueError(f"{path} has trailing content after its JSON array")
                return
            if not expect_value:
                if position >= len(buffer):
                    raise ValueError(f"{path} has an unterminated JSON array")
                if buffer[position] != ",":
                    raise ValueError(f"{path} JSON array entries must be comma-separated")
                position += 1
                expect_value = True
                after_comma = True
                continue

            record_start = position
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as error:
                    if reader.finished:
                        raise ValueError(f"{path} contains invalid JSON") from error
                    buffered_record_chars = len(buffer) - record_start
                    if buffered_record_chars > max_record_chars:
                        raise ValueError(
                            f"{path} contains a record exceeding {max_record_chars} characters"
                        ) from error
                    # Retry only after approximately doubling the buffered record. The stdlib
                    # decoder rescans from the record start, so fixed-size retries become
                    # quadratic for the 100-MiB-plus DOM records present in the protected split.
                    next_read_bytes = max(
                        _JSON_CHUNK_CHARS,
                        min(
                            max_record_chars - buffered_record_chars + 1,
                            max(buffered_record_chars, 1),
                        ),
                    )
                    fill(next_read_bytes)
                    record_start = 0
                    continue
                if end - record_start > max_record_chars:
                    raise ValueError(
                        f"{path} contains a record exceeding {max_record_chars} characters"
                    )
                position = end
                expect_value = False
                after_comma = False
                yield value
                break


def _assert_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema drift: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class Mind2WebRankerInput:
    """The complete field projection admitted to the production DOM ranker."""

    confirmed_task: str
    cleaned_html: str


@dataclass(frozen=True)
class _RankedPrompt:
    prompt: str
    ranking: RankedDom
    prompt_bytes: int
    framed_prompt_bytes: int
    prompt_sha256: str


@dataclass
class _RankingCounters:
    rows: int = 0
    full_html_bytes: int = 0
    selected_bytes: int = 0
    parsed_nodes: int = 0
    eligible_nodes: int = 0
    selected_nodes: int = 0
    max_full_html_bytes: int = 0
    max_selected_bytes: int = 0
    max_parsed_nodes: int = 0
    max_selected_nodes: int = 0
    max_prompt_bytes: int = 0
    max_framed_prompt_bytes: int = 0

    def add(self, ranked: _RankedPrompt) -> None:
        self.rows += 1
        self.full_html_bytes += ranked.ranking.full_html_bytes
        self.selected_bytes += ranked.ranking.selected_bytes
        self.parsed_nodes += ranked.ranking.parsed_nodes
        self.eligible_nodes += ranked.ranking.eligible_nodes
        self.selected_nodes += ranked.ranking.selected_nodes
        self.max_full_html_bytes = max(
            self.max_full_html_bytes,
            ranked.ranking.full_html_bytes,
        )
        self.max_selected_bytes = max(
            self.max_selected_bytes,
            ranked.ranking.selected_bytes,
        )
        self.max_parsed_nodes = max(
            self.max_parsed_nodes,
            ranked.ranking.parsed_nodes,
        )
        self.max_selected_nodes = max(
            self.max_selected_nodes,
            ranked.ranking.selected_nodes,
        )
        self.max_prompt_bytes = max(self.max_prompt_bytes, ranked.prompt_bytes)
        self.max_framed_prompt_bytes = max(
            self.max_framed_prompt_bytes,
            ranked.framed_prompt_bytes,
        )

    def merge(self, other: _RankingCounters) -> None:
        self.rows += other.rows
        self.full_html_bytes += other.full_html_bytes
        self.selected_bytes += other.selected_bytes
        self.parsed_nodes += other.parsed_nodes
        self.eligible_nodes += other.eligible_nodes
        self.selected_nodes += other.selected_nodes
        self.max_full_html_bytes = max(
            self.max_full_html_bytes,
            other.max_full_html_bytes,
        )
        self.max_selected_bytes = max(
            self.max_selected_bytes,
            other.max_selected_bytes,
        )
        self.max_parsed_nodes = max(
            self.max_parsed_nodes,
            other.max_parsed_nodes,
        )
        self.max_selected_nodes = max(
            self.max_selected_nodes,
            other.max_selected_nodes,
        )
        self.max_prompt_bytes = max(self.max_prompt_bytes, other.max_prompt_bytes)
        self.max_framed_prompt_bytes = max(
            self.max_framed_prompt_bytes,
            other.max_framed_prompt_bytes,
        )

    def as_audit(self) -> dict[str, int]:
        return {
            "eligible_nodes": self.eligible_nodes,
            "full_html_bytes": self.full_html_bytes,
            "max_framed_prompt_bytes": self.max_framed_prompt_bytes,
            "max_full_html_bytes": self.max_full_html_bytes,
            "max_parsed_nodes": self.max_parsed_nodes,
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_selected_bytes": self.max_selected_bytes,
            "max_selected_nodes": self.max_selected_nodes,
            "parsed_nodes": self.parsed_nodes,
            "rows": self.rows,
            "selected_bytes": self.selected_bytes,
            "selected_nodes": self.selected_nodes,
        }


def render_mind2web_step_prompt(
    confirmed_task: str,
    cleaned_html: str,
    *,
    prior_actions: Sequence[str] = (),
    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
) -> str:
    """Render the versioned label-free prompt for one Mind2Web state."""

    _positive_int(max_prompt_bytes, label="max_prompt_bytes")
    task = _nonempty_string(confirmed_task, label="confirmed_task")
    context = _nonempty_string(cleaned_html, label="cleaned_html")
    if isinstance(prior_actions, (str, bytes)) or not isinstance(prior_actions, Sequence):
        raise TypeError("prior_actions must be an array of strings")
    history = [
        _nonempty_string(action, label=f"prior_actions[{index}]")
        for index, action in enumerate(prior_actions)
    ]
    history_text = "\n".join(
        f"{index + 1}. {action}" for index, action in enumerate(history)
    )
    if not history_text:
        history_text = "<none>"
    prompt = (
        f"[{MIND2WEB_PROMPT_ADAPTER_VERSION}]\n"
        "Complete the browser task by choosing the next action from the current cleaned HTML.\n"
        f"<task>\n{task}\n</task>\n"
        f"<prior_actions>\n{history_text}\n</prior_actions>\n"
        f"<cleaned_html>\n{context}\n</cleaned_html>\n"
        "Return only the next browser action."
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_prompt_bytes:
        raise ValueError(
            f"rendered Mind2Web prompt is {prompt_bytes} bytes, exceeding "
            f"max_prompt_bytes={max_prompt_bytes}"
        )
    return prompt


def project_mind2web_ranker_input(
    task: Mapping[str, Any],
    step_index: int,
    *,
    label: str = "Mind2Web task",
) -> Mind2WebRankerInput:
    """Project one production step without dereferencing any gold-bearing field value."""

    if not isinstance(task, Mapping):
        raise TypeError(f"{label} must be an object")
    _assert_keys(task, _TASK_KEYS, label=label)
    confirmed_task = _nonempty_string(
        task["confirmed_task"],
        label=f"{label}.confirmed_task",
    )
    actions = task["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{label}.actions must be a non-empty array")
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
        or step_index >= len(actions)
    ):
        raise ValueError(f"{label} step_index is out of range")
    action = actions[step_index]
    action_label = f"{label}.actions[{step_index}]"
    if not isinstance(action, Mapping):
        raise TypeError(f"{action_label} must be an object")
    _assert_keys(action, _ACTION_KEYS, label=action_label)
    cleaned_html = _nonempty_string(
        action["cleaned_html"],
        label=f"{action_label}.cleaned_html",
    )
    return Mind2WebRankerInput(
        confirmed_task=confirmed_task,
        cleaned_html=cleaned_html,
    )


def _ranked_prompt_parts(confirmed_task: str) -> tuple[str, str]:
    task = _nonempty_string(confirmed_task, label="confirmed_task")
    return (
        (
            f"[{MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION}]\n"
            "Complete the browser task by choosing the next action from the ranked current DOM.\n"
            f"<task>\n{task}\n</task>\n"
            "<ranked_dom>\n"
        ),
        "\n</ranked_dom>\nReturn only the next browser action.",
    )


def _render_ranked_mind2web_step_prompt(
    ranker_input: Mind2WebRankerInput,
    *,
    config: Mind2WebDomRankerConfig,
) -> _RankedPrompt:
    prefix, suffix = _ranked_prompt_parts(ranker_input.confirmed_task)
    base_framed = (
        config.user_marker + prefix + suffix + config.assistant_marker
    ).encode("utf-8")
    context_byte_budget = config.max_framed_prompt_bytes - len(base_framed)
    ranking = rank_mind2web_dom(
        ranker_input.confirmed_task,
        ranker_input.cleaned_html,
        context_byte_budget=context_byte_budget,
        config=config,
    )
    prompt = prefix + ranking.context + suffix
    prompt_payload = prompt.encode("utf-8")
    framed_payload = (
        config.user_marker + prompt + config.assistant_marker
    ).encode("utf-8")
    if len(prompt_payload) > config.max_unframed_prompt_bytes:
        raise AssertionError("ranked prompt exceeded its unframed byte budget")
    if len(framed_payload) > config.max_framed_prompt_bytes:
        raise AssertionError("ranked prompt exceeded its fully framed byte budget")
    return _RankedPrompt(
        prompt=prompt,
        ranking=ranking,
        prompt_bytes=len(prompt_payload),
        framed_prompt_bytes=len(framed_payload),
        prompt_sha256=_sha256(prompt_payload),
    )


def _case_id(revision: str, annotation_id: str, action_uid: str, step_index: int) -> str:
    identity = {
        "action_uid": action_uid,
        "annotation_id": annotation_id,
        "revision": revision,
        "step_index": step_index,
    }
    return f"mind2web-step-v1:{_sha256(_canonical_json_bytes(identity))}"


def _ranked_case_id(
    revision: str,
    source_name: str,
    source_index: int,
    step_index: int,
) -> str:
    identity = {
        "revision": revision,
        "source_index": source_index,
        "source_name": source_name,
        "step_index": step_index,
    }
    return f"mind2web-step-v2:{_sha256(_canonical_json_bytes(identity))}"


def _stage_task(
    connection: sqlite3.Connection,
    task: Any,
    *,
    revision: str,
    source_name: str,
    source_index: int,
    max_prompt_bytes: int,
) -> tuple[int, str]:
    label = f"{source_name}:task[{source_index}]"
    if not isinstance(task, dict):
        raise TypeError(f"{label} must be an object")
    _assert_keys(task, _TASK_KEYS, label=label)
    for field in ("website", "domain", "subdomain"):
        _nonempty_string(task[field], label=f"{label}.{field}")
    annotation_id = _nonempty_string(task["annotation_id"], label=f"{label}.annotation_id")
    confirmed_task = _nonempty_string(task["confirmed_task"], label=f"{label}.confirmed_task")
    actions = task["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{label}.actions must be a non-empty array")
    action_reprs = task["action_reprs"]
    if not isinstance(action_reprs, list):
        raise TypeError(f"{label}.action_reprs must be an array")
    if len(action_reprs) != len(actions):
        raise ValueError(
            f"{label}.action_reprs must align one-to-one with actions: "
            f"expected {len(actions)}, got {len(action_reprs)}"
        )
    action_reprs = [
        _nonempty_string(value, label=f"{label}.action_reprs[{index}]")
        for index, value in enumerate(action_reprs)
    ]
    try:
        connection.execute(
            "INSERT INTO tasks(task_identity, source_name) VALUES (?, ?)",
            (annotation_id, source_name),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(f"duplicate Mind2Web annotation_id: {annotation_id!r}") from error

    rows = 0
    action_uids: set[str] = set()
    for step_index, action in enumerate(actions):
        action_label = f"{label}.actions[{step_index}]"
        if not isinstance(action, dict):
            raise TypeError(f"{action_label} must be an object")
        _assert_keys(action, _ACTION_KEYS, label=action_label)
        action_uid = _nonempty_string(
            action["action_uid"], label=f"{action_label}.action_uid"
        )
        if action_uid in action_uids:
            raise ValueError(f"{label} contains duplicate action_uid {action_uid!r}")
        action_uids.add(action_uid)
        if not isinstance(action["raw_html"], str):
            raise TypeError(f"{action_label}.raw_html must be a string")
        for candidate_field in ("pos_candidates", "neg_candidates"):
            if not isinstance(action[candidate_field], list):
                raise TypeError(f"{action_label}.{candidate_field} must be an array")
        operation = action["operation"]
        if not isinstance(operation, dict):
            raise TypeError(f"{action_label}.operation must be an object")
        _assert_keys(operation, _OPERATION_KEYS, label=f"{action_label}.operation")
        normalized_operation = _nonempty_string(
            operation["op"], label=f"{action_label}.operation.op"
        ).upper()
        if normalized_operation not in _NORMALIZED_OPERATIONS:
            raise ValueError(
                f"{action_label}.operation.op is unsupported: {normalized_operation!r}"
            )
        _nonempty_string(
            operation["original_op"], label=f"{action_label}.operation.original_op"
        )
        if not isinstance(operation["value"], str):
            raise TypeError(f"{action_label}.operation.value must be a string")

        prompt = render_mind2web_step_prompt(
            confirmed_task,
            action["cleaned_html"],
            prior_actions=action_reprs[:step_index],
            max_prompt_bytes=max_prompt_bytes,
        )
        source_case_id = _case_id(
            revision,
            annotation_id,
            action_uid,
            step_index,
        )
        try:
            connection.execute(
                """
                INSERT INTO prompt_rows(
                    source_name, sort_key, step_index, source_case_id, prompt,
                    ranking_receipt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_name,
                    annotation_id,
                    step_index,
                    source_case_id,
                    prompt,
                    "",
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"duplicate Mind2Web source case {source_case_id}") from error
        rows += 1
    return rows, annotation_id


def _stage_ranked_task(
    connection: sqlite3.Connection,
    task: Any,
    *,
    revision: str,
    source_name: str,
    source_index: int,
    ranker_config: Mind2WebDomRankerConfig,
) -> tuple[int, _RankingCounters]:
    label = f"{source_name}:task[{source_index}]"
    if not isinstance(task, Mapping):
        raise TypeError(f"{label} must be an object")
    _assert_keys(task, _TASK_KEYS, label=label)
    actions = task["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{label}.actions must be a non-empty array")
    task_identity = _sha256(
        _canonical_json_bytes(
            {
                "source_index": source_index,
                "source_name": source_name,
            }
        )
    )
    try:
        connection.execute(
            "INSERT INTO tasks(task_identity, source_name) VALUES (?, ?)",
            (task_identity, source_name),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"duplicate production Mind2Web task ordinal: {source_name}:{source_index}"
        ) from error

    counters = _RankingCounters()
    sort_key = f"{source_index:012d}"
    for step_index in range(len(actions)):
        ranker_input = project_mind2web_ranker_input(
            task,
            step_index,
            label=label,
        )
        ranked = _render_ranked_mind2web_step_prompt(
            ranker_input,
            config=ranker_config,
        )
        source_case_id = _ranked_case_id(
            revision,
            source_name,
            source_index,
            step_index,
        )
        receipt = {
            "eligible_nodes": ranked.ranking.eligible_nodes,
            "framed_prompt_bytes": ranked.framed_prompt_bytes,
            "full_html_bytes": ranked.ranking.full_html_bytes,
            "full_html_sha256": ranked.ranking.full_html_sha256,
            "parsed_nodes": ranked.ranking.parsed_nodes,
            "prompt_bytes": ranked.prompt_bytes,
            "prompt_sha256": ranked.prompt_sha256,
            "ranked_dom_sha256": ranked.ranking.ranked_dom_sha256,
            "selected_bytes": ranked.ranking.selected_bytes,
            "selected_nodes": ranked.ranking.selected_nodes,
            "source_case_id": source_case_id,
        }
        try:
            connection.execute(
                """
                INSERT INTO prompt_rows(
                    source_name, sort_key, step_index, source_case_id, prompt,
                    ranking_receipt
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_name,
                    sort_key,
                    step_index,
                    source_case_id,
                    ranked.prompt,
                    _canonical_json_bytes(receipt).decode("utf-8"),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"duplicate Mind2Web source case {source_case_id}"
            ) from error
        counters.add(ranked)
    return len(actions), counters


def _write_staged_rows(
    connection: sqlite3.Connection,
    staged_path: Path,
    *,
    max_output_bytes: int,
) -> tuple[int, int, str, str, str | None]:
    output_digest = hashlib.sha256()
    case_digest = hashlib.sha256()
    ranking_digest = hashlib.sha256()
    output_bytes = 0
    rows = 0
    first_case = True
    ranked_rows = 0
    with staged_path.open("wb") as handle:
        for source_case_id, prompt, ranking_receipt in connection.execute(
            """
            SELECT source_case_id, prompt, ranking_receipt
            FROM prompt_rows
            ORDER BY source_name, sort_key, step_index
            """
        ):
            row = {"prompt": prompt, "source_case_id": source_case_id}
            if set(row) != {"source_case_id", "prompt"}:
                raise AssertionError("prompt-only row gained an unexpected field")
            payload = _canonical_json_bytes(row) + b"\n"
            output_bytes += len(payload)
            if output_bytes > max_output_bytes:
                raise ValueError(
                    f"prompt output exceeds max_output_bytes={max_output_bytes}"
                )
            handle.write(payload)
            output_digest.update(payload)
            if not first_case:
                case_digest.update(b"\n")
            case_digest.update(source_case_id.encode("ascii"))
            first_case = False
            if ranking_receipt:
                ranking_digest.update(ranking_receipt.encode("utf-8"))
                ranking_digest.update(b"\n")
                ranked_rows += 1
            rows += 1
        handle.flush()
        os.fsync(handle.fileno())
    return (
        rows,
        output_bytes,
        output_digest.hexdigest(),
        case_digest.hexdigest(),
        ranking_digest.hexdigest() if ranked_rows else None,
    )


def _existing_matches(path: Path, *, expected_bytes: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == expected_bytes
        and _hash_file(path) == expected_sha256
    )


def _preflight_output(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    if path.exists() and not _existing_matches(
        path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"refusing to overwrite drifted {label}: {path}")


def _publish_staged(
    staged_path: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> tuple[int, int] | None:
    created_by_this_call = False
    if not destination.exists():
        try:
            os.link(staged_path, destination)
            created_by_this_call = True
        except FileExistsError:
            pass
    if not _existing_matches(
        destination,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    ):
        raise RuntimeError(f"concurrently published {label} does not match: {destination}")
    if not created_by_this_call:
        return None
    staged_stat = staged_path.stat()
    destination_stat = destination.lstat()
    if (
        not stat.S_ISREG(destination_stat.st_mode)
        or staged_stat.st_dev != destination_stat.st_dev
        or staged_stat.st_ino != destination_stat.st_ino
    ):
        raise RuntimeError(f"published {label} inode changed during verification: {destination}")
    return destination_stat.st_dev, destination_stat.st_ino


def _unlink_if_same_inode(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(observed.st_mode) and (observed.st_dev, observed.st_ino) == identity:
        path.unlink()


def export_mind2web_prompt_rows(
    sources: Sequence[Mind2WebSource],
    output_path: str | Path,
    *,
    revision: str,
    split: str = "fixture",
    archive: Mind2WebArchive | None = None,
    audit_path: str | Path | None = None,
    ranker_config_path: str | Path | None = None,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_total_source_bytes: int = DEFAULT_MAX_TOTAL_SOURCE_BYTES,
    max_sources: int = DEFAULT_MAX_SOURCES,
    max_compression_ratio: int = DEFAULT_MAX_COMPRESSION_RATIO,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
    max_prompt_bytes: int | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Validate local shards and atomically export canonical two-field prompt rows.

    Fixture exports preserve the v1 full-HTML adapter.  The exact production revision/split
    requires an explicit frozen ranker config and emits only v2 ranked prompts.
    """

    revision = _validate_revision(revision)
    split = _nonempty_string(split, label="split")
    mode = (
        "production"
        if revision == PRODUCTION_MIND2WEB_REVISION
        and split == PRODUCTION_MIND2WEB_SPLIT
        else "fixture"
    )
    for value, label in (
        (max_archive_bytes, "max_archive_bytes"),
        (max_source_bytes, "max_source_bytes"),
        (max_total_source_bytes, "max_total_source_bytes"),
        (max_sources, "max_sources"),
        (max_compression_ratio, "max_compression_ratio"),
        (max_record_chars, "max_record_chars"),
        (max_output_bytes, "max_output_bytes"),
        (max_rows, "max_rows"),
    ):
        _positive_int(value, label=label)
    if max_prompt_bytes is not None:
        _positive_int(max_prompt_bytes, label="max_prompt_bytes")
    ranker_config: Mind2WebDomRankerConfig | None = None
    if mode == "production":
        if ranker_config_path is None:
            raise ValueError(
                "production Mind2Web export requires an explicit ranker_config_path"
            )
        ranker_config = load_mind2web_dom_ranker_config(ranker_config_path)
        effective_max_prompt_bytes = ranker_config.max_unframed_prompt_bytes
        if (
            max_prompt_bytes is not None
            and max_prompt_bytes != effective_max_prompt_bytes
        ):
            raise ValueError(
                "production max_prompt_bytes is pinned by the ranker config to "
                f"{effective_max_prompt_bytes}"
            )
    else:
        if ranker_config_path is not None:
            raise ValueError("fixture Mind2Web export must not declare a ranker config")
        effective_max_prompt_bytes = (
            DEFAULT_MAX_PROMPT_BYTES
            if max_prompt_bytes is None
            else max_prompt_bytes
        )
    active_adapter = (
        MIND2WEB_RANKED_PROMPT_ADAPTER_VERSION
        if mode == "production"
        else MIND2WEB_PROMPT_ADAPTER_VERSION
    )
    adapter_implementation = (
        _adapter_implementation_identity() if mode == "production" else None
    )
    audit_schema_version = (
        MIND2WEB_RANKED_AUDIT_SCHEMA_VERSION
        if mode == "production"
        else MIND2WEB_AUDIT_SCHEMA_VERSION
    )
    if not sources:
        raise ValueError("at least one immutable Mind2Web source is required")
    if len(sources) > max_sources:
        raise ValueError(f"Mind2Web export exceeds max_sources={max_sources}")
    expected_member_splits: dict[str, str] = {}
    if mode == "production":
        if archive is None:
            raise ValueError(
                "production Mind2Web export requires the protected test.zip archive"
            )
        expected_archive_identity = (
            PRODUCTION_MIND2WEB_ARCHIVE_BYTES,
            PRODUCTION_MIND2WEB_ARCHIVE_SHA256,
        )
        if (archive.bytes, archive.sha256) != expected_archive_identity:
            raise ValueError(
                "production Mind2Web archive identity mismatch: "
                f"expected bytes={expected_archive_identity[0]}, "
                f"sha256={expected_archive_identity[1]}"
            )
        expected_member_splits = _production_member_splits()
        declared_members: list[str] = []
        for source in sources:
            if not isinstance(source.archive_member, str) or not source.archive_member:
                raise ValueError(
                    "production Mind2Web sources require an exact archive_member"
                )
            _validate_archive_member_name(source.archive_member)
            declared_members.append(source.archive_member)
        if len(declared_members) != len(set(declared_members)):
            raise ValueError("production Mind2Web sources contain duplicate archive_member values")
        expected_members = frozenset(expected_member_splits)
        observed_members = frozenset(declared_members)
        if observed_members != expected_members:
            missing = sorted(expected_members - observed_members)
            extra = sorted(observed_members - expected_members)
            raise ValueError(
                "production Mind2Web extracted source member set mismatch: "
                f"missing={missing}, extra={extra}"
            )
    else:
        if archive is not None:
            raise ValueError("fixture Mind2Web export must not declare a production archive")
        if any(source.archive_member is not None for source in sources):
            raise ValueError("fixture Mind2Web sources must not declare archive_member")

    output = Path(output_path)
    audit_output = Path(audit_path) if audit_path is not None else None
    ranker_config_file = (
        ranker_config.path if ranker_config is not None else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() == audit_output.resolve():
            raise ValueError("output_path and audit_path must be different files")
    if ranker_config_file is not None and (
        ranker_config_file.resolve() == output.resolve()
        or (
            audit_output is not None
            and ranker_config_file.resolve() == audit_output.resolve()
        )
    ):
        raise ValueError("ranker config and output paths must be distinct")

    declared: list[tuple[Mind2WebSource, Path]] = []
    names: set[str] = set()
    for source in sources:
        path = _validated_source_path(source, max_source_bytes=max_source_bytes)
        name = source.archive_member if mode == "production" else path.name
        if name is None:
            raise AssertionError("production source lost its archive member")
        if name in names:
            raise ValueError(f"Mind2Web source identities must be unique: {name!r}")
        names.add(name)
        if path.resolve() == output.resolve() or (
            audit_output is not None and path.resolve() == audit_output.resolve()
        ):
            raise ValueError("source and output paths must be distinct")
        if (
            ranker_config_file is not None
            and path.resolve() == ranker_config_file.resolve()
        ):
            raise ValueError("source and ranker config paths must be distinct")
        declared.append((source, path))
    total_source_bytes = sum(source.bytes for source, _ in declared)
    if total_source_bytes > max_total_source_bytes:
        raise ValueError(
            "Mind2Web sources exceed "
            f"max_total_source_bytes={max_total_source_bytes}"
        )
    if (
        mode == "production"
        and total_source_bytes != PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES
    ):
        raise ValueError(
            "production Mind2Web extracted source byte total mismatch: "
            f"expected {PRODUCTION_MIND2WEB_TOTAL_UNCOMPRESSED_BYTES}, "
            f"got {total_source_bytes}"
        )
    declared.sort(
        key=lambda item: (
            item[0].archive_member
            if item[0].archive_member is not None
            else item[1].name
        )
    )

    archive_path: Path | None = None
    if archive is not None:
        archive_path = _validated_source_path(
            archive,
            max_source_bytes=max_archive_bytes,
            label="archive",
        )
        if archive_path.resolve() == output.resolve() or (
            audit_output is not None and archive_path.resolve() == audit_output.resolve()
        ):
            raise ValueError("archive and output paths must be distinct")
        if any(archive_path.resolve() == path.resolve() for _, path in declared):
            raise ValueError("archive and extracted source paths must be distinct")
        if (
            ranker_config_file is not None
            and archive_path.resolve() == ranker_config_file.resolve()
        ):
            raise ValueError("archive and ranker config paths must be distinct")

    staged_output_name: str | None = None
    staged_audit_name: str | None = None
    production_archive_audit: dict[str, int | str] | None = None
    production_member_audits: list[dict[str, int | str]] = []
    tasks_by_split = {
        split_name: 0
        for split_name in (
            PRODUCTION_MIND2WEB_TASK_COUNTS if mode == "production" else ()
        )
    }
    ranking_totals = _RankingCounters()
    ranking_source_audits: list[dict[str, Any]] = []
    ranking_receipts_sha256: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="localagent-mind2web-prompts-") as temporary:
            temporary_path = Path(temporary)
            snapshot_directory = temporary_path / "sources"
            snapshot_directory.mkdir(mode=0o700)
            verified: list[tuple[Path, dict[str, int | str]]] = []
            for index, (source, path) in enumerate(declared):
                snapshot, identity = _snapshot_verified_source(
                    source,
                    snapshot_directory / f"{index:04d}-{path.name}",
                    max_source_bytes=max_source_bytes,
                )
                if source.archive_member is not None:
                    identity["archive_member"] = source.archive_member
                    identity["split"] = expected_member_splits[source.archive_member]
                verified.append((snapshot, identity))

            if mode == "production":
                if archive is None or archive_path is None:
                    raise AssertionError("production archive disappeared after validation")
                archive_snapshot, production_archive_audit = _snapshot_verified_source(
                    archive,
                    temporary_path / "protected-test.zip",
                    max_source_bytes=max_archive_bytes,
                    label="archive",
                )
                production_member_audits = _attest_production_archive(
                    archive_snapshot,
                    source_identities={
                        str(identity["archive_member"]): identity
                        for _, identity in verified
                    },
                    max_source_bytes=max_source_bytes,
                    max_total_source_bytes=max_total_source_bytes,
                    max_sources=max_sources,
                    max_compression_ratio=max_compression_ratio,
                )

            database_path = temporary_path / "rows.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute(
                    """
                    CREATE TABLE tasks (
                        task_identity TEXT PRIMARY KEY,
                        source_name TEXT NOT NULL
                    ) WITHOUT ROWID
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE prompt_rows (
                        source_name TEXT NOT NULL,
                        sort_key TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        source_case_id TEXT NOT NULL UNIQUE,
                        prompt TEXT NOT NULL,
                        ranking_receipt TEXT NOT NULL,
                        PRIMARY KEY (source_name, sort_key, step_index)
                    ) WITHOUT ROWID
                    """
                )
                source_audits: list[dict[str, int | str]] = []
                total_tasks = 0
                total_rows = 0
                for path, identity in verified:
                    source_name = str(identity.get("archive_member", identity["name"]))
                    source_rows = 0
                    source_tasks = 0
                    source_ranking = _RankingCounters()
                    for task_index, task in enumerate(
                        _iter_json_array(
                            path,
                            max_bytes=max_source_bytes,
                            max_record_chars=max_record_chars,
                        )
                    ):
                        if mode == "production":
                            if ranker_config is None:
                                raise AssertionError("production ranker config disappeared")
                            rows, task_ranking = _stage_ranked_task(
                                connection,
                                task,
                                revision=revision,
                                source_name=source_name,
                                source_index=task_index,
                                ranker_config=ranker_config,
                            )
                            source_ranking.merge(task_ranking)
                        else:
                            rows, _ = _stage_task(
                                connection,
                                task,
                                revision=revision,
                                source_name=source_name,
                                source_index=task_index,
                                max_prompt_bytes=effective_max_prompt_bytes,
                            )
                        source_rows += rows
                        source_tasks += 1
                        total_rows += rows
                        if total_rows > max_rows:
                            raise ValueError(
                                f"Mind2Web export exceeds max_rows={max_rows}"
                            )
                    if source_tasks == 0:
                        raise ValueError(f"Mind2Web source contains no tasks: {path}")
                    total_tasks += source_tasks
                    if mode == "production":
                        source_split = str(identity["split"])
                        tasks_by_split[source_split] += source_tasks
                        ranking_totals.merge(source_ranking)
                        ranking_source_audits.append(
                            {
                                "source": source_name,
                                **source_ranking.as_audit(),
                            }
                        )
                    source_audits.append(
                        {
                            **identity,
                            "rows": source_rows,
                            "tasks": source_tasks,
                        }
                    )
                if mode == "production":
                    expected_task_counts = dict(PRODUCTION_MIND2WEB_TASK_COUNTS)
                    if tasks_by_split != expected_task_counts:
                        raise ValueError(
                            "production Mind2Web task counts mismatch: "
                            f"expected {expected_task_counts}, got {tasks_by_split}"
                        )
                    source_audit_by_member = {
                        str(source["archive_member"]): source for source in source_audits
                    }
                    for member_audit in production_member_audits:
                        source_audit = source_audit_by_member[str(member_audit["member"])]
                        member_audit["rows"] = int(source_audit["rows"])
                        member_audit["tasks"] = int(source_audit["tasks"])
                connection.commit()

                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=output.parent,
                    prefix=f".{output.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as staged_handle:
                    staged_output_name = staged_handle.name
                (
                    written_rows,
                    output_bytes,
                    output_sha256,
                    case_ids_sha256,
                    ranking_receipts_sha256,
                ) = _write_staged_rows(
                    connection,
                    Path(staged_output_name),
                    max_output_bytes=max_output_bytes,
                )
                if written_rows != total_rows:
                    raise AssertionError("staged Mind2Web row count changed")
                if mode == "production" and (
                    ranking_receipts_sha256 is None
                    or ranking_totals.rows != written_rows
                ):
                    raise AssertionError("ranked Mind2Web row accounting changed")
            finally:
                connection.close()

        if mode == "production":
            if ranker_config is None or adapter_implementation is None:
                raise AssertionError("production implementation binding disappeared")
            replayed_config = load_mind2web_dom_ranker_config(ranker_config.path)
            if (
                replayed_config.sha256 != ranker_config.sha256
                or dict(replayed_config.implementation)
                != dict(ranker_config.implementation)
                or dict(replayed_config.runtime) != dict(ranker_config.runtime)
                or implementation_identity() != dict(ranker_config.implementation)
                or runtime_identity() != dict(ranker_config.runtime)
                or _adapter_implementation_identity() != adapter_implementation
            ):
                raise ValueError(
                    "ranker config, ranker/adapter implementation, or runtime "
                    "changed during export"
                )

        source_attestation: dict[str, Any] | None = None
        if mode == "production":
            if production_archive_audit is None:
                raise AssertionError("production archive audit was not created")
            source_attestation = {
                "archive": production_archive_audit,
                "archive_format": {
                    "compression": "deflate",
                    "encryption": (
                        "zipcrypto"
                        if PRODUCTION_MIND2WEB_ARCHIVE_ENCRYPTED
                        else "fixture_unencrypted"
                    ),
                    "members": len(production_member_audits),
                },
                "kind": MIND2WEB_ARCHIVE_ATTESTATION_KIND,
                "members": production_member_audits,
                "members_sha256": _sha256(
                    _canonical_json_bytes(production_member_audits)
                ),
                "schema_version": MIND2WEB_ARCHIVE_ATTESTATION_SCHEMA_VERSION,
                "tasks_by_split": tasks_by_split,
                "total_tasks": sum(tasks_by_split.values()),
                "total_uncompressed_bytes": sum(
                    int(member["bytes"]) for member in production_member_audits
                ),
            }

        freeze_binding: dict[str, Any] = {
            "adapter": active_adapter,
            "benchmark": "mind2web",
            "mode": mode,
            "revision": revision,
            "split": split,
            "prompt_only": True,
            "contains_current_step_labels": False,
            "output": {
                "bytes": output_bytes,
                "sha256": output_sha256,
                "records": written_rows,
            },
        }
        ranking_audit: dict[str, Any] | None = None
        if mode == "production":
            if (
                ranker_config is None
                or ranking_receipts_sha256 is None
                or adapter_implementation is None
            ):
                raise AssertionError("production ranking audit inputs disappeared")
            ranker_identity = ranker_config.audit_identity()
            freeze_binding["ranker"] = ranker_identity
            freeze_binding["adapter_implementation"] = adapter_implementation
            ranking_audit = {
                "adapter_implementation": adapter_implementation,
                "budget": {
                    "assistant_marker_bytes": len(
                        ranker_config.assistant_marker.encode("utf-8")
                    ),
                    "generation_reserve_tokens_including_eos": (
                        ranker_config.generation_reserve_tokens_including_eos
                    ),
                    "max_framed_prompt_bytes": (
                        ranker_config.max_framed_prompt_bytes
                    ),
                    "max_unframed_prompt_bytes": (
                        ranker_config.max_unframed_prompt_bytes
                    ),
                    "minimum_dom_bytes": ranker_config.minimum_dom_bytes,
                    "model_max_seq_len": ranker_config.model_max_seq_len,
                    "user_marker_bytes": len(
                        ranker_config.user_marker.encode("utf-8")
                    ),
                },
                "dependencies": {
                    "action_representations_used_by_ranker": False,
                    "action_uids_used_by_ranker": False,
                    "model_used": False,
                    "negative_candidates_used_by_ranker": False,
                    "operations_used_by_ranker": False,
                    "positive_candidates_used_by_ranker": False,
                    "raw_html_used_by_ranker": False,
                    "tokenizer_used": False,
                },
                "input_projection": {
                    "allowed": ["confirmed_task", "cleaned_html"],
                    "forbidden": [
                        "action_reprs",
                        "action_uid",
                        "neg_candidates",
                        "operation",
                        "pos_candidates",
                        "raw_html",
                    ],
                },
                "kind": MIND2WEB_DOM_RANKING_AUDIT_KIND,
                "ordered_row_receipts_sha256": ranking_receipts_sha256,
                "ranker": ranker_identity,
                "recall_ceiling_measured": False,
                "schema_version": MIND2WEB_DOM_RANKING_AUDIT_SCHEMA_VERSION,
                "scores_emitted": False,
                "sources": ranking_source_audits,
                "totals": ranking_totals.as_audit(),
            }

        audit_without_hash: dict[str, Any] = {
            "adapter": active_adapter,
            "adapter_version": active_adapter,
            "benchmark": "mind2web",
            "freeze_binding": freeze_binding,
            "kind": MIND2WEB_AUDIT_KIND,
            "label_isolation": {
                "current_action_emitted": False,
                "expected_calls_emitted": False,
                "negative_candidates_emitted": False,
                "positive_candidates_emitted": False,
                "prior_action_representations_emitted": mode == "fixture",
                "scores_emitted": False,
            },
            "limits": {
                "max_archive_bytes": max_archive_bytes,
                "max_compression_ratio": max_compression_ratio,
                "max_output_bytes": max_output_bytes,
                "max_prompt_bytes": effective_max_prompt_bytes,
                "max_record_chars": max_record_chars,
                "max_rows": max_rows,
                "max_source_bytes": max_source_bytes,
                "max_sources": max_sources,
                "max_total_source_bytes": max_total_source_bytes,
            },
            "ordering": (
                "archive_member_then_upstream_task_index_then_upstream_step_index"
                if mode == "production"
                else "source_basename_then_annotation_id_then_upstream_step_index"
            ),
            "mode": mode,
            "output": {
                "bytes": output_bytes,
                "path": output.name,
                "rows": written_rows,
                "sha256": output_sha256,
                "source_case_ids_sha256": case_ids_sha256,
            },
            "privacy": {
                "contains_private_heldout_prompts": True,
                "redistribution_authorized": False,
            },
            "purpose": "prompt_only_pretraining_decontamination_not_official_scoring",
            "revision": revision,
            "schema_version": audit_schema_version,
            "sources": source_audits,
            "split": split,
            "tasks": total_tasks,
            **(
                {"source_attestation": source_attestation}
                if source_attestation is not None
                else {}
            ),
            **({"ranking": ranking_audit} if ranking_audit is not None else {}),
        }
        audit = {
            **audit_without_hash,
            "audit_self_sha256": _sha256(_canonical_json_bytes(audit_without_hash)),
        }
        audit_payload = _canonical_json_bytes(audit) + b"\n"
        audit_sha256 = _sha256(audit_payload)

        _preflight_output(
            output,
            expected_bytes=output_bytes,
            expected_sha256=output_sha256,
            label="Mind2Web prompt export",
        )
        if audit_output is not None:
            _preflight_output(
                audit_output,
                expected_bytes=len(audit_payload),
                expected_sha256=audit_sha256,
                label="Mind2Web adapter audit",
            )
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=audit_output.parent,
                prefix=f".{audit_output.name}.",
                suffix=".tmp",
                delete=False,
            ) as audit_handle:
                staged_audit_name = audit_handle.name
                audit_handle.write(audit_payload)
                audit_handle.flush()
                os.fsync(audit_handle.fileno())

        published_output_identity: tuple[int, int] | None = None
        try:
            published_output_identity = _publish_staged(
                Path(staged_output_name),
                output,
                expected_bytes=output_bytes,
                expected_sha256=output_sha256,
                label="Mind2Web prompt export",
            )
            if audit_output is not None and staged_audit_name is not None:
                _publish_staged(
                    Path(staged_audit_name),
                    audit_output,
                    expected_bytes=len(audit_payload),
                    expected_sha256=audit_sha256,
                    label="Mind2Web adapter audit",
                )
        except Exception:
            if (
                audit_output is not None
                and published_output_identity is not None
            ):
                _unlink_if_same_inode(output, published_output_identity)
            raise
        return audit
    finally:
        if staged_output_name is not None:
            Path(staged_output_name).unlink(missing_ok=True)
        if staged_audit_name is not None:
            Path(staged_audit_name).unlink(missing_ok=True)
