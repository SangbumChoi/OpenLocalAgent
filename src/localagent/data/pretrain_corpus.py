"""Deterministic pretraining corpus preparation.

The on-disk format is deliberately boring and inspectable:

* raw documents are UTF-8 ``.txt`` or JSONL with a ``text``/``content``/``code`` field;
* filtering emits ``filtered.jsonl`` with provenance and a stable document id;
* packing emits two-dimensional NumPy token shards plus a ``manifest.json``;
* train/validation assignment is document-level, so a document cannot leak across splits.

Rows contain ``seq_len + 1`` tokens.  A matching lengths shard records the non-padding length,
allowing the loader to mask padding from next-token loss.  This is a small, dependency-free
version of the BOS-aligned packing used by nanochat and is suitable for memory mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import urllib.request
import uuid
from bisect import insort
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

SAMPLE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
MANIFEST_VERSION = 2
STAGING_VERSION = 2
SPLIT_ASSIGNMENT_FORMAT = "localagent_document_split_jsonl"
SPLIT_ASSIGNMENT_VERSION = 1
MAX_MANIFEST_GROUPS = 64
DEFAULT_MAX_RAW_DOCUMENT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_EVALUATION_DENYLIST_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_EVALUATION_DENYLIST_ENTRIES = 1_000_000
MAX_SPLIT_ASSIGNMENT_LINE_BYTES = 1024 * 1024
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_SHINGLE_HASH_PERSON = b"localagent-data"
_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}
_CODE_SOURCE_HINTS = (
    "codeparrot",
    "github",
    "starcoder",
    "the-stack",
    "source-code",
)
_HASH_TEXT_CHARS = 64 * 1024


def _update_framed_utf8(digest: Any, label: str, value: str) -> None:
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    for start in range(0, len(value), _HASH_TEXT_CHARS):
        encoded = value[start : start + _HASH_TEXT_CHARS].encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    digest.update((0).to_bytes(4, "big"))


def _sha256_utf8(value: str) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(value), _HASH_TEXT_CHARS):
        digest.update(value[start : start + _HASH_TEXT_CHARS].encode("utf-8"))
    return digest.hexdigest()


def _update_raw_document_stream_hash(digest: Any, document: CorpusDocument) -> None:
    """Hash raw provenance with bounded text encoding allocations."""

    digest.update(b"localagent-raw-document-v2\0")
    _update_framed_utf8(digest, "text", document.text)
    _update_framed_utf8(digest, "source", document.source)
    _update_framed_utf8(digest, "doc_id", document.doc_id)
    _update_framed_utf8(digest, "license", document.license)
    _update_framed_utf8(digest, "meta", _canonical_meta(document.meta))


@dataclass(frozen=True)
class CorpusDocument:
    """A pretraining document with enough provenance to audit or remove it later."""

    text: str
    source: str = "unknown"
    doc_id: str = ""
    license: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def with_stable_id(self) -> CorpusDocument:
        if self.doc_id:
            return self
        digest = _sha256_utf8(self.text)
        return CorpusDocument(self.text, self.source, digest, self.license, self.meta)


@dataclass(frozen=True)
class FrozenSplitAssignment:
    """Verified reference to an immutable document/content-to-split assignment artifact."""

    path: Path
    bytes: int
    sha256: str
    records: int
    assignment_sha256: str
    seed: int
    val_fraction: float
    source_manifest: dict[str, Any]


def _coerce_document(doc: CorpusDocument | str | dict[str, Any]) -> CorpusDocument:
    if isinstance(doc, CorpusDocument):
        return doc.with_stable_id()
    if isinstance(doc, str):
        return CorpusDocument(doc).with_stable_id()
    for key in ("text", "content", "code"):
        if isinstance(doc.get(key), str):
            return CorpusDocument(
                text=doc[key],
                source=str(doc.get("source", "unknown")),
                doc_id=str(doc.get("doc_id", doc.get("id", ""))),
                license=str(doc.get("license", "unknown")),
                meta=dict(doc.get("meta", {})),
            ).with_stable_id()
    raise ValueError("document dict needs a string 'text', 'content', or 'code' field")


def iter_documents(
    paths: str | Path | Sequence[str | Path],
    *,
    max_document_bytes: int = DEFAULT_MAX_RAW_DOCUMENT_BYTES,
) -> Iterator[CorpusDocument]:
    """Read bounded plain-text and JSONL documents without a dataset framework.

    The byte guard is enforced while reading, rather than after an unbounded ``read()`` or JSONL
    line allocation. It protects local ingestion; upstream dataset iterators must impose their own
    transport/row limits before yielding already-materialized Python strings.
    """

    if max_document_bytes < 1:
        raise ValueError("max_document_bytes must be positive")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            expanded.append(path)

    for path in expanded:
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("rb") as handle:
                line_no = 0
                while True:
                    line = handle.readline(max_document_bytes + 1)
                    if not line:
                        break
                    line_no += 1
                    if len(line) > max_document_bytes:
                        raise ValueError(
                            f"{path}:{line_no}: JSONL record exceeds {max_document_bytes} bytes"
                        )
                    if not line.strip():
                        continue
                    raw = json.loads(line.decode("utf-8"))
                    raw.setdefault("source", str(path))
                    # A line number is not document identity: reordering otherwise changes the
                    # split and audit hashes. _coerce_document derives a content id when neither
                    # doc_id nor id is supplied, while preserving explicit upstream identifiers.
                    yield _coerce_document(raw)
        elif suffix in {".txt", ".md", ".rst", ".py", ".js", ".ts", ".java", ".rs", ".go"}:
            with path.open("rb") as handle:
                payload = handle.read(max_document_bytes + 1)
            if len(payload) > max_document_bytes:
                raise ValueError(f"{path}: document exceeds {max_document_bytes} bytes")
            text = payload.decode("utf-8", errors="replace")
            yield CorpusDocument(text, source=str(path), doc_id=str(path)).with_stable_id()


def download_sample(out_dir: str, *, url: str = SAMPLE_URL) -> Path:
    """Fetch a public-domain toy corpus and write auditable JSONL.

    The sample is only for smoke tests; it is not presented as a useful production corpus.
    A temporary file is replaced atomically so an interrupted download is never treated as valid.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "gutenberg_alice.jsonl"
    if destination.exists():
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "LocalAgent/0.0.1 corpus-smoke"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    # Split at chapter boundaries to exercise document-level splitting and packing.
    pieces = [p.strip() for p in re.split(r"(?im)^\s*chapter\s+[ivxlcdm0-9]+\b.*$", text)]
    pieces = [p for p in pieces if len(p) >= 200]
    tmp = destination.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for index, piece in enumerate(pieces):
            doc = CorpusDocument(
                text=piece,
                source=url,
                doc_id=f"alice-{index:03d}",
                license="Project Gutenberg public domain",
            )
            handle.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    tmp.replace(destination)
    return destination


def _repetition_ratio(text: str) -> float:
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 20]
    if not lines:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def _canonical_meta(meta: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(meta),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _quality_candidate(
    raw: CorpusDocument | str | dict[str, Any],
    *,
    min_chars: int,
    max_chars: int,
    max_control_ratio: float,
    max_repetition_ratio: float,
) -> tuple[CorpusDocument | None, str | None, str]:
    doc = _coerce_document(raw)
    # Reject before newline normalization/strip can copy an arbitrarily large Python string.
    # This bounds our additional allocation even when an upstream iterator already materialized it.
    if len(doc.text) > max_chars:
        return None, None, "too_long"
    text = doc.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) < min_chars:
        return None, None, "too_short"
    controls = sum(ord(ch) < 32 and ch not in "\n\t" for ch in text)
    if controls / max(1, len(text)) > max_control_ratio:
        return None, None, "control_characters"
    if "\ufffd" in text and text.count("\ufffd") / len(text) > 0.001:
        return None, None, "replacement_characters"
    if _repetition_ratio(text) > max_repetition_ratio:
        return None, None, "repetitive"
    # Preserve case: case-only changes can alter identifiers and behavior in source code.
    normalized = re.sub(r"\s+", " ", text).strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return (
        CorpusDocument(
            text,
            doc.source,
            doc.doc_id or digest,
            doc.license,
            doc.meta,
        ),
        digest,
        "",
    )


def _exact_dedup_key(document: CorpusDocument) -> tuple[str, str, str, bytes, str]:
    return (
        document.doc_id,
        document.source,
        document.license,
        hashlib.sha256(document.text.encode("utf-8")).digest(),
        _canonical_meta(document.meta),
    )


def quality_filter(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    *,
    min_chars: int = 200,
    max_chars: int = 2_000_000,
    max_control_ratio: float = 0.01,
    max_repetition_ratio: float = 0.35,
    deduplicate: bool = True,
) -> list[CorpusDocument]:
    """Apply deterministic, explainable quality filters and exact-content deduplication.

    The thresholds intentionally retain source code and structured tool traces; an
    English-only alphabetic-ratio filter would discard precisely the data this project needs.
    """

    accepted: list[CorpusDocument] = []
    accepted_by_digest: dict[str, CorpusDocument] = {}
    for raw in docs:
        candidate, digest, _ = _quality_candidate(
            raw,
            min_chars=min_chars,
            max_chars=max_chars,
            max_control_ratio=max_control_ratio,
            max_repetition_ratio=max_repetition_ratio,
        )
        if candidate is None or digest is None:
            continue
        if not deduplicate:
            accepted.append(candidate)
            continue
        previous = accepted_by_digest.get(digest)
        if previous is None or _exact_dedup_key(candidate) < _exact_dedup_key(previous):
            accepted_by_digest[digest] = candidate
    if deduplicate:
        accepted = [accepted_by_digest[digest] for digest in sorted(accepted_by_digest)]
    return accepted


def _normalized_tokens(text: str, *, case_sensitive: bool = False) -> list[str]:
    return _TOKEN_PATTERN.findall(text if case_sensitive else text.casefold())


def _normalized_match_text(text: str) -> str:
    return " ".join(_normalized_tokens(text))


def _hashed_token_shingles(
    tokens: Sequence[str],
    shingle_size: int,
    *,
    max_shingles: int | None = None,
) -> frozenset[int]:
    if shingle_size < 1:
        raise ValueError("shingle_size must be positive")
    total = len(tokens) - shingle_size + 1
    if total <= 0:
        return frozenset()
    if max_shingles is not None:
        if max_shingles < 1:
            raise ValueError("max_shingles must be positive")
        if total > max_shingles:
            positions: Iterable[int] = (
                index * total // max_shingles for index in range(max_shingles)
            )
        else:
            positions = range(total)
    else:
        positions = range(total)
    hashes: set[int] = set()
    for index in positions:
        shingle = "\0".join(tokens[index : index + shingle_size]).encode("utf-8")
        digest = hashlib.blake2b(
            shingle,
            digest_size=8,
            person=_SHINGLE_HASH_PERSON,
        ).digest()
        hashes.add(int.from_bytes(digest, "big"))
    return frozenset(hashes)


def _hashed_shingles(
    text: str,
    shingle_size: int,
    *,
    max_shingles: int | None = None,
    case_sensitive: bool = False,
) -> frozenset[int]:
    return _hashed_token_shingles(
        _normalized_tokens(text, case_sensitive=case_sensitive),
        shingle_size,
        max_shingles=max_shingles,
    )


def _is_code_like(
    text: str,
    source: str,
    meta: Mapping[str, Any] | str | None = None,
) -> bool:
    """Conservatively identify documents for which near-dedup is semantically unsafe."""

    metadata: Mapping[str, Any] = {}
    if isinstance(meta, str):
        try:
            decoded = json.loads(meta)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            metadata = decoded
    elif isinstance(meta, Mapping):
        metadata = meta

    parsed = urlsplit(source)
    source_path = parsed.path if parsed.scheme else source
    if Path(source_path).suffix.casefold() in _CODE_SUFFIXES:
        return True
    provenance = " ".join(
        [
            source,
            *(str(metadata.get(key, "")) for key in ("mixture_source", "dataset", "language")),
        ]
    ).casefold()
    if any(hint in provenance for hint in _CODE_SOURCE_HINTS):
        return True

    sample = text[:16_384]
    signals = 0
    signals += bool(re.search(r"(?m)^\s*(def|class|import|from|function|interface)\s+\w+", sample))
    signals += bool(re.search(r"(?m)^\s*(const|let|var)\s+\w+\s*=", sample))
    signals += bool(re.search(r"[{};]\s*(//.*)?$", sample, flags=re.MULTILINE))
    signals += bool(re.search(r"(?m)^\s*#(include|define|!/)", sample))
    return signals >= 2


def _simhash64(features: frozenset[int]) -> int:
    if not features:
        return 0
    scores = [-len(features)] * 64
    for feature in features:
        remaining = feature
        while remaining:
            lowest_bit = remaining & -remaining
            scores[lowest_bit.bit_length() - 1] += 2
            remaining ^= lowest_bit
    signature = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            signature |= 1 << bit
    return signature


def _fingerprint_strings(values: Iterable[str]) -> str:
    members = sorted(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values)
    return hashlib.sha256("\n".join(members).encode("ascii")).hexdigest()


def near_deduplicate(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    *,
    shingle_size: int = 5,
    max_shingles: int = 256,
    max_hamming_distance: int = 3,
    min_jaccard: float = 0.95,
    min_features: int = 8,
    lsh_bands: int = 8,
    max_bucket_size: int = 64,
    feature_cache_size: int = 128,
) -> tuple[list[CorpusDocument], dict[str, Any]]:
    """Remove conservative near duplicates with bounded SimHash LSH candidate search.

    Documents are processed in a canonical quality order (longer first, then stable hashes), so
    the retained membership and audit fingerprint do not depend on input order. SimHash only
    proposes candidates; sampled shingle Jaccard makes the removal decision. The LSH buckets and
    feature cache are bounded, so this is a scalable heuristic rather than an exhaustive
    all-pairs duplicate proof.
    """

    if not 0 <= max_hamming_distance < lsh_bands:
        raise ValueError("max_hamming_distance must be in [0, lsh_bands)")
    if 64 % lsh_bands:
        raise ValueError("lsh_bands must divide 64")
    if not 0.0 < min_jaccard <= 1.0:
        raise ValueError("min_jaccard must be in (0, 1]")
    if min_features < 1 or max_bucket_size < 1 or feature_cache_size < 1:
        raise ValueError("near-dedup bounds must be positive")
    if max_shingles < min_features:
        raise ValueError("max_shingles must be >= min_features")

    prepared = [_coerce_document(document) for document in docs]
    ranked = sorted(
        prepared,
        key=lambda document: (
            -len(document.text),
            hashlib.sha256(document.text.encode("utf-8")).digest(),
            document.doc_id,
            document.source,
        ),
    )
    band_bits = 64 // lsh_bands
    band_mask = (1 << band_bits) - 1
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    retained: list[CorpusDocument] = []
    signatures: list[int] = []
    feature_cache: OrderedDict[int, frozenset[int]] = OrderedDict()
    removed: list[CorpusDocument] = []
    removal_pairs: list[str] = []
    candidate_pairs = 0
    jaccard_checks = 0
    bucket_overflows = 0
    code_documents_bypassed = 0

    def cached_features(index: int) -> frozenset[int]:
        if index in feature_cache:
            value = feature_cache.pop(index)
            feature_cache[index] = value
            return value
        value = _hashed_shingles(
            retained[index].text,
            shingle_size,
            max_shingles=max_shingles,
            case_sensitive=True,
        )
        feature_cache[index] = value
        if len(feature_cache) > feature_cache_size:
            feature_cache.popitem(last=False)
        return value

    for document in ranked:
        if _is_code_like(document.text, document.source, document.meta):
            # Exact-content deduplication happens before this stage in the corpus pipeline.
            # Heuristic near-dedup is unsafe for code because a case or identifier-only change
            # can alter behavior while preserving almost all shingles.
            retained.append(document)
            signatures.append(0)
            code_documents_bypassed += 1
            continue
        features = _hashed_shingles(
            document.text,
            shingle_size,
            max_shingles=max_shingles,
            case_sensitive=True,
        )
        if len(features) < min_features:
            retained.append(document)
            signatures.append(0)
            continue
        signature = _simhash64(features)
        band_keys = [
            (band, (signature >> (band * band_bits)) & band_mask) for band in range(lsh_bands)
        ]
        candidates = sorted(
            {index for band_key in band_keys for index in buckets.get(band_key, ())}
        )
        duplicate_of: CorpusDocument | None = None
        for index in candidates:
            candidate_pairs += 1
            if (signature ^ signatures[index]).bit_count() > max_hamming_distance:
                continue
            jaccard_checks += 1
            other_features = cached_features(index)
            union_size = len(features | other_features)
            similarity = len(features & other_features) / max(1, union_size)
            if similarity >= min_jaccard:
                duplicate_of = retained[index]
                break
        if duplicate_of is not None:
            removed.append(document)
            removal_pairs.append(
                f"{_document_identity(document)}:{_document_identity(duplicate_of)}"
            )
            continue

        index = len(retained)
        retained.append(document)
        signatures.append(signature)
        feature_cache[index] = features
        if len(feature_cache) > feature_cache_size:
            feature_cache.popitem(last=False)
        for band_key in band_keys:
            bucket = buckets[band_key]
            if len(bucket) < max_bucket_size:
                bucket.append(index)
            else:
                bucket_overflows += 1

    audit = {
        "enabled": True,
        "method": ("case_sensitive_simhash64_lsh_then_sampled_shingle_jaccard_with_code_bypass"),
        "exhaustive": False,
        "input_documents": len(prepared),
        "retained_documents": len(retained),
        "removed_documents": len(removed),
        "retained_document_ids_sha256": _document_ids_sha256(retained),
        "removed_document_ids_sha256": _document_ids_sha256(removed),
        "removal_pairs_sha256": _fingerprint_strings(removal_pairs),
        "shingle_size": shingle_size,
        "max_shingles_per_document": max_shingles,
        "max_hamming_distance": max_hamming_distance,
        "min_jaccard": min_jaccard,
        "min_features": min_features,
        "lsh_bands": lsh_bands,
        "max_bucket_size": max_bucket_size,
        "candidate_pairs": candidate_pairs,
        "jaccard_checks": jaccard_checks,
        "bucket_overflows": bucket_overflows,
        "case_sensitive_shingles": True,
        "code_documents_bypassed": code_documents_bypassed,
        "code_policy": "exact_content_deduplication_only",
        "limitations": (
            "Bounded heuristic screening; LSH bucket caps and sampled shingles can miss "
            "near duplicates. Code-like documents bypass heuristic near-dedup after exact "
            "content deduplication to preserve potentially semantic case/identifier changes."
        ),
    }
    return retained, audit


def _denylist_text_from_record(record: dict[str, Any]) -> str | None:
    for key in ("prompt", "query", "instruction", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = record.get("messages")
    if isinstance(messages, list):
        user_parts = [
            message.get("content", "")
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        if user_parts:
            return "\n".join(user_parts)
    return None


def read_evaluation_denylist(
    paths: str | Path | Sequence[str | Path],
    *,
    max_total_bytes: int = DEFAULT_MAX_EVALUATION_DENYLIST_BYTES,
    max_entries: int = DEFAULT_MAX_EVALUATION_DENYLIST_ENTRIES,
    max_record_bytes: int = DEFAULT_MAX_RAW_DOCUMENT_BYTES,
) -> list[str]:
    """Read bounded evaluation prompts from JSON, JSONL, or one-prompt-per-line text files."""

    for name, value in (
        ("max_total_bytes", max_total_bytes),
        ("max_entries", max_entries),
        ("max_record_bytes", max_record_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if isinstance(paths, (str, Path)):
        paths = [paths]
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(
                sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
            )
        else:
            expanded.append(path)

    entries: list[str] = []
    total_bytes = 0

    def account_bytes(path: Path, count: int) -> None:
        nonlocal total_bytes
        total_bytes += count
        if total_bytes > max_total_bytes:
            raise ValueError(
                f"evaluation denylist inputs exceed {max_total_bytes} total bytes at {path}"
            )

    def append_entry(path: Path, text: str) -> None:
        entries.append(text)
        if len(entries) > max_entries:
            raise ValueError(
                f"evaluation denylist inputs exceed {max_entries} prompt entries at {path}"
            )

    for path in expanded:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open("rb") as handle:
                payload = handle.read(max_record_bytes + 1)
            account_bytes(path, len(payload))
            if len(payload) > max_record_bytes:
                raise ValueError(
                    f"{path}: benchmark suite exceeds {max_record_bytes} bytes"
                )
            suite = json.loads(payload.decode("utf-8"))
            if not isinstance(suite, dict):
                raise ValueError(f"{path}: benchmark suite must be a JSON object")
            version = suite.get("schema_version", suite.get("version"))
            if (
                isinstance(version, bool)
                or not isinstance(version, (int, str))
                or not str(version).strip()
            ):
                raise ValueError(f"{path}: benchmark suite needs a schema_version/version")
            cases = suite.get("cases")
            if not isinstance(cases, list):
                raise ValueError(f"{path}: benchmark suite needs a top-level cases array")
            for case_index, case in enumerate(cases):
                if not isinstance(case, dict):
                    raise ValueError(f"{path}: cases[{case_index}] must be an object")
                text = _denylist_text_from_record(case)
                if text is None:
                    raise ValueError(
                        f"{path}: cases[{case_index}] needs a query/prompt/text/messages field"
                    )
                append_entry(path, text)
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("rb") as handle:
                line_no = 0
                while True:
                    raw = handle.readline(max_record_bytes + 1)
                    if not raw:
                        break
                    line_no += 1
                    account_bytes(path, len(raw))
                    if len(raw) > max_record_bytes:
                        raise ValueError(
                            f"{path}:{line_no}: denylist row exceeds "
                            f"{max_record_bytes} bytes"
                        )
                    if not raw.strip():
                        continue
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"{path}:{line_no}: invalid UTF-8 JSON denylist row"
                        ) from error
                    if not isinstance(record, dict):
                        raise ValueError(f"{path}:{line_no}: denylist JSONL row must be an object")
                    text = _denylist_text_from_record(record)
                    if text is None:
                        raise ValueError(
                            f"{path}:{line_no}: denylist row needs a prompt/text/messages field"
                        )
                    append_entry(path, text)
        else:
            with path.open("rb") as handle:
                line_no = 0
                while True:
                    raw = handle.readline(max_record_bytes + 1)
                    if not raw:
                        break
                    line_no += 1
                    account_bytes(path, len(raw))
                    if len(raw) > max_record_bytes:
                        raise ValueError(
                            f"{path}:{line_no}: denylist line exceeds "
                            f"{max_record_bytes} bytes"
                        )
                    try:
                        line = raw.decode("utf-8").strip()
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            f"{path}:{line_no}: denylist line is not valid UTF-8"
                        ) from error
                    if line:
                        append_entry(path, line)

    unique: dict[str, str] = {}
    for entry in entries:
        normalized = _normalized_match_text(entry)
        if normalized:
            unique.setdefault(normalized, entry)
    return [unique[key] for key in sorted(unique)]


@dataclass
class _EvaluationDenylistMatcher:
    normalized_entries: list[str]
    long_entries: list[tuple[str, frozenset[int]]]
    short_entries: list[str]
    anchor_index: dict[int, list[int]]
    shingle_size: int
    min_shingles: int
    min_coverage: float
    anchors_per_entry: int
    max_denylist_shingles: int
    candidate_checks: int = 0

    def match(self, text: str) -> str | None:
        tokens = _normalized_tokens(text)
        normalized_document = " ".join(tokens)
        for short_entry in self.short_entries:
            if short_entry in normalized_document:
                return short_entry
        if not self.long_entries:
            return None
        document_features = _hashed_token_shingles(tokens, self.shingle_size)
        candidates = sorted(
            {index for feature in document_features for index in self.anchor_index.get(feature, ())}
        )
        for index in candidates:
            self.candidate_checks += 1
            normalized_entry, entry_features = self.long_entries[index]
            coverage = len(document_features & entry_features) / len(entry_features)
            if coverage >= self.min_coverage:
                return normalized_entry
        return None


def _build_evaluation_denylist_matcher(
    denylist: Iterable[str],
    *,
    shingle_size: int,
    min_shingles: int,
    min_coverage: float,
    anchors_per_entry: int,
    max_denylist_shingles: int,
) -> _EvaluationDenylistMatcher:
    if shingle_size < 1 or min_shingles < 1 or anchors_per_entry < 1:
        raise ValueError("decontamination shingle bounds must be positive")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if max_denylist_shingles < min_shingles:
        raise ValueError("max_denylist_shingles must be >= min_shingles")

    normalized_entries = sorted(
        {normalized for text in denylist if (normalized := _normalized_match_text(str(text)))}
    )
    long_entries: list[tuple[str, frozenset[int]]] = []
    short_entries: list[str] = []
    anchor_index: dict[int, list[int]] = defaultdict(list)
    for normalized in normalized_entries:
        features = _hashed_shingles(
            normalized,
            shingle_size,
            max_shingles=max_denylist_shingles,
        )
        if len(features) < min_shingles:
            short_entries.append(normalized)
            continue
        entry_index = len(long_entries)
        long_entries.append((normalized, features))
        for anchor in sorted(features)[:anchors_per_entry]:
            anchor_index[anchor].append(entry_index)
    return _EvaluationDenylistMatcher(
        normalized_entries=normalized_entries,
        long_entries=long_entries,
        short_entries=short_entries,
        anchor_index=dict(anchor_index),
        shingle_size=shingle_size,
        min_shingles=min_shingles,
        min_coverage=min_coverage,
        anchors_per_entry=anchors_per_entry,
        max_denylist_shingles=max_denylist_shingles,
    )


def _decontamination_audit(
    matcher: _EvaluationDenylistMatcher,
    *,
    input_documents: int,
    retained_documents: int,
    removed_document_ids_sha256: str,
    matched_entries: Iterable[str],
) -> dict[str, Any]:
    matched = set(matched_entries)
    return {
        "enabled": bool(matcher.normalized_entries),
        "method": "normalized_exact_short_or_anchor_shingle_containment",
        "exhaustive": False,
        "denylist_entries": len(matcher.normalized_entries),
        "denylist_sha256": _fingerprint_strings(matcher.normalized_entries),
        "input_documents": input_documents,
        "retained_documents": retained_documents,
        "removed_documents": input_documents - retained_documents,
        "removed_document_ids_sha256": removed_document_ids_sha256,
        "matched_denylist_entries": len(matched),
        "matched_denylist_entries_sha256": _fingerprint_strings(matched),
        "shingle_size": matcher.shingle_size,
        "min_shingles": matcher.min_shingles,
        "min_coverage": matcher.min_coverage,
        "anchors_per_entry": matcher.anchors_per_entry,
        "max_denylist_shingles": matcher.max_denylist_shingles,
        "candidate_checks": matcher.candidate_checks,
        "limitations": (
            "Screens only the supplied denylist with conservative normalized/shingle matching; "
            "it is not proof that the corpus is free of benchmark contamination."
        ),
    }


def screen_evaluation_contamination(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    denylist: Iterable[str],
    *,
    shingle_size: int = 5,
    min_shingles: int = 8,
    min_coverage: float = 0.9,
    anchors_per_entry: int = 8,
    max_denylist_shingles: int = 2048,
) -> tuple[list[CorpusDocument], dict[str, Any]]:
    """Exclude documents containing denylisted evaluation prompts or close token-shingle copies.

    Long prompts use anchor-indexed shingle containment. Short prompts require an exact normalized
    substring to avoid filtering broad swaths of the corpus on generic phrases. This only screens
    the provided denylist and must not be described as proof that the corpus is contamination-free.
    """

    prepared = [_coerce_document(document) for document in docs]
    matcher = _build_evaluation_denylist_matcher(
        denylist,
        shingle_size=shingle_size,
        min_shingles=min_shingles,
        min_coverage=min_coverage,
        anchors_per_entry=anchors_per_entry,
        max_denylist_shingles=max_denylist_shingles,
    )

    retained: list[CorpusDocument] = []
    removed: list[CorpusDocument] = []
    matched_entries: set[str] = set()
    for document in prepared:
        matched = matcher.match(document.text)
        if matched is None:
            retained.append(document)
        else:
            removed.append(document)
            matched_entries.add(matched)

    audit = _decontamination_audit(
        matcher,
        input_documents=len(prepared),
        retained_documents=len(retained),
        removed_document_ids_sha256=_document_ids_sha256(removed),
        matched_entries=matched_entries,
    )
    return retained, audit


def _document_identity(document: CorpusDocument) -> str:
    """Return a stable identity suitable for split assignment and audit hashes."""

    return hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()


def _document_set_sha256(documents: Sequence[CorpusDocument]) -> str:
    identities = sorted(_document_identity(document) for document in documents)
    payload = "\n".join(identities).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _document_ids_sha256(documents: Sequence[CorpusDocument]) -> str:
    payload = "\n".join(sorted(document.doc_id for document in documents)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _joined_values_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    first = True
    for value in values:
        if not first:
            digest.update(b"\n")
        digest.update(value.encode("utf-8"))
        first = False
    return digest.hexdigest()


def _split_assignment_value(identity: str, document_sha256: str, split: str) -> str:
    return f"{identity}:{document_sha256}:{split}"


def _staging_split_assignment_sha256(connection: sqlite3.Connection) -> str:
    return _joined_values_sha256(
        _staging_values(
            connection,
            """
            SELECT identity || ':' || raw_text_sha || ':' || split
            FROM documents
            WHERE decontaminated = 1 AND near_keep = 1
            ORDER BY identity, raw_text_sha
            """,
        )
    )


def _canonical_document_json(document: CorpusDocument) -> str:
    return json.dumps(
        asdict(document),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _staging_connection(
    path: Path,
    *,
    read_only: bool = False,
    allow_thread_handoff: bool = False,
) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            check_same_thread=not allow_thread_handoff,
        )
    else:
        if allow_thread_handoff:
            raise ValueError("thread handoff is only supported for read-only staging connections")
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _staging_values(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence[Any] = (),
) -> Iterator[str]:
    for row in connection.execute(query, parameters):
        yield str(row[0])


def _staging_document_ids_sha256(
    connection: sqlite3.Connection,
    where: str,
    parameters: Sequence[Any] = (),
) -> str:
    return _joined_values_sha256(
        _staging_values(
            connection,
            f"SELECT doc_id FROM documents WHERE {where} ORDER BY doc_id",
            parameters,
        )
    )


def _staging_document_set_sha256(
    connection: sqlite3.Connection,
    where: str,
    parameters: Sequence[Any] = (),
) -> str:
    return _joined_values_sha256(
        _staging_values(
            connection,
            f"SELECT identity FROM documents WHERE {where} ORDER BY identity",
            parameters,
        )
    )


@dataclass(frozen=True)
class DiskBackedCorpus:
    """A deterministic SQLite-backed corpus view used by paper-scale preparation.

    The database owns document text and document-count-sized indexes. Iterators keep at most one
    document in Python memory, except for the explicitly bounded near-dedup feature cache and shard
    row buffer used while building it.
    """

    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with closing(_staging_connection(self.path, read_only=True)) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'staging_version'"
            ).fetchone()
        if row is None or int(json.loads(row[0])) != STAGING_VERSION:
            raise ValueError(f"{self.path} is not a supported LocalAgent staging database")

    @property
    def corpus_audit(self) -> dict[str, Any]:
        with closing(_staging_connection(self.path, read_only=True)) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'corpus_audit'"
            ).fetchone()
        if row is None:
            raise ValueError(f"{self.path} has no corpus audit")
        return dict(json.loads(row[0]))

    @property
    def staging_config(self) -> dict[str, Any]:
        with closing(_staging_connection(self.path, read_only=True)) as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'staging_config'"
            ).fetchone()
        if row is None:
            raise ValueError(f"{self.path} has no staging config")
        return dict(json.loads(row[0]))

    def iter_documents(self, split: str | None = None) -> Iterator[CorpusDocument]:
        # ``tokenizers.train_from_iterator`` can request the first item on its caller thread and
        # continue consuming on a Rust worker thread. This generator is single-consumer and
        # read-only, so permitting that sequential handoff preserves constant-memory fitting.
        connection = _staging_connection(
            self.path,
            read_only=True,
            allow_thread_handoff=True,
        )
        try:
            if split is None:
                rows = connection.execute(
                    """
                    SELECT text, source, doc_id, license, meta_json
                    FROM documents
                    WHERE decontaminated = 1 AND near_keep = 1
                    ORDER BY near_rank
                    """
                )
            else:
                if split not in {"train", "val"}:
                    raise ValueError(f"unknown split {split!r}")
                rows = connection.execute(
                    """
                    SELECT text, source, doc_id, license, meta_json
                    FROM documents
                    WHERE decontaminated = 1 AND near_keep = 1 AND split = ?
                    ORDER BY near_rank
                    """,
                    (split,),
                )
            for row in rows:
                yield CorpusDocument(
                    text=str(row["text"]),
                    source=str(row["source"]),
                    doc_id=str(row["doc_id"]),
                    license=str(row["license"]),
                    meta=dict(json.loads(row["meta_json"])),
                )
        finally:
            connection.close()

    def count(self, split: str | None = None) -> int:
        where = "decontaminated = 1 AND near_keep = 1"
        parameters: tuple[str, ...] = ()
        if split is not None:
            if split not in {"train", "val"}:
                raise ValueError(f"unknown split {split!r}")
            where += " AND split = ?"
            parameters = (split,)
        with closing(_staging_connection(self.path, read_only=True)) as connection:
            return int(
                connection.execute(
                    f"SELECT COUNT(*) FROM documents WHERE {where}",
                    parameters,
                ).fetchone()[0]
            )

    def iter_exact_dedup_aliases(self) -> Iterator[dict[str, Any]]:
        """Stream provenance for occurrences removed by normalized-content dedup."""

        connection = _staging_connection(self.path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT
                    provenance.digest,
                    provenance.provenance_sha256,
                    provenance.doc_id,
                    provenance.source,
                    provenance.license,
                    provenance.meta_json,
                    provenance.raw_text_sha,
                    provenance.identity,
                    provenance.occurrences,
                    (
                        provenance.doc_id = documents.doc_id
                        AND provenance.source = documents.source
                        AND provenance.license = documents.license
                        AND provenance.meta_json = documents.meta_json
                        AND provenance.raw_text_sha = documents.raw_text_sha
                    ) AS matches_retained_provenance
                FROM exact_dedup_aliases AS provenance
                JOIN documents USING (digest)
                ORDER BY provenance.digest, provenance.provenance_sha256
                """
            )
            for row in rows:
                yield {
                    "digest": str(row["digest"]),
                    "provenance_sha256": str(row["provenance_sha256"]),
                    "doc_id": str(row["doc_id"]),
                    "source": str(row["source"]),
                    "license": str(row["license"]),
                    "meta": dict(json.loads(row["meta_json"])),
                    "raw_text_sha256": str(row["raw_text_sha"]),
                    "identity": str(row["identity"]),
                    "occurrences": int(row["occurrences"]),
                    "matches_retained_provenance": bool(row["matches_retained_provenance"]),
                }
        finally:
            connection.close()

    def write_filtered_jsonl(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for document in self.iter_documents():
                handle.write(_canonical_document_json(document) + "\n")
        temporary.replace(destination)
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
        }

    def artifact(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
            "sha256": _sha256_file(self.path),
            "staging_version": STAGING_VERSION,
        }


def _create_staging_schema(connection: sqlite3.Connection) -> None:
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
            char_count INTEGER NOT NULL,
            identity TEXT NOT NULL,
            decontaminated INTEGER NOT NULL DEFAULT 1,
            contamination_match TEXT,
            near_keep INTEGER,
            near_rank INTEGER,
            signature_hex TEXT,
            duplicate_identity TEXT,
            removal_pair_hash TEXT,
            split TEXT
        ) WITHOUT ROWID;
        CREATE TABLE exact_dedup_aliases (
            digest TEXT NOT NULL,
            provenance_sha256 TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            license TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            raw_text_sha TEXT NOT NULL,
            identity TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY (digest, provenance_sha256)
        ) WITHOUT ROWID;
        CREATE TEMP TABLE exact_dedup_provenance_accumulator (
            digest TEXT NOT NULL,
            provenance_sha256 TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            source TEXT NOT NULL,
            license TEXT NOT NULL,
            meta_json TEXT NOT NULL,
            raw_text_sha TEXT NOT NULL,
            identity TEXT NOT NULL,
            occurrences INTEGER NOT NULL,
            PRIMARY KEY (digest, provenance_sha256)
        ) WITHOUT ROWID;
        CREATE TABLE near_queue (
            queue_rank INTEGER PRIMARY KEY,
            digest TEXT NOT NULL UNIQUE
        );
        CREATE TABLE near_index (
            band INTEGER NOT NULL,
            band_value INTEGER NOT NULL,
            near_rank INTEGER NOT NULL,
            digest TEXT NOT NULL,
            signature_hex TEXT NOT NULL,
            identity TEXT NOT NULL,
            PRIMARY KEY (band, band_value, near_rank)
        ) WITHOUT ROWID;
        CREATE TABLE near_buckets (
            band INTEGER NOT NULL,
            band_value INTEGER NOT NULL,
            entries INTEGER NOT NULL,
            PRIMARY KEY (band, band_value)
        ) WITHOUT ROWID;
        CREATE TABLE split_identities (
            identity TEXT PRIMARY KEY,
            rank_key BLOB NOT NULL,
            split TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )


def _stage_exact_documents(
    connection: sqlite3.Connection,
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    *,
    min_chars: int,
    max_chars: int,
    max_control_ratio: float,
    max_repetition_ratio: float,
) -> dict[str, Any]:
    raw_documents = 0
    quality_passed = 0
    rejection_counts: Counter[str] = Counter()
    raw_stream_hash = hashlib.sha256()
    insert = """
        INSERT INTO documents (
            digest, text, source, doc_id, license, meta_json, raw_text_sha,
            char_count, identity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(digest) DO UPDATE SET
            text = excluded.text,
            source = excluded.source,
            doc_id = excluded.doc_id,
            license = excluded.license,
            meta_json = excluded.meta_json,
            raw_text_sha = excluded.raw_text_sha,
            char_count = excluded.char_count,
            identity = excluded.identity
        WHERE (
            excluded.doc_id,
            excluded.source,
            excluded.license,
            excluded.raw_text_sha,
            excluded.meta_json
        ) < (
            documents.doc_id,
            documents.source,
            documents.license,
            documents.raw_text_sha,
            documents.meta_json
        )
    """
    insert_provenance = """
        INSERT INTO exact_dedup_provenance_accumulator (
            digest, provenance_sha256, doc_id, source, license, meta_json,
            raw_text_sha, identity, occurrences
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(digest, provenance_sha256)
        DO UPDATE SET occurrences = occurrences + 1
    """
    for raw in docs:
        raw_document = _coerce_document(raw)
        _update_raw_document_stream_hash(raw_stream_hash, raw_document)
        raw_documents += 1
        candidate, digest, rejection = _quality_candidate(
            raw_document,
            min_chars=min_chars,
            max_chars=max_chars,
            max_control_ratio=max_control_ratio,
            max_repetition_ratio=max_repetition_ratio,
        )
        if candidate is None or digest is None:
            rejection_counts[rejection] += 1
            continue
        quality_passed += 1
        raw_text_sha = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        meta_json = _canonical_meta(candidate.meta)
        identity = _document_identity(candidate)
        provenance_json = json.dumps(
            [
                candidate.doc_id,
                candidate.source,
                candidate.license,
                raw_text_sha,
                meta_json,
                identity,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        provenance_sha256 = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
        connection.execute(
            insert_provenance,
            (
                digest,
                provenance_sha256,
                candidate.doc_id,
                candidate.source,
                candidate.license,
                meta_json,
                raw_text_sha,
                identity,
            ),
        )
        connection.execute(
            insert,
            (
                digest,
                candidate.text,
                candidate.source,
                candidate.doc_id,
                candidate.license,
                meta_json,
                raw_text_sha,
                len(candidate.text),
                identity,
            ),
        )
        if raw_documents % 10_000 == 0:
            connection.commit()
    connection.commit()
    exact_documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    # Every normalized-content group retains exactly one canonical document. Copy only remaining
    # occurrences from the file-backed SQLite TEMP accumulator into the durable alias table, so
    # the staging artifact scales with aliases rather than duplicating every canonical provenance.
    connection.execute(
        """
        INSERT INTO exact_dedup_aliases (
            digest, provenance_sha256, doc_id, source, license, meta_json,
            raw_text_sha, identity, occurrences
        )
        SELECT
            accumulator.digest,
            accumulator.provenance_sha256,
            accumulator.doc_id,
            accumulator.source,
            accumulator.license,
            accumulator.meta_json,
            accumulator.raw_text_sha,
            accumulator.identity,
            accumulator.occurrences - CASE WHEN (
                accumulator.doc_id = documents.doc_id
                AND accumulator.source = documents.source
                AND accumulator.license = documents.license
                AND accumulator.meta_json = documents.meta_json
                AND accumulator.raw_text_sha = documents.raw_text_sha
            ) THEN 1 ELSE 0 END
        FROM exact_dedup_provenance_accumulator AS accumulator
        JOIN documents USING (digest)
        WHERE
            accumulator.occurrences - CASE WHEN (
                accumulator.doc_id = documents.doc_id
                AND accumulator.source = documents.source
                AND accumulator.license = documents.license
                AND accumulator.meta_json = documents.meta_json
                AND accumulator.raw_text_sha = documents.raw_text_sha
            ) THEN 1 ELSE 0 END > 0
        """
    )
    connection.execute("DROP TABLE exact_dedup_provenance_accumulator")
    connection.commit()
    alias_records = int(
        connection.execute("SELECT COUNT(*) FROM exact_dedup_aliases").fetchone()[0]
    )
    alias_occurrences = int(
        connection.execute(
            "SELECT COALESCE(SUM(occurrences), 0) FROM exact_dedup_aliases"
        ).fetchone()[0]
    )
    expected_alias_occurrences = quality_passed - exact_documents
    if alias_occurrences != expected_alias_occurrences:
        raise RuntimeError(
            "exact-dedup alias accounting mismatch: "
            f"expected {expected_alias_occurrences}, got {alias_occurrences}"
        )
    alias_fingerprint = _joined_values_sha256(
        _staging_values(
            connection,
            """
            SELECT digest || ':' || provenance_sha256 || ':' || occurrences
            FROM exact_dedup_aliases
            ORDER BY digest, provenance_sha256
            """,
        )
    )
    return {
        "input_documents": raw_documents,
        "quality_passed_documents": quality_passed,
        "retained_documents": exact_documents,
        "removed_documents": raw_documents - exact_documents,
        "quality_rejection_counts": dict(sorted(rejection_counts.items())),
        "exact_duplicates_removed": quality_passed - exact_documents,
        "exact_content_deduplication": True,
        "normalized_content_sha256": True,
        "exact_dedup_provenance": {
            "storage": "sqlite:exact_dedup_aliases",
            "alias_records": alias_records,
            "alias_occurrences": alias_occurrences,
            "alias_fingerprint_sha256": alias_fingerprint,
            "constant_python_memory": True,
            "storage_scales_with": "deduplicated_alias_provenance_not_retained_documents",
            "stored_fields": [
                "normalized_content_digest",
                "provenance_sha256",
                "doc_id",
                "source",
                "license",
                "meta_json",
                "raw_text_sha256",
                "identity",
                "occurrences",
            ],
        },
        "raw_document_stream_hash_format": "framed_utf8_v2",
        "raw_document_stream_sha256": raw_stream_hash.hexdigest(),
    }


def _stage_evaluation_decontamination(
    connection: sqlite3.Connection,
    denylist: Iterable[str],
    *,
    shingle_size: int,
    min_shingles: int,
    min_coverage: float,
    anchors_per_entry: int,
    max_denylist_shingles: int,
) -> dict[str, Any]:
    matcher = _build_evaluation_denylist_matcher(
        denylist,
        shingle_size=shingle_size,
        min_shingles=min_shingles,
        min_coverage=min_coverage,
        anchors_per_entry=anchors_per_entry,
        max_denylist_shingles=max_denylist_shingles,
    )
    matched_entries: set[str] = set()
    input_documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    last_digest = ""
    processed = 0
    while True:
        row = connection.execute(
            """
            SELECT digest, text
            FROM documents
            WHERE digest > ?
            ORDER BY digest
            LIMIT 1
            """,
            (last_digest,),
        ).fetchone()
        if row is None:
            break
        matched = matcher.match(str(row["text"]))
        if matched is not None:
            connection.execute(
                """
                UPDATE documents
                SET decontaminated = 0, contamination_match = ?
                WHERE digest = ?
                """,
                (matched, row["digest"]),
            )
            matched_entries.add(matched)
        last_digest = str(row["digest"])
        processed += 1
        if processed % 1000 == 0:
            connection.commit()
    connection.commit()
    retained_documents = int(
        connection.execute("SELECT COUNT(*) FROM documents WHERE decontaminated = 1").fetchone()[0]
    )
    removed_ids_hash = _staging_document_ids_sha256(
        connection,
        "decontaminated = 0",
    )
    return _decontamination_audit(
        matcher,
        input_documents=input_documents,
        retained_documents=retained_documents,
        removed_document_ids_sha256=removed_ids_hash,
        matched_entries=matched_entries,
    )


def _validate_near_dedup_bounds(
    *,
    max_shingles: int,
    max_hamming_distance: int,
    min_jaccard: float,
    min_features: int,
    lsh_bands: int,
    max_bucket_size: int,
    feature_cache_size: int,
) -> None:
    if not 0 <= max_hamming_distance < lsh_bands:
        raise ValueError("max_hamming_distance must be in [0, lsh_bands)")
    if 64 % lsh_bands:
        raise ValueError("lsh_bands must divide 64")
    if not 0.0 < min_jaccard <= 1.0:
        raise ValueError("min_jaccard must be in (0, 1]")
    if min_features < 1 or max_bucket_size < 1 or feature_cache_size < 1:
        raise ValueError("near-dedup bounds must be positive")
    if max_shingles < min_features:
        raise ValueError("max_shingles must be >= min_features")


def _stage_near_deduplication(
    connection: sqlite3.Connection,
    *,
    enabled: bool,
    shingle_size: int,
    max_shingles: int,
    max_hamming_distance: int,
    min_jaccard: float,
    min_features: int,
    lsh_bands: int,
    max_bucket_size: int,
    feature_cache_size: int,
) -> dict[str, Any]:
    input_documents = int(
        connection.execute("SELECT COUNT(*) FROM documents WHERE decontaminated = 1").fetchone()[0]
    )
    connection.execute(
        """
        INSERT INTO near_queue (queue_rank, digest)
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY char_count DESC, raw_text_sha, doc_id, source
            ) - 1,
            digest
        FROM documents
        WHERE decontaminated = 1
        """
        if enabled
        else """
        INSERT INTO near_queue (queue_rank, digest)
        SELECT ROW_NUMBER() OVER (ORDER BY digest) - 1, digest
        FROM documents
        WHERE decontaminated = 1
        """
    )
    if not enabled:
        connection.execute(
            """
            UPDATE documents
            SET
                near_keep = 1,
                near_rank = (
                    SELECT near_queue.queue_rank
                    FROM near_queue
                    WHERE near_queue.digest = documents.digest
                )
            WHERE decontaminated = 1
            """
        )
        connection.commit()
        return {
            "enabled": False,
            "exhaustive": False,
            "input_documents": input_documents,
            "retained_documents": input_documents,
            "removed_documents": 0,
            "limitations": "Near-duplicate screening was disabled.",
        }

    _validate_near_dedup_bounds(
        max_shingles=max_shingles,
        max_hamming_distance=max_hamming_distance,
        min_jaccard=min_jaccard,
        min_features=min_features,
        lsh_bands=lsh_bands,
        max_bucket_size=max_bucket_size,
        feature_cache_size=feature_cache_size,
    )
    band_bits = 64 // lsh_bands
    band_mask = (1 << band_bits) - 1
    feature_cache: OrderedDict[int, frozenset[int]] = OrderedDict()
    retained_documents = 0
    removed_documents = 0
    candidate_pairs = 0
    jaccard_checks = 0
    bucket_overflows = 0
    code_documents_bypassed = 0

    def cached_features(rank: int, digest: str) -> frozenset[int]:
        if rank in feature_cache:
            value = feature_cache.pop(rank)
            feature_cache[rank] = value
            return value
        row = connection.execute(
            "SELECT text FROM documents WHERE digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"near-dedup index references missing digest {digest}")
        value = _hashed_shingles(
            str(row["text"]),
            shingle_size,
            max_shingles=max_shingles,
            case_sensitive=True,
        )
        feature_cache[rank] = value
        if len(feature_cache) > feature_cache_size:
            feature_cache.popitem(last=False)
        return value

    for queue_rank in range(input_documents):
        row = connection.execute(
            """
            SELECT
                documents.digest,
                documents.text,
                documents.identity,
                documents.source,
                documents.meta_json
            FROM near_queue
            JOIN documents USING (digest)
            WHERE near_queue.queue_rank = ?
            """,
            (queue_rank,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"near-dedup queue is missing rank {queue_rank}")
        digest = str(row["digest"])
        identity = str(row["identity"])
        if _is_code_like(str(row["text"]), str(row["source"]), str(row["meta_json"])):
            connection.execute(
                """
                UPDATE documents
                SET near_keep = 1, near_rank = ?, signature_hex = ?
                WHERE digest = ?
                """,
                (retained_documents, "0000000000000000", digest),
            )
            retained_documents += 1
            code_documents_bypassed += 1
            if (queue_rank + 1) % 1000 == 0:
                connection.commit()
            continue
        features = _hashed_shingles(
            str(row["text"]),
            shingle_size,
            max_shingles=max_shingles,
            case_sensitive=True,
        )
        if len(features) < min_features:
            connection.execute(
                """
                UPDATE documents
                SET near_keep = 1, near_rank = ?, signature_hex = ?
                WHERE digest = ?
                """,
                (retained_documents, "0000000000000000", digest),
            )
            retained_documents += 1
            if (queue_rank + 1) % 1000 == 0:
                connection.commit()
            continue

        signature = _simhash64(features)
        signature_hex = f"{signature:016x}"
        band_keys = [
            (band, (signature >> (band * band_bits)) & band_mask) for band in range(lsh_bands)
        ]
        candidates: dict[int, tuple[str, int, str]] = {}
        for band, band_value in band_keys:
            for candidate in connection.execute(
                """
                SELECT near_rank, digest, signature_hex, identity
                FROM near_index
                WHERE band = ? AND band_value = ?
                """,
                (band, band_value),
            ):
                candidates[int(candidate["near_rank"])] = (
                    str(candidate["digest"]),
                    int(str(candidate["signature_hex"]), 16),
                    str(candidate["identity"]),
                )

        duplicate_identity: str | None = None
        for rank in sorted(candidates):
            candidate_pairs += 1
            other_digest, other_signature, other_identity = candidates[rank]
            if (signature ^ other_signature).bit_count() > max_hamming_distance:
                continue
            jaccard_checks += 1
            other_features = cached_features(rank, other_digest)
            union_size = len(features | other_features)
            similarity = len(features & other_features) / max(1, union_size)
            if similarity >= min_jaccard:
                duplicate_identity = other_identity
                break

        if duplicate_identity is not None:
            pair = f"{identity}:{duplicate_identity}"
            connection.execute(
                """
                UPDATE documents
                SET
                    near_keep = 0,
                    duplicate_identity = ?,
                    removal_pair_hash = ?
                WHERE digest = ?
                """,
                (
                    duplicate_identity,
                    hashlib.sha256(pair.encode("utf-8")).hexdigest(),
                    digest,
                ),
            )
            removed_documents += 1
        else:
            rank = retained_documents
            connection.execute(
                """
                UPDATE documents
                SET near_keep = 1, near_rank = ?, signature_hex = ?
                WHERE digest = ?
                """,
                (rank, signature_hex, digest),
            )
            feature_cache[rank] = features
            if len(feature_cache) > feature_cache_size:
                feature_cache.popitem(last=False)
            for band, band_value in band_keys:
                bucket = connection.execute(
                    """
                    SELECT entries
                    FROM near_buckets
                    WHERE band = ? AND band_value = ?
                    """,
                    (band, band_value),
                ).fetchone()
                entries = 0 if bucket is None else int(bucket["entries"])
                if entries >= max_bucket_size:
                    bucket_overflows += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO near_buckets (band, band_value, entries)
                    VALUES (?, ?, 1)
                    ON CONFLICT(band, band_value)
                    DO UPDATE SET entries = entries + 1
                    """,
                    (band, band_value),
                )
                connection.execute(
                    """
                    INSERT INTO near_index (
                        band, band_value, near_rank, digest, signature_hex, identity
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (band, band_value, rank, digest, signature_hex, identity),
                )
            retained_documents += 1
        if (queue_rank + 1) % 1000 == 0:
            connection.commit()
    connection.commit()

    retained_ids_hash = _staging_document_ids_sha256(
        connection,
        "decontaminated = 1 AND near_keep = 1",
    )
    removed_ids_hash = _staging_document_ids_sha256(
        connection,
        "decontaminated = 1 AND near_keep = 0",
    )
    removal_pairs_hash = _joined_values_sha256(
        _staging_values(
            connection,
            """
            SELECT removal_pair_hash
            FROM documents
            WHERE decontaminated = 1 AND near_keep = 0
            ORDER BY removal_pair_hash
            """,
        )
    )
    return {
        "enabled": True,
        "method": (
            "sqlite_case_sensitive_simhash64_lsh_then_sampled_shingle_jaccard_with_code_bypass"
        ),
        "exhaustive": False,
        "input_documents": input_documents,
        "retained_documents": retained_documents,
        "removed_documents": removed_documents,
        "retained_document_ids_sha256": retained_ids_hash,
        "removed_document_ids_sha256": removed_ids_hash,
        "removal_pairs_sha256": removal_pairs_hash,
        "shingle_size": shingle_size,
        "max_shingles_per_document": max_shingles,
        "max_hamming_distance": max_hamming_distance,
        "min_jaccard": min_jaccard,
        "min_features": min_features,
        "lsh_bands": lsh_bands,
        "max_bucket_size": max_bucket_size,
        "feature_cache_size": feature_cache_size,
        "candidate_pairs": candidate_pairs,
        "jaccard_checks": jaccard_checks,
        "bucket_overflows": bucket_overflows,
        "case_sensitive_shingles": True,
        "code_documents_bypassed": code_documents_bypassed,
        "code_policy": "exact_content_deduplication_only",
        "limitations": (
            "Bounded heuristic screening; LSH bucket caps and sampled shingles can miss "
            "near duplicates. Code-like documents bypass heuristic near-dedup after exact "
            "content deduplication to preserve potentially semantic case/identifier changes."
        ),
    }


def _stage_split_assignment(
    connection: sqlite3.Connection,
    *,
    val_fraction: float,
    seed: int,
    frozen_assignment: FrozenSplitAssignment | None = None,
) -> dict[str, Any]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if frozen_assignment is not None:
        if seed != frozen_assignment.seed:
            raise ValueError(
                "frozen split manifest seed mismatch: "
                f"expected {frozen_assignment.seed}, got {seed}"
            )
        if val_fraction != frozen_assignment.val_fraction:
            raise ValueError(
                "frozen split manifest val_fraction mismatch: "
                f"expected {frozen_assignment.val_fraction}, got {val_fraction}"
            )
        return _stage_frozen_split_assignment(connection, frozen_assignment)

    last_identity = ""
    while True:
        identities = connection.execute(
            """
            SELECT DISTINCT identity
            FROM documents
            WHERE decontaminated = 1 AND near_keep = 1 AND identity > ?
            ORDER BY identity
            LIMIT 5000
            """,
            (last_identity,),
        ).fetchall()
        if not identities:
            break
        connection.executemany(
            """
            INSERT INTO split_identities (identity, rank_key, split)
            VALUES (?, ?, 'train')
            """,
            [
                (
                    str(row["identity"]),
                    hashlib.sha256(f"{seed}:{row['identity']}".encode("ascii")).digest(),
                )
                for row in identities
            ],
        )
        last_identity = str(identities[-1]["identity"])
        connection.commit()

    unique_identities = int(
        connection.execute("SELECT COUNT(*) FROM split_identities").fetchone()[0]
    )
    n_val = 0
    if val_fraction > 0 and unique_identities > 1:
        n_val = max(
            1,
            min(
                unique_identities - 1,
                round(unique_identities * val_fraction),
            ),
        )
    connection.execute(
        """
        UPDATE split_identities
        SET split = 'val'
        WHERE identity IN (
            SELECT identity
            FROM split_identities
            ORDER BY rank_key, identity
            LIMIT ?
        )
        """,
        (n_val,),
    )
    connection.execute(
        """
        UPDATE documents
        SET split = (
            SELECT split_identities.split
            FROM split_identities
            WHERE split_identities.identity = documents.identity
        )
        WHERE decontaminated = 1 AND near_keep = 1
        """
    )
    connection.commit()
    return {
        "mode": "computed",
        "algorithm": "sha256_seeded_exact_validation_count_v1",
        "seed": seed,
        "val_fraction": val_fraction,
        "identities": unique_identities,
        "documents": int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE decontaminated = 1 AND near_keep = 1
                """
            ).fetchone()[0]
        ),
        "assignment_sha256": _staging_split_assignment_sha256(connection),
    }


def _canonical_jsonl_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_split_assignment_record(line: bytes, *, path: Path, line_no: int) -> dict[str, Any]:
    if not line.endswith(b"\n"):
        raise ValueError(f"{path}:{line_no}: split assignment line must end with a newline")
    try:
        decoded = line.decode("utf-8")
        record = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}:{line_no}: invalid split assignment JSON") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_no}: split assignment row must be an object")
    if line != _canonical_jsonl_bytes(record):
        raise ValueError(
            f"{path}:{line_no}: split assignment JSON must use compact sorted-key encoding"
        )
    return record


def _stage_frozen_split_assignment(
    connection: sqlite3.Connection,
    reference: FrozenSplitAssignment,
) -> dict[str, Any]:
    """Apply a verified base-corpus assignment and reject missing or changed documents."""

    path = reference.path
    if not path.is_file():
        raise ValueError(f"frozen split assignment artifact is missing: {path}")
    if path.is_symlink():
        raise ValueError(f"frozen split assignment artifact must not be a symbolic link: {path}")
    if path.stat().st_size != reference.bytes:
        raise ValueError(
            "frozen split assignment byte-size mismatch: "
            f"expected {reference.bytes}, got {path.stat().st_size}"
        )

    file_digest = hashlib.sha256()
    assignment_digest = hashlib.sha256()
    assignment_first = True
    records = 0
    identities = 0
    matched_documents = 0
    previous_key: tuple[str, str] | None = None
    previous_identity = ""
    previous_identity_split = ""
    expected_header = {
        "format": SPLIT_ASSIGNMENT_FORMAT,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
    }

    with path.open("rb") as handle:
        line_no = 1
        header_line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
        if len(header_line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
            raise ValueError(f"{path}:1: split assignment line is too large")
        if not header_line:
            raise ValueError(f"{path}: split assignment artifact is empty")
        file_digest.update(header_line)
        header = _read_split_assignment_record(header_line, path=path, line_no=line_no)
        if header != expected_header:
            raise ValueError(
                f"{path}: unsupported split assignment header; expected {expected_header}"
            )

        while True:
            line_no += 1
            line = handle.readline(MAX_SPLIT_ASSIGNMENT_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_SPLIT_ASSIGNMENT_LINE_BYTES:
                raise ValueError(f"{path}:{line_no}: split assignment line is too large")
            file_digest.update(line)
            row = _read_split_assignment_record(line, path=path, line_no=line_no)
            if set(row) != {"document_id", "document_sha256", "identity_sha256", "split"}:
                raise ValueError(f"{path}:{line_no}: split assignment row has unexpected fields")
            document_id = row["document_id"]
            document_sha256 = row["document_sha256"]
            identity = row["identity_sha256"]
            split = row["split"]
            if not isinstance(document_id, str) or not document_id:
                raise ValueError(f"{path}:{line_no}: document_id must be a non-empty string")
            if not _valid_sha256(document_sha256):
                raise ValueError(f"{path}:{line_no}: document_sha256 must be lowercase SHA-256")
            if not _valid_sha256(identity):
                raise ValueError(f"{path}:{line_no}: identity_sha256 must be lowercase SHA-256")
            expected_identity = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
            if identity != expected_identity:
                raise ValueError(f"{path}:{line_no}: identity_sha256 does not match document_id")
            if split not in {"train", "val"}:
                raise ValueError(f"{path}:{line_no}: split must be 'train' or 'val'")

            key = (identity, document_sha256)
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    f"{path}:{line_no}: split assignment rows must be unique and sorted"
                )
            if identity != previous_identity:
                identities += 1
                previous_identity = identity
                previous_identity_split = str(split)
            elif split != previous_identity_split:
                raise ValueError(
                    f"{path}:{line_no}: one document_id is assigned to multiple splits"
                )
            previous_key = key

            value = _split_assignment_value(identity, document_sha256, str(split))
            if not assignment_first:
                assignment_digest.update(b"\n")
            assignment_digest.update(value.encode("ascii"))
            assignment_first = False
            records += 1
            cursor = connection.execute(
                """
                UPDATE documents
                SET split = ?
                WHERE
                    decontaminated = 1
                    AND near_keep = 1
                    AND identity = ?
                    AND raw_text_sha = ?
                """,
                (split, identity, document_sha256),
            )
            matched_documents += cursor.rowcount
            if records % 5000 == 0:
                connection.commit()

    actual_sha256 = file_digest.hexdigest()
    if actual_sha256 != reference.sha256:
        raise ValueError(
            "frozen split assignment SHA-256 mismatch: "
            f"expected {reference.sha256}, got {actual_sha256}"
        )
    if records != reference.records:
        raise ValueError(
            "frozen split assignment record-count mismatch: "
            f"expected {reference.records}, got {records}"
        )
    actual_assignment_sha256 = assignment_digest.hexdigest()
    if actual_assignment_sha256 != reference.assignment_sha256:
        raise ValueError(
            "frozen split assignment content fingerprint mismatch: "
            f"expected {reference.assignment_sha256}, got {actual_assignment_sha256}"
        )

    missing_documents = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE decontaminated = 1 AND near_keep = 1 AND split IS NULL
            """
        ).fetchone()[0]
    )
    if missing_documents:
        examples = [
            f"{row['doc_id']}@{str(row['raw_text_sha'])[:12]}"
            for row in connection.execute(
                """
                SELECT doc_id, raw_text_sha
                FROM documents
                WHERE decontaminated = 1 AND near_keep = 1 AND split IS NULL
                ORDER BY identity, raw_text_sha
                LIMIT 5
                """
            )
        ]
        raise ValueError(
            f"frozen split assignment is missing {missing_documents} retained document/content "
            f"binding(s): {', '.join(examples)}"
        )

    conflicting_identity = connection.execute(
        """
        SELECT identity
        FROM documents
        WHERE decontaminated = 1 AND near_keep = 1
        GROUP BY identity
        HAVING COUNT(DISTINCT split) > 1
        LIMIT 1
        """
    ).fetchone()
    if conflicting_identity is not None:
        raise ValueError("frozen split assignment maps one retained document_id to multiple splits")
    connection.execute(
        """
        INSERT INTO split_identities (identity, rank_key, split)
        SELECT identity, X'', MIN(split)
        FROM documents
        WHERE decontaminated = 1 AND near_keep = 1
        GROUP BY identity
        """
    )
    connection.commit()
    target_assignment_sha256 = _staging_split_assignment_sha256(connection)
    return {
        "mode": "frozen",
        "binding": "sha256(document_id)+sha256(filtered_utf8_text)",
        "seed": reference.seed,
        "val_fraction": reference.val_fraction,
        "source_records": records,
        "source_identities": identities,
        "matched_documents": matched_documents,
        "ignored_source_records": records - matched_documents,
        "missing_documents": 0,
        "source_assignment_sha256": actual_assignment_sha256,
        "assignment_sha256": target_assignment_sha256,
        "source_artifact": {
            "path": str(path),
            "bytes": reference.bytes,
            "sha256": reference.sha256,
        },
        "source_manifest": dict(reference.source_manifest),
    }


def build_disk_backed_corpus(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    database_path: str | Path,
    *,
    min_chars: int = 200,
    max_chars: int = 2_000_000,
    max_control_ratio: float = 0.01,
    max_repetition_ratio: float = 0.35,
    denylist: Iterable[str] = (),
    decontam_shingle_size: int = 5,
    decontam_min_shingles: int = 8,
    decontam_coverage: float = 0.9,
    decontam_anchors_per_entry: int = 8,
    decontam_max_denylist_shingles: int = 2048,
    near_dedup: bool = True,
    near_dedup_shingle_size: int = 5,
    near_dedup_max_shingles: int = 256,
    near_dedup_hamming: int = 3,
    near_dedup_jaccard: float = 0.95,
    near_dedup_min_features: int = 8,
    near_dedup_lsh_bands: int = 8,
    near_dedup_max_bucket_size: int = 64,
    near_dedup_feature_cache_size: int = 128,
    val_fraction: float = 0.01,
    seed: int = 42,
    frozen_split_assignment: FrozenSplitAssignment | None = None,
) -> DiskBackedCorpus:
    """Stage a corpus on disk without materializing all document text in Python RAM.

    The operation is deterministic and replaces ``database_path`` atomically only after exact
    filtering, denylist screening, near-deduplication, and split assignment all succeed.
    """

    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_suffix(database.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = _staging_connection(temporary)
    try:
        _create_staging_schema(connection)
        quality_audit = _stage_exact_documents(
            connection,
            docs,
            min_chars=min_chars,
            max_chars=max_chars,
            max_control_ratio=max_control_ratio,
            max_repetition_ratio=max_repetition_ratio,
        )
        if quality_audit["retained_documents"] == 0:
            raise ValueError("all documents were rejected by the quality filters")
        decontamination_audit = _stage_evaluation_decontamination(
            connection,
            denylist,
            shingle_size=decontam_shingle_size,
            min_shingles=decontam_min_shingles,
            min_coverage=decontam_coverage,
            anchors_per_entry=decontam_anchors_per_entry,
            max_denylist_shingles=decontam_max_denylist_shingles,
        )
        near_dedup_audit = _stage_near_deduplication(
            connection,
            enabled=near_dedup,
            shingle_size=near_dedup_shingle_size,
            max_shingles=near_dedup_max_shingles,
            max_hamming_distance=near_dedup_hamming,
            min_jaccard=near_dedup_jaccard,
            min_features=near_dedup_min_features,
            lsh_bands=near_dedup_lsh_bands,
            max_bucket_size=near_dedup_max_bucket_size,
            feature_cache_size=near_dedup_feature_cache_size,
        )
        if near_dedup_audit["retained_documents"] == 0:
            raise ValueError("all documents were rejected by corpus hygiene filters")
        split_assignment_audit = _stage_split_assignment(
            connection,
            val_fraction=val_fraction,
            seed=seed,
            frozen_assignment=frozen_split_assignment,
        )
        corpus_audit = {
            "quality_and_exact_deduplication": quality_audit,
            "evaluation_decontamination": decontamination_audit,
            "near_deduplication": near_dedup_audit,
            "split_assignment": split_assignment_audit,
        }
        staging_config = {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "max_control_ratio": max_control_ratio,
            "max_repetition_ratio": max_repetition_ratio,
            "decontam_shingle_size": decontam_shingle_size,
            "decontam_min_shingles": decontam_min_shingles,
            "decontam_coverage": decontam_coverage,
            "decontam_anchors_per_entry": decontam_anchors_per_entry,
            "decontam_max_denylist_shingles": decontam_max_denylist_shingles,
            "near_dedup": near_dedup,
            "near_dedup_shingle_size": near_dedup_shingle_size,
            "near_dedup_max_shingles": near_dedup_max_shingles,
            "near_dedup_hamming": near_dedup_hamming,
            "near_dedup_jaccard": near_dedup_jaccard,
            "near_dedup_min_features": near_dedup_min_features,
            "near_dedup_lsh_bands": near_dedup_lsh_bands,
            "near_dedup_max_bucket_size": near_dedup_max_bucket_size,
            "near_dedup_feature_cache_size": near_dedup_feature_cache_size,
            "val_fraction": val_fraction,
            "seed": seed,
            "split_assignment_mode": split_assignment_audit["mode"],
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("staging_version", json.dumps(STAGING_VERSION)),
                (
                    "corpus_audit",
                    json.dumps(
                        corpus_audit,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
                (
                    "staging_config",
                    json.dumps(staging_config, sort_keys=True, separators=(",", ":")),
                ),
            ],
        )
        connection.commit()
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    connection.close()
    temporary.replace(database)
    return DiskBackedCorpus(database)


def split_documents(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> dict[str, list[CorpusDocument]]:
    """Assign whole documents to deterministic, disjoint train and validation splits.

    Assignment depends on the seed and each document's stable id, not on input order. The returned
    lists retain input order so packing remains reproducible for a fixed corpus. Callers that train
    a tokenizer should fit it only on the returned ``train`` list and pass this same mapping to
    :func:`pack_shards`.
    """

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    prepared = [_coerce_document(document) for document in docs]
    identities = [_document_identity(document) for document in prepared]
    ranked_identities = sorted(
        set(identities),
        key=lambda identity: hashlib.sha256(f"{seed}:{identity}".encode("ascii")).digest(),
    )
    n_val = 0
    if val_fraction > 0 and len(ranked_identities) > 1:
        n_val = max(
            1,
            min(
                len(ranked_identities) - 1,
                round(len(ranked_identities) * val_fraction),
            ),
        )
    val_identities = set(ranked_identities[:n_val])
    return {
        "train": [
            document
            for identity, document in zip(identities, prepared, strict=True)
            if identity not in val_identities
        ],
        "val": [
            document
            for identity, document in zip(identities, prepared, strict=True)
            if identity in val_identities
        ],
    }


def _validate_precomputed_splits(
    documents: Sequence[CorpusDocument],
    splits: Mapping[str, Sequence[CorpusDocument]],
) -> None:
    if set(splits) != {"train", "val"}:
        raise ValueError("precomputed_splits must contain exactly 'train' and 'val'")
    expected = sorted(_document_identity(document) for document in documents)
    assigned = sorted(
        _document_identity(document)
        for split_documents_ in splits.values()
        for document in split_documents_
    )
    if assigned != expected:
        raise ValueError("precomputed_splits must assign every input document exactly once")
    train_identities = {_document_identity(document) for document in splits["train"]}
    val_identities = {_document_identity(document) for document in splits["val"]}
    if train_identities & val_identities:
        raise ValueError("precomputed_splits must be document-disjoint")


def _split_assignment_sha256(splits: Mapping[str, Sequence[CorpusDocument]]) -> str:
    assignments = sorted(
        {
            _split_assignment_value(
                _document_identity(document),
                _sha256_utf8(document.text),
                split,
            )
            for split, documents in splits.items()
            for document in documents
        }
    )
    return hashlib.sha256("\n".join(assignments).encode("ascii")).hexdigest()


def _packed_rows(
    documents: Iterable[CorpusDocument],
    tokenizer,
    row_tokens: int,
    *,
    source_token_counts: dict[str, int] | None = None,
    source_group: Callable[[CorpusDocument], str] | None = None,
) -> Iterator[list[int]]:
    """Best-fit-in-order packing where every row and document boundary has an EOS marker."""

    if (source_token_counts is None) != (source_group is None):
        raise ValueError("source token accounting needs both a counter and grouping function")
    eos = int(tokenizer.eos_id)
    current = [eos]
    for doc in documents:
        group = source_group(doc) if source_group is not None else None
        tokens = tokenizer.encode(doc.text, add_eos=False)
        cursor = 0
        while cursor < len(tokens):
            available = row_tokens - len(current) - 1  # reserve closing EOS
            if available <= 0:
                yield [*current, eos] if current[-1] != eos else current
                current = [eos]
                available = row_tokens - 2
            take = min(available, len(tokens) - cursor)
            current.extend(tokens[cursor : cursor + take])
            cursor += take
            if source_token_counts is not None and group is not None:
                # Every encoded token is a next-token target. The closing EOS below is also a
                # packed target and is attributed to the document that caused it.
                source_token_counts[group] += take + 1
            if cursor < len(tokens):
                current.append(eos)
                yield current
                current = [eos]
            else:
                current.append(eos)  # also separates the next document
        if not tokens and current[-1] != eos:
            current.append(eos)
    if len(current) > 1:
        yield current


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split_assignment_artifact(
    directory: Path,
    relative_directory: Path,
    rows: Iterable[tuple[str, str, str, str]],
) -> dict[str, Any]:
    """Write canonical, content-bound split rows into the immutable shard generation."""

    name = "split-assignment.jsonl"
    path = directory / name
    assignment_digest = hashlib.sha256()
    assignment_first = True
    records = 0
    identities = 0
    previous_key: tuple[str, str] | None = None
    previous_identity = ""
    previous_identity_split = ""
    with path.open("wb") as handle:
        handle.write(
            _canonical_jsonl_bytes(
                {
                    "format": SPLIT_ASSIGNMENT_FORMAT,
                    "schema_version": SPLIT_ASSIGNMENT_VERSION,
                }
            )
        )
        for document_id, document_sha256, identity, split in rows:
            key = (identity, document_sha256)
            if previous_key is not None and key <= previous_key:
                raise ValueError("split assignment rows must be unique and sorted")
            if identity != hashlib.sha256(document_id.encode("utf-8")).hexdigest():
                raise ValueError("split assignment identity does not match document_id")
            if not _valid_sha256(document_sha256) or not _valid_sha256(identity):
                raise ValueError("split assignment contains an invalid SHA-256")
            if split not in {"train", "val"}:
                raise ValueError(f"unknown split {split!r}")
            if identity != previous_identity:
                identities += 1
                previous_identity = identity
                previous_identity_split = split
            elif split != previous_identity_split:
                raise ValueError("one document_id cannot be assigned to multiple splits")
            previous_key = key
            handle.write(
                _canonical_jsonl_bytes(
                    {
                        "document_id": document_id,
                        "document_sha256": document_sha256,
                        "identity_sha256": identity,
                        "split": split,
                    }
                )
            )
            value = _split_assignment_value(identity, document_sha256, split)
            if not assignment_first:
                assignment_digest.update(b"\n")
            assignment_digest.update(value.encode("ascii"))
            assignment_first = False
            records += 1
        handle.flush()
        os.fsync(handle.fileno())
    if not records:
        raise ValueError("cannot write an empty split assignment")
    return {
        "format": SPLIT_ASSIGNMENT_FORMAT,
        "schema_version": SPLIT_ASSIGNMENT_VERSION,
        "path": str(relative_directory / name),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "records": records,
        "identities": identities,
        "assignment_sha256": assignment_digest.hexdigest(),
        "binding": "sha256(document_id)+sha256(filtered_utf8_text)",
    }


def _split_assignment_rows_from_splits(
    splits: Mapping[str, Sequence[CorpusDocument]],
) -> Iterator[tuple[str, str, str, str]]:
    assignments: dict[tuple[str, str], tuple[str, str]] = {}
    identity_splits: dict[str, str] = {}
    for split, documents in splits.items():
        if split not in {"train", "val"}:
            raise ValueError(f"unknown split {split!r}")
        for document in documents:
            identity = _document_identity(document)
            document_sha256 = _sha256_utf8(document.text)
            previous_split = identity_splits.setdefault(identity, split)
            if previous_split != split:
                raise ValueError("one document_id cannot be assigned to multiple splits")
            assignments[(identity, document_sha256)] = (document.doc_id, split)
    for (identity, document_sha256), (document_id, split) in sorted(assignments.items()):
        yield document_id, document_sha256, identity, split


def load_frozen_split_assignment_manifest(
    manifest_path: str | Path,
) -> FrozenSplitAssignment:
    """Load and verify the immutable split artifact referenced by a base corpus manifest."""

    source = Path(manifest_path)
    if not source.is_file():
        raise ValueError(f"frozen split manifest is missing: {source}")
    if source.is_symlink():
        raise ValueError(f"frozen split manifest must not be a symbolic link: {source}")
    with source.open("rb") as handle:
        payload = handle.read(DEFAULT_MAX_RAW_DOCUMENT_BYTES + 1)
    if len(payload) > DEFAULT_MAX_RAW_DOCUMENT_BYTES:
        raise ValueError(
            f"{source}: packed manifest exceeds {DEFAULT_MAX_RAW_DOCUMENT_BYTES} bytes"
        )
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source}: invalid packed manifest JSON") from error
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{source}: unsupported packed corpus manifest")
    generation = manifest.get("generation")
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise ValueError(f"{source}: packed manifest has an invalid generation")
    artifact = manifest.get("split_assignment")
    if not isinstance(artifact, dict):
        raise ValueError(f"{source}: packed manifest has no frozen split_assignment artifact")
    if artifact.get("format") != SPLIT_ASSIGNMENT_FORMAT:
        raise ValueError(f"{source}: unsupported split assignment format")
    if artifact.get("schema_version") != SPLIT_ASSIGNMENT_VERSION:
        raise ValueError(f"{source}: unsupported split assignment schema version")

    relative_value = artifact.get("path")
    expected_bytes = artifact.get("bytes")
    expected_sha256 = artifact.get("sha256")
    records = artifact.get("records")
    assignment_sha256 = artifact.get("assignment_sha256")
    if not isinstance(relative_value, str):
        raise ValueError(f"{source}: split assignment artifact path is missing")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
    ):
        raise ValueError(f"{source}: split assignment artifact bytes are invalid")
    if isinstance(records, bool) or not isinstance(records, int) or records < 1:
        raise ValueError(f"{source}: split assignment record count is invalid")
    if not _valid_sha256(expected_sha256) or not _valid_sha256(assignment_sha256):
        raise ValueError(f"{source}: split assignment artifact hashes are invalid")
    if manifest.get("split_assignment_sha256") != assignment_sha256:
        raise ValueError(f"{source}: split assignment fingerprints do not match")

    relative = Path(relative_value)
    expected_relative = Path("generations") / generation / "split-assignment.jsonl"
    if relative.is_absolute() or relative != expected_relative:
        raise ValueError(
            f"{source}: split assignment is not part of manifest generation {generation}"
        )
    root = source.parent.resolve()
    unresolved_path = source.parent / relative
    generation_directory = source.parent / "generations" / generation
    if unresolved_path.is_symlink() or generation_directory.is_symlink():
        raise ValueError(f"{source}: split assignment path must not be a symbolic link")
    artifact_path = unresolved_path.resolve()
    if (
        not artifact_path.is_relative_to(root)
        or artifact_path.parent != generation_directory.resolve()
    ):
        raise ValueError(f"{source}: split assignment path escapes the corpus directory")
    if not artifact_path.is_file():
        raise ValueError(f"{source}: split assignment artifact is missing")
    actual_bytes = artifact_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{source}: split assignment byte-size mismatch: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    actual_sha256 = _sha256_file(artifact_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{source}: split assignment SHA-256 mismatch")

    seed = manifest.get("seed")
    val_fraction = manifest.get("val_fraction")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{source}: split seed is invalid")
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise ValueError(f"{source}: split val_fraction is invalid")
    val_fraction = float(val_fraction)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"{source}: split val_fraction is out of range")
    return FrozenSplitAssignment(
        path=artifact_path,
        bytes=expected_bytes,
        sha256=expected_sha256,
        records=records,
        assignment_sha256=assignment_sha256,
        seed=seed,
        val_fraction=val_fraction,
        source_manifest={
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "generation": generation,
        },
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; some platforms do not permit directory fsync."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _bounded_label(value: str, *, max_chars: int = 160) -> str:
    """Keep manifest group keys bounded while retaining a stable full-value fingerprint."""

    normalized = value.strip() or "unknown"
    if len(normalized) <= max_chars:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"{normalized[: max_chars - 22]}…#{digest}"


def _source_family(source: str, meta: Mapping[str, Any] | str | None = None) -> str:
    """Collapse row-level provenance into a dataset/source family suitable for a manifest."""

    metadata: Mapping[str, Any] = {}
    if isinstance(meta, str):
        try:
            decoded = json.loads(meta)
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            metadata = decoded
    elif isinstance(meta, Mapping):
        metadata = meta

    mixture_source = metadata.get("mixture_source")
    if isinstance(mixture_source, str) and mixture_source.strip():
        return _bounded_label(f"mixture:{mixture_source}")
    dataset = metadata.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return _bounded_label(f"dataset:{dataset}")

    parsed = urlsplit(source)
    if parsed.scheme == "hf" and parsed.netloc == "datasets":
        parts = [part for part in parsed.path.split("/") if part]
        dataset_path = "/".join(parts[:2]) if parts else "unknown"
        return _bounded_label(f"dataset:{dataset_path}")
    if parsed.scheme in {"http", "https"}:
        return _bounded_label(f"web:{parsed.hostname or 'unknown'}")
    if parsed.scheme:
        return _bounded_label(f"scheme:{parsed.scheme}")
    if "/" in source or "\\" in source:
        suffix = Path(source).suffix.lower() or "no-extension"
        return _bounded_label(f"file:{suffix}")
    return _bounded_label(f"named:{source}")


def _license_family(license_name: str) -> str:
    return _bounded_label(license_name.casefold())


def _bounded_group_counts(
    values: Callable[[], Iterable[str]],
    *,
    max_groups: int = MAX_MANIFEST_GROUPS,
) -> tuple[dict[str, int], dict[str, Any], frozenset[str]]:
    """Count a deterministic bounded set of group labels and fold the rest into overflow."""

    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    selected: list[str] = []
    selected_set: set[str] = set()
    for value in values():
        if value in selected_set:
            continue
        if len(selected) < max_groups:
            insort(selected, value)
            selected_set.add(value)
        elif value < selected[-1]:
            selected_set.remove(selected.pop())
            insort(selected, value)
            selected_set.add(value)

    counts = dict.fromkeys(selected, 0)
    overflow_documents = 0
    documents = 0
    for value in values():
        documents += 1
        if value in selected_set:
            counts[value] += 1
        else:
            overflow_documents += 1

    overflow_key: str | None = None
    if overflow_documents:
        overflow_key = "__other__"
        while overflow_key in counts:
            overflow_key += "_"
        counts[overflow_key] = overflow_documents
    audit = {
        "max_named_groups": max_groups,
        "named_groups": len(selected),
        "overflow_documents": overflow_documents,
        "overflow_key": overflow_key,
        "selection": "lexicographically_first_distinct_groups",
        "documents": documents,
    }
    return counts, audit, frozenset(selected)


def _multiset_provenance_fingerprint(values: Iterable[str]) -> dict[str, Any]:
    """Build an order-independent, constant-memory fingerprint over provenance records."""

    modulus = 1 << 256
    digest_sum = 0
    digest_xor = 0
    records = 0
    for value in values:
        item = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest(), "big")
        digest_sum = (digest_sum + item) % modulus
        digest_xor ^= item
        records += 1
    payload = f"localagent-provenance-multiset-v1\n{records}\n{digest_sum:064x}\n{digest_xor:064x}"
    return {
        "algorithm": "sha256_over_count_sum256_xor256_v1",
        "records": records,
        "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }


def _provenance_value(
    source: str,
    license_name: str,
    meta: Mapping[str, Any] | str,
) -> str:
    if isinstance(meta, str):
        meta_json = meta
    else:
        meta_json = _canonical_meta(meta)
    return json.dumps(
        [source, license_name, meta_json],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _source_group_function(
    selected: frozenset[str],
    overflow_key: str | None,
) -> Callable[[CorpusDocument], str]:
    def group(document: CorpusDocument) -> str:
        family = _source_family(document.source, document.meta)
        if family in selected:
            return family
        if overflow_key is None:
            raise RuntimeError(f"unaccounted source family {family!r}")
        return overflow_key

    return group


def _new_shard_generation(out: Path) -> tuple[str, Path, Path, Path]:
    """Create a private generation directory whose final name is never reused."""

    generations = out / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    temporary = generations / f".{generation}.tmp"
    final = generations / generation
    temporary.mkdir()
    relative = Path("generations") / generation
    return generation, temporary, final, relative


def _write_packed_shard(
    directory: Path,
    relative_directory: Path,
    *,
    split: str,
    shard_index: int,
    rows: Sequence[Sequence[int]],
    row_tokens: int,
    pad_id: int,
    dtype: type[np.unsignedinteger[Any]],
) -> dict[str, Any]:
    array = np.full((len(rows), row_tokens), pad_id, dtype=dtype)
    lengths = np.empty(len(rows), dtype=np.uint32)
    for index, row in enumerate(rows):
        array[index, : len(row)] = row
        lengths[index] = len(row)
    token_name = f"{split}-{shard_index:05d}.npy"
    length_name = f"{split}-{shard_index:05d}.lengths.npy"
    token_path = directory / token_name
    length_path = directory / length_name
    np.save(token_path, array, allow_pickle=False)
    np.save(length_path, lengths, allow_pickle=False)
    _fsync_file(token_path)
    _fsync_file(length_path)
    return {
        "tokens": str(relative_directory / token_name),
        "lengths": str(relative_directory / length_name),
        "rows": len(rows),
        "bytes": token_path.stat().st_size,
        "lengths_bytes": length_path.stat().st_size,
        "sha256": _sha256_file(token_path),
        "lengths_sha256": _sha256_file(length_path),
    }


def _publish_shard_generation(
    out: Path,
    generation: str,
    temporary_directory: Path,
    final_directory: Path,
    manifest: dict[str, Any],
) -> None:
    """Commit completed immutable shards first, then atomically publish their manifest."""

    temporary_directory.replace(final_directory)
    _fsync_directory(final_directory.parent)
    manifest["generation"] = generation
    manifest_path = out / "manifest.json"
    temporary_manifest = out / f".manifest.{generation}.tmp"
    try:
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_manifest.replace(manifest_path)
        _fsync_directory(out)
    finally:
        temporary_manifest.unlink(missing_ok=True)


def _discard_temporary_generation(temporary_directory: Path) -> None:
    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)


def pack_shards(
    docs: Iterable[CorpusDocument | str | dict[str, Any]],
    tokenizer,
    seq_len: int,
    shards_dir: str,
    *,
    rows_per_shard: int = 2048,
    val_fraction: float = 0.01,
    seed: int = 42,
    precomputed_splits: Mapping[str, Sequence[CorpusDocument | str | dict[str, Any]]] | None = None,
    tokenizer_training: Mapping[str, Any] | None = None,
    corpus_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokenize documents into memory-mappable train/validation shards.

    Returns the same manifest written to disk. Existing ``*.npy`` shards in the target are not
    silently mixed in: only filenames listed in the new manifest are consumed by the loader.
    ``precomputed_splits`` lets tokenizer training and packing share one exact assignment.
    """

    if seq_len < 8:
        raise ValueError("seq_len must be >= 8")
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be >= 1")
    prepared = [_coerce_document(doc) for doc in docs]
    if not prepared:
        raise ValueError("cannot pack an empty corpus")
    if precomputed_splits is None:
        split_docs = split_documents(prepared, val_fraction, seed)
    else:
        split_docs = {
            split: [_coerce_document(document) for document in documents]
            for split, documents in precomputed_splits.items()
        }
        _validate_precomputed_splits(prepared, split_docs)
    vocab_size = int(tokenizer.vocab_size)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    row_tokens = seq_len + 1
    source_counts, source_aggregation, selected_sources = _bounded_group_counts(
        lambda: (_source_family(doc.source, doc.meta) for doc in prepared)
    )
    license_counts, license_aggregation, _ = _bounded_group_counts(
        lambda: (_license_family(doc.license) for doc in prepared)
    )
    provenance_fingerprint = _multiset_provenance_fingerprint(
        _provenance_value(doc.source, doc.license, doc.meta) for doc in prepared
    )
    source_group = _source_group_function(
        selected_sources,
        source_aggregation["overflow_key"],
    )
    out = Path(shards_dir)
    out.mkdir(parents=True, exist_ok=True)
    generation, generation_tmp, generation_final, generation_relative = _new_shard_generation(out)
    try:
        split_assignment = _write_split_assignment_artifact(
            generation_tmp,
            generation_relative,
            _split_assignment_rows_from_splits(split_docs),
        )
    except BaseException:
        _discard_temporary_generation(generation_tmp)
        raise
    expected_assignment_sha256 = _split_assignment_sha256(split_docs)
    if split_assignment["assignment_sha256"] != expected_assignment_sha256:
        _discard_temporary_generation(generation_tmp)
        raise RuntimeError("split assignment artifact fingerprint mismatch")
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "format": "bos_aligned_rows",
        "seq_len": seq_len,
        "row_tokens": row_tokens,
        "token_dtype": np.dtype(dtype).name,
        "vocab_size": vocab_size,
        "seed": seed,
        "val_fraction": val_fraction,
        "split_assignment_sha256": expected_assignment_sha256,
        "split_assignment": split_assignment,
        "source_counts": source_counts,
        "source_token_counts": dict.fromkeys(source_counts, 0),
        "license_counts": license_counts,
        "provenance_summary": {
            "source_count_aggregation": source_aggregation,
            "license_count_aggregation": license_aggregation,
            "retained_document_provenance": provenance_fingerprint,
            "source_token_accounting": (
                "Encoded document tokens plus packing EOS targets, attributed to the source "
                "family of the document that caused each target."
            ),
        },
        "splits": {},
    }
    if corpus_audit is not None:
        manifest["corpus_audit"] = dict(corpus_audit)
    try:
        for split, documents in split_docs.items():
            rows: list[list[int]] = []
            shard_index = 0
            split_token_counts = dict.fromkeys(source_counts, 0)
            split_meta = {
                "documents": len(documents),
                "document_ids_sha256": _document_ids_sha256(documents),
                "document_set_sha256": _document_set_sha256(documents),
                "rows": 0,
                "tokens": 0,
                "source_token_counts": split_token_counts,
                "shards": [],
            }

            def flush(split_name=split, split_stats=split_meta) -> None:
                nonlocal rows, shard_index
                if not rows:
                    return
                entry = _write_packed_shard(
                    generation_tmp,
                    generation_relative,
                    split=split_name,
                    shard_index=shard_index,
                    rows=rows,
                    row_tokens=row_tokens,
                    pad_id=tokenizer.pad_id,
                    dtype=dtype,
                )
                split_stats["shards"].append(entry)
                split_stats["rows"] += len(rows)
                split_stats["tokens"] += int(sum(len(row) - 1 for row in rows))
                shard_index += 1
                rows = []

            for row in _packed_rows(
                documents,
                tokenizer,
                row_tokens,
                source_token_counts=split_token_counts,
                source_group=source_group,
            ):
                rows.append(row)
                if len(rows) >= rows_per_shard:
                    flush()
            flush()
            if sum(split_token_counts.values()) != split_meta["tokens"]:
                raise RuntimeError(f"{split} source-token accounting does not match packed tokens")
            manifest["splits"][split] = split_meta
            for family, count in split_token_counts.items():
                manifest["source_token_counts"][family] += count

        if tokenizer_training is not None:
            training_audit = dict(tokenizer_training)
            training_split = training_audit.get("split")
            if training_split is None:
                training_audit["documents"] = 0
            else:
                if not isinstance(training_split, str) or training_split not in split_docs:
                    raise ValueError("tokenizer_training split must name a packed split")
                training_documents = split_docs[training_split]
                training_audit.update(
                    {
                        "documents": len(training_documents),
                        "document_ids_sha256": _document_ids_sha256(training_documents),
                        "document_set_sha256": _document_set_sha256(training_documents),
                        "excluded_documents": len(prepared) - len(training_documents),
                    }
                )
            manifest["tokenizer_training"] = training_audit

        manifest["total_documents"] = len(prepared)
        manifest["total_tokens"] = sum(split["tokens"] for split in manifest["splits"].values())
        manifest["train_tokens"] = manifest["splits"]["train"]["tokens"]
        if sum(manifest["source_token_counts"].values()) != manifest["total_tokens"]:
            raise RuntimeError("source-token accounting does not match total packed tokens")
        _publish_shard_generation(
            out,
            generation,
            generation_tmp,
            generation_final,
            manifest,
        )
    except BaseException:
        _discard_temporary_generation(generation_tmp)
        raise
    return manifest


def pack_disk_backed_shards(
    corpus: DiskBackedCorpus,
    tokenizer,
    seq_len: int,
    shards_dir: str,
    *,
    rows_per_shard: int = 2048,
    tokenizer_training: Mapping[str, Any] | None = None,
    preparation_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack a staged corpus while keeping document text on disk.

    Split membership is read from the staging database created before tokenizer training. This
    prevents a caller from accidentally recomputing a different validation split for packing.
    """

    if seq_len < 8:
        raise ValueError("seq_len must be >= 8")
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be >= 1")
    staging_config = corpus.staging_config
    seed = int(staging_config["seed"])
    val_fraction = float(staging_config["val_fraction"])
    vocab_size = int(tokenizer.vocab_size)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    row_tokens = seq_len + 1

    with closing(_staging_connection(corpus.path, read_only=True)) as connection:
        retained_where = "decontaminated = 1 AND near_keep = 1"
        total_documents = int(
            connection.execute(f"SELECT COUNT(*) FROM documents WHERE {retained_where}").fetchone()[
                0
            ]
        )
        if total_documents == 0:
            raise ValueError("cannot pack an empty corpus")
        split_assignment_sha256 = _staging_split_assignment_sha256(connection)
        source_counts, source_aggregation, selected_sources = _bounded_group_counts(
            lambda: (
                _source_family(str(row["source"]), str(row["meta_json"]))
                for row in connection.execute(
                    f"""
                    SELECT source, meta_json
                    FROM documents
                    WHERE {retained_where}
                    """
                )
            )
        )
        license_counts, license_aggregation, _ = _bounded_group_counts(
            lambda: (
                _license_family(str(row["license"]))
                for row in connection.execute(
                    f"""
                    SELECT license
                    FROM documents
                    WHERE {retained_where}
                    """
                )
            )
        )
        provenance_fingerprint = _multiset_provenance_fingerprint(
            _provenance_value(
                str(row["source"]),
                str(row["license"]),
                str(row["meta_json"]),
            )
            for row in connection.execute(
                f"""
                SELECT source, license, meta_json
                FROM documents
                WHERE {retained_where}
                """
            )
        )
        split_metadata = {}
        for split in ("train", "val"):
            where = "decontaminated = 1 AND near_keep = 1 AND split = ?"
            split_metadata[split] = {
                "documents": int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM documents WHERE {where}",
                        (split,),
                    ).fetchone()[0]
                ),
                "document_ids_sha256": _staging_document_ids_sha256(
                    connection,
                    where,
                    (split,),
                ),
                "document_set_sha256": _staging_document_set_sha256(
                    connection,
                    where,
                    (split,),
                ),
                "rows": 0,
                "tokens": 0,
                "source_token_counts": dict.fromkeys(source_counts, 0),
                "shards": [],
            }

    source_group = _source_group_function(
        selected_sources,
        source_aggregation["overflow_key"],
    )
    out = Path(shards_dir)
    out.mkdir(parents=True, exist_ok=True)
    generation, generation_tmp, generation_final, generation_relative = _new_shard_generation(out)
    try:
        with closing(_staging_connection(corpus.path, read_only=True)) as connection:
            split_assignment = _write_split_assignment_artifact(
                generation_tmp,
                generation_relative,
                (
                    (
                        str(row["doc_id"]),
                        str(row["raw_text_sha"]),
                        str(row["identity"]),
                        str(row["split"]),
                    )
                    for row in connection.execute(
                        """
                        SELECT doc_id, raw_text_sha, identity, split
                        FROM documents
                        WHERE decontaminated = 1 AND near_keep = 1
                        ORDER BY identity, raw_text_sha
                        """
                    )
                ),
            )
    except BaseException:
        _discard_temporary_generation(generation_tmp)
        raise
    if split_assignment["assignment_sha256"] != split_assignment_sha256:
        _discard_temporary_generation(generation_tmp)
        raise RuntimeError("split assignment artifact fingerprint mismatch")
    preparation: dict[str, Any] = {
        "mode": "sqlite_disk_backed",
        "staging_database": corpus.artifact(),
        "staging_config": staging_config,
        "python_memory_scope": (
            "No full document corpus is materialized in Python RAM. Memory still scales with one "
            "quality-bounded document, the evaluation denylist, the bounded near-dedup feature "
            "cache/candidate set, tokenizer trainer state, and one output shard buffer."
        ),
        "limitations": (
            "Near-deduplication is a bounded non-exhaustive heuristic. SQLite staging needs "
            "additional local disk, and the tokenizer library's internal trainer state is not "
            "claimed to be constant-memory."
        ),
    }
    if preparation_provenance is not None:
        preparation["provenance"] = dict(preparation_provenance)
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "format": "bos_aligned_rows",
        "seq_len": seq_len,
        "row_tokens": row_tokens,
        "token_dtype": np.dtype(dtype).name,
        "vocab_size": vocab_size,
        "seed": seed,
        "val_fraction": val_fraction,
        "split_assignment_sha256": split_assignment_sha256,
        "split_assignment": split_assignment,
        "source_counts": source_counts,
        "source_token_counts": dict.fromkeys(source_counts, 0),
        "license_counts": license_counts,
        "provenance_summary": {
            "source_count_aggregation": source_aggregation,
            "license_count_aggregation": license_aggregation,
            "retained_document_provenance": provenance_fingerprint,
            "source_token_accounting": (
                "Encoded document tokens plus packing EOS targets, attributed to the source "
                "family of the document that caused each target."
            ),
        },
        "splits": split_metadata,
        "corpus_audit": corpus.corpus_audit,
        "preparation": preparation,
    }

    try:
        for split in ("train", "val"):
            rows: list[list[int]] = []
            shard_index = 0
            split_meta = manifest["splits"][split]
            split_token_counts = split_meta["source_token_counts"]

            def flush(split_name: str = split, split_stats: dict[str, Any] = split_meta) -> None:
                nonlocal rows, shard_index
                if not rows:
                    return
                entry = _write_packed_shard(
                    generation_tmp,
                    generation_relative,
                    split=split_name,
                    shard_index=shard_index,
                    rows=rows,
                    row_tokens=row_tokens,
                    pad_id=tokenizer.pad_id,
                    dtype=dtype,
                )
                split_stats["shards"].append(entry)
                split_stats["rows"] += len(rows)
                split_stats["tokens"] += int(sum(len(row) - 1 for row in rows))
                shard_index += 1
                rows = []

            for row in _packed_rows(
                corpus.iter_documents(split),
                tokenizer,
                row_tokens,
                source_token_counts=split_token_counts,
                source_group=source_group,
            ):
                rows.append(row)
                if len(rows) >= rows_per_shard:
                    flush()
            flush()
            if sum(split_token_counts.values()) != split_meta["tokens"]:
                raise RuntimeError(f"{split} source-token accounting does not match packed tokens")
            for family, count in split_token_counts.items():
                manifest["source_token_counts"][family] += count

        if tokenizer_training is not None:
            training_audit = dict(tokenizer_training)
            training_split = training_audit.get("split")
            if training_split is None:
                training_audit["documents"] = 0
            else:
                if training_split not in {"train", "val"}:
                    raise ValueError("tokenizer_training split must name a packed split")
                training_split_meta = manifest["splits"][training_split]
                training_audit.update(
                    {
                        "documents": training_split_meta["documents"],
                        "document_ids_sha256": training_split_meta["document_ids_sha256"],
                        "document_set_sha256": training_split_meta["document_set_sha256"],
                        "excluded_documents": (total_documents - training_split_meta["documents"]),
                    }
                )
            manifest["tokenizer_training"] = training_audit

        manifest["total_documents"] = total_documents
        manifest["total_tokens"] = sum(split["tokens"] for split in manifest["splits"].values())
        manifest["train_tokens"] = manifest["splits"]["train"]["tokens"]
        if sum(manifest["source_token_counts"].values()) != manifest["total_tokens"]:
            raise RuntimeError("source-token accounting does not match total packed tokens")
        _publish_shard_generation(
            out,
            generation,
            generation_tmp,
            generation_final,
            manifest,
        )
    except BaseException:
        _discard_temporary_generation(generation_tmp)
        raise
    return manifest


class PackedShardDataset:
    """Random-access, memory-mapped view over one split of packed token shards."""

    def __init__(self, shards_dir: str | Path, split: str = "train"):
        self.root = Path(shards_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(
                f"unsupported packed-shard manifest version {self.manifest.get('version')!r}"
            )
        generation = self.manifest.get("generation")
        if (
            not isinstance(generation, str)
            or len(generation) != 32
            or any(character not in "0123456789abcdef" for character in generation)
        ):
            raise ValueError("packed-shard manifest has an invalid generation")
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        entries = self.manifest["splits"][split]["shards"]
        if not entries:
            raise ValueError(f"split {split!r} has no rows")
        token_paths = [
            self._verified_artifact(
                entry,
                path_key="tokens",
                size_key="bytes",
                sha_key="sha256",
                generation=generation,
            )
            for entry in entries
        ]
        length_paths = [
            self._verified_artifact(
                entry,
                path_key="lengths",
                size_key="lengths_bytes",
                sha_key="lengths_sha256",
                generation=generation,
            )
            for entry in entries
        ]
        self.tokens = [np.load(path, mmap_mode="r", allow_pickle=False) for path in token_paths]
        self.lengths = [np.load(path, mmap_mode="r", allow_pickle=False) for path in length_paths]
        row_tokens = int(self.manifest["row_tokens"])
        expected_dtype = np.dtype(self.manifest["token_dtype"])
        for entry, tokens, lengths in zip(entries, self.tokens, self.lengths, strict=True):
            if tokens.ndim != 2 or tokens.shape[1] != row_tokens:
                raise ValueError(f"token shard {entry['tokens']!r} has an invalid shape")
            if tokens.dtype != expected_dtype:
                raise ValueError(f"token shard {entry['tokens']!r} has an invalid dtype")
            if lengths.ndim != 1 or len(lengths) != len(tokens):
                raise ValueError(f"length shard {entry['lengths']!r} has an invalid shape")
            if len(tokens) != entry.get("rows"):
                raise ValueError(f"shard {entry['tokens']!r} row count does not match manifest")
        self._offsets = np.cumsum([0, *(len(array) for array in self.tokens)])

    def _verified_artifact(
        self,
        entry: Mapping[str, Any],
        *,
        path_key: str,
        size_key: str,
        sha_key: str,
        generation: str,
    ) -> Path:
        relative_value = entry.get(path_key)
        expected_size = entry.get(size_key)
        expected_sha = entry.get(sha_key)
        if not isinstance(relative_value, str):
            raise ValueError(f"shard manifest is missing {path_key!r}")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ValueError(f"shard manifest is missing integer {size_key!r}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise ValueError(f"shard manifest has invalid {sha_key!r}")

        relative = Path(relative_value)
        expected_parent = Path("generations") / generation
        if relative.is_absolute() or relative.parent != expected_parent:
            raise ValueError(
                f"shard {relative_value!r} is not part of manifest generation {generation}"
            )
        root = self.root.resolve()
        generation_directory = self.root / expected_parent
        unresolved_path = self.root / relative
        if generation_directory.is_symlink() or unresolved_path.is_symlink():
            raise ValueError(f"shard path {relative_value!r} must not be a symbolic link")
        resolved_generation = generation_directory.resolve()
        path = unresolved_path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"shard path {relative_value!r} escapes the shard directory")
        if path.parent != resolved_generation:
            raise ValueError(
                f"shard {relative_value!r} resolves outside manifest generation {generation}"
            )
        if not path.is_file():
            raise ValueError(f"shard artifact {relative_value!r} is missing")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"shard artifact {relative_value!r} size mismatch: "
                f"expected {expected_size}, got {actual_size}"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"shard artifact {relative_value!r} SHA-256 mismatch")
        return path

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def seq_len(self) -> int:
        return int(self.manifest["seq_len"])

    def row_length(self, index: int) -> int:
        """Return one row's non-padding token length without reading its token payload."""

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = index - int(self._offsets[shard])
        return int(self.lengths[shard][local])

    def row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local = index - int(self._offsets[shard])
        raw = np.asarray(self.tokens[shard][local], dtype=np.int64)
        length = int(self.lengths[shard][local])
        x = raw[:-1].copy()
        y = raw[1:].copy()
        # y positions at and beyond length-1 predict padding and must not affect CE.
        y[max(0, length - 1) :] = -100
        return x, y

    def sample_batch_token_counts(self, batch_size: int, rng) -> tuple[int, int]:
        """Sample row indices and return exact runner counts without token tensors.

        The draws intentionally mirror :meth:`sample_batch`. Packed rows expose fixed-width
        inputs, but accounting excludes padding: a row of ``length`` tokens contributes
        ``length - 1`` supervised targets and one additional input token, capped at ``seq_len``.
        """

        indices = [rng.randrange(len(self)) for _ in range(batch_size)]
        loss_lengths = [max(0, min(self.seq_len, self.row_length(index) - 1)) for index in indices]
        loss_tokens = sum(loss_lengths)
        input_tokens = sum(min(loss_length + 1, self.seq_len) for loss_length in loss_lengths)
        return input_tokens, loss_tokens

    def sample_batch(self, batch_size: int, rng, device):
        import torch

        indices = [rng.randrange(len(self)) for _ in range(batch_size)]
        rows = [self.row(index) for index in indices]
        x = torch.tensor(np.stack([row[0] for row in rows]), dtype=torch.long, device=device)
        y = torch.tensor(np.stack([row[1] for row in rows]), dtype=torch.long, device=device)
        return x, y


def suggested_training_tokens(num_params: int, *, tokens_per_param: float = 20.0) -> int:
    """Chinchilla-style starting point; data-quality ablations should tune this per tier."""

    if num_params <= 0 or tokens_per_param <= 0:
        raise ValueError("num_params and tokens_per_param must be positive")
    return math.ceil(num_params * tokens_per_param)
