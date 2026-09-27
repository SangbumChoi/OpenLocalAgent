"""Deterministic, lineage-bound evaluation for pretraining checkpoints.

The evaluator intentionally scores raw held-out documents rather than packed token rows.  The
paper corpus manifest binds an immutable SQLite staging database containing the filtered text and
document-level split assignment.  Reading that database lets us report a real bits-per-byte value
whose denominator is the original UTF-8 source bytes.

Only next-token language-model quality is measured here.  Tool and browser behavior require their
own post-training checkpoints and are deliberately outside this scorecard.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localagent.data.pretrain_corpus import (
    MANIFEST_VERSION,
    STAGING_VERSION,
    CorpusDocument,
    DiskBackedCorpus,
    _source_family,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer
from localagent.train.device import autocast_ctx, resolve_device, resolve_dtype
from localagent.train.stage_data import canonical_sha256, tokenizer_identity

SCORECARD_KIND = "localagent_pretrain_checkpoint_scorecard"
SCORECARD_SCHEMA_VERSION = 1
DOCUMENT_SIDECAR_KIND = "localagent_pretrain_document_metrics"
DOCUMENT_SIDECAR_SCHEMA_VERSION = 1
_DOCUMENT_SIDECAR_FIELDS = frozenset(
    {
        "document_identity_sha256",
        "document_content_sha256",
        "source_family",
        "source_group",
        "utf8_bytes",
        "tokens",
        "nll_nats",
        "correct_tokens",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GROUP_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_artifact(path: str | Path, *, label: str) -> tuple[Path, dict[str, int | str]]:
    artifact = Path(path)
    if artifact.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"{label} is missing or is not a file: {artifact}")
    return artifact, {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": _sha256_file(artifact),
    }


def parse_source_groups(specifications: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Parse repeated ``NAME=SOURCE_FAMILY[,SOURCE_FAMILY...]`` CLI values."""

    groups: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"source group {specification!r} must use NAME=SOURCE_FAMILY[,SOURCE_FAMILY...]"
            )
        name, raw_selectors = specification.split("=", 1)
        name = name.strip()
        if _GROUP_NAME.fullmatch(name) is None:
            raise ValueError(f"source group name {name!r} is invalid")
        if name == "overall":
            raise ValueError("source group name 'overall' is reserved")
        if name in groups:
            raise ValueError(f"duplicate source group name {name!r}")
        selectors = tuple(
            selector.strip() for selector in raw_selectors.split(",") if selector.strip()
        )
        if not selectors:
            raise ValueError(f"source group {name!r} has no source-family selectors")
        if len(set(selectors)) != len(selectors):
            raise ValueError(f"source group {name!r} repeats a source-family selector")
        for selector in selectors:
            previous = owners.setdefault(selector, name)
            if previous != name:
                raise ValueError(
                    f"source-family selector {selector!r} belongs to both "
                    f"{previous!r} and {name!r}"
                )
        groups[name] = selectors
    if not groups:
        raise ValueError("at least one source group is required")
    return groups


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, int | str]]:
    manifest_path, artifact = _file_artifact(path, label="packed corpus manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{manifest_path}: invalid packed corpus manifest JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path}: packed corpus manifest must be an object")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"{manifest_path}: unsupported packed corpus manifest version "
            f"{manifest.get('version')!r}"
        )
    if manifest.get("format") != "bos_aligned_rows":
        raise ValueError(f"{manifest_path}: unsupported packed corpus format")
    generation = manifest.get("generation")
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise ValueError(f"{manifest_path}: packed corpus generation is invalid")
    assignment_sha256 = manifest.get("split_assignment_sha256")
    assignment = manifest.get("split_assignment")
    if not _valid_sha256(assignment_sha256) or not isinstance(assignment, dict):
        raise ValueError(f"{manifest_path}: frozen split assignment metadata is missing")
    if assignment.get("assignment_sha256") != assignment_sha256:
        raise ValueError(f"{manifest_path}: frozen split assignment fingerprints disagree")
    splits = manifest.get("splits")
    validation = splits.get("val") if isinstance(splits, dict) else None
    if not isinstance(validation, dict):
        raise ValueError(f"{manifest_path}: validation split metadata is missing")
    documents = validation.get("documents")
    if isinstance(documents, bool) or not isinstance(documents, int) or documents < 1:
        raise ValueError(f"{manifest_path}: validation split is empty")
    if not _valid_sha256(validation.get("document_set_sha256")):
        raise ValueError(f"{manifest_path}: validation document-set fingerprint is missing")
    artifact = {
        **artifact,
        "canonical_sha256": canonical_sha256(manifest),
        "generation": generation,
        "split_assignment_sha256": assignment_sha256,
    }
    return manifest_path, manifest, artifact


def _resolve_staging_database(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    override: str | Path | None,
) -> tuple[Path, dict[str, int | str]]:
    preparation = manifest.get("preparation")
    expected = preparation.get("staging_database") if isinstance(preparation, Mapping) else None
    if not isinstance(expected, Mapping):
        raise ValueError("packed corpus manifest has no immutable staging_database artifact")
    expected_bytes = expected.get("bytes")
    expected_sha256 = expected.get("sha256")
    declared_path = expected.get("path")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or not _valid_sha256(expected_sha256)
        or not isinstance(declared_path, str)
        or not declared_path
    ):
        raise ValueError("packed corpus staging_database artifact metadata is invalid")

    if override is not None:
        database_path = Path(override)
    else:
        declared = Path(declared_path)
        candidates = [declared] if declared.is_absolute() else [
            declared,
            manifest_path.parent / declared,
            manifest_path.parent / declared.name,
        ]
        matches = [candidate for candidate in candidates if candidate.is_file()]
        if not matches:
            raise ValueError(
                "packed corpus staging database is unavailable; pass --corpus-db with the "
                "artifact recorded by the manifest"
            )
        database_path = matches[0]

    database_path, actual = _file_artifact(
        database_path,
        label="raw held-out staging database",
    )
    if actual["bytes"] != expected_bytes:
        raise ValueError(
            "raw held-out staging database byte-size does not match packed corpus manifest"
        )
    if actual["sha256"] != expected_sha256:
        raise ValueError(
            "raw held-out staging database SHA-256 does not match packed corpus manifest"
        )
    return database_path, actual


def _manifest_tokenizer_sha256(manifest: Mapping[str, Any]) -> str | None:
    direct = manifest.get("tokenizer_sha256")
    if _valid_sha256(direct):
        return str(direct)
    tokenizer = manifest.get("tokenizer")
    if isinstance(tokenizer, Mapping) and _valid_sha256(tokenizer.get("sha256")):
        return str(tokenizer["sha256"])
    training = manifest.get("tokenizer_training")
    artifact = training.get("artifact") if isinstance(training, Mapping) else None
    if isinstance(artifact, Mapping) and _valid_sha256(artifact.get("sha256")):
        return str(artifact["sha256"])
    return None


def _resolve_tokenizer_path(
    checkpoint_path: Path,
    recorded_path: object,
    override: str | Path | None,
) -> Path | None:
    if override is not None:
        return Path(override)
    if not isinstance(recorded_path, str) or not recorded_path:
        return None
    declared = Path(recorded_path)
    candidates = [declared] if declared.is_absolute() else [
        declared,
        checkpoint_path.parent / declared,
        checkpoint_path.parent / declared.name,
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), declared)


@dataclass(frozen=True)
class _LoadedCheckpoint:
    model: LocalAgentLM
    tokenizer: Any
    config: ModelConfig
    checkpoint: Mapping[str, Any]
    checkpoint_artifact: dict[str, int | str]
    tokenizer_artifact: dict[str, Any]


def _load_checkpoint(
    checkpoint_path: str | Path,
    *,
    manifest: Mapping[str, Any],
    manifest_canonical_sha256: str,
    tokenizer_kind: str | None,
    tokenizer_path: str | Path | None,
) -> _LoadedCheckpoint:
    path, checkpoint_artifact = _file_artifact(checkpoint_path, label="pretrain checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("pretrain checkpoint payload must be a mapping")
    if checkpoint.get("stage") != "pretrain":
        raise ValueError("checkpoint stage must be 'pretrain'")

    raw_config = checkpoint.get("cfg")
    if isinstance(raw_config, Mapping):
        config_values = dict(raw_config)
    elif hasattr(raw_config, "__dict__"):
        config_values = dict(vars(raw_config))
    else:
        raise ValueError("pretrain checkpoint has no model configuration")
    config = ModelConfig(
        **{
            key: value
            for key, value in config_values.items()
            if key in ModelConfig.__dataclass_fields__
        }
    )
    config.assert_within_budget()

    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("pretrain checkpoint has no lineage metadata")
    if lineage.get("stage") != "pretrain":
        raise ValueError("checkpoint lineage stage must be 'pretrain'")
    pretrain_config_sha256 = lineage.get("config_sha256")
    if not _valid_sha256(pretrain_config_sha256):
        raise ValueError("checkpoint pretraining configuration lineage is missing")
    expected_model_sha256 = canonical_sha256(config.__dict__)
    if lineage.get("model_config_sha256") != expected_model_sha256:
        raise ValueError("checkpoint model configuration lineage mismatch")

    data = checkpoint.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("pretrain checkpoint has no packed-corpus data lineage")
    if data.get("kind") != "packed_shards" or data.get("split") != "train":
        raise ValueError("checkpoint data lineage must identify packed training shards")
    if data.get("manifest_sha256") != manifest_canonical_sha256:
        raise ValueError("checkpoint packed-corpus manifest lineage mismatch")
    expected_data_sha256 = canonical_sha256(
        {
            "kind": "packed_shards",
            "manifest_sha256": manifest_canonical_sha256,
            "split": "train",
        }
    )
    if lineage.get("data_sha256") != expected_data_sha256:
        raise ValueError("checkpoint packed-corpus lineage fingerprint mismatch")

    tokenizer_metadata = checkpoint.get("tokenizer")
    if not isinstance(tokenizer_metadata, Mapping):
        raise ValueError("pretrain checkpoint has no tokenizer lineage")
    recorded_kind = tokenizer_metadata.get("kind")
    if recorded_kind not in {"byte", "bpe"}:
        raise ValueError("checkpoint tokenizer kind is invalid")
    selected_kind = tokenizer_kind or str(recorded_kind)
    if selected_kind != recorded_kind:
        raise ValueError("configured tokenizer kind does not match checkpoint lineage")
    selected_path = _resolve_tokenizer_path(
        path,
        tokenizer_metadata.get("path"),
        tokenizer_path,
    )
    if selected_kind == "byte" and selected_path is not None:
        raise ValueError("byte tokenizer must not use a tokenizer artifact path")
    if selected_kind == "bpe" and selected_path is None:
        raise ValueError("BPE checkpoint requires --tokenizer-path")
    tokenizer = load_tokenizer(selected_kind, selected_path)
    identity = tokenizer_identity(
        selected_kind,
        vocab_size=tokenizer.vocab_size,
        path=selected_path,
    )
    if tokenizer_metadata.get("sha256") != identity["sha256"]:
        raise ValueError("checkpoint tokenizer artifact lineage mismatch")
    if lineage.get("tokenizer_sha256") != identity["sha256"]:
        raise ValueError("checkpoint tokenizer lineage fingerprint mismatch")
    manifest_tokenizer_sha256 = _manifest_tokenizer_sha256(manifest)
    if (
        manifest_tokenizer_sha256 is not None
        and manifest_tokenizer_sha256 != identity["sha256"]
    ):
        raise ValueError("packed corpus tokenizer artifact does not match checkpoint tokenizer")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match checkpoint model configuration")
    if manifest.get("vocab_size") != config.vocab_size:
        raise ValueError("packed corpus vocabulary does not match checkpoint model configuration")

    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("pretrain checkpoint has no state_dict")
    model = LocalAgentLM(config)
    model.load_state_dict(state, strict=True)
    tokenizer_artifact: dict[str, Any] = {
        "kind": selected_kind,
        "vocab_size": tokenizer.vocab_size,
        "sha256": identity["sha256"],
    }
    if selected_path is not None:
        _, file_artifact = _file_artifact(selected_path, label="tokenizer artifact")
        tokenizer_artifact["artifact"] = file_artifact
    return _LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        checkpoint=checkpoint,
        checkpoint_artifact={
            **checkpoint_artifact,
            "model_config_sha256": expected_model_sha256,
            "pretrain_config_sha256": str(pretrain_config_sha256),
        },
        tokenizer_artifact=tokenizer_artifact,
    )


def _document_identity(document: CorpusDocument) -> str:
    return hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()


def _document_content_binding(document: CorpusDocument) -> str:
    identity = _document_identity(document)
    text_sha256 = _document_text_sha256(document)
    return f"{identity}:{text_sha256}"


def _document_text_sha256(document: CorpusDocument) -> str:
    return hashlib.sha256(document.text.encode("utf-8")).hexdigest()


def _fingerprint(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("ascii")).hexdigest()


@dataclass
class _DocumentAudit:
    documents: int = 0
    utf8_bytes: int = 0
    identities: list[str] = field(default_factory=list)
    content_bindings: list[str] = field(default_factory=list)

    def add(self, document: CorpusDocument) -> None:
        byte_count = len(document.text.encode("utf-8"))
        if byte_count < 1:
            raise ValueError(f"held-out document {document.doc_id!r} has no UTF-8 bytes")
        self.documents += 1
        self.utf8_bytes += byte_count
        self.identities.append(_document_identity(document))
        self.content_bindings.append(_document_content_binding(document))

    def fingerprints(self) -> dict[str, int | str]:
        return {
            "documents": self.documents,
            "utf8_bytes": self.utf8_bytes,
            "document_set_sha256": _fingerprint(self.identities),
            "document_content_sha256": _fingerprint(self.content_bindings),
        }


@dataclass
class _MetricAccumulator:
    audit: _DocumentAudit = field(default_factory=_DocumentAudit)
    nll_nats: float = 0.0
    correct_tokens: int = 0
    tokens: int = 0

    def add_document(self, document: CorpusDocument) -> None:
        self.audit.add(document)

    def add_predictions(self, nll_nats: float, correct_tokens: int, tokens: int) -> None:
        self.nll_nats += nll_nats
        self.correct_tokens += correct_tokens
        self.tokens += tokens

    def result(self) -> dict[str, int | float | str]:
        if self.audit.documents < 1:
            raise ValueError("cannot report an empty evaluation group")
        if self.audit.utf8_bytes < 1 or self.tokens < 1:
            raise ValueError("evaluation group has no scoreable UTF-8 bytes or tokens")
        cross_entropy = self.nll_nats / self.tokens
        bits_per_byte = self.nll_nats / (math.log(2.0) * self.audit.utf8_bytes)
        accuracy = self.correct_tokens / self.tokens
        if not all(math.isfinite(value) for value in (cross_entropy, bits_per_byte, accuracy)):
            raise ValueError("evaluation produced a non-finite metric")
        return {
            **self.audit.fingerprints(),
            "tokens": self.tokens,
            "correct_tokens": self.correct_tokens,
            "nll_nats": self.nll_nats,
            "cross_entropy_nats_per_token": cross_entropy,
            "bits_per_byte": bits_per_byte,
            "top1_accuracy": accuracy,
        }


@dataclass
class _DocumentPrediction:
    document_identity_sha256: str
    document_content_sha256: str
    source_family: str
    source_group: str | None
    utf8_bytes: int
    expected_tokens: int
    nll_nats: float = 0.0
    correct_tokens: int = 0
    tokens: int = 0

    @classmethod
    def from_document(
        cls,
        document: CorpusDocument,
        *,
        source_family: str,
        source_group: str | None,
        tokens: int,
    ) -> _DocumentPrediction:
        return cls(
            document_identity_sha256=_document_identity(document),
            document_content_sha256=_document_text_sha256(document),
            source_family=source_family,
            source_group=source_group,
            utf8_bytes=len(document.text.encode("utf-8")),
            expected_tokens=tokens,
        )

    def add_predictions(self, nll_nats: float, correct_tokens: int, tokens: int) -> None:
        self.nll_nats += nll_nats
        self.correct_tokens += correct_tokens
        self.tokens += tokens

    def record(self) -> dict[str, int | float | str | None]:
        if self.tokens != self.expected_tokens:
            raise RuntimeError(
                "per-document token accounting does not match tokenizer output for "
                f"{self.document_identity_sha256}"
            )
        if (
            self.utf8_bytes < 1
            or self.tokens < 1
            or not 0 <= self.correct_tokens <= self.tokens
            or not math.isfinite(self.nll_nats)
            or self.nll_nats < 0.0
        ):
            raise RuntimeError(
                f"invalid per-document metrics for {self.document_identity_sha256}"
            )
        return {
            "document_identity_sha256": self.document_identity_sha256,
            "document_content_sha256": self.document_content_sha256,
            "source_family": self.source_family,
            "source_group": self.source_group,
            "utf8_bytes": self.utf8_bytes,
            "tokens": self.tokens,
            "nll_nats": self.nll_nats,
            "correct_tokens": self.correct_tokens,
        }


@dataclass(frozen=True)
class _ScoreRow:
    inputs: list[int]
    targets: list[int]
    group_name: str | None
    document: _DocumentPrediction | None


def _selector_owners(
    source_groups: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    normalized: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for name, raw_selectors in source_groups.items():
        if _GROUP_NAME.fullmatch(name) is None or name == "overall":
            raise ValueError(f"source group name {name!r} is invalid or reserved")
        selectors = tuple(str(selector) for selector in raw_selectors)
        if not selectors or any(not selector for selector in selectors):
            raise ValueError(f"source group {name!r} has no source-family selectors")
        if len(set(selectors)) != len(selectors):
            raise ValueError(f"source group {name!r} repeats a source-family selector")
        for selector in selectors:
            previous = owners.setdefault(selector, name)
            if previous != name:
                raise ValueError(
                    f"source-family selector {selector!r} belongs to multiple groups"
                )
        normalized[name] = selectors
    if not normalized:
        raise ValueError("at least one source group is required")
    return normalized, owners


def _audit_validation_documents(
    corpus: DiskBackedCorpus,
    *,
    expected_documents: int,
    expected_document_set_sha256: str,
    selector_owners: Mapping[str, str],
    group_names: Sequence[str],
) -> tuple[dict[str, int | str], Counter[str], Counter[str]]:
    overall = _DocumentAudit()
    group_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for document in corpus.iter_documents("val"):
        overall.add(document)
        family = _source_family(document.source, document.meta)
        source_counts[family] += 1
        owner = selector_owners.get(family)
        if owner is not None:
            group_counts[owner] += 1
    audit = overall.fingerprints()
    if audit["documents"] != expected_documents:
        raise ValueError(
            "raw held-out document count does not match packed corpus validation metadata"
        )
    if audit["document_set_sha256"] != expected_document_set_sha256:
        raise ValueError(
            "raw held-out document set does not match packed corpus validation metadata"
        )
    empty = [name for name in group_names if group_counts[name] == 0]
    if empty:
        raise ValueError("requested source group(s) are empty: " + ", ".join(sorted(empty)))
    return audit, group_counts, source_counts


def _score_batch(
    model: LocalAgentLM,
    rows: Sequence[_ScoreRow],
    *,
    pad_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    overall: _MetricAccumulator,
    groups: Mapping[str, _MetricAccumulator],
) -> None:
    width = max(len(row.inputs) for row in rows)
    inputs = torch.full(
        (len(rows), width),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(rows), width),
        -100,
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(rows):
        inputs[index, : len(row.inputs)] = torch.tensor(
            row.inputs,
            dtype=torch.long,
            device=device,
        )
        targets[index, : len(row.targets)] = torch.tensor(
            row.targets,
            dtype=torch.long,
            device=device,
        )

    with torch.inference_mode(), autocast_ctx(device, amp_dtype):
        logits, _ = model(inputs)
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)
    predictions = logits.argmax(dim=-1)
    for index, row in enumerate(rows):
        mask = targets[index] != -100
        token_count = int(mask.sum().item())
        nll_nats = float(losses[index][mask].sum().item())
        correct_tokens = int((predictions[index][mask] == targets[index][mask]).sum().item())
        overall.add_predictions(nll_nats, correct_tokens, token_count)
        if row.document is not None:
            row.document.add_predictions(nll_nats, correct_tokens, token_count)
        if row.group_name is not None:
            groups[row.group_name].add_predictions(nll_nats, correct_tokens, token_count)


def _checkpoint_token_accounting(
    checkpoint: Mapping[str, Any],
) -> dict[str, int | str] | None:
    accounting = checkpoint.get("token_accounting")
    if isinstance(accounting, Mapping):
        values = {
            "input_tokens": accounting.get("input_tokens"),
            "loss_tokens": accounting.get("loss_tokens"),
        }
        source = "checkpoint.token_accounting"
    else:
        values = {
            "input_tokens": checkpoint.get("input_tokens_seen"),
            "loss_tokens": checkpoint.get("tokens_seen"),
        }
        source = "checkpoint.legacy_token_fields"
    result: dict[str, int | str] = {"source": source}
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {name} accounting is invalid")
        result[name] = value
    return result if len(result) > 1 else None


def write_document_sidecar(
    header: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> dict[str, int | str]:
    """Atomically write compact header-plus-document JSONL without raw document text."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {**header, "record_type": "header"},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        for document in documents:
            if set(document) != _DOCUMENT_SIDECAR_FIELDS:
                raise ValueError(
                    "per-document metric sidecar records must contain only hashed bindings, "
                    "source labels, counts, and metrics"
                )
            handle.write(
                json.dumps(
                    {**document, "record_type": "document"},
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(destination)
    _, artifact = _file_artifact(destination, label="per-document metric sidecar")
    return {
        **artifact,
        "kind": DOCUMENT_SIDECAR_KIND,
        "schema_version": DOCUMENT_SIDECAR_SCHEMA_VERSION,
        "documents": len(documents),
    }


def evaluate_pretrain_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    source_groups: Mapping[str, Sequence[str]],
    *,
    corpus_db_path: str | Path | None = None,
    tokenizer_kind: str | None = None,
    tokenizer_path: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 1,
    chunk_length: int | None = None,
    document_sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every raw document in the manifest-bound validation split."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    normalized_groups, owners = _selector_owners(source_groups)
    manifest_path_resolved, manifest, manifest_artifact = _load_manifest(manifest_path)
    database_path, database_artifact = _resolve_staging_database(
        manifest_path_resolved,
        manifest,
        corpus_db_path,
    )
    loaded = _load_checkpoint(
        checkpoint_path,
        manifest=manifest,
        manifest_canonical_sha256=str(manifest_artifact["canonical_sha256"]),
        tokenizer_kind=tokenizer_kind,
        tokenizer_path=tokenizer_path,
    )

    resolved_chunk_length = (
        loaded.config.max_seq_len if chunk_length is None else int(chunk_length)
    )
    if not 1 <= resolved_chunk_length <= loaded.config.max_seq_len:
        raise ValueError(
            f"chunk_length must be in [1, {loaded.config.max_seq_len}]"
        )
    device_obj = resolve_device(device)
    amp_dtype = resolve_dtype(device_obj, dtype)
    loaded.model.to(device_obj).eval()

    corpus = DiskBackedCorpus(database_path)
    validation_meta = manifest["splits"]["val"]
    validation_audit, expected_group_counts, source_counts = _audit_validation_documents(
        corpus,
        expected_documents=int(validation_meta["documents"]),
        expected_document_set_sha256=str(validation_meta["document_set_sha256"]),
        selector_owners=owners,
        group_names=tuple(normalized_groups),
    )

    overall = _MetricAccumulator()
    group_metrics = {name: _MetricAccumulator() for name in normalized_groups}
    document_metrics: list[_DocumentPrediction] = []
    pending: list[_ScoreRow] = []
    for document in corpus.iter_documents("val"):
        family = _source_family(document.source, document.meta)
        group_name = owners.get(family)
        token_ids = loaded.tokenizer.encode(document.text, add_eos=False)
        if not token_ids:
            raise ValueError(f"held-out document {document.doc_id!r} has no tokenizer tokens")
        document_metric = None
        if document_sidecar_path is not None:
            document_metric = _DocumentPrediction.from_document(
                document,
                source_family=family,
                source_group=group_name,
                tokens=len(token_ids),
            )
            document_metrics.append(document_metric)
        overall.add_document(document)
        if group_name is not None:
            group_metrics[group_name].add_document(document)
        sequence = [int(loaded.tokenizer.eos_id), *map(int, token_ids)]
        for start in range(0, len(token_ids), resolved_chunk_length):
            stop = min(start + resolved_chunk_length, len(token_ids))
            pending.append(
                _ScoreRow(
                    inputs=sequence[start:stop],
                    targets=sequence[start + 1 : stop + 1],
                    group_name=group_name,
                    document=document_metric,
                )
            )
            if len(pending) >= batch_size:
                _score_batch(
                    loaded.model,
                    pending,
                    pad_id=int(loaded.tokenizer.pad_id),
                    device=device_obj,
                    amp_dtype=amp_dtype,
                    overall=overall,
                    groups=group_metrics,
                )
                pending = []
    if pending:
        _score_batch(
            loaded.model,
            pending,
            pad_id=int(loaded.tokenizer.pad_id),
            device=device_obj,
            amp_dtype=amp_dtype,
            overall=overall,
            groups=group_metrics,
        )

    aggregate = overall.result()
    if {
        key: aggregate[key]
        for key in ("documents", "utf8_bytes", "document_set_sha256", "document_content_sha256")
    } != validation_audit:
        raise RuntimeError("validation documents changed between audit and model evaluation")
    results = {name: group_metrics[name].result() for name in sorted(group_metrics)}
    for name, expected_count in expected_group_counts.items():
        if results[name]["documents"] != expected_count:
            raise RuntimeError(f"source group {name!r} changed during model evaluation")

    checkpoint_step = loaded.checkpoint.get("step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        checkpoint_step = None
    checkpoint_training_seed = loaded.checkpoint.get("training_seed")
    if checkpoint_training_seed is not None and (
        isinstance(checkpoint_training_seed, bool)
        or not isinstance(checkpoint_training_seed, int)
        or not 0 <= checkpoint_training_seed < 2**63
    ):
        raise ValueError("checkpoint training seed is invalid")
    token_accounting = _checkpoint_token_accounting(loaded.checkpoint)
    checkpoint_report = {
        **loaded.checkpoint_artifact,
        "stage": "pretrain",
        "step": checkpoint_step,
    }
    if token_accounting is not None:
        checkpoint_report["token_accounting"] = token_accounting
    if checkpoint_training_seed is not None:
        checkpoint_report["training_seed"] = checkpoint_training_seed
    report = {
        "kind": SCORECARD_KIND,
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "checkpoint": checkpoint_report,
        "tokenizer": loaded.tokenizer_artifact,
        "dataset": {
            "manifest": manifest_artifact,
            "staging_database": {
                **database_artifact,
                "staging_version": STAGING_VERSION,
            },
            "split": "val",
            "selection": "all_verified_validation_documents",
            "validation": validation_audit,
            "source_families": dict(sorted(source_counts.items())),
            "groups": {
                name: list(normalized_groups[name]) for name in sorted(normalized_groups)
            },
        },
        "evaluation": {
            "device": str(device_obj),
            "dtype": str(amp_dtype).removeprefix("torch."),
            "batch_size": batch_size,
            "chunk_length": resolved_chunk_length,
            "boundary_policy": (
                "predict source tokens only; prepend EOS once at document start; use "
                "non-overlapping target chunks with exact source-token coverage; later chunks "
                "reset model state but overlap the preceding source token as their first input; "
                "exclude closing EOS from CE and BPB"
            ),
        },
        "aggregate": aggregate,
        "groups": results,
    }
    if document_sidecar_path is not None:
        document_records = [document.record() for document in document_metrics]
        sidecar_header = {
            "kind": DOCUMENT_SIDECAR_KIND,
            "schema_version": DOCUMENT_SIDECAR_SCHEMA_VERSION,
            "bindings": {
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "model_config_sha256": report["checkpoint"]["model_config_sha256"],
                "pretrain_config_sha256": report["checkpoint"][
                    "pretrain_config_sha256"
                ],
                "tokenizer_sha256": report["tokenizer"]["sha256"],
                "manifest_sha256": report["dataset"]["manifest"]["sha256"],
                "manifest_canonical_sha256": report["dataset"]["manifest"][
                    "canonical_sha256"
                ],
                "staging_database_sha256": report["dataset"]["staging_database"]["sha256"],
                "split_assignment_sha256": report["dataset"]["manifest"][
                    "split_assignment_sha256"
                ],
                "validation_document_set_sha256": aggregate["document_set_sha256"],
                "validation_document_content_sha256": aggregate[
                    "document_content_sha256"
                ],
            },
            "checkpoint_step": checkpoint_step,
            "checkpoint_token_accounting": token_accounting,
            "groups": report["dataset"]["groups"],
            "evaluation": report["evaluation"],
            "validation": {
                key: aggregate[key]
                for key in (
                    "documents",
                    "utf8_bytes",
                    "tokens",
                    "correct_tokens",
                    "nll_nats",
                    "cross_entropy_nats_per_token",
                    "bits_per_byte",
                    "top1_accuracy",
                    "document_set_sha256",
                    "document_content_sha256",
                )
            },
        }
        if checkpoint_training_seed is not None:
            sidecar_header["checkpoint_training_seed"] = checkpoint_training_seed
        report["document_sidecar"] = write_document_sidecar(
            sidecar_header,
            document_records,
            document_sidecar_path,
        )
    return report


def write_scorecard(report: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write one canonical, finite JSON scorecard."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(destination)
