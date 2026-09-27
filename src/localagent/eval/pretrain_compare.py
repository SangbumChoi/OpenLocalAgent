"""Paired document-bootstrap comparison for pretraining scorecard sidecars."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from localagent.eval.pretrain_scorecard import (
    DOCUMENT_SIDECAR_KIND,
    DOCUMENT_SIDECAR_SCHEMA_VERSION,
)
from localagent.train.stage_data import canonical_sha256

COMPARISON_KIND = "localagent_paired_pretrain_comparison"
COMPARISON_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_HEADER_FIELDS = frozenset(
    {
        "record_type",
        "kind",
        "schema_version",
        "bindings",
        "checkpoint_step",
        "checkpoint_token_accounting",
        "groups",
        "evaluation",
        "validation",
    }
)
_OPTIONAL_HEADER_FIELDS = frozenset({"checkpoint_training_seed"})
_HEADER_FIELDS = _REQUIRED_HEADER_FIELDS | _OPTIONAL_HEADER_FIELDS
_DOCUMENT_FIELDS = frozenset(
    {
        "record_type",
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
_SHARED_BINDINGS = (
    "tokenizer_sha256",
    "manifest_sha256",
    "manifest_canonical_sha256",
    "staging_database_sha256",
    "split_assignment_sha256",
    "validation_document_set_sha256",
    "validation_document_content_sha256",
)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path, *, label: str) -> tuple[Path, dict[str, int | str]]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {source}")
    if not source.is_file():
        raise ValueError(f"{label} is missing or is not a file: {source}")
    return source, {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _fingerprint(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("ascii")).hexdigest()


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


@dataclass(frozen=True)
class DocumentMetric:
    document_identity_sha256: str
    document_content_sha256: str
    source_family: str
    source_group: str | None
    utf8_bytes: int
    tokens: int
    nll_nats: float
    correct_tokens: int

    @property
    def key(self) -> tuple[str, str]:
        return self.document_identity_sha256, self.document_content_sha256


@dataclass(frozen=True)
class LoadedSidecar:
    header: Mapping[str, Any]
    documents: Mapping[tuple[str, str], DocumentMetric]
    artifact: Mapping[str, int | str]


def _group_owners(groups: object) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("document sidecar has no source-group mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    owners: dict[str, str] = {}
    for raw_name, raw_selectors in groups.items():
        if not isinstance(raw_name, str) or not raw_name or raw_name == "overall":
            raise ValueError("document sidecar contains an invalid source-group name")
        if (
            not isinstance(raw_selectors, list)
            or not raw_selectors
            or any(not isinstance(selector, str) or not selector for selector in raw_selectors)
        ):
            raise ValueError(f"document sidecar source group {raw_name!r} is invalid")
        selectors = tuple(raw_selectors)
        if len(set(selectors)) != len(selectors):
            raise ValueError(f"document sidecar source group {raw_name!r} repeats a selector")
        for selector in selectors:
            previous = owners.setdefault(selector, raw_name)
            if previous != raw_name:
                raise ValueError(
                    f"document sidecar source-family selector {selector!r} overlaps groups"
                )
        normalized[raw_name] = selectors
    return normalized, owners


def _parse_document(record: object, *, line_number: int) -> DocumentMetric:
    if not isinstance(record, dict):
        raise ValueError(f"document sidecar line {line_number} must be a JSON object")
    unexpected = sorted(set(record) - _DOCUMENT_FIELDS)
    missing = sorted(_DOCUMENT_FIELDS - set(record))
    if unexpected or missing:
        detail = []
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        if missing:
            detail.append("missing=" + ",".join(missing))
        raise ValueError(
            f"document sidecar line {line_number} has invalid fields: {'; '.join(detail)}"
        )
    if record.get("record_type") != "document":
        raise ValueError(f"document sidecar line {line_number} has invalid record_type")
    identity = record.get("document_identity_sha256")
    content = record.get("document_content_sha256")
    if not _valid_sha256(identity) or not _valid_sha256(content):
        raise ValueError(f"document sidecar line {line_number} has invalid document bindings")
    source_family = record.get("source_family")
    source_group = record.get("source_group")
    if not isinstance(source_family, str) or not source_family:
        raise ValueError(f"document sidecar line {line_number} has invalid source_family")
    if source_group is not None and (not isinstance(source_group, str) or not source_group):
        raise ValueError(f"document sidecar line {line_number} has invalid source_group")
    utf8_bytes = _positive_integer(
        record.get("utf8_bytes"),
        label=f"document sidecar line {line_number} utf8_bytes",
    )
    tokens = _positive_integer(
        record.get("tokens"),
        label=f"document sidecar line {line_number} tokens",
    )
    correct_tokens = record.get("correct_tokens")
    if (
        isinstance(correct_tokens, bool)
        or not isinstance(correct_tokens, int)
        or not 0 <= correct_tokens <= tokens
    ):
        raise ValueError(
            f"document sidecar line {line_number} correct_tokens is out of range"
        )
    nll_nats = _finite_number(
        record.get("nll_nats"),
        label=f"document sidecar line {line_number} nll_nats",
    )
    if nll_nats < 0.0:
        raise ValueError(f"document sidecar line {line_number} nll_nats is negative")
    return DocumentMetric(
        document_identity_sha256=str(identity),
        document_content_sha256=str(content),
        source_family=source_family,
        source_group=source_group,
        utf8_bytes=utf8_bytes,
        tokens=tokens,
        nll_nats=nll_nats,
        correct_tokens=correct_tokens,
    )


def load_document_sidecar(path: str | Path) -> LoadedSidecar:
    """Load and fully validate one compact per-document metric sidecar."""

    source, artifact = _artifact(path, label="pretrain document sidecar")
    with source.open(encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"document sidecar is empty: {source}")
        try:
            header = json.loads(first_line)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"document sidecar header is invalid JSON: {source}") from error
        if not isinstance(header, dict) or header.get("record_type") != "header":
            raise ValueError(f"document sidecar first line is not a header: {source}")
        if (
            not _REQUIRED_HEADER_FIELDS.issubset(header)
            or not set(header).issubset(_HEADER_FIELDS)
        ):
            raise ValueError(
                f"document sidecar header has unexpected or missing fields: {source}"
            )
        if header.get("kind") != DOCUMENT_SIDECAR_KIND:
            raise ValueError(f"document sidecar kind is unsupported: {source}")
        if header.get("schema_version") != DOCUMENT_SIDECAR_SCHEMA_VERSION:
            raise ValueError(f"document sidecar schema version is unsupported: {source}")
        bindings = header.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"document sidecar has no artifact bindings: {source}")
        required_bindings = {
            "checkpoint_sha256",
            "model_config_sha256",
            *_SHARED_BINDINGS,
        }
        invalid_bindings = sorted(
            key for key in required_bindings if not _valid_sha256(bindings.get(key))
        )
        if invalid_bindings:
            raise ValueError(
                "document sidecar has invalid artifact binding(s): "
                + ", ".join(invalid_bindings)
            )
        groups, owners = _group_owners(header.get("groups"))
        if not isinstance(header.get("evaluation"), Mapping):
            raise ValueError(f"document sidecar has no evaluation protocol: {source}")
        token_accounting = header.get("checkpoint_token_accounting")
        if token_accounting is not None:
            if not isinstance(token_accounting, Mapping):
                raise ValueError(
                    f"document sidecar checkpoint token accounting is invalid: {source}"
                )
            for key in ("input_tokens", "loss_tokens"):
                value = token_accounting.get(key)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ValueError(
                        f"document sidecar checkpoint {key} accounting is invalid: {source}"
                    )
        validation = header.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError(f"document sidecar has no validation summary: {source}")

        documents: dict[tuple[str, str], DocumentMetric] = {}
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                raise ValueError(f"document sidecar line {line_number} is blank")
            try:
                raw_document = json.loads(line)
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(
                    f"document sidecar line {line_number} is invalid JSON"
                ) from error
            document = _parse_document(raw_document, line_number=line_number)
            if document.key in documents:
                raise ValueError(
                    f"document sidecar repeats document binding {document.key[0]}"
                )
            expected_group = owners.get(document.source_family)
            if document.source_group != expected_group:
                raise ValueError(
                    f"document sidecar source-group assignment disagrees for {document.key[0]}"
                )
            documents[document.key] = document

    if not documents:
        raise ValueError(f"document sidecar has no document rows: {source}")
    expected_documents = _positive_integer(
        validation.get("documents"),
        label="document sidecar validation documents",
    )
    expected_bytes = _positive_integer(
        validation.get("utf8_bytes"),
        label="document sidecar validation utf8_bytes",
    )
    expected_tokens = _positive_integer(
        validation.get("tokens"),
        label="document sidecar validation tokens",
    )
    expected_correct = validation.get("correct_tokens")
    if (
        isinstance(expected_correct, bool)
        or not isinstance(expected_correct, int)
        or not 0 <= expected_correct <= expected_tokens
    ):
        raise ValueError("document sidecar validation correct_tokens is invalid")
    if len(documents) != expected_documents:
        raise ValueError("document sidecar row count does not match its validation summary")
    if sum(document.utf8_bytes for document in documents.values()) != expected_bytes:
        raise ValueError("document sidecar byte count does not match its validation summary")
    if sum(document.tokens for document in documents.values()) != expected_tokens:
        raise ValueError("document sidecar token count does not match its validation summary")
    if sum(document.correct_tokens for document in documents.values()) != expected_correct:
        raise ValueError(
            "document sidecar correct-token count does not match its validation summary"
        )
    expected_nll = _finite_number(
        validation.get("nll_nats"),
        label="document sidecar validation nll_nats",
    )
    actual_nll = sum(document.nll_nats for document in documents.values())
    if expected_nll < 0.0 or not math.isclose(
        actual_nll,
        expected_nll,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise ValueError("document sidecar NLL does not match its validation summary")
    expected_metrics = {
        "cross_entropy_nats_per_token": actual_nll / expected_tokens,
        "bits_per_byte": actual_nll / (math.log(2.0) * expected_bytes),
        "top1_accuracy": expected_correct / expected_tokens,
    }
    for metric, expected_value in expected_metrics.items():
        recorded = _finite_number(
            validation.get(metric),
            label=f"document sidecar validation {metric}",
        )
        if not math.isclose(recorded, expected_value, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(
                f"document sidecar {metric} does not match its validation summary"
            )
    document_set_sha256 = _fingerprint(
        [document.document_identity_sha256 for document in documents.values()]
    )
    content_sha256 = _fingerprint(
        [
            f"{document.document_identity_sha256}:{document.document_content_sha256}"
            for document in documents.values()
        ]
    )
    if validation.get("document_set_sha256") != document_set_sha256:
        raise ValueError("document sidecar document-set fingerprint mismatch")
    if validation.get("document_content_sha256") != content_sha256:
        raise ValueError("document sidecar document-content fingerprint mismatch")
    if bindings.get("validation_document_set_sha256") != document_set_sha256:
        raise ValueError("document sidecar bound document-set fingerprint mismatch")
    if bindings.get("validation_document_content_sha256") != content_sha256:
        raise ValueError("document sidecar bound document-content fingerprint mismatch")
    present_groups = {document.source_group for document in documents.values()}
    empty_groups = sorted(set(groups) - present_groups)
    if empty_groups:
        raise ValueError(
            "document sidecar contains empty source group(s): " + ", ".join(empty_groups)
        )
    return LoadedSidecar(
        header=header,
        documents=documents,
        artifact={
            **artifact,
            "kind": DOCUMENT_SIDECAR_KIND,
            "schema_version": DOCUMENT_SIDECAR_SCHEMA_VERSION,
            "documents": len(documents),
        },
    )


def _arm_summary(documents: Sequence[DocumentMetric]) -> dict[str, int | float]:
    tokens = sum(document.tokens for document in documents)
    utf8_bytes = sum(document.utf8_bytes for document in documents)
    nll_nats = sum(document.nll_nats for document in documents)
    correct_tokens = sum(document.correct_tokens for document in documents)
    return {
        "documents": len(documents),
        "tokens": tokens,
        "utf8_bytes": utf8_bytes,
        "nll_nats": nll_nats,
        "correct_tokens": correct_tokens,
        "cross_entropy_nats_per_token": nll_nats / tokens,
        "bits_per_byte": nll_nats / (math.log(2.0) * utf8_bytes),
        "top1_accuracy": correct_tokens / tokens,
    }


def _distribution_summary(
    values: np.ndarray,
    *,
    estimate: float,
    confidence: float,
    higher_is_better: bool,
) -> dict[str, Any]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    if higher_is_better:
        attention_wins = values > 0.0
        hybrid_wins = values < 0.0
    else:
        attention_wins = values < 0.0
        hybrid_wins = values > 0.0
    ties = values == 0.0
    return {
        "estimate": estimate,
        "percentile_ci": {
            "confidence": confidence,
            "lower": float(lower),
            "upper": float(upper),
        },
        "attention_win_fraction": float(attention_wins.mean()),
        "hybrid_win_fraction": float(hybrid_wins.mean()),
        "tie_fraction": float(ties.mean()),
    }


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_subset(
    attention: Sequence[DocumentMetric],
    hybrid: Sequence[DocumentMetric],
    *,
    label: str,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    if len(attention) != len(hybrid) or not attention:
        raise ValueError(f"paired subset {label!r} is empty or unaligned")
    attention_summary = _arm_summary(attention)
    hybrid_summary = _arm_summary(hybrid)
    tokens = np.asarray([document.tokens for document in attention], dtype=np.float64)
    utf8_bytes = np.asarray(
        [document.utf8_bytes for document in attention],
        dtype=np.float64,
    )
    delta_nll = np.asarray(
        [
            attention_document.nll_nats - hybrid_document.nll_nats
            for attention_document, hybrid_document in zip(attention, hybrid, strict=True)
        ],
        dtype=np.float64,
    )
    delta_correct = np.asarray(
        [
            attention_document.correct_tokens - hybrid_document.correct_tokens
            for attention_document, hybrid_document in zip(attention, hybrid, strict=True)
        ],
        dtype=np.float64,
    )
    estimates = {
        "cross_entropy_nats_per_token": float(delta_nll.sum() / tokens.sum()),
        "bits_per_byte": float(delta_nll.sum() / (math.log(2.0) * utf8_bytes.sum())),
        "top1_accuracy": float(delta_correct.sum() / tokens.sum()),
    }

    distributions = {
        "cross_entropy_nats_per_token": np.empty(resamples, dtype=np.float64),
        "bits_per_byte": np.empty(resamples, dtype=np.float64),
        "top1_accuracy": np.empty(resamples, dtype=np.float64),
    }
    rng = np.random.default_rng(_derived_seed(seed, label))
    document_count = len(attention)
    batch_size = max(1, min(256, 1_000_000 // document_count))
    for start in range(0, resamples, batch_size):
        batch = min(batch_size, resamples - start)
        indices = rng.integers(0, document_count, size=(batch, document_count))
        sampled_tokens = tokens[indices].sum(axis=1)
        sampled_bytes = utf8_bytes[indices].sum(axis=1)
        sampled_delta_nll = delta_nll[indices].sum(axis=1)
        sampled_delta_correct = delta_correct[indices].sum(axis=1)
        stop = start + batch
        distributions["cross_entropy_nats_per_token"][start:stop] = (
            sampled_delta_nll / sampled_tokens
        )
        distributions["bits_per_byte"][start:stop] = (
            sampled_delta_nll / (math.log(2.0) * sampled_bytes)
        )
        distributions["top1_accuracy"][start:stop] = (
            sampled_delta_correct / sampled_tokens
        )

    difference = {
        metric: _distribution_summary(
            distribution,
            estimate=estimates[metric],
            confidence=confidence,
            higher_is_better=metric == "top1_accuracy",
        )
        for metric, distribution in distributions.items()
    }
    return {
        "documents": document_count,
        "tokens": int(tokens.sum()),
        "utf8_bytes": int(utf8_bytes.sum()),
        "attention": attention_summary,
        "hybrid": hybrid_summary,
        "difference_attention_minus_hybrid": difference,
    }


def compare_pretrain_sidecars(
    attention_path: str | Path,
    hybrid_path: str | Path,
    *,
    seed: int = 2026,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compare two aligned sidecars with a paired nonparametric document bootstrap."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("bootstrap seed must be an integer in [0, 2**63)")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 10_000:
        raise ValueError("paired document bootstrap requires at least 10000 resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    attention = load_document_sidecar(attention_path)
    hybrid = load_document_sidecar(hybrid_path)
    attention_bindings = attention.header["bindings"]
    hybrid_bindings = hybrid.header["bindings"]
    attention_training_seed = attention.header.get("checkpoint_training_seed")
    hybrid_training_seed = hybrid.header.get("checkpoint_training_seed")
    if (attention_training_seed is None) != (hybrid_training_seed is None):
        raise ValueError("paired sidecars must both record their checkpoint training seed")
    if attention_training_seed is not None:
        if (
            isinstance(attention_training_seed, bool)
            or not isinstance(attention_training_seed, int)
            or not 0 <= attention_training_seed < 2**63
            or hybrid_training_seed != attention_training_seed
        ):
            raise ValueError("paired sidecars have invalid or mismatched training seeds")
    attention_config_sha256 = attention_bindings.get("pretrain_config_sha256")
    hybrid_config_sha256 = hybrid_bindings.get("pretrain_config_sha256")
    if (attention_config_sha256 is None) != (hybrid_config_sha256 is None):
        raise ValueError(
            "paired sidecars must both bind their pretraining configurations"
        )
    if attention_config_sha256 is not None and (
        not _valid_sha256(attention_config_sha256)
        or not _valid_sha256(hybrid_config_sha256)
    ):
        raise ValueError("paired sidecars have invalid pretraining configuration bindings")
    mismatched_bindings = [
        key
        for key in _SHARED_BINDINGS
        if attention_bindings.get(key) != hybrid_bindings.get(key)
    ]
    if mismatched_bindings:
        raise ValueError(
            "paired sidecars have mismatched shared binding(s): "
            + ", ".join(mismatched_bindings)
        )
    if attention.header.get("groups") != hybrid.header.get("groups"):
        raise ValueError("paired sidecars use different source-group mappings")
    if attention.header.get("evaluation") != hybrid.header.get("evaluation"):
        raise ValueError("paired sidecars use different evaluation protocols")
    if set(attention.documents) != set(hybrid.documents):
        raise ValueError("paired sidecars do not contain the same document bindings")

    ordered_keys = sorted(attention.documents)
    attention_documents = [attention.documents[key] for key in ordered_keys]
    hybrid_documents = [hybrid.documents[key] for key in ordered_keys]
    for attention_document, hybrid_document in zip(
        attention_documents,
        hybrid_documents,
        strict=True,
    ):
        shared = (
            "source_family",
            "source_group",
            "utf8_bytes",
            "tokens",
        )
        mismatches = [
            field
            for field in shared
            if getattr(attention_document, field) != getattr(hybrid_document, field)
        ]
        if mismatches:
            raise ValueError(
                "paired document metadata mismatch for "
                f"{attention_document.document_identity_sha256}: {', '.join(mismatches)}"
            )

    groups = sorted(attention.header["groups"])
    group_results = {}
    for group in groups:
        positions = [
            index
            for index, document in enumerate(attention_documents)
            if document.source_group == group
        ]
        group_results[group] = _paired_subset(
            [attention_documents[index] for index in positions],
            [hybrid_documents[index] for index in positions],
            label=f"group:{group}",
            seed=seed,
            resamples=resamples,
            confidence=confidence,
        )
    overall = _paired_subset(
        attention_documents,
        hybrid_documents,
        label="overall",
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )
    matched_bindings = {
        key: attention_bindings[key] for key in _SHARED_BINDINGS
    }
    comparison_identity = canonical_sha256(
        {
            "kind": COMPARISON_KIND,
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "method": "paired_nonparametric_document_bootstrap",
            "attention_sidecar_sha256": attention.artifact["sha256"],
            "hybrid_sidecar_sha256": hybrid.artifact["sha256"],
            "seed": seed,
            "resamples": resamples,
            "confidence": confidence,
        }
    )
    attention_input = {
        "sidecar": dict(attention.artifact),
        "checkpoint_sha256": attention_bindings["checkpoint_sha256"],
        "model_config_sha256": attention_bindings["model_config_sha256"],
        "token_accounting": attention.header.get("checkpoint_token_accounting"),
    }
    hybrid_input = {
        "sidecar": dict(hybrid.artifact),
        "checkpoint_sha256": hybrid_bindings["checkpoint_sha256"],
        "model_config_sha256": hybrid_bindings["model_config_sha256"],
        "token_accounting": hybrid.header.get("checkpoint_token_accounting"),
    }
    if attention_config_sha256 is not None:
        attention_input["pretrain_config_sha256"] = attention_config_sha256
        hybrid_input["pretrain_config_sha256"] = hybrid_config_sha256
    if attention_training_seed is not None:
        attention_input["training_seed"] = attention_training_seed
        hybrid_input["training_seed"] = hybrid_training_seed
    return {
        "kind": COMPARISON_KIND,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "comparison_sha256": comparison_identity,
        "inputs": {
            "attention": attention_input,
            "hybrid": hybrid_input,
        },
        "matched_bindings": matched_bindings,
        "evaluation": dict(attention.header["evaluation"]),
        "bootstrap": {
            "unit": "document",
            "pairing": "document_identity_sha256+document_content_sha256",
            "method": "nonparametric_resampling_with_replacement",
            "random_generator": "numpy.PCG64",
            "seed": seed,
            "resamples": resamples,
            "confidence": confidence,
            "difference": "attention_minus_hybrid",
            "win_fraction": (
                "fraction of bootstrap replicates favoring attention; lower is better for "
                "CE/BPB and higher is better for accuracy"
            ),
        },
        "overall": overall,
        "groups": group_results,
    }
