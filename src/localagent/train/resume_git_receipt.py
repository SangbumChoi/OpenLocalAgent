"""Explicit, external authorization for one SFT resume across Git worktree-only drift."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from localagent.train.stage_data import canonical_sha256

RECEIPT_KIND = "localagent.sft_resume_git_lineage_migration"
RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_FIELDS = {
    "kind",
    "schema_version",
    "stage",
    "checkpoint_sha256",
    "recorded_lineage",
    "expected_lineage",
    "reason",
    "evidence",
    "receipt_self_sha256",
}
_GIT_FIELDS = {
    "commit",
    "repository_sha256",
    "dirty",
    "worktree_sha256",
}


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_git_object_id(value: Any, *, label: str) -> str:
    """Accept object IDs from either SHA-1 or SHA-256 Git repositories."""

    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be 40 or 64 lowercase hexadecimal characters")
    return value


def _validated_git_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _GIT_FIELDS:
        raise ValueError(f"{label} must contain exactly {', '.join(sorted(_GIT_FIELDS))}")
    commit = _require_git_object_id(value.get("commit"), label=f"{label}.commit")
    repository_sha256 = _require_sha256(
        value.get("repository_sha256"),
        label=f"{label}.repository_sha256",
    )
    worktree_sha256 = _require_sha256(
        value.get("worktree_sha256"),
        label=f"{label}.worktree_sha256",
    )
    dirty = value.get("dirty")
    if not isinstance(dirty, bool):
        raise ValueError(f"{label}.dirty must be boolean")
    return {
        "commit": commit,
        "repository_sha256": repository_sha256,
        "dirty": dirty,
        "worktree_sha256": worktree_sha256,
    }


def _validated_git_only_lineage_migration(
    recorded_lineage: Any,
    expected_lineage: Any,
    *,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(recorded_lineage, Mapping) or not isinstance(expected_lineage, Mapping):
        raise TypeError("resume Git receipt lineages must be mappings")
    recorded = dict(recorded_lineage)
    expected = dict(expected_lineage)
    if recorded.get("stage") != stage or expected.get("stage") != stage:
        raise ValueError(f"resume Git receipt lineages must both record stage {stage!r}")
    recorded_non_git = {key: value for key, value in recorded.items() if key != "git"}
    expected_non_git = {key: value for key, value in expected.items() if key != "git"}
    if recorded_non_git != expected_non_git:
        mismatches = sorted(
            key
            for key in set(recorded_non_git) | set(expected_non_git)
            if recorded_non_git.get(key) != expected_non_git.get(key)
        )
        raise ValueError(
            "resume Git receipt cannot authorize non-Git lineage drift: "
            + ", ".join(mismatches)
        )
    recorded_git = _validated_git_identity(recorded.get("git"), label="recorded lineage.git")
    expected_git = _validated_git_identity(expected.get("git"), label="expected lineage.git")
    for key in ("commit", "repository_sha256"):
        if recorded_git[key] != expected_git[key]:
            raise ValueError(
                f"resume Git receipt cannot authorize lineage.git.{key} drift"
            )
    if recorded_git == expected_git:
        raise ValueError("resume Git receipt is unnecessary because Git lineages already match")
    return recorded, expected


def build_resume_git_receipt(
    *,
    checkpoint_sha256: str,
    recorded_lineage: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    stage: str,
    reason: str,
    evidence: Sequence[str],
) -> dict[str, Any]:
    """Build a deterministic, self-hashed receipt without touching the checkpoint."""

    if stage != "sft":
        raise ValueError("resume Git receipts currently support only stage 'sft'")
    checkpoint_digest = _require_sha256(
        checkpoint_sha256,
        label="resume checkpoint sha256",
    )
    recorded, expected = _validated_git_only_lineage_migration(
        recorded_lineage,
        expected_lineage,
        stage=stage,
    )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resume Git receipt reason must be a non-empty string")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("resume Git receipt evidence must be a sequence of strings")
    normalized_evidence = []
    for item in evidence:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("resume Git receipt evidence entries must be non-empty strings")
        normalized_evidence.append(item.strip())
    if not normalized_evidence:
        raise ValueError("resume Git receipt requires at least one evidence entry")
    payload = {
        "kind": RECEIPT_KIND,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "stage": stage,
        "checkpoint_sha256": checkpoint_digest,
        "recorded_lineage": recorded,
        "expected_lineage": expected,
        "reason": reason.strip(),
        "evidence": normalized_evidence,
    }
    payload["receipt_self_sha256"] = canonical_sha256(payload)
    return payload


def assert_resume_git_receipt(
    receipt: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    recorded_lineage: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    stage: str,
) -> None:
    """Validate one receipt against the exact checkpoint and current full lineage."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError(
            "resume Git receipt must contain exactly " + ", ".join(sorted(_RECEIPT_FIELDS))
        )
    if receipt.get("kind") != RECEIPT_KIND:
        raise ValueError("resume Git receipt kind is unsupported")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise ValueError("resume Git receipt schema version is unsupported")
    if receipt.get("stage") != stage:
        raise ValueError("resume Git receipt stage mismatch")
    declared_self_hash = _require_sha256(
        receipt.get("receipt_self_sha256"),
        label="resume Git receipt self sha256",
    )
    without_self_hash = dict(receipt)
    without_self_hash.pop("receipt_self_sha256")
    if canonical_sha256(without_self_hash) != declared_self_hash:
        raise ValueError("resume Git receipt self-hash mismatch")
    expected_checkpoint_sha256 = _require_sha256(
        checkpoint_sha256,
        label="resume checkpoint sha256",
    )
    if receipt.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("resume Git receipt checkpoint sha256 mismatch")
    recorded, expected = _validated_git_only_lineage_migration(
        recorded_lineage,
        expected_lineage,
        stage=stage,
    )
    if receipt.get("recorded_lineage") != recorded:
        raise ValueError("resume Git receipt recorded lineage mismatch")
    if receipt.get("expected_lineage") != expected:
        raise ValueError("resume Git receipt expected current lineage mismatch")
    reason = receipt.get("reason")
    evidence = receipt.get("evidence")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("resume Git receipt reason is malformed")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(item, str) or not item.strip() for item in evidence)
    ):
        raise ValueError("resume Git receipt evidence is malformed")


def write_resume_git_receipt(
    path: str | Path,
    *,
    checkpoint_sha256: str,
    recorded_lineage: Mapping[str, Any],
    expected_lineage: Mapping[str, Any],
    stage: str,
    reason: str,
    evidence: Sequence[str],
) -> dict[str, Any]:
    """Create a receipt atomically enough for a new file and never overwrite an existing path."""

    receipt = build_resume_git_receipt(
        checkpoint_sha256=checkpoint_sha256,
        recorded_lineage=recorded_lineage,
        expected_lineage=expected_lineage,
        stage=stage,
        reason=reason,
        evidence=evidence,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"resume Git receipt contains duplicate key {key!r}")
        result[key] = value
    return result


def load_resume_git_receipt(path: str | Path) -> Mapping[str, Any]:
    """Load JSON without permitting duplicate keys; semantic validation happens separately."""

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load resume Git receipt: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("resume Git receipt root must be a mapping")
    return value
