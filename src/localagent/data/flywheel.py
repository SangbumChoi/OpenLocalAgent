"""Deterministic Conversation flywheel: ingest -> mine -> verify -> append.

The runtime's future SQLite store is still a separate Phase-8 deliverable.  This module closes the
data side of the loop today using canonical ``Conversation`` JSONL exports, including normalized
public-agent artifacts.  It never treats an unverified eval row as training data.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from localagent.data.conversation_artifact import conversation_semantic_sha256
from localagent.data.prompt_contract import assistant_training_turns
from localagent.data.schema import Conversation

_MAX_STORE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ROW_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def ingest(store_path: str | Path) -> list[Conversation]:
    """Read a canonical Conversation JSONL export without changing its provenance metadata."""

    path = Path(store_path)
    if not path.is_file():
        raise ValueError(f"conversation export is missing: {path}")
    if path.stat().st_size > _MAX_STORE_BYTES:
        raise ValueError(f"conversation export exceeds {_MAX_STORE_BYTES} bytes")
    conversations: list[Conversation] = []
    with path.open("rb") as handle:
        for row_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > _MAX_ROW_BYTES:
                raise ValueError(f"conversation row {row_number} exceeds {_MAX_ROW_BYTES} bytes")
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"conversation row {row_number} is missing its LF terminator")
            try:
                line = raw_line[:-1].decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError(f"conversation row {row_number} is not UTF-8") from error
            conversation = Conversation.from_json(line)
            if conversation.to_json() != line:
                raise ValueError(
                    f"conversation row {row_number} is not canonical schema serialization"
                )
            conversations.append(conversation)
    return conversations


def _feedback_selected(meta: dict[str, Any]) -> bool:
    feedback = meta.get("feedback")
    if feedback is None:
        # Provenance-bound public rows are already selected by their source adapter and quality
        # policy.  An explicit quality rejection still wins.
        quality = meta.get("quality", {})
        return bool(meta.get("public_data")) and not (
            isinstance(quality, dict) and quality.get("accepted") is False
        )
    if not isinstance(feedback, dict):
        return False
    if feedback.get("rejected") is True or feedback.get("missing_knowledge") is True:
        return False
    positive = (
        feedback.get("preference_winner") is True
        or feedback.get("adopted") is True
        or feedback.get("knowledge_relevant") is True
    )
    uncertainty = feedback.get("uncertainty", 0.0)
    if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
        return False
    return positive and float(uncertainty) <= 0.5


def verify(conversation: Conversation) -> bool:
    """Rule-check one training candidate against schema, split, and provenance policy."""

    if not isinstance(conversation, Conversation):
        return False
    if conversation.meta.get("split") != "train":
        return False
    if conversation.meta.get("rule_verified") is not True:
        return False
    model_verified = conversation.meta.get("model_verified")
    if model_verified is not None and not isinstance(model_verified, bool):
        return False
    environment_executed = conversation.meta.get("environment_executed")
    if environment_executed is not None and not isinstance(environment_executed, bool):
        return False
    if conversation.meta.get("public_data") is True:
        if (
            conversation.meta.get("model_verified") is not False
            or conversation.meta.get("environment_executed") is not False
        ):
            return False
        provenance = conversation.meta.get("provenance")
        if not isinstance(provenance, dict):
            return False
        required = {
            "dataset",
            "subset",
            "revision",
            "record_id",
            "url",
            "license",
            "file_sha256",
            "source_line",
        }
        if set(provenance) != required:
            return False
        text_fields = required - {"source_line"}
        if not all(
            isinstance(provenance[field], str) and provenance[field]
            for field in text_fields
        ):
            return False
        if not str(provenance["url"]).startswith("https://"):
            return False
        if _SHA256_RE.fullmatch(str(provenance["file_sha256"])) is None:
            return False
        source_line = provenance["source_line"]
        if (
            isinstance(source_line, bool)
            or not isinstance(source_line, int)
            or source_line < 1
        ):
            return False
        capabilities = conversation.meta.get("capabilities")
        if (
            not isinstance(capabilities, list)
            or not all(isinstance(value, str) and value for value in capabilities)
            or capabilities != sorted(set(capabilities))
        ):
            return False
        action_count = conversation.meta.get("action_count")
        observed_actions = sum(
            len(message.tool_calls) for message in conversation.messages
        )
        if (
            isinstance(action_count, bool)
            or not isinstance(action_count, int)
            or action_count != observed_actions
        ):
            return False
    try:
        turns = assistant_training_turns(conversation)
    except (TypeError, ValueError):
        return False
    return bool(turns)


def mine(conversations: Iterable[Conversation]) -> list[Conversation]:
    """Select verified, positive/adopted or provenance-bound public TRAIN trajectories."""

    selected: dict[str, Conversation] = {}
    for conversation in conversations:
        if not verify(conversation) or not _feedback_selected(conversation.meta):
            continue
        fingerprint = conversation_semantic_sha256(conversation)
        selected.setdefault(fingerprint, conversation)
    return [selected[fingerprint] for fingerprint in sorted(selected)]


def _publish(path: Path, conversations: Sequence[Conversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for conversation in conversations:
                handle.write((conversation.to_json() + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_training_pool(
    out_jsonl: str | Path,
    *,
    store_path: str | Path | None = None,
    conversations: Sequence[Conversation] | None = None,
) -> int:
    """Ingest -> mine -> verify -> append; return the number of new semantic rows.

    Exactly one source is required: ``store_path`` for a canonical JSONL export or an in-memory
    ``conversations`` sequence.  Existing output rows are retained, and semantically duplicate
    candidates are skipped.
    """

    if (store_path is None) == (conversations is None):
        raise ValueError("provide exactly one of store_path or conversations")
    candidates = ingest(store_path) if store_path is not None else list(conversations or ())
    mined = mine(candidates)
    output = Path(out_jsonl)
    existing = ingest(output) if output.exists() else []
    fingerprints = {conversation_semantic_sha256(row) for row in existing}
    additions = [
        row
        for row in mined
        if conversation_semantic_sha256(row) not in fingerprints
    ]
    # ``mine`` is fingerprint-sorted, so appending is deterministic for the same source bytes.
    _publish(output, [*existing, *additions])
    return len(additions)


def pool_sha256(path: str | Path) -> str:
    """Return the exact byte identity of a published flywheel pool."""

    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
