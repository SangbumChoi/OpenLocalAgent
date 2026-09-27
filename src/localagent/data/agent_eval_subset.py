"""Deterministically derive the bounded WebGPU capability-pilot evaluation subset.

The production subset is deliberately derived from the already-frozen ``agent_eval.jsonl``
artifact. Generating a fresh evaluation seed after pretraining would weaken the held-out claim:
the proxy pretraining corpus was decontaminated against this exact frozen source.

Selection never reserializes a conversation. Selected JSONL lines are copied byte-for-byte and
published in their original source order.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localagent.data.schema import Conversation, Role

PRODUCTION_SOURCE_SHA256 = (
    "e2c2406865a076f21ca1dd8747187ae2c2a2af0c06888b207b702aa6f2ebfb07"
)
PRODUCTION_SOURCE_MANIFEST_SHA256 = (
    "17457caea77ca7d2501d2d3f33cc57508a95d0c0b026a4acbac315bdd29ba234"
)
PRODUCTION_TRAIN_SOURCE_SHA256 = (
    "161c4b8baab4adb75fe7f968f2bfc15b3e1828c5e376a16b494ee61072179a00"
)
PRODUCTION_OUTPUT_SHA256 = (
    "a46aaba70b0135cdb10abba760de23f17ddc61ad9515d277b1b6d21f1b4ee179"
)
PRODUCTION_OUTPUT_BYTES = 659_063
PRODUCTION_SOURCE_ROWS = 5_000
PRODUCTION_TRAIN_SOURCE_ROWS = 5_000
PRODUCTION_SINGLE_TURN_CATEGORIES = 53
PRODUCTION_MULTI_TURN_ROWS = 12
PRODUCTION_OUTPUT_ROWS = 65
PRODUCTION_EPISODE_KINDS = frozenset(
    {
        "coding_episode",
        "computer_use_episode",
        "planner_episode",
        "productivity_episode",
    }
)
PRODUCTION_PLANNER_LENGTHS = frozenset(range(5))

_ARTIFACT_TYPE = "localagent_derived_conversation_eval_subset"
_ALGORITHM = (
    "minimum_raw_line_sha256_per_single_turn_category_then_episode_kind_and_"
    "planner_length_coverage_then_raw_line_sha256_fill_v1"
)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SubsetContract:
    """Immutable identities and coverage requirements for one derivation."""

    source_sha256: str
    source_rows: int
    single_turn_categories: int
    multi_turn_rows: int
    episode_kinds: frozenset[str]
    planner_lengths: frozenset[int]
    output_rows: int
    output_bytes: int | None = None
    output_sha256: str | None = None
    source_manifest_sha256: str | None = None
    train_source_sha256: str | None = None
    train_source_rows: int | None = None


PRODUCTION_CONTRACT = SubsetContract(
    source_sha256=PRODUCTION_SOURCE_SHA256,
    source_manifest_sha256=PRODUCTION_SOURCE_MANIFEST_SHA256,
    train_source_sha256=PRODUCTION_TRAIN_SOURCE_SHA256,
    source_rows=PRODUCTION_SOURCE_ROWS,
    train_source_rows=PRODUCTION_TRAIN_SOURCE_ROWS,
    single_turn_categories=PRODUCTION_SINGLE_TURN_CATEGORIES,
    multi_turn_rows=PRODUCTION_MULTI_TURN_ROWS,
    episode_kinds=PRODUCTION_EPISODE_KINDS,
    planner_lengths=PRODUCTION_PLANNER_LENGTHS,
    output_rows=PRODUCTION_OUTPUT_ROWS,
    output_bytes=PRODUCTION_OUTPUT_BYTES,
    output_sha256=PRODUCTION_OUTPUT_SHA256,
)


@dataclass(frozen=True)
class _Row:
    line_number: int
    raw: bytes
    raw_sha256: str
    semantic_sha256: str
    single_turn_prompt_sha256: str | None
    category: str | None
    episode_kind: str | None
    planner_length: int | None

    @property
    def is_single_turn(self) -> bool:
        return self.category is not None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": _path_label(path),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_WORKSPACE_ROOT))
    except ValueError:
        return str(resolved)


def _semantic_sha256(conversation: Conversation) -> str:
    # Match the train runners' content-fingerprint contract: normalize schema defaults through
    # Conversation first, then discard split/generator metadata.
    semantic = json.loads(conversation.to_json())
    semantic.pop("meta", None)
    encoded = json.dumps(
        semantic,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(encoded)


def _parse_rows(path: Path, *, expected_rows: int) -> tuple[bytes, list[_Row]]:
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if b"".join(lines) != payload:
        raise RuntimeError(f"{path} line splitting changed source bytes")
    if len(lines) != expected_rows:
        raise ValueError(f"{path} must contain {expected_rows} rows, observed {len(lines)}")
    if not lines or not lines[-1].endswith(b"\n"):
        raise ValueError(f"{path} must end every JSONL row with a newline")

    rows: list[_Row] = []
    for line_number, raw in enumerate(lines, start=1):
        if not raw.endswith(b"\n"):
            raise ValueError(f"{path}:{line_number} is not newline-terminated")
        if not raw.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            decoded = raw.decode("utf-8")
            record = json.loads(decoded)
            if not isinstance(record, dict):
                raise TypeError("Conversation row must be a JSON object")
            conversation = Conversation.from_json(decoded)
        except (
            AttributeError,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(f"{path}:{line_number} is not a valid Conversation row") from error
        if not isinstance(conversation.meta, dict):
            raise ValueError(f"{path}:{line_number} meta must be a JSON object")
        if any(not isinstance(message.content, str) for message in conversation.messages):
            raise ValueError(f"{path}:{line_number} message content must be text")

        single_turn = (
            len(conversation.messages) == 2
            and conversation.messages[0].role == Role.user
            and conversation.messages[1].role == Role.assistant
        )
        category = None
        prompt_sha256 = None
        episode_kind = None
        planner_length = None
        if single_turn:
            raw_category = conversation.meta.get("category")
            if not isinstance(raw_category, str) or not raw_category:
                raise ValueError(
                    f"{path}:{line_number} single-turn row has no non-empty meta.category"
                )
            category = raw_category
            prompt_sha256 = _sha256(conversation.messages[0].content.encode("utf-8"))
        else:
            if len(conversation.messages) <= 2:
                raise ValueError(
                    f"{path}:{line_number} is neither a simple user/assistant row "
                    "nor a multi-turn episode"
                )
            raw_kind = conversation.meta.get("kind")
            if not isinstance(raw_kind, str) or not raw_kind:
                raise ValueError(
                    f"{path}:{line_number} multi-turn row has no non-empty meta.kind"
                )
            episode_kind = raw_kind
            raw_length = conversation.meta.get("plan_len")
            if raw_length is not None:
                if isinstance(raw_length, bool) or not isinstance(raw_length, int):
                    raise ValueError(f"{path}:{line_number} has an invalid meta.plan_len")
                planner_length = raw_length

        rows.append(
            _Row(
                line_number=line_number,
                raw=raw,
                raw_sha256=_sha256(raw),
                semantic_sha256=_semantic_sha256(conversation),
                single_turn_prompt_sha256=prompt_sha256,
                category=category,
                episode_kind=episode_kind,
                planner_length=planner_length,
            )
        )
    return payload, rows


def _minimum_sha_row(rows: list[_Row]) -> _Row:
    if not rows:
        raise ValueError("coverage bucket has no candidate rows")
    return min(rows, key=lambda row: (row.raw_sha256, row.line_number))


def _select_rows(rows: list[_Row], contract: SubsetContract) -> list[_Row]:
    single_turn = [row for row in rows if row.is_single_turn]
    multi_turn = [row for row in rows if not row.is_single_turn]

    categories = sorted({str(row.category) for row in single_turn})
    if len(categories) != contract.single_turn_categories:
        raise ValueError(
            "single-turn category-count mismatch: "
            f"expected {contract.single_turn_categories}, observed {len(categories)}"
        )
    selected: dict[int, _Row] = {}
    for category in categories:
        candidates = [row for row in single_turn if row.category == category]
        row = _minimum_sha_row(candidates)
        selected[row.line_number] = row

    observed_kinds = {str(row.episode_kind) for row in multi_turn}
    if observed_kinds != set(contract.episode_kinds):
        raise ValueError(
            "multi-turn episode-kind coverage mismatch: "
            f"expected {sorted(contract.episode_kinds)}, observed {sorted(observed_kinds)}"
        )
    selected_multi: dict[int, _Row] = {}
    for kind in sorted(contract.episode_kinds):
        row = _minimum_sha_row([row for row in multi_turn if row.episode_kind == kind])
        selected_multi[row.line_number] = row

    observed_lengths = {
        row.planner_length for row in multi_turn if row.planner_length is not None
    }
    if observed_lengths != set(contract.planner_lengths):
        raise ValueError(
            "planner-length coverage mismatch: "
            f"expected {sorted(contract.planner_lengths)}, observed {sorted(observed_lengths)}"
        )
    for planner_length in sorted(contract.planner_lengths):
        row = _minimum_sha_row(
            [row for row in multi_turn if row.planner_length == planner_length]
        )
        selected_multi[row.line_number] = row

    if len(selected_multi) > contract.multi_turn_rows:
        raise ValueError(
            "mandatory multi-turn coverage exceeds configured subset size: "
            f"{len(selected_multi)} > {contract.multi_turn_rows}"
        )
    for row in sorted(multi_turn, key=lambda row: (row.raw_sha256, row.line_number)):
        if len(selected_multi) == contract.multi_turn_rows:
            break
        selected_multi[row.line_number] = row
    if len(selected_multi) != contract.multi_turn_rows:
        raise ValueError(
            f"needed {contract.multi_turn_rows} multi-turn rows, selected {len(selected_multi)}"
        )
    selected.update(selected_multi)

    ordered = [selected[line_number] for line_number in sorted(selected)]
    if len(ordered) != contract.output_rows:
        raise ValueError(
            f"needed {contract.output_rows} total rows, selected {len(ordered)}"
        )
    if sum(row.is_single_turn for row in ordered) != contract.single_turn_categories:
        raise RuntimeError("selected single-turn row count drifted from category coverage")
    if sum(not row.is_single_turn for row in ordered) != contract.multi_turn_rows:
        raise RuntimeError("selected multi-turn row count drifted from contract")
    if [row.line_number for row in ordered] != sorted(row.line_number for row in ordered):
        raise RuntimeError("selected rows are not in original source order")
    return ordered


def _leakage_audit(train_rows: list[_Row], selected_rows: list[_Row]) -> dict[str, Any]:
    train_semantic = {row.semantic_sha256 for row in train_rows}
    eval_semantic = {row.semantic_sha256 for row in selected_rows}
    semantic_overlap = train_semantic & eval_semantic

    train_prompts = {
        row.single_turn_prompt_sha256
        for row in train_rows
        if row.single_turn_prompt_sha256 is not None
    }
    eval_prompts = {
        row.single_turn_prompt_sha256
        for row in selected_rows
        if row.single_turn_prompt_sha256 is not None
    }
    prompt_overlap = train_prompts & eval_prompts
    if semantic_overlap or prompt_overlap:
        raise ValueError(
            "derived evaluation subset overlaps training data: "
            f"canonical_rows={len(semantic_overlap)}, single_turn_prompts={len(prompt_overlap)}"
        )
    return {
        "contract": (
            "sha256(canonical Conversation JSON excluding meta); "
            "sha256(exact single-turn user content)"
        ),
        "canonical_conversation_overlap": 0,
        "single_turn_prompt_overlap": 0,
        "train_rows": len(train_rows),
        "eval_rows": len(selected_rows),
        "eval_single_turn_rows": sum(row.is_single_turn for row in selected_rows),
    }


def _assert_expected_identity(
    *,
    label: str,
    payload: bytes,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> None:
    observed_sha256 = _sha256(payload)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {observed_sha256}"
        )
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise ValueError(
            f"{label} byte-size mismatch: expected {expected_bytes}, got {len(payload)}"
        )


def _assert_existing_exact(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise RuntimeError(f"refusing to overwrite drifted derived artifact: {path}")


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise RuntimeError(f"published artifact failed byte-for-byte verification: {path}")


def derive_agent_eval_pilot_subset(
    source_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    train_source_path: str | Path | None = None,
    contract: SubsetContract = PRODUCTION_CONTRACT,
) -> dict[str, Any]:
    """Derive, verify, and atomically publish one deterministic held-out subset.

    Existing output or manifest files are accepted only when their bytes match the derivation
    exactly. Any source, coverage, leakage, or output-identity drift fails before publication.
    """

    source = Path(source_path)
    output = Path(output_path)
    manifest_file = Path(manifest_path)
    source_manifest = (
        Path(source_manifest_path) if source_manifest_path is not None else None
    )
    train_source = Path(train_source_path) if train_source_path is not None else None

    resolved_targets = {output.resolve(), manifest_file.resolve()}
    if len(resolved_targets) != 2:
        raise ValueError("output_path and manifest_path must be different files")
    protected_sources = {source.resolve()}
    if source_manifest is not None:
        protected_sources.add(source_manifest.resolve())
    if train_source is not None:
        protected_sources.add(train_source.resolve())
    if resolved_targets & protected_sources:
        raise ValueError("derived outputs must not overwrite source artifacts")

    source_payload, rows = _parse_rows(source, expected_rows=contract.source_rows)
    _assert_expected_identity(
        label="frozen evaluation source",
        payload=source_payload,
        expected_sha256=contract.source_sha256,
    )

    source_manifest_identity = None
    if contract.source_manifest_sha256 is not None:
        if source_manifest is None:
            raise ValueError("the subset contract requires source_manifest_path")
        source_manifest_payload = source_manifest.read_bytes()
        _assert_expected_identity(
            label="frozen evaluation source manifest",
            payload=source_manifest_payload,
            expected_sha256=contract.source_manifest_sha256,
        )
        source_manifest_identity = _identity(source_manifest, source_manifest_payload)

    selected_rows = _select_rows(rows, contract)
    output_payload = b"".join(row.raw for row in selected_rows)
    if contract.output_sha256 is not None:
        _assert_expected_identity(
            label="derived evaluation subset",
            payload=output_payload,
            expected_sha256=contract.output_sha256,
            expected_bytes=contract.output_bytes,
        )
    elif contract.output_bytes is not None and len(output_payload) != contract.output_bytes:
        raise ValueError(
            "derived evaluation subset byte-size mismatch: "
            f"expected {contract.output_bytes}, got {len(output_payload)}"
        )

    train_identity = None
    leakage_audit = None
    if contract.train_source_sha256 is not None:
        if train_source is None or contract.train_source_rows is None:
            raise ValueError("the subset contract requires train_source_path and row count")
        train_payload, train_rows = _parse_rows(
            train_source,
            expected_rows=contract.train_source_rows,
        )
        _assert_expected_identity(
            label="frozen training source",
            payload=train_payload,
            expected_sha256=contract.train_source_sha256,
        )
        train_identity = _identity(train_source, train_payload)
        leakage_audit = _leakage_audit(train_rows, selected_rows)

    categories = {
        str(row.category): row.line_number
        for row in selected_rows
        if row.is_single_turn
    }
    multi_rows = [row for row in selected_rows if not row.is_single_turn]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "algorithm": _ALGORITHM,
        "source": {
            **_identity(source, source_payload),
            "rows": len(rows),
            "manifest": source_manifest_identity,
        },
        "train_source": train_identity,
        "contract": {
            "single_turn_categories": contract.single_turn_categories,
            "multi_turn_rows": contract.multi_turn_rows,
            "episode_kinds": sorted(contract.episode_kinds),
            "planner_lengths": sorted(contract.planner_lengths),
            "output_rows": contract.output_rows,
            "raw_line_hash_scope": "exact UTF-8 JSONL line bytes including newline",
            "output_order": "ascending original one-based source line number",
            "serialization": "selected source lines copied byte-for-byte; no JSON rewrite",
        },
        "selection": {
            "single_turn_category_lines": dict(sorted(categories.items())),
            "multi_turn_lines": [row.line_number for row in multi_rows],
            "original_line_numbers": [row.line_number for row in selected_rows],
            "raw_line_sha256": [row.raw_sha256 for row in selected_rows],
            "covered_episode_kinds": sorted(
                {str(row.episode_kind) for row in multi_rows}
            ),
            "covered_planner_lengths": sorted(
                {
                    row.planner_length
                    for row in multi_rows
                    if row.planner_length is not None
                }
            ),
        },
        "leakage_audit": leakage_audit,
        "output": {
            **_identity(output, output_payload),
            "rows": len(selected_rows),
            "single_turn_rows": sum(row.is_single_turn for row in selected_rows),
            "multi_turn_rows": sum(not row.is_single_turn for row in selected_rows),
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    # Check every existing destination before publishing either missing destination.
    _assert_existing_exact(output, output_payload)
    _assert_existing_exact(manifest_file, manifest_payload)
    _publish_atomic(output, output_payload)
    _publish_atomic(manifest_file, manifest_payload)
    return manifest
