from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from localagent.data.agent_eval_subset import (
    PRODUCTION_EPISODE_KINDS,
    PRODUCTION_OUTPUT_BYTES,
    PRODUCTION_OUTPUT_ROWS,
    PRODUCTION_OUTPUT_SHA256,
    PRODUCTION_PLANNER_LENGTHS,
    PRODUCTION_SINGLE_TURN_CATEGORIES,
    SubsetContract,
    derive_agent_eval_pilot_subset,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SOURCE = ROOT / "data" / "synth" / "agent_eval.jsonl"
PRODUCTION_SOURCE_MANIFEST = (
    ROOT / "data" / "synth" / "agent_eval.jsonl.manifest.json"
)
PRODUCTION_TRAIN_SOURCE = ROOT / "data" / "synth" / "agent_sft.jsonl"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(record: dict) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _single_turn(category: str, variant: int) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"{category} request {variant}"},
            {"role": "assistant", "content": f"{category} answer {variant}"},
        ],
        "tools": [],
        "meta": {"category": category, "split": "eval", "variant": variant},
    }


def _multi_turn(kind: str, planner_length: int | None, variant: int) -> dict:
    meta: dict[str, object] = {"kind": kind, "split": "eval", "variant": variant}
    if planner_length is not None:
        meta["plan_len"] = planner_length
    return {
        "messages": [
            {"role": "user", "content": f"episode request {variant}"},
            {"role": "assistant", "content": f"episode progress {variant}"},
            {"role": "user", "content": f"episode follow-up {variant}"},
        ],
        "tools": [],
        "meta": meta,
    }


def _synthetic_source(tmp_path: Path) -> tuple[Path, list[bytes], SubsetContract]:
    records = [
        *(
            _single_turn(category, variant)
            for category in ("alpha", "beta", "gamma")
            for variant in range(2)
        ),
        *(
            _multi_turn(
                ("coding", "computer", "planner", "productivity")[variant % 4],
                variant % 5 if variant < 10 else None,
                variant,
            )
            for variant in range(12)
        ),
    ]
    # The order is intentionally unrelated to either category or content hash.
    source_order = [7, 1, 12, 4, 17, 0, 9, 3, 15, 6, 11, 2, 16, 5, 8, 14, 10, 13]
    records = [records[index] for index in source_order]
    lines = [_jsonl(record) for record in records]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(lines))
    contract = SubsetContract(
        source_sha256=_sha256(source.read_bytes()),
        source_rows=len(lines),
        single_turn_categories=3,
        multi_turn_rows=10,
        episode_kinds=frozenset({"coding", "computer", "planner", "productivity"}),
        planner_lengths=frozenset(range(5)),
        output_rows=13,
    )
    return source, lines, contract


def _expected_synthetic_selection(lines: list[bytes], multi_turn_rows: int) -> list[int]:
    decoded = [json.loads(line) for line in lines]
    singles = [
        (index, line, record)
        for index, (line, record) in enumerate(zip(lines, decoded, strict=True), start=1)
        if len(record["messages"]) == 2
    ]
    multis = [
        (index, line, record)
        for index, (line, record) in enumerate(zip(lines, decoded, strict=True), start=1)
        if len(record["messages"]) > 2
    ]
    selected = {
        min(
            (row for row in singles if row[2]["meta"]["category"] == category),
            key=lambda row: (_sha256(row[1]), row[0]),
        )[0]
        for category in ("alpha", "beta", "gamma")
    }
    mandatory_multi = {
        min(
            (row for row in multis if row[2]["meta"]["kind"] == kind),
            key=lambda row: (_sha256(row[1]), row[0]),
        )[0]
        for kind in ("coding", "computer", "planner", "productivity")
    }
    mandatory_multi.update(
        min(
            (
                row
                for row in multis
                if row[2]["meta"].get("plan_len") == planner_length
            ),
            key=lambda row: (_sha256(row[1]), row[0]),
        )[0]
        for planner_length in range(5)
    )
    for index, line, _ in sorted(multis, key=lambda row: (_sha256(row[1]), row[0])):
        if len(mandatory_multi) == multi_turn_rows:
            break
        mandatory_multi.add(index)
    selected.update(mandatory_multi)
    return sorted(selected)


def test_subset_selection_is_deterministic_byte_preserving_and_idempotent(
    tmp_path: Path,
) -> None:
    source, lines, contract = _synthetic_source(tmp_path)
    expected_lines = _expected_synthetic_selection(lines, contract.multi_turn_rows)
    expected_payload = b"".join(lines[index - 1] for index in expected_lines)
    output = tmp_path / "pilot.jsonl"
    manifest_path = tmp_path / "pilot.manifest.json"

    manifest = derive_agent_eval_pilot_subset(
        source,
        output,
        manifest_path,
        contract=contract,
    )

    assert output.read_bytes() == expected_payload
    assert manifest["selection"]["original_line_numbers"] == expected_lines
    assert manifest["output"]["rows"] == 13
    assert manifest["output"]["single_turn_rows"] == 3
    assert manifest["output"]["multi_turn_rows"] == 10
    assert manifest["selection"]["covered_episode_kinds"] == [
        "coding",
        "computer",
        "planner",
        "productivity",
    ]
    assert manifest["selection"]["covered_planner_lengths"] == list(range(5))

    output_mtime = output.stat().st_mtime_ns
    manifest_payload = manifest_path.read_bytes()
    second = derive_agent_eval_pilot_subset(
        source,
        output,
        manifest_path,
        contract=contract,
    )
    assert second == manifest
    assert output.stat().st_mtime_ns == output_mtime
    assert manifest_path.read_bytes() == manifest_payload


def test_subset_derivation_fails_closed_on_contract_or_destination_drift(
    tmp_path: Path,
) -> None:
    source, _, contract = _synthetic_source(tmp_path)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        derive_agent_eval_pilot_subset(
            source,
            tmp_path / "bad-sha.jsonl",
            tmp_path / "bad-sha.manifest.json",
            contract=replace(contract, source_sha256="0" * 64),
        )
    assert not (tmp_path / "bad-sha.jsonl").exists()

    with pytest.raises(ValueError, match="must contain"):
        derive_agent_eval_pilot_subset(
            source,
            tmp_path / "bad-count.jsonl",
            tmp_path / "bad-count.manifest.json",
            contract=replace(contract, source_rows=contract.source_rows + 1),
        )
    assert not (tmp_path / "bad-count.jsonl").exists()

    with pytest.raises(ValueError, match="category-count mismatch"):
        derive_agent_eval_pilot_subset(
            source,
            tmp_path / "bad-coverage.jsonl",
            tmp_path / "bad-coverage.manifest.json",
            contract=replace(contract, single_turn_categories=4),
        )
    assert not (tmp_path / "bad-coverage.jsonl").exists()

    drifted_output = tmp_path / "drifted.jsonl"
    drifted_output.write_bytes(b"do not overwrite me\n")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        derive_agent_eval_pilot_subset(
            source,
            drifted_output,
            tmp_path / "drifted.manifest.json",
            contract=contract,
        )
    assert drifted_output.read_bytes() == b"do not overwrite me\n"
    assert not (tmp_path / "drifted.manifest.json").exists()


def test_subset_derivation_rejects_single_turn_prompt_leakage(tmp_path: Path) -> None:
    source, lines, contract = _synthetic_source(tmp_path)
    selected_lines = _expected_synthetic_selection(lines, contract.multi_turn_rows)
    selected_single = next(
        json.loads(lines[index - 1])
        for index in selected_lines
        if len(json.loads(lines[index - 1])["messages"]) == 2
    )
    selected_single["messages"][1]["content"] = "different training answer"
    selected_single["meta"] = {"category": "training-only", "split": "train"}
    train_source = tmp_path / "train.jsonl"
    train_source.write_bytes(_jsonl(selected_single))
    audited_contract = replace(
        contract,
        train_source_sha256=_sha256(train_source.read_bytes()),
        train_source_rows=1,
    )

    with pytest.raises(ValueError, match="overlaps training data"):
        derive_agent_eval_pilot_subset(
            source,
            tmp_path / "leaked.jsonl",
            tmp_path / "leaked.manifest.json",
            train_source_path=train_source,
            contract=audited_contract,
        )
    assert not (tmp_path / "leaked.jsonl").exists()


def test_subset_derivation_rejects_normalized_conversation_leakage(
    tmp_path: Path,
) -> None:
    source, lines, contract = _synthetic_source(tmp_path)
    selected_lines = _expected_synthetic_selection(lines, contract.multi_turn_rows)
    selected_multi = next(
        json.loads(lines[index - 1])
        for index in selected_lines
        if len(json.loads(lines[index - 1])["messages"]) > 2
    )
    # Explicit schema defaults must normalize to the same semantic row as their omitted form.
    for message in selected_multi["messages"]:
        message["tool_calls"] = []
        message["tool_response"] = None
    selected_multi["meta"] = {"kind": "training-episode", "split": "train"}
    train_source = tmp_path / "train.jsonl"
    train_source.write_bytes(_jsonl(selected_multi))
    audited_contract = replace(
        contract,
        train_source_sha256=_sha256(train_source.read_bytes()),
        train_source_rows=1,
    )

    with pytest.raises(ValueError, match="overlaps training data"):
        derive_agent_eval_pilot_subset(
            source,
            tmp_path / "semantic-leak.jsonl",
            tmp_path / "semantic-leak.manifest.json",
            train_source_path=train_source,
            contract=audited_contract,
        )
    assert not (tmp_path / "semantic-leak.jsonl").exists()


@pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            PRODUCTION_SOURCE,
            PRODUCTION_SOURCE_MANIFEST,
            PRODUCTION_TRAIN_SOURCE,
        )
    ),
    reason="frozen generated artifacts are not present in this checkout",
)
def test_production_subset_matches_pinned_identity_and_coverage(tmp_path: Path) -> None:
    output = tmp_path / "agent_eval_pilot65.jsonl"
    manifest_path = tmp_path / "agent_eval_pilot65.jsonl.manifest.json"

    manifest = derive_agent_eval_pilot_subset(
        PRODUCTION_SOURCE,
        output,
        manifest_path,
        source_manifest_path=PRODUCTION_SOURCE_MANIFEST,
        train_source_path=PRODUCTION_TRAIN_SOURCE,
    )

    payload = output.read_bytes()
    assert len(payload) == PRODUCTION_OUTPUT_BYTES
    assert _sha256(payload) == PRODUCTION_OUTPUT_SHA256
    assert len(payload.splitlines()) == PRODUCTION_OUTPUT_ROWS
    assert Path(manifest["output"]["path"]).name == output.name
    assert manifest["output"]["bytes"] == PRODUCTION_OUTPUT_BYTES
    assert manifest["output"]["sha256"] == PRODUCTION_OUTPUT_SHA256
    assert manifest["output"]["rows"] == PRODUCTION_OUTPUT_ROWS
    assert (
        manifest["output"]["single_turn_rows"]
        == PRODUCTION_SINGLE_TURN_CATEGORIES
    )
    assert manifest["output"]["multi_turn_rows"] == 12
    assert manifest["selection"]["covered_episode_kinds"] == sorted(
        PRODUCTION_EPISODE_KINDS
    )
    assert manifest["selection"]["covered_planner_lengths"] == sorted(
        PRODUCTION_PLANNER_LENGTHS
    )
    assert manifest["leakage_audit"]["canonical_conversation_overlap"] == 0
    assert manifest["leakage_audit"]["single_turn_prompt_overlap"] == 0

    source_lines = PRODUCTION_SOURCE.read_bytes().splitlines(keepends=True)
    selected_line_numbers = manifest["selection"]["original_line_numbers"]
    assert selected_line_numbers == sorted(selected_line_numbers)
    assert payload == b"".join(
        source_lines[line_number - 1] for line_number in selected_line_numbers
    )
