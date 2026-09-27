from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from openlocalagent.data.conversation_artifact import (
    CONVERSATION_FORMAT,
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    canonical_json_bytes,
    load_verified_conversation_artifact,
    self_hashed_manifest,
)
from openlocalagent.data.synth.format_bootstrap import (
    FORMAT_BOOTSTRAP_ALGORITHM,
    FORMAT_BOOTSTRAP_CONFIG_KIND,
    FORMAT_BOOTSTRAP_PHASES,
    FORMAT_BOOTSTRAP_SCHEMA_VERSION,
    build_format_bootstrap,
    classify_format_bootstrap_phase,
    select_format_bootstrap,
)
from openlocalagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from openlocalagent.model.tokenizer import ByteTokenizer, train_bpe


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="ping",
            description="Ping the service.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="lookup",
            description="Look up one query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            name="route",
            description="Route one item.",
            parameters={
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "item": {"type": "string"},
                },
                "required": ["destination", "item"],
                "additionalProperties": False,
            },
        ),
    ]


def _conversation(
    prompt: str,
    *,
    calls: list[ToolCall] | None = None,
    text: str = "",
    split: str = "train",
    category: str = "test",
) -> Conversation:
    return Conversation(
        messages=[
            Message(role=Role.user, content=prompt),
            Message(role=Role.assistant, content=text, tool_calls=calls or []),
        ],
        tools=_tools(),
        meta={
            "category": category,
            "environment_executed": False,
            "rule_verified": True,
            "split": split,
        },
    )


def _curriculum_rows() -> list[Conversation]:
    return [
        *[
            _conversation(
                "p" * length,
                calls=[ToolCall(name="ping", arguments={})],
                category="ping",
            )
            for length in (3, 8, 13, 21)
        ],
        *[
            _conversation(
                "look " + "q" * length,
                calls=[ToolCall(name="lookup", arguments={"query": "q" * length})],
                category="lookup",
            )
            for length in (2, 5, 9, 14)
        ],
        *[
            _conversation(
                "route " + str(index),
                calls=[
                    ToolCall(
                        name="route",
                        arguments={"destination": f"d{index}", "item": f"i{index}"},
                    )
                ],
                category="route",
            )
            for index in range(4)
        ],
        *[
            _conversation(
                "parallel " + str(index),
                calls=[
                    ToolCall(name="ping", arguments={}),
                    ToolCall(name="lookup", arguments={"query": f"q{index}"}),
                ],
                category="parallel",
            )
            for index in range(4)
        ],
        *[
            _conversation(
                "thanks " + str(index),
                text="OK." if index % 2 else "You're welcome.",
                category="no_tool" if index % 2 else "text",
            )
            for index in range(4)
        ],
    ]


def _publish_verified_artifact(
    directory: Path,
    *,
    name: str,
    conversations: list[Conversation],
    split: str,
) -> tuple[Path, Path, Path, dict[str, object]]:
    config_path = directory / f"{name}.yaml"
    config_path.write_text(f"name: {name}\nsplit: {split}\n", encoding="utf-8")
    config_identity = FileIdentity.from_bytes(config_path.read_bytes())
    data_path = directory / f"{name}.jsonl"
    payload = b"".join((conversation.to_json() + "\n").encode() for conversation in conversations)
    data_path.write_bytes(payload)
    data_identity = FileIdentity.from_bytes(payload)
    manifest = {
        "argument_schema_coverage": {},
        "argument_value_counts": {},
        "behavior_counts": {
            "enum_arguments": 0,
            "explicit_restraint": 0,
            "integer_arguments": 0,
            "multiple_arguments": 0,
            "parallel_calls": 0,
            "tool_response_grounded_followups": 0,
            "verified_error_recovery": 0,
        },
        "behavior_definitions": {},
        "complexity_contract": {},
        "conversation_serialization": CONVERSATION_SERIALIZATION,
        "coverage_contract": {},
        "environment_executed": False,
        "exact_prompt_holdouts": {},
        "format": CONVERSATION_FORMAT,
        "generator_config": config_identity.as_dict(),
        "irrelevance": 0,
        "kind": MANIFEST_KIND,
        "level": 1,
        "model_verified": False,
        "multi_turn": 0,
        "output_bytes": data_identity.bytes,
        "output_sha256": data_identity.sha256,
        "plan_length_counts": {},
        "rows": len(conversations),
        "rule_verification_scope": ["test_fixture"],
        "rule_verified": True,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": 0,
        "single_turn": len(conversations),
        "split": split,
        "split_contract": {},
        "structural_counts": {},
        "verification_claim": "test_fixture",
    }
    sealed, manifest_payload = self_hashed_manifest(manifest)
    manifest_path = directory / f"{name}.jsonl.manifest.v1.json"
    manifest_path.write_bytes(manifest_payload)
    manifest_identity = FileIdentity.from_bytes(manifest_payload)
    identity = {
        "generator_config": config_identity.as_dict(),
        "jsonl": data_identity.as_dict(),
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sidecar": {
            **manifest_identity.as_dict(),
            "manifest_self_sha256": sealed["manifest_self_sha256"],
        },
        "split": split,
    }
    return data_path, manifest_path, config_path, identity


def test_phase_classifier_and_balanced_shortest_selection_are_deterministic():
    rows = _curriculum_rows()
    expected_phases = [
        *(["format_core"] * 8),
        *(["multi_argument"] * 4),
        *(["parallel"] * 4),
        *(["text"] * 4),
    ]

    assert [classify_format_bootstrap_phase(row) for row in rows] == expected_phases
    multi_turn = Conversation(
        messages=[
            Message(role=Role.user, content="Do two steps."),
            Message(
                role=Role.assistant,
                tool_calls=[ToolCall(name="ping", arguments={})],
            ),
            Message(role=Role.tool, tool_response="done"),
            Message(role=Role.assistant, content="Done."),
        ],
        tools=_tools(),
    )
    assert classify_format_bootstrap_phase(multi_turn) is None

    phase_rows = {
        "format_core": 4,
        "multi_argument": 2,
        "parallel": 2,
        "text": 2,
    }
    first = select_format_bootstrap(rows, ByteTokenizer(), phase_rows)
    replay = select_format_bootstrap(rows, ByteTokenizer(), phase_rows)

    assert first == replay
    assert [record.phase for record in first.records] == [
        *(["format_core"] * 4),
        *(["multi_argument"] * 2),
        *(["parallel"] * 2),
        *(["text"] * 2),
    ]
    assert first.audit["phase_order"] == list(FORMAT_BOOTSTRAP_PHASES)
    assert first.audit["consumption_contract"] == {
        "configured_data_sampling": "absent",
        "resolved_sft_mode": "source_order_wrapping_v1",
        "shuffle": False,
        "warning": (
            "quota_stratified_no_replacement_v1 is intentionally incompatible because it "
            "reorders assistant decisions across curriculum phases"
        ),
    }
    assert first.audit["output"]["assistant_decisions"] == 10
    assert first.audit["output"]["unique_semantic_rows"] == 10
    assert len(first.audit["audit_sha256"]) == 64
    assert set(record.bucket for record in first.records[:4]) == {"lookup", "ping"}
    assert first.audit["phases"]["format_core"]["tokens"]["selected"][
        "target_tokens_including_eos"
    ]["count"] == 4


def test_selector_rejects_missing_phase_capacity_and_invalid_phase_mapping():
    rows = _curriculum_rows()
    with pytest.raises(ValueError, match="only 4 unique eligible rows"):
        select_format_bootstrap(
            rows,
            ByteTokenizer(),
            {
                "format_core": 1,
                "multi_argument": 5,
                "parallel": 1,
                "text": 1,
            },
        )
    with pytest.raises(ValueError, match="keys mismatch"):
        select_format_bootstrap(rows, ByteTokenizer(), {"format_core": 1})


def test_builder_seals_replayable_subset_and_binds_every_input(tmp_path: Path):
    train_rows = _curriculum_rows()
    eval_rows = [
        _conversation(
            "heldout lookup",
            calls=[ToolCall(name="lookup", arguments={"query": "heldout"})],
            split="eval",
            category="lookup",
        )
    ]
    train_path, train_manifest, train_config, train_identity = _publish_verified_artifact(
        tmp_path,
        name="train",
        conversations=train_rows,
        split="train",
    )
    eval_path, eval_manifest, eval_config, eval_identity = _publish_verified_artifact(
        tmp_path,
        name="eval",
        conversations=eval_rows,
        split="eval",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    train_bpe(
        [row.to_json() for row in [*train_rows, *eval_rows]],
        tokenizer_path,
        vocab_size=320,
    )
    tokenizer_identity = FileIdentity.from_bytes(tokenizer_path.read_bytes()).as_dict()
    output_path = tmp_path / "bootstrap.jsonl"
    output_manifest = tmp_path / "bootstrap.jsonl.manifest.v1.json"
    receipt_path = tmp_path / "bootstrap.receipt.json"
    build_config = {
        "algorithm": FORMAT_BOOTSTRAP_ALGORITHM,
        "evaluation_holdout": {
            "expected_identity": eval_identity,
            "generator_config": str(eval_config),
            "manifest": str(eval_manifest),
            "path": str(eval_path),
        },
        "kind": FORMAT_BOOTSTRAP_CONFIG_KIND,
        "manifest": str(output_manifest),
        "out": str(output_path),
        "phases": {
            "format_core": 2,
            "multi_argument": 1,
            "parallel": 1,
            "text": 1,
        },
        "prompt_contract": "openai_full_catalog_v1",
        "receipt": str(receipt_path),
        "schema_version": FORMAT_BOOTSTRAP_SCHEMA_VERSION,
        "source": {
            "expected_identity": train_identity,
            "generator_config": str(train_config),
            "manifest": str(train_manifest),
            "path": str(train_path),
        },
        "tokenizer": {
            "identity": tokenizer_identity,
            "kind": "bpe",
            "path": str(tokenizer_path),
        },
    }
    build_config_path = tmp_path / "bootstrap.yaml"
    build_config_path.write_text(
        yaml.safe_dump(build_config, sort_keys=True),
        encoding="utf-8",
    )

    first = build_format_bootstrap(build_config_path)
    first_bytes = (
        output_path.read_bytes(),
        output_manifest.read_bytes(),
        receipt_path.read_bytes(),
    )
    replay = build_format_bootstrap(build_config_path)

    assert replay == first
    assert (
        output_path.read_bytes(),
        output_manifest.read_bytes(),
        receipt_path.read_bytes(),
    ) == first_bytes
    assert first["output"]["rows"] == 5
    assert first["overlap_audit"]["semantic_overlap"] == 0
    assert first["overlap_audit"]["rendered_prompt_overlap"] == 0
    assert first["selection"]["output"]["assistant_decisions"] == 5
    assert first["corpus_analysis"]["source"]["assistant_decisions"] == len(train_rows)
    assert first["corpus_analysis"]["evaluation_holdout"]["assistant_decisions"] == 1
    assert first["corpus_analysis"]["source"]["tokens"]["totals"]["loss_tokens"] > 0
    unsigned_receipt = dict(first)
    receipt_self_sha256 = unsigned_receipt.pop("receipt_self_sha256")
    assert receipt_self_sha256 == hashlib.sha256(
        canonical_json_bytes(unsigned_receipt)
    ).hexdigest()

    rebound = load_verified_conversation_artifact(
        output_path,
        config_path=build_config_path,
        expected_split="train",
        manifest_path=output_manifest,
        expected_rule_verified=True,
        environment_policy="forbid",
    )
    assert len(rebound.conversations) == 5
    source_semantics = {
        json.dumps(
            {key: value for key, value in asdict(row).items() if key != "meta"},
            default=lambda value: value.value,
            sort_keys=True,
        )
        for row in train_rows
    }
    assert all(
        json.dumps(
            {key: value for key, value in asdict(row).items() if key != "meta"},
            default=lambda value: value.value,
            sort_keys=True,
        )
        in source_semantics
        for row in rebound.conversations
    )

    unsafe = json.loads(json.dumps(build_config))
    unsafe["out"] = str(train_path)
    build_config_path.write_text(
        yaml.safe_dump(unsafe, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overwrite a bound input"):
        build_format_bootstrap(build_config_path)

    tampered = json.loads(json.dumps(build_config))
    tampered["tokenizer"]["identity"]["bytes"] += 1
    build_config_path.write_text(
        yaml.safe_dump(tampered, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer identity mismatch"):
        build_format_bootstrap(build_config_path)
