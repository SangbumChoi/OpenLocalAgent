from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

import localagent.eval.confirmatory_eval_split as confirmatory
from localagent.data.conversation_artifact import (
    CONVERSATION_FORMAT,
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    assert_no_conversation_overlap,
    canonical_json_bytes,
    conversation_semantic_sha256,
    load_verified_conversation_artifact,
    self_hashed_manifest,
)
from localagent.data.schema import Conversation, Message, Role
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.model.tokenizer import train_bpe

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = ROOT / "configs/eval/paper-confirmatory-eval-split-v2.yaml"
PRODUCTION_OUTPUT = ROOT / "data/synth/agent_eval_confirmatory_v2.jsonl"
PRODUCTION_MANIFEST = ROOT / "data/synth/agent_eval_confirmatory_v2.jsonl.manifest.v1.json"
PRODUCTION_PROVENANCE = ROOT / "data/provenance/paper/agent-eval-confirmatory-v2.json"
PRODUCTION_SOURCE_CONFIG = ROOT / "configs/data/agent_synth_eval.yaml"
PRODUCTION_SOURCE_OUTPUT = ROOT / "data/synth/agent_eval.jsonl"
PRODUCTION_SOURCE_MANIFEST = ROOT / "data/synth/agent_eval.jsonl.manifest.v1.json"
PRODUCTION_ARTIFACTS = (
    PRODUCTION_CONFIG,
    PRODUCTION_OUTPUT,
    PRODUCTION_MANIFEST,
    PRODUCTION_PROVENANCE,
)
PRODUCTION_SOURCE_ARTIFACTS = (
    PRODUCTION_SOURCE_CONFIG,
    PRODUCTION_SOURCE_OUTPUT,
    PRODUCTION_SOURCE_MANIFEST,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _conversation(index: int, *, split: str, candidate_metric: float | None = None):
    meta = {
        "category": "text",
        "environment_executed": False,
        "rule_verified": True,
        "split": split,
    }
    if candidate_metric is not None:
        meta["candidate_metric"] = candidate_metric
    return Conversation(
        messages=[
            Message(role=Role.user, content=f"fixture prompt {split} {index}"),
            Message(role=Role.assistant, content=f"fixture answer {index}"),
        ],
        tools=[],
        meta=meta,
    )


def _write_verified_artifact(
    root: Path,
    *,
    name: str,
    split: str,
    conversations: list[Conversation],
) -> dict:
    data_path = root / f"{name}.jsonl"
    config_path = root / f"{name}.yaml"
    manifest_path = root / f"{name}.jsonl.manifest.v1.json"
    config_path.write_text(f"kind: fixture\nname: {name}\n", encoding="utf-8")
    output = b"".join(
        (conversation.to_json() + "\n").encode("utf-8") for conversation in conversations
    )
    data_path.write_bytes(output)
    config_identity = FileIdentity.from_bytes(config_path.read_bytes())
    output_identity = FileIdentity.from_bytes(output)
    manifest_core = {
        "argument_schema_coverage": {},
        "argument_value_counts": {},
        "behavior_counts": {},
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
        "output_bytes": output_identity.bytes,
        "output_sha256": output_identity.sha256,
        "plan_length_counts": {},
        "rows": len(conversations),
        "rule_verification_scope": ["fixture"],
        "rule_verified": True,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": 1,
        "single_turn": len(conversations),
        "split": split,
        "split_contract": {},
        "structural_counts": {
            "assistant_tool_calls": 0,
            "irrelevance_conversations": 0,
            "multi_turn_conversations": 0,
            "parallel_call_conversations": 0,
            "single_call_conversations": 0,
            "text_conversations": len(conversations),
        },
        "verification_claim": "fixture",
    }
    manifest, manifest_payload = self_hashed_manifest(manifest_core)
    manifest_path.write_bytes(manifest_payload)
    return {
        "path": str(data_path),
        "generator_config": str(config_path),
        "manifest": str(manifest_path),
        "expected_identity": {
            "kind": MANIFEST_KIND,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "split": split,
            "jsonl": output_identity.as_dict(),
            "sidecar": {
                **FileIdentity.from_bytes(manifest_payload).as_dict(),
                "manifest_self_sha256": manifest["manifest_self_sha256"],
            },
            "generator_config": config_identity.as_dict(),
        },
    }


def _write_fixture_config(tmp_path: Path) -> Path:
    source_rows = [_conversation(index, split="eval") for index in range(8)]
    train_rows = [_conversation(index + 100, split="train") for index in range(4)]
    source = _write_verified_artifact(
        tmp_path,
        name="eval",
        split="eval",
        conversations=source_rows,
    )
    train = _write_verified_artifact(
        tmp_path,
        name="train",
        split="train",
        conversations=train_rows,
    )
    selection = confirmatory.derive_confirmatory_eval_selection(
        source_rows,
        primary_max_rows=2,
        confirmatory_max_rows=2,
    )
    tokenizer_path = tmp_path / "fixture-tokenizer.json"
    tokenizer = train_bpe(
        (
            message.content
            for conversation in (*source_rows, *train_rows)
            for message in conversation.messages
        ),
        tokenizer_path,
        vocab_size=384,
        min_frequency=1,
    )
    accounting = confirmatory._token_accounting(  # noqa: SLF001
        selection.conversations,
        tokenizer,
        max_seq_len=4096,
    )
    output = b"".join(
        (conversation.to_json() + "\n").encode("utf-8") for conversation in selection.conversations
    )
    output_identity = FileIdentity.from_bytes(output)
    config = {
        "kind": confirmatory.CONFIG_KIND,
        "schema_version": confirmatory.SCHEMA_VERSION,
        "algorithm": confirmatory.ALGORITHM,
        "prompt_contract": "openai_full_catalog_v1",
        "primary_source": source,
        "train_sources": [{"name": "fixture_train", "artifact": train}],
        "tokenizer": {
            "kind": "bpe",
            "path": str(tokenizer_path),
            "expected_identity": FileIdentity.from_bytes(tokenizer_path.read_bytes()).as_dict(),
        },
        "max_rows": {"primary": 2, "confirmatory": 2},
        "token_accounting": {"max_seq_len": 4096},
        "rejected_preflight_reference": {
            "status": "rejected",
            "reason": "production semantic-only preflight retained rendered prefixes",
            "algorithm": confirmatory.REJECTED_PREFLIGHT_ALGORITHM,
            "confirmatory_assistant_decisions": (
                confirmatory.REJECTED_PREFLIGHT_CONFIRMATORY_ASSISTANT_DECISIONS
            ),
            "confirmatory_selected_semantic_set_sha256": (
                confirmatory.REJECTED_PREFLIGHT_CONFIRMATORY_SEMANTIC_SET_SHA256
            ),
            "filtered_selection_audit_sha256": (
                confirmatory.REJECTED_PREFLIGHT_FILTERED_AUDIT_SHA256
            ),
            "original_source_row_numbers_sha256": (
                confirmatory.REJECTED_PREFLIGHT_ORIGINAL_ROWS_SHA256
            ),
            "reference_contract_sha256": (
                confirmatory.REJECTED_PREFLIGHT_REFERENCE_CONTRACT_SHA256
            ),
            "rendered_prompt_overlap": (confirmatory.REJECTED_PREFLIGHT_RENDERED_PROMPT_OVERLAP),
        },
        "expected": {
            "source_rows": len(source_rows),
            "source_semantic_set_sha256": (selection.primary.audit.source_semantic_set_sha256),
            "source_rendered_prompt_set_sha256": (selection.source_rendered_prompt_set_sha256),
            "primary_selected_rows": 2,
            "primary_selected_assistant_decisions": (
                selection.primary.audit.selected_assistant_decisions
            ),
            "primary_selected_semantic_set_sha256": (
                selection.primary.audit.selected_semantic_set_sha256
            ),
            "primary_rendered_prompt_set_sha256": (selection.primary_rendered_prompt_set_sha256),
            "primary_selection_audit_sha256": selection.primary.audit.as_dict()["audit_sha256"],
            "semantic_excluded_rows": selection.semantic_excluded_rows,
            "rendered_prompt_excluded_rows": (selection.rendered_prompt_excluded_rows),
            "remaining_rows": selection.remaining_rows,
            "confirmatory_rows": 2,
            "confirmatory_assistant_decisions": (
                selection.filtered.audit.selected_assistant_decisions
            ),
            "confirmatory_selected_semantic_set_sha256": (
                selection.filtered.audit.selected_semantic_set_sha256
            ),
            "confirmatory_rendered_prompt_set_sha256": (
                selection.confirmatory_rendered_prompt_set_sha256
            ),
            "filtered_selection_audit_sha256": selection.filtered.audit.as_dict()["audit_sha256"],
            "original_source_row_numbers_sha256": (selection.original_source_row_numbers_sha256),
            "reference_contract_sha256": selection.reference_contract_sha256,
            "assistant_loss_tokens": accounting["assistant_loss_tokens"],
            "confirmatory_max_post_shift_input_tokens": accounting["max_post_shift_input_tokens"],
            "output_bytes": output_identity.bytes,
            "output_sha256": output_identity.sha256,
        },
        "out": str(tmp_path / "confirm.jsonl"),
        "manifest": str(tmp_path / "confirm.jsonl.manifest.v1.json"),
        "provenance": str(tmp_path / "confirm.provenance.json"),
    }
    config_path = tmp_path / "confirm.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.skipif(
    not all(path.is_file() for path in PRODUCTION_ARTIFACTS),
    reason="requires the exact local sealed confirmatory config, JSONL, manifest, and provenance",
)
def test_production_artifacts_pin_exact_counts_hashes_and_disjointness() -> None:
    config = confirmatory.load_confirmatory_eval_split_config(PRODUCTION_CONFIG)
    manifest_payload = PRODUCTION_MANIFEST.read_bytes()
    provenance_payload = PRODUCTION_PROVENANCE.read_bytes()
    assert len(manifest_payload) == confirmatory.PRODUCTION_MANIFEST_BYTES
    assert _sha256(manifest_payload) == confirmatory.PRODUCTION_MANIFEST_SHA256
    assert len(provenance_payload) == confirmatory.PRODUCTION_PROVENANCE_BYTES
    assert _sha256(provenance_payload) == confirmatory.PRODUCTION_PROVENANCE_SHA256
    receipt = confirmatory.load_confirmatory_eval_receipt(PRODUCTION_PROVENANCE)
    assert receipt["receipt_self_sha256"] == (confirmatory.PRODUCTION_PROVENANCE_SELF_SHA256)
    output = PRODUCTION_OUTPUT.read_bytes()
    assert len(output) == confirmatory.PRODUCTION_OUTPUT_BYTES
    assert _sha256(output) == confirmatory.PRODUCTION_OUTPUT_SHA256
    assert receipt["output"]["jsonl"] == {
        "bytes": confirmatory.PRODUCTION_OUTPUT_BYTES,
        "path": config["out"],
        "sha256": confirmatory.PRODUCTION_OUTPUT_SHA256,
    }
    assert receipt["output"]["rows"] == confirmatory.PRODUCTION_CONFIRMATORY_ROWS
    assert (
        receipt["filtered_selection"]["selected"]["assistant_decisions"]
        == confirmatory.PRODUCTION_CONFIRMATORY_ASSISTANT_DECISIONS
    )
    assert (
        receipt["filtered_selection"]["selected"]["semantic_set_sha256"]
        == confirmatory.PRODUCTION_CONFIRMATORY_SEMANTIC_SET_SHA256
    )
    assert (
        receipt["filtered_selection"]["audit_sha256"]
        == confirmatory.PRODUCTION_FILTERED_AUDIT_SHA256
    )
    assert (
        receipt["original_source_mapping"]["source_row_numbers_sha256"]
        == confirmatory.PRODUCTION_ORIGINAL_ROWS_SHA256
    )
    assert receipt["reference_contract_sha256"] == confirmatory.PRODUCTION_REFERENCE_CONTRACT_SHA256
    assert (
        receipt["token_accounting"]["assistant_loss_tokens"]
        == confirmatory.PRODUCTION_ASSISTANT_LOSS_TOKENS
    )
    assert (
        receipt["token_accounting"]["max_post_shift_input_tokens"]
        == confirmatory.PRODUCTION_MAX_POST_SHIFT_INPUT_TOKENS
    )
    development = receipt["overlap_evidence"]["development"]
    assert development["semantic_overlap"] == 0
    assert development["rendered_prompt_overlap"] == 0
    assert len(receipt["overlap_evidence"]["train_sources"]) == len(config["train_sources"])
    for record in receipt["overlap_evidence"]["train_sources"]:
        assert record["audit"]["semantic_overlap"] == 0
        assert record["audit"]["rendered_prompt_overlap"] == 0

    rebound = load_verified_conversation_artifact(
        PRODUCTION_OUTPUT,
        config_path=PRODUCTION_CONFIG,
        expected_split="eval",
        manifest_path=PRODUCTION_MANIFEST,
        expected_rule_verified=True,
        environment_policy="forbid",
    )
    assert len(rebound.conversations) == confirmatory.PRODUCTION_CONFIRMATORY_ROWS
    manifest_contract = rebound.manifest["coverage_contract"]["confirmatory_eval_split"]
    assert rebound.manifest["manifest_self_sha256"] == (
        confirmatory.PRODUCTION_MANIFEST_SELF_SHA256
    )
    assert (
        manifest_contract["reference_contract_sha256"]
        == confirmatory.PRODUCTION_REFERENCE_CONTRACT_SHA256
    )
    assert manifest_contract["token_accounting"] == receipt["token_accounting"]


def test_builder_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path)
    first = confirmatory.build_confirmatory_eval_split(config_path)
    paths = (
        tmp_path / "confirm.jsonl",
        tmp_path / "confirm.jsonl.manifest.v1.json",
        tmp_path / "confirm.provenance.json",
    )
    first_payloads = tuple(path.read_bytes() for path in paths)
    second = confirmatory.build_confirmatory_eval_split(config_path)
    assert second == first
    assert tuple(path.read_bytes() for path in paths) == first_payloads
    confirmatory.assert_confirmatory_eval_receipt(second)


@pytest.mark.skipif(
    not all(path.is_file() for path in PRODUCTION_SOURCE_ARTIFACTS),
    reason="requires the exact local sealed agent-eval JSONL, manifest, and generator config",
)
def test_semantic_only_preflight_collides_but_v2_is_prefix_disjoint() -> None:
    source = load_verified_conversation_artifact(
        PRODUCTION_SOURCE_OUTPUT,
        config_path=PRODUCTION_SOURCE_CONFIG,
        expected_split="eval",
        manifest_path=PRODUCTION_SOURCE_MANIFEST,
        expected_rule_verified=True,
        environment_policy="forbid",
    )
    primary = select_stratified_eval_subset(source.conversations, max_rows=512)
    primary_semantic = {
        conversation_semantic_sha256(conversation) for conversation in primary.conversations
    }
    semantic_only_remaining = tuple(
        conversation
        for conversation in source.conversations
        if conversation_semantic_sha256(conversation) not in primary_semantic
    )
    rejected = select_stratified_eval_subset(
        semantic_only_remaining,
        max_rows=512,
    )
    with pytest.raises(ValueError, match="rendered_prompts=7"):
        assert_no_conversation_overlap(
            primary.conversations,
            rejected.conversations,
            left_label="primary",
            right_label="semantic-only preflight",
            conversation_prompt_contract="openai_full_catalog_v1",
        )
    assert (
        rejected.audit.selected_semantic_set_sha256
        == confirmatory.REJECTED_PREFLIGHT_CONFIRMATORY_SEMANTIC_SET_SHA256
    )
    assert rejected.audit.as_dict()["audit_sha256"] == (
        confirmatory.REJECTED_PREFLIGHT_FILTERED_AUDIT_SHA256
    )

    corrected = confirmatory.derive_confirmatory_eval_selection(
        source.conversations,
        primary_max_rows=512,
        confirmatory_max_rows=512,
    )
    clean = assert_no_conversation_overlap(
        corrected.primary.conversations,
        corrected.conversations,
        left_label="primary",
        right_label="confirmatory v2",
        conversation_prompt_contract="openai_full_catalog_v1",
    )
    assert clean.semantic_overlap_sha256 == ()
    assert clean.rendered_prompt_overlap_sha256 == ()
    assert corrected.rendered_prompt_excluded_rows == 40
    assert corrected.reference_contract_sha256 == (
        confirmatory.PRODUCTION_REFERENCE_CONTRACT_SHA256
    )


def test_builder_rejects_tampered_existing_artifact(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path)
    confirmatory.build_confirmatory_eval_split(config_path)
    output = tmp_path / "confirm.jsonl"
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="refusing to replace drifted"):
        confirmatory.build_confirmatory_eval_split(config_path)


def test_selection_ignores_unrecognized_candidate_metric_metadata() -> None:
    source = [_conversation(index, split="eval") for index in range(8)]
    with_metrics = copy.deepcopy(source)
    for index, conversation in enumerate(with_metrics):
        conversation.meta["candidate_metric"] = float(index) / 10
    baseline = confirmatory.derive_confirmatory_eval_selection(
        source,
        primary_max_rows=2,
        confirmatory_max_rows=2,
    )
    mutated = confirmatory.derive_confirmatory_eval_selection(
        with_metrics,
        primary_max_rows=2,
        confirmatory_max_rows=2,
    )
    assert mutated.original_source_row_numbers == baseline.original_source_row_numbers
    assert mutated.original_source_row_numbers_sha256 == baseline.original_source_row_numbers_sha256
    assert mutated.primary.audit.canonical_bytes() == baseline.primary.audit.canonical_bytes()
    assert mutated.filtered.audit.canonical_bytes() == baseline.filtered.audit.canonical_bytes()
    assert mutated.reference_contract_sha256 == baseline.reference_contract_sha256


def test_config_and_receipt_validation_fail_closed(tmp_path: Path) -> None:
    config_path = _write_fixture_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["candidate_metrics"] = {"mean_loss": 0.0}
    drifted_config = tmp_path / "drifted.yaml"
    drifted_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="keys mismatch"):
        confirmatory.load_confirmatory_eval_split_config(drifted_config)

    duplicate_config = tmp_path / "duplicate.yaml"
    duplicate_config.write_text(
        config_path.read_text(encoding="utf-8") + f"\nkind: {confirmatory.CONFIG_KIND}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML key 'kind'"):
        confirmatory.load_confirmatory_eval_split_config(duplicate_config)

    receipt = confirmatory.build_confirmatory_eval_split(config_path)
    receipt["token_accounting"]["assistant_loss_tokens"] += 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        confirmatory.assert_confirmatory_eval_receipt(receipt)


@pytest.mark.skipif(
    not PRODUCTION_PROVENANCE.is_file(),
    reason="requires the exact local sealed confirmatory provenance receipt",
)
def test_reference_hash_uses_canonical_json_with_trailing_lf() -> None:
    receipt = confirmatory.load_confirmatory_eval_receipt(PRODUCTION_PROVENANCE)
    contract = receipt["reference_contract"]
    assert _sha256(canonical_json_bytes(contract)) == (
        confirmatory.PRODUCTION_REFERENCE_CONTRACT_SHA256
    )
