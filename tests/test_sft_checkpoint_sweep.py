"""Focused contracts for the canonical teacher-forced SFT checkpoint sweep."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from localagent.data.agent_synth import Sample
from localagent.data.conversation_artifact import (
    CONVERSATION_FORMAT,
    CONVERSATION_SERIALIZATION,
    MANIFEST_KIND,
    MANIFEST_SCHEMA_VERSION,
    FileIdentity,
    canonical_json_bytes,
    self_hashed_manifest,
)
from localagent.data.prompt_contract import LEGACY_CONVERSATION_PROMPT_CONTRACT
from localagent.data.schema import Conversation, Message, Role
from localagent.eval.sft_checkpoint_sweep import (
    CONFIG_KIND,
    RESULT_KIND,
    SCHEMA_VERSION,
    _checkpoint_paths,
    assert_sft_checkpoint_sweep_result,
    load_sweep_context,
    run_sft_checkpoint_sweep,
    write_sft_checkpoint_sweep_result,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import ByteTokenizer
from localagent.train.sft import (
    _evaluate_conversations,
    _sealed_resume_sha256,
    sft,
)
from localagent.train.stage_data import canonical_sha256, tokenizer_identity


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="sft-sweep-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=128,
        dropout=0.0,
    )


def _write_verified_conversations(
    tmp_path: Path,
    conversations: list[Conversation],
    *,
    split: str,
) -> dict:
    path = tmp_path / f"{split}.jsonl"
    generator_config = tmp_path / f"{split}.generator.yaml"
    generator_config.write_text(
        yaml.safe_dump(
            {
                "kind": "sft_sweep_eval_fixture",
                "schema_version": 1,
                "out": str(path),
                "split": split,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = []
    for conversation in conversations:
        row = Conversation.from_json(conversation.to_json())
        row.meta.update(
            {
                "split": split,
                "rule_verified": True,
                "environment_executed": False,
            }
        )
        rows.append(row)
    path.write_text(
        "".join(f"{conversation.to_json()}\n" for conversation in rows),
        encoding="utf-8",
    )
    output_identity = FileIdentity.from_bytes(path.read_bytes())
    config_identity = FileIdentity.from_bytes(generator_config.read_bytes())
    manifest_core = {
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "format": CONVERSATION_FORMAT,
        "conversation_serialization": CONVERSATION_SERIALIZATION,
        "generator_config": config_identity.as_dict(),
        "rows": len(rows),
        "output_bytes": output_identity.bytes,
        "output_sha256": output_identity.sha256,
        "single_turn": len(rows),
        "multi_turn": 0,
        "irrelevance": 0,
        "seed": 0,
        "level": 0,
        "split": split,
        "rule_verified": True,
        "rule_verification_scope": ["canonical_conversation_schema"],
        "model_verified": False,
        "environment_executed": False,
        "verification_claim": "test_fixture_rule_audited_not_environment_executed",
        "split_contract": {"fixture": True},
        "exact_prompt_holdouts": {"fixture": True},
        "structural_counts": {},
        "behavior_counts": {},
        "behavior_definitions": {},
        "argument_value_counts": {},
        "argument_schema_coverage": {},
        "plan_length_counts": {},
        "complexity_contract": {"fixture": True},
        "coverage_contract": {"fixture": True},
    }
    _manifest, manifest_payload = self_hashed_manifest(manifest_core)
    manifest = path.with_suffix(path.suffix + ".manifest.v1.json")
    manifest.write_bytes(manifest_payload)
    return {
        "path": str(path),
        "artifact": {
            "generator_config": str(generator_config),
            "manifest": str(manifest),
            "expected_split": split,
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
    }


def _sweep_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = _model_config()
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(cfg.__dict__, sort_keys=True),
        encoding="utf-8",
    )
    eval_source = _write_verified_conversations(
        tmp_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="say alpha"),
                    Message(role=Role.assistant, content="alpha"),
                ]
            ),
            Conversation(
                messages=[
                    Message(role=Role.user, content="say beta"),
                    Message(role=Role.assistant, content="beta"),
                ]
            ),
        ],
        split="eval",
    )
    train_source = _write_verified_conversations(
        tmp_path,
        [
            Conversation(
                messages=[
                    Message(role=Role.user, content="train gamma"),
                    Message(role=Role.assistant, content="gamma"),
                ]
            )
        ],
        split="train",
    )
    training_config = {
        "stage": "sft",
        "model_config": str(model_config_path),
        "data": {
            "conversation_prompt_contract": LEGACY_CONVERSATION_PROMPT_CONTRACT,
            "strict_conversation_artifacts": True,
            "conversations": [train_source],
            "eval_conversations": [eval_source],
            "tokenizer": {"kind": "byte"},
            "seq_len": 128,
        },
        "evaluation": {"batch_size": 2},
        "runtime": {"device": "cpu", "dtype": "fp32", "seed": 71},
    }
    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        yaml.safe_dump(training_config, sort_keys=False),
        encoding="utf-8",
    )
    context = load_sweep_context(
        training_config_path,
        expected_conversations=2,
        expected_assistant_decisions=2,
        expected_assistant_loss_tokens=1,
    )
    torch.manual_seed(901)
    model = LocalAgentLM(cfg)
    baseline = _evaluate_conversations(
        model,
        context.conversations,
        context.tokenizer,
        max_seq_len=context.max_seq_len,
        batch_size=context.batch_size,
        device="cpu",
        conversation_prompt_contract=context.prompt_contract,
    )
    tokenizer = tokenizer_identity("byte", vocab_size=256)
    parent_checkpoint_sha256 = "5" * 64
    checkpoint_path = tmp_path / "latest.pt"
    sft(
        model,
        [
            Sample(
                category="text",
                group="text",
                prompt="say alpha",
                kind="text",
                target="alpha",
            )
        ],
        ByteTokenizer(),
        steps=2,
        batch_size=1,
        warmup=0,
        joint_tool_head=False,
        max_seq_len=128,
        conversation_prompt_contract=LEGACY_CONVERSATION_PROMPT_CONTRACT,
        seed=71,
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        archive_checkpoints=True,
        lineage={
            "version": 1,
            "stage": "sft",
            "config_sha256": context.training_config_sha256,
            "model_config_sha256": canonical_sha256(cfg.__dict__),
            "data_sha256": "1" * 64,
            "tokenizer_sha256": tokenizer["sha256"],
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "git": {
                "commit": "2" * 40,
                "repository_sha256": "3" * 64,
                "dirty": True,
                "worktree_sha256": "4" * 64,
            },
        },
        tokenizer_metadata={"kind": "byte", "sha256": tokenizer["sha256"]},
        data_metadata={
            "heldout_content_overlap": 0,
            "heldout_rendered_prompt_overlap": 0,
            "conversation_overlap_audit": copy.deepcopy(dict(context.overlap_audit)),
        },
        execution={"fixture": True},
        heldout_baseline={
            "contract": copy.deepcopy(dict(context.eval_contract)),
            "pre": baseline,
        },
        log=lambda *_: None,
    )
    sweep_config = {
        "kind": CONFIG_KIND,
        "schema_version": SCHEMA_VERSION,
        "training_config": str(training_config_path),
        "expected_parent_checkpoint_sha256": parent_checkpoint_sha256,
        "checkpoints": {
            "directory": str(tmp_path),
            "pattern": "latest.step-*.pt",
        },
        "expected_eval": {
            "conversations": 2,
            "assistant_decisions": 2,
            "assistant_loss_tokens": baseline["assistant_loss_tokens"],
        },
        "expected_baseline": {
            "metrics": baseline,
            "absolute_tolerances": {
                "mean_loss": 0.0,
                "assistant_token_accuracy": 0.0,
                "assistant_sequence_accuracy": 0.0,
            },
        },
        "thresholds": {
            "max_mean_loss_increase": 100.0,
            "max_assistant_token_accuracy_drop": 1.0,
            "max_assistant_sequence_accuracy_drop": 1.0,
        },
    }
    sweep_config_path = tmp_path / "sweep.yaml"
    sweep_config_path.write_text(
        yaml.safe_dump(sweep_config, sort_keys=False),
        encoding="utf-8",
    )
    return (
        sweep_config_path,
        tmp_path / "latest.step-00000001.pt",
        tmp_path / "latest.step-00000002.pt",
    )


def test_sweep_is_canonical_selects_best_and_never_mutates_checkpoints(
    tmp_path: Path,
) -> None:
    sweep_config, first, second = _sweep_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (first, second)}

    result = run_sft_checkpoint_sweep(sweep_config)

    assert_sft_checkpoint_sweep_result(result)
    assert [row["completed_steps"] for row in result["checkpoints"]] == [1, 2]
    assert result["heldout"]["conversations"] == 2
    assert result["heldout"]["assistant_decisions"] == 2
    assert result["heldout"]["assistant_loss_tokens"] == 11
    assert result["heldout"]["leakage_assurance"]["heldout_content_overlap"] == 0
    assert result["heldout"]["leakage_assurance"]["heldout_rendered_prompt_overlap"] == 0
    assert result["inputs"]["expected_baseline"]["absolute_tolerances"] == {
        "assistant_sequence_accuracy": 0.0,
        "assistant_token_accuracy": 0.0,
        "mean_loss": 0.0,
    }
    assert result["summary"]["retention_eligible_checkpoints"] == 2
    assert result["summary"]["best_retention_eligible_checkpoint"] is not None
    assert {path: path.read_bytes() for path in (first, second)} == before

    output = tmp_path / "sweep.json"
    write_sft_checkpoint_sweep_result(result, output)
    assert output.read_bytes() == canonical_json_bytes(result)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_sft_checkpoint_sweep_result(result, first)
    assert {path: path.read_bytes() for path in (first, second)} == before

    tampered = copy.deepcopy(result)
    tampered["summary"]["status"] = "tampered"
    with pytest.raises(ValueError, match="self-hash"):
        assert_sft_checkpoint_sweep_result(tampered)


@pytest.mark.parametrize("schema_version", [2.0, True])
def test_sweep_result_schema_version_requires_non_boolean_integer(
    schema_version: object,
) -> None:
    result = {
        "kind": RESULT_KIND,
        "schema_version": schema_version,
    }
    result["result_sha256"] = canonical_sha256(result)

    with pytest.raises(ValueError, match="kind/schema"):
        assert_sft_checkpoint_sweep_result(result)


@pytest.mark.parametrize("schema_version", [2.0, True])
def test_sweep_config_schema_version_requires_non_boolean_integer(
    tmp_path: Path,
    schema_version: object,
) -> None:
    config_path = tmp_path / "invalid-schema.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "kind": CONFIG_KIND,
                "schema_version": schema_version,
                "training_config": None,
                "checkpoints": None,
                "expected_parent_checkpoint_sha256": None,
                "expected_eval": None,
                "expected_baseline": None,
                "thresholds": None,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint sweep config must be"):
        run_sft_checkpoint_sweep(config_path)


@pytest.mark.parametrize("mutation", ["missing", "divergent"])
def test_sweep_fails_closed_on_missing_or_divergent_baseline(
    tmp_path: Path,
    mutation: str,
) -> None:
    sweep_config, first, second = _sweep_fixture(tmp_path)
    checkpoint = torch.load(second, map_location="cpu", weights_only=True)
    if mutation == "missing":
        checkpoint["heldout_baseline"] = None
        config = yaml.safe_load(sweep_config.read_text(encoding="utf-8"))
        config["checkpoints"] = {"paths": [str(second)]}
        sweep_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    else:
        checkpoint["heldout_baseline"]["pre"]["mean_loss"] += 0.125
    checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(checkpoint)
    torch.save(checkpoint, second)

    message = (
        "missing or malformed"
        if mutation == "missing"
        else "differs from the configured baseline"
    )
    with pytest.raises(ValueError, match=message):
        run_sft_checkpoint_sweep(sweep_config)
    assert first.is_file()
    assert second.is_file()


@pytest.mark.parametrize(
    "pattern",
    ["../*.pt", "**/*.pt", "/tmp/*.pt", r"nested\\*.pt"],
)
def test_checkpoint_directory_pattern_rejects_traversal(
    tmp_path: Path,
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match="basename-only"):
        _checkpoint_paths({"directory": str(tmp_path), "pattern": pattern})


def test_checkpoint_paths_require_archive_style_names(tmp_path: Path) -> None:
    rolling = tmp_path / "latest.pt"

    with pytest.raises(ValueError, match="archive-style filenames"):
        _checkpoint_paths({"paths": [str(rolling)]})


def test_sweep_rejects_wrong_parent_and_loss_token_pins(tmp_path: Path) -> None:
    sweep_config, _first, _second = _sweep_fixture(tmp_path)
    config = yaml.safe_load(sweep_config.read_text(encoding="utf-8"))
    config["expected_parent_checkpoint_sha256"] = "9" * 64
    sweep_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the configured parent"):
        run_sft_checkpoint_sweep(sweep_config)

    config["expected_parent_checkpoint_sha256"] = "5" * 64
    config["expected_eval"]["assistant_loss_tokens"] += 1
    config["expected_baseline"]["metrics"]["assistant_loss_tokens"] += 1
    sweep_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="assistant loss tokens do not match expected_eval"):
        run_sft_checkpoint_sweep(sweep_config)


def test_sweep_baseline_numeric_tolerance_is_explicit(tmp_path: Path) -> None:
    sweep_config, _first, _second = _sweep_fixture(tmp_path)
    config = yaml.safe_load(sweep_config.read_text(encoding="utf-8"))
    config["expected_baseline"]["metrics"]["mean_loss"] += 0.01
    config["expected_baseline"]["absolute_tolerances"]["mean_loss"] = 0.02
    sweep_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_sft_checkpoint_sweep(sweep_config)

    assert result["summary"]["retention_eligible_checkpoints"] == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("heldout_content_overlap", 1, "must be exactly zero"),
        ("heldout_rendered_prompt_overlap", 1, "must be exactly zero"),
        (
            "conversation_overlap_audit",
            {"tampered": True},
            "does not match the reconstructed audit",
        ),
    ],
)
def test_sweep_rejects_invalid_sealed_overlap_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    sweep_config, _first, second = _sweep_fixture(tmp_path)
    config = yaml.safe_load(sweep_config.read_text(encoding="utf-8"))
    config["checkpoints"] = {"paths": [str(second)]}
    sweep_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint = torch.load(second, map_location="cpu", weights_only=True)
    checkpoint["data"][field] = value
    checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(checkpoint)
    torch.save(checkpoint, second)

    with pytest.raises((TypeError, ValueError), match=message):
        run_sft_checkpoint_sweep(sweep_config)


def test_sweep_reconstructs_and_rejects_train_eval_overlap(tmp_path: Path) -> None:
    cfg = _model_config()
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(cfg.__dict__, sort_keys=True),
        encoding="utf-8",
    )
    shared = [
        Conversation(
            messages=[
                Message(role=Role.user, content="shared prompt"),
                Message(role=Role.assistant, content="shared answer"),
            ]
        )
    ]
    training_config_path = tmp_path / "training.yaml"
    training_config_path.write_text(
        yaml.safe_dump(
            {
                "stage": "sft",
                "model_config": str(model_config_path),
                "data": {
                    "conversation_prompt_contract": LEGACY_CONVERSATION_PROMPT_CONTRACT,
                    "strict_conversation_artifacts": True,
                    "conversations": [
                        _write_verified_conversations(tmp_path, shared, split="train")
                    ],
                    "eval_conversations": [
                        _write_verified_conversations(tmp_path, shared, split="eval")
                    ],
                    "tokenizer": {"kind": "byte"},
                    "seq_len": 128,
                },
                "evaluation": {"batch_size": 1},
                "runtime": {"device": "cpu", "dtype": "fp32"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conversation contamination"):
        load_sweep_context(
            training_config_path,
            expected_conversations=1,
            expected_assistant_decisions=1,
            expected_assistant_loss_tokens=1,
        )


def test_sweep_prefers_earliest_checkpoint_on_exact_metric_tie(tmp_path: Path) -> None:
    sweep_config, first, second = _sweep_fixture(tmp_path)
    first_checkpoint = torch.load(first, map_location="cpu", weights_only=True)
    second_checkpoint = torch.load(second, map_location="cpu", weights_only=True)
    second_checkpoint["state_dict"] = copy.deepcopy(first_checkpoint["state_dict"])
    second_checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(second_checkpoint)
    torch.save(second_checkpoint, second)

    result = run_sft_checkpoint_sweep(sweep_config)

    assert result["summary"]["best_retention_eligible_checkpoint"]["completed_steps"] == 1
    assert result["selection_contract"]["ranking"][-2] == "completed_steps_asc"
