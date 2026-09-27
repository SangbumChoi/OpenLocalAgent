from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import localagent.eval.sft_candidate_promotion as promotion_module
from localagent.data.conversation_artifact import (
    canonical_json_bytes,
    conversation_semantic_sha256,
    self_hashed_manifest,
)
from localagent.data.prompt_contract import OPENAI_FULL_CATALOG_V1
from localagent.data.schema import Conversation, Message, Role, ToolCall, ToolSpec
from localagent.data.stratified_eval_selector import (
    ALGORITHM as STRATIFIED_EVAL_ALGORITHM,
)
from localagent.data.stratified_eval_selector import select_stratified_eval_subset
from localagent.eval.sft_candidate_promotion import (
    BINDING_KIND,
    DECISION_KIND,
    PROMOTION_THRESHOLDS,
    assert_sft_candidate_promotion_decision,
    load_validated_candidate_binding,
    prepare_sft_candidate,
    verify_sft_candidate_promotion,
    write_sft_candidate_promotion_decision,
)
from localagent.eval.sft_candidate_promotion import main as promotion_main
from localagent.eval.sft_checkpoint_sweep import RESULT_KIND as SWEEP_RESULT_KIND
from localagent.model import ModelConfig
from localagent.train.stage_data import canonical_sha256, tokenizer_identity


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: str, payload: bytes) -> dict[str, int | str]:
    return {"path": path, "bytes": len(payload), "sha256": _sha256(payload)}


def _write_conversation_artifact(
    root: Path,
    *,
    stem: str,
    rows: int,
    prefix: str,
) -> tuple[str, str, str, tuple[Conversation, ...]]:
    data_name = f"{stem}.jsonl"
    manifest_name = f"{stem}.jsonl.manifest.v1.json"
    config_name = f"{stem}.yaml"
    data_path = root / data_name
    manifest_path = root / manifest_name
    config_path = root / config_name
    config_payload = yaml.safe_dump(
        {"kind": "test_generator", "prefix": prefix, "rows": rows},
        sort_keys=True,
    ).encode()
    config_path.write_bytes(config_payload)

    tool = ToolSpec(
        name="lookup",
        description="Look up one exact item.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    conversations: list[Conversation] = []
    for index in range(rows):
        no_tool = index % 5 == 0
        assistant = (
            Message(role=Role.assistant, content=f"{prefix} no action {index}")
            if no_tool
            else Message(
                role=Role.assistant,
                tool_calls=[
                    ToolCall(
                        name="lookup",
                        arguments={"query": f"{prefix}-{index}"},
                    )
                ],
            )
        )
        conversations.append(
            Conversation(
                messages=[
                    Message(role=Role.user, content=f"{prefix} request {index}"),
                    assistant,
                ],
                tools=[tool],
                meta={
                    "split": "eval",
                    "rule_verified": True,
                    "environment_executed": False,
                    "category": "no_tool" if no_tool else "lookup",
                },
            )
        )
    data_payload = "".join(item.to_json() + "\n" for item in conversations).encode()
    data_path.write_bytes(data_payload)
    manifest_core = {
        "argument_schema_coverage": {},
        "argument_value_counts": {},
        "behavior_counts": {},
        "behavior_definitions": {},
        "complexity_contract": {},
        "conversation_serialization": "schema_roundtrip_jsonl_utf8_lf_v1",
        "coverage_contract": {},
        "environment_executed": False,
        "exact_prompt_holdouts": {},
        "format": "localagent.data.schema.Conversation",
        "generator_config": {
            "bytes": len(config_payload),
            "sha256": _sha256(config_payload),
        },
        "irrelevance": 0,
        "kind": "localagent_synthetic_conversation_artifact",
        "level": 5,
        "model_verified": False,
        "multi_turn": 0,
        "output_bytes": len(data_payload),
        "output_sha256": _sha256(data_payload),
        "plan_length_counts": {},
        "rows": rows,
        "rule_verification_scope": ["test_fixture"],
        "rule_verified": True,
        "schema_version": 1,
        "seed": 17,
        "single_turn": rows,
        "split": "eval",
        "split_contract": {},
        "structural_counts": {},
        "verification_claim": "test_fixture",
    }
    _manifest, manifest_payload = self_hashed_manifest(manifest_core)
    manifest_path.write_bytes(manifest_payload)
    return data_name, manifest_name, config_name, tuple(conversations)


def _selection_config(conversations: tuple[Conversation, ...]) -> tuple[dict, dict]:
    selection = select_stratified_eval_subset(conversations, max_rows=512)
    audit = selection.audit.as_dict()
    return (
        {
            "algorithm": audit["algorithm"],
            "max_rows": audit["capacity"]["max_rows"],
            "expected_source_rows": audit["source"]["rows"],
            "expected_source_assistant_decisions": audit["source"]["assistant_decisions"],
            "expected_source_semantic_set_sha256": audit["source"]["semantic_set_sha256"],
            "expected_selected_rows": audit["selected"]["rows"],
            "expected_selected_assistant_decisions": audit["selected"]["assistant_decisions"],
            "expected_selected_semantic_set_sha256": audit["selected"]["semantic_set_sha256"],
            "expected_audit_sha256": audit["audit_sha256"],
        },
        audit,
    )


def _semantic_set(conversations: tuple[Conversation, ...]) -> str:
    values = sorted({conversation_semantic_sha256(item) for item in conversations})
    return _sha256("\n".join(values).encode("ascii"))


def _sweep_record(
    root: Path,
    *,
    path: str,
    completed_steps: int,
    mean_loss: float,
    baseline: dict,
    thresholds: dict,
) -> dict:
    payload = (root / path).read_bytes()
    metrics = {
        "rows": baseline["rows"],
        "assistant_loss_tokens": baseline["assistant_loss_tokens"],
        "mean_loss": mean_loss,
        "assistant_token_accuracy": baseline["assistant_token_accuracy"],
        "assistant_sequence_accuracy": baseline["assistant_sequence_accuracy"],
    }
    delta = {
        "mean_loss": mean_loss - baseline["mean_loss"],
        "assistant_token_accuracy": 0.0,
        "assistant_sequence_accuracy": 0.0,
    }
    gates = {
        "mean_loss_non_inferiority": {
            "observed_increase": delta["mean_loss"],
            "maximum_increase": thresholds["max_mean_loss_increase"],
            "passed": delta["mean_loss"] <= thresholds["max_mean_loss_increase"],
        },
        "assistant_token_accuracy_non_inferiority": {
            "observed_drop": -delta["assistant_token_accuracy"],
            "maximum_drop": thresholds["max_assistant_token_accuracy_drop"],
            "passed": True,
        },
        "assistant_sequence_accuracy_non_inferiority": {
            "observed_drop": -delta["assistant_sequence_accuracy"],
            "maximum_drop": thresholds["max_assistant_sequence_accuracy_drop"],
            "passed": True,
        },
    }
    return {
        "artifact": _record(path, payload),
        "checkpoint_step": completed_steps - 1,
        "completed_steps": completed_steps,
        "planned_steps": 20,
        "metrics": metrics,
        "delta_from_baseline": delta,
        "gates": gates,
        "retention_eligible": all(item["passed"] for item in gates.values()),
    }


def _write_sweep(root: Path, training_name: str, training_hash: str, lineage: dict) -> None:
    sweep_config = {
        "kind": "test_sft_checkpoint_sweep_config",
        "schema_version": 2,
        "checkpoint_paths": [
            "runs/candidate.step-00000010.pt",
            "runs/runner-up.step-00000020.pt",
        ],
    }
    sweep_config_payload = yaml.safe_dump(sweep_config, sort_keys=True).encode()
    (root / "sweep.yaml").write_bytes(sweep_config_payload)
    baseline = {
        "rows": 512,
        "assistant_loss_tokens": 512,
        "mean_loss": 1.0,
        "assistant_token_accuracy": 0.5,
        "assistant_sequence_accuracy": 0.1,
    }
    thresholds = {
        "max_mean_loss_increase": 0.1,
        "max_assistant_token_accuracy_drop": 0.0,
        "max_assistant_sequence_accuracy_drop": 0.0,
    }
    candidate = _sweep_record(
        root,
        path="runs/candidate.step-00000010.pt",
        completed_steps=10,
        mean_loss=1.0,
        baseline=baseline,
        thresholds=thresholds,
    )
    runner_up = _sweep_record(
        root,
        path="runs/runner-up.step-00000020.pt",
        completed_steps=20,
        mean_loss=1.01,
        baseline=baseline,
        thresholds=thresholds,
    )
    training_payload = (root / training_name).read_bytes()
    training_config = yaml.safe_load(training_payload)
    model_name = training_config["model_config"]
    model_payload = (root / model_name).read_bytes()
    eval_sources_raw = training_config["data"]["eval_conversations"]
    eval_source_specs = (
        eval_sources_raw if isinstance(eval_sources_raw, list) else [eval_sources_raw]
    )
    heldout_sources = []
    for source in eval_source_specs:
        data_path = source["path"]
        artifact = source["artifact"]
        manifest_path = artifact["manifest"]
        generator_path = artifact["generator_config"]
        manifest_payload = (root / manifest_path).read_bytes()
        manifest = json.loads(manifest_payload)
        generator_payload = (root / generator_path).read_bytes()
        heldout_sources.append(
            {
                "path": data_path,
                "kind": manifest["kind"],
                "schema_version": manifest["schema_version"],
                "split": manifest["split"],
                "jsonl": {
                    "bytes": manifest["output_bytes"],
                    "sha256": manifest["output_sha256"],
                },
                "sidecar": {
                    "bytes": len(manifest_payload),
                    "sha256": _sha256(manifest_payload),
                    "manifest_self_sha256": manifest["manifest_self_sha256"],
                },
                "generator_config": {
                    "bytes": len(generator_payload),
                    "sha256": _sha256(generator_payload),
                },
            }
        )
    best = {
        "artifact": candidate["artifact"],
        "checkpoint_step": candidate["checkpoint_step"],
        "completed_steps": candidate["completed_steps"],
        "metrics": candidate["metrics"],
    }
    core = {
        "kind": SWEEP_RESULT_KIND,
        "schema_version": 2,
        "inputs": {
            "sweep_config": _record("sweep.yaml", sweep_config_payload),
            "sweep_config_sha256": canonical_sha256(sweep_config),
            "training_config": _record(training_name, training_payload),
            "training_config_sha256": training_hash,
            "checkpoint_discovery": {"mode": "paths"},
            "expected_parent_checkpoint_sha256": "c" * 64,
            "expected_eval": {
                "conversations": 512,
                "assistant_decisions": 512,
                "assistant_loss_tokens": 512,
            },
            "expected_baseline": {
                "metrics": baseline,
                "absolute_tolerances": {
                    "mean_loss": 0.0,
                    "assistant_token_accuracy": 0.0,
                    "assistant_sequence_accuracy": 0.0,
                },
            },
        },
        "identity": {
            "model_config": _record(model_name, model_payload),
            "model_config_sha256": lineage["model_config_sha256"],
            "tokenizer": {
                "kind": "byte",
                "vocab_size": 256,
                "sha256": lineage["tokenizer_sha256"],
            },
            "lineage": lineage,
            "training_contract": {},
        },
        "heldout": {
            "sources": heldout_sources,
            "conversations": 512,
            "assistant_decisions": 512,
            "assistant_loss_tokens": 512,
            "contract": {},
            "baseline": baseline,
            "leakage_assurance": {},
        },
        "thresholds": thresholds,
        "execution": {},
        "selection_contract": {
            "eligible_filter": "all_non_inferiority_gates_pass",
            "ranking": [
                "assistant_sequence_accuracy_desc",
                "assistant_token_accuracy_desc",
                "mean_loss_asc",
                "completed_steps_asc",
                "checkpoint_sha256_desc",
            ],
        },
        "checkpoints": [candidate, runner_up],
        "summary": {
            "evaluated_checkpoints": 2,
            "retention_eligible_checkpoints": 2,
            "failed_checkpoints": 0,
            "status": "retention_eligible_checkpoint_found",
            "best_retention_eligible_checkpoint": best,
        },
    }
    core["result_sha256"] = canonical_sha256(core)
    payload = canonical_json_bytes(core)
    (root / "sweep.json").write_bytes(payload)
    (root / "sweep-replay.json").write_bytes(payload)


def _write_confirmatory_provenance(
    root: Path,
    *,
    dev_audit: dict,
    confirm_conversations: tuple[Conversation, ...],
    confirm_data: str,
    confirm_manifest: str,
    confirm_config: str,
) -> None:
    confirm_semantic = _semantic_set(confirm_conversations)
    manifest = json.loads((root / confirm_manifest).read_text())
    data_payload = (root / confirm_data).read_bytes()
    config_payload = (root / confirm_config).read_bytes()
    manifest["output_bytes"] = len(data_payload)
    manifest["output_sha256"] = _sha256(data_payload)
    filtered_audit = "8" * 64
    reference = {
        "algorithm": (
            "exclude_primary_semantic_and_rendered_prompt_rows_then_"
            "greedy_uncovered_strata_then_semantic_sha256_fill_v1"
        ),
        "confirm_rows": 512,
        "confirm_assistant_decisions": 512,
        "confirm_semantic_set_sha256": confirm_semantic,
        "primary_selected_semantic_set_sha256": dev_audit["selected"]["semantic_set_sha256"],
        "inner_filtered_selection_audit_sha256": filtered_audit,
        "prompt_contract": OPENAI_FULL_CATALOG_V1,
    }
    filtered_selection = {
        "algorithm": STRATIFIED_EVAL_ALGORITHM,
        "audit_sha256": filtered_audit,
        "source": {
            "rows": 512,
            "assistant_decisions": 512,
            "semantic_set_sha256": confirm_semantic,
        },
        "selected": {
            "rows": 512,
            "assistant_decisions": 512,
            "semantic_set_sha256": confirm_semantic,
        },
    }
    development_overlap = {
        "left_rows": 512,
        "right_rows": 512,
        "left_semantic_set_sha256": dev_audit["selected"]["semantic_set_sha256"],
        "right_semantic_set_sha256": confirm_semantic,
        "semantic_overlap": 0,
        "semantic_overlap_sha256": [],
        "rendered_prompt_overlap": 0,
        "rendered_prompt_overlap_sha256": [],
    }
    manifest.pop("manifest_self_sha256")
    manifest["coverage_contract"] = {
        "confirmatory_eval_split": {
            "filtered_selection": filtered_selection,
            "reference_contract": reference,
            "reference_contract_sha256": _sha256(canonical_json_bytes(reference)),
        }
    }
    manifest["split_contract"] = {
        "confirmatory_eval_split": {
            "development_overlap": development_overlap,
        }
    }
    manifest, manifest_payload = self_hashed_manifest(manifest)
    (root / confirm_manifest).write_bytes(manifest_payload)
    core = {
        "kind": "localagent_confirmatory_eval_split_receipt",
        "schema_version": 2,
        "config": _record(confirm_config, config_payload),
        "output": {
            "jsonl": _record(confirm_data, data_payload),
            "manifest": {
                **_record(confirm_manifest, manifest_payload),
                "manifest_self_sha256": manifest["manifest_self_sha256"],
            },
            "rows": 512,
        },
        "development_selection": {
            "algorithm": dev_audit["algorithm"],
            "audit_sha256": dev_audit["audit_sha256"],
            "selected": {
                "rows": dev_audit["selected"]["rows"],
                "assistant_decisions": dev_audit["selected"]["assistant_decisions"],
                "semantic_set_sha256": dev_audit["selected"]["semantic_set_sha256"],
            },
        },
        "filtered_selection": filtered_selection,
        "reference_contract": reference,
        "reference_contract_sha256": _sha256(canonical_json_bytes(reference)),
        "overlap_evidence": {"development": development_overlap},
    }
    core["receipt_self_sha256"] = _sha256(canonical_json_bytes(core))
    (root / "confirm-provenance.json").write_bytes(canonical_json_bytes(core))


def _rate(correct: int, total: int) -> dict:
    return {"correct": correct, "total": total, "accuracy": correct / total}


def _write_scorecard_result(
    root: Path,
    *,
    binding: dict,
    name: str,
    out: str,
    checkpoint_override: dict | None = None,
    eos: int | None = None,
) -> None:
    scorecard_binding = binding["scorecards"][name]
    expected = scorecard_binding["expected_provenance"]
    case_set = copy.deepcopy(scorecard_binding["cases"]["case_set"])
    decisions = case_set["assistant_decisions"]
    eos = decisions if eos is None else eos
    length = decisions - eos
    provenance = {
        "evaluator": {
            "source_tree": {
                "commit": "1" * 40,
                "repository_sha256": "2" * 64,
                "worktree_sha256": "3" * 64,
                "dirty": True,
            },
            "modules": copy.deepcopy(expected["evaluator_modules"]),
        },
        **{
            key: copy.deepcopy(expected[key])
            for key in (
                "scorecard_config",
                "checkpoint",
                "checkpoint_lineage",
                "training_config",
                "model_config",
                "tokenizer",
                "training_corpus",
                "cases",
            )
        },
        "generation": {
            "requested_device": "mps",
            "resolved_device": "mps",
            "requested_dtype": "fp32",
            "resolved_dtype": "fp32",
            "temperature": 0.0,
            "max_new_tokens": 96,
            "serial_generation_calls": decisions,
            "serial_prefill_calls": decisions,
            "generation_batch_size": 1,
            "maximum_non_eos_new_tokens": decisions * 96,
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "truncation": "forbidden",
            "generation_reserve_tokens": 96,
            "prompt_budget": {
                "assistant_decisions": decisions,
                "truncation": "forbidden",
                "generation_reserve_tokens": 96,
            },
            "gold_output_budget": {
                "assistant_decisions": decisions,
                "max_new_tokens": 96,
                "fits_generation_budget": True,
            },
        },
    }
    if checkpoint_override is not None:
        provenance["checkpoint"] = copy.deepcopy(checkpoint_override)
    metrics = {
        "generation_completion": _rate(eos, decisions),
        "format_validity": _rate(decisions, decisions),
        "tool_format_validity_on_tool_decisions": _rate(
            case_set["tool_decisions"],
            case_set["tool_decisions"],
        ),
        "schema_validity_on_tool_decisions": _rate(
            case_set["tool_decisions"],
            case_set["tool_decisions"],
        ),
        "tool_name": {
            "case_exact": _rate(
                case_set["tool_decisions"],
                case_set["tool_decisions"],
            )
        },
        "whole_call_exact": _rate(
            case_set["tool_decisions"],
            case_set["tool_decisions"],
        ),
        "abstention": _rate(
            case_set["no_tool_decisions"],
            case_set["no_tool_decisions"],
        ),
    }
    finish_reasons = {"eos": eos}
    if length:
        finish_reasons["length"] = length
    core = {
        "kind": "localagent_internal_agent_scorecard_result",
        "schema_version": 1,
        "benchmark": {
            "name": "test",
            "official_bfcl": False,
            "external_native_benchmark": False,
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
        },
        "provenance": provenance,
        "scorecard": {
            "contract": {},
            "case_set": case_set,
            "metrics": metrics,
            "by_category": {},
            "predictions": {
                "sha256": "4" * 64,
                "records": decisions,
                "finish_reasons": finish_reasons,
                "complete": eos,
                "terminated_by_eos": eos,
                "raw_outputs_retained": False,
            },
        },
        "limitations": [],
    }
    core["result_self_sha256"] = _sha256(canonical_json_bytes(core))
    payload = canonical_json_bytes(core)
    (root / out).write_bytes(payload)
    if out == f"{name}-result.json":
        (root / f"{name}-replay.json").write_bytes(payload)


def _build_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dev_data, dev_manifest, dev_generator, dev_conversations = _write_conversation_artifact(
        root,
        stem="development",
        rows=600,
        prefix="development",
    )
    (
        confirm_data,
        confirm_manifest,
        confirm_generator,
        confirm_conversations,
    ) = _write_conversation_artifact(
        root,
        stem="confirmatory",
        rows=512,
        prefix="confirmatory",
    )
    selection, dev_audit = _selection_config(dev_conversations)
    _write_confirmatory_provenance(
        root,
        dev_audit=dev_audit,
        confirm_conversations=confirm_conversations,
        confirm_data=confirm_data,
        confirm_manifest=confirm_manifest,
        confirm_config=confirm_generator,
    )

    model = ModelConfig(
        name="promotion-fixture",
        vocab_size=256,
        d_model=32,
        embed_dim=32,
        n_layers=1,
        n_loops=1,
        n_heads=4,
        n_kv_heads=1,
        ffn_hidden=64,
        max_seq_len=4096,
    )
    model_payload = yaml.safe_dump(model.__dict__, sort_keys=True).encode()
    (root / "model.yaml").write_bytes(model_payload)
    training = {
        "stage": "sft",
        "model_config": "model.yaml",
        "data": {
            "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
            "tokenizer": {"kind": "byte"},
            "strict_conversation_artifacts": True,
            "eval_conversations": {
                "path": dev_data,
                "artifact": {
                    "manifest": dev_manifest,
                    "generator_config": dev_generator,
                    "expected_split": "eval",
                    "expected_rule_verified": True,
                    "environment_policy": "forbid",
                },
            },
        },
        "runtime": {"resume": "runs/latest.pt"},
    }
    training_payload = yaml.safe_dump(training, sort_keys=True).encode()
    (root / "training.yaml").write_bytes(training_payload)
    training_hash = canonical_sha256({**training, "runtime": {}})
    tokenizer = tokenizer_identity("byte", vocab_size=256)
    lineage = {
        "version": 1,
        "stage": "sft",
        "config_sha256": training_hash,
        "model_config_sha256": canonical_sha256(model.__dict__),
        "data_sha256": "5" * 64,
        "tokenizer_sha256": tokenizer["sha256"],
        "git": {
            "commit": "6" * 40,
            "repository_sha256": "7" * 64,
            "worktree_sha256": "8" * 64,
            "dirty": True,
        },
        "parent_checkpoint_sha256": "9" * 64,
    }
    (root / "runs").mkdir()
    (root / "runs/candidate.step-00000010.pt").write_bytes(b"selected candidate")
    (root / "runs/runner-up.step-00000020.pt").write_bytes(b"eligible runner up")
    _write_sweep(root, "training.yaml", training_hash, lineage)

    base = {
        "kind": "localagent_internal_agent_scorecard_config",
        "schema_version": 1,
        "checkpoint": "runs/old.pt",
        "training_config": "old-training.yaml",
        "model_config": "model.yaml",
        "tokenizer": {"kind": "byte"},
        "cases": {
            "path": dev_data,
            "manifest": dev_manifest,
            "generator_config": dev_generator,
            "expected_split": "eval",
            "expected_rule_verified": True,
            "environment_policy": "forbid",
            "selection": selection,
        },
        "generation": {"device": "auto", "max_new_tokens": 12},
    }
    (root / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=True))
    sweep_replay = json.loads((root / "sweep-replay.json").read_text())
    with patch.object(
        promotion_module,
        "run_sft_checkpoint_sweep",
        return_value=copy.deepcopy(sweep_replay),
    ):
        binding = prepare_sft_candidate(
            "sweep.json",
            "development-scorecard.yaml",
            "confirmatory-scorecard.yaml",
            "binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases=confirm_data,
            confirmatory_manifest=confirm_manifest,
            confirmatory_generator_config=confirm_generator,
            confirmatory_provenance="confirm-provenance.json",
            repository_root=root,
        )
    assert binding["kind"] == BINDING_KIND
    _write_scorecard_result(
        root,
        binding=binding,
        name="development",
        out="development-result.json",
    )
    _write_scorecard_result(
        root,
        binding=binding,
        name="confirmatory",
        out="confirmatory-result.json",
    )


@pytest.fixture(scope="module")
def fixture_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sft-candidate-template")
    _build_fixture(root)
    return root


@pytest.fixture
def prepared_root(fixture_template: Path, tmp_path: Path) -> Path:
    root = tmp_path / "fixture"
    shutil.copytree(fixture_template, root)
    return root


@pytest.fixture(autouse=True)
def deterministic_replays(
    prepared_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replay_sweep(_config_path: str | Path) -> dict:
        return json.loads((prepared_root / "sweep-replay.json").read_text())

    def replay_scorecard(config_path: str | Path) -> dict:
        name = Path(config_path).name
        if name == "development-scorecard.yaml":
            replay = "development-replay.json"
        elif name == "confirmatory-scorecard.yaml":
            replay = "confirmatory-replay.json"
        else:
            raise AssertionError(f"unexpected scorecard replay config: {config_path}")
        return json.loads((prepared_root / replay).read_text())

    monkeypatch.setattr(
        promotion_module,
        "run_sft_checkpoint_sweep",
        replay_sweep,
    )
    monkeypatch.setattr(promotion_module, "run_scorecard", replay_scorecard)


def _reseal(path: Path, *, self_field: str) -> dict:
    value = json.loads(path.read_text())
    value.pop(self_field, None)
    value[self_field] = _sha256(canonical_json_bytes(value))
    path.write_bytes(canonical_json_bytes(value))
    return value


def test_prepare_binds_summary_winner_and_both_configs(prepared_root: Path):
    binding = load_validated_candidate_binding(
        "binding.json",
        repository_root=prepared_root,
    )
    assert binding["candidate"]["artifact"]["path"] == ("runs/candidate.step-00000010.pt")
    assert binding["candidate"]["retention_eligible"] is True
    assert binding["sweep"]["model_config"]["path"] == "model.yaml"
    assert binding["sweep"]["tokenizer"]["kind"] == "byte"
    assert binding["sweep"]["heldout_sources"][0]["files"]["jsonl"]["path"] == ("development.jsonl")
    development = yaml.safe_load((prepared_root / "development-scorecard.yaml").read_text())
    confirmatory = yaml.safe_load((prepared_root / "confirmatory-scorecard.yaml").read_text())
    for config in (development, confirmatory):
        assert config["checkpoint"] == binding["candidate"]["artifact"]["path"]
        assert config["training_config"] == "training.yaml"
        assert config["generation"] == {"device": "mps", "max_new_tokens": 96}
    assert "selection" in development["cases"]
    assert "selection" not in confirmatory["cases"]
    assert binding["scorecards"]["confirmatory"]["cases"]["case_set"]["assistant_decisions"] == 512


def test_sweep_accepts_observed_baseline_within_configured_tolerance(
    prepared_root: Path,
):
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    observed_mean_loss = 1.0 - 1.0e-8
    sweep["inputs"]["expected_baseline"]["absolute_tolerances"]["mean_loss"] = 1.0e-6
    sweep["heldout"]["baseline"]["mean_loss"] = observed_mean_loss
    threshold = sweep["thresholds"]["max_mean_loss_increase"]
    for record in sweep["checkpoints"]:
        observed_increase = record["metrics"]["mean_loss"] - observed_mean_loss
        record["delta_from_baseline"]["mean_loss"] = observed_increase
        gate = record["gates"]["mean_loss_non_inferiority"]
        gate["observed_increase"] = observed_increase
        gate["passed"] = observed_increase <= threshold
        record["retention_eligible"] = all(
            item["passed"] for item in record["gates"].values()
        )
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_path.write_bytes(canonical_json_bytes(sweep))

    validated = promotion_module.load_validated_sweep_result(
        "sweep.json",
        repository_root=prepared_root,
    )

    assert validated.candidate_artifact["path"] == "runs/candidate.step-00000010.pt"


def test_sweep_rejects_observed_baseline_outside_configured_tolerance(
    prepared_root: Path,
):
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    sweep["inputs"]["expected_baseline"]["absolute_tolerances"]["mean_loss"] = 1.0e-6
    sweep["heldout"]["baseline"]["mean_loss"] = 1.0 + 1.0e-5
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_path.write_bytes(canonical_json_bytes(sweep))

    with pytest.raises(ValueError, match="mean_loss absolute_difference"):
        promotion_module.load_validated_sweep_result(
            "sweep.json",
            repository_root=prepared_root,
        )


def test_sweep_rejects_negative_baseline_tolerance(prepared_root: Path):
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    sweep["inputs"]["expected_baseline"]["absolute_tolerances"]["mean_loss"] = -1.0
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_path.write_bytes(canonical_json_bytes(sweep))

    with pytest.raises(ValueError, match="absolute_tolerances.mean_loss must be non-negative"):
        promotion_module.load_validated_sweep_result(
            "sweep.json",
            repository_root=prepared_root,
        )


def test_sweep_tolerances_never_cover_baseline_count_mismatch(prepared_root: Path):
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    sweep["heldout"]["baseline"]["rows"] += 1
    for key in sweep["inputs"]["expected_baseline"]["absolute_tolerances"]:
        sweep["inputs"]["expected_baseline"]["absolute_tolerances"][key] = 1.0
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_path.write_bytes(canonical_json_bytes(sweep))

    with pytest.raises(ValueError, match="baseline counts disagree"):
        promotion_module.load_validated_sweep_result(
            "sweep.json",
            repository_root=prepared_root,
        )


def test_verify_requires_confirmatory_before_authorizing(prepared_root: Path):
    decision = verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        repository_root=prepared_root,
    )
    assert decision["kind"] == DECISION_KIND
    assert decision["decision"] == {
        "development_passed": True,
        "confirmatory_supplied": False,
        "confirmatory_passed": False,
        "promotion_allowed": False,
        "status": "confirmatory_scorecard_required",
        "fallback_checkpoint_allowed": False,
    }


def test_verify_cli_writes_fail_closed_decision_and_exits_nonzero(
    prepared_root: Path,
    capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit) as stopped:
        promotion_main(
            [
                "verify",
                "--binding",
                "binding.json",
                "--development-scorecard",
                "development-result.json",
                "--decision-out",
                "cli-decision.json",
                "--repository-root",
                str(prepared_root),
            ]
        )
    assert stopped.value.code == 1
    assert "confirmatory_scorecard_required" in capsys.readouterr().err
    decision = json.loads((prepared_root / "cli-decision.json").read_text())
    assert decision["decision"]["promotion_allowed"] is False


def test_verify_authorizes_only_two_passing_bound_scorecards(prepared_root: Path):
    decision = verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        "confirmatory-result.json",
        repository_root=prepared_root,
    )
    assert decision["decision"]["promotion_allowed"] is True
    assert decision["thresholds"] == PROMOTION_THRESHOLDS
    output = prepared_root / "decision.json"
    write_sft_candidate_promotion_decision(
        decision,
        "decision.json",
        binding_path="binding.json",
        development_scorecard_path="development-result.json",
        confirmatory_scorecard_path="confirmatory-result.json",
        repository_root=prepared_root,
    )
    assert output.read_bytes() == canonical_json_bytes(decision)
    with pytest.raises(FileExistsError, match="refusing"):
        write_sft_candidate_promotion_decision(
            decision,
            "decision.json",
            binding_path="binding.json",
            development_scorecard_path="development-result.json",
            confirmatory_scorecard_path="confirmatory-result.json",
            repository_root=prepared_root,
        )


def test_completion_and_metric_floors_are_enforced(prepared_root: Path):
    binding = json.loads((prepared_root / "binding.json").read_text())
    _write_scorecard_result(
        prepared_root,
        binding=binding,
        name="development",
        out="development-result.json",
        eos=450,
    )
    completion_failure = verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        "confirmatory-result.json",
        repository_root=prepared_root,
    )
    gate = completion_failure["evaluations"]["development"]["gate"]
    assert gate["eos_completion"]["passed"] is False
    assert gate["truncation"]["passed"] is False
    assert completion_failure["decision"]["promotion_allowed"] is False

    _write_scorecard_result(
        prepared_root,
        binding=binding,
        name="development",
        out="development-result.json",
    )
    result_path = prepared_root / "development-result.json"
    result = json.loads(result_path.read_text())
    abstention = result["scorecard"]["metrics"]["abstention"]
    abstention["correct"] = 0
    abstention["accuracy"] = 0.0
    result.pop("result_self_sha256")
    result["result_self_sha256"] = _sha256(canonical_json_bytes(result))
    result_payload = canonical_json_bytes(result)
    result_path.write_bytes(result_payload)
    (prepared_root / "development-replay.json").write_bytes(result_payload)
    metric_failure = verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        "confirmatory-result.json",
        repository_root=prepared_root,
    )
    assert (
        metric_failure["evaluations"]["development"]["gate"]["metrics"][
            "expected_no_tool_structural_abstention"
        ]["passed"]
        is False
    )
    assert metric_failure["decision"]["promotion_allowed"] is False


def test_checkpoint_hash_drift_is_rejected_without_runner_up_fallback(prepared_root: Path):
    (prepared_root / "runs/candidate.step-00000010.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint sweep archive 0 byte identity"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            "confirmatory-result.json",
            repository_root=prepared_root,
        )


def test_confirmatory_runner_up_cannot_authorize(prepared_root: Path):
    binding = json.loads((prepared_root / "binding.json").read_text())
    runner = json.loads((prepared_root / "sweep.json").read_text())["checkpoints"][1]
    override = {
        **runner["artifact"],
        "stage": "sft",
        "step": runner["checkpoint_step"],
        "conversation_prompt_contract": OPENAI_FULL_CATALOG_V1,
    }
    _write_scorecard_result(
        prepared_root,
        binding=binding,
        name="confirmatory",
        out="runner-up-result.json",
        checkpoint_override=override,
    )
    with pytest.raises(ValueError, match="replayed confirmatory scorecard"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            "runner-up-result.json",
            repository_root=prepared_root,
        )


def test_confirmatory_reselection_is_rejected_even_when_resealed(prepared_root: Path):
    config_path = prepared_root / "confirmatory-scorecard.yaml"
    config = yaml.safe_load(config_path.read_text())
    development = yaml.safe_load((prepared_root / "development-scorecard.yaml").read_text())
    config["cases"]["selection"] = development["cases"]["selection"]
    config_payload = yaml.safe_dump(config, allow_unicode=True, sort_keys=True).encode()
    config_path.write_bytes(config_payload)

    binding_path = prepared_root / "binding.json"
    binding = json.loads(binding_path.read_text())
    config_record = binding["scorecards"]["confirmatory"]["config"]
    config_record.update(
        {
            "bytes": len(config_payload),
            "sha256": _sha256(config_payload),
            "canonical_sha256": canonical_sha256(config),
        }
    )
    expected_config = binding["scorecards"]["confirmatory"]["expected_provenance"][
        "scorecard_config"
    ]
    expected_config.update(
        {
            "bytes": len(config_payload),
            "sha256": _sha256(config_payload),
            "canonical_sha256": canonical_sha256(config),
        }
    )
    binding.pop("binding_self_sha256")
    binding["binding_self_sha256"] = _sha256(canonical_json_bytes(binding))
    binding_path.write_bytes(canonical_json_bytes(binding))
    with pytest.raises(ValueError, match="must not re-select"):
        load_validated_candidate_binding(
            "binding.json",
            repository_root=prepared_root,
        )


def test_scorecard_self_hash_and_integer_types_fail_closed(prepared_root: Path):
    result_path = prepared_root / "development-result.json"
    result = json.loads(result_path.read_text())
    result["scorecard"]["predictions"]["records"] = 512.0
    result.pop("result_self_sha256")
    result["result_self_sha256"] = _sha256(canonical_json_bytes(result))
    malformed_payload = canonical_json_bytes(result)
    result_path.write_bytes(malformed_payload)
    (prepared_root / "development-replay.json").write_bytes(malformed_payload)
    with pytest.raises(ValueError, match="must be an integer"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )

    result["scorecard"]["predictions"]["records"] = 512
    result_path.write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_absolute_archive_path_is_rejected_even_with_rehashed_sweep(prepared_root: Path):
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    absolute = str((prepared_root / sweep["checkpoints"][0]["artifact"]["path"]).resolve())
    sweep["checkpoints"][0]["artifact"]["path"] = absolute
    sweep["summary"]["best_retention_eligible_checkpoint"]["artifact"]["path"] = absolute
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_path.write_bytes(canonical_json_bytes(sweep))
    with pytest.raises(ValueError, match="literal relative path"):
        prepare_sft_candidate(
            "sweep.json",
            "new-development.yaml",
            "new-confirmatory.yaml",
            "new-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_prepare_refuses_existing_or_symlink_outputs(prepared_root: Path):
    existing = prepared_root / "existing.yaml"
    existing.write_text("preserve me")
    symlink = prepared_root / "symlink.yaml"
    symlink.symlink_to(existing.name)
    with pytest.raises(FileExistsError, match="refusing"):
        prepare_sft_candidate(
            "sweep.json",
            "existing.yaml",
            "new-confirmatory.yaml",
            "new-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )
    with pytest.raises(ValueError, match="symlink component"):
        prepare_sft_candidate(
            "sweep.json",
            "new-development.yaml",
            "symlink.yaml",
            "new-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_sweep_replay_is_mandatory_and_must_match(
    prepared_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    replay = json.loads((prepared_root / "sweep-replay.json").read_text())
    calls: list[str] = []

    def matching(config_path: str | Path) -> dict:
        calls.append(str(config_path))
        return copy.deepcopy(replay)

    monkeypatch.setattr(promotion_module, "run_sft_checkpoint_sweep", matching)
    load_validated_candidate_binding("binding.json", repository_root=prepared_root)
    assert calls == ["sweep.yaml"]
    prepare_sft_candidate(
        "sweep.json",
        "replayed-development.yaml",
        "replayed-confirmatory.yaml",
        "replayed-binding.json",
        base_scorecard_config="base.yaml",
        confirmatory_cases="confirmatory.jsonl",
        confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
        confirmatory_generator_config="confirmatory.yaml",
        confirmatory_provenance="confirm-provenance.json",
        repository_root=prepared_root,
    )
    assert calls == ["sweep.yaml", "sweep.yaml"]

    mismatched = copy.deepcopy(replay)
    mismatched["execution"] = {"hand_authored": True}
    monkeypatch.setattr(
        promotion_module,
        "run_sft_checkpoint_sweep",
        lambda _config_path: mismatched,
    )
    with pytest.raises(ValueError, match="replayed SFT checkpoint sweep"):
        load_validated_candidate_binding("binding.json", repository_root=prepared_root)


def test_scorecard_replay_is_mandatory_and_must_match(
    prepared_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def matching(config_path: str | Path) -> dict:
        name = Path(config_path).name
        calls.append(name)
        replay = (
            "development-replay.json"
            if name == "development-scorecard.yaml"
            else "confirmatory-replay.json"
        )
        return json.loads((prepared_root / replay).read_text())

    monkeypatch.setattr(promotion_module, "run_scorecard", matching)
    verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        "confirmatory-result.json",
        repository_root=prepared_root,
    )
    assert calls == ["development-scorecard.yaml", "confirmatory-scorecard.yaml"]

    def mismatching(config_path: str | Path) -> dict:
        replay = matching(config_path)
        replay["limitations"] = [*replay["limitations"], "hand-authored evidence"]
        return replay

    monkeypatch.setattr(promotion_module, "run_scorecard", mismatching)
    with pytest.raises(ValueError, match="replayed development scorecard"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_missing_runner_up_archive_is_rejected(prepared_root: Path):
    (prepared_root / "runs/runner-up.step-00000020.pt").unlink()
    with pytest.raises(ValueError, match="checkpoint sweep archive 1"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_resealed_runner_up_archive_is_rejected_by_fresh_sweep(
    prepared_root: Path,
):
    runner_path = prepared_root / "runs/runner-up.step-00000020.pt"
    runner_path.write_bytes(b"resealed runner-up")
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    runner_record = _record(
        "runs/runner-up.step-00000020.pt",
        runner_path.read_bytes(),
    )
    sweep["checkpoints"][1]["artifact"] = runner_record
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_payload = canonical_json_bytes(sweep)
    sweep_path.write_bytes(sweep_payload)

    binding_path = prepared_root / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["sweep"]["result"] = _record("sweep.json", sweep_payload)
    binding["sweep"]["result_self_sha256"] = sweep["result_sha256"]
    binding["sweep"]["checkpoint_artifacts"][1] = runner_record
    binding.pop("binding_self_sha256")
    binding["binding_self_sha256"] = _sha256(canonical_json_bytes(binding))
    binding_path.write_bytes(canonical_json_bytes(binding))

    with pytest.raises(ValueError, match="replayed SFT checkpoint sweep"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_direct_overlap_recomputation_rejects_511_concealed_rows(
    prepared_root: Path,
):
    development = tuple(
        Conversation.from_json(line)
        for line in (prepared_root / "development.jsonl").read_text().splitlines()
    )
    confirmatory = tuple(
        Conversation.from_json(line)
        for line in (prepared_root / "confirmatory.jsonl").read_text().splitlines()
    )
    selected = select_stratified_eval_subset(development, max_rows=512).conversations
    concealed = tuple(selected[:511]) + (confirmatory[-1],)
    (prepared_root / "confirmatory.jsonl").write_bytes(
        "".join(item.to_json() + "\n" for item in concealed).encode()
    )
    _selection, development_audit = _selection_config(development)
    _write_confirmatory_provenance(
        prepared_root,
        dev_audit=development_audit,
        confirm_conversations=concealed,
        confirm_data="confirmatory.jsonl",
        confirm_manifest="confirmatory.jsonl.manifest.v1.json",
        confirm_config="confirmatory.yaml",
    )

    with pytest.raises(ValueError, match="semantic=511"):
        prepare_sft_candidate(
            "sweep.json",
            "overlap-development.yaml",
            "overlap-confirmatory.yaml",
            "overlap-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_direct_overlap_recomputation_rejects_511_rendered_prompt_only_rows(
    prepared_root: Path,
):
    development = tuple(
        Conversation.from_json(line)
        for line in (prepared_root / "development.jsonl").read_text().splitlines()
    )
    confirmatory = tuple(
        Conversation.from_json(line)
        for line in (prepared_root / "confirmatory.jsonl").read_text().splitlines()
    )
    selected = select_stratified_eval_subset(development, max_rows=512).conversations
    prompt_duplicates = copy.deepcopy(tuple(selected[:511]))
    for index, conversation in enumerate(prompt_duplicates):
        assistant = conversation.messages[-1]
        if assistant.tool_calls:
            assistant.tool_calls[0].arguments["query"] += "-different-gold"
        else:
            assistant.content += f" different gold {index}"
    concealed = tuple(prompt_duplicates) + (confirmatory[-1],)
    (prepared_root / "confirmatory.jsonl").write_bytes(
        "".join(item.to_json() + "\n" for item in concealed).encode()
    )
    _selection, development_audit = _selection_config(development)
    _write_confirmatory_provenance(
        prepared_root,
        dev_audit=development_audit,
        confirm_conversations=concealed,
        confirm_data="confirmatory.jsonl",
        confirm_manifest="confirmatory.jsonl.manifest.v1.json",
        confirm_config="confirmatory.yaml",
    )

    with pytest.raises(ValueError, match="semantic=0, rendered_prompts=511"):
        prepare_sft_candidate(
            "sweep.json",
            "prompt-overlap-development.yaml",
            "prompt-overlap-confirmatory.yaml",
            "prompt-overlap-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_byte_identical_model_alias_is_rejected_before_mutation_window(
    prepared_root: Path,
):
    model_payload = (prepared_root / "model.yaml").read_bytes()
    (prepared_root / "model-scorecard-alias.yaml").write_bytes(model_payload)
    base_path = prepared_root / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["model_config"] = "model-scorecard-alias.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=True))

    with pytest.raises(ValueError, match="different model paths"):
        prepare_sft_candidate(
            "sweep.json",
            "model-alias-development.yaml",
            "model-alias-confirmatory.yaml",
            "model-alias-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_byte_identical_sweep_model_alias_is_rejected(
    prepared_root: Path,
):
    model_payload = (prepared_root / "model.yaml").read_bytes()
    sweep_alias = "model-sweep-alias.yaml"
    (prepared_root / sweep_alias).write_bytes(model_payload)
    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    sweep["identity"]["model_config"] = _record(sweep_alias, model_payload)
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_payload = canonical_json_bytes(sweep)
    sweep_path.write_bytes(sweep_payload)
    (prepared_root / "sweep-replay.json").write_bytes(sweep_payload)

    with pytest.raises(ValueError, match="different model paths"):
        prepare_sft_candidate(
            "sweep.json",
            "sweep-model-alias-development.yaml",
            "sweep-model-alias-confirmatory.yaml",
            "sweep-model-alias-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_byte_identical_tokenizer_alias_is_rejected_before_mutation_window(
    prepared_root: Path,
):
    tokenizer_payload = b'{"fixture":"byte-identical-path-alias"}\n'
    training_tokenizer = "tokenizer-training.json"
    scorecard_tokenizer = "tokenizer-scorecard-alias.json"
    (prepared_root / training_tokenizer).write_bytes(tokenizer_payload)
    (prepared_root / scorecard_tokenizer).write_bytes(tokenizer_payload)

    training_path = prepared_root / "training.yaml"
    training = yaml.safe_load(training_path.read_text())
    training["data"]["tokenizer"] = {
        "kind": "bpe",
        "path": training_tokenizer,
    }
    training_payload = yaml.safe_dump(training, sort_keys=True).encode()
    training_path.write_bytes(training_payload)
    normalized_training = copy.deepcopy(training)
    normalized_training["runtime"].pop("resume")
    training_hash = canonical_sha256(normalized_training)

    base_path = prepared_root / "base.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["tokenizer"] = {
        "kind": "bpe",
        "path": scorecard_tokenizer,
    }
    base_path.write_text(yaml.safe_dump(base, sort_keys=True))

    sweep_path = prepared_root / "sweep.json"
    sweep = json.loads(sweep_path.read_text())
    tokenizer_sha256 = _sha256(tokenizer_payload)
    sweep["inputs"]["training_config"] = _record("training.yaml", training_payload)
    sweep["inputs"]["training_config_sha256"] = training_hash
    sweep["identity"]["tokenizer"] = {
        "kind": "bpe",
        "vocab_size": 256,
        "sha256": tokenizer_sha256,
        "artifact": _record(training_tokenizer, tokenizer_payload),
    }
    sweep["identity"]["lineage"]["config_sha256"] = training_hash
    sweep["identity"]["lineage"]["tokenizer_sha256"] = tokenizer_sha256
    sweep.pop("result_sha256")
    sweep["result_sha256"] = canonical_sha256(sweep)
    sweep_payload = canonical_json_bytes(sweep)
    sweep_path.write_bytes(sweep_payload)
    (prepared_root / "sweep-replay.json").write_bytes(sweep_payload)

    with pytest.raises(ValueError, match="different tokenizer paths"):
        prepare_sft_candidate(
            "sweep.json",
            "tokenizer-alias-development.yaml",
            "tokenizer-alias-confirmatory.yaml",
            "tokenizer-alias-binding.json",
            base_scorecard_config="base.yaml",
            confirmatory_cases="confirmatory.jsonl",
            confirmatory_manifest="confirmatory.jsonl.manifest.v1.json",
            confirmatory_generator_config="confirmatory.yaml",
            confirmatory_provenance="confirm-provenance.json",
            repository_root=prepared_root,
        )


def test_model_config_toctou_is_caught_by_final_rehash(
    prepared_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def mutating_replay(config_path: str | Path) -> dict:
        replay = json.loads((prepared_root / "development-replay.json").read_text())
        model_path = prepared_root / "model.yaml"
        model_path.write_bytes(model_path.read_bytes() + b"\n")
        return replay

    monkeypatch.setattr(promotion_module, "run_scorecard", mutating_replay)
    with pytest.raises(ValueError, match="bound .*model config byte identity"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_sweep_reported_heldout_input_toctou_is_caught(
    prepared_root: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def mutating_replay(config_path: str | Path) -> dict:
        replay = json.loads((prepared_root / "development-replay.json").read_text())
        manifest_path = prepared_root / "development.jsonl.manifest.v1.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        return replay

    monkeypatch.setattr(promotion_module, "run_scorecard", mutating_replay)
    with pytest.raises(
        ValueError,
        match="bound sweep-reported heldout source 0 manifest byte identity",
    ):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )


def test_decision_validation_rejects_empty_and_internally_forged_receipts(
    prepared_root: Path,
):
    with pytest.raises(ValueError, match="keys mismatch"):
        assert_sft_candidate_promotion_decision({})

    decision = verify_sft_candidate_promotion(
        "binding.json",
        "development-result.json",
        "confirmatory-result.json",
        repository_root=prepared_root,
    )
    assert_sft_candidate_promotion_decision(decision)

    forged = copy.deepcopy(decision)
    forged["evaluations"] = {}
    forged.pop("decision_self_sha256")
    forged["decision_self_sha256"] = _sha256(canonical_json_bytes(forged))
    with pytest.raises(ValueError, match=r"evaluations.*keys mismatch"):
        assert_sft_candidate_promotion_decision(forged)

    forged = copy.deepcopy(decision)
    forged["decision"]["status"] = "promotion_authorized_by_hand"
    forged.pop("decision_self_sha256")
    forged["decision_self_sha256"] = _sha256(canonical_json_bytes(forged))
    with pytest.raises(ValueError, match="outcome is inconsistent"):
        write_sft_candidate_promotion_decision(
            forged,
            "forged-decision.json",
            binding_path="binding.json",
            development_scorecard_path="development-result.json",
            confirmatory_scorecard_path="confirmatory-result.json",
            repository_root=prepared_root,
        )
    assert not (prepared_root / "forged-decision.json").exists()


def test_external_and_ancestor_symlink_inputs_are_rejected(prepared_root: Path):
    absolute_result = str((prepared_root / "development-result.json").resolve())
    with pytest.raises(ValueError, match="literal relative path"):
        verify_sft_candidate_promotion(
            "binding.json",
            absolute_result,
            repository_root=prepared_root,
        )
    with pytest.raises(ValueError, match="literal relative path"):
        verify_sft_candidate_promotion(
            "../binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )

    (prepared_root / "linked-inputs").symlink_to(".", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        verify_sft_candidate_promotion(
            "linked-inputs/binding.json",
            "development-result.json",
            repository_root=prepared_root,
        )

    linked_root = prepared_root.parent / "linked-root"
    linked_root.symlink_to(prepared_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink ancestor"):
        verify_sft_candidate_promotion(
            "binding.json",
            "development-result.json",
            repository_root=linked_root,
        )
