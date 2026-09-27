"""One-update training preflight contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from localagent.data.agent_synth import synthesize
from localagent.data.conversation_artifact import canonical_json_bytes
from localagent.data.decision_quota_order import QUOTA_SAMPLING_MODE
from localagent.data.schema import Conversation, Message, Role
from localagent.data.stratified_eval_selector import ALGORITHM as STRATIFIED_EVAL_ALGORITHM
from localagent.data.stratified_eval_selector import (
    InsufficientStratumCapacityError,
    select_stratified_eval_subset,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.replay_sampling import (
    MIXED_REPLAY_SAMPLING_MODE,
    PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
)
from localagent.train.stage_data import (
    canonical_sha256,
    file_identity,
    load_conversation_source,
    tokenizer_identity,
)
from localagent.train.update_preflight import (
    PREFLIGHT_KIND,
    PREFLIGHT_SCHEMA_VERSION,
    RL_EVAL_COVERAGE_KIND,
    RL_EVAL_COVERAGE_SCHEMA_VERSION,
    _ResourceSampler,
    _bind_sft_parent_checkpoint_identity,
    _derive_sft_data_identity_and_sampling,
    _derive_sft_preflight_execution_contract,
    _expected_sft_executed_learning_rates,
    assert_preflight_receipt,
    build_one_update_pretrain_config,
    build_one_update_rl_config,
    build_one_update_sft_config,
    derive_rl_eval_minimum_coverage,
    run_one_update_rl_preflight,
    run_one_update_sft_preflight,
    seal_preflight_receipt,
)


def test_resource_sampler_reports_mps_working_set_ratios() -> None:
    sampler = _ResourceSampler()
    sampler.peak_rss_bytes = 200
    sampler.peak_mps_allocated_bytes = 25
    sampler.peak_mps_driver_bytes = 125
    sampler.mps_recommended_max_memory_bytes = 100

    memory = sampler.as_dict(baseline_rss_bytes=75)

    assert memory["peak_process_rss_delta_from_loaded_runner_bytes"] == 125
    assert memory["mps_recommended_max_memory_bytes"] == 100
    assert memory["peak_mps_allocated_to_recommended_ratio"] == 0.25
    assert memory["peak_mps_driver_to_recommended_ratio"] == 1.25
    assert memory["peak_mps_driver_within_recommended_working_set"] is False


def _source_config() -> dict:
    return {
        "stage": "pretrain",
        "model_config": "configs/model/webgpu-1m-bpe-router.yaml",
        "data": {"shards_dir": "data/shards/paper-all"},
        "schedule": {"type": "wsd", "warmup_steps": 12, "total_steps": 599},
        "batch": {"micro_batch_size": 2, "grad_accum_steps": 8},
        "runtime": {"device": "auto", "dtype": "auto", "seed": 2026, "resume": True},
        "log": {
            "out_dir": "runs/production",
            "mirror_dir": "elsewhere",
            "ckpt_every": 50,
            "eval_every": 50,
        },
    }


def test_build_one_update_pretrain_config_isolates_all_writes(tmp_path) -> None:
    source = _source_config()
    original = copy.deepcopy(source)
    effective = build_one_update_pretrain_config(
        source,
        work_dir=tmp_path / "preflight",
        device="mps",
    )

    assert source == original
    assert effective["schedule"]["total_steps"] == 1
    assert effective["batch"] == source["batch"]
    assert effective["runtime"]["resume"] is False
    assert effective["runtime"]["device"] == "mps"
    assert effective["log"]["out_dir"] == str(tmp_path / "preflight" / "run")
    assert effective["log"]["ckpt_every"] == 1
    assert effective["log"]["eval_every"] == 0
    assert "mirror_dir" not in effective["log"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("stage", "sft", "currently supports stage 'pretrain'"),
        ("steps", 0, "schedule.total_steps"),
        ("micro_batch_size", 0, "batch.micro_batch_size"),
        ("grad_accum_steps", False, "batch.grad_accum_steps"),
    ],
)
def test_build_one_update_pretrain_config_rejects_invalid_contract(
    tmp_path,
    field,
    value,
    match,
) -> None:
    source = _source_config()
    if field == "stage":
        source["stage"] = value
    elif field == "steps":
        source["schedule"]["total_steps"] = value
    else:
        source["batch"][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_one_update_pretrain_config(source, work_dir=tmp_path / "preflight")


def test_preflight_receipt_self_hash_fails_on_drift() -> None:
    sealed = seal_preflight_receipt(
        {
            "kind": PREFLIGHT_KIND,
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "passed",
        }
    )
    assert_preflight_receipt(sealed)

    sealed["status"] = "failed"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        assert_preflight_receipt(sealed)


@pytest.mark.parametrize("schema_version", [1.0, True])
def test_preflight_receipt_schema_version_requires_non_boolean_integer(
    schema_version: object,
) -> None:
    sealed = seal_preflight_receipt(
        {
            "kind": PREFLIGHT_KIND,
            "schema_version": schema_version,
            "status": "passed",
        }
    )

    with pytest.raises(ValueError, match="unsupported one-update preflight receipt"):
        assert_preflight_receipt(sealed)


def test_build_one_update_sft_config_preserves_format_bootstrap_contract(
    tmp_path,
) -> None:
    config_path = Path("configs/train/sft-paper-tier-1m-format-bootstrap.yaml")
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original = copy.deepcopy(source)

    effective = build_one_update_sft_config(
        source,
        work_dir=tmp_path / "preflight",
        device="mps",
    )

    assert source == original
    assert effective["schedule"]["total_steps"] == 1
    assert effective["batch"] == {
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "pad_to_input_tokens": 3_529,
    }
    assert effective["evaluation"]["pad_to_input_tokens"] == 3_598
    assert effective["init_from"] == source["init_from"]
    assert effective["continuation"] == {
        "mode": "fresh_optimizer_sft_child_v1"
    }
    assert effective["data"] == source["data"]
    assert effective["evaluation"] == source["evaluation"]
    assert effective["runtime"]["resume"] is False
    assert effective["runtime"]["device"] == "mps"
    assert effective["log"]["out_dir"] == str(tmp_path / "preflight" / "run")
    assert effective["log"]["ckpt_every"] == 1
    model = ModelConfig.from_yaml(effective["model_config"])
    model.assert_within_budget()


def test_build_one_update_sft_config_preserves_mixed_replay_cycle(
    tmp_path,
) -> None:
    config_path = Path(
        "configs/train/sft-paper-tier-1m-mixed-replay-pilot.yaml"
    )
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original = copy.deepcopy(source)

    effective = build_one_update_sft_config(
        source,
        work_dir=tmp_path / "preflight",
        device="mps",
    )

    assert source == original
    assert effective["schedule"]["total_steps"] == source["schedule"]["total_steps"]
    assert effective["data"]["sampling"] == source["data"]["sampling"]
    assert len(effective["data"]["sampling"]["cycle"]) == (
        effective["batch"]["micro_batch_size"]
        * effective["batch"]["grad_accum_steps"]
    )
    assert effective["optim"]["loss_normalization"] == (
        "assistant_token_mean_per_update_v1"
    )
    assert effective["batch"]["pad_to_input_tokens"] == 3_684
    assert effective["evaluation"]["pad_to_input_tokens"] == 3_598
    assert effective["log"]["archive_checkpoints"] is True
    assert effective["log"]["ckpt_every"] == 1


def test_build_one_update_sft_config_preserves_parent_anchor_horizon_and_freeze(
    tmp_path,
) -> None:
    config_path = Path(
        "configs/train/sft-paper-tier-1m-parent-anchor-pulse-pilot.yaml"
    )
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original = copy.deepcopy(source)

    effective = build_one_update_sft_config(
        source,
        work_dir=tmp_path / "preflight",
        device="mps",
    )

    assert source == original
    assert effective["schedule"]["total_steps"] == 372
    assert effective["data"]["sampling"] == source["data"]["sampling"]
    assert (
        effective["data"]["sampling"]["mode"]
        == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    )
    assert effective["data"]["sampling"]["update_decisions"] == (
        effective["batch"]["micro_batch_size"]
        * effective["batch"]["grad_accum_steps"]
    )
    assert effective["optim"]["loss_normalization"] == "microbatch_mean_v1"
    assert effective["optim"]["freeze_parameters"] == [
        "loop_embed",
        "embed.weight",
        "in_proj.weight",
        "out_proj.weight",
    ]
    assert effective["log"]["archive_checkpoints"] is True
    assert effective["log"]["ckpt_every"] == 1


def test_mixed_replay_identity_seals_production_contract_and_first_update(
    tmp_path,
    monkeypatch,
) -> None:
    general_path = tmp_path / "general.jsonl"
    format_path = tmp_path / "format.jsonl"
    _write_conversation_rows(general_path, prefix="general", rows=36)
    _write_conversation_rows(format_path, prefix="format", rows=4)
    cycle = [
        "general",
        "general",
        "general",
        "format_core",
        "general",
        "general",
        "general",
        "multi_argument",
        "general",
        "general",
        "general",
        "parallel",
        "general",
        "general",
        "general",
        "text",
    ]
    source = {
        "data": {
            "conversation_prompt_contract": "openai_full_catalog_v1",
            "conversations": [str(general_path), str(format_path)],
            "shuffle": False,
            "sampling": {
                "mode": MIXED_REPLAY_SAMPLING_MODE,
                "general_source_index": 0,
                "format_source_index": 1,
                "exclude_format_semantic_overlap": True,
                "cycle": cycle,
            },
        },
        "schedule": {"total_steps": 2},
        "batch": {"micro_batch_size": 2, "grad_accum_steps": 8},
    }
    ordered_keys = tuple((index, 1) for index in range(40))
    production_contract = {
        "mode": MIXED_REPLAY_SAMPLING_MODE,
        "selected_decisions": 32,
        "cycle": {"length": 16, "labels": cycle},
        "complete_order_sha256": "a" * 64,
        "selected_order_sha256": "b" * 64,
    }

    def fake_mixed_replay_sampling_window(
        source_conversations,
        *,
        selected_decisions,
        sampling_config,
    ):
        assert len(source_conversations) == 2
        assert selected_decisions == 32
        assert sampling_config["mode"] == MIXED_REPLAY_SAMPLING_MODE
        return ordered_keys, production_contract

    monkeypatch.setattr(
        "localagent.train.replay_sampling.mixed_replay_sampling_window",
        fake_mixed_replay_sampling_window,
    )

    identity, evidence = _derive_sft_data_identity_and_sampling(source)

    assert identity["decision_sampling"] == production_contract
    assert evidence is not None
    assert evidence["production"] == {
        "selected_decisions": 32,
        "sampling_contract": production_contract,
        "sampling_contract_sha256": canonical_sha256(production_contract),
    }
    prefix = evidence["exercised_prefix"]
    expected_keys = [[index, 1] for index in range(16)]
    assert prefix == {
        "decisions": 16,
        "decision_keys": expected_keys,
        "decision_keys_sha256": canonical_sha256(expected_keys),
        "equals_production_order_prefix": True,
    }


def test_parent_anchored_sampling_identity_binds_exact_parent_and_rejects_drift() -> None:
    sampling_contract = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "selected_decisions": 48,
    }
    identity = {"decision_sampling": sampling_contract}
    evidence = {
        "production": {
            "selected_decisions": 48,
            "sampling_contract": sampling_contract,
            "sampling_contract_sha256": canonical_sha256(sampling_contract),
        }
    }
    binding = {
        "parent_lm_sampling_mode": QUOTA_SAMPLING_MODE,
        "parent_no_replacement": True,
        "parent_order_sha256": "a" * 64,
        "parent_completed_steps": 1,
        "parent_completed_lm_cursor": 6,
        "parent_update_decisions": 6,
    }

    bound_identity, bound_evidence = _bind_sft_parent_checkpoint_identity(
        identity,
        evidence,
        parent_checkpoint_binding=binding,
    )

    expected_contract = {
        **sampling_contract,
        "parent_checkpoint_binding": binding,
    }
    assert bound_identity["decision_sampling"] == expected_contract
    assert bound_evidence["production"]["sampling_contract"] == expected_contract
    assert bound_evidence["production"]["sampling_contract_sha256"] == canonical_sha256(
        expected_contract
    )
    assert identity["decision_sampling"] == sampling_contract

    drifted = copy.deepcopy(identity)
    drifted["decision_sampling"]["parent_checkpoint_binding"] = {
        **binding,
        "parent_completed_lm_cursor": 0,
    }
    with pytest.raises(ValueError, match="drifted parent binding"):
        _bind_sft_parent_checkpoint_identity(
            drifted,
            evidence,
            parent_checkpoint_binding=binding,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_continuation", "requires continuation.mode"),
        ("route_optimizer", "train_route_head=false"),
        ("dense_optimizer", "train_dense_selector=false"),
    ],
)
def test_build_one_update_sft_config_rejects_ambiguous_optimizer_scope(
    tmp_path,
    mutation,
    match,
) -> None:
    source = yaml.safe_load(
        Path("configs/train/sft-paper-tier-1m-format-bootstrap.yaml").read_text(
            encoding="utf-8"
        )
    )
    if mutation == "missing_continuation":
        source.pop("continuation")
    elif mutation == "route_optimizer":
        source["heads"]["train_route_head"] = True
    else:
        source["heads"]["train_dense_selector"] = True

    with pytest.raises(ValueError, match=match):
        build_one_update_sft_config(source, work_dir=tmp_path / "preflight")


def _rl_source_config() -> dict:
    return {
        "stage": "rl",
        "model_config": "configs/model/webgpu-1m-bpe-router.yaml",
        "init_from": "runs/sft-paper-tier-1m/latest.pt",
        "data": {
            "conversations": ["train.jsonl"],
            "eval_conversations": ["eval.jsonl"],
            "tokenizer": {"kind": "bpe", "path": "tokenizer.json"},
        },
        "evaluation": {
            "max_conversations": 512,
            "selection": STRATIFIED_EVAL_ALGORITHM,
        },
        "rollout": {
            "prompts_per_step": 8,
            "group_size": 4,
            "max_new_tokens": 256,
            "temperature": 1.0,
        },
        "policy": {
            "clip_ratio": 0.2,
            "kl_beta": 0.02,
            "epochs_per_rollout": 2,
        },
        "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
        "schedule": {"warmup_steps": 1, "total_steps": 18},
        "runtime": {"device": "auto", "dtype": "auto", "seed": 2026, "resume": True},
        "log": {
            "out_dir": "runs/rl-production",
            "mirror_dir": "elsewhere",
            "ckpt_every": 2,
        },
    }


def _minimum_coverage_contract(minimum_rows: int = 3) -> dict:
    audit_core = {
        "algorithm": STRATIFIED_EVAL_ALGORITHM,
        "capacity": {
            "max_rows": minimum_rows,
            "coverage_rows": minimum_rows,
            "fill_rows": 0,
        },
        "mandatory_strata": minimum_rows + 1,
        "selected": {"rows": minimum_rows},
    }
    audit = {
        **audit_core,
        "audit_sha256": hashlib.sha256(canonical_json_bytes(audit_core)).hexdigest(),
    }
    payload = {
        "kind": RL_EVAL_COVERAGE_KIND,
        "schema_version": RL_EVAL_COVERAGE_SCHEMA_VERSION,
        "selector": STRATIFIED_EVAL_ALGORITHM,
        "production_max_conversations": 512,
        "minimum_coverage_rows": minimum_rows,
        "mandatory_strata": minimum_rows + 1,
        "verified_eval_artifacts": [{"path": "eval.jsonl", "sha256": "a" * 64}],
        "selection_audit": audit,
    }
    payload["derivation_sha256"] = canonical_sha256(payload)
    return payload


def test_build_one_update_rl_config_preserves_policy_and_isolates_writes(tmp_path) -> None:
    source = _rl_source_config()
    original = copy.deepcopy(source)
    effective = build_one_update_rl_config(
        source,
        work_dir=tmp_path / "preflight",
        device="mps",
        evaluation_coverage=_minimum_coverage_contract(),
    )

    assert source == original
    assert effective["schedule"] == source["schedule"]
    assert effective["rollout"] == source["rollout"]
    assert effective["policy"] == source["policy"]
    assert effective["reward"] == source["reward"]
    assert effective["runtime"]["resume"] is False
    assert effective["runtime"]["device"] == "mps"
    assert effective["evaluation"]["max_conversations"] == 3
    assert effective["evaluation"]["selection"] == source["evaluation"]["selection"]
    assert (
        effective["evaluation"]["preflight_minimum_coverage"]
        == _minimum_coverage_contract()
    )
    assert effective["log"]["out_dir"] == str(tmp_path / "preflight" / "run")
    assert effective["log"]["ckpt_every"] == 1
    assert "mirror_dir" not in effective["log"]


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("root", "stage", "sft", "requires stage 'rl'"),
        ("schedule", "total_steps", 0, "schedule.total_steps"),
        ("rollout", "prompts_per_step", 0, "rollout.prompts_per_step"),
        ("rollout", "group_size", 1, "rollout.group_size"),
        ("rollout", "max_new_tokens", False, "rollout.max_new_tokens"),
        ("policy", "epochs_per_rollout", 0, "policy.epochs_per_rollout"),
        ("evaluation", "max_conversations", 0, "evaluation.max_conversations"),
    ],
)
def test_build_one_update_rl_config_rejects_invalid_contract(
    tmp_path,
    section,
    field,
    value,
    match,
) -> None:
    source = _rl_source_config()
    if section == "root":
        source[field] = value
    else:
        source[section][field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_one_update_rl_config(
            source,
            work_dir=tmp_path / "preflight",
            evaluation_coverage=_minimum_coverage_contract(),
        )


def test_build_one_update_rl_config_rejects_drifted_or_preexisting_derivation(
    tmp_path,
) -> None:
    source = _rl_source_config()
    drifted = _minimum_coverage_contract()
    drifted["minimum_coverage_rows"] += 1
    with pytest.raises(ValueError, match="minimum-coverage derivation is invalid"):
        build_one_update_rl_config(
            source,
            work_dir=tmp_path / "drifted",
            evaluation_coverage=drifted,
        )

    source["evaluation"]["preflight_minimum_coverage"] = _minimum_coverage_contract()
    with pytest.raises(ValueError, match="derived-only"):
        build_one_update_rl_config(
            source,
            work_dir=tmp_path / "preexisting",
            evaluation_coverage=_minimum_coverage_contract(),
        )


def _write_conversation_rows(path: Path, *, prefix: str, rows: int) -> None:
    conversations = [
        Conversation(
            messages=[
                Message(role=Role.user, content=f"{prefix} prompt {index}"),
                Message(role=Role.assistant, content=f"{prefix} answer {index}"),
            ]
        )
        for index in range(rows)
    ]
    path.write_text(
        "".join(f"{conversation.to_json()}\n" for conversation in conversations),
        encoding="utf-8",
    )


def _write_completed_sft_parent(path: Path, model_config: ModelConfig) -> None:
    from localagent.data.prompt_contract import LEGACY_CONVERSATION_PROMPT_CONTRACT
    from localagent.train.sft import _sealed_resume_sha256

    tokenizer = tokenizer_identity("byte", vocab_size=model_config.vocab_size)
    training_contract = {
        "steps": 1,
        "batch_size": 2,
        "accum_steps": 3,
        "lm_sampling": {
            "mode": QUOTA_SAMPLING_MODE,
            "no_replacement": True,
            "ordering": {"order_sha256": "a" * 64},
        },
    }
    token_accounting = {
        "input_tokens": 1,
        "loss_tokens": 1,
        "sources": {"parent.jsonl": {"input_tokens": 1, "loss_tokens": 1, "rows": 1}},
    }
    payload = {
        "resume_format": "localagent.sft_resume",
        "resume_version": 1,
        "cfg": model_config.__dict__,
        "state_dict": LocalAgentLM(model_config).state_dict(),
        "tool_head": None,
        "ptr_head": None,
        "optimizer": {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        },
        "grad_scaler": None,
        "step": 0,
        "loss_history": [1.0],
        "dataset_token_accounting": {"main": token_accounting, "decay": None},
        "token_accounting": token_accounting,
        "token_accounting_scope": "language_model_microbatches",
        "sampling_state": {
            "lm_cursor": 6,
            "completed_steps": 1,
            "completed_microbatches": 3,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": None,
        "mps_rng_state": None,
        "xpu_rng_state_all": None,
        "stage": "sft",
        "training_seed": 7,
        "training_contract": training_contract,
        "lineage": {
            "version": 1,
            "stage": "sft",
            "tokenizer_sha256": tokenizer["sha256"],
        },
        "conversation_prompt_contract": LEGACY_CONVERSATION_PROMPT_CONTRACT,
        "tokenizer": {
            "kind": "byte",
            "path": None,
            "sha256": tokenizer["sha256"],
        },
        "data": {},
        "execution": {
            "requested_device": "cpu",
            "resolved_device": "cpu",
        },
        "heldout_baseline": None,
    }
    payload["resume_integrity_sha256"] = _sealed_resume_sha256(payload)
    torch.save(payload, path)


def _sft_parent_pins(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    training_contract = checkpoint["training_contract"]
    sampling_state = checkpoint["sampling_state"]
    return {
        "checkpoint_sha256": file_identity(path)["sha256"],
        "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
        "training_contract_sha256": canonical_sha256(training_contract),
        "lm_sampling_sha256": canonical_sha256(training_contract["lm_sampling"]),
        "completed_steps": sampling_state["completed_steps"],
        "completed_lm_cursor": sampling_state["lm_cursor"],
    }


def _write_sft_preflight_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_config = ModelConfig(
        name="sft-preflight-test",
        vocab_size=256,
        d_model=16,
        embed_dim=8,
        n_layers=1,
        n_loops=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_seq_len=64,
        dropout=0.0,
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(model_config.__dict__, sort_keys=False),
        encoding="utf-8",
    )
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_conversation_rows(train_path, prefix="train", rows=8)
    _write_conversation_rows(eval_path, prefix="eval", rows=1)
    parent_path = tmp_path / "completed-sft-parent.pt"
    _write_completed_sft_parent(parent_path, model_config)
    production_path = tmp_path / "production-sft"
    config_path = tmp_path / "sft.yaml"
    config = {
        "stage": "sft",
        "model_config": str(model_path),
        "init_from": str(parent_path),
        "continuation": {"mode": "fresh_optimizer_sft_child_v1"},
        "data": {
            "conversations": [str(train_path)],
            "eval_conversations": [str(eval_path)],
            "tokenizer": {"kind": "byte"},
            "seq_len": 64,
            "function_masking": False,
            "shuffle": False,
        },
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0},
        "schedule": {
            "type": "wsd",
            "warmup_steps": 10,
            "total_steps": 7,
            "decay_frac": 0.2,
        },
        "batch": {"micro_batch_size": 2, "grad_accum_steps": 3},
        "evaluation": {"batch_size": 1},
        "heads": {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "fp32",
            "seed": 2026,
            "resume": True,
        },
        "log": {
            "out_dir": str(production_path),
            "mirror_dir": str(tmp_path / "mirror"),
            "ckpt_every": 4,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, production_path, parent_path


def _write_fake_sft_outputs(
    effective_config_path: str,
    *,
    optimizer_step: int | None = None,
    optimizer_parameter_count_delta: int = 0,
    optimizer_model_parameter_names_override: list[str] | None = None,
    optimizer_contract_override: dict | None = None,
    mutate_frozen_parameter: bool = False,
    change_unfrozen_parameter: bool | None = None,
    production_mutation: Path | None = None,
    parent_mutation: Path | None = None,
) -> None:
    from localagent.data.prompt_contract import LEGACY_CONVERSATION_PROMPT_CONTRACT
    from localagent.train.sft import (
        _sealed_resume_sha256,
        _validate_parent_anchored_sampling_parent,
    )
    from localagent.train.update_preflight import _derive_sft_data_identity

    effective = yaml.safe_load(Path(effective_config_path).read_text(encoding="utf-8"))
    out_dir = Path(effective["log"]["out_dir"])
    out_dir.mkdir(parents=True)
    model_config = ModelConfig.from_yaml(effective["model_config"])
    parent_checkpoint = torch.load(
        effective["init_from"],
        map_location="cpu",
        weights_only=True,
    )
    data_identity = _derive_sft_data_identity(effective)
    sampling_config = effective.get("data", {}).get("sampling")
    if (
        isinstance(sampling_config, dict)
        and sampling_config.get("mode")
        == PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    ):
        parent_binding = _validate_parent_anchored_sampling_parent(
            parent_checkpoint,
            sampling_config,
        )
        decision_sampling = dict(data_identity["decision_sampling"])
        decision_sampling["parent_checkpoint_binding"] = parent_binding
        data_identity["decision_sampling"] = decision_sampling
    execution_contract = _derive_sft_preflight_execution_contract(
        effective,
        data_identity,
    )
    execution_update_limit = execution_contract[
        "execution_optimizer_update_limit"
    ]
    if optimizer_step is None:
        optimizer_step = execution_update_limit
    expected_learning_rates = _expected_sft_executed_learning_rates(
        effective,
        execution_update_limit=execution_update_limit,
    )
    model_parameter_names = [
        name for name, _ in LocalAgentLM(model_config).named_parameters()
    ]
    frozen_parameter_names = list(
        effective.get("optim", {}).get("freeze_parameters", [])
    )
    frozen_parameter_set = set(frozen_parameter_names)
    optimizer_model_parameter_names = [
        name
        for name in model_parameter_names
        if name not in frozen_parameter_set
    ]
    state_dict = {
        name: tensor.detach().clone()
        for name, tensor in parent_checkpoint["state_dict"].items()
    }
    should_change_unfrozen = (
        any(learning_rate != 0.0 for learning_rate in expected_learning_rates)
        if change_unfrozen_parameter is None
        else change_unfrozen_parameter
    )
    if should_change_unfrozen:
        changed_name = optimizer_model_parameter_names[0]
        state_dict[changed_name].view(-1)[0].add_(1.0)
    if mutate_frozen_parameter:
        state_dict[frozen_parameter_names[0]].view(-1)[0].add_(1.0)
    tokenizer = tokenizer_identity("byte", vocab_size=model_config.vocab_size)
    lineage_config = copy.deepcopy(effective)
    lineage_config["runtime"].pop("resume", None)
    lineage = {
        "version": 1,
        "stage": "sft",
        "config_sha256": canonical_sha256(lineage_config),
        "model_config_sha256": canonical_sha256(model_config.__dict__),
        "data_sha256": canonical_sha256(data_identity),
        "tokenizer_sha256": tokenizer["sha256"],
        "parent_checkpoint_sha256": hashlib.sha256(
            Path(effective["init_from"]).read_bytes()
        ).hexdigest(),
        "git": None,
    }
    train_path = str(Path(effective["data"]["conversations"][0]))
    eval_path = str(Path(effective["data"]["eval_conversations"][0]))
    micro_batch_size = int(effective["batch"]["micro_batch_size"])
    grad_accum_steps = int(effective["batch"]["grad_accum_steps"])
    realized_rows = (
        execution_update_limit * micro_batch_size * grad_accum_steps
    )
    dataset_rows = sum(
        1 for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line
    )
    dataset_source = {
        "input_tokens": dataset_rows * 5,
        "loss_tokens": dataset_rows * 2,
        "rows": dataset_rows,
    }
    realized_source = {
        "input_tokens": realized_rows * 5,
        "loss_tokens": realized_rows * 2,
        "rows": realized_rows,
    }
    dataset_accounting = {
        "main": {
            "input_tokens": dataset_source["input_tokens"],
            "loss_tokens": dataset_source["loss_tokens"],
            "sources": {train_path: dataset_source},
        },
        "decay": None,
    }
    token_accounting = {
        "input_tokens": realized_source["input_tokens"],
        "loss_tokens": realized_source["loss_tokens"],
        "sources": {train_path: realized_source},
    }
    lm_sampling = data_identity.get(
        "decision_sampling",
        {"mode": "source_order_wrapping_v1"},
    )
    data = {
        "conversation_rows": dataset_rows,
        "single_turn_rows": dataset_rows,
        "probe_decision_rows": dataset_rows,
        "paths": [train_path],
        "eval_conversation_rows": 1,
        "eval_source_conversation_rows": 1,
        "eval_paths": [eval_path],
        "heldout_content_overlap": 0,
        "heldout_rendered_prompt_overlap": 0,
        **(
            {"decision_sampling": lm_sampling}
            if "decision_sampling" in data_identity
            else {}
        ),
    }
    execution = {
        "requested_device": effective["runtime"]["device"],
        "resolved_device": effective["runtime"]["device"],
    }
    heldout_baseline = {
        "contract": {"kind": "deterministic_test_eval"},
        "pre": {
            "mean_loss": 1.0,
            "assistant_token_accuracy": 0.0,
            "assistant_sequence_accuracy": 0.0,
        },
    }
    heldout_eval = {
        "contract": heldout_baseline["contract"],
        "pre": heldout_baseline["pre"],
        "post": {
            "mean_loss": 0.9,
            "assistant_token_accuracy": 0.1,
            "assistant_sequence_accuracy": 0.0,
        },
        "delta": {
            "mean_loss": -0.1,
            "assistant_token_accuracy": 0.1,
            "assistant_sequence_accuracy": 0.0,
        },
    }
    training_contract = {
        "steps": int(effective["schedule"]["total_steps"]),
        "batch_size": micro_batch_size,
        "accum_steps": grad_accum_steps,
        "lm_sampling": lm_sampling,
        "loss_normalization": effective.get("optim", {}).get(
            "loss_normalization",
            "microbatch_mean_v1",
        ),
        **(
            {
                "freeze_parameters": list(
                    effective["optim"]["freeze_parameters"]
                ),
                "optimizer_model_parameter_names": (
                    optimizer_model_parameter_names_override
                    if optimizer_model_parameter_names_override is not None
                    else optimizer_model_parameter_names
                ),
            }
            if "freeze_parameters" in effective.get("optim", {})
            else {}
        ),
        "optimizer": (
            optimizer_contract_override
            if optimizer_contract_override is not None
            else {
                "kind": "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": float(
                    effective.get("optim", {}).get("weight_decay", 0.0)
                ),
                "grad_clip": float(
                    effective.get("optim", {}).get("grad_clip", 1.0)
                ),
            }
        ),
        "initial_model_sha256": "a" * 64,
    }
    checkpoint_path = out_dir / "latest.pt"
    checkpoint = {
        "resume_format": "localagent.sft_resume",
        "resume_version": 1,
        "cfg": model_config.__dict__,
        "state_dict": state_dict,
        "tool_head": None,
        "ptr_head": None,
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(float(optimizer_step)),
                    "exp_avg": torch.tensor([0.0]),
                    "exp_avg_sq": torch.tensor([0.0]),
                }
            },
            "param_groups": [
                {
                    "params": list(
                        range(
                            len(optimizer_model_parameter_names)
                            + optimizer_parameter_count_delta
                        )
                    ),
                    "lr": expected_learning_rates[-1],
                }
            ],
        },
        "grad_scaler": None,
        "step": execution_update_limit - 1,
        "loss_history": [1.25] * execution_update_limit,
        "dataset_token_accounting": dataset_accounting,
        "token_accounting": token_accounting,
        "token_accounting_scope": "language_model_microbatches",
        "sampling_state": {
            "rng_state": None,
            "lm_cursor": (
                realized_rows
                if isinstance(lm_sampling, dict)
                and lm_sampling.get("no_replacement") is True
                else realized_rows % dataset_rows
            ),
            "completed_steps": execution_update_limit,
            "completed_microbatches": execution_update_limit * grad_accum_steps,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": None,
        "mps_rng_state": None,
        "xpu_rng_state_all": None,
        "stage": "sft",
        "training_seed": int(effective["runtime"]["seed"]),
        "training_contract": training_contract,
        "lineage": lineage,
        "conversation_prompt_contract": LEGACY_CONVERSATION_PROMPT_CONTRACT,
        "tokenizer": {
            "kind": "byte",
            "path": None,
            "sha256": tokenizer["sha256"],
        },
        "data": data,
        "execution": execution,
        "heldout_baseline": heldout_baseline,
    }
    checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(checkpoint)
    fixed_horizon_progress = {
        "planned_optimizer_updates": int(effective["schedule"]["total_steps"]),
        "completed_optimizer_updates": execution_update_limit,
        "partial": (
            int(effective["schedule"]["total_steps"]) > execution_update_limit
        ),
    }
    checkpoint.update(
        {
            "heldout_eval": heldout_eval,
            "heldout_structured_eval": None,
            "continuation": copy.deepcopy(effective["continuation"]),
            "fixed_horizon_progress": fixed_horizon_progress,
        }
    )
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "stage": "sft",
        "checkpoint": str(checkpoint_path),
        "conversation_rows": dataset_rows,
        "single_turn_rows": dataset_rows,
        "probe_decision_rows": dataset_rows,
        "loss_last": 1.25,
        "loss_steps": execution_update_limit,
        "dataset_token_accounting": dataset_accounting,
        "token_accounting": token_accounting,
        "token_accounting_scope": "language_model_microbatches",
        "lm_sampling": lm_sampling,
        "fixed_horizon_progress": fixed_horizon_progress,
        "lineage": lineage,
        "data": data,
        "heldout_eval": heldout_eval,
        "heldout_structured_eval": None,
        "execution": execution,
        "continuation": copy.deepcopy(effective["continuation"]),
        "structured_heads": {
            "tool_pointer": False,
            "route": False,
            "dense_selector": False,
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if production_mutation is not None:
        production_mutation.mkdir(parents=True)
        (production_mutation / "latest.pt").write_bytes(b"forbidden production mutation")
    if parent_mutation is not None:
        parent_mutation.write_bytes(parent_mutation.read_bytes() + b"forbidden parent mutation")


def _write_rl_preflight_fixture(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    parent_path = tmp_path / "sft-parent.pt"
    production_path = tmp_path / "production-rl"
    config_path = tmp_path / "rl.yaml"
    train_path.write_text('{"row":"train"}\n', encoding="utf-8")
    eval_path.write_text('{"row":"eval"}\n', encoding="utf-8")
    model_config_path = Path("configs/model/ultra-tiny-1m.yaml").resolve()
    model_config = ModelConfig.from_yaml(model_config_path)
    torch.manual_seed(17)
    parent_model = LocalAgentLM(model_config)
    torch.save({"state_dict": parent_model.state_dict()}, parent_path)
    config = {
        "stage": "rl",
        "model_config": str(model_config_path),
        "init_from": str(parent_path),
        "data": {
            "conversations": [str(train_path)],
            "eval_conversations": [str(eval_path)],
            "tokenizer": {"kind": "byte"},
        },
        "evaluation": {},
        "environment": {"name": "canonical_toolcalls", "learned_judge": False},
        "rollout": {
            "prompts_per_step": 1,
            "group_size": 2,
            "max_new_tokens": 4,
            "temperature": 1.0,
        },
        "policy": {
            "clip_ratio": 0.2,
            "kl_beta": 0.0,
            "epochs_per_rollout": 1,
        },
        "reward": {"format_weight": 0.1, "truncation_penalty": 0.05},
        "optim": {"lr": 2.0e-5},
        "schedule": {"warmup_steps": 1, "total_steps": 3},
        "runtime": {
            "device": "cpu",
            "dtype": "fp32",
            "seed": 2026,
            "resume": True,
        },
        "log": {"out_dir": str(production_path), "ckpt_every": 2},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, production_path


def _verified_synth_source(tmp_path: Path, *, split: str, rows: int) -> dict:
    output = tmp_path / f"verified-{split}.jsonl"
    generator_config = tmp_path / f"verified-{split}.yaml"
    generator_config.write_text(
        yaml.safe_dump(
            {
                "out": str(output),
                "n_samples": rows,
                "seed": 17 if split == "train" else 909,
                "level": 5,
                "split": split,
                "generator": {"backend": "deterministic_templates"},
                "complexity": {"multi_turn": 0},
                "irrelevance_fraction": 0.2,
                "verification": {"rule_based": True, "model_based": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    synthesize(str(generator_config))
    legacy_manifest = output.with_suffix(output.suffix + ".manifest.json")
    versioned_manifest = output.with_suffix(output.suffix + ".manifest.v1.json")
    versioned_manifest.write_bytes(legacy_manifest.read_bytes())
    return {
        "path": str(output),
        "artifact": {
            "generator_config": str(generator_config),
            "manifest": str(versioned_manifest),
            "expected_split": split,
            "expected_rule_verified": True,
            "environment_policy": "forbid",
        },
    }


def _write_fake_rl_outputs(
    effective_config_path: str,
    *,
    execution_rollout_step_limit: int,
    production_mutation: Path | None = None,
    mutate_policy: bool = True,
    learning_rate_drift: bool = False,
    execution_drift: bool = False,
) -> None:
    effective = yaml.safe_load(Path(effective_config_path).read_text(encoding="utf-8"))
    out_dir = Path(effective["log"]["out_dir"])
    out_dir.mkdir(parents=True)
    model_config = ModelConfig.from_yaml(effective["model_config"])
    assert execution_rollout_step_limit == 2
    expected_learning_rates = [0.0, float(effective["optim"]["lr"])]
    actual_learning_rates = list(expected_learning_rates)
    if learning_rate_drift:
        actual_learning_rates[-1] /= 2
    tokenizer = tokenizer_identity("byte", vocab_size=model_config.vocab_size)
    strict = effective["data"].get("strict_conversation_artifacts", False)
    if strict:
        loaded_train = load_conversation_source(
            effective["data"]["conversations"][0],
            require_verified=True,
            expected_split="train",
        )
        loaded_eval = load_conversation_source(
            effective["data"]["eval_conversations"][0],
            require_verified=True,
            expected_split="eval",
        )
        train_path = loaded_train.path
        eval_path = loaded_eval.path
        train_identity = dict(loaded_train.identity)
        eval_identity = dict(loaded_eval.identity)
    else:
        train_path = Path(effective["data"]["conversations"][0])
        eval_path = Path(effective["data"]["eval_conversations"][0])
        train_identity = file_identity(train_path)
        eval_identity = file_identity(eval_path)
    data = {
        "paths": [str(train_path)],
        "eval_paths": [str(eval_path)],
        "train_artifacts": [{"path": str(train_path), **train_identity}],
        "eval_artifacts": [{"path": str(eval_path), **eval_identity}],
        "split_audit": {
            "train_dataset_sha256": "1" * 64,
            "eval_dataset_sha256": "2" * 64,
            "train_scored_rows_sha256": "3" * 64,
            "eval_scored_rows_sha256": "4" * 64,
            "train_scored_prompts_sha256": "5" * 64,
            "eval_scored_prompts_sha256": "6" * 64,
            "row_overlap": 0,
            "prompt_overlap": 0,
        },
        "selected_eval_split_audit": {
            "train_dataset_sha256": "1" * 64,
            "eval_dataset_sha256": "2" * 64,
            "train_scored_rows_sha256": "3" * 64,
            "eval_scored_rows_sha256": "4" * 64,
            "train_scored_prompts_sha256": "5" * 64,
            "eval_scored_prompts_sha256": "6" * 64,
            "row_overlap": 0,
            "prompt_overlap": 0,
        },
    }
    coverage = effective["evaluation"].get("preflight_minimum_coverage")
    if coverage is not None:
        data["eval_selection"] = coverage["selection_audit"]
        data["preflight_minimum_coverage"] = coverage
    lineage_config = copy.deepcopy(effective)
    lineage_config["runtime"].pop("resume", None)
    lineage_data = {
        key: data[key]
        for key in (
            "train_artifacts",
            "eval_artifacts",
            "split_audit",
            "selected_eval_split_audit",
        )
    }
    for optional in ("eval_selection", "preflight_minimum_coverage"):
        if optional in data:
            lineage_data[optional] = data[optional]
    lineage = {
        "version": 1,
        "stage": "rl",
        "config_sha256": canonical_sha256(lineage_config),
        "model_config_sha256": canonical_sha256(model_config.__dict__),
        "data_sha256": canonical_sha256(lineage_data),
        "tokenizer_sha256": tokenizer["sha256"],
        "parent_checkpoint_sha256": hashlib.sha256(
            Path(effective["init_from"]).read_bytes()
        ).hexdigest(),
        "git": None,
    }
    prompt_accounting = {
        "selected_prompts": 2,
        "selected_prompt_tokens": 6,
        "rollout_prompt_tokens": 12,
        "generated_tokens": 6,
        "generated_eos_tokens": 2,
        "truncated_rollouts": 0,
        "informative_steps": 2,
        "informative_scoring_input_slots": 14,
        "model_forward_token_slots": {
            "phases": {
                "rollout_prefill": 12,
                "rollout_cached_decode": 4,
                "old_policy_scoring": 14,
                "reference_policy_scoring": 0,
                "current_policy_optimization": 14,
            },
            "total": 44,
        },
        "sample_draws": [2],
    }
    rollout_observability = {
        "reward": {
            "distribution": [
                {"reward": 0.0, "reward_hex": "0x0.0p+0", "count": 2},
                {"reward": 1.0, "reward_hex": "0x1.0000000000000p+0", "count": 2},
            ],
            "unique_values": 2,
            "exact_success_rollouts": 1,
        },
        "parsing": {
            "parser_format_valid_rollouts": 4,
            "complete_parser_format_valid_rollouts": 4,
            "parser_tool_syntax_rollouts": 0,
            "tool_reward_rollouts": 0,
            "text_reward_rollouts": 4,
            "strict_tool_format_valid_rollouts": 0,
        },
        "truncation": {"truncated_rollouts": 0},
        "tokens": {
            "selected_prompt_tokens": 6,
            "rollout_prompt_tokens": 12,
            "generated_tokens": 6,
            "generated_eos_tokens": 2,
            "model_forward_token_slots": prompt_accounting["model_forward_token_slots"],
        },
    }
    accounting = {
        "attempted_rollout_steps": 2,
        "attempted_groups": 2,
        "attempted_rollouts": 4,
        "zero_signal_steps": 0,
        "informative_groups": 2,
        "realized_optimizer_updates": 2,
        "policy_epochs_per_informative_batch": 1,
        "generated_tokens": 6,
        "generated_eos_tokens": 2,
        "truncated_rollouts": 0,
        "informative_scoring_input_slots": 14,
        "model_forward_token_slots": prompt_accounting["model_forward_token_slots"],
        "learning_rate_history": actual_learning_rates,
        "fixed_horizon_progress": {
            "planned_rollout_steps": 3,
            "completed_rollout_steps": 2,
            "execution_rollout_step_limit": 2,
            "bounded_prefix": True,
        },
        "rollout_observability": rollout_observability,
    }
    parent = torch.load(
        effective["init_from"],
        map_location="cpu",
        weights_only=True,
    )
    child_state = {
        name: tensor.clone() for name, tensor in parent["state_dict"].items()
    }
    if mutate_policy:
        first_name = next(iter(child_state))
        first_tensor = child_state[first_name]
        first_tensor.reshape(-1)[0] += 1
    execution = {
        "requested_device": "mps" if execution_drift else "cpu",
        "resolved_device": "mps" if execution_drift else "cpu",
        "requested_dtype": "fp32",
        "resolved_dtype": "fp32",
    }
    checkpoint_path = out_dir / "latest.pt"
    checkpoint = {
        "stage": "rl",
        "step": 1,
        "state_dict": child_state,
        "optimizer": {
            "param_groups": [{"lr": actual_learning_rates[-1]}],
        },
        "lineage": lineage,
        "tokenizer": {
            "kind": "byte",
            "path": None,
            "sha256": tokenizer["sha256"],
        },
        "data": data,
        "prompt_accounting": prompt_accounting,
        "rl_accounting": accounting,
        "execution": execution,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics = {
        "stage": "rl",
        "checkpoint": str(checkpoint_path),
        "reward_steps": 2,
        "mean_reward_last": 0.5,
        "lineage": lineage,
        "execution": execution,
        "data": data,
        "prompt_accounting": prompt_accounting,
        "rl_accounting": accounting,
        "heldout_eval": {},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if production_mutation is not None:
        production_mutation.mkdir(parents=True)
        (production_mutation / "latest.pt").write_bytes(b"forbidden production mutation")


def test_run_one_update_rl_preflight_seals_lineage_and_observability(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, production_path = _write_rl_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda config_path, *, resume, _execution_rollout_step_limit: (
            _write_fake_rl_outputs(
                config_path,
                execution_rollout_step_limit=_execution_rollout_step_limit,
            )
        ),
    )
    work_dir = tmp_path / "isolated"
    receipt_path = tmp_path / "receipt.json"

    receipt = run_one_update_rl_preflight(
        config_path,
        work_dir=work_dir,
        receipt_path=receipt_path,
        device="cpu",
    )

    assert receipt["status"] == "passed"
    assert_preflight_receipt(receipt)
    assert not production_path.exists()
    assert receipt["source"]["production_rl_output_untouched"] is True
    assert receipt["source"]["source_artifacts_untouched"] is True
    contract = receipt["effective"]["contract"]
    assert contract["rollout_steps"] == 2
    assert contract["execution_rollout_step_limit"] == 2
    assert contract["production_schedule_total_steps"] == 3
    assert contract["first_nonzero_learning_rate_step"] == 1
    assert contract["expected_learning_rates"] == [0.0, 2.0e-5]
    assert contract["realized_optimizer_updates"] == 2
    observation = receipt["measurement"]["rollout_observability"]
    assert observation["reward"]["unique_values"] == 2
    assert observation["parsing"]["parser_format_valid_rollouts"] == 4
    assert observation["truncation"]["truncated_rollouts"] == 0
    assert observation["tokens"]["generated_tokens"] == 6
    transition = receipt["measurement"]["policy_transition"]
    assert transition["at_least_one_policy_tensor_changed"] is True
    assert transition["changed_model_parameter_count"] == 1
    assert transition["actual_learning_rates"] == [0.0, 2.0e-5]
    assert transition["actual_learning_rates_match_expected"] is True
    assert transition["nonzero_learning_rate_executed"] is True
    assert transition["final_optimizer_learning_rate_matches_expected"] is True
    assert receipt_path.read_text(encoding="utf-8").endswith("\n")


def test_rl_preflight_derives_verified_selector_minimum_instead_of_one_row(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, production_path = _write_rl_preflight_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"] = {
        "strict_conversation_artifacts": True,
        "conversations": [_verified_synth_source(tmp_path, split="train", rows=18)],
        "eval_conversations": [_verified_synth_source(tmp_path, split="eval", rows=24)],
        "tokenizer": {"kind": "byte"},
    }
    config["evaluation"] = {
        "max_conversations": 24,
        "selection": STRATIFIED_EVAL_ALGORITHM,
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    derivation = derive_rl_eval_minimum_coverage(config)
    assert derivation is not None
    assert derivation["minimum_coverage_rows"] > 1
    loaded_eval = load_conversation_source(
        config["data"]["eval_conversations"][0],
        require_verified=True,
        expected_split="eval",
    )
    with pytest.raises(InsufficientStratumCapacityError) as one_row:
        select_stratified_eval_subset(loaded_eval.conversations, max_rows=1)
    assert one_row.value.required_rows == derivation["minimum_coverage_rows"]
    exact_selection = select_stratified_eval_subset(
        loaded_eval.conversations,
        max_rows=derivation["minimum_coverage_rows"],
    )
    assert exact_selection.audit.as_dict() == derivation["selection_audit"]
    assert derivation["selection_audit"]["capacity"] == {
        "max_rows": derivation["minimum_coverage_rows"],
        "coverage_rows": derivation["minimum_coverage_rows"],
        "fill_rows": 0,
    }
    assert derivation["mandatory_strata"] > derivation["minimum_coverage_rows"]

    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda config_path, *, resume, _execution_rollout_step_limit: (
            _write_fake_rl_outputs(
                config_path,
                execution_rollout_step_limit=_execution_rollout_step_limit,
            )
        ),
    )
    receipt = run_one_update_rl_preflight(
        config_path,
        work_dir=tmp_path / "coverage-isolated",
        receipt_path=tmp_path / "coverage-receipt.json",
        device="cpu",
    )

    assert receipt["status"] == "passed"
    assert_preflight_receipt(receipt)
    effective_evaluation = receipt["effective"]["config_payload"]["evaluation"]
    assert (
        effective_evaluation["max_conversations"]
        == derivation["minimum_coverage_rows"]
    )
    assert effective_evaluation["preflight_minimum_coverage"] == derivation
    assert (
        receipt["source"]["evaluation_minimum_coverage_derivation"]
        == derivation
    )
    contract = receipt["effective"]["contract"]["heldout_minimum_coverage"]
    assert contract == {
        "derivation_sha256": derivation["derivation_sha256"],
        "minimum_coverage_rows": derivation["minimum_coverage_rows"],
        "mandatory_strata": derivation["mandatory_strata"],
        "selection_audit_sha256": derivation["selection_audit"]["audit_sha256"],
    }
    assert (
        receipt["metrics"]["data"]["preflight_minimum_coverage"]
        == derivation
    )
    assert not production_path.exists()


def test_run_one_update_rl_preflight_fails_if_production_output_changes(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, production_path = _write_rl_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda config_path, *, resume, _execution_rollout_step_limit: (
            _write_fake_rl_outputs(
                config_path,
                execution_rollout_step_limit=_execution_rollout_step_limit,
                production_mutation=production_path,
            )
        ),
    )
    receipt_path = tmp_path / "failed-receipt.json"

    with pytest.raises(RuntimeError, match="RL preflight failed"):
        run_one_update_rl_preflight(
            config_path,
            work_dir=tmp_path / "isolated",
            receipt_path=receipt_path,
            device="cpu",
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert_preflight_receipt(receipt)
    assert receipt["status"] == "failed"
    assert receipt["error"]["type"] == "ProductionRLOutputMutation"
    assert receipt["source"]["production_rl_output_untouched"] is False


@pytest.mark.parametrize(
    ("fake_kwargs", "validation_error"),
    [
        (
            {"mutate_policy": False},
            "isolated RL nonzero-LR prefix changed no policy model tensor",
        ),
        (
            {"learning_rate_drift": True},
            "isolated RL actual learning-rate sequence drifted from production",
        ),
        (
            {"execution_drift": True},
            "isolated RL execution requested_device mismatch",
        ),
    ],
)
def test_run_one_update_rl_preflight_requires_real_production_schedule_update(
    tmp_path,
    monkeypatch,
    fake_kwargs,
    validation_error,
) -> None:
    config_path, production_path = _write_rl_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda config_path, *, resume, _execution_rollout_step_limit: (
            _write_fake_rl_outputs(
                config_path,
                execution_rollout_step_limit=_execution_rollout_step_limit,
                **fake_kwargs,
            )
        ),
    )
    receipt_path = tmp_path / "failed-update-receipt.json"

    with pytest.raises(RuntimeError, match="RL preflight failed"):
        run_one_update_rl_preflight(
            config_path,
            work_dir=tmp_path / "isolated",
            receipt_path=receipt_path,
            device="cpu",
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert_preflight_receipt(receipt)
    assert receipt["status"] == "failed"
    assert validation_error in receipt["validation_errors"]
    assert receipt["error"]["type"] == "RLPreflightValidationError"
    assert not production_path.exists()


@pytest.mark.parametrize("preexisting", ["work", "receipt", "receipt_tmp"])
def test_run_one_update_rl_preflight_fails_closed_on_preexisting_destination(
    tmp_path,
    monkeypatch,
    preexisting,
) -> None:
    config_path, _ = _write_rl_preflight_fixture(tmp_path)
    work_dir = tmp_path / "rl-isolated"
    receipt_path = tmp_path / "rl-receipt.json"
    if preexisting == "work":
        work_dir.mkdir()
    elif preexisting == "receipt":
        receipt_path.write_text("occupied", encoding="utf-8")
    else:
        receipt_path.with_suffix(".json.tmp").write_text(
            "occupied",
            encoding="utf-8",
        )
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.rl.run", forbidden_run)
    with pytest.raises(FileExistsError):
        run_one_update_rl_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=receipt_path,
            device="cpu",
        )
    assert called is False


def test_run_one_update_rl_preflight_rejects_receipt_inside_work_directory(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _ = _write_rl_preflight_fixture(tmp_path)
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.rl.run", forbidden_run)
    work_dir = tmp_path / "isolated"
    with pytest.raises(ValueError, match="must be disjoint"):
        run_one_update_rl_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=work_dir / "receipt.json",
            device="cpu",
        )
    assert called is False
    assert not work_dir.exists()


def test_run_one_update_sft_preflight_seals_exact_update_and_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, production_path, parent_path = _write_sft_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path
        ),
    )
    receipt_path = tmp_path / "sft-receipt.json"

    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=receipt_path,
        device="cpu",
    )

    assert receipt["status"] == "passed"
    assert_preflight_receipt(receipt)
    assert not production_path.exists()
    assert receipt["source"]["production_sft_output_untouched"] is True
    assert receipt["source"]["source_artifacts_untouched"] is True
    assert receipt["source"]["sft_parent_checkpoint_untouched"] is True
    assert receipt["source"]["data_artifacts_untouched"] is True
    assert (
        receipt["source"]["sft_parent_checkpoint"]["sha256"]
        == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    )
    assert (
        receipt["source"]["sft_data_lineage"]["sha256"]
        == receipt["metrics"]["lineage"]["data_sha256"]
    )
    contract = receipt["effective"]["contract"]
    assert contract["optimizer_updates"] == contract["realized_optimizer_updates"] == 1
    assert contract["optimizer_parameter_step_values"] == [1]
    assert contract["optimizer_learning_rates"] == [0.0]
    assert contract["expected_step_zero_learning_rate"] == 0.0
    assert contract["micro_batch_size"] == 2
    assert contract["grad_accum_steps"] == 3
    assert contract["effective_batch_size"] == 6
    assert contract["source_order_prefix_rows"] == 6
    assert contract["lm_sampling"] == {"mode": "source_order_wrapping_v1"}
    assert contract["parent_pins"] == _sft_parent_pins(parent_path)
    assert receipt["source"]["sft_parent_checkpoint"]["completion"] == {
        **_sft_parent_pins(parent_path),
        "seal_configured": False,
        "step": 0,
        "planned_steps": 1,
        "micro_batch_size": 2,
        "grad_accum_steps": 3,
        "completed_microbatches": 3,
    }
    parameter_scope = contract["model_parameter_scope"]
    assert parameter_scope["configured_frozen_model_parameter_names"] == []
    assert parameter_scope["expected_trainable_model_tensor_count"] == len(
        parameter_scope["model_named_parameter_names"]
    )
    assert parameter_scope["frozen_model_tensors_exactly_preserved"] is True
    assert parameter_scope["unfrozen_model_transition_required"] is False
    assert parameter_scope["first_changed_unfrozen_model_parameter"] is None
    assert receipt["metrics"]["token_accounting"]["sources"][
        str(tmp_path / "train.jsonl")
    ]["rows"] == 6
    assert receipt["artifacts"]["checkpoint"]["sha256"] == file_identity(
        tmp_path / "sft-isolated" / "run" / "latest.pt"
    )["sha256"]
    assert (
        receipt["artifacts"]["checkpoint"]["resume_integrity_sha256"]
        == torch.load(
            tmp_path / "sft-isolated" / "run" / "latest.pt",
            map_location="cpu",
            weights_only=True,
        )["resume_integrity_sha256"]
    )
    assert receipt["artifacts"]["metrics"]["sha256"] == file_identity(
        tmp_path / "sft-isolated" / "run" / "metrics.json"
    )["sha256"]
    assert receipt["measurement"]["wall_seconds"] >= 0
    assert receipt["measurement"]["memory"]["peak_process_rss_bytes"] > 0
    assert "not learning evidence" in receipt["measurement"]["interpretation"]
    assert (
        receipt["measurement"]["non_learning_limitation"]
        == parameter_scope["non_learning_limitation"]
    )
    assert "all bounded executed learning rates are zero" in (
        receipt["measurement"]["non_learning_limitation"]
    )
    assert "first effective-batch prefix" in receipt["measurement"]["sampling_limitation"]
    assert receipt_path.read_text(encoding="utf-8").endswith("\n")


def test_sft_preflight_validates_and_records_exact_parent_seal_before_run(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, parent_path = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["continuation"]["parent"] = _sft_parent_pins(parent_path)
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path
        ),
    )

    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=tmp_path / "sft-receipt.json",
        device="cpu",
    )

    completion = receipt["source"]["sft_parent_checkpoint"]["completion"]
    assert completion["seal_configured"] is True
    assert {key: completion[key] for key in source["continuation"]["parent"]} == (
        source["continuation"]["parent"]
    )
    assert receipt["effective"]["contract"]["parent_pins"] == source[
        "continuation"
    ]["parent"]


def test_sft_preflight_rejects_parent_pin_mismatch_before_run(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, parent_path = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["continuation"]["parent"] = _sft_parent_pins(parent_path)
    source["continuation"]["parent"]["completed_lm_cursor"] += 1
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.sft.run", forbidden_run)
    with pytest.raises(ValueError, match="completed_lm_cursor mismatch"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=tmp_path / "sft-receipt.json",
            device="cpu",
        )

    assert called is False
    assert not (tmp_path / "sft-isolated").exists()


def test_parent_anchored_sft_preflight_requires_declared_parent_seal(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["data"]["sampling"] = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE
    }
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.sft.run", forbidden_run)
    with pytest.raises(
        ValueError,
        match="parent-anchored SFT preflight requires continuation.parent",
    ):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=tmp_path / "sft-receipt.json",
            device="cpu",
        )

    assert called is False
    assert not (tmp_path / "sft-isolated").exists()


def test_parent_anchored_sft_preflight_rejects_semantic_parent_drift_before_run(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, parent_path = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["continuation"]["parent"] = _sft_parent_pins(parent_path)
    source["data"]["sampling"] = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "parent_prefix_decisions": 6,
        "update_decisions": 6,
        "expected_parent_order_sha256": "b" * 64,
    }
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.sft.run", forbidden_run)
    with pytest.raises(ValueError, match="parent.*order.*mismatch"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=tmp_path / "sft-receipt.json",
            device="cpu",
        )

    assert called is False
    assert not (tmp_path / "sft-isolated").exists()


def test_sft_preflight_proves_freeze_and_optimizer_model_scope(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["optim"]["freeze_parameters"] = ["embed.weight", "in_proj.weight"]
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path
        ),
    )

    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=tmp_path / "sft-receipt.json",
        device="cpu",
    )

    scope = receipt["effective"]["contract"]["model_parameter_scope"]
    assert scope["configured_frozen_model_parameter_names"] == [
        "embed.weight",
        "in_proj.weight",
    ]
    assert scope["compared_frozen_model_parameter_names"] == [
        "embed.weight",
        "in_proj.weight",
    ]
    assert scope["frozen_model_tensors_exactly_preserved"] is True
    assert scope["expected_optimizer_model_parameter_names"] == [
        name
        for name in scope["model_named_parameter_names"]
        if name not in {"embed.weight", "in_proj.weight"}
    ]
    isolated = torch.load(
        tmp_path / "sft-isolated" / "run" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert isolated["training_contract"]["optimizer_model_parameter_names"] == (
        scope["expected_optimizer_model_parameter_names"]
    )
    assert len(isolated["optimizer"]["param_groups"][0]["params"]) == scope[
        "expected_trainable_model_tensor_count"
    ]
    assert scope["training_contract_frozen_model_parameter_names"] == [
        "embed.weight",
        "in_proj.weight",
    ]
    assert scope["training_contract_optimizer_model_parameter_names"] == scope[
        "expected_optimizer_model_parameter_names"
    ]
    assert scope["all_auxiliary_heads_disabled"] is True
    assert scope["optimizer_param_group_parameter_count"] == scope[
        "expected_trainable_model_tensor_count"
    ]
    assert scope["optimizer_parameter_count_matches_expected"] is True


def test_sft_preflight_requires_unfrozen_transition_at_nonzero_step_zero_lr(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["schedule"]["warmup_steps"] = 0
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path
        ),
    )

    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=tmp_path / "sft-receipt.json",
        device="cpu",
    )

    scope = receipt["effective"]["contract"]["model_parameter_scope"]
    assert scope["step_zero_learning_rate"] == source["optim"]["lr"]
    assert scope["unfrozen_model_transition_required"] is True
    assert scope["first_changed_unfrozen_model_parameter"] is not None
    assert scope["non_learning_limitation"] is None
    assert receipt["measurement"]["non_learning_limitation"] is None


def test_parent_anchored_sft_preflight_executes_through_first_pulse(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, parent_path = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["continuation"]["parent"] = _sft_parent_pins(parent_path)
    source["schedule"]["total_steps"] = 12
    source["schedule"]["warmup_steps"] = 24
    source["optim"]["freeze_parameters"] = ["embed.weight"]
    source["data"]["sampling"] = {
        "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
        "parent_prefix_decisions": 6,
        "update_decisions": 6,
        "expected_parent_order_sha256": "a" * 64,
    }
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    real_derive = _derive_sft_data_identity_and_sampling

    def derive_with_parent_pulse(config):
        identity_source = copy.deepcopy(config)
        identity_source["data"].pop("sampling")
        identity, _ = real_derive(identity_source)
        effective_batch = (
            int(config["batch"]["micro_batch_size"])
            * int(config["batch"]["grad_accum_steps"])
        )
        planned_updates = int(config["schedule"]["total_steps"])
        selected_decisions = planned_updates * effective_batch
        sampling_contract = {
            "mode": PARENT_ANCHORED_FORMAT_PULSE_SAMPLING_MODE,
            "no_replacement": True,
            "selected_decisions": selected_decisions,
            "update_layout": {
                "update_decisions": effective_batch,
                "total_updates": planned_updates,
                "pulse_positions_index_base": 0,
                "pulse_positions_zero_based": [7, 10],
            },
        }
        identity["decision_sampling"] = sampling_contract
        execution = _derive_sft_preflight_execution_contract(config, identity)
        decision_keys = [
            [index, 1] for index in range(execution["executed_lm_decisions"])
        ]
        evidence = {
            "kind": "localagent_sft_preflight_mixed_replay_prefix",
            "schema_version": 1,
            "production": {
                "selected_decisions": selected_decisions,
                "sampling_contract": sampling_contract,
                "sampling_contract_sha256": canonical_sha256(sampling_contract),
            },
            "exercised_prefix": {
                "decisions": execution["executed_lm_decisions"],
                "decision_keys": decision_keys,
                "decision_keys_sha256": canonical_sha256(decision_keys),
                "equals_production_order_prefix": True,
            },
            "bounded_execution": execution,
        }
        return identity, evidence

    monkeypatch.setattr(
        "localagent.train.update_preflight._derive_sft_data_identity_and_sampling",
        derive_with_parent_pulse,
    )
    observed_limits = []

    def fake_run(config_path, *, resume, _max_optimizer_updates):
        assert resume is False
        observed_limits.append(_max_optimizer_updates)
        _write_fake_sft_outputs(config_path)

    monkeypatch.setattr("localagent.train.sft.run", fake_run)
    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=tmp_path / "sft-receipt.json",
        device="cpu",
    )

    assert observed_limits == [8]
    contract = receipt["effective"]["contract"]
    assert contract["execution_optimizer_update_limit"] == 8
    assert contract["first_pulse_update_zero_based"] == 7
    assert contract["executed_through_first_pulse"] is True
    assert contract["executed_lm_decisions"] == 48
    assert contract["expected_completed_lm_cursor"] == 48
    assert contract["observed_completed_lm_cursor"] == 48
    expected_parent_anchor_binding = {
        "parent_lm_sampling_mode": QUOTA_SAMPLING_MODE,
        "parent_no_replacement": True,
        "parent_order_sha256": "a" * 64,
        "parent_completed_steps": 1,
        "parent_completed_lm_cursor": 6,
        "parent_update_decisions": 6,
    }
    assert contract["parent_anchor_binding"] == expected_parent_anchor_binding
    assert receipt["source"]["sft_parent_checkpoint"]["completion"][
        "parent_anchor_binding"
    ] == expected_parent_anchor_binding
    expected_sampling = receipt["source"]["sft_data_lineage"]["identity"][
        "decision_sampling"
    ]
    assert (
        expected_sampling["parent_checkpoint_binding"]
        == expected_parent_anchor_binding
    )
    sampling_production = receipt["source"]["sft_sampling_lineage"]["production"]
    assert sampling_production["sampling_contract"] == expected_sampling
    assert sampling_production["sampling_contract_sha256"] == canonical_sha256(
        expected_sampling
    )
    assert receipt["metrics"]["lm_sampling"] == expected_sampling
    assert contract["optimizer_updates"] == 8
    assert contract["realized_optimizer_updates"] == 8
    assert contract["optimizer_parameter_step_values"] == [8]
    assert len(contract["expected_executed_learning_rates"]) == 8
    assert contract["expected_step_zero_learning_rate"] == 0.0
    assert contract["expected_last_executed_learning_rate"] > 0.0
    assert contract["any_executed_learning_rate_nonzero"] is True
    assert contract["exercised_lm_prefix"]["decisions"] == 48
    assert len(contract["exercised_lm_prefix"]["decision_keys"]) == 48
    scope = contract["model_parameter_scope"]
    assert scope["frozen_model_tensors_exactly_preserved"] is True
    assert scope["execution_optimizer_update_limit"] == 8
    assert scope["any_executed_learning_rate_nonzero"] is True
    assert scope["last_executed_learning_rate"] > 0.0
    assert scope["first_changed_unfrozen_model_parameter"] is not None
    assert scope["expected_completed_lm_cursor"] == 48
    assert scope["observed_completed_lm_cursor"] == 48
    assert scope["completed_lm_cursor_matches_expected"] is True
    assert receipt["measurement"]["non_learning_limitation"] is None

    isolated = torch.load(
        tmp_path / "sft-isolated" / "run" / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert isolated["step"] == 7
    assert len(isolated["loss_history"]) == 8
    assert isolated["fixed_horizon_progress"] == {
        "planned_optimizer_updates": 12,
        "completed_optimizer_updates": 8,
        "partial": True,
    }
    assert isolated["sampling_state"]["completed_steps"] == 8
    assert isolated["sampling_state"]["completed_microbatches"] == 24
    assert isolated["sampling_state"]["lm_cursor"] == 48


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("frozen_tensor", "changed configured frozen model tensors"),
        (
            "optimizer_names",
            "optimizer model parameter names do not equal model named_parameters",
        ),
        ("optimizer_count", "optimizer parameter count does not match"),
        ("optimizer_name", "optimizer kind does not match optim.name"),
        ("weight_decay", "optimizer weight_decay does not match optim.weight_decay"),
        ("grad_clip", "optimizer grad_clip does not match optim.grad_clip"),
        (
            "no_unfrozen_transition",
            "bounded prefix includes a nonzero learning rate but changed no unfrozen model tensor",
        ),
    ],
)
def test_sft_preflight_rejects_false_freeze_optimizer_or_learning_evidence(
    tmp_path,
    monkeypatch,
    mutation,
    expected_error,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["optim"]["freeze_parameters"] = ["embed.weight"]
    if mutation == "no_unfrozen_transition":
        source["schedule"]["warmup_steps"] = 0
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    fake_kwargs = {
        "mutate_frozen_parameter": mutation == "frozen_tensor",
        "optimizer_model_parameter_names_override": (
            [] if mutation == "optimizer_names" else None
        ),
        "optimizer_parameter_count_delta": -1 if mutation == "optimizer_count" else 0,
        "change_unfrozen_parameter": (
            False if mutation == "no_unfrozen_transition" else None
        ),
        "optimizer_contract_override": (
            {
                "kind": "SGD" if mutation == "optimizer_name" else "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": 0.5 if mutation == "weight_decay" else 0.0,
                "grad_clip": 2.0 if mutation == "grad_clip" else 1.0,
            }
            if mutation in {"optimizer_name", "weight_decay", "grad_clip"}
            else None
        ),
    }
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path,
            **fake_kwargs,
        ),
    )
    receipt_path = tmp_path / "sft-receipt.json"

    with pytest.raises(RuntimeError, match="SFT preflight failed"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=receipt_path,
            device="cpu",
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert any(expected_error in error for error in receipt["validation_errors"])


def test_sft_preflight_receipt_binds_production_mixed_replay_and_exercised_prefix(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["data"]["sampling"] = {"mode": MIXED_REPLAY_SAMPLING_MODE}
    config_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    real_derive = _derive_sft_data_identity_and_sampling

    def derive_with_mixed_replay(config):
        identity_source = copy.deepcopy(config)
        identity_source["data"].pop("sampling")
        identity, _ = real_derive(identity_source)
        effective_batch = (
            int(config["batch"]["micro_batch_size"])
            * int(config["batch"]["grad_accum_steps"])
        )
        selected_decisions = int(config["schedule"]["total_steps"]) * effective_batch
        production_contract = {
            "mode": MIXED_REPLAY_SAMPLING_MODE,
            "no_replacement": True,
            "selected_decisions": selected_decisions,
            "selected_order_sha256": "c" * 64,
            "complete_order_sha256": "d" * 64,
        }
        decision_keys = [[index, 1] for index in range(effective_batch)]
        evidence = {
            "kind": "localagent_sft_preflight_mixed_replay_prefix",
            "schema_version": 1,
            "production": {
                "selected_decisions": selected_decisions,
                "sampling_contract": production_contract,
                "sampling_contract_sha256": canonical_sha256(production_contract),
            },
            "exercised_prefix": {
                "decisions": effective_batch,
                "decision_keys": decision_keys,
                "decision_keys_sha256": canonical_sha256(decision_keys),
                "equals_production_order_prefix": True,
            },
        }
        identity["decision_sampling"] = production_contract
        return identity, evidence

    monkeypatch.setattr(
        "localagent.train.update_preflight._derive_sft_data_identity_and_sampling",
        derive_with_mixed_replay,
    )
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path
        ),
    )
    receipt = run_one_update_sft_preflight(
        config_path,
        work_dir=tmp_path / "sft-isolated",
        receipt_path=tmp_path / "sft-receipt.json",
        device="cpu",
    )

    sampling_lineage = receipt["source"]["sft_sampling_lineage"]
    production = sampling_lineage["production"]
    exercised = sampling_lineage["exercised_prefix"]
    assert receipt["effective"]["config_payload"]["schedule"]["total_steps"] == 7
    assert production["selected_decisions"] == 42
    assert production["sampling_contract_sha256"] == canonical_sha256(
        production["sampling_contract"]
    )
    assert exercised["decisions"] == 6
    assert exercised["decision_keys_sha256"] == canonical_sha256(
        exercised["decision_keys"]
    )
    contract = receipt["effective"]["contract"]
    assert contract["execution_optimizer_update_limit"] == 1
    assert contract["production_lm_sampling_sha256"] == production[
        "sampling_contract_sha256"
    ]
    assert contract["exercised_lm_prefix"] == exercised
    assert receipt["metrics"]["fixed_horizon_progress"] == {
        "planned_optimizer_updates": 7,
        "completed_optimizer_updates": 1,
        "partial": True,
    }


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("production", "ProductionSFTOutputMutation"),
        ("parent", "SourceArtifactMutation"),
    ],
)
def test_run_one_update_sft_preflight_detects_external_mutation(
    tmp_path,
    monkeypatch,
    mutation,
    error_type,
) -> None:
    config_path, production_path, parent_path = _write_sft_preflight_fixture(tmp_path)

    def fake_run(
        config_path: str,
        *,
        resume: bool,
        _max_optimizer_updates: int,
    ) -> None:
        assert resume is False
        assert _max_optimizer_updates == 1
        _write_fake_sft_outputs(
            config_path,
            production_mutation=production_path if mutation == "production" else None,
            parent_mutation=parent_path if mutation == "parent" else None,
        )

    monkeypatch.setattr("localagent.train.sft.run", fake_run)
    receipt_path = tmp_path / "failed-sft-receipt.json"
    with pytest.raises(RuntimeError, match="SFT preflight failed"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=receipt_path,
            device="cpu",
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert_preflight_receipt(receipt)
    assert receipt["status"] == "failed"
    assert receipt["error"]["type"] == error_type
    if mutation == "production":
        assert receipt["source"]["production_sft_output_untouched"] is False
    else:
        assert receipt["source"]["sft_parent_checkpoint_untouched"] is False


def test_run_one_update_sft_preflight_rejects_second_adam_update(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        "localagent.train.sft.run",
        lambda config_path, *, resume, _max_optimizer_updates: _write_fake_sft_outputs(
            config_path,
            optimizer_step=2,
        ),
    )
    receipt_path = tmp_path / "invalid-step-receipt.json"

    with pytest.raises(RuntimeError, match="SFT preflight failed"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=tmp_path / "sft-isolated",
            receipt_path=receipt_path,
            device="cpu",
        )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert_preflight_receipt(receipt)
    assert receipt["error"]["type"] == "SFTPreflightValidationError"
    assert any(
        "exactly one optimizer update" in message
        for message in receipt["validation_errors"]
    )


@pytest.mark.parametrize("preexisting", ["work", "receipt", "receipt_tmp"])
def test_run_one_update_sft_preflight_fails_closed_on_preexisting_destination(
    tmp_path,
    monkeypatch,
    preexisting,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    work_dir = tmp_path / "sft-isolated"
    receipt_path = tmp_path / "sft-receipt.json"
    if preexisting == "work":
        work_dir.mkdir()
    elif preexisting == "receipt":
        receipt_path.write_text("occupied", encoding="utf-8")
    else:
        receipt_path.with_suffix(".json.tmp").write_text("occupied", encoding="utf-8")
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.sft.run", forbidden_run)
    with pytest.raises(FileExistsError):
        run_one_update_sft_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=receipt_path,
            device="cpu",
        )
    assert called is False


def test_run_one_update_sft_preflight_rejects_receipt_inside_work_directory(
    tmp_path,
    monkeypatch,
) -> None:
    config_path, _, _ = _write_sft_preflight_fixture(tmp_path)
    called = False

    def forbidden_run(*_args, **_kwargs) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("localagent.train.sft.run", forbidden_run)
    work_dir = tmp_path / "isolated"
    with pytest.raises(ValueError, match="must be disjoint"):
        run_one_update_sft_preflight(
            config_path,
            work_dir=work_dir,
            receipt_path=work_dir / "receipt.json",
            device="cpu",
        )
    assert called is False
    assert not work_dir.exists()
