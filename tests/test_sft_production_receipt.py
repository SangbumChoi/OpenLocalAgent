from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from localagent.model import LocalAgentLM, ModelConfig
from localagent.train.loop import cosine_lr
from localagent.train.sft import (
    _SFT_RESUME_FORMAT,
    _SFT_RESUME_VERSION,
    _resume_sha256,
    _sealed_resume_sha256,
)
from localagent.train.stage_budget import (
    canonical_plan_bytes,
    seal_stage_budget_plan,
)
from localagent.train.stage_data import canonical_sha256
from localagent.train.update_preflight import seal_preflight_receipt
from localagent.eval.sft_production_receipt import (
    PRODUCTION_CHECKPOINT_EVERY,
    PRODUCTION_FROZEN_PARAMETERS,
    PRODUCTION_TOTAL_STEPS,
    assert_sft_production_receipt,
    verify_sft_production_receipt_against_artifacts,
    verify_sft_production_receipt_bytes,
    verify_sft_production_run,
    write_sft_production_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT_ORDER_SHA256 = "8" * 64
PROMPT_CONTRACT = "openai_full_catalog_v1"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_resume_payload(
    *,
    cfg: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    optimizer: dict[str, Any],
    step: int,
    loss_history: list[float],
    dataset_accounting: dict[str, Any],
    token_accounting: dict[str, Any],
    training_contract: dict[str, Any],
    lineage: dict[str, Any] | None,
    tokenizer: dict[str, Any] | None,
    data: dict[str, Any] | None,
    execution: dict[str, Any] | None,
    heldout_baseline: dict[str, Any] | None,
    lm_cursor: int,
    accum_steps: int,
) -> dict[str, Any]:
    completed_steps = step + 1
    payload = {
        "resume_format": _SFT_RESUME_FORMAT,
        "resume_version": _SFT_RESUME_VERSION,
        "cfg": cfg,
        "state_dict": state_dict,
        "tool_head": None,
        "ptr_head": None,
        "optimizer": optimizer,
        "grad_scaler": None,
        "step": step,
        "loss_history": loss_history,
        "dataset_token_accounting": dataset_accounting,
        "token_accounting": token_accounting,
        "token_accounting_scope": "language_model_microbatches",
        "sampling_state": {
            "rng_state": (3, (1, 2, 3), None),
            "lm_cursor": lm_cursor,
            "completed_steps": completed_steps,
            "completed_microbatches": completed_steps * accum_steps,
        },
        "torch_rng_state": torch.arange(8, dtype=torch.uint8),
        "cuda_rng_state_all": None,
        "mps_rng_state": None,
        "xpu_rng_state_all": None,
        "stage": "sft",
        "training_seed": 2026,
        "training_contract": training_contract,
        "lineage": lineage,
        "conversation_prompt_contract": PROMPT_CONTRACT,
        "tokenizer": tokenizer,
        "data": data,
        "execution": execution,
        "heldout_baseline": heldout_baseline,
    }
    payload["resume_integrity_sha256"] = _sealed_resume_sha256(payload)
    return payload


def _optimizer(
    state_dict: dict[str, torch.Tensor],
    trainable_names: list[str],
    completed_steps: int,
) -> dict[str, Any]:
    state = {}
    for parameter_id, name in enumerate(trainable_names):
        state[parameter_id] = {
            "step": torch.tensor(float(completed_steps)),
            "exp_avg": torch.zeros_like(state_dict[name]),
            "exp_avg_sq": torch.ones_like(state_dict[name]),
        }
    learning_rate = cosine_lr(
        completed_steps - 1,
        PRODUCTION_TOTAL_STEPS,
        1.0e-6,
        24,
        0.1,
    )
    probe = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    group = torch.optim.AdamW(
        [probe],
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    ).state_dict()["param_groups"][0]
    group["params"] = list(range(len(trainable_names)))
    return {"state": state, "param_groups": [group]}


def _accounting(completed_steps: int) -> dict[str, Any]:
    tokens = 16 * completed_steps
    return {
        "input_tokens": tokens,
        "loss_tokens": tokens,
        "sources": {
            "fixture-train.jsonl": {
                "input_tokens": tokens,
                "loss_tokens": tokens,
                "rows": tokens,
            }
        },
    }


def _plan_total_accounting() -> dict[str, Any]:
    tokens = 16 * PRODUCTION_TOTAL_STEPS
    return {
        "updates": PRODUCTION_TOTAL_STEPS,
        "input_tokens": tokens,
        "loss_tokens": tokens,
        "sources": {
            "fixture-train.jsonl": {
                "draws": tokens,
                "rows": tokens,
                "input_tokens": tokens,
                "loss_tokens": tokens,
            }
        },
    }


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    torch.save(checkpoint, path)


def _reseal_checkpoint(path: Path, mutate) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    mutate(checkpoint)
    checkpoint["resume_integrity_sha256"] = _sealed_resume_sha256(checkpoint)
    torch.save(checkpoint, path)


@pytest.fixture
def production_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    model_config = ModelConfig(
        name="sft-production-receipt-fixture",
        vocab_size=256,
        d_model=8,
        embed_dim=4,
        n_layers=1,
        n_loops=2,
        n_heads=1,
        n_kv_heads=1,
        ffn_hidden=16,
        max_seq_len=64,
        tie_embeddings=True,
    )
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(model_config.__dict__, sort_keys=False),
        encoding="utf-8",
    )
    model = LocalAgentLM(model_config)
    parent_state = {
        name: torch.zeros_like(tensor.detach().cpu())
        for name, tensor in model.state_dict().items()
    }
    model_names = list(parent_state)
    optimizer_names = [
        name for name in model_names if name not in PRODUCTION_FROZEN_PARAMETERS
    ]

    parent_lm_sampling = {
        "mode": "quota_stratified_no_replacement_v1",
        "no_replacement": True,
        "ordering": {"order_sha256": PARENT_ORDER_SHA256},
    }
    parent_training_contract = {
        "steps": 348,
        "batch_size": 2,
        "accum_steps": 8,
        "lm_sampling": parent_lm_sampling,
    }
    parent_payload = _base_resume_payload(
        cfg=model_config.__dict__,
        state_dict=parent_state,
        optimizer={"state": {}, "param_groups": []},
        step=347,
        loss_history=[1.0] * 348,
        dataset_accounting={"main": _accounting(348), "decay": None},
        token_accounting=_accounting(348),
        training_contract=parent_training_contract,
        lineage=None,
        tokenizer=None,
        data=None,
        execution=None,
        heldout_baseline=None,
        lm_cursor=5_568,
        accum_steps=8,
    )
    parent_path = tmp_path / "parent.pt"
    _write_checkpoint(parent_path, parent_payload)
    parent_sha256 = _sha256_file(parent_path)
    parent_pins = {
        "checkpoint_sha256": parent_sha256,
        "resume_integrity_sha256": parent_payload["resume_integrity_sha256"],
        "training_contract_sha256": canonical_sha256(parent_training_contract),
        "lm_sampling_sha256": canonical_sha256(parent_lm_sampling),
        "completed_steps": 348,
        "completed_lm_cursor": 5_568,
    }
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    fixture_train_path = tmp_path / "fixture-train.jsonl"
    fixture_eval_path = tmp_path / "fixture-eval.jsonl"
    tokenizer_path = tmp_path / "tokenizer.json"
    fixture_train_path.write_text("{}\n", encoding="utf-8")
    fixture_eval_path.write_text("{}\n", encoding="utf-8")
    tokenizer_path.write_text('{"fixture":"byte-tokenizer"}\n', encoding="utf-8")
    tokenizer_sha256 = _sha256_file(tokenizer_path)
    config = {
        "stage": "sft",
        "model_config": str(model_config_path),
        "init_from": str(parent_path),
        "continuation": {
            "mode": "fresh_optimizer_sft_child_v1",
            "parent": parent_pins,
        },
        "data": {
            "conversation_prompt_contract": PROMPT_CONTRACT,
            "strict_conversation_artifacts": True,
            "conversations": [{"path": str(fixture_train_path)}],
            "eval_conversations": [{"path": str(fixture_eval_path)}],
            "tokenizer": {"kind": "byte", "path": str(tokenizer_path)},
            "seq_len": 64,
            "function_masking": False,
            "shuffle": False,
            "sampling": {
                "mode": "parent_quota_update_blocks_with_format_pulses_v3",
                "general_source_index": 0,
                "format_source_index": 1,
                "parent_prefix_decisions": 5_568,
                "update_decisions": 16,
                "expected_parent_order_sha256": PARENT_ORDER_SHA256,
                "expected_parent_prefix_sha256": "7" * 64,
                "format_pulses": {
                    "count": 24,
                    "rows_per_phase": 4,
                    "phase_order": [
                        "format_core",
                        "multi_argument",
                        "parallel",
                        "text",
                    ],
                    "within_pulse_order": "phase_round_robin_v1",
                    "position_contract": "centered_update_quantiles_v1",
                },
            },
        },
        "optim": {
            "name": "adamw",
            "lr": 1.0e-6,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "loss_normalization": "microbatch_mean_v1",
            "freeze_parameters": list(PRODUCTION_FROZEN_PARAMETERS),
        },
        "schedule": {"type": "cosine", "warmup_steps": 24, "total_steps": 372},
        "batch": {
            "micro_batch_size": 2,
            "grad_accum_steps": 8,
            "pad_to_input_tokens": 63,
        },
        "evaluation": {
            "batch_size": 8,
            "max_conversations": 3,
            "selection": "greedy_uncovered_strata_then_semantic_sha256_fill_v1",
            "pad_to_input_tokens": 62,
        },
        "heads": {
            "joint_tool_pointer": False,
            "train_route_head": False,
            "train_dense_selector": False,
            "example_centroids": False,
            "multi_turn_batch_size": 0,
        },
        "runtime": {"device": "auto", "dtype": "auto", "seed": 2026},
        "log": {
            "out_dir": str(run_directory),
            "ckpt_every": 12,
            "archive_checkpoints": True,
        },
    }
    config_path = tmp_path / "production.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_payload = config_path.read_bytes()
    model_payload = model_config_path.read_bytes()

    parent_binding = {
        "parent_lm_sampling_mode": "quota_stratified_no_replacement_v1",
        "parent_no_replacement": True,
        "parent_order_sha256": PARENT_ORDER_SHA256,
        "parent_completed_steps": 348,
        "parent_completed_lm_cursor": 5_568,
        "parent_update_decisions": 16,
    }
    plan_sampling = {
        "mode": "parent_quota_update_blocks_with_format_pulses_v3",
        "no_replacement": True,
        "selected_decisions": 5_952,
        "update_layout": {
            "update_decisions": 16,
            "total_updates": 372,
            "pulse_positions_index_base": 0,
            "pulse_positions_zero_based": [7 + 15 * index for index in range(24)],
        },
        "format_pulses": {"phase_order": ["format_core", "multi_argument", "parallel", "text"]},
    }
    expected_sampling = {**plan_sampling, "parent_checkpoint_binding": parent_binding}
    execution_contract = {
        "execution_optimizer_update_limit": 8,
        "first_pulse_update_zero_based": 7,
        "executed_through_first_pulse": True,
        "executed_lm_decisions": 128,
    }
    exercised_keys = [[index, 1] for index in range(128)]
    sampling_evidence = {
        "kind": "localagent_sft_preflight_mixed_replay_prefix",
        "schema_version": 1,
        "production": {
            "selected_decisions": 5_952,
            "sampling_contract": expected_sampling,
            "sampling_contract_sha256": canonical_sha256(expected_sampling),
        },
        "exercised_prefix": {
            "decisions": 128,
            "decision_keys": exercised_keys,
            "decision_keys_sha256": canonical_sha256(exercised_keys),
            "equals_production_order_prefix": True,
        },
        "bounded_execution": execution_contract,
    }
    eval_selection = {"selected": {"rows": 3, "assistant_decisions": 3}}
    train_identity = {"bytes": 11, "sha256": "1" * 64}
    eval_identity = {"bytes": 12, "sha256": "2" * 64}
    overlap_audit = {"conversation_overlap": 0}
    plan_data = {
        "conversation_prompt_contract": PROMPT_CONTRACT,
        "conversations": [
            {"path": "fixture-train.jsonl", "verified": True, "artifact": train_identity}
        ],
        "eval_conversations": [
            {"path": "fixture-eval.jsonl", "verified": True, "artifact": eval_identity}
        ],
        "decay_conversations": [],
        "conversation_overlap_audit": overlap_audit,
        "eval_selection": eval_selection,
        "dataset_token_accounting": {
            "main": {
                "input_tokens": 10_000,
                "loss_tokens": 7_000,
                "sources": {
                    "fixture-train.jsonl": {
                        "input_tokens": 10_000,
                        "loss_tokens": 7_000,
                        "rows": 6_000,
                    }
                },
            },
            "decay": None,
        },
        "heldout_eval_token_accounting": {
            "accounting_kind": "exact_shifted_masked_language_model_tokens",
            "rows": 3,
            "input_tokens": 21,
            "loss_tokens": 7,
        },
        "decision_sampling": plan_sampling,
    }
    updates = [
        {
            "step": step,
            "input_tokens": 16,
            "loss_tokens": 16,
            "sources": {
                "fixture-train.jsonl": {
                    "draws": 16,
                    "rows": 16,
                    "input_tokens": 16,
                    "loss_tokens": 16,
                }
            },
        }
        for step in range(PRODUCTION_TOTAL_STEPS)
    ]
    plan = seal_stage_budget_plan(
        {
            "kind": "localagent_stage_budget_plan",
            "schema_version": 2,
            "stage": "sft",
            "request": {
                "config_path": str(config_path),
                "configured_steps": 372,
                "max_steps": 372,
                "min_supervised_tokens": None,
                "max_supervised_tokens": None,
            },
            "identity": {
                "config": {
                    "path": str(config_path),
                    "bytes": len(config_payload),
                    "sha256": hashlib.sha256(config_payload).hexdigest(),
                    "canonical_sha256": canonical_sha256(config),
                },
                "model_config": {
                    "path": str(model_config_path),
                    "bytes": len(model_payload),
                    "sha256": hashlib.sha256(model_payload).hexdigest(),
                    "canonical_sha256": canonical_sha256(model_config.__dict__),
                },
                "tokenizer": {
                    "path": str(tokenizer_path),
                    "kind": "byte",
                    "vocab_size": 256,
                    "sha256": tokenizer_sha256,
                },
            },
            "data": plan_data,
            "schedule": {
                "seed": 2026,
                "lr_schedule": "cosine",
                "decay_frac": 0.2,
                "loss_normalization": "microbatch_mean_v1",
                "freeze_parameters": list(PRODUCTION_FROZEN_PARAMETERS),
                "shuffle": False,
                "lm_sampling": plan_sampling,
                "micro_batch_size": 2,
                "grad_accum_steps": 8,
                "pad_to_input_tokens": 63,
                "evaluation_pad_to_input_tokens": 62,
                "joint_tool_pointer": False,
                "multi_turn_batch_size": 0,
                "distillation_enabled": False,
                "continuation": config["continuation"],
            },
            "planned": {
                "accounting_kind": "exact_shifted_masked_language_model_tokens",
                "updates": updates,
                "horizon_totals": _plan_total_accounting(),
                "selected_totals": _plan_total_accounting(),
            },
            "calibration": {
                "contract": "full fixed horizon",
                "mode": "full_horizon",
                "min_supervised_tokens": None,
                "max_supervised_tokens": None,
                "selected_steps": 372,
                "previous_prefix_loss_tokens": 5_936,
                "selected_prefix_loss_tokens": 5_952,
            },
        }
    )
    plan_path = tmp_path / "budget.json"
    plan_path.write_bytes(canonical_plan_bytes(plan))

    data_identity = {
        "conversations": [train_identity],
        "eval_conversations": [eval_identity],
        "decay_conversations": [],
        "conversation_overlap_audit": overlap_audit,
        "eval_selection": eval_selection,
        "conversation_prompt_contract": PROMPT_CONTRACT,
        "decision_sampling": expected_sampling,
    }
    lineage = {
        "version": 1,
        "stage": "sft",
        "config_sha256": canonical_sha256(config),
        "model_config_sha256": canonical_sha256(model_config.__dict__),
        "data_sha256": canonical_sha256(data_identity),
        "tokenizer_sha256": tokenizer_sha256,
        "git": {
            "commit": "3" * 40,
            "repository_sha256": "4" * 64,
            "dirty": True,
            "worktree_sha256": "5" * 64,
        },
        "parent_checkpoint_sha256": parent_sha256,
    }
    heldout_contract = {
        "kind": "deterministic_teacher_forced_assistant_tokens",
        "row_order": "configured_jsonl_assistant_decision_order",
        "same_rows_pre_post": True,
        "max_seq_len": 64,
        "pad_to_input_tokens": 62,
        "dataset_sha256": "6" * 64,
        "selection": eval_selection,
        "conversation_prompt_contract": PROMPT_CONTRACT,
    }
    heldout_pre = {
        "rows": 3,
        "assistant_loss_tokens": 7,
        "mean_loss": 1.5,
        "assistant_token_accuracy": 0.25,
        "assistant_sequence_accuracy": 0.0,
    }
    heldout_baseline = {"contract": heldout_contract, "pre": heldout_pre}
    tokenizer_metadata = {
        "kind": "byte",
        "path": str(tokenizer_path),
        "sha256": tokenizer_sha256,
    }
    data_metadata = {
        "conversation_rows": 6_000,
        "eval_conversation_rows": 3,
        "decision_sampling": expected_sampling,
    }
    execution_environment = {
        "torch_version": str(torch.__version__),
        "python_version": "3.12.2",
        "platform": "macOS-15-test-arm64",
        "cuda_available": False,
        "mps_built": True,
        "mps_available": True,
        "torch_intraop_threads": 4,
        "torch_interop_threads": 10,
    }
    execution = {
        "requested_device": "auto",
        "resolved_device": "mps",
        "requested_dtype": "auto",
        "resolved_dtype": "fp32",
        **execution_environment,
    }
    training_contract = {
        "version": 1,
        "steps": 372,
        "batch_size": 2,
        "accum_steps": 8,
        "lr": 1.0e-6,
        "warmup": 24,
        "lr_schedule": "cosine",
        "decay_frac": 0.2,
        "shuffle": False,
        "lm_sampling": expected_sampling,
        "joint_tool_head": False,
        "aux_weight": 1.0,
        "ptr_weight": 0.15,
        "mt_weight": 1.0,
        "multi_turn_batch_size": 0,
        "kd_type": "topk",
        "kd_k": 16,
        "kd_weight": 0.5,
        "kd_temperature": 2.0,
        "kd_enabled": False,
        "teacher_state_sha256": None,
        "teacher_cache_sha256": None,
        "max_seq_len": 64,
        "pad_to_input_tokens": 63,
        "amp_dtype": "torch.float32",
        "seed": 2026,
        "conversation_prompt_contract": PROMPT_CONTRACT,
        "tokenizer": {
            "class": "localagent.model.tokenizer.ByteTokenizer",
            "vocab_size": 256,
            "pad_id": 0,
            "eos_id": 1,
        },
        "prepared_data_sha256": "a" * 64,
        "initial_model_sha256": _resume_sha256(parent_state),
        "initial_tool_head_sha256": None,
        "initial_ptr_head_sha256": None,
        "optimizer": {
            "kind": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "grad_clip": 1.0,
        },
        "loss_normalization": "microbatch_mean_v1",
        "freeze_parameters": list(PRODUCTION_FROZEN_PARAMETERS),
        "optimizer_model_parameter_names": optimizer_names,
        "archive_checkpoints": True,
        "checkpoint_archive_every": 12,
        "checkpoint_archive_format": "immutable_periodic_sft_v1",
    }
    dataset_accounting = plan_data["dataset_token_accounting"]
    final_checkpoint = None
    for completed_steps in range(
        PRODUCTION_CHECKPOINT_EVERY,
        PRODUCTION_TOTAL_STEPS + 1,
        PRODUCTION_CHECKPOINT_EVERY,
    ):
        child_state = {name: tensor.clone() for name, tensor in parent_state.items()}
        child_state[optimizer_names[0]].view(-1)[0] = completed_steps / 1000
        checkpoint = _base_resume_payload(
            cfg=model_config.__dict__,
            state_dict=child_state,
            optimizer=_optimizer(child_state, optimizer_names, completed_steps),
            step=completed_steps - 1,
            loss_history=[float(index + 1) for index in range(completed_steps)],
            dataset_accounting=dataset_accounting,
            token_accounting=_accounting(completed_steps),
            training_contract=training_contract,
            lineage=lineage,
            tokenizer=tokenizer_metadata,
            data=data_metadata,
            execution=execution,
            heldout_baseline=heldout_baseline,
            lm_cursor=16 * completed_steps,
            accum_steps=8,
        )
        archive_path = run_directory / f"latest.step-{completed_steps:08d}.pt"
        _write_checkpoint(archive_path, checkpoint)
        final_checkpoint = checkpoint
    assert final_checkpoint is not None
    heldout_post = {
        "rows": 3,
        "assistant_loss_tokens": 7,
        "mean_loss": 1.25,
        "assistant_token_accuracy": 0.5,
        "assistant_sequence_accuracy": 1 / 3,
    }
    heldout_eval = {
        "contract": heldout_contract,
        "pre": heldout_pre,
        "post": heldout_post,
        "delta": {
            "mean_loss": heldout_post["mean_loss"] - heldout_pre["mean_loss"],
            "assistant_token_accuracy": (
                heldout_post["assistant_token_accuracy"]
                - heldout_pre["assistant_token_accuracy"]
            ),
            "assistant_sequence_accuracy": (
                heldout_post["assistant_sequence_accuracy"]
                - heldout_pre["assistant_sequence_accuracy"]
            ),
        },
    }
    latest = {
        **final_checkpoint,
        "route_head": None,
        "dense_selector": None,
        "selector_proj": 4,
        "examples": {},
        "fixed_horizon_progress": {
            "planned_optimizer_updates": 372,
            "completed_optimizer_updates": 372,
            "partial": False,
        },
        "heldout_eval": heldout_eval,
        "heldout_structured_eval": None,
        "continuation": config["continuation"],
    }
    latest_path = run_directory / "latest.pt"
    _write_checkpoint(latest_path, latest)
    metrics = {
        "stage": "sft",
        "checkpoint": str(latest_path),
        "conversation_rows": 6_000,
        "single_turn_rows": 6_000,
        "probe_decision_rows": 6_000,
        "loss_last": 372.0,
        "loss_steps": 372,
        "dataset_token_accounting": dataset_accounting,
        "token_accounting": _accounting(372),
        "token_accounting_scope": "language_model_microbatches",
        "lm_sampling": expected_sampling,
        "fixed_horizon_progress": latest["fixed_horizon_progress"],
        "conversation_prompt_contract": PROMPT_CONTRACT,
        "lineage": lineage,
        "data": data_metadata,
        "heldout_eval": heldout_eval,
        "heldout_structured_eval": None,
        "execution": execution,
        "continuation": config["continuation"],
        "structured_heads": {
            "tool_pointer": False,
            "route": False,
            "dense_selector": False,
        },
    }
    (run_directory / "metrics.json").write_bytes(
        (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode()
    )

    config_identity = {
        "exists": True,
        "path": str(config_path),
        "kind": "file",
        "bytes": len(config_payload),
        "sha256": hashlib.sha256(config_payload).hexdigest(),
        "canonical_sha256": canonical_sha256(config),
    }
    model_identity = {
        "exists": True,
        "path": str(model_config_path),
        "kind": "file",
        "bytes": len(model_payload),
        "sha256": hashlib.sha256(model_payload).hexdigest(),
        "canonical_sha256": canonical_sha256(model_config.__dict__),
    }
    parent_identity = {
        "exists": True,
        "path": str(parent_path),
        "kind": "file",
        "bytes": parent_path.stat().st_size,
        "sha256": parent_sha256,
        "completion": {
            "completed_steps": 348,
            "completed_lm_cursor": 5_568,
        },
    }
    config_after = {key: value for key, value in config_identity.items() if key != "canonical_sha256"}
    model_after = {key: value for key, value in model_identity.items() if key != "canonical_sha256"}
    parent_after = {key: value for key, value in parent_identity.items() if key != "completion"}
    expected_learning_rates = [
        cosine_lr(step, PRODUCTION_TOTAL_STEPS, 1.0e-6, 24, 0.1)
        for step in range(8)
    ]
    scope = {
        "model_named_parameter_names": model_names,
        "configured_frozen_model_parameter_names": list(PRODUCTION_FROZEN_PARAMETERS),
        "expected_optimizer_model_parameter_names": optimizer_names,
        "expected_trainable_model_tensor_count": len(optimizer_names),
        "training_contract_frozen_model_parameter_names": list(
            PRODUCTION_FROZEN_PARAMETERS
        ),
        "training_contract_optimizer_model_parameter_names": optimizer_names,
        "all_auxiliary_heads_disabled": True,
        "optimizer_param_group_parameter_count": len(optimizer_names),
        "optimizer_parameter_count_matches_expected": True,
        "compared_frozen_model_parameter_names": list(PRODUCTION_FROZEN_PARAMETERS),
        "frozen_model_tensors_exactly_preserved": True,
        "execution_optimizer_update_limit": 8,
        "executed_learning_rates": expected_learning_rates,
        "step_zero_learning_rate": expected_learning_rates[0],
        "last_executed_learning_rate": expected_learning_rates[-1],
        "any_executed_learning_rate_nonzero": True,
        "expected_completed_lm_cursor": 128,
        "observed_completed_lm_cursor": 128,
        "completed_lm_cursor_matches_expected": True,
        "unfrozen_model_transition_required": True,
        "first_changed_unfrozen_model_parameter": optimizer_names[0],
        "non_learning_limitation": None,
    }

    preflight_work = tmp_path / "preflight-work"
    preflight_run = preflight_work / "run"
    preflight_run.mkdir(parents=True)
    effective_config = copy.deepcopy(config)
    effective_config["runtime"] = {
        **effective_config["runtime"],
        "resume": False,
        "device": "mps",
    }
    effective_config["log"] = {
        **effective_config["log"],
        "out_dir": str(preflight_run),
        "ckpt_every": 1,
    }
    effective_path = preflight_work / "effective.yaml"
    effective_path.write_text(
        yaml.safe_dump(effective_config, sort_keys=False),
        encoding="utf-8",
    )
    effective_payload = effective_path.read_bytes()
    effective_identity = {
        "exists": True,
        "path": str(effective_path),
        "kind": "file",
        "bytes": len(effective_payload),
        "sha256": hashlib.sha256(effective_payload).hexdigest(),
        "canonical_sha256": canonical_sha256(effective_config),
    }
    preflight_execution = {
        "requested_device": "mps",
        "resolved_device": "mps",
        "requested_dtype": "auto",
        "resolved_dtype": "fp32",
        **execution_environment,
    }
    preflight_state = {name: tensor.clone() for name, tensor in parent_state.items()}
    preflight_state[optimizer_names[0]].view(-1)[0] = 0.008
    preflight_training_contract = copy.deepcopy(training_contract)
    preflight_training_contract["checkpoint_archive_every"] = 1
    preflight_checkpoint = _base_resume_payload(
        cfg=model_config.__dict__,
        state_dict=preflight_state,
        optimizer=_optimizer(preflight_state, optimizer_names, 8),
        step=7,
        loss_history=[float(index + 1) for index in range(8)],
        dataset_accounting=dataset_accounting,
        token_accounting=_accounting(8),
        training_contract=preflight_training_contract,
        lineage=lineage,
        tokenizer=tokenizer_metadata,
        data=data_metadata,
        execution=preflight_execution,
        heldout_baseline=heldout_baseline,
        lm_cursor=128,
        accum_steps=8,
    )
    preflight_checkpoint_path = preflight_run / "latest.pt"
    _write_checkpoint(preflight_checkpoint_path, preflight_checkpoint)
    preflight_metrics = {
        "stage": "sft",
        "checkpoint": str(preflight_checkpoint_path),
        "loss_steps": 8,
        "loss_last": 8.0,
        "lm_sampling": expected_sampling,
        "execution": preflight_execution,
        "data": data_metadata,
        "lineage": lineage,
        "token_accounting": _accounting(8),
        "dataset_token_accounting": dataset_accounting,
    }
    preflight_metrics_path = preflight_run / "metrics.json"
    preflight_metrics_path.write_bytes(_canonical_json(preflight_metrics))
    checkpoint_artifact = {
        "path": str(preflight_checkpoint_path),
        "bytes": preflight_checkpoint_path.stat().st_size,
        "sha256": _sha256_file(preflight_checkpoint_path),
        "resume_integrity_sha256": preflight_checkpoint["resume_integrity_sha256"],
    }
    metrics_artifact = {
        "path": str(preflight_metrics_path),
        "bytes": preflight_metrics_path.stat().st_size,
        "sha256": _sha256_file(preflight_metrics_path),
    }
    data_artifacts = [
        {
            "role": "train",
            "index": 0,
            "jsonl": {
                "exists": True,
                "path": str(fixture_train_path),
                "kind": "file",
                "bytes": fixture_train_path.stat().st_size,
                "sha256": _sha256_file(fixture_train_path),
            },
        },
        {
            "role": "eval",
            "index": 0,
            "jsonl": {
                "exists": True,
                "path": str(fixture_eval_path),
                "kind": "file",
                "bytes": fixture_eval_path.stat().st_size,
                "sha256": _sha256_file(fixture_eval_path),
            },
        },
    ]
    tokenizer_snapshot = {
        "exists": True,
        "path": str(tokenizer_path),
        "kind": "file",
        "bytes": tokenizer_path.stat().st_size,
        "sha256": tokenizer_sha256,
    }
    production_snapshot = {"exists": False, "path": str(run_directory)}
    preflight = seal_preflight_receipt(
        {
            "kind": "localagent_one_update_training_preflight",
            "schema_version": 1,
            "status": "passed",
            "started_at_utc": "2026-07-30T00:00:00+00:00",
            "finished_at_utc": "2026-07-30T00:01:00+00:00",
            "source": {
                "config": config_identity,
                "config_after": config_after,
                "model_config": model_identity,
                "model_config_after": model_after,
                "sft_parent_checkpoint": parent_identity,
                "sft_parent_checkpoint_after": parent_after,
                "sft_parent_checkpoint_untouched": True,
                "tokenizer": {
                    **tokenizer_snapshot,
                    "tokenizer_kind": "byte",
                    "lineage_sha256": tokenizer_sha256,
                },
                "tokenizer_after": tokenizer_snapshot,
                "data_artifacts": data_artifacts,
                "data_artifacts_after": copy.deepcopy(data_artifacts),
                "data_artifacts_untouched": True,
                "source_artifacts_untouched": True,
                "production_sft_output_before": production_snapshot,
                "production_sft_output_after": copy.deepcopy(production_snapshot),
                "production_sft_output_untouched": True,
                "sft_data_lineage": {
                    "identity": data_identity,
                    "sha256": canonical_sha256(data_identity),
                },
                "sft_sampling_lineage": sampling_evidence,
            },
            "effective": {
                "config": effective_identity,
                "config_after": {
                    key: value
                    for key, value in effective_identity.items()
                    if key != "canonical_sha256"
                },
                "config_untouched": True,
                "config_payload": effective_config,
                "contract": {
                    "stage": "sft",
                    "optimizer_updates": 8,
                    "realized_optimizer_updates": 8,
                    "optimizer_parameter_step_values": [8],
                    "optimizer_learning_rates": [expected_learning_rates[-1]],
                    "expected_step_zero_learning_rate": expected_learning_rates[0],
                    "expected_executed_learning_rates": expected_learning_rates,
                    "expected_last_executed_learning_rate": expected_learning_rates[-1],
                    "any_executed_learning_rate_nonzero": True,
                    "resume": False,
                    "checkpoint_every": 1,
                    "checkpoint_output": "isolated_work_directory",
                    "production_schedule_total_steps": 372,
                    **execution_contract,
                    "expected_completed_lm_cursor": 128,
                    "observed_completed_lm_cursor": 128,
                    "micro_batch_size": 2,
                    "grad_accum_steps": 8,
                    "effective_batch_size": 16,
                    "pad_to_input_tokens": 63,
                    "continuation": config["continuation"],
                    "parent_checkpoint_sha256": parent_sha256,
                    "parent_pins": parent_pins,
                    "parent_anchor_binding": parent_binding,
                    "lm_sampling": expected_sampling,
                    "production_lm_sampling_sha256": canonical_sha256(
                        expected_sampling
                    ),
                    "exercised_lm_prefix": sampling_evidence["exercised_prefix"],
                    "model_parameter_scope": scope,
                }
            },
            "model": {
                "name": model_config.name,
                "exact_parameters": model_config.estimate_params(),
                "max_seq_len": model_config.max_seq_len,
                "vocab_size": model_config.vocab_size,
            },
            "environment": {
                "python": "3.test",
                "platform": "test",
                "torch": str(torch.__version__),
                "mps_built": True,
                "mps_available": True,
                "cuda_available": False,
            },
            "measurement": {
                "scope": "synthetic eight-update fixture",
                "interpretation": "optimizer and transition proof",
                "non_learning_limitation": None,
            },
            "artifacts": {
                "checkpoint": checkpoint_artifact,
                "metrics": metrics_artifact,
            },
            "metrics": {
                "stage": "sft",
                "loss_steps": 8,
                "execution": preflight_execution,
            },
            "validation_errors": [],
            "error": None,
        }
    )
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_bytes(_canonical_json(preflight))
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt.verify_stage_budget_plan",
        lambda path: copy.deepcopy(plan),
    )
    unbound_identity = copy.deepcopy(data_identity)
    unbound_identity["decision_sampling"].pop("parent_checkpoint_binding")
    unbound_evidence = copy.deepcopy(sampling_evidence)
    unbound_evidence["production"]["sampling_contract"].pop(
        "parent_checkpoint_binding"
    )
    unbound_evidence["production"]["sampling_contract_sha256"] = canonical_sha256(
        unbound_evidence["production"]["sampling_contract"]
    )
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt._derive_sft_data_identity_and_sampling",
        lambda source: (copy.deepcopy(unbound_identity), copy.deepcopy(unbound_evidence)),
    )
    materialization = {
        "training_contract": training_contract,
        "tokenizer_lineage": {
            "kind": "byte",
            "vocab_size": 256,
            "sha256": tokenizer_sha256,
        },
        "tokenizer_metadata": tokenizer_metadata,
        "data_metadata": data_metadata,
        "execution": execution,
        "heldout_contract": heldout_contract,
    }
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt._derive_expected_runner_materialization",
        lambda **kwargs: copy.deepcopy(materialization),
    )
    preflight_sha256 = _sha256_file(preflight_path)
    roots = {
        "config_file_sha256": _sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "model_config_file_sha256": _sha256_file(model_config_path),
        "model_config_canonical_sha256": canonical_sha256(model_config.__dict__),
        "tokenizer_sha256": tokenizer_sha256,
        "data_sha256": canonical_sha256(data_identity),
        "parent_checkpoint_sha256": parent_sha256,
        "budget_plan_file_sha256": _sha256_file(plan_path),
        "budget_plan_self_sha256": plan["plan_self_sha256"],
        "preflight_file_sha256": preflight_sha256,
        "preflight_self_sha256": preflight["receipt_self_sha256"],
    }
    constant_overrides = {
        "PRODUCTION_CONFIG_FILE_SHA256": roots["config_file_sha256"],
        "PRODUCTION_CONFIG_CANONICAL_SHA256": roots["config_canonical_sha256"],
        "PRODUCTION_MODEL_CONFIG_FILE_SHA256": roots["model_config_file_sha256"],
        "PRODUCTION_MODEL_CONFIG_CANONICAL_SHA256": roots[
            "model_config_canonical_sha256"
        ],
        "PRODUCTION_TOKENIZER_SHA256": tokenizer_sha256,
        "PRODUCTION_PARENT_CHECKPOINT_SHA256": parent_sha256,
        "PRODUCTION_DATA_SHA256": roots["data_sha256"],
        "PRODUCTION_BUDGET_PLAN_FILE_SHA256": roots["budget_plan_file_sha256"],
        "PRODUCTION_PREFLIGHT_FILE_SHA256": preflight_sha256,
        "PRODUCTION_PREFLIGHT_SELF_SHA256_PREFIX": preflight[
            "receipt_self_sha256"
        ][:8],
        "PRODUCTION_MODEL_NAME": model_config.name,
        "PRODUCTION_MODEL_PARAMETERS": model_config.estimate_params(),
    }
    for name, value in constant_overrides.items():
        monkeypatch.setattr(
            f"localagent.eval.sft_production_receipt.{name}",
            value,
        )
    return {
        "config": config_path,
        "plan": plan_path,
        "preflight": preflight_path,
        "run": run_directory,
        "parent": parent_path,
        "model": model_config_path,
        "source": fixture_train_path,
        "tokenizer": tokenizer_path,
        "preflight_effective": effective_path,
        "preflight_checkpoint": preflight_checkpoint_path,
        "preflight_metrics": preflight_metrics_path,
        "plan_payload": plan,
        "optimizer_names": optimizer_names,
        "roots": roots,
    }


def _verify(fixture: dict[str, Any]) -> dict[str, Any]:
    return verify_sft_production_run(
        fixture["config"],
        fixture["plan"],
        fixture["preflight"],
        expected_roots=fixture["roots"],
    )


def _reroot_preflight_after_checkpoint_change(
    fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = fixture["preflight_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    preflight_path = fixture["preflight"]
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["artifacts"]["checkpoint"] = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256_file(checkpoint_path),
        "resume_integrity_sha256": checkpoint["resume_integrity_sha256"],
    }
    preflight = seal_preflight_receipt(
        {
            key: value
            for key, value in preflight.items()
            if key != "receipt_self_sha256"
        }
    )
    preflight_path.write_bytes(_canonical_json(preflight))
    file_sha256 = _sha256_file(preflight_path)
    fixture["roots"]["preflight_file_sha256"] = file_sha256
    fixture["roots"]["preflight_self_sha256"] = preflight["receipt_self_sha256"]
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt.PRODUCTION_PREFLIGHT_FILE_SHA256",
        file_sha256,
    )
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt.PRODUCTION_PREFLIGHT_SELF_SHA256_PREFIX",
        preflight["receipt_self_sha256"][:8],
    )


def test_completed_run_builds_canonical_integrity_only_receipt(
    production_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipt = _verify(production_fixture)

    assert receipt["contract"]["archive_count"] == 31
    assert receipt["contract"]["final_lm_cursor"] == 5_952
    assert receipt["scope"]["quality_claimed"] is False
    assert receipt["scope"]["retention_claimed"] is False
    latest_checkpoint = torch.load(
        production_fixture["run"] / "latest.pt",
        map_location="cpu",
        weights_only=True,
    )
    isolated_checkpoint = torch.load(
        production_fixture["preflight_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    assert latest_checkpoint["training_contract"]["checkpoint_archive_every"] == 12
    assert isolated_checkpoint["training_contract"]["checkpoint_archive_every"] == 1
    assert {
        key: latest_checkpoint["execution"][key]
        for key in (
            "requested_device",
            "resolved_device",
            "requested_dtype",
            "resolved_dtype",
        )
    } == {
        "requested_device": "auto",
        "resolved_device": "mps",
        "requested_dtype": "auto",
        "resolved_dtype": "fp32",
    }
    assert len(receipt["artifacts"]["archives"]) == 31
    assert receipt["validation"]["complete_live_evidence_inventory_rehashed"] is True
    assert {
        item["kind"] for item in receipt["artifacts"]["source_inputs"]
    } == {"jsonl", "tokenizer"}
    assert set(receipt["artifacts"]["preflight_evidence"]) == {
        "effective_config",
        "isolated_checkpoint",
        "isolated_metrics",
    }
    inventory = receipt["artifacts"]["live_evidence_inventory"]
    assert inventory["count"] == len(inventory["entries"])
    assert receipt["artifacts"]["latest_checkpoint"]["resume_integrity_sha256"] == (
        receipt["artifacts"]["archives"][-1]["resume_integrity_sha256"]
    )
    assert_sft_production_receipt(receipt)

    output = tmp_path / "receipt.json"
    write_sft_production_receipt(output, receipt)
    assert verify_sft_production_receipt_bytes(output.read_bytes()) == receipt
    assert verify_sft_production_receipt_against_artifacts(
        output,
        expected_receipt_file_sha256=_sha256_file(output),
    ) == receipt
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_sft_production_receipt(output, receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [("resolved_device", "cpu"), ("resolved_dtype", "fp16")],
)
def test_production_execution_must_resolve_to_mps_fp32(
    production_fixture: dict[str, Any],
    field: str,
    value: str,
) -> None:
    latest = production_fixture["run"] / "latest.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["execution"][field] = value

    _reseal_checkpoint(latest, mutate)
    with pytest.raises(ValueError, match=field):
        _verify(production_fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [("checkpoint_archive_every", 12), ("warmup", 25)],
)
def test_isolated_preflight_contract_allows_only_archive_interval_one(
    production_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    checkpoint_path = production_fixture["preflight_checkpoint"]

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["training_contract"][field] = value

    _reseal_checkpoint(checkpoint_path, mutate)
    _reroot_preflight_after_checkpoint_change(production_fixture, monkeypatch)
    with pytest.raises(ValueError, match="preflight isolated complete training contract"):
        _verify(production_fixture)


@pytest.mark.parametrize("failure", ["missing", "extra", "symlink"])
def test_directory_set_missing_extra_and_symlink_fail_closed(
    production_fixture: dict[str, Any],
    failure: str,
) -> None:
    run = production_fixture["run"]
    first = run / "latest.step-00000012.pt"
    if failure == "missing":
        first.unlink()
    elif failure == "extra":
        (run / "training.tmp").write_bytes(b"partial")
    else:
        first.unlink()
        first.symlink_to(run / "latest.step-00000024.pt")

    with pytest.raises(ValueError, match="directory|regular non-symlink"):
        _verify(production_fixture)


def test_frozen_tensor_drift_is_rejected(production_fixture: dict[str, Any]) -> None:
    archive = production_fixture["run"] / "latest.step-00000012.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"]["embed.weight"].view(-1)[0] += 1.0

    _reseal_checkpoint(archive, mutate)
    with pytest.raises(ValueError, match="frozen tensor"):
        _verify(production_fixture)


@pytest.mark.parametrize("mutation", ["cursor", "accounting"])
def test_cursor_and_prefix_accounting_drift_are_rejected(
    production_fixture: dict[str, Any],
    mutation: str,
) -> None:
    archive = production_fixture["run"] / "latest.step-00000012.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        if mutation == "cursor":
            checkpoint["sampling_state"]["lm_cursor"] += 1
        else:
            checkpoint["token_accounting"]["loss_tokens"] += 1

    _reseal_checkpoint(archive, mutate)
    with pytest.raises(ValueError, match="lm_cursor|prefix token accounting"):
        _verify(production_fixture)


def test_latest_must_equal_step372_sealed_resume_content(
    production_fixture: dict[str, Any],
) -> None:
    latest = production_fixture["run"] / "latest.pt"
    changed_name = production_fixture["optimizer_names"][0]

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"][changed_name].view(-1)[0] += 0.5

    _reseal_checkpoint(latest, mutate)
    with pytest.raises(ValueError, match="differs from the final archive|resume seals differ"):
        _verify(production_fixture)


def test_optimizer_parameter_scope_and_order_are_exact(
    production_fixture: dict[str, Any],
) -> None:
    archive = production_fixture["run"] / "latest.step-00000012.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["optimizer"]["param_groups"][0]["params"].reverse()

    _reseal_checkpoint(archive, mutate)
    with pytest.raises(ValueError, match="optimizer.*parameter-group contract"):
        _verify(production_fixture)


def test_optimizer_missing_moment_is_rejected(
    production_fixture: dict[str, Any],
) -> None:
    archive = production_fixture["run"] / "latest.step-00000012.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["optimizer"]["state"][0].pop("exp_avg_sq")

    _reseal_checkpoint(archive, mutate)
    with pytest.raises(ValueError, match="moment fields"):
        _verify(production_fixture)


def test_model_tensor_wrong_shape_is_rejected(
    production_fixture: dict[str, Any],
) -> None:
    archive = production_fixture["run"] / "latest.step-00000012.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        changed_name = next(
            name
            for name in production_fixture["optimizer_names"]
            if checkpoint["state_dict"][name].ndim > 1
        )
        tensor = checkpoint["state_dict"][changed_name]
        checkpoint["state_dict"][changed_name] = tensor.reshape(-1)

    _reseal_checkpoint(archive, mutate)
    with pytest.raises(ValueError, match="shape mismatch"):
        _verify(production_fixture)


@pytest.mark.parametrize("field", ["conversation_prompt_contract", "prepared_data_sha256"])
def test_complete_recomputed_training_contract_rejects_prompt_or_prepared_hash(
    production_fixture: dict[str, Any],
    field: str,
) -> None:
    latest = production_fixture["run"] / "latest.pt"

    def mutate(checkpoint: dict[str, Any]) -> None:
        checkpoint["training_contract"][field] = (
            "legacy" if field == "conversation_prompt_contract" else "f" * 64
        )

    _reseal_checkpoint(latest, mutate)
    with pytest.raises(ValueError, match="complete recomputed SFT training contract"):
        _verify(production_fixture)


def test_self_sealed_hand_authored_preflight_is_rejected(
    production_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = production_fixture["preflight"]
    forged = json.loads(preflight_path.read_text(encoding="utf-8"))
    forged.pop("artifacts")
    forged.pop("environment")
    forged.pop("measurement")
    forged.pop("model")
    forged = seal_preflight_receipt(
        {key: value for key, value in forged.items() if key != "receipt_self_sha256"}
    )
    preflight_path.write_bytes(_canonical_json(forged))
    file_sha256 = _sha256_file(preflight_path)
    production_fixture["roots"]["preflight_file_sha256"] = file_sha256
    production_fixture["roots"]["preflight_self_sha256"] = forged[
        "receipt_self_sha256"
    ]
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt.PRODUCTION_PREFLIGHT_FILE_SHA256",
        file_sha256,
    )
    monkeypatch.setattr(
        "localagent.eval.sft_production_receipt.PRODUCTION_PREFLIGHT_SELF_SHA256_PREFIX",
        forged["receipt_self_sha256"][:8],
    )

    with pytest.raises(ValueError, match="preflight receipt keys mismatch"):
        _verify(production_fixture)


def test_receipt_against_artifacts_requires_existing_real_path(
    production_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    missing = tmp_path / "forged-receipt.json"
    with pytest.raises(ValueError, match="missing|inaccessible"):
        verify_sft_production_receipt_against_artifacts(
            missing,
            expected_receipt_file_sha256="0" * 64,
        )


def test_top_level_dotdot_and_ancestor_symlink_paths_are_rejected(
    production_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    dotdot = f"{tmp_path}/nested/../{production_fixture['config'].name}"
    with pytest.raises(ValueError, match=r"'\.\.'"):
        verify_sft_production_run(
            dotdot,
            production_fixture["plan"],
            production_fixture["preflight"],
            expected_roots=production_fixture["roots"],
        )

    alias = tmp_path / "config-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify_sft_production_run(
            alias / production_fixture["config"].name,
            production_fixture["plan"],
            production_fixture["preflight"],
            expected_roots=production_fixture["roots"],
        )


@pytest.mark.parametrize("artifact", ["plan", "preflight"])
def test_budget_and_preflight_self_hashes_are_required(
    production_fixture: dict[str, Any],
    artifact: str,
) -> None:
    path = production_fixture[artifact]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if artifact == "plan":
        payload["request"]["max_steps"] = 371
        path.write_bytes(canonical_plan_bytes(payload))
    else:
        payload["source"]["source_artifacts_untouched"] = False
        path.write_bytes(_canonical_json(payload))
    with pytest.raises(ValueError, match="self-hash|external root"):
        _verify(production_fixture)


def test_receipt_bool_or_float_counters_fail_even_with_fresh_self_hash(
    production_fixture: dict[str, Any],
) -> None:
    receipt = _verify(production_fixture)
    for invalid in (True, 12.0):
        mutated = copy.deepcopy(receipt)
        mutated["artifacts"]["archives"][0]["completed_steps"] = invalid
        mutated.pop("receipt_self_sha256")
        mutated["receipt_self_sha256"] = canonical_sha256(mutated)
        with pytest.raises(ValueError, match="completed_steps"):
            assert_sft_production_receipt(mutated)


@pytest.mark.parametrize(
    "artifact",
    [
        "production_metrics",
        "preflight_metrics",
        "preflight_effective",
        "preflight_checkpoint",
        "source",
        "tokenizer",
    ],
)
def test_final_rehash_detects_concurrent_artifact_mutation(
    production_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    import localagent.eval.sft_production_receipt as module

    original = module._rehash_stable
    target = (
        production_fixture["run"] / "metrics.json"
        if artifact == "production_metrics"
        else production_fixture[artifact]
    )
    mutated = False

    def mutate_then_rehash(path, **kwargs):
        nonlocal mutated
        if not mutated:
            mutated = True
            target.write_bytes(target.read_bytes() + b" ")
        return original(path, **kwargs)

    monkeypatch.setattr(module, "_rehash_stable", mutate_then_rehash)
    with pytest.raises(RuntimeError, match="changed"):
        _verify(production_fixture)


def test_output_symlink_and_output_inside_run_are_refused(
    production_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    receipt = _verify(production_fixture)
    target = tmp_path / "target.json"
    target.write_bytes(b"keep")
    symlink = tmp_path / "receipt.json"
    symlink.symlink_to(target)
    with pytest.raises((FileExistsError, ValueError), match="refusing to replace|symlink"):
        write_sft_production_receipt(symlink, receipt)
    assert target.read_bytes() == b"keep"
    with pytest.raises(ValueError, match="outside"):
        write_sft_production_receipt(
            production_fixture["run"] / "receipt.json",
            receipt,
        )


def test_cli_requires_all_explicit_evidence_arguments() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_sft_production_run.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--config" in result.stderr
    assert "--budget-plan" in result.stderr
    assert "--preflight" in result.stderr
    assert "--output" in result.stderr
    assert "--expected-preflight-file-sha256" in result.stderr
    assert "--expected-preflight-self-sha256" in result.stderr
    assert "--expected-budget-plan-file-sha256" in result.stderr
    assert "--expected-config-canonical-sha256" in result.stderr
