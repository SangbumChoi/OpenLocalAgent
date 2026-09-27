from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from localagent.eval.rl_readiness import (
    CONFIG_KIND,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SUMMARY_KIND,
    assert_rl_readiness_summary,
    load_rl_readiness_summary,
    reproduce_historical_rl_readiness_v1,
    run_ready_rl,
    summarize_rl_readiness,
    write_rl_readiness_summary,
)

_PROMPT_CONTRACT = "openai_full_catalog_v1"
_CHECKPOINT_SHA256 = "1" * 64
_EVAL_JSONL_SHA256 = "2" * 64
_EVAL_SEMANTIC_SHA256 = "3" * 64
_SELECTED_EVAL_SEMANTIC_SHA256 = "4" * 64
ROOT = Path(__file__).resolve().parents[1]
SEALED_V1_CONFIG = ROOT / "configs/eval/paper-tier-1m-rl-readiness.yaml"
SEALED_V1_SUMMARY = ROOT / "data/provenance/paper/rl-readiness-paper-tier-1m.json"
SEALED_V1_SCORECARD = ROOT / "runs/eval/webgpu-1m-sft-scorecard.json"
SEALED_V1_PREFLIGHT = (
    ROOT / "data/provenance/paper/preflights/rl-paper-tier-1m-mps-v2.json"
)
SEALED_V1_INPUTS = (
    SEALED_V1_CONFIG,
    SEALED_V1_SCORECARD,
    SEALED_V1_PREFLIGHT,
)


def _canonical_sha256(value: object, *, trailing_lf: bool = False) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if trailing_lf:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rate(correct: int, total: int) -> dict[str, int | float | None]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scorecard(
    *,
    decisions: int,
    completed: int,
    complete_format: int,
    schema_valid: int,
    schema_attempts: int,
    exact: int,
) -> dict:
    case_set_sha256 = "5" * 64
    result = {
        "kind": "localagent_internal_agent_scorecard_result",
        "schema_version": 1,
        "benchmark": {
            "name": "LocalAgent BFCL-style internal agent scorecard",
            "official_bfcl": False,
            "external_native_benchmark": False,
            "conversation_prompt_contract": _PROMPT_CONTRACT,
        },
        "provenance": {
            "checkpoint": {
                "stage": "sft",
                "sha256": _CHECKPOINT_SHA256,
                "conversation_prompt_contract": _PROMPT_CONTRACT,
            },
            "cases": {
                "split": "eval",
                "rule_verified": True,
                "environment_executed": False,
                "case_set_sha256": case_set_sha256,
                "jsonl": {"sha256": _EVAL_JSONL_SHA256},
                "selection": {
                    "algorithm": "greedy_uncovered_strata_then_semantic_sha256_fill_v1",
                    "source": {"semantic_set_sha256": _EVAL_SEMANTIC_SHA256},
                },
            },
            "generation": {
                "conversation_prompt_contract": _PROMPT_CONTRACT,
                "truncation": "forbidden",
                "temperature": 0.0,
            },
        },
        "scorecard": {
            "contract": {
                "official_bfcl": False,
                "external_native_benchmark": False,
                "conversation_prompt_contract": _PROMPT_CONTRACT,
            },
            "case_set": {
                "sha256": case_set_sha256,
                "conversations": decisions,
                "assistant_decisions": decisions,
                "tool_decisions": decisions,
                "no_tool_decisions": 0,
            },
            "metrics": {
                "generation_completion": _rate(completed, decisions),
                "format_validity": _rate(complete_format, decisions),
                "schema_validity_on_tool_attempts": _rate(
                    schema_valid,
                    schema_attempts,
                ),
                "action_exact": _rate(exact, decisions),
            },
            "by_category": {},
            "predictions": {
                "records": decisions,
                "complete": completed,
                "terminated_by_eos": completed,
                "raw_outputs_retained": False,
                "finish_reasons": {
                    **({"eos": completed} if completed else {}),
                    **({"length": decisions - completed} if completed < decisions else {}),
                },
            },
        },
        "limitations": [],
    }
    result["result_self_sha256"] = _canonical_sha256(result, trailing_lf=True)
    return result


def _preflight(
    *,
    status: str,
    syntax: int,
    complete_parser: int,
    schema_valid: int,
    exact: int,
    rewards: list[tuple[float, int]],
    informative_groups: int,
    realized_updates: int,
    truncated: int = 0,
) -> dict:
    attempted_groups = 8
    attempted_rollouts = 32
    generated_eos = attempted_rollouts - truncated
    validation_errors = (
        []
        if status == "passed"
        else ["isolated RL rollouts did not demonstrate a learnable update"]
    )
    result = {
        "kind": "localagent_one_update_training_preflight",
        "schema_version": 1,
        "status": status,
        "source": {
            "source_artifacts_untouched": True,
            "production_rl_output_untouched": True,
            "sft_parent_checkpoint": {"sha256": _CHECKPOINT_SHA256},
        },
        "effective": {
            "contract": {
                "stage": "rl",
                "rollout_steps": 1,
                "resume": False,
                "checkpoint_output": "isolated_work_directory",
                "group_size": 4,
                "prompts_per_step": attempted_groups,
                "configured_policy_epochs_preserved": 2,
                "realized_optimizer_updates": realized_updates,
            }
        },
        "measurement": {
            "rollout_observability": {
                "parsing": {
                    "parser_format_valid_rollouts": complete_parser,
                    "complete_parser_format_valid_rollouts": complete_parser,
                    "parser_tool_syntax_rollouts": syntax,
                    "tool_reward_rollouts": attempted_rollouts,
                    "text_reward_rollouts": 0,
                    "strict_tool_format_valid_rollouts": schema_valid,
                },
                "reward": {
                    "distribution": [
                        {
                            "reward": reward,
                            "reward_hex": float(reward).hex(),
                            "count": count,
                        }
                        for reward, count in rewards
                    ],
                    "unique_values": len(rewards),
                    "exact_success_rollouts": exact,
                },
                "truncation": {"truncated_rollouts": truncated},
                "tokens": {"generated_eos_tokens": generated_eos},
            }
        },
        "metrics": {
            "stage": "rl",
            "rl_accounting": {
                "attempted_rollout_steps": 1,
                "attempted_groups": attempted_groups,
                "attempted_rollouts": attempted_rollouts,
                "informative_groups": informative_groups,
                "zero_signal_steps": int(informative_groups == 0),
                "realized_optimizer_updates": realized_updates,
                "policy_epochs_per_informative_batch": 2,
                "truncated_rollouts": truncated,
                "generated_eos_tokens": generated_eos,
            },
            "data": {
                "conversation_prompt_contract": _PROMPT_CONTRACT,
                "train_artifacts": [
                    {
                        "split": "train",
                        "path": "train.jsonl",
                        "jsonl": {"sha256": "6" * 64},
                    }
                ],
                "eval_artifacts": [
                    {
                        "split": "eval",
                        "path": "eval.jsonl",
                        "jsonl": {"sha256": _EVAL_JSONL_SHA256},
                    }
                ],
                "split_audit": {
                    "row_overlap": 0,
                    "prompt_overlap": 0,
                    "eval_dataset_sha256": _EVAL_SEMANTIC_SHA256,
                },
                "selected_eval_split_audit": {
                    "row_overlap": 0,
                    "prompt_overlap": 0,
                    "eval_dataset_sha256": _SELECTED_EVAL_SEMANTIC_SHA256,
                },
                "preflight_minimum_coverage": {
                    "selection_audit": {
                        "source": {"semantic_set_sha256": _EVAL_SEMANTIC_SHA256},
                        "selected": {"semantic_set_sha256": _SELECTED_EVAL_SEMANTIC_SHA256},
                    }
                },
            },
            "heldout_eval": {
                "contract": {
                    "split": "explicit_disjoint_eval_conversations",
                    "same_rows_pre_post": True,
                    "current_gold_in_prompt": False,
                    "conversation_prompt_contract": _PROMPT_CONTRACT,
                }
            },
        },
        "validation_errors": validation_errors,
        "error": (
            None
            if status == "passed"
            else {
                "type": "RLPreflightValidationError",
                "message": validation_errors[0],
            }
        ),
    }
    result["receipt_self_sha256"] = _canonical_sha256(result)
    return result


def _thresholds() -> dict:
    return {
        "scorecard": {
            "min_assistant_decisions": 512,
            "min_generation_completion_rate": 0.90,
            "max_generation_truncation_rate": 0.10,
            "min_complete_format_rate": 0.05,
            "min_schema_valid_attempt_rate": 0.05,
            "min_action_exact_successes": 1,
        },
        "rl_preflight": {
            "min_attempted_groups": 8,
            "min_attempted_rollouts": 32,
            "min_tool_syntax_rate": 0.10,
            "min_complete_parser_valid_rate": 0.05,
            "min_schema_valid_tool_rate": 0.05,
            "min_exact_successes": 1,
            "min_reward_unique_values": 2,
            "min_informative_groups": 1,
            "min_informative_group_rate": 0.125,
            "min_realized_optimizer_updates": 1,
            "max_truncation_rate": 0.05,
        },
    }


def _write_inputs(
    tmp_path: Path,
    scorecard: dict,
    preflight: dict,
    *,
    thresholds: dict | None = None,
) -> Path:
    scorecard_path = tmp_path / "scorecard.json"
    preflight_path = tmp_path / "preflight.json"
    _write_json(scorecard_path, scorecard)
    _write_json(preflight_path, preflight)
    config = {
        "kind": CONFIG_KIND,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "evidence": {
            "scorecard": {
                "path": str(scorecard_path),
                "expected_self_sha256": scorecard["result_self_sha256"],
            },
            "rl_preflight": {
                "path": str(preflight_path),
                "expected_self_sha256": preflight["receipt_self_sha256"],
            },
        },
        "thresholds": _thresholds() if thresholds is None else thresholds,
    }
    config_path = tmp_path / "readiness.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _passing_scorecard() -> dict:
    return _scorecard(
        decisions=512,
        completed=500,
        complete_format=100,
        schema_valid=80,
        schema_attempts=100,
        exact=8,
    )


def _passing_preflight() -> dict:
    return _preflight(
        status="passed",
        syntax=24,
        complete_parser=20,
        schema_valid=8,
        exact=4,
        rewards=[(0.0, 24), (1.1, 8)],
        informative_groups=2,
        realized_updates=2,
    )


def _gates(summary: dict) -> dict[str, dict]:
    return {gate["id"]: gate for gate in summary["gates"]}


def _v2_thresholds() -> dict:
    return {
        "scorecard": {
            "min_assistant_decisions": 512,
            "min_generation_completion_rate": 0.90,
            "max_generation_truncation_rate": 0.10,
            "min_complete_format_successes": 1,
            "min_complete_format_rate": 0.05,
            "min_tool_format_successes": 1,
            "min_tool_format_rate": 0.05,
            "min_schema_valid_tool_successes": 1,
            "min_schema_valid_tool_rate": 0.05,
            "min_tool_name_case_exact_successes": 1,
            "min_tool_name_case_exact_rate": 0.05,
            "min_whole_call_exact_successes": 1,
            "min_whole_call_exact_rate": 0.05,
            "min_abstention_successes": 1,
            "min_abstention_rate": 0.05,
        },
        "sft_metrics": {
            "max_mean_loss_increase": 0.01,
            "max_assistant_token_accuracy_drop": 0.01,
            "max_assistant_sequence_accuracy_drop": 0.01,
        },
        "rl_preflight": _thresholds()["rl_preflight"],
    }


def _rehash_scorecard(scorecard: dict) -> None:
    scorecard.pop("result_self_sha256", None)
    scorecard["result_self_sha256"] = _canonical_sha256(scorecard, trailing_lf=True)


def _rehash_preflight(preflight: dict) -> None:
    preflight.pop("receipt_self_sha256", None)
    preflight["receipt_self_sha256"] = _canonical_sha256(preflight)


def _v2_scorecard(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    tool_exact: bool = True,
    zero_tool_decisions: bool = False,
) -> dict:
    tool_decisions = 0 if zero_tool_decisions else 400
    no_tool_decisions = 512 - tool_decisions
    schema_attempts = 0 if zero_tool_decisions else 250
    abstention_correct = 300 if zero_tool_decisions else 80
    scorecard = _scorecard(
        decisions=512,
        completed=500,
        complete_format=330,
        schema_valid=220 if tool_exact and not zero_tool_decisions else 0,
        schema_attempts=schema_attempts,
        exact=(
            abstention_correct if zero_tool_decisions else 260 if tool_exact else abstention_correct
        ),
    )
    scorecard["provenance"]["checkpoint"].update(
        {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
        }
    )
    scorecard["provenance"]["cases"]["selection"]["selected"] = {
        "rows": 512,
        "assistant_decisions": 512,
        "semantic_set_sha256": _SELECTED_EVAL_SEMANTIC_SHA256,
    }
    scorecard["scorecard"]["case_set"]["tool_decisions"] = tool_decisions
    scorecard["scorecard"]["case_set"]["no_tool_decisions"] = no_tool_decisions
    metrics = scorecard["scorecard"]["metrics"]
    metrics["tool_format_validity_on_tool_decisions"] = _rate(
        250 if tool_exact and not zero_tool_decisions else 0,
        tool_decisions,
    )
    metrics["schema_validity_on_tool_decisions"] = _rate(
        220 if tool_exact and not zero_tool_decisions else 0,
        tool_decisions,
    )
    metrics["tool_name"] = {
        "case_exact": _rate(
            210 if tool_exact and not zero_tool_decisions else 0,
            tool_decisions,
        ),
    }
    metrics["whole_call_exact"] = _rate(
        180 if tool_exact and not zero_tool_decisions else 0,
        tool_decisions,
    )
    metrics["abstention"] = _rate(abstention_correct, no_tool_decisions)
    _rehash_scorecard(scorecard)
    return scorecard


def _sft_metrics(
    checkpoint_path: Path,
    *,
    post_loss: float = 1.9,
    post_token_accuracy: float = 0.71,
    post_sequence_accuracy: float = 0.11,
) -> dict:
    pre_loss = 2.0
    pre_token_accuracy = 0.70
    pre_sequence_accuracy = 0.10
    return {
        "stage": "sft",
        "checkpoint": str(checkpoint_path),
        "conversation_prompt_contract": _PROMPT_CONTRACT,
        "data": {
            "heldout_content_overlap": 0,
            "heldout_rendered_prompt_overlap": 0,
        },
        "heldout_eval": {
            "contract": {
                "kind": "deterministic_teacher_forced_assistant_tokens",
                "same_rows_pre_post": True,
                "conversation_prompt_contract": _PROMPT_CONTRACT,
                "row_order": "configured_jsonl_assistant_decision_order",
                "selection": {
                    "algorithm": "greedy_uncovered_strata_then_semantic_sha256_fill_v1",
                    "source": {
                        "semantic_set_sha256": _EVAL_SEMANTIC_SHA256,
                    },
                    "selected": {
                        "rows": 512,
                        "assistant_decisions": 512,
                        "semantic_set_sha256": _SELECTED_EVAL_SEMANTIC_SHA256,
                    },
                },
            },
            "pre": {
                "rows": 512,
                "assistant_loss_tokens": 4096,
                "mean_loss": pre_loss,
                "assistant_token_accuracy": pre_token_accuracy,
                "assistant_sequence_accuracy": pre_sequence_accuracy,
            },
            "post": {
                "rows": 512,
                "assistant_loss_tokens": 4096,
                "mean_loss": post_loss,
                "assistant_token_accuracy": post_token_accuracy,
                "assistant_sequence_accuracy": post_sequence_accuracy,
            },
            "delta": {
                "mean_loss": post_loss - pre_loss,
                "assistant_token_accuracy": post_token_accuracy - pre_token_accuracy,
                "assistant_sequence_accuracy": post_sequence_accuracy - pre_sequence_accuracy,
            },
        },
    }


def _sft_sweep_result(checkpoint_path: Path, checkpoint_sha256: str) -> dict:
    baseline = {
        "rows": 512,
        "assistant_loss_tokens": 4096,
        "mean_loss": 2.0,
        "assistant_token_accuracy": 0.70,
        "assistant_sequence_accuracy": 0.10,
    }
    post = {
        "rows": 512,
        "assistant_loss_tokens": 4096,
        "mean_loss": 1.9,
        "assistant_token_accuracy": 0.71,
        "assistant_sequence_accuracy": 0.11,
    }
    thresholds = _v2_thresholds()["sft_metrics"]
    artifact = {
        "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint_sha256,
    }
    delta = {
        "mean_loss": post["mean_loss"] - baseline["mean_loss"],
        "assistant_token_accuracy": (
            post["assistant_token_accuracy"] - baseline["assistant_token_accuracy"]
        ),
        "assistant_sequence_accuracy": (
            post["assistant_sequence_accuracy"] - baseline["assistant_sequence_accuracy"]
        ),
    }
    record = {
        "artifact": artifact,
        "checkpoint_step": 1,
        "completed_steps": 2,
        "planned_steps": 512,
        "metrics": post,
        "delta_from_baseline": delta,
        "gates": {
            "mean_loss_non_inferiority": {
                "observed_increase": delta["mean_loss"],
                "maximum_increase": thresholds["max_mean_loss_increase"],
                "passed": True,
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
        },
        "retention_eligible": True,
    }
    result = {
        "kind": "localagent_sft_checkpoint_sweep_result",
        "schema_version": 2,
        "inputs": {
            "sweep_config": {"path": "sweep.yaml", "bytes": 1, "sha256": "8" * 64},
            "sweep_config_sha256": "9" * 64,
            "training_config": {
                "path": "training.yaml",
                "bytes": 1,
                "sha256": "a" * 64,
            },
            "training_config_sha256": "b" * 64,
            "checkpoint_discovery": {"mode": "explicit_paths"},
            "expected_parent_checkpoint_sha256": "7" * 64,
            "expected_eval": {
                "conversations": 512,
                "assistant_decisions": 512,
                "assistant_loss_tokens": 4096,
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
            "model_config": {"path": "model.yaml", "bytes": 1, "sha256": "c" * 64},
            "model_config_sha256": "d" * 64,
            "tokenizer": {"kind": "bpe", "sha256": "e" * 64},
            "lineage": {
                "parent_checkpoint_sha256": "7" * 64,
            },
            "training_contract": {"steps": 512},
        },
        "heldout": {
            "sources": [
                {
                    "path": "eval.jsonl",
                    "bytes": 1,
                    "sha256": _EVAL_JSONL_SHA256,
                }
            ],
            "conversations": 512,
            "assistant_decisions": 512,
            "assistant_loss_tokens": 4096,
            "contract": {
                "kind": "deterministic_teacher_forced_assistant_tokens",
                "same_rows_pre_post": True,
                "conversation_prompt_contract": _PROMPT_CONTRACT,
                "row_order": "configured_jsonl_assistant_decision_order",
                "selection": {
                    "algorithm": "greedy_uncovered_strata_then_semantic_sha256_fill_v1",
                    "source": {
                        "semantic_set_sha256": _EVAL_SEMANTIC_SHA256,
                    },
                    "selected": {
                        "rows": 512,
                        "assistant_decisions": 512,
                        "semantic_set_sha256": _SELECTED_EVAL_SEMANTIC_SHA256,
                    },
                },
            },
            "baseline": baseline,
            "leakage_assurance": {
                "heldout_content_overlap": 0,
                "heldout_rendered_prompt_overlap": 0,
                "conversation_overlap_audit": {
                    "fingerprint_contract": {
                        "conversation_prompt_contract": _PROMPT_CONTRACT,
                        "semantic_row": "semantic-row-contract",
                        "rendered_prompt": "rendered-prompt-contract",
                        "set_aggregation": "set-aggregation-contract",
                    },
                    "left_rows": 8192,
                    "right_rows": 5000,
                    "left_rendered_prompts": 8192,
                    "right_rendered_prompts": 7963,
                    "left_semantic_set_sha256": "1" * 64,
                    "right_semantic_set_sha256": _EVAL_SEMANTIC_SHA256,
                    "left_rendered_prompt_set_sha256": "3" * 64,
                    "right_rendered_prompt_set_sha256": "4" * 64,
                    "semantic_overlap": 0,
                    "rendered_prompt_overlap": 0,
                    "semantic_overlap_sha256": [],
                    "rendered_prompt_overlap_sha256": [],
                },
            },
        },
        "thresholds": thresholds,
        "execution": {"resolved_device": "cpu"},
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
        "checkpoints": [record],
        "summary": {
            "evaluated_checkpoints": 1,
            "retention_eligible_checkpoints": 1,
            "failed_checkpoints": 0,
            "status": "retention_eligible_checkpoint_found",
            "best_retention_eligible_checkpoint": {
                "artifact": artifact,
                "checkpoint_step": 1,
                "completed_steps": 2,
                "metrics": post,
            },
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def _write_v2_inputs(
    tmp_path: Path,
    *,
    tool_exact: bool = True,
    zero_tool_decisions: bool = False,
    metrics: dict | None = None,
    thresholds: dict | None = None,
    teacher_evidence: str = "metrics",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = tmp_path / "latest.pt"
    checkpoint_path.write_bytes(b"sealed-test-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    production_out = tmp_path / "production-rl"
    production_config_path = tmp_path / "production-rl.yaml"
    production_config = {
        "stage": "rl",
        "init_from": str(checkpoint_path),
        "runtime": {
            "device": "cpu",
            "dtype": "fp32",
            "resume": False,
        },
        "optim": {"lr": 2.0e-5},
        "schedule": {"warmup_steps": 0, "total_steps": 1},
        "log": {"out_dir": str(production_out)},
    }
    production_config_path.write_text(
        yaml.safe_dump(production_config, sort_keys=False),
        encoding="utf-8",
    )
    scorecard = _v2_scorecard(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        tool_exact=tool_exact,
        zero_tool_decisions=zero_tool_decisions,
    )
    preflight = _passing_preflight()
    preflight["source"]["config"] = {
        "path": str(production_config_path),
        "canonical_sha256": _canonical_sha256(production_config),
    }
    preflight["source"]["sft_parent_checkpoint"].update(
        {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        }
    )
    preflight["effective"]["contract"].update(
        {
            "execution_rollout_step_limit": 1,
            "production_schedule_total_steps": 1,
            "first_nonzero_learning_rate_step": 0,
            "expected_learning_rates": [2.0e-5],
        }
    )
    preflight["metrics"]["execution"] = {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "requested_dtype": "fp32",
        "resolved_dtype": "fp32",
    }
    preflight["metrics"]["rl_accounting"].update(
        {
            "learning_rate_history": [2.0e-5],
            "fixed_horizon_progress": {
                "planned_rollout_steps": 1,
                "completed_rollout_steps": 1,
                "execution_rollout_step_limit": 1,
                "bounded_prefix": False,
            },
        }
    )
    preflight["measurement"]["policy_transition"] = {
        "contract": "exact_named_policy_parameter_comparison_v1",
        "model_named_parameter_names": ["weight"],
        "model_parameter_count": 1,
        "compared_model_parameter_names": ["weight"],
        "compared_model_parameter_count": 1,
        "changed_model_parameter_names": ["weight"],
        "changed_model_parameter_count": 1,
        "first_changed_model_parameter": "weight",
        "initial_model_state_sha256": "a" * 64,
        "final_model_state_sha256": "b" * 64,
        "at_least_one_policy_tensor_changed": True,
        "production_schedule_total_steps": 1,
        "execution_rollout_step_limit": 1,
        "first_nonzero_learning_rate_step": 0,
        "expected_learning_rates": [2.0e-5],
        "actual_learning_rates": [2.0e-5],
        "actual_learning_rates_match_expected": True,
        "nonzero_learning_rate_executed": True,
        "final_optimizer_learning_rates": [2.0e-5],
        "final_optimizer_learning_rate_matches_expected": True,
    }
    _rehash_preflight(preflight)
    if teacher_evidence not in {"metrics", "sweep"}:
        raise ValueError("teacher_evidence must be metrics or sweep")
    if teacher_evidence == "sweep" and metrics is not None:
        raise ValueError("metrics override is unsupported for sweep evidence")
    teacher_payload = (
        (_sft_metrics(checkpoint_path) if metrics is None else metrics)
        if teacher_evidence == "metrics"
        else _sft_sweep_result(
            checkpoint_path,
            checkpoint_sha256,
        )
    )

    scorecard_path = tmp_path / "scorecard-v2.json"
    preflight_path = tmp_path / "preflight-v2.json"
    teacher_path = tmp_path / (
        "metrics-v2.json" if teacher_evidence == "metrics" else "sweep-v2.json"
    )
    _write_json(scorecard_path, scorecard)
    _write_json(preflight_path, preflight)
    _write_json(teacher_path, teacher_payload)
    teacher_spec = (
        {
            "sft_metrics": {
                "path": str(teacher_path),
                "expected_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            }
        }
        if teacher_evidence == "metrics"
        else {
            "sft_checkpoint_sweep": {
                "path": str(teacher_path),
                "expected_self_sha256": teacher_payload["result_sha256"],
                "selected_checkpoint_sha256": checkpoint_sha256,
            }
        }
    )
    config = {
        "kind": CONFIG_KIND,
        "schema_version": SCHEMA_VERSION,
        "production": {
            "rl_config": {
                "path": str(production_config_path),
                "expected_canonical_sha256": _canonical_sha256(
                    production_config
                ),
            },
            "parent_checkpoint": {
                "path": str(checkpoint_path),
                "expected_sha256": checkpoint_sha256,
            },
            "execution": {
                "production_requested_device": "cpu",
                "production_requested_dtype": "fp32",
                "preflight_requested_device": "cpu",
                "preflight_requested_dtype": "fp32",
                "resolved_device": "cpu",
                "resolved_dtype": "fp32",
            },
        },
        "evidence": {
            "scorecard": {
                "path": str(scorecard_path),
                "expected_self_sha256": scorecard["result_self_sha256"],
            },
            "rl_preflight": {
                "path": str(preflight_path),
                "expected_self_sha256": preflight["receipt_self_sha256"],
            },
            **teacher_spec,
        },
        "thresholds": _v2_thresholds() if thresholds is None else thresholds,
    }
    config_path = tmp_path / "readiness-v2.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, teacher_path


def _rewrite_v2_preflight(config_path: Path, preflight: dict) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight_path = Path(config["evidence"]["rl_preflight"]["path"])
    _rehash_preflight(preflight)
    _write_json(preflight_path, preflight)
    config["evidence"]["rl_preflight"]["expected_self_sha256"] = preflight[
        "receipt_self_sha256"
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_zero_signal_summary_preserves_each_readiness_rung(tmp_path: Path) -> None:
    scorecard = _scorecard(
        decisions=820,
        completed=153,
        complete_format=0,
        schema_valid=0,
        schema_attempts=820,
        exact=0,
    )
    preflight = _preflight(
        status="failed",
        syntax=26,
        complete_parser=6,
        schema_valid=0,
        exact=0,
        rewards=[(0.0, 32)],
        informative_groups=0,
        realized_updates=0,
    )
    config_path = _write_inputs(tmp_path, scorecard, preflight)

    summary = reproduce_historical_rl_readiness_v1(config_path)

    assert summary["kind"] == SUMMARY_KIND
    assert summary["contract"]["official_bfcl"] is False
    assert summary["evidence"]["pairing"]["heldout_split"] == "verified_disjoint_eval"
    assert summary["funnel"]["greedy_heldout"]["generation_completion"] == {
        "count": 153,
        "total": 820,
        "rate": 153 / 820,
    }
    sampled = summary["funnel"]["sampled_preflight"]
    assert sampled["tool_syntax_presence"] == {"count": 26, "total": 32, "rate": 26 / 32}
    assert sampled["complete_parser_validity"] == {
        "count": 6,
        "total": 32,
        "rate": 6 / 32,
    }
    assert sampled["schema_valid_tool_actions"]["count"] == 0
    assert sampled["exact_success"]["count"] == 0
    assert sampled["generation_truncation"]["count"] == 0
    signal = summary["funnel"]["optimization_signal"]
    assert signal["unique_reward_values"] == 1
    assert signal["informative_groups"] == 0
    assert signal["realized_optimizer_updates"] == 0

    gates = _gates(summary)
    assert gates["sampled_tool_syntax"]["passed"] is True
    assert gates["sampled_complete_parser_validity"]["passed"] is True
    assert gates["sampled_schema_valid_tool_actions"]["passed"] is False
    assert gates["reward_diversity"]["passed"] is False
    assert gates["informative_groups"]["passed"] is False
    assert summary["decision"]["status"] == "not_ready_for_rl"
    assert summary["decision"]["learnable_signal_observed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert_rl_readiness_summary(summary)


def test_readiness_promotes_only_after_signal_and_updates_are_realized(
    tmp_path: Path,
) -> None:
    config_path = _write_inputs(tmp_path, _passing_scorecard(), _passing_preflight())

    summary = reproduce_historical_rl_readiness_v1(config_path)

    assert all(gate["passed"] for gate in summary["gates"])
    assert summary["decision"] == {
        "status": "ready_for_production_rl",
        "promotion_allowed": True,
        "learnable_signal_observed": True,
        "recommended_action": "promote_to_production_rl",
        "failed_gate_ids": [],
    }


def test_global_reward_diversity_does_not_replace_informative_groups(
    tmp_path: Path,
) -> None:
    preflight = _preflight(
        status="failed",
        syntax=24,
        complete_parser=20,
        schema_valid=8,
        exact=4,
        rewards=[(0.0, 16), (1.1, 16)],
        informative_groups=0,
        realized_updates=0,
    )
    config_path = _write_inputs(tmp_path, _passing_scorecard(), preflight)

    summary = reproduce_historical_rl_readiness_v1(config_path)
    gates = _gates(summary)

    assert gates["reward_diversity"]["passed"] is True
    assert gates["informative_groups"]["passed"] is False
    assert gates["realized_optimizer_updates"]["passed"] is False
    assert summary["decision"]["learnable_signal_observed"] is False
    assert summary["decision"]["promotion_allowed"] is False


def test_readiness_rejects_tampered_sealed_evidence(tmp_path: Path) -> None:
    scorecard = _passing_scorecard()
    config_path = _write_inputs(tmp_path, scorecard, _passing_preflight())
    scorecard["benchmark"]["official_bfcl"] = True
    _write_json(tmp_path / "scorecard.json", scorecard)

    with pytest.raises(ValueError, match="self-hash mismatch"):
        reproduce_historical_rl_readiness_v1(config_path)


def test_readiness_rejects_declared_train_eval_leakage(tmp_path: Path) -> None:
    preflight = _passing_preflight()
    preflight["metrics"]["data"]["train_artifacts"][0]["jsonl"]["sha256"] = _EVAL_JSONL_SHA256
    preflight.pop("receipt_self_sha256")
    preflight["receipt_self_sha256"] = _canonical_sha256(preflight)
    config_path = _write_inputs(tmp_path, _passing_scorecard(), preflight)

    with pytest.raises(ValueError, match="train/eval leakage"):
        reproduce_historical_rl_readiness_v1(config_path)


def test_readiness_config_cannot_allow_constant_rewards(tmp_path: Path) -> None:
    thresholds = _thresholds()
    thresholds["rl_preflight"]["min_reward_unique_values"] = 1
    config_path = _write_inputs(
        tmp_path,
        _passing_scorecard(),
        _passing_preflight(),
        thresholds=thresholds,
    )

    with pytest.raises(ValueError, match="must be at least 2"):
        reproduce_historical_rl_readiness_v1(config_path)


def test_summary_write_is_deterministic_and_self_hash_detects_drift(
    tmp_path: Path,
) -> None:
    config_path = _write_inputs(tmp_path, _passing_scorecard(), _passing_preflight())
    summary = reproduce_historical_rl_readiness_v1(config_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_rl_readiness_summary(summary, first)
    write_rl_readiness_summary(summary, second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == summary
    tampered = copy.deepcopy(summary)
    tampered["decision"]["promotion_allowed"] = False
    with pytest.raises(ValueError, match="self-hash mismatch"):
        assert_rl_readiness_summary(tampered)


def test_schema_v2_uses_tool_conditioned_exactness_and_sft_non_inferiority(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path)

    summary = summarize_rl_readiness(config_path)
    gates = _gates(summary)

    assert summary["schema_version"] == SCHEMA_VERSION
    assert "scorecard_action_exact" not in gates
    assert gates["scorecard_tool_format_successes"]["passed"] is True
    assert gates["scorecard_schema_valid_tool_successes"]["passed"] is True
    assert gates["scorecard_tool_name_case_exact_successes"]["passed"] is True
    assert gates["scorecard_whole_call_exact_successes"]["passed"] is True
    assert gates["scorecard_abstention_successes"]["passed"] is True
    assert gates["sft_mean_loss_non_inferiority"]["passed"] is True
    assert gates["sft_token_accuracy_non_inferiority"]["passed"] is True
    assert gates["production_learning_rate_sequence"]["passed"] is True
    assert gates["nonzero_learning_rate_executed"]["passed"] is True
    assert gates["final_optimizer_learning_rate"]["passed"] is True
    assert gates["policy_tensor_transition"]["passed"] is True
    assert summary["funnel"]["greedy_heldout"]["action_exact_promotion_evidence"] is False
    assert summary["production"]["rl_config"]["canonical_sha256"]
    assert summary["production"]["parent_checkpoint"]["sha256"]
    assert summary["production"]["execution"] == {
        "production_requested_device": "cpu",
        "production_requested_dtype": "fp32",
        "preflight_requested_device": "cpu",
        "preflight_requested_dtype": "fp32",
        "resolved_device": "cpu",
        "resolved_dtype": "fp32",
    }
    assert summary["decision"]["promotion_allowed"] is True
    assert_rl_readiness_summary(summary)


@pytest.mark.parametrize("invalid_schema_version", [2.0, True])
def test_schema_versions_require_non_boolean_integers(
    tmp_path: Path,
    invalid_schema_version: object,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path)
    summary = summarize_rl_readiness(config_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = invalid_schema_version
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported RL-readiness config"):
        summarize_rl_readiness(config_path)

    summary["schema_version"] = invalid_schema_version
    summary.pop("summary_self_sha256")
    summary["summary_self_sha256"] = _canonical_sha256(summary)
    with pytest.raises(ValueError, match="unsupported RL-readiness summary"):
        assert_rl_readiness_summary(summary)


@pytest.mark.parametrize("evidence_kind", ["scorecard", "rl_preflight", "sft_checkpoint_sweep"])
@pytest.mark.parametrize("version_kind", ["float", "bool"])
def test_schema_v2_rejects_non_integer_evidence_schema_versions(
    tmp_path: Path,
    evidence_kind: str,
    version_kind: str,
) -> None:
    config_path, teacher_path = _write_v2_inputs(
        tmp_path,
        teacher_evidence=(
            "sweep" if evidence_kind == "sft_checkpoint_sweep" else "metrics"
        ),
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if evidence_kind == "scorecard":
        evidence_path = Path(config["evidence"]["scorecard"]["path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["schema_version"] = 1.0 if version_kind == "float" else True
        _rehash_scorecard(evidence)
        configured_hash_key = "expected_self_sha256"
        evidence_hash_key = "result_self_sha256"
    elif evidence_kind == "rl_preflight":
        evidence_path = Path(config["evidence"]["rl_preflight"]["path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["schema_version"] = 1.0 if version_kind == "float" else True
        _rehash_preflight(evidence)
        configured_hash_key = "expected_self_sha256"
        evidence_hash_key = "receipt_self_sha256"
    else:
        evidence_path = teacher_path
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["schema_version"] = 2.0 if version_kind == "float" else True
        evidence.pop("result_sha256")
        evidence["result_sha256"] = _canonical_sha256(evidence)
        configured_hash_key = "expected_self_sha256"
        evidence_hash_key = "result_sha256"
    _write_json(evidence_path, evidence)
    config["evidence"][evidence_kind][configured_hash_key] = evidence[evidence_hash_key]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    expected_message = {
        "scorecard": "unsupported internal scorecard",
        "rl_preflight": "unsupported one-update RL preflight",
        "sft_checkpoint_sweep": "unsupported SFT checkpoint sweep",
    }[evidence_kind]
    with pytest.raises(ValueError, match=expected_message):
        summarize_rl_readiness(config_path)


def test_schema_v2_rejects_drifted_production_config_and_parent(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    production_config_path = Path(config["production"]["rl_config"]["path"])
    production_config = yaml.safe_load(
        production_config_path.read_text(encoding="utf-8")
    )
    production_config["optim"]["lr"] *= 2
    production_config_path.write_text(
        yaml.safe_dump(production_config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config canonical SHA-256"):
        summarize_rl_readiness(config_path)

    config_path, _ = _write_v2_inputs(tmp_path / "parent")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent_path = Path(config["production"]["parent_checkpoint"]["path"])
    parent_path.write_bytes(parent_path.read_bytes() + b"-drift")

    with pytest.raises(ValueError, match="parent checkpoint SHA-256"):
        summarize_rl_readiness(config_path)


def test_schema_v2_rejects_preflight_config_and_execution_binding_drift(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path / "config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    production_config_path = Path(config["production"]["rl_config"]["path"])
    production_config = yaml.safe_load(
        production_config_path.read_text(encoding="utf-8")
    )
    production_config["optim"]["lr"] *= 2
    production_config_path.write_text(
        yaml.safe_dump(production_config, sort_keys=False),
        encoding="utf-8",
    )
    config["production"]["rl_config"]["expected_canonical_sha256"] = (
        _canonical_sha256(production_config)
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the preflight source config"):
        summarize_rl_readiness(config_path)

    config_path, _ = _write_v2_inputs(tmp_path / "execution")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["production"]["execution"]["preflight_requested_device"] = "mps"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight execution identity"):
        summarize_rl_readiness(config_path)


@pytest.mark.parametrize("failure", ["unchanged_policy", "zero_learning_rate"])
def test_schema_v2_seals_failed_policy_update_as_not_ready(
    tmp_path: Path,
    failure: str,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preflight_path = Path(config["evidence"]["rl_preflight"]["path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    transition = preflight["measurement"]["policy_transition"]
    preflight["status"] = "failed"
    preflight["validation_errors"] = [f"test {failure} rejection"]
    preflight["error"] = {
        "type": "RLPreflightValidationError",
        "message": preflight["validation_errors"][0],
    }
    if failure == "unchanged_policy":
        transition.update(
            {
                "changed_model_parameter_names": [],
                "changed_model_parameter_count": 0,
                "first_changed_model_parameter": None,
                "final_model_state_sha256": transition[
                    "initial_model_state_sha256"
                ],
                "at_least_one_policy_tensor_changed": False,
            }
        )
    else:
        transition.update(
            {
                "actual_learning_rates": [0.0],
                "actual_learning_rates_match_expected": False,
                "nonzero_learning_rate_executed": False,
                "final_optimizer_learning_rates": [0.0],
                "final_optimizer_learning_rate_matches_expected": False,
            }
        )
        preflight["metrics"]["rl_accounting"]["learning_rate_history"] = [0.0]
    _rewrite_v2_preflight(config_path, preflight)

    summary = summarize_rl_readiness(config_path)
    gates = _gates(summary)

    assert summary["decision"]["promotion_allowed"] is False
    assert gates["preflight_status"]["passed"] is False
    if failure == "unchanged_policy":
        assert gates["policy_tensor_transition"]["passed"] is False
    else:
        assert gates["production_learning_rate_sequence"]["passed"] is False
        assert gates["nonzero_learning_rate_executed"]["passed"] is False
        assert gates["final_optimizer_learning_rate"]["passed"] is False
    assert_rl_readiness_summary(summary)


def test_schema_v2_cannot_promote_abstention_only_action_exact(tmp_path: Path) -> None:
    config_path, _ = _write_v2_inputs(tmp_path, tool_exact=False)

    summary = summarize_rl_readiness(config_path)
    gates = _gates(summary)

    assert summary["funnel"]["greedy_heldout"]["action_exact"]["count"] == 80
    assert gates["scorecard_abstention_successes"]["passed"] is True
    assert gates["scorecard_tool_format_successes"]["passed"] is False
    assert gates["scorecard_schema_valid_tool_successes"]["passed"] is False
    assert gates["scorecard_tool_name_case_exact_successes"]["passed"] is False
    assert gates["scorecard_whole_call_exact_successes"]["passed"] is False
    assert summary["decision"]["promotion_allowed"] is False


def test_schema_v2_zero_tool_decisions_is_sealed_not_ready(tmp_path: Path) -> None:
    config_path, _ = _write_v2_inputs(tmp_path, zero_tool_decisions=True)

    summary = summarize_rl_readiness(config_path)
    gates = _gates(summary)

    assert summary["funnel"]["greedy_heldout"]["tool_decisions"] == 0
    assert (
        summary["funnel"]["greedy_heldout"]["tool_format_validity_on_tool_decisions"]["rate"]
        is None
    )
    assert gates["scorecard_tool_format_rate"]["observed"] is None
    assert gates["scorecard_tool_format_rate"]["passed"] is False
    assert gates["scorecard_whole_call_exact_rate"]["passed"] is False
    assert summary["decision"]["promotion_allowed"] is False
    assert_rl_readiness_summary(summary)


def test_schema_v2_rejects_teacher_forced_regression_even_when_actions_pass(
    tmp_path: Path,
) -> None:
    config_path, metrics_path = _write_v2_inputs(tmp_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["heldout_eval"]["post"]["mean_loss"] = 3.6
    metrics["heldout_eval"]["post"]["assistant_token_accuracy"] = 0.4
    metrics["heldout_eval"]["delta"]["mean_loss"] = 1.6
    metrics["heldout_eval"]["delta"]["assistant_token_accuracy"] = -0.3
    _write_json(metrics_path, metrics)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence"]["sft_metrics"]["expected_sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = summarize_rl_readiness(config_path)
    gates = _gates(summary)

    assert gates["scorecard_whole_call_exact_successes"]["passed"] is True
    assert gates["sft_mean_loss_non_inferiority"]["observed"] == pytest.approx(1.6)
    assert gates["sft_mean_loss_non_inferiority"]["passed"] is False
    assert gates["sft_token_accuracy_non_inferiority"]["observed"] == pytest.approx(0.3)
    assert gates["sft_token_accuracy_non_inferiority"]["passed"] is False
    assert summary["decision"]["promotion_allowed"] is False


def test_schema_v2_rejects_unpinned_sft_metrics_drift(tmp_path: Path) -> None:
    config_path, metrics_path = _write_v2_inputs(tmp_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["heldout_eval"]["post"]["mean_loss"] = 9.0
    _write_json(metrics_path, metrics)

    with pytest.raises(ValueError, match="does not match the configured identity"):
        summarize_rl_readiness(config_path)


def test_schema_v2_accepts_exact_sweep_selected_checkpoint_evidence(
    tmp_path: Path,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path, teacher_evidence="sweep")

    summary = summarize_rl_readiness(config_path)

    assert summary["contract"]["teacher_forced_evidence_kind"] == "checkpoint_sweep"
    sweep = summary["evidence"]["sft_checkpoint_sweep"]
    assert sweep["selected_checkpoint_retention_eligible"] is True
    assert (
        sweep["selected_checkpoint"]["sha256"]
        == summary["evidence"]["scorecard"]["checkpoint_sha256"]
    )
    assert summary["funnel"]["teacher_forced_sft"]["evidence_kind"] == "checkpoint_sweep"
    assert summary["decision"]["promotion_allowed"] is True
    assert_rl_readiness_summary(summary)


def test_schema_v2_sweep_rejects_baseline_post_token_count_drift(
    tmp_path: Path,
) -> None:
    config_path, sweep_path = _write_v2_inputs(
        tmp_path,
        teacher_evidence="sweep",
    )
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["checkpoints"][0]["metrics"]["assistant_loss_tokens"] = 4095
    sweep["summary"]["best_retention_eligible_checkpoint"]["metrics"]["assistant_loss_tokens"] = (
        4095
    )
    sweep.pop("result_sha256")
    sweep["result_sha256"] = _canonical_sha256(sweep)
    _write_json(sweep_path, sweep)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence"]["sft_checkpoint_sweep"]["expected_self_sha256"] = sweep["result_sha256"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="assistant_loss_tokens changed"):
        summarize_rl_readiness(config_path)


def test_schema_v2_sweep_rejects_checkpoint_identity_drift(tmp_path: Path) -> None:
    config_path, sweep_path = _write_v2_inputs(
        tmp_path,
        teacher_evidence="sweep",
    )
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep["checkpoints"][0]["artifact"]["bytes"] += 1
    sweep["summary"]["best_retention_eligible_checkpoint"]["artifact"]["bytes"] += 1
    sweep.pop("result_sha256")
    sweep["result_sha256"] = _canonical_sha256(sweep)
    _write_json(sweep_path, sweep)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence"]["sft_checkpoint_sweep"]["expected_self_sha256"] = sweep["result_sha256"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="path/bytes/SHA-256 changed"):
        summarize_rl_readiness(config_path)


def test_guarded_rl_entry_revalidates_summary_and_passes_exact_bindings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path, _ = _write_v2_inputs(tmp_path)
    summary = summarize_rl_readiness(config_path)
    summary_path = tmp_path / "ready.json"
    write_rl_readiness_summary(summary, summary_path)
    loaded = load_rl_readiness_summary(summary_path)
    assert loaded == summary
    out_dir = Path(summary["production"]["out_dir"])
    calls: list[dict] = []

    def fake_run(config_path: str, **kwargs) -> None:
        assert not out_dir.exists()
        calls.append({"config_path": config_path, **kwargs})

    monkeypatch.setattr("localagent.train.rl.run", fake_run)

    result = run_ready_rl(summary_path)

    assert len(calls) == 1
    assert calls[0] == {
        "config_path": summary["production"]["rl_config"]["reference"],
        "resume": False,
        "_expected_config_canonical_sha256": summary["production"]["rl_config"][
            "canonical_sha256"
        ],
        "_expected_parent_checkpoint_sha256": summary["production"][
            "parent_checkpoint"
        ]["sha256"],
        "_expected_execution": {
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "requested_dtype": "fp32",
            "resolved_dtype": "fp32",
        },
        "_require_fresh_output_dir": True,
    }
    assert result["readiness_summary_sha256"] == summary["summary_self_sha256"]
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "invalidity",
    [
        "missing",
        "tampered",
        "stale_config",
        "stale_parent",
        "execution_drift",
        "not_ready",
    ],
)
def test_guarded_rl_entry_rejects_invalid_or_stale_readiness_before_output(
    tmp_path: Path,
    monkeypatch,
    invalidity: str,
) -> None:
    config_path, _ = _write_v2_inputs(
        tmp_path,
        tool_exact=invalidity != "not_ready",
    )
    summary = summarize_rl_readiness(config_path)
    summary_path = tmp_path / "ready.json"
    out_dir = Path(summary["production"]["out_dir"])
    calls: list[object] = []
    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    if invalidity == "missing":
        candidate = tmp_path / "missing.json"
    elif invalidity == "tampered":
        summary["decision"]["promotion_allowed"] = False
        _write_json(summary_path, summary)
        candidate = summary_path
    else:
        write_rl_readiness_summary(summary, summary_path)
        candidate = summary_path
        if invalidity == "stale_config":
            production_config_path = Path(
                summary["production"]["rl_config"]["reference"]
            )
            production_config = yaml.safe_load(
                production_config_path.read_text(encoding="utf-8")
            )
            production_config["optim"]["lr"] *= 2
            production_config_path.write_text(
                yaml.safe_dump(production_config, sort_keys=False),
                encoding="utf-8",
            )
        elif invalidity == "stale_parent":
            parent_path = Path(
                summary["production"]["parent_checkpoint"]["reference"]
            )
            parent_path.write_bytes(parent_path.read_bytes() + b"-drift")
        elif invalidity == "execution_drift":
            monkeypatch.setattr(
                "localagent.train.device.resolve_device",
                lambda _requested: torch.device("mps"),
            )

    with pytest.raises((FileNotFoundError, ValueError)):
        run_ready_rl(candidate)

    assert calls == []
    assert not out_dir.exists()


def test_guarded_rl_entry_never_authorizes_historical_schema_v1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_inputs(
        tmp_path,
        _passing_scorecard(),
        _passing_preflight(),
    )
    summary = reproduce_historical_rl_readiness_v1(config_path)
    summary_path = tmp_path / "historical.json"
    write_rl_readiness_summary(summary, summary_path)
    calls: list[object] = []
    monkeypatch.setattr(
        "localagent.train.rl.run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="historical analysis-only"):
        run_ready_rl(summary_path)

    assert calls == []


def test_production_summarizer_refuses_schema_v1(tmp_path: Path) -> None:
    config_path = _write_inputs(tmp_path, _passing_scorecard(), _passing_preflight())

    with pytest.raises(ValueError, match="historical verification-only"):
        summarize_rl_readiness(config_path)

    historical = reproduce_historical_rl_readiness_v1(config_path)
    assert historical["schema_version"] == LEGACY_SCHEMA_VERSION
    assert historical["decision"]["promotion_allowed"] is True


@pytest.mark.skipif(
    not all(path.is_file() for path in (*SEALED_V1_INPUTS, SEALED_V1_SUMMARY)),
    reason="requires the exact local sealed schema-v1 summary, scorecard, and RL preflight",
)
def test_schema_v1_sealed_summary_remains_verifiable() -> None:
    summary = json.loads(SEALED_V1_SUMMARY.read_text(encoding="utf-8"))

    assert summary["schema_version"] == LEGACY_SCHEMA_VERSION
    assert_rl_readiness_summary(summary)
    reproduced = reproduce_historical_rl_readiness_v1(
        "configs/eval/paper-tier-1m-rl-readiness.yaml"
    )
    assert reproduced == summary


@pytest.mark.skipif(
    not all(path.is_file() for path in SEALED_V1_INPUTS),
    reason="requires the exact local sealed schema-v1 scorecard and RL preflight",
)
def test_cli_requires_explicit_historical_v1_route() -> None:
    command = [
        sys.executable,
        "scripts/summarize_rl_readiness.py",
        "configs/eval/paper-tier-1m-rl-readiness.yaml",
    ]

    production = subprocess.run(command, check=False, capture_output=True, text=True)
    assert production.returncode != 0
    assert "historical verification-only" in production.stderr

    historical = subprocess.run(
        [*command, "--historical-verify-v1"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert historical.returncode == 0
    assert json.loads(historical.stdout)["summary_self_sha256"] == (
        "264a4f47939824c35942e012f7ba33da05bc94cd2dc156d268a455413e243812"
    )

    invalid_promotion = subprocess.run(
        [*command, "--historical-verify-v1", "--require-ready"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid_promotion.returncode != 0
    assert "cannot be combined" in invalid_promotion.stderr
