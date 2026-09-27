from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from localagent.data.schema import Conversation, Message, Role
from localagent.eval.stage_pilot_summary import (
    STAGE_ORDER,
    StagePilotInput,
    summarize_stage_pilot,
    write_stage_pilot_summary,
)
from localagent.train.stage_data import canonical_sha256, file_identity, sha256_file


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lineage(
    stage: str,
    *,
    config_sha256: str,
    data_sha256: str,
    model_sha256: str,
    tokenizer_sha256: str,
    parent_sha256: str,
) -> dict:
    return {
        "version": 1,
        "stage": stage,
        "config_sha256": config_sha256,
        "model_config_sha256": model_sha256,
        "data_sha256": data_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "parent_checkpoint_sha256": parent_sha256,
        "git": {
            "commit": "a" * 40,
            "repository_sha256": _sha256("repository"),
            "dirty": False,
            "worktree_sha256": _sha256("tree"),
        },
    }


def _config_sha256(config: dict) -> str:
    normalized = copy.deepcopy(config)
    runtime = normalized.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume", None)
    return canonical_sha256(normalized)


def _heldout_contracts(eval_path: Path) -> dict[str, dict]:
    conversation = Conversation(
        messages=[
            Message(role=Role.user, content="Say ready"),
            Message(role=Role.assistant, content="ready"),
        ]
    )
    eval_path.write_text(conversation.to_json() + "\n", encoding="utf-8")
    payload = json.loads(conversation.to_json())
    payload.pop("meta", None)
    row_sha256 = canonical_sha256(payload)
    sft_dataset_sha256 = canonical_sha256([row_sha256])
    rl_dataset_sha256 = hashlib.sha256(row_sha256.encode("ascii")).hexdigest()
    return {
        "midtrain": {
            "contract": {
                "kind": "deterministic_teacher_forced_next_token",
                "sources": ["agent_eval"],
                "same_draws_pre_post": True,
            },
            "pre": {"mean_loss": 2.0, "token_accuracy": 0.2},
            "post": {"mean_loss": 1.8, "token_accuracy": 0.3},
            "delta": {"mean_loss": -0.2, "token_accuracy": 0.1},
        },
        "sft": {
            "contract": {
                "kind": "deterministic_teacher_forced_assistant_tokens",
                "dataset_sha256": sft_dataset_sha256,
                "same_rows_pre_post": True,
            },
            "pre": {"mean_loss": 1.8, "assistant_token_accuracy": 0.3},
            "post": {"mean_loss": 1.4, "assistant_token_accuracy": 0.5},
            "delta": {"mean_loss": -0.4, "assistant_token_accuracy": 0.2},
        },
        "rl": {
            "contract": {
                "split": "explicit_disjoint_eval_conversations",
                "dataset_sha256": rl_dataset_sha256,
                "same_rows_pre_post": True,
            },
            "pre": {"mean_reward": 0.2, "exact_match_accuracy": 0.2},
            "post": {"mean_reward": 0.3, "exact_match_accuracy": 0.3},
            "delta": {"mean_reward": 0.1, "exact_match_accuracy": 0.1},
        },
    }


def _token_accounting(input_tokens: int, loss_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "loss_tokens": loss_tokens,
        "sources": {
            "fixture": {
                "input_tokens": input_tokens,
                "loss_tokens": loss_tokens,
                "rows": 2,
            }
        },
    }


def _pilot_inputs(
    tmp_path: Path,
    *,
    sft_parent_mismatch: bool = False,
    missing_execution_stage: str | None = None,
    nonfinite_midtrain_loss: bool = False,
    realized_optimizer_updates: int = 2,
) -> list[StagePilotInput]:
    eval_path = tmp_path / "eval.jsonl"
    heldout = _heldout_contracts(eval_path)
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        Conversation(
            messages=[
                Message(role=Role.user, content="Say training"),
                Message(role=Role.assistant, content="training"),
            ]
        ).to_json()
        + "\n",
        encoding="utf-8",
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"kind":"fixture"}\n', encoding="utf-8")
    eval_identity = {"path": str(eval_path), **file_identity(eval_path)}
    train_identity = {"path": str(train_path), **file_identity(train_path)}
    model_config = {"name": "pilot-fixture", "vocab_size": 256, "d_model": 16}
    model_sha256 = canonical_sha256(model_config)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    tokenizer = {
        "kind": "bpe",
        "path": str(tokenizer_path),
        "sha256": tokenizer_sha256,
    }
    execution = {
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "requested_dtype": "fp32",
        "resolved_dtype": "fp32",
    }
    midtrain_accounting = _token_accounting(40, 32)
    sft_accounting = _token_accounting(24, 12)
    invalidated_heads = ["tool_head", "ptr_head", "route_head", "dense_selector"]
    reward_contract = {
        "environment": "canonical_toolcalls",
        "correctness": "exact normalized tool AST; exact text match",
        "learned_judge": False,
    }
    policy_contract = {
        "objective": "sampled_token_clipped_grpo",
        "ratio_scope": "generated_tokens_only",
    }
    rl_accounting = {
        "attempted_rollout_steps": 2,
        "attempted_groups": 4,
        "attempted_rollouts": 8,
        "zero_signal_steps": int(realized_optimizer_updates == 0),
        "informative_groups": int(realized_optimizer_updates > 0),
        "realized_optimizer_updates": realized_optimizer_updates,
    }

    midtrain_dir = tmp_path / "midtrain"
    sft_dir = tmp_path / "sft"
    rl_dir = tmp_path / "rl"
    for directory in (midtrain_dir, sft_dir, rl_dir):
        directory.mkdir()
    initial_parent = tmp_path / "pretrain.pt"
    initial_parent.write_bytes(b"fixture pretrain checkpoint")
    checkpoints = {
        "midtrain": midtrain_dir / "latest.pt",
        "sft": sft_dir / "latest.pt",
        "rl": rl_dir / "latest.pt",
    }
    metrics_paths = {
        "midtrain": midtrain_dir / "metrics.json",
        "sft": sft_dir / "metrics.json",
        "rl": rl_dir / "metrics.json",
    }
    config_paths = {
        "midtrain": tmp_path / "midtrain.yaml",
        "sft": tmp_path / "sft.yaml",
        "rl": tmp_path / "rl.yaml",
    }
    configs = {
        "midtrain": {
            "stage": "midtrain",
            "init_from": str(initial_parent),
            "data": {
                "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                "sources": [
                    {
                        "name": "fixture_train",
                        "type": "conversations",
                        "path": str(train_path),
                        "weight": 1.0,
                    }
                ],
                "eval_sources": [
                    {
                        "name": "agent_eval",
                        "type": "conversations",
                        "path": str(eval_path),
                    }
                ],
            },
            "runtime": {"resume": True},
        },
        "sft": {
            "stage": "sft",
            "init_from": str(checkpoints["midtrain"]),
            "data": {
                "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
            },
        },
        "rl": {
            "stage": "rl",
            "init_from": str(checkpoints["sft"]),
            "data": {
                "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
                "conversations": [str(train_path)],
                "eval_conversations": [str(eval_path)],
            },
        },
    }
    for stage, config in configs.items():
        config_paths[stage].write_text(
            yaml.safe_dump(config, sort_keys=True),
            encoding="utf-8",
        )

    midtrain_data = {
        "sources": [
            {
                "name": "fixture_train",
                "type": "conversations",
                "split": None,
                "start_weight": 1.0,
                "end_weight": 1.0,
                "artifact": file_identity(train_path),
            }
        ],
        "eval_sources": [
            {
                "name": "agent_eval",
                "type": "conversations",
                "split": None,
                "artifact": file_identity(eval_path),
            }
        ],
        "packed_holdout_audit": {},
    }
    midtrain_lineage = _lineage(
        "midtrain",
        config_sha256=_config_sha256(configs["midtrain"]),
        data_sha256=canonical_sha256(midtrain_data),
        model_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        parent_sha256=sha256_file(initial_parent),
    )
    midtrain_loss = float("nan") if nonfinite_midtrain_loss else 1.25
    midtrain_checkpoint = {
        "stage": "midtrain",
        "cfg": model_config,
        "state_dict": {},
        "tokenizer": tokenizer,
        "loss_history": [1.5, midtrain_loss],
        "token_accounting": midtrain_accounting,
        "lineage": midtrain_lineage,
        "heldout_eval": heldout["midtrain"],
        "execution": execution,
        "data": {
            "sources": [
                {
                    "name": "fixture_train",
                    "type": "conversations",
                    "path": str(train_path),
                    "split": None,
                }
            ],
            "eval_sources": [
                {
                    "name": "agent_eval",
                    "type": "conversations",
                    "path": str(eval_path),
                    "split": None,
                }
            ],
            "packed_holdout_audit": {},
        },
    }
    if missing_execution_stage == "midtrain":
        midtrain_checkpoint.pop("execution")
    torch.save(midtrain_checkpoint, checkpoints["midtrain"])

    sft_data_identity = {
        "conversations": [file_identity(train_path)],
        "eval_conversations": [file_identity(eval_path)],
        "decay_conversations": [],
    }
    sft_lineage = _lineage(
        "sft",
        config_sha256=_config_sha256(configs["sft"]),
        data_sha256=canonical_sha256(sft_data_identity),
        model_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        parent_sha256=(
            _sha256("wrong-parent")
            if sft_parent_mismatch
            else sha256_file(checkpoints["midtrain"])
        ),
    )
    head_state = {"weight": torch.tensor([1.0])}
    sft_data = {
        "paths": [str(train_path)],
        "eval_paths": [str(eval_path)],
    }
    sft_checkpoint = {
        "stage": "sft",
        "cfg": model_config,
        "state_dict": {},
        "tokenizer": tokenizer,
        "loss_history": [1.1, 0.9],
        "token_accounting": sft_accounting,
        "lineage": sft_lineage,
        "heldout_eval": heldout["sft"],
        "execution": execution,
        "data": sft_data,
        "tool_head": head_state,
        "ptr_head": head_state,
        "route_head": head_state,
        "dense_selector": head_state,
    }
    if missing_execution_stage == "sft":
        sft_checkpoint.pop("execution")
    torch.save(sft_checkpoint, checkpoints["sft"])

    split_audit = {
        "eval_scored_rows_sha256": heldout["rl"]["contract"]["dataset_sha256"]
    }
    rl_data_identity = {
        "train_artifacts": [train_identity],
        "eval_artifacts": [eval_identity],
        "split_audit": split_audit,
    }
    rl_lineage = _lineage(
        "rl",
        config_sha256=_config_sha256(configs["rl"]),
        data_sha256=canonical_sha256(rl_data_identity),
        model_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        parent_sha256=sha256_file(checkpoints["sft"]),
    )
    rl_data = {
        "paths": [str(train_path)],
        "eval_paths": [str(eval_path)],
        "train_artifacts": [train_identity],
        "eval_artifacts": [eval_identity],
        "split_audit": split_audit,
    }
    rl_checkpoint = {
        "stage": "rl",
        "cfg": model_config,
        "state_dict": {},
        "tokenizer": tokenizer,
        "reward_history": [0.1, 0.25],
        "rl_accounting": rl_accounting,
        "reward_contract": reward_contract,
        "policy_contract": policy_contract,
        "structured_heads_available": False,
        "invalidated_structured_heads": invalidated_heads,
        "lineage": rl_lineage,
        "heldout_eval": heldout["rl"],
        "execution": execution,
        "data": rl_data,
    }
    if missing_execution_stage == "rl":
        rl_checkpoint.pop("execution")
    torch.save(rl_checkpoint, checkpoints["rl"])

    checkpoint_payloads = {
        "midtrain": midtrain_checkpoint,
        "sft": sft_checkpoint,
        "rl": rl_checkpoint,
    }
    metrics = {
        "midtrain": {
            "stage": "midtrain",
            "checkpoint": str(checkpoints["midtrain"]),
            "checkpoint_identity": file_identity(checkpoints["midtrain"]),
            "loss_last": midtrain_loss,
            "loss_steps": 2,
            "steps_completed": 2,
            "token_accounting": midtrain_accounting,
            "lineage": midtrain_lineage,
            "heldout_eval": heldout["midtrain"],
            "execution": execution,
        },
        "sft": {
            "stage": "sft",
            "checkpoint": str(checkpoints["sft"]),
            "checkpoint_identity": file_identity(checkpoints["sft"]),
            "loss_last": 0.9,
            "loss_steps": 2,
            "token_accounting": sft_accounting,
            "lineage": sft_lineage,
            "heldout_eval": heldout["sft"],
            "execution": execution,
            "data": sft_data,
            "structured_heads": {
                "tool_pointer": True,
                "route": True,
                "dense_selector": True,
            },
        },
        "rl": {
            "stage": "rl",
            "checkpoint": str(checkpoints["rl"]),
            "checkpoint_identity": file_identity(checkpoints["rl"]),
            "mean_reward_last": 0.25,
            "reward_steps": 2,
            "rl_accounting": rl_accounting,
            "reward_contract": reward_contract,
            "policy_contract": policy_contract,
            "structured_heads_available": False,
            "invalidated_structured_heads": invalidated_heads,
            "lineage": rl_lineage,
            "heldout_eval": heldout["rl"],
            "execution": execution,
            "data": rl_data,
        },
    }
    if missing_execution_stage is not None:
        metrics[missing_execution_stage].pop("execution")
        assert "execution" not in checkpoint_payloads[missing_execution_stage]
    for stage in STAGE_ORDER:
        _write_json(metrics_paths[stage], metrics[stage])
    return [
        StagePilotInput(
            stage,
            metrics_paths[stage],
            checkpoints[stage],
            config_paths[stage],
        )
        for stage in STAGE_ORDER
    ]


def test_stage_pilot_summary_happy_path_is_canonical_and_preserves_heldout(tmp_path):
    inputs = _pilot_inputs(tmp_path)

    summary = summarize_stage_pilot(inputs)

    assert summary["validation"]["status"] == "mechanically_valid"
    assert all(summary["validation"]["checks"].values())
    assert [stage["stage"] for stage in summary["stages"]] == list(STAGE_ORDER)
    assert summary["rl_optimization_outcome"] == {
        "classification": "optimizer_updates_realized",
        "attempted_rollouts": 8,
        "realized_optimizer_updates": 2,
    }
    for stage_input, stage_summary in zip(inputs, summary["stages"], strict=True):
        metrics = json.loads(Path(stage_input.metrics_path).read_text(encoding="utf-8"))
        assert stage_summary["heldout_eval"] == metrics["heldout_eval"]
        assert stage_summary["lineage"] == metrics["lineage"]
        assert set(stage_summary["lineage"]) == {
            "version",
            "stage",
            "config_sha256",
            "model_config_sha256",
            "data_sha256",
            "tokenizer_sha256",
            "parent_checkpoint_sha256",
            "git",
        }
        assert stage_summary["lineage"]["git"] == {
            "commit": "a" * 40,
            "repository_sha256": _sha256("repository"),
            "dirty": False,
            "worktree_sha256": _sha256("tree"),
        }
        assert stage_summary["artifacts"]["checkpoint"]["sha256"] == sha256_file(
            stage_input.checkpoint_path
        )
        assert stage_summary["artifacts"]["metrics"]["sha256"] == sha256_file(
            stage_input.metrics_path
        )
        inputs_summary = stage_summary["artifacts"]["inputs"]
        assert inputs_summary["config"]["canonical_sha256"] == metrics["lineage"][
            "config_sha256"
        ]
        assert inputs_summary["config"]["sha256"] == sha256_file(stage_input.config_path)
        assert inputs_summary["tokenizer"]["sha256"] == metrics["lineage"][
            "tokenizer_sha256"
        ]
        assert inputs_summary["parent_checkpoint"]["sha256"] == metrics["lineage"][
            "parent_checkpoint_sha256"
        ]
        assert inputs_summary["data"]["canonical_data_sha256"] == metrics["lineage"][
            "data_sha256"
        ]
        assert len(inputs_summary["data"]["artifacts"]) == 2
        assert all(
            {"path", "bytes", "sha256", "availability", "role", "type"}
            <= artifact.keys()
            for artifact in inputs_summary["data"]["artifacts"]
        )
        assert inputs_summary["data"]["artifact_set_sha256"] == canonical_sha256(
            inputs_summary["data"]["artifacts"]
        )
    assert {item["id"] for item in summary["limitations"]} == {
        "single_seed",
        "offline_canonical_reward",
        "no_browsergym",
        "browser_action_result_separate",
        "artifact_identity_not_publication",
    }
    unsigned = copy.deepcopy(summary)
    recorded_sha256 = unsigned.pop("summary_sha256")
    assert recorded_sha256 == canonical_sha256(unsigned)

    first_output = tmp_path / "summary-first.json"
    second_output = tmp_path / "summary-second.json"
    write_stage_pilot_summary(summary, first_output)
    write_stage_pilot_summary(summary, second_output)
    assert first_output.read_bytes() == second_output.read_bytes()
    assert json.loads(first_output.read_text(encoding="utf-8")) == summary


def test_stage_pilot_summary_rejects_parent_chain_mismatch(tmp_path):
    inputs = _pilot_inputs(tmp_path, sft_parent_mismatch=True)

    with pytest.raises(ValueError, match="SFT parent_checkpoint_sha256"):
        summarize_stage_pilot(inputs)


def test_stage_pilot_summary_classifies_zero_updates_as_zero_signal(tmp_path):
    inputs = _pilot_inputs(tmp_path, realized_optimizer_updates=0)

    summary = summarize_stage_pilot(inputs)

    assert summary["validation"]["status"] == "mechanically_valid"
    assert summary["rl_optimization_outcome"] == {
        "classification": "zero_signal",
        "attempted_rollouts": 8,
        "realized_optimizer_updates": 0,
    }
    assert summary["stages"][2]["optimization"]["classification"] == "zero_signal"


def test_stage_pilot_summary_rejects_missing_execution(tmp_path):
    inputs = _pilot_inputs(tmp_path, missing_execution_stage="sft")

    with pytest.raises(ValueError, match="execution"):
        summarize_stage_pilot(inputs)


def test_stage_pilot_summary_rejects_nonfinite_loss(tmp_path):
    inputs = _pilot_inputs(tmp_path, nonfinite_midtrain_loss=True)

    with pytest.raises(ValueError, match="non-finite"):
        summarize_stage_pilot(inputs)


def test_stage_pilot_summary_rejects_stage_order_mismatch(tmp_path):
    inputs = _pilot_inputs(tmp_path)

    with pytest.raises(ValueError, match="stage order"):
        summarize_stage_pilot([inputs[1], inputs[0], inputs[2]])


def test_stage_pilot_summary_rejects_missing_required_lineage(tmp_path):
    inputs = _pilot_inputs(tmp_path)
    stage_input = inputs[0]
    checkpoint = torch.load(stage_input.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["lineage"]["git"].pop("worktree_sha256")
    torch.save(checkpoint, stage_input.checkpoint_path)
    metrics = json.loads(Path(stage_input.metrics_path).read_text(encoding="utf-8"))
    metrics["lineage"]["git"].pop("worktree_sha256")
    metrics["checkpoint_identity"] = file_identity(stage_input.checkpoint_path)
    _write_json(Path(stage_input.metrics_path), metrics)

    with pytest.raises(ValueError, match="git.worktree_sha256"):
        summarize_stage_pilot(inputs)


def test_stage_pilot_summary_rejects_tampered_data_lineage(tmp_path):
    inputs = _pilot_inputs(tmp_path)
    stage_input = inputs[2]
    checkpoint = torch.load(stage_input.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["lineage"]["data_sha256"] = _sha256("tampered-data")
    torch.save(checkpoint, stage_input.checkpoint_path)
    metrics = json.loads(Path(stage_input.metrics_path).read_text(encoding="utf-8"))
    metrics["lineage"]["data_sha256"] = _sha256("tampered-data")
    metrics["checkpoint_identity"] = file_identity(stage_input.checkpoint_path)
    _write_json(Path(stage_input.metrics_path), metrics)

    with pytest.raises(ValueError, match="canonical data identity"):
        summarize_stage_pilot(inputs)


def test_stage_pilot_summary_rejects_tampered_canonical_config(tmp_path):
    inputs = _pilot_inputs(tmp_path)
    config_path = Path(inputs[1].config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["optim"] = {"lr": 9.9}
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical config"):
        summarize_stage_pilot(inputs)


def test_tracked_seed2027_summary_preserves_all_stage_lineages():
    path = Path("docs/paper/results/webgpu-proxy-pilot-seed2027.summary.json")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "midtrain": {
            "config_sha256": "f72225ad0ef364bac0a664d34acbf5e315820c19b964c82c8c85deadf887c496",
            "data_sha256": "784a290afe86b59d8987907559c46fc98b0ee08d260e7593db2a67d3ab71982a",
            "parent_checkpoint_sha256": (
                "1952ad7f956e02352a395ee9b34d37e893b210f9943935c0bfc3af3c4057bd2b"
            ),
            "artifact_set_sha256": (
                "d323dbd4f053d704c9b3370ebaa48d77ee812dd7a913aa2d6df4a09302d8b394"
            ),
        },
        "sft": {
            "config_sha256": "990cf69db5a91bfaba93776ef5aeee49d916d98aea0eb2f866115241f1eac006",
            "data_sha256": "845221fe3f127f402647477d544c53b6f33876aa118007f5789a7a823d268de1",
            "parent_checkpoint_sha256": (
                "65bbd67ddbab22d07795686ea41627a4453945ef112d30b40025a0cf156aef9a"
            ),
            "artifact_set_sha256": (
                "db86d84d88c0ee134e6cc3d1069bedbce585ebf0ed0d1997b0d3f10824652fd5"
            ),
        },
        "rl": {
            "config_sha256": "78cb54cc78824c4a95b3f095b1e879045c3dd9486b03501aa7d988c9e43ea313",
            "data_sha256": "2b42b1b0ef9e3afd9afc9764dc3f832654d1cbe1c44afaa18063261cfaee309e",
            "parent_checkpoint_sha256": (
                "79387105de75d332413262e8d8ddb847b6cc13bc03f5e4df3c81663d9897aef1"
            ),
            "artifact_set_sha256": (
                "db86d84d88c0ee134e6cc3d1069bedbce585ebf0ed0d1997b0d3f10824652fd5"
            ),
        },
    }
    expected_git = {
        "commit": "dced0ccb93600b2fbe5693796bd14807af0bc3b8",
        "dirty": True,
        "repository_sha256": (
            "4f3f386c8a99593d11431d8772a6bc3b3f67365462192b3068675ebbdee038bf"
        ),
        "worktree_sha256": (
            "79c8ee35595f201343a3093af1976d47e3891a2982aac32ac05dd8c315f782a1"
        ),
    }
    stages = {stage["stage"]: stage for stage in summary["stages"]}
    assert set(stages) == set(expected)
    for stage_name, expected_stage in expected.items():
        stage = stages[stage_name]
        lineage = stage["lineage"]
        for key in ("config_sha256", "data_sha256", "parent_checkpoint_sha256"):
            assert lineage[key] == expected_stage[key]
        assert lineage["git"] == expected_git
        assert (
            stage["artifacts"]["inputs"]["data"]["artifact_set_sha256"]
            == expected_stage["artifact_set_sha256"]
        )
        assert stage["artifacts"]["inputs"]["data"]["artifacts"]
        assert all(
            artifact["availability"] == "local_file_verified"
            for artifact in stage["artifacts"]["inputs"]["data"]["artifacts"]
        )
    unsigned = copy.deepcopy(summary)
    recorded = unsigned.pop("summary_sha256")
    assert recorded == (
        "2bfc209a8b98f5d1cf09554dc5a24b708662a95232321e62089b9af1664b4486"
    )
    assert recorded == canonical_sha256(unsigned)
