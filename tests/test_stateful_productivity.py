import hashlib
import json
from pathlib import Path

from localagent.data.stateful_productivity import (
    SUITE_ID,
    StatefulRuntime,
    apply_action,
    build_tasks,
    canonical_json,
    conversation_for_task,
    initial_state,
    state_prompt,
    stateful_reward,
    stateful_reward_spec,
    suite_inventory,
    task_complete,
    task_prompts,
)
from localagent.data.schema import Role


def test_train_and_eval_tasks_are_slot_and_prompt_disjoint() -> None:
    train = build_tasks("train")
    evaluation = build_tasks("eval")
    train_text = "\n".join(task.goal + "\n" + "\n".join(task_prompts(task)) for task in train)
    eval_text = "\n".join(task.goal + "\n" + "\n".join(task_prompts(task)) for task in evaluation)
    assert train_text != eval_text
    assert "maya@example.com" in train_text
    assert "zoe@example.com" in eval_text
    assert "zoe@example.com" not in train_text
    assert "maya@example.com" not in eval_text


def test_email_notion_browser_and_recovery_transitions() -> None:
    for task in build_tasks("eval"):
        state = initial_state()
        for index, action in enumerate(task.actions):
            result = apply_action(task, index, state, action.tool, action.arguments)
            assert result.schema_valid
            assert result.exact_action
            assert result.state_transition, (task.task_id, index, result.error)
            state = result.state
        assert task_complete(task, state), task.task_id


def test_recovery_keeps_intermediate_error_and_abstention_is_noop() -> None:
    recovery = next(task for task in build_tasks("train") if task.family == "recovery")
    state = initial_state()
    first = apply_action(recovery, 0, state, "open_url", recovery.actions[0].arguments)
    assert first.state["browser"]["last_error"]
    assert first.state["browser"]["page"] is None
    abstain = next(task for task in build_tasks("train") if task.family == "abstention")
    result = apply_action(abstain, 0, initial_state(), None, {})
    assert result.closed_loop_success
    assert result.state == initial_state()
    assert stateful_reward(result, terminal=True) == 1.0
    assert sum(stateful_reward_spec().values()) == 1.0


def test_recovery_urls_are_visible_and_numeric_arguments_are_semantically_equal() -> None:
    recovery = next(task for task in build_tasks("eval") if task.family == "recovery")
    prompts = task_prompts(recovery)
    assert "https://local.test/retired" in prompts[0]
    assert "https://local.test/reports" in prompts[1]
    click = next(task for task in build_tasks("eval") if task.family == "email").actions[1]
    result = apply_action(
        next(task for task in build_tasks("eval") if task.family == "email"),
        1,
        initial_state(),
        click.tool,
        {"x": 120.0, "y": 220.0},
    )
    assert result.exact_args


def test_runtime_retries_without_advancing_and_oracle_completes() -> None:
    task = next(task for task in build_tasks("eval") if task.family == "notion")
    runtime = StatefulRuntime(task)
    rejected = runtime.execute("notion_create_page", {"title": "wrong", "content": "wrong"})
    assert not rejected.closed_loop_success
    assert runtime.step_index == 0
    assert runtime.prompt().endswith("error=action_mismatch")
    for action in task.actions:
        result = runtime.execute(action.tool, action.arguments)
        assert result.closed_loop_success
    assert runtime.done
    assert runtime.step_index == len(task.actions)
    assert len(runtime.events) == len(task.actions) + 1


def test_conversation_uses_canonical_schema_and_state_prompt() -> None:
    task = next(task for task in build_tasks("eval") if task.family == "notion")
    conversation = conversation_for_task(task)
    assert conversation.meta["suite"] == SUITE_ID
    assert conversation.messages[0].role == Role.user
    assert any(message.role == Role.tool for message in conversation.messages)
    prompt = state_prompt(task, 0, task.initial_state)
    assert canonical_json(task.initial_state) in prompt
    assert "notion_create_page" not in prompt
    assert len(conversation.tools) >= 60


def test_suite_inventory_is_balanced_and_prompt_free() -> None:
    inventory = suite_inventory("eval")
    assert inventory == {
        "suite": SUITE_ID,
        "split": "eval",
        "tasks": 5,
        "families": {"abstention": 1, "browser": 1, "email": 1, "notion": 1, "recovery": 1},
        "steps": 16,
        "recovery_tasks": 1,
        "abstention_tasks": 1,
        "conversation_rows": 5,
    }


def test_published_stateful_probe_receipt_is_self_hashed_and_negative() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m55-stateful-productivity-transfer-v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["source"]["public_benchmark_text_used"] is False
    assert receipt["arms"]["pretrained_frozen_backbone"]["closed_loop"]["task_complete_rate"] == 0.2
    assert receipt["arms"]["pretrained_frozen_backbone"]["closed_loop"]["recovery_task_complete_rate"] == 0.0


def test_published_lowrate_transfer_receipt_binds_all_three_arms() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m56-stateful-productivity-transfer-ablation-v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert set(receipt["arms"]) == {
        "pretrained_frozen_backbone",
        "pretrained_lowrate_unfrozen_backbone",
        "matched_random_backbone",
    }
    lowrate = receipt["arms"]["pretrained_lowrate_unfrozen_backbone"]
    assert lowrate["weight_movement"]["backbone"] > 0.0
    assert lowrate["closed_loop"]["task_complete_rate"] == 0.2
    assert receipt["source"]["native_runtime_executed"] is False


def test_published_lowrate_deployment_receipts_fail_closed_on_public_claims() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    web = json.loads((root / "m57-stateful-webgpu-deploy-verification-v1.json").read_text())
    hub = json.loads((root / "m58-stateful-hf-local-export-v1.json").read_text())
    assert web["verified"] is True
    assert web["parity_gate"]["passed"] is True
    assert web["hub"] == {
        "authenticated": False,
        "uploaded": False,
        "reason": "hf auth whoami reports no login; no public model or Space URL is claimed",
    }
    assert hub["export_verified_locally"] is True
    assert hub["hub"]["uploaded"] is False
    assert hub["parameter_count"] == 10524544


def test_published_runtime_receipt_has_oracle_and_model_boundaries() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m66-stateful-runtime-evaluation-v1.json"
    receipt = json.loads(path.read_text())
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["runtime"] == {
        "environment_executed": True,
        "external_accounts_used": False,
        "kind": "local_resettable_state_machine",
        "public_benchmark": False,
        "tool_side_effects": "in_memory_only",
    }
    assert receipt["oracle"]["task_complete_rate"] == 1.0
    assert receipt["model"]["task_complete_rate"] == 0.2
    assert receipt["model"]["accepted_steps"] == 4


def test_published_stateful_grpo_receipt_records_real_updates_and_negative_accuracy() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m67-stateful-productivity-grpo-v1.json"
    receipt = json.loads(path.read_text())
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["configuration"]["reward_environment"] == "stateful_productivity"
    assert receipt["training"]["rl_accounting"]["informative_groups"] == 5
    assert receipt["training"]["rl_accounting"]["realized_optimizer_updates"] == 4
    assert receipt["training"]["exact_match_accuracy_post"] == 0.0
    assert receipt["source"]["native_runtime_executed"] is False


def test_published_public_adaptation_runtime_comparison_is_self_hashed_and_fail_closed() -> None:
    path = Path(__file__).parents[1] / (
        "docs/paper/results/raw/m71-public-adaptation-stateful-runtime-comparison-v1.json"
    )
    receipt = json.loads(path.read_text())
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["source"] == {
        "external_accounts_used": False,
        "native_runtime_executed": False,
        "public_benchmark_text_used": True,
        "public_datasets": ["xlangai/AgentNet", "osunlp/Mind2Web"],
        "runtime_kind": "local_resettable_state_machine",
        "tool_side_effects": "in_memory_only",
    }
    assert receipt["oracle"]["task_complete_rate"] == 1.0
    assert receipt["oracle"]["accepted_steps"] == 16
    for arm in ("m69", "m70"):
        assert receipt["arms"][arm]["model"]["task_complete_rate"] == 0.0
        assert receipt["arms"][arm]["model"]["accepted_steps"] == 0
    assert receipt["comparison"] == {
        "accepted_steps_m69": 0,
        "accepted_steps_m70": 0,
        "oracle_task_complete_rate": 1.0,
        "same_model_event_sha256": True,
        "task_complete_rate_m69": 0.0,
        "task_complete_rate_m70": 0.0,
    }


def test_published_stateful_adoption_receipts_prove_closed_loop_pass_and_claim_boundary() -> None:
    root = Path(__file__).parents[1] / "docs/paper/results/raw"
    adoption = json.loads((root / "m89-public-to-stateful-closed-loop-adoption-v1.json").read_text())
    expected = adoption.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(adoption, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    runtime = json.loads((root / "m89-stateful-runtime-public-adaptation-v2.json").read_text())
    runtime_expected = runtime.pop("receipt_self_sha256")
    runtime_actual = hashlib.sha256(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert runtime_actual == runtime_expected
    assert runtime["model"]["task_complete_rate"] == 1.0
    assert runtime["model"]["accepted_steps"] == runtime["model"]["expected_steps"] == 16
    assert runtime["model"]["attempts"] == 20
    assert adoption["comparison"] == {
        "accepted_steps_delta_vs_m71_m70": 16,
        "all_steps_accepted": True,
        "task_complete_rate_delta_vs_m71_m70": 1.0,
        "workflow_passed": True,
    }
    assert adoption["source"]["native_runtime_executed"] is False
    assert "official" in adoption["claim_boundary"]
