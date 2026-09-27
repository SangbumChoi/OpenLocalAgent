import hashlib
import json
from pathlib import Path

from scripts.train_mobile_dispatch_pilot import _synthetic_mobile_samples


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_m125_mobilegym_answer_adaptation_is_hash_bound_and_non_gating() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m125-mobilegym-answer-adaptation-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["native_benchmark_id"] == "mobilegym"
    assert receipt["benchmark_id"] == "mobilegym_model_adaptation_probe"
    assert receipt["native_receipt_eligible"] is False
    assert receipt["environment_executed"] is True
    assert receipt["official_split"] == "test"
    assert receipt["official_split_verified"] is True
    assert receipt["data"]["train_rows"] == 4096
    assert receipt["data"]["eval_rows"] == 904
    assert receipt["training"]["tool_pool_size"] == 63
    arms = receipt["training"]["arms"]
    assert set(arms) == {"warm", "random"}
    for arm in arms.values():
        assert arm["native_success_rate"] == 0.0
        assert arm["native_judge_passed"] is False
        assert arm["native_judge_issue_count"] == 3
        assert arm["weight_report"]["shared_tensors"] == 51
        assert arm["weight_report"]["config_compatible"] is True
        assert arm["weight_report"]["tokenizer_compatible"] is True


def test_mobile_answer_synthetic_rows_have_pointer_supervision() -> None:
    rows = [sample for sample in _synthetic_mobile_samples() if sample.ref_name == "mobile_submit_answer"]
    assert len(rows) == 8
    messages = [json.loads(sample.ref_args)["message"] for sample in rows]
    assert messages == [
        "alpha, beta",
        "one; two",
        "会议、购物清单",
        "第一项，第二项",
        "red, blue",
        "完成",
        "已确认",
        "7",
    ]


def test_m126_pointer_adaptation_is_hash_bound_and_non_gating() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m126-mobilegym-pointer-adaptation-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["native_benchmark_id"] == "mobilegym"
    assert receipt["native_receipt_eligible"] is False
    assert receipt["training"]["pointer_steps"] == 300
    assert receipt["data"]["pointer_train_rows"] == 4225
    for arm in receipt["training"]["arms"].values():
        assert arm["held_out_pointer_exact"] == 0.25
        assert arm["native_success_rate"] == 0.0
        assert arm["native_arguments_nonempty"] is True
        assert arm["weight_report"]["backbone_relative_delta_l2"] == 0.0


def test_m127_notes_sweep_is_hash_bound_and_bounded() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m127-mobilegym-notes-sweep-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["benchmark_id"] == "mobilegym_notes_model_sweep"
    assert receipt["native_receipt_eligible"] is False
    assert receipt["official_split_verified"] is True
    assert receipt["task_count"] == 5
    assert receipt["max_steps"] == 2
    assert receipt["success_rate"] == 0.0
    assert len(receipt["tasks"]) == 5
    assert {task["task_id"] for task in receipt["tasks"]} == {
        "notes.CreateFolderAndMoveNote",
        "notes.CreateNoteWithReminder",
        "notes.DeleteAllCompletedTodos",
        "notes.DeleteTodo",
        "notes.ReadTodoText",
    }
    assert all(task["tools"] == ["mobile_submit_answer"] for task in receipt["tasks"])


def test_m130_state_routing_rerank_receipt_is_hash_bound_and_non_gating() -> None:
    receipt = json.loads(
        Path("docs/paper/results/raw/m130-mobilegym-state-routing-rerank-v1.json").read_text(
            encoding="utf-8"
        )
    )
    self_hash = receipt.pop("receipt_self_sha256")
    assert self_hash == _canonical_sha256(receipt)
    assert receipt["native_receipt_eligible"] is False
    assert receipt["data"]["synthetic_mobile_rows"] == 105
    assert receipt["data"]["state_conditioned_trajectory_rows"] == 31
    assert receipt["arms"]["warm"]["native_top1"]["tools"] == ["mobile_submit_answer"]
    assert receipt["arms"]["warm"]["native_top8_rerank"]["tools"] == ["mobile_wait"]
    assert receipt["arms"]["random"]["native_top8_rerank"]["success_rate"] == 0.0
