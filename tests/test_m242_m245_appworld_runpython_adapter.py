from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def test_appworld_train_dev_manifests_are_self_hashed_and_disjoint() -> None:
    train = _load("m242-appworld-train-manifest-v1.json")
    dev = _load("m242-appworld-dev-manifest-v1.json")
    for manifest in (train, dev):
        expected = manifest.pop("manifest_self_sha256")
        assert _canonical_hash(manifest) == expected
        assert manifest["source"]["dataset"] == "appworld"
        assert manifest["source"]["purpose"] in {"train", "eval"}
        assert manifest["source"]["split"] in {"train", "dev"}
        assert manifest["rows"] == len(manifest["tasks"])
    assert train["source"]["purpose"] == "train"
    assert train["source"]["split"] == "train"
    assert dev["source"]["purpose"] == "eval"
    assert dev["source"]["split"] == "dev"
    assert {task["task_id"] for task in train["tasks"]}.isdisjoint(
        {task["task_id"] for task in dev["tasks"]}
    )


def test_appworld_runpython_sft_and_head_adapter_report_real_metrics() -> None:
    sft = _load("m242-appworld-runpython-sft-v1.json")
    head = _load("m244-appworld-runpython-head-adapter-v1.json")
    assert sft["source"]["dataset"] == head["source"]["dataset"] == "appworld"
    assert sft["rows"] == {"train": 24, "eval": 12}
    assert sft["after"]["eval"]["assistant_token_accuracy"] > sft["before"]["eval"]["assistant_token_accuracy"]
    assert sft["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert head["hyperparameters"]["selector_pool"] == "standard"
    assert head["heads"]["before"]["selector_top1_accuracy"] == 0.0
    assert head["heads"]["after"]["selector_top1_accuracy"] == 1.0
    assert head["heads"]["after"]["route_accuracy"] == 1.0


def test_appworld_runpython_native_receipt_is_fail_closed() -> None:
    receipt = _load("m245-appworld-runpython-native-dev-v1.json")
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["configuration"]["action_translation"] == "appworld_run_python"
    assert receipt["configuration"]["replay_run_python"] is True
    assert receipt["configuration"]["selector_first"] is True
    assert receipt["summary"] == {
        "action_replayed": 12,
        "native_api_calls": 0,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert all(task["predicted_tool"] == "run_python" for task in receipt["tasks"])
    assert all(task["action_replayed"] is True for task in receipt["tasks"])
    assert "not an AppWorld leaderboard result" in receipt["claim_boundary"]


def test_appworld_weight_report_records_compatible_low_rate_movement() -> None:
    report = _load("m243-appworld-runpython-weight-transfer-v1.json")
    assert report["kind"] == "localagent_weight_transfer_analysis"
    assert report["compatibility"]["config_mismatches"] == {}
    assert report["compatibility"]["shape_mismatches"] == {}
    assert report["compatibility"]["tokenizer_sha256_equal"] is True
    assert report["groups"]["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.0 < report["groups"]["embedding"]["relative_delta_l2"] < 0.01
    assert 0.0 < report["groups"]["attention_or_mixer"]["relative_delta_l2"] < 0.01
    assert 0.0 < report["groups"]["ffn"]["relative_delta_l2"] < 0.01
