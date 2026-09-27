from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/paper/results/raw"
_SCRIPT = ROOT / "scripts/evaluate_appworld_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_appworld_checkpoint", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_parse_appworld_api_code = _MODULE._parse_appworld_api_code


def _canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load(name: str) -> dict:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def test_appworld_action_step_manifests_are_self_hashed_and_disjoint() -> None:
    train = _load("m247-appworld-action-train-manifest-v1.json")
    dev = _load("m247-appworld-action-dev-manifest-v1.json")
    for manifest in (train, dev):
        expected = manifest.pop("manifest_self_sha256")
        assert _canonical_hash(manifest) == expected
        assert manifest["kind"] == "localagent_appworld_action_step_export"
        assert manifest["source"]["dataset"] == "appworld"
        assert manifest["source"]["split"] in {"train", "dev"}
        assert manifest["rows"] == len(manifest["tasks"])
        assert all(task["first_action"]["app"] for task in manifest["tasks"])
        assert all(task["first_action"]["api"] for task in manifest["tasks"])
        assert all(task["first_action"]["code"]["sha256"] for task in manifest["tasks"])
    assert train["source"]["purpose"] == "train"
    assert dev["source"]["purpose"] == "eval"
    assert {task["task_id"] for task in train["tasks"]}.isdisjoint(
        {task["task_id"] for task in dev["tasks"]}
    )
    assert train["rows"] == 24
    assert dev["rows"] == 12


def test_appworld_action_step_sft_report_is_teacher_forced_only() -> None:
    report = _load("m247-appworld-action-step-sft-v1.json")
    assert report["source"]["dataset"] == "appworld_action_step"
    assert report["rows"] == {"train": 24, "eval": 12}
    assert report["hyperparameters"]["selector_pool"] == "standard"
    assert report["after"]["eval"]["assistant_token_accuracy"] > report["before"]["eval"][
        "assistant_token_accuracy"
    ]
    assert report["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert report["heads"]["after"]["selector_top1_accuracy"] == 1.0
    assert report["child"]["sha256"] == "79456920e5012de5bc63a4004cd310b271ff80f31ecc6ea8e908c8f5809264e1"


def test_appworld_api_step_native_receipt_proves_execution_not_success() -> None:
    receipt = _load("m250-appworld-action-step-schema-native-dev-v1.json")
    expected = receipt.pop("receipt_self_sha256")
    assert _canonical_hash(receipt) == expected
    assert receipt["configuration"]["action_translation"] == "appworld_api_step"
    assert receipt["configuration"]["schema_ground_appworld_api_step"] is True
    assert receipt["environment"]["native_runtime_executed"] is True
    assert receipt["summary"] == {
        "action_replayed": 12,
        "native_action_api_calls": 12,
        "native_api_calls": 48,
        "native_bootstrap_api_calls": 36,
        "native_success_rate": 0.0,
        "native_successes": 0,
        "tasks": 12,
    }
    assert all(task["schema_translation_applied"] for task in receipt["tasks"])
    assert all(task["native_action_api_calls"] == 1 for task in receipt["tasks"])
    assert all(task["native_bootstrap_api_calls"] == 3 for task in receipt["tasks"])
    assert "not an AppWorld leaderboard result" in receipt["claim_boundary"]


def test_appworld_api_step_parser_is_strict_and_literal_only() -> None:
    assert _parse_appworld_api_code("apis.spotify.show_song_library(page_index=0)") == (
        "spotify",
        "show_song_library",
        {"page_index": 0},
    )
    assert _parse_appworld_api_code("apis.phone.search_contacts(query='Kristin')") == (
        "phone",
        "search_contacts",
        {"query": "Kristin"},
    )
    assert _parse_appworld_api_code("apis.spotify.show_song_library(__import__('os'))") is None
    assert _parse_appworld_api_code("apis.spotify.show_song_library(page_index=x)") is None
    assert _parse_appworld_api_code("apis.spotify.show_song_library(); print('extra')") is None
    assert _parse_appworld_api_code("apis.spotify.show_song_library(**kwargs)") is None


def test_appworld_action_step_weight_report_is_compatible_and_low_rate() -> None:
    report = _load("m251-appworld-action-step-weight-transfer-v1.json")
    assert report["kind"] == "localagent_weight_transfer_analysis"
    assert report["compatibility"]["config_mismatches"] == {}
    assert report["compatibility"]["shape_mismatches"] == {}
    assert report["compatibility"]["tokenizer_sha256_equal"] is True
    groups = report["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    assert 0.02 < groups["embedding"]["relative_delta_l2"] < 0.022
    assert 0.008 < groups["attention_or_mixer"]["relative_delta_l2"] < 0.009
    assert 0.01 < groups["ffn"]["relative_delta_l2"] < 0.011
    assert groups["normalization"]["relative_delta_l2"] < 0.001
