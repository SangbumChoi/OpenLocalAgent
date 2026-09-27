"""Integrity checks for the public Mind2Web realistic SFT continuation."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m448-mind2web-realistic-sft-weight-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m448_receipt_binds_public_source_and_parent() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mind2web_realistic_sft_and_weight_receipt"
    assert payload["dataset"]["name"] == "osunlp/Mind2Web"
    assert payload["dataset"]["revision"] == "17ece8eb89862368edc0cc806acee6fca5163474"
    assert payload["dataset"]["train_rows"] == 96
    assert payload["dataset"]["eval_rows"] == 32
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256


def test_m448_held_out_accuracy_improves_without_large_backbone_movement() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    before = payload["teacher_forced_before"]["eval"]["assistant_token_accuracy"]
    after = payload["teacher_forced"]["eval"]["assistant_token_accuracy"]
    assert after > before
    assert payload["teacher_forced"]["eval"]["assistant_sequence_accuracy"] == 0.0
    groups = payload["weight_transfer"]["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    for name in ("embedding", "attention_or_mixer", "ffn", "normalization"):
        assert groups[name]["relative_delta_l2"] < 0.002
    assert payload["decision"]["native_replay_required"] is True
    assert payload["decision"]["webgpu_export_allowed"] is False


def test_m448_native_canary_separates_protocol_progress_from_task_success() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    canary = payload["native_canary"]
    assert canary["summary"] == {
        "runtime_errors": 0,
        "tasks": 1,
        "verifier_failures": 1,
        "verifier_passes": 0,
    }
    turns = canary["results"][0]["turns"]
    assert [turn["tool"] for turn in turns[:2]] == ["browser_navigate", "browser_snapshot"]
    assert turns[0]["arguments"]["url"] == "https://eval-web.mcpmark.ai/extraction"
    assert turns[1]["arguments"] == {}
    assert canary["results"][0]["verifier_exit_code"] == 1
