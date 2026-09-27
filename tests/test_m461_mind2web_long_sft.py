"""Integrity checks for the longer current-parent Mind2Web continuation."""

import hashlib
import json
from pathlib import Path


RECEIPT = Path("docs/paper/results/raw/m461-mind2web-long-sft-weight-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_m461_binds_public_source_parent_and_long_protocol() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["kind"] == "localagent_mind2web_long_sft_and_weight_receipt"
    assert payload["dataset"]["name"] == "osunlp/Mind2Web"
    assert payload["dataset"]["revision"] == "17ece8eb89862368edc0cc806acee6fca5163474"
    assert payload["dataset"]["train_rows"] == 96
    assert payload["dataset"]["eval_rows"] == 32
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["protocol"]["steps"] == 24
    assert payload["protocol"]["head_steps"] == 0


def test_m461_improves_held_out_tokens_but_rejects_promotion() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    before = payload["teacher_forced_before"]["eval"]
    after = payload["teacher_forced"]["eval"]
    assert after["assistant_token_accuracy"] > before["assistant_token_accuracy"]
    assert after["assistant_sequence_accuracy"] == 0.0
    groups = payload["weight_transfer"]["groups"]
    assert groups["action_heads"]["relative_delta_l2"] == 0.0
    assert groups["embedding"]["relative_delta_l2"] > 0.002
    assert groups["embedding"]["relative_delta_l2"] < 0.004
    assert payload["decision"]["native_replay_required"] is True
    assert payload["decision"]["webgpu_export_allowed"] is False
