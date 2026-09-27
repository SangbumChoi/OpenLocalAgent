"""Integrity checks for the Android-Control split and transfer receipts."""

import hashlib
import json
from pathlib import Path

from scripts.split_androidcontrol_episodes import split_rows


TRANSFER_RECEIPT = Path("docs/paper/results/raw/m475-androidcontrol-current-warm-random-v1.json")
MOBILE_RECEIPT = Path("docs/paper/results/raw/m475-androidcontrol-mobilegym-replay-v1.json")
PARENT_SHA256 = "6a6520264f5f81fc68c54f80d462ddde64ac2f442e6e30077c909b702939dd45"


def _self_hash(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_self_sha256"}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row(episode: str, index: int) -> dict:
    return {
        "meta": {
            "image_references": [f"and_ctrl/out_episode_{episode}_step_{index:03d}.png"],
            "source_row_index": index,
        },
        "messages": [],
    }


def test_androidcontrol_split_is_whole_episode_and_deterministic() -> None:
    rows = [_row("1", 0), _row("1", 1), _row("2", 2), _row("3", 3), _row("4", 4)]
    train_a, eval_a, manifest_a = split_rows(rows, eval_percent=20)
    train_b, eval_b, manifest_b = split_rows(rows, eval_percent=20)
    assert train_a == train_b
    assert eval_a == eval_b
    assert manifest_a == manifest_b
    train_ids = {row["meta"]["parent_record_id"] for row in train_a}
    eval_ids = {row["meta"]["parent_record_id"] for row in eval_a}
    assert not train_ids & eval_ids
    assert all(row["meta"]["split_contract"] == "whole_episode_sha256_bucket_v1" for row in train_a + eval_a)


def test_m475_transfer_binds_public_androidcontrol_and_random_control() -> None:
    payload = json.loads(TRANSFER_RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["parent_checkpoint"]["sha256"] == PARENT_SHA256
    assert payload["source"]["dataset"] == "google/androidcontrol"
    assert payload["source"]["revision"] == "hf:OfficerChul/Android-Control-84k@train4096"
    assert payload["protocol"]["split_contract"]["no_episode_overlap"] is True
    assert payload["comparison"]["aggregate"]["warm_after_token_accuracy"] == 0.6339285714285714
    assert payload["comparison"]["aggregate"]["random_after_token_accuracy"] == 0.0
    assert payload["warm"]["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["random"]["after"]["eval"]["assistant_sequence_accuracy"] == 0.0
    assert payload["decision"]["export_child_to_webgpu"] is False


def test_m475_native_mobile_replay_rejects_promotion() -> None:
    payload = json.loads(MOBILE_RECEIPT.read_text(encoding="utf-8"))
    assert payload["receipt_self_sha256"] == _self_hash(payload)
    assert payload["source"]["revision"] == "093a3292d13fc4186e279af4ef1b005ac8e4d2b7"
    assert payload["parent"]["checkpoint_sha256"] == PARENT_SHA256
    assert payload["warm"]["success_rate"] == 0.0
    assert payload["warm"]["progress_sum"] == 0.0
    assert payload["warm"]["tool_counts"] == {"mobile_input_text": 4}
    assert payload["decision"]["adopt_androidcontrol_child_for_native_mobile"] is False
