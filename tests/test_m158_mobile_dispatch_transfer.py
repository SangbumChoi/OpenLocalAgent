from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m158-mobile-dispatch-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m158_receipt_self_hash_and_public_boundary() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["kind"] == "localagent_realistic_mobile_dispatch_transfer_receipt"
    assert payload["decision"] == "diagnostic_only"
    assert "not Android emulator/AndroidWorld success" in payload["claim_boundary"]
    assert payload["source"]["visual_input_omitted"] is True


def test_m158_held_out_mobile_metrics_and_native_canary_are_bound() -> None:
    payload = _load()
    training = payload["training"]
    held_out = training["held_out"]
    native = payload["native_canary"]
    parent = payload["parent_native_canary"]
    assert training["train_rows"] == 4096
    assert training["held_out_rows"] == 904
    assert held_out["rows"] == 904
    assert held_out["route_accuracy"] == 1.0
    assert held_out["selector_top1"] > 0.4
    assert training["pointer_training"]["held_out"]["exact_rate"] == 0.25
    assert native["environment_executed"] is True
    assert native["official_split_verified"] is True
    assert native["task_count"] == 20
    assert native["passed_tasks"] == 1
    assert native["errors"] == []
    assert parent["passed_tasks"] == 1
    assert payload["comparison"]["native_success_delta_vs_parent_same_range"] == 0.0


def test_m158_transfer_reuses_compatible_body_and_moves_heads() -> None:
    payload = _load()
    compatibility = payload["training"]["weight_transfer"]["compatibility"]
    groups = payload["training"]["weight_transfer"]["groups"]
    assert compatibility["tokenizer_sha256_equal"] is True
    assert compatibility["config_mismatches"] == {}
    assert compatibility["shape_mismatches"] == {}
    assert groups["embedding"]["relative_delta_l2"] == 0.0
    assert groups["attention_or_mixer"]["relative_delta_l2"] == 0.0
    assert groups["ffn"]["relative_delta_l2"] == 0.0
    assert groups["action_heads"]["relative_delta_l2"] > 1.0
