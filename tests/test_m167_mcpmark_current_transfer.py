from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/paper/results/raw/m167-mcpmark-current-transfer-v1.json"


def _load() -> dict:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_m167_receipt_self_hash_and_public_boundary() -> None:
    payload = _load()
    claimed = payload.pop("receipt_self_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert claimed == expected
    assert payload["dataset"] == "Jakumetsu/mcpmark-trajectory-log"
    assert payload["source_revision"] == "e50578f0ab904d8e6a7c576c387c1e76ae482c89"
    assert payload["decision"] == "diagnostic_only"
    assert "not an official MCPMark score" in payload["claim_boundary"]


def test_m167_split_is_redacted_and_disjoint() -> None:
    payload = _load()
    inputs = payload["inputs"]
    assert inputs["train"]["rows"] == 8
    assert inputs["eval"]["rows"] == 2
    assert inputs["train"]["parent_records"] == 8
    assert inputs["eval"]["parent_records"] == 2
    assert inputs["parent_record_disjoint"] is True
    assert inputs["tool_outputs_redacted"] is True
    assert inputs["assistant_text_redacted"] is True
    assert inputs["visual_input_omitted"] is True


def test_m167_warm_current_child_beats_random_and_preserves_body() -> None:
    payload = _load()
    training = payload["training"]
    comparison = payload["comparison"]
    assert comparison["warm_start_better_after"] is True
    assert comparison["warm_minus_random_after_pp"] == 38.46153846
    assert comparison["warm_after_sequence_accuracy"] == 0.0
    assert comparison["random_after_sequence_accuracy"] == 0.0
    assert training["warm_child"]["after_eval"]["assistant_token_accuracy"] > training[
        "warm_child"
    ]["before_eval"]["assistant_token_accuracy"]
    assert training["random_child"]["after_eval"]["assistant_token_accuracy"] > training[
        "random_child"
    ]["before_eval"]["assistant_token_accuracy"]
    warm = training["warm_child"]["relative_delta_l2"]
    random = training["random_child"]["relative_delta_l2"]
    assert warm["embedding"] < 0.01
    assert warm["attention_or_mixer"] < 0.01
    assert warm["ffn"] < 0.01
    assert warm["normalization"] < 0.001
    assert random["embedding"] > 1.0
    assert random["attention_or_mixer"] > 0.5
    assert random["ffn"] > 0.5
    assert random["normalization"] > 0.05
    assert payload["compatibility"] == {
        "config_mismatches": {},
        "shape_mismatches": {},
        "shared_tensor_count": 51,
        "tokenizer_sha256_equal": True,
    }
